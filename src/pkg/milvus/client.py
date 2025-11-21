from __future__ import annotations

import hashlib
import logging
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

from langchain_openai import OpenAIEmbeddings
from pymilvus import AnnSearchRequest, MilvusClient, WeightedRanker

logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning("Invalid float for %s=%s, using default %s", name, value, default)
        return default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid int for %s=%s, using default %s", name, value, default)
        return default


@dataclass
class MilvusSettings:
    """Configuration holder for knowledge base retrieval."""

    uri: str = os.getenv("MILVUS_ADDR", "")
    username: str = os.getenv("MILVUS_USERNAME", "")
    password: str = os.getenv("MILVUS_PASSWORD", "")
    collection: str = os.getenv("MILVUS_COLLECTION", "knowledge_base")
    partition_key_field: str = os.getenv("MILVUS_PARTITION_KEY_FIELD", "workspace_id")

    dense_field: str = os.getenv("MILVUS_DENSE_FIELD", "dense_vector")
    dense_metric: str = os.getenv("MILVUS_DENSE_METRIC", "COSINE")
    dense_weight: float = _env_float("MILVUS_DENSE_WEIGHT", 0.6)

    sparse_field: str = os.getenv("MILVUS_SPARSE_FIELD", "sparse_vector")
    sparse_metric: str = os.getenv("MILVUS_SPARSE_METRIC", "BM25")
    sparse_weight: float = _env_float("MILVUS_SPARSE_WEIGHT", 0.4)
    sparse_drop_ratio: float = _env_float("MILVUS_SPARSE_DROP", 0.2)
    sparse_vocab_size: int = _env_int("MILVUS_SPARSE_VOCAB_SIZE", 500_000)

    text_field: str = os.getenv("MILVUS_TEXT_FIELD", "text")
    top_k: int = _env_int("MILVUS_TOP_K", 5)

    embedding_model: str = os.getenv(
        "AGENT_KNOWLEDGE_OPENAI_EMBED_MODEL", "text-embedding-3-small"
    )
    embedding_api_key: Optional[str] = os.getenv("AGENT_KNOWLEDGE_OPENAI_API_KEY")


@dataclass
class KnowledgeHit:
    """A normalized representation of a Milvus hit."""

    id: str
    score: float
    text: str
    metadata: Dict[str, Any]


class SimpleSparseEncoder:
    """
    Lightweight deterministic sparse encoder.

    NOTE: This must mirror the encoding strategy used when writing data into Milvus.
    It uses hashing to produce integer token ids so it can run without extra model
    dependencies. If your collection was built with a different encoder, plug that
    in by swapping this class.
    """

    token_pattern = re.compile(r"\b[\w-]+\b", re.UNICODE)

    def __init__(self, vocab_size: int = 500_000):
        self.vocab_size = vocab_size

    def _token_id(self, token: str) -> int:
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        return int(digest, 16) % self.vocab_size

    def encode(self, text: str) -> Dict[int, float]:
        tokens = self.token_pattern.findall(text.lower())
        if not tokens:
            return {}
        counts = Counter(tokens)
        total = float(sum(counts.values()))
        # Normalize counts so values are comparable across queries.
        return {self._token_id(tok): freq / total for tok, freq in counts.items()}


class MilvusHybridRetriever:
    """Hybrid dense + sparse retriever backed by Milvus."""

    def __init__(self, settings: Optional[MilvusSettings] = None):
        self.settings = settings or MilvusSettings()
        token = f"{self.settings.username}:{self.settings.password}" if self.settings.username else None
        self.client = MilvusClient(uri=self.settings.uri, token=token)
        self.embeddings = OpenAIEmbeddings(
            model=self.settings.embedding_model,
            api_key=self.settings.embedding_api_key,
        )
        self.sparse_encoder = SimpleSparseEncoder(vocab_size=self.settings.sparse_vocab_size)

    @classmethod
    def from_env(cls) -> "MilvusHybridRetriever":
        return cls(MilvusSettings())

    def _build_expr(self, workspace_id: Optional[str]) -> Optional[str]:
        if not workspace_id:
            return None
        escaped = workspace_id.replace('"', r"\"")
        return f'{self.settings.partition_key_field} == "{escaped}"'

    def _default_output_fields(self) -> List[str]:
        fields = [
            self.settings.text_field,
            "doc_name",
            "chunk_index",
            "total_chunks",
            "metadata",
            "document_id",
        ]
        if self.settings.partition_key_field:
            fields.append(self.settings.partition_key_field)
        return fields

    def _sparse_query_vector(self, query: str) -> Dict[int, float]:
        try:
            return self.sparse_encoder.encode(query)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Failed to encode sparse query vector: %s", exc)
            return {}

    def _parse_hits(self, search_result: Sequence[Sequence[Dict[str, Any]]]) -> List[KnowledgeHit]:
        if not search_result:
            return []

        hits = search_result[0]
        parsed: List[KnowledgeHit] = []
        for hit in hits:
            # Hit is a dict-like object with keys id, distance, entity
            hit_id = hit.get("id")
            score = hit.get("distance", 0.0)
            entity = hit.get("entity", {}) or {}
            text_value = self._extract_text(entity)
            parsed.append(
                KnowledgeHit(
                    id=str(hit_id) if hit_id is not None else "",
                    score=float(score),
                    text=text_value,
                    metadata={k: v for k, v in entity.items() if k != self.settings.text_field},
                )
            )
        return parsed

    def _extract_text(self, entity: Dict[str, Any]) -> str:
        if not entity:
            return ""
        if self.settings.text_field in entity and entity[self.settings.text_field] is not None:
            return str(entity[self.settings.text_field])

        # Fallback to common field names if text_field is missing.
        for field in ("content", "page_content", "body"):
            if field in entity and entity[field] is not None:
                return str(entity[field])
        return ""

    def search(
        self, query: str, workspace_id: Optional[str] = None, top_k: Optional[int] = None
    ) -> List[KnowledgeHit]:
        """
        Run a hybrid search against the Milvus knowledge base.

        Falls back to dense-only search if sparse encoding is unavailable for the query.
        """
        limit = top_k or self.settings.top_k
        expr = self._build_expr(workspace_id) or ""
        dense_vector = self.embeddings.embed_query(query)

        requests: List[AnnSearchRequest] = [
            AnnSearchRequest(
                data=[dense_vector],
                anns_field=self.settings.dense_field,
                param={"metric_type": self.settings.dense_metric},
                limit=limit,
                expr=expr or None,
            )
        ]

        weights: List[float] = [self.settings.dense_weight]

        # Check if we are using BM25 (server-side) or client-side sparse vectors
        if self.settings.sparse_metric == "BM25":
            # For BM25, we pass the raw text as data
            requests.append(
                AnnSearchRequest(
                    data=[query],
                    anns_field=self.settings.sparse_field,
                    param={
                        "metric_type": self.settings.sparse_metric,
                        "drop_ratio_search": self.settings.sparse_drop_ratio,
                    },
                    limit=limit,
                    expr=expr or None,
                )
            )
            weights.append(self.settings.sparse_weight)
        else:
            # Client-side sparse vector generation
            sparse_vector = self._sparse_query_vector(query)
            if sparse_vector:
                requests.append(
                    AnnSearchRequest(
                        data=[sparse_vector],
                        anns_field=self.settings.sparse_field,
                        param={
                            "metric_type": self.settings.sparse_metric,
                            "drop_ratio_search": self.settings.sparse_drop_ratio,
                        },
                        limit=limit,
                        expr=expr or None,
                    )
                )
                weights.append(self.settings.sparse_weight)

        if len(requests) == 1:
            logger.debug("Executing dense-only search against %s", self.settings.collection)
            result = self.client.search(
                collection_name=self.settings.collection,
                data=[dense_vector],
                anns_field=self.settings.dense_field,
                search_params={"metric_type": self.settings.dense_metric},
                limit=limit,
                output_fields=self._default_output_fields(),
                filter=expr,
            )
        else:
            logger.debug(
                "Executing hybrid search against %s with weights %s", self.settings.collection, weights
            )
            ranker = WeightedRanker(*weights)
            result = self.client.hybrid_search(
                collection_name=self.settings.collection,
                reqs=requests,
                ranker=ranker,
                limit=limit,
                output_fields=self._default_output_fields(),
                partition_names=None,
            )

        return self._parse_hits(result)
