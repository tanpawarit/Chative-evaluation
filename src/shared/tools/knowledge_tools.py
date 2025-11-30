from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

from pkg.milvus.client import MilvusHybridRetriever

# Singleton retriever so we reuse connections across tool calls.
retriever = MilvusHybridRetriever.from_env()

KNOWLEDGE_SEARCH_TOOL_NAME = "knowledge_search"


def _to_agent_chunk(hit: Any) -> Dict[str, Any]:
    """Convert a Milvus hit into the agent-friendly payload."""
    meta = hit.metadata or {}
    source = (
        meta.get("doc_name")
        or meta.get("docName")
        or meta.get("source")
        or meta.get("document_name")
    )
    chunk_id = meta.get("chunk_id") or meta.get("chunkId") or hit.id
    chunk: Dict[str, Any] = {
        "text": hit.text,
        "source": source,
        "score": float(hit.score),
        "chunk_id": str(chunk_id) if chunk_id is not None else "",
    }
    if "chunk_index" in meta:
        chunk["chunk_index"] = meta["chunk_index"]
    if "total_chunks" in meta:
        chunk["total_chunks"] = meta["total_chunks"]
    if "doc_hash" in meta:
        chunk["doc_hash"] = meta["doc_hash"]
    if "workspace_id" in meta:
        chunk["workspace_id"] = meta["workspace_id"]
    return chunk


@tool(KNOWLEDGE_SEARCH_TOOL_NAME)
def knowledge_search(
    query: str, workspace_id: Optional[str] = None, limit: int = 5
) -> Dict[str, Any]:
    """
    Search the workspace knowledge base and return top passages as structured chunks with provenance.
    Use for factual/policy/product answers; prefer retrieved snippets over memory and cite by index.

    - query: Natural-language search query (≈5–25 words) including the user's ask plus key context
             like entities, product/plan, region, version, or date.
    - workspace_id: Optional partition key to restrict results.
    - limit: How many passages to return (default 5).
    """
    query = (query or "").strip()
    if not query:
        raise ValueError("query is required")

    top_k = limit if limit and limit > 0 else 5
    hits = retriever.search(query=query, workspace_id=workspace_id, top_k=top_k) or []
    chunks: List[Dict[str, Any]] = [_to_agent_chunk(hit) for hit in hits]

    res = {
        "query": query,
        "total": len(chunks),
        "chunks": chunks,
    }
    print(res)
    return res


# Convenience export for agent wiring.
KNOWLEDGE_TOOLS = [knowledge_search]
