"""
DeepEval harness for the Entity Agent.

- Generates test cases from seed goldens (optionally augmented via Synthesizer)
- Invokes the entity agent prompt with runtime intent + entity schema
- Scores extraction quality with a lightweight precision/recall metric

Usage:
    python eval_generation/intent_agent/run_eval.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src"))

from agents.prompt_utils import apply_mock_template_vars  # noqa: E402
from agents.entity_agent import prompts as entity_prompts  # noqa: E402
from eval_generation.intent_agent.goldens import (  # noqa: E402
    SEED_GOLDENS,
    EntityGolden,
    synthesize_from_seed,
)


def build_prompt(intent: str, entity_schema: Sequence[dict]) -> ChatPromptTemplate:
    """Render the entity prompt with runtime intent + entity schema."""
    overrides = {
        "{{.TupleDelimiter}}": "\t",
        "{{.RecordDelimiter}}": "##",
        "{{.CompletedDelimiter}}": "<|COMPLETED|>",
        "{{.Intent}}": intent,
        "{{.Entities}}": json.dumps(entity_schema, ensure_ascii=False),
    }
    prompt_text = apply_mock_template_vars(entity_prompts.SYSTEM_PROMPT, overrides)
    return ChatPromptTemplate.from_messages(
        [
            ("system", prompt_text),
            ("human", "{input}"),
        ]
    )


def build_chain(intent: str, entity_schema: Sequence[dict], llm: ChatOpenAI):
    prompt = build_prompt(intent, entity_schema)
    parser = StrOutputParser()
    return prompt | llm | parser


def canonicalize(value: str) -> str:
    """Normalize text for matching."""
    value = value.lower().strip()
    # Remove separators and punctuation; keep Thai and alphanumerics.
    return re.sub(r"[^0-9a-zก-๙]+", "", value)


def parse_entity_output(
    output: str,
    record_delimiter: str = "##",
    tuple_delimiter: str = "\t",
) -> List[dict]:
    """Parse the entity agent raw string into structured rows."""
    rows: List[dict] = []
    if not output:
        return rows

    for raw_row in output.split(record_delimiter):
        row = raw_row.strip()
        if not row or row == "<|COMPLETED|>":
            continue
        if row.startswith("(") and row.endswith(")"):
            row = row[1:-1]
        parts = row.split(tuple_delimiter)
        if len(parts) < 7 or parts[0] != "entity":
            continue
        _, name, original_text, normalized_value, *_rest = parts
        value = normalized_value or original_text
        rows.append({"name": name.strip(), "value": value.strip()})
    return rows


class EntityExtractionMetric(BaseMetric):
    """Simple precision/recall metric over normalized entity values."""

    name = "entity_extraction_f1"

    def __init__(self, threshold: float = 0.7):
        super().__init__()
        self.threshold = threshold

    def measure(self, test_case: LLMTestCase) -> float:
        expected = json.loads(test_case.expected_output or "[]")
        actual = parse_entity_output(test_case.actual_output)

        expected_set = {(e["name"], canonicalize(e["value"])) for e in expected}
        actual_set = {(e["name"], canonicalize(e["value"])) for e in actual}

        tp = len(expected_set & actual_set)
        precision = tp / len(actual_set) if actual_set else 0.0
        recall = tp / len(expected_set) if expected_set else 0.0
        score = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        self.score = score
        self.reason = (
            f"tp={tp}, expected={len(expected_set)}, actual={len(actual_set)}, "
            f"precision={precision:.2f}, recall={recall:.2f}"
        )
        self.success = score >= self.threshold
        return score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)


def build_test_cases(
    goldens: Iterable[EntityGolden], llm: ChatOpenAI
) -> List[LLMTestCase]:
    test_cases: List[LLMTestCase] = []
    for golden in goldens:
        chain = build_chain(golden.intent, golden.entity_schema, llm)
        output = chain.invoke({"input": golden.input})
        test_cases.append(
            LLMTestCase(
                input=golden.input,
                actual_output=output,
                expected_output=json.dumps(golden.expected_entities, ensure_ascii=False),
                context=[json.dumps(golden.entity_schema, ensure_ascii=False)],
                tags=[golden.name, golden.intent],
            )
        )
    return test_cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DeepEval for the entity agent.") 
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="F1 threshold for passing the metric.",
    )
    parser.add_argument(
        "--use-synthesized",
        action="store_true",
        help="Include synthesized goldens from DeepEval Synthesizer.",
    )
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()

    llm = ChatOpenAI(model=args.model, temperature=0)
    goldens = list(SEED_GOLDENS)
    if args.use_synthesized:
        # Optional: append Synthesizer-generated goldens for broader coverage.
        goldens.extend(synthesize_from_seed())

    test_cases = build_test_cases(goldens, llm)
    metric = EntityExtractionMetric(threshold=args.threshold)
    evaluate(test_cases=test_cases, metrics=[metric])


if __name__ == "__main__":
    main()
