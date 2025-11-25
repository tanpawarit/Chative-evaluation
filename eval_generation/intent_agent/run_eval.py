"""
DeepEval harness for the Intent Agent.

- Generates test cases from seed goldens (optionally augmented via Synthesizer)
- Invokes the intent agent prompt with a list of candidate intents
- Scores extraction quality with simple accuracy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List

from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "src"))

from agents.prompt_utils import apply_mock_template_vars  # noqa: E402
from agents.intent_agent import prompts as intent_prompts  # noqa: E402
from eval_generation.intent_agent.goldens import (  # noqa: E402
    GOLDENS_PATH,
    INTENTS_PATH,
    IntentGolden,
    load_goldens,
)


def build_prompt(intents_str: str) -> ChatPromptTemplate:
    """Render the intent prompt with candidate intents."""
    overrides = {
        "{{.TupleDelimiter}}": "\t",
        "{{.RecordDelimiter}}": "##",
        "{{.CompletedDelimiter}}": "<|COMPLETED|>",
        "{{.Intents}}": intents_str,
    }
    prompt_text = apply_mock_template_vars(intent_prompts.SYSTEM_PROMPT, overrides)
    return ChatPromptTemplate.from_messages(
        [
            ("system", prompt_text),
            ("human", "{input}"),
        ]
    )


def build_chain(intents_str: str, llm: ChatOpenAI):
    prompt = build_prompt(intents_str)
    parser = StrOutputParser()
    return prompt | llm | parser


def parse_intent_output(
    output: str,
    record_delimiter: str = "##",
    tuple_delimiter: str = "\t",
) -> str | None:
    """Parse the intent agent raw string to extract the top intent."""
    if not output:
        return None

    for raw_row in output.split(record_delimiter):
        row = raw_row.strip()
        if not row or row == "<|COMPLETED|>":
            continue
        if row.startswith("(") and row.endswith(")"):
            row = row[1:-1]
        parts = row.split(tuple_delimiter)
        # Format: intent, intent_name, confidence, priority
        if len(parts) >= 2 and parts[0] == "intent":
            return parts[1].strip()
            
    return None


class IntentAccuracyMetric(BaseMetric):
    """Simple accuracy metric for intent classification."""

    name = "intent_accuracy"

    def __init__(self, threshold: float = 1.0):
        super().__init__()
        self.threshold = threshold

    def is_successful(self) -> bool:
        return self.success

    def measure(self, test_case: LLMTestCase) -> float:
        expected = test_case.expected_output
        actual = parse_intent_output(test_case.actual_output)

        score = 1.0 if expected == actual else 0.0

        self.score = score
        self.reason = (
            f"expected='{expected}', actual='{actual}'"
        )
        self.success = score >= self.threshold
        return score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)


def build_test_cases(
    goldens: Iterable[IntentGolden], llm: ChatOpenAI
) -> List[LLMTestCase]:
    # Collect all unique intents from goldens
    all_intents = sorted(set(g.intent for g in goldens))

    # Load priorities from intents.json
    with open(INTENTS_PATH, "r", encoding="utf-8") as f:
        intents_data = json.load(f)

    intent_priorities = {}
    if isinstance(intents_data, dict):
        # Fallback: if config is a simple mapping, default priority to 1.0
        intent_priorities = {name: 1.0 for name in intents_data.keys()}
    else:
        for item in intents_data:
            name = item.get("intent")
            if not name:
                continue
            priority = item.get("priority", 1.0)
            try:
                intent_priorities[name] = float(priority)
            except (TypeError, ValueError):
                intent_priorities[name] = 1.0

    # Build intents string like: "greet:0.30,inquire_product:0.90,..."
    intents_str = ",".join(
        f"{intent}:{intent_priorities.get(intent, 1.0):.2f}" for intent in all_intents
    )

    test_cases: List[LLMTestCase] = []
    chain = build_chain(intents_str, llm)

    for golden in goldens:
        output = chain.invoke({"input": golden.input})
        test_cases.append(
            LLMTestCase(
                input=golden.input,
                actual_output=output,
                expected_output=golden.intent,
                context=[intents_str],
                tags=[golden.name, golden.intent],
            )
        )
    return test_cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DeepEval for the intent agent.") 
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="OpenAI model to use.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Accuracy threshold for passing the metric.",
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
    
    # Load goldens from JSON
    goldens_file = GOLDENS_PATH
    if not goldens_file.exists():
        print(f"Error: Goldens file not found at {goldens_file}")
        print("Please run 'uv run eval_generation/intent_agent/generate_goldens.py' first.")
        sys.exit(1)

    goldens = load_goldens(goldens_file)

    test_cases = build_test_cases(goldens, llm)
    metric = IntentAccuracyMetric(threshold=args.threshold)
    evaluate(test_cases=test_cases, metrics=[metric])


if __name__ == "__main__":
    main()


# uv run eval_generation/intent_agent/run_eval.py
