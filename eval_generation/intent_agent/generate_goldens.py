import asyncio
from typing import Iterable, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from eval_generation.intent_agent.goldens import (  # noqa: E402
    IntentGolden,
    load_intents,
    save_goldens,
)

GENERATE_PROMPT = """You are a synthetic data generator for an intent classification system.
Your task is to generate realistic user queries in Thai (and some English mixed is okay) for a specific intent.

Intent: {intent}
Description: {description}

Existing examples:
{examples}

Generate {count} NEW and DIVERSE user queries for this intent.
- Vary the length, tone, and specific details (product names, error codes, etc.).
- Make them sound like natural chat messages.
- Return ONLY the queries, one per line.
- Do not number the lines.
"""

async def generate_for_intent(
    llm,
    intent: str,
    description: str,
    seed_examples: Iterable[IntentGolden] | None = None,
    count: int = 5,
) -> List[IntentGolden]:
    # Find existing examples for this intent
    seed_examples = list(seed_examples or [])
    existing_inputs = [g.input for g in seed_examples]
    examples_str = "\n".join(f"- {ex}" for ex in existing_inputs)

    prompt = ChatPromptTemplate.from_template(GENERATE_PROMPT)
    chain = prompt | llm | StrOutputParser()

    print(f"Generating {count} examples for intent: {intent}...")
    output = await chain.ainvoke({
        "intent": intent,
        "description": description,
        "examples": examples_str,
        "count": count
    })

    new_goldens: List[IntentGolden] = []
    existing_set = set(existing_inputs)

    for line in output.strip().split("\n"):
        line = line.strip()
        if line and line not in existing_set:
            # Create a simple name based on intent and hash or counter
            name = f"{intent}_gen_{hash(line) % 10000:04d}"
            new_goldens.append(IntentGolden(name=name, input=line, intent=intent))
            existing_set.add(line)
    
    return new_goldens

async def main():
    load_dotenv()
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7) # Higher temp for diversity

    intents = load_intents()
    all_goldens: List[IntentGolden] = []
    
    for intent, description in intents.items():
        new_goldens = await generate_for_intent(
            llm, intent, description, seed_examples=None, count=5
        )
        all_goldens.extend(new_goldens)

    save_goldens(all_goldens)

    print(
        f"Generated {len(all_goldens)} goldens in"
        f" {PROJECT_ROOT / 'eval_generation/intent_agent/data/goldens.json'}"
    )

if __name__ == "__main__":
    asyncio.run(main())

# uv run eval_generation/intent_agent/generate_goldens.py
