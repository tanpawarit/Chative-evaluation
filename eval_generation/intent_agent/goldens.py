"""Seed goldens and optional synthesis helpers for the entity agent eval."""

from dataclasses import dataclass
from typing import Dict, List, Sequence


@dataclass
class EntityGolden:
    """Single test scenario for entity extraction."""

    name: str
    input: str
    intent: str
    entity_schema: Sequence[Dict[str, str]]
    expected_entities: Sequence[Dict[str, str]]


SEED_GOLDENS: List[EntityGolden] = [
    EntityGolden(
        name="gpu_price_cap",
        input="หาการ์ดจอ RTX 4070 ราคาไม่เกิน 25,000 บาท พร้อมส่ง",
        intent="inquire_product",
        entity_schema=[
            {"name": "product_type", "type": "text", "description": "product category"},
            {"name": "product_model", "type": "text", "description": "model or series"},
            {"name": "price_max", "type": "currency", "description": "maximum budget in THB"},
            {"name": "availability", "type": "text", "description": "stock or delivery timing"},
        ],
        expected_entities=[
            {"name": "product_type", "value": "การ์ดจอ"},
            {"name": "product_model", "value": "RTX 4070"},
            {"name": "price_max", "value": "25000"},
            {"name": "availability", "value": "พร้อมส่ง"},
        ],
    )
]

style = StylingConfig(
    input_format="Question in Thai language",
    expected_output_format="Answer in Thai language"
)

def synthesize_from_seed(contexts_per_seed: int = 1):
    """
    Optionally expand the golden set using DeepEval's Synthesizer.

    Requires an OPENAI_API_KEY. Returns a list of deepeval.dataset.Golden objects.
    """
    from deepeval.synthesizer import Synthesizer, StylingConfig

    synth = Synthesizer()
    contexts = [[golden.input] * contexts_per_seed for golden in SEED_GOLDENS]
    return synth.generate_goldens_from_contexts(contexts=contexts, synthetic_input_quality_threshold=0.8, styling_config=style)
