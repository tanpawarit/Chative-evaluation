from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "eval_generation/intent_agent/data"
GOLDENS_PATH = DATA_DIR / "goldens.json"
INTENTS_PATH = DATA_DIR / "intents.json"


def ensure_data_dir(path: Path = DATA_DIR) -> None:
    path.mkdir(parents=True, exist_ok=True)


@dataclass
class IntentGolden:
    """Single test scenario for intent extraction."""

    name: str
    input: str
    intent: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "name": self.name,
            "input": self.input,
            "intent": self.intent,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "IntentGolden":
        return cls(name=data["name"], input=data["input"], intent=data["intent"])


def load_intents(path: Path = INTENTS_PATH) -> Dict[str, str]:
    """Load intent descriptions from JSON file.

    Supports two formats for convenience:
    - Mapping of intent -> description
    - List of objects: {"intent": "...", "description": "..."}
    """

    if not path.exists():
        raise FileNotFoundError(f"Intent definition file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        return {str(k): str(v) for k, v in data.items()}

    intents: Dict[str, str] = {}
    for item in data:
        intent = item.get("intent")
        description = item.get("description")
        if intent and description:
            intents[str(intent)] = str(description)

    return intents


def load_goldens(path: Path = GOLDENS_PATH) -> List[IntentGolden]:
    if not path.exists():
        return []

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [IntentGolden.from_dict(item) for item in raw]


def save_goldens(goldens: Iterable[IntentGolden], path: Path = GOLDENS_PATH) -> None:
    ensure_data_dir()
    with open(path, "w", encoding="utf-8") as f:
        json.dump([g.to_dict() for g in goldens], f, ensure_ascii=False, indent=2)
