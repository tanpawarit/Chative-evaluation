"""Shared helpers for patching prompt templates."""

import re
from typing import Dict, Mapping, Optional

# Hard-coded mock values for the Go-style template placeholders used in prompts.
# Extend here if new placeholders appear in any agent prompt.
MOCK_TEMPLATE_VARIABLES: Dict[str, str] = {
    # ====== Delimiters for intent agent placeholders ======
    "{{.TupleDelimiter}}": "<||>",
    "{{.RecordDelimiter}}": "##",
    "{{.CompletedDelimiter}}": "<|COMPLETED|>",
    # Intent agent/runtime placeholders
    "{{.Intents}}": "greet:0.30,inquire_product:0.90,check_stock:0.80,warranty_claim:0.70,technical_support:0.60,buy_product:0.90",
    # ====== Entity agent placeholders ======
    "{{.Intent}}": "greet",
    "{{.Entities}}": "[]",
    "{{.MissingEntities}}": "[]",
    # ====== Entity agent placeholders ======
    "{{.Language}}": "Thai",
    "{{.Sentiment}}": "neutral",
    "{{.Formality}}": "friendly",
    "{{.Instruction}}": "Be concise and helpful.",
    "{{.Restriction}}": "",
    "{{.Action}}": "knowledge_search",
    "{{.AllowedTools}}": "['knowledge_search', 'calculator']",
    "{{.UnknownIntent}}": "",
}


def _is_truthy(value: str) -> bool:
    return bool(value and value.strip())


def _render_restriction_block(prompt: str, mapping: Mapping[str, str]) -> str:
    """Handle {{if .Restriction}} blocks by picking the appropriate branch."""
    pattern = re.compile(r"{{if\s+\.Restriction}}(.*?){{else}}(.*?){{end}}", re.DOTALL)

    def repl(match: re.Match[str]) -> str:
        content_if = match.group(1)
        content_else = match.group(2)
        return content_if if _is_truthy(mapping.get("{{.Restriction}}", "")) else content_else

    return pattern.sub(repl, prompt)


def _render_language_block(prompt: str, mapping: Mapping[str, str]) -> str:
    """Resolve the language conditional block."""
    pattern = re.compile(
        r"{{if eq \.Language \"Thai\"}}(.*?){{else if eq \.Language \"English\"}}(.*?){{else}}(.*?){{end}}",
        re.DOTALL,
    )

    def repl(match: re.Match[str]) -> str:
        lang = mapping.get("{{.Language}}", "").lower()
        if lang == "thai":
            return match.group(1)
        if lang == "english":
            return match.group(2)
        return match.group(3)

    return pattern.sub(repl, prompt)


def _render_formality_block(prompt: str, mapping: Mapping[str, str]) -> str:
    """Resolve the formality conditional block."""
    pattern = re.compile(
        r"{{if eq \.Formality \"formal\"}}(.*?){{else if eq \.Formality \"friendly\"}}(.*?){{else if eq \.Formality \"casual\"}}(.*?){{else if eq \.Formality \"playful\"}}(.*?){{end}}",
        re.DOTALL,
    )

    def repl(match: re.Match[str]) -> str:
        formality = mapping.get("{{.Formality}}", "").lower()
        if formality == "formal":
            return match.group(1)
        if formality == "friendly":
            return match.group(2)
        if formality == "casual":
            return match.group(3)
        if formality == "playful":
            return match.group(4)
        # Default to friendly if unknown
        return match.group(2)

    return pattern.sub(repl, prompt)


def _render_sentiment_block(prompt: str, mapping: Mapping[str, str]) -> str:
    """Resolve the sentiment conditional block."""
    pattern = re.compile(
        r"{{if eq \.Sentiment \"negative\"}}(.*?){{else if eq \.Sentiment \"positive\"}}(.*?){{else}}(.*?){{end}}",
        re.DOTALL,
    )

    def repl(match: re.Match[str]) -> str:
        sentiment = mapping.get("{{.Sentiment}}", "").lower()
        if sentiment == "negative":
            return match.group(1)
        if sentiment == "positive":
            return match.group(2)
        return match.group(3)

    return pattern.sub(repl, prompt)


def _render_unknown_intent_block(prompt: str, mapping: Mapping[str, str]) -> str:
    """Handle {{if .UnknownIntent}} blocks."""
    pattern = re.compile(r"{{if\s+\.UnknownIntent}}(.*?){{end}}", re.DOTALL)

    def repl(match: re.Match[str]) -> str:
        content_if = match.group(1)
        return content_if if _is_truthy(mapping.get("{{.UnknownIntent}}", "")) else ""

    return pattern.sub(repl, prompt)


def _render_entities_block(prompt: str, mapping: Mapping[str, str]) -> str:
    """Handle {{if .Entities}} blocks (including {{- if .Entities}})."""
    pattern = re.compile(r"{{-?\s*if\s+\.Entities}}(.*?){{-?\s*end}}", re.DOTALL)

    def repl(match: re.Match[str]) -> str:
        content_if = match.group(1)
        # Check if Entities is not empty JSON array "[]" and not empty string
        entities = mapping.get("{{.Entities}}", "[]")
        is_present = _is_truthy(entities) and entities != "[]"
        return content_if if is_present else ""

    return pattern.sub(repl, prompt)


def _render_conditionals(prompt: str, mapping: Mapping[str, str]) -> str:
    """Resolve the limited set of Go-template conditionals we use."""
    prompt = _render_restriction_block(prompt, mapping)
    prompt = _render_language_block(prompt, mapping)
    prompt = _render_formality_block(prompt, mapping)
    prompt = _render_sentiment_block(prompt, mapping)
    prompt = _render_unknown_intent_block(prompt, mapping)
    prompt = _render_entities_block(prompt, mapping)
    return prompt


def apply_mock_template_vars(
    prompt: str, overrides: Optional[Mapping[str, str]] = None
) -> str:
    """
    Replace known template placeholders with mock values (overridable) and
    resolve the simple conditional branches in our prompts.
    """
    mapping = {**MOCK_TEMPLATE_VARIABLES, **(overrides or {})}
    prompt = _render_conditionals(prompt, mapping)
    for placeholder, value in mapping.items():
        prompt = prompt.replace(placeholder, value)
    return prompt
