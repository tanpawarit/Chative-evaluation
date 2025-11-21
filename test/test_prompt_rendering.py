import os
import sys

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from agents.prompt_utils import apply_mock_template_vars
from agents.response_agent.prompts import SYSTEM_PROMPT

def test_rendering(name, overrides):
    print(f"--- Test Case: {name} ---")
    rendered = apply_mock_template_vars(SYSTEM_PROMPT, overrides)
    
    # Extract relevant sections to keep output concise
    # We'll look for specific markers to see which block was rendered
    
    print(f"Overrides: {overrides}")
    
    # Check Language
    if "Language: Thai" in rendered:
        print("[Language] Detected: Thai")
    elif "Language: English" in rendered:
        print("[Language] Detected: English")
    else:
        # Check for custom language
        lang = overrides.get("{{.Language}}", "Unknown")
        if f"Language: {lang}" in rendered:
             print(f"[Language] Detected: {lang}")

    # Check Formality
    if "**Formal Gathering:**" in rendered or "- Complete sentences, no contractions" in rendered:
        print("[Formality] Detected: Formal")
    elif "**Friendly Gathering:**" in rendered or "- Warm and personable" in rendered:
        print("[Formality] Detected: Friendly")
    elif "**Casual Gathering:**" in rendered or "- Relaxed, natural speech" in rendered:
        print("[Formality] Detected: Casual")
    elif "**Playful Gathering:**" in rendered or "- Upbeat and energetic" in rendered:
        print("[Formality] Detected: Playful")
        
    # Check Sentiment
    if "**Customer Frustrated:**" in rendered or "**Customer frustrated/unhappy:**" in rendered:
        print("[Sentiment] Detected: Negative")
    elif "**Customer Happy:**" in rendered or "**Customer happy/excited:**" in rendered:
        print("[Sentiment] Detected: Positive")
    elif "**Customer Neutral:**" in rendered or "**Customer neutral:**" in rendered:
        print("[Sentiment] Detected: Neutral")

    # Check Entities
    if "Customer Info:" in rendered:
        print("[Entities] Detected: Present")
    else:
        print("[Entities] Detected: Absent")
        
    # Check Unknown Intent
    if "<unknown_intent_handling>" in rendered:
        print("[UnknownIntent] Detected: True")
    else:
        print("[UnknownIntent] Detected: False")

    print("\n")

def main():
    # Case 1: Default (Thai, Friendly, Neutral, No Entities)
    test_rendering("Default", {})

    # Case 2: English, Formal, Negative
    test_rendering("English Formal Negative", {
        "{{.Language}}": "English",
        "{{.Formality}}": "formal",
        "{{.Sentiment}}": "negative"
    })

    # Case 3: Casual, Positive, With Entities
    test_rendering("Casual Positive Entities", {
        "{{.Formality}}": "casual",
        "{{.Sentiment}}": "positive",
        "{{.Entities}}": '[{"name": "order_id", "value": "123"}]'
    })
    
    # Case 4: Unknown Intent
    test_rendering("Unknown Intent", {
        "{{.UnknownIntent}}": "true"
    })

    # Case 5: Empty Entities (should not show Customer Info)
    test_rendering("Empty Entities List", {
        "{{.Entities}}": "[]"
    })

if __name__ == "__main__":
    main()
