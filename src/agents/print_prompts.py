import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from agents.intent_agent.prompts import get_prompt as get_intent_prompt
from agents.entity_agent.prompts import get_prompt as get_entity_prompt
from agents.response_agent.prompts import get_prompt as get_response_prompt

def print_prompt(agent_name, prompt_template):
    try:
        # Access the system message template
        system_message_template = prompt_template.messages[0]
        content = system_message_template.prompt.template
        
        print(f"\n{'='*20} {agent_name} PROMPT {'='*20}\n")
        print(content)
        print(f"\n{'='*60}\n")
            
    except Exception as e:
        print(f"ERROR printing {agent_name}: {e}")

print_prompt("Intent Agent", get_intent_prompt())
print_prompt("Entity Agent", get_entity_prompt())
print_prompt("Response Agent", get_response_prompt())
