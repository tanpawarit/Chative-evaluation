from langchain_core.output_parsers import StrOutputParser

from agents.response_agent.prompts import get_prompt
from pkg.openrouter.client import get_chat_openrouter


def create_agent():
    """Create a simple LangChain agent for DeepTeam red teaming."""
    prompt = get_prompt()
    model = get_chat_openrouter(model="openai/gpt-4o-mini", temperature=0)
    parser = StrOutputParser()
    return prompt | model | parser


# Eagerly create a reusable agent instance
agent = create_agent()
