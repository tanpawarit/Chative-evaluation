from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from agents.intent_agent.prompts import get_prompt


def create_agent():
    """Create a simple LangChain agent for DeepTeam red teaming."""
    prompt = get_prompt()
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    parser = StrOutputParser()
    return prompt | model | parser


# Eagerly create a reusable agent instance
agent = create_agent()
