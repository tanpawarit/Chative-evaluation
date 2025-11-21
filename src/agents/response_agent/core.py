from typing import Optional

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_openai import ChatOpenAI

from agents.response_agent.prompts import get_prompt
from shared.tools.common_tools import COMMON_TOOLS
from shared.tools.knowledge_tools import KNOWLEDGE_TOOLS


def create_agent(model: Optional[ChatOpenAI] = None) -> AgentExecutor:
    """Create a LangChain agent wired with knowledge retrieval and utility tools."""
    prompt = get_prompt()
    llm = model or ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = KNOWLEDGE_TOOLS + COMMON_TOOLS
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools)


# Eagerly create a reusable agent executor instance
agent = create_agent()
