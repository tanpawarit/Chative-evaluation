from typing import Optional

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from pkg.openrouter.client import get_chat_openrouter

from agents.response_agent.prompts import get_prompt
from shared.tools.common_tools import COMMON_TOOLS
from shared.tools.knowledge_tools import KNOWLEDGE_TOOLS


def create_agent(model: Optional[ChatOpenAI] = None) -> AgentExecutor:
    """Create a LangChain agent wired with knowledge retrieval and utility tools."""
    prompt = get_prompt()
    llm = model or get_chat_openrouter(
        model="x-ai/grok-4.1-fast:free", 
        temperature=0.5,
        extra_body={"include_reasoning": False}
    )
    tools = KNOWLEDGE_TOOLS + COMMON_TOOLS
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools)


# Eagerly create a reusable agent executor instance
agent = create_agent()
