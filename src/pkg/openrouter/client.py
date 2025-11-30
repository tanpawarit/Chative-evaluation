import os
from typing import Optional

from langchain_openai import ChatOpenAI


def get_chat_openrouter(
    model: str = "openai/gpt-4o-mini",
    temperature: float = 0.5,
    **kwargs,
) -> ChatOpenAI:
    """
    Create a ChatOpenAI client configured for OpenRouter.
    
    Args:
        model: The model name to use (default: "openai/gpt-4o-mini").
        temperature: The temperature to use (default: 0.5).
        **kwargs: Additional arguments to pass to ChatOpenAI.
        
    Returns:
        A configured ChatOpenAI instance.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is not set")

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        **kwargs,
    )
