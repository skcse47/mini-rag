from functools import lru_cache

from rag.config import (
    LLM_PROVIDER,
    LLM_MODEL,
    LLM_MAX_TOKENS,
    OPENROUTER_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
)


@lru_cache(maxsize=1)
def get_chat_model():
    """
    Return the LangChain chat model selected by LLM_PROVIDER.
    Both providers support token streaming, which LangGraph
    forwards through stream_mode="messages".
    """

    if LLM_PROVIDER == "local":

        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=LLM_MODEL,
            keep_alive="10m",
            num_predict=LLM_MAX_TOKENS,
        )

    elif LLM_PROVIDER == "openrouter":

        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=OPENROUTER_MODEL,
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
            max_tokens=LLM_MAX_TOKENS,
            streaming=True,
        )

    else:

        raise ValueError(
            f"Unsupported LLM provider: {LLM_PROVIDER}"
        )
