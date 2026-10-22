from functools import lru_cache

from langchain_openai import OpenAIEmbeddings

from rag.config import (
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    EMBEDDING_BATCH_SIZE,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
)


@lru_cache(maxsize=1)
def get_embeddings():
    """
    LangChain embeddings client backed by OpenRouter's
    OpenAI-compatible embeddings endpoint.
    """

    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIMENSION,
        chunk_size=EMBEDDING_BATCH_SIZE,
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
        # OpenRouter expects raw strings, not tiktoken token ids.
        check_embedding_ctx_length=False,
    )


def create_embeddings(texts):
    """
    Generate embeddings for a string or list of strings.
    Always returns a list of vectors, in the same order as the input.
    """

    if isinstance(texts, str):
        texts = [texts]

    return get_embeddings().embed_documents(texts)
