from openai import OpenAI

from rag.config import (
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    EMBEDDING_BATCH_SIZE,
    OPENROUTER_API_KEY,
)

embedding_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)


def create_embeddings(texts):
    """
    Generate embeddings for a string or list of strings via the OpenRouter API.
    Always returns a list of vectors, in the same order as the input.
    """

    if isinstance(texts, str):
        texts = [texts]

    embeddings = []

    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[start:start + EMBEDDING_BATCH_SIZE]

        response = embedding_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
            dimensions=EMBEDDING_DIMENSION,
        )

        ordered = sorted(response.data, key=lambda item: item.index)
        embeddings.extend(item.embedding for item in ordered)

    return embeddings
