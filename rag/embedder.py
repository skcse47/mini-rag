from ollama import embed
from rag.config import EMBEDDING_MODEL

def create_embeddings(texts):
    """
    Generate embeddings for a string or list of strings.
    """

    response = embed(
        model=EMBEDDING_MODEL,
        input=texts
    )

    return response["embeddings"]