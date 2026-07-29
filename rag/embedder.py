from ollama import embed


def create_embeddings(texts):
    """
    Generate embeddings for a string or list of strings.
    """

    response = embed(
        model="nomic-embed-text",
        input=texts
    )

    return response["embeddings"]