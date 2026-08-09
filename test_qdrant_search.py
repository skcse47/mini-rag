from rag.embedder import create_embeddings
from rag.qdrant_store import search


query = "How many vacation days do employees get?"

query_embedding = create_embeddings(query)[0]

results = search(
    query_embedding,
    top_k=3
)

for result in results:

    print("\nScore:", result.score)

    print(
        "Text:",
        result.payload["text"]
    )

    print(
        "Metadata:",
        result.payload["metadata"]
    )