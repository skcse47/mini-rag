from rag.qdrant_store import search


query = "How many vacation days do employees get?"

results = search(
    query,
    top_k=3
)

for document, score in results:

    print("\nScore:", score)

    print(
        "Text:",
        document.page_content
    )

    print(
        "Metadata:",
        document.metadata
    )