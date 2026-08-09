from rag.chunker import chunk_text
from rag.embedder import create_embeddings
from rag.qdrant_store import (
    create_collection,
    insert_documents
)


with open(
    "data/company",
    "r",
    encoding="utf-8"
) as file:

    text = file.read()


chunks = chunk_text(
    text,
    2,
    1
)

embeddings = create_embeddings(chunks)

create_collection()

insert_documents(
    chunks,
    embeddings
)

print("Qdrant index created successfully.")