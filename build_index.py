from rag.chunker import chunk_text
from rag.embedder import create_embeddings
from rag.vector_store import save_vector_store
from rag.config import CHUNK_SIZE, CHUNK_OVERLAP



with open("data/company", "r", encoding="utf-8") as file:
    text = file.read()

chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

embeddings = create_embeddings(chunks)

vector_store = []

for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):

    vector_store.append({
        "id": idx,
        "text": chunk,
        "embedding": embedding,
        "metadata": {
            "source": "company.txt"
        }
    })

save_vector_store(
    vector_store,
    "storage/vector_store.json"
)

print("Index created successfully.")