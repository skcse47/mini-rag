import math
import faiss
import numpy as np
from rag.vector_store import load_vector_store

vector_store = load_vector_store(
    "storage/vector_store.json"
)

index = faiss.read_index("storage/faiss.index")

def retrieve_with_faiss(query_embedding, top_k=3):
    query_vector = np.array(query_embedding, dtype="float32").reshape(1, -1)

    faiss.normalize_L2(query_vector)

    distances, indices = index.search(query_vector, top_k)

    results = []

    for idx, distance in zip(indices[0], distances[0]):

        if idx == -1:
            continue

        if idx < len(vector_store):
            results.append({
                "score": float(distance),
                "record": vector_store[idx]
            })

    return results
def cosine_similarity(vec1, vec2):
    dot = sum(a * b for a, b in zip(vec1, vec2))

    mag_a = math.sqrt(sum(a * a for a in vec1))
    mag_b = math.sqrt(sum(b * b for b in vec2))

    return dot / (mag_a * mag_b)


def retrieve(query_embedding, vector_store, top_k=3):

    results = []

    for record in vector_store:

        score = cosine_similarity(
            query_embedding,
            record["embedding"]
        )

        results.append({
            "score": score,
            "record": record
        })

    results.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return results[:top_k]