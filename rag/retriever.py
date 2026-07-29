import math


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