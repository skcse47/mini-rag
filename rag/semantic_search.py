from ollama import embed
import math
from rag.config import EMBEDDING_MODEL


def cosine_similarity(vec1, vec2):
    dot = sum(a * b for a, b in zip(vec1, vec2))

    mag_a = math.sqrt(sum(a * a for a in vec1))
    mag_b = math.sqrt(sum(b * b for b in vec2))

    return dot / (mag_a * mag_b)


sentences = [
    "Employees receive 24 paid leave days every year.",
    "Remote work is allowed two days per week.",
    "The office is open from 9 AM to 6 PM."
]

response = embed(
    model=EMBEDDING_MODEL,
    input=sentences
)

embeddings = response["embeddings"]


# query = "How many vacation days do employees get?"
query = "can work remotely?"
queryResponse = embed(
    model=EMBEDDING_MODEL,
    input=query
)

queryEmbeddings = queryResponse["embeddings"]


vector_store = []

for index, (chunk, embedding) in enumerate( zip(sentences, embeddings)):
    vector_store.append({
        "id": index,
        "text": chunk,
        "embedding": embedding,
        "metadata": {
            "source": "company_policy.txt"
        }
    })

# results = []

# for chunk, embedding in zip(sentences, embeddings):
#     score = cosine_similarity(
#         queryEmbeddings[0],
#         embedding
#     )

#     results.append({
#         "chunk": chunk,
#         "score": score
#     })

results = []

for record in vector_store:

    score = cosine_similarity(
        queryEmbeddings[0],
        record["embedding"]
    )

    results.append({
        "score": score,
        "record": record
    })

results.sort(
    key=lambda item: item["score"],
    reverse= True
)

top_k = 2

for result in results[:2]:

    record = result["record"]

    print(record["text"])
    print(record["metadata"])