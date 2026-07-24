from ollama import embed
import math


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
    model="nomic-embed-text",
    input=sentences
)

embeddings = response["embeddings"]


# query = "How many vacation days do employees get?"
query = "Can I work from home?"
queryResponse = embed(
    model="nomic-embed-text",
    input=query
)

queryEmbeddings = queryResponse["embeddings"]
results = []

for chunk, embedding in zip(sentences, embeddings):
    score = cosine_similarity(
        queryEmbeddings[0],
        embedding
    )

    results.append({
        "chunk": chunk,
        "score": score
    })

results.sort(
    key=lambda item: item["score"],
    reverse= True
)

top_k = 2

for val in results[:top_k]:
    print(
        val["score"], " -> ",
        val["chunk"]
    )