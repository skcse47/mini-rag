from ollama import embed

response = embed(
    model="nomic-embed-text",
    input="I love Python."
)

response1 = embed(
    model="nomic-embed-text",
    input="I enjoy programming in Python."
)


response2 = embed(
    model="nomic-embed-text",
    input="The weather is sunny today."
)

# "I love Python."
# "I enjoy programming in Python."
# "The weather is sunny today."

print(type(response))
print(response["embeddings"][0][:10])
print(response1["embeddings"][0][:10])
print(response2["embeddings"][0][:10])

