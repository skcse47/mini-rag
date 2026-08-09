from rag.embedder import create_embeddings
# from rag.retriever import retrieve, retrieve_with_faiss
from rag.llm import generate_answer
from rag.config import TOP_K
from rag.prompt import build_rag_prompt
from rag.qdrant_store import search, close_qdrant


query = input("Ask a question: ")

query_embedding = create_embeddings(query)[0]

# results = retrieve(
#     query_embedding,
#     vector_store,
#     top_k=TOP_K
# )

# results = retrieve_with_faiss(
#     query_embedding,
#     top_k=TOP_K,
#     metadata_filters={
#         "department": "HR"
#     }
# )

results = search(
    query_embedding,
    top_k=TOP_K,
    document_id='fec13e14-10bf-4120-a382-20b62ba4d3f2'
)

# context = "\n".join(
#         result["record"]["text"]
#         for result in results
# )

context = "\n".join(
    result.payload["text"]
    for result in results
)

score = "\n".join(
    str(result.score)
    for result in results
)

# print('-'*10)
# print("Context:")
# print(context)


# print('-'*10)
# print("score:")
# print(score)

print("\n===== Retrieved Results =====")

for result in results:

    print(f"\nScore: {result.score}")

    print("Text:")
    print(result.payload["text"])

    print("Metadata:")
    print(result.payload["metadata"])

print("=============================\n")

prompt = build_rag_prompt(context, query)
answer = generate_answer(prompt)

print(answer)

close_qdrant()