from rag.embedder import create_embeddings
from rag.retriever import retrieve, retrieve_with_faiss
from rag.llm import generate_answer
from rag.config import TOP_K
from rag.prompt import build_rag_prompt


query = input("Ask a question: ")

query_embedding = create_embeddings(query)[0]

# results = retrieve(
#     query_embedding,
#     vector_store,
#     top_k=TOP_K
# )

results = retrieve_with_faiss(
    query_embedding,
    top_k=TOP_K
)

context = "\n".join(
        result["record"]["text"]
        for result in results
)

prompt = build_rag_prompt(context, query)
answer = generate_answer(prompt)

print(answer)