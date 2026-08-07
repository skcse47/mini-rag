from rag.embedder import create_embeddings
from rag.vector_store import load_vector_store
from rag.retriever import retrieve
from rag.llm import generate_answer
from rag.config import TOP_K
from rag.prompt import build_rag_prompt



vector_store = load_vector_store(
    "storage/vector_store.json"
)

query = input("Ask a question: ")

query_embedding = create_embeddings(query)[0]

results = retrieve(
    query_embedding,
    vector_store,
    top_k=TOP_K
)

context = "\n".join(
        result["record"]["text"]
        for result in results
)
print("\n===== Retrieved Context =====")
print(context)
print("=============================\n")

prompt = build_rag_prompt(context, query)
answer = generate_answer(prompt)

print(answer)