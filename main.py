from rag.embedder import create_embeddings
from rag.vector_store import load_vector_store
from rag.retriever import retrieve
from rag.llm import generate_answer


vector_store = load_vector_store(
    "storage/vector_store.json"
)

query = input("Ask a question: ")

query_embedding = create_embeddings(query)[0]

results = retrieve(
    query_embedding,
    vector_store,
    top_k=2
)

context = "\n".join(
        result["record"]["text"]
        for result in results
)

answer = generate_answer(context, question=query)

print(answer)