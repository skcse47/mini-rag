from rag.embedder import create_embeddings
from rag.qdrant_store import search
from rag.prompt import build_rag_prompt
from rag.llm import generate_answer
from rag.config import TOP_K


def answer_question(question, document_id):

    # 1. Convert question into embedding
    query_embedding = create_embeddings(question)[0]

    # 2. Search only inside selected document
    results = search(
        query_embedding,
        top_k=TOP_K,
        document_id=document_id
    )

    # 3. Build context
    context = "\n\n".join(
        result.payload["text"]
        for result in results
    )

    # 4. Build RAG prompt
    prompt = build_rag_prompt(
        context,
        question
    )

    # 5. Ask Qwen
    answer = generate_answer(prompt)

    return {
        "answer": answer,
        "sources": [
            {
                "filename": result.payload["metadata"]["filename"],
                "page": result.payload["metadata"]["page"],
                "score": result.score
            }
            for result in results
        ]
    }