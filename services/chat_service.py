from rag.embedder import create_embeddings
from rag.qdrant_store import search
from rag.prompt import build_rag_prompt
from rag.llm import generate_answer
from rag.config import TOP_K, SIMILARITY_THRESHOLD
import asyncio
import time


async def answer_question(question, document_id):
    start = time.perf_counter()
    # 1. Convert question into embedding
    query_embedding = create_embeddings(question)[0]

    embedding_time = time.perf_counter() - start

    retrieval_start = time.perf_counter()
    # 2. Search only inside selected document
    results = search(
        query_embedding,
        top_k=TOP_K,
        document_id=document_id
    )

    results = [
        result
        for result in results
        if result.score >= SIMILARITY_THRESHOLD
    ]
    
    # if not results:
    #     return {
    #         "answer": "I don't know.",
    #         "sources": []
    #     }
    
    retrieval_time = time.perf_counter() - retrieval_start
    # 3. Build context
    context = "\n\n".join(
        result.payload["text"]
        for result in results
    )

    # for result in results:
    #     print("Score:", result.score)
    #     print("Text:", result.payload["text"])
    #     print()
    # 4. Build RAG prompt
    prompt = build_rag_prompt(
        context,
        question
    )

    prompt_time = time.perf_counter() - retrieval_start

    llm_start = time.perf_counter()

    # 5. Generate complete answer
    answer = generate_answer(prompt)

    llm_time = time.perf_counter() - llm_start

#     print(f"""
# Embedding:  {embedding_time:.2f}s
# Retrieval:  {retrieval_time:.2f}s
# Prompt:     {prompt_time:.2f}s
# LLM:        {llm_time:.2f}s
# """)
#     print("Prompt characters:", len(prompt))
#     print("Context characters:", len(context))
    # 6. Simulate streaming
    for word in answer.split():
        yield {
            "type": "text",
            "text": word + " "
        }

        await asyncio.sleep(0.03)

    # 7. Send sources
    yield {
        "type": "sources",
        "sources": [
            {
                "filename": result.payload["metadata"]["filename"],
                "page": result.payload["metadata"]["page"],
                "score": result.score
            }
            for result in results
        ]
    }