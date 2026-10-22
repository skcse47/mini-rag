from rag.graph import rag_graph


async def answer_question(question, document_id):
    """
    Run the RAG graph and yield streamed answer text, then sources.
    """

    sources = []

    async for mode, data in rag_graph.astream(
        {
            "question": question,
            "document_id": document_id
        },
        stream_mode=["messages", "updates"]
    ):

        # 1. Real token streaming from the LLM inside the generate node
        if mode == "messages":

            message_chunk, metadata = data

            if (
                metadata.get("langgraph_node") == "generate"
                and message_chunk.content
            ):
                yield {
                    "type": "text",
                    "text": message_chunk.content
                }

            continue

        # 2. Node outputs: keep retrieved chunks for sources and
        #    emit the fixed reply when nothing relevant was found
        if "retrieve" in data:

            sources = [
                {
                    "filename": document.metadata["filename"],
                    "page": document.metadata["page"],
                    "score": score
                }
                for document, score in data["retrieve"]["documents"]
            ]

        elif "no_answer" in data:

            yield {
                "type": "text",
                "text": data["no_answer"]["answer"]
            }

    # 3. Send sources
    yield {
        "type": "sources",
        "sources": sources
    }
