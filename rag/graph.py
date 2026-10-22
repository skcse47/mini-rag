import asyncio
from typing import TypedDict

from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END

from rag.config import TOP_K, SIMILARITY_THRESHOLD
from rag.llm import get_chat_model
from rag.prompt import RAG_PROMPT, NO_ANSWER
from rag.qdrant_store import search


class RAGState(TypedDict, total=False):
    question: str
    document_id: str
    documents: list[tuple[Document, float]]
    answer: str


async def retrieve(state: RAGState):
    """
    Embed the question and search only inside the selected document.
    Chunks below SIMILARITY_THRESHOLD are dropped.
    """

    # The Qdrant client and embedding call are synchronous.
    results = await asyncio.to_thread(
        search,
        state["question"],
        top_k=TOP_K,
        document_id=state.get("document_id")
    )

    return {
        "documents": [
            (document, score)
            for document, score in results
            if score >= SIMILARITY_THRESHOLD
        ]
    }


def route_after_retrieve(state: RAGState):

    return "generate" if state["documents"] else "no_answer"


async def generate(state: RAGState):

    context = "\n\n".join(
        document.page_content
        for document, _ in state["documents"]
    )

    chain = RAG_PROMPT | get_chat_model()

    # Tokens are streamed to callers through stream_mode="messages".
    response = await chain.ainvoke({
        "context": context,
        "question": state["question"]
    })

    return {"answer": response.content}


def no_answer(state: RAGState):

    return {"answer": NO_ANSWER}


def build_graph():

    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("no_answer", no_answer)

    graph.add_edge(START, "retrieve")
    graph.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        ["generate", "no_answer"]
    )
    graph.add_edge("generate", END)
    graph.add_edge("no_answer", END)

    return graph.compile()


rag_graph = build_graph()
