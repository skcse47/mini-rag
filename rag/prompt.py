from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """
Answer only from the provided context.
If the answer is not present, reply exactly: I don't know.
Do not guess or use outside knowledge.
Be concise.
"""

NO_ANSWER = "I don't know."

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Context:\n{context}\n\nQuestion:\n{question}"),
])
