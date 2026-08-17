SYSTEM_PROMPT = """
Answer only from the provided context.
If the answer is not present, reply exactly: I don't know.
Do not guess or use outside knowledge.
Be concise.
"""


def build_rag_prompt(context, question):

    return (
        f"Context:\n{context}\n\n"
        f"Question:\n{question}"
    )