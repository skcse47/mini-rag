from ollama import chat
from rag.config import LLM_MODEL

def generate_answer(context, question):
    prompt = f"""
    You are a question-answering assistant.

    Answer strictly ONLY using the provided context.

    If the answer cannot be found in the context, reply exactly:

    I don't know.

    Do not use outside knowledge.
    Do not guess.
    Do not explain why you are guessing.


    Context:
    {context}

    Question:
    {question}
    """

    response = chat(
        model=LLM_MODEL,
        messages= [
            {
                "role": "user",
                "content": prompt
            },
             {
                "role": "system",
                "content": (
                    "You answer ONLY from the provided context. "
                    "If the answer is not in the context, reply exactly: "
                    "'I don't know.'"
                )
            }
        ]
    )

    return response.message.content