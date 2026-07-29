from ollama import chat


def generate_answer(context, question):
    prompt = f"""
    You are a question-answering assistant.

    Answer ONLY using the provided context.

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
        model="qwen2.5:0.5b",
        messages= [
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.message.content