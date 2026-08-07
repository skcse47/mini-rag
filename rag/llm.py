from ollama import chat
from rag.config import LLM_MODEL
from rag.prompt import SYSTEM_PROMPT


def generate_answer(prompt):
    response = chat(
        model=LLM_MODEL,
        messages= [
            {
                "role": "user",
                "content": prompt
            },
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }
        ]
    )

    return response.message.content