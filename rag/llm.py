from rag.config import (LLM_PROVIDER,
    LLM_MODEL,
    OPENROUTER_MODEL,
    OPENROUTER_API_KEY
)
from rag.prompt import SYSTEM_PROMPT
from openai import OpenAI 


openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)


def generate_answer(prompt):

    if LLM_PROVIDER == "local":

        return generate_local_answer(prompt)

    elif LLM_PROVIDER == "openrouter":

        return generate_openrouter_answer(prompt)

    else:

        raise ValueError(
            f"Unsupported LLM provider: {LLM_PROVIDER}"
        )

def generate_local_answer(prompt):

    from ollama import chat

    response = chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        keep_alive="10m"
    )

    print("Provider: Ollama")
    print("Model:", LLM_MODEL)

    print(
        "Load duration:",
        response.get("load_duration")
    )

    print(
        "Prompt eval count:",
        response.get("prompt_eval_count")
    )

    print(
        "Prompt eval duration:",
        response.get("prompt_eval_duration")
    )

    print(
        "Eval count:",
        response.get("eval_count")
    )

    print(
        "Eval duration:",
        response.get("eval_duration")
    )

    return response.message.content

def generate_openrouter_answer(prompt):

    response = openrouter_client.chat.completions.create(
        model=OPENROUTER_MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    print("Provider: OpenRouter")
    print("Model:", OPENROUTER_MODEL)

    return response.choices[0].message.content