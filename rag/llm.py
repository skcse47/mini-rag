from ollama import chat
from rag.config import LLM_MODEL
from rag.prompt import SYSTEM_PROMPT

def generate_answer(prompt):

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

    # print("Model:", LLM_MODEL)
    # print(
    #     "Load duration:",
    #     response.get("load_duration")
    # )
    # print(
    #     "Prompt eval count:",
    #     response.get("prompt_eval_count")
    # )
    # print(
    #     "Prompt eval duration:",
    #     response.get("prompt_eval_duration")
    # )
    # print(
    #     "Eval count:",
    #     response.get("eval_count")
    # )
    # print(
    #     "Eval duration:",
    #     response.get("eval_duration")
    # )

    return response.message.content