from dotenv import load_dotenv
import os

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:0.5b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

TOP_K = int(os.getenv("TOP_K", 2))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 2))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 1))