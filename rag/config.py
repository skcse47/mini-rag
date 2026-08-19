from dotenv import load_dotenv
import os

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:0.5b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

TOP_K = int(os.getenv("TOP_K", 2))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 2))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 1))
COLLECTION_NAME = str(os.getenv("COLLECTION_NAME","company_documents"))
EMBEDDING_DIMENSION  = int(os.getenv("EMBEDDING_DIMENSION", 768))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.60))
LLM_PROVIDER = os.getenv(
    "LLM_PROVIDER",
    "local"
)

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "qwen2.5:3b-instruct"
)

OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "openrouter/free"
)

OPENROUTER_API_KEY = os.getenv(
    "OPENROUTER_API_KEY"
)