from dotenv import load_dotenv
import os

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", 100))
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", 768))

TOP_K = int(os.getenv("TOP_K", 2))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 2))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 1))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.60))

COLLECTION_NAME = str(os.getenv("COLLECTION_NAME", "company_documents"))
REGISTRY_COLLECTION_NAME = f"{COLLECTION_NAME}_registry"
QDRANT_PATH = os.getenv("QDRANT_PATH", "storage/qdrant")

LLM_PROVIDER = os.getenv(
    "LLM_PROVIDER",
    "local"
)

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "qwen2.5:3b-instruct"
)

LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 1024))

OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "openrouter/free"
)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

OPENROUTER_API_KEY = os.getenv(
    "OPENROUTER_API_KEY"
)
