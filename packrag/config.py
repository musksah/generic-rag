"""Settings read from environment variables, with defaults for local development."""

import os
from pathlib import Path

# Google Cloud (Vertex AI). PROJECT_ID falls back to the ADC default project when unset.
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-005")

# Chroma server
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "state_of_the_union")

# Retrieval
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "10"))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "3"))
RERANK_MODEL = os.getenv("RERANK_MODEL", "ms-marco-MiniLM-L-12-v2")

DOCUMENT_PATH = os.getenv(
    "DOCUMENT_PATH",
    str(Path(__file__).resolve().parent.parent / "documents" / "2024_state_of_the_union.txt"),
)
