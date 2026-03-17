import os
from dotenv import load_dotenv

load_dotenv(override=True)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-brain")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# Gemini embedding model — multimodal
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSION = 3072  # dimension for this model

def validate():
    """Check all required keys are set."""
    missing = []
    if not ANTHROPIC_API_KEY or "xxxxx" in ANTHROPIC_API_KEY:
        missing.append("ANTHROPIC_API_KEY")
    if not GOOGLE_API_KEY or "xxxxx" in GOOGLE_API_KEY:
        missing.append("GOOGLE_API_KEY")
    if not PINECONE_API_KEY or "xxxxx" in PINECONE_API_KEY:
        missing.append("PINECONE_API_KEY")
    if missing:
        raise ValueError(
            f"Missing or placeholder API keys: {', '.join(missing)}\n"
            f"Copy .env.example to .env and fill in your keys."
        )
