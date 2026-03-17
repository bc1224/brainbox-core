"""
Embedding via Google Gemini multimodal embedding model.
"""

from google import genai
import config


def get_client():
    return genai.Client(api_key=config.GOOGLE_API_KEY)


def embed_texts(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    """
    Embed a batch of texts using Gemini embedding model.
    
    task_type should be:
      - "RETRIEVAL_DOCUMENT" when embedding docs for storage
      - "RETRIEVAL_QUERY" when embedding a user query
    """
    client = get_client()
    
    # Gemini embedding API supports batching
    result = client.models.embed_content(
        model=config.EMBEDDING_MODEL,
        contents=texts,
        config={
            "task_type": task_type,
            "output_dimensionality": config.EMBEDDING_DIMENSION,
        }
    )
    
    return [e.values for e in result.embeddings]


def embed_query(query: str) -> list[float]:
    """Embed a single query for retrieval."""
    results = embed_texts([query], task_type="RETRIEVAL_QUERY")
    return results[0]
