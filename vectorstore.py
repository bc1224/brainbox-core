"""
Pinecone vector store operations.
"""

from pinecone import Pinecone, ServerlessSpec
import config

_pc = None
_index = None


def get_pinecone():
    global _pc
    if _pc is None:
        _pc = Pinecone(api_key=config.PINECONE_API_KEY)
    return _pc


def get_index():
    """Get or create the Pinecone index."""
    global _index
    if _index is not None:
        return _index
    
    pc = get_pinecone()
    
    # Check if index exists
    existing = [idx.name for idx in pc.list_indexes()]
    
    if config.PINECONE_INDEX_NAME not in existing:
        pc.create_index(
            name=config.PINECONE_INDEX_NAME,
            dimension=config.EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
    
    _index = pc.Index(config.PINECONE_INDEX_NAME)
    return _index


def upsert_vectors(vectors: list[dict], namespace: str = "default"):
    """
    Upsert vectors to Pinecone.
    Each vector dict should have: id, values, metadata
    """
    index = get_index()
    
    # Pinecone batch limit is 100
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        records = [
            {
                "id": v["id"],
                "values": v["values"],
                "metadata": v["metadata"],
            }
            for v in batch
        ]
        index.upsert(vectors=records, namespace=namespace)


def query_vectors(query_embedding: list[float], top_k: int = 5, namespace: str = "default") -> list[dict]:
    """
    Query Pinecone for similar vectors.
    Returns list of matches with score and metadata.
    """
    index = get_index()
    
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace,
    )
    
    return [
        {
            "id": match.id,
            "score": match.score,
            "metadata": match.metadata,
        }
        for match in results.matches
    ]


def get_stats():
    """Get index stats."""
    index = get_index()
    return index.describe_index_stats()


def delete_all(namespace: str = "default"):
    """Delete all vectors in a namespace."""
    index = get_index()
    index.delete(delete_all=True, namespace=namespace)
