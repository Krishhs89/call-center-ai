"""
ChromaDB-backed vector store for semantic retrieval of past call summaries.
Enables RAG (Retrieval-Augmented Generation) in the call processing pipeline.
Gracefully degrades if ChromaDB or OpenAI embeddings are unavailable.
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

VECTOR_DB_DIR = Path(__file__).parent.parent / "data" / "vector_db"


def _get_collection():
    """Return ChromaDB collection, or None if unavailable."""
    try:
        import chromadb
        from chromadb.config import Settings

        VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(
            path=str(VECTOR_DB_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        return client.get_or_create_collection(
            name="call_summaries",
            metadata={"hnsw:space": "cosine"},
        )
    except Exception as e:
        logger.warning(f"[VECTORDB] ChromaDB unavailable: {e}")
        return None


def _get_embed_fn():
    """Return ChromaDB OpenAI embedding function, or None if unavailable."""
    try:
        from config.settings import settings
        if not settings.OPENAI_API_KEY:
            logger.warning("[VECTORDB] OPENAI_API_KEY not set — embeddings unavailable")
            return None
        from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
        return OpenAIEmbeddingFunction(
            api_key=settings.OPENAI_API_KEY,
            model_name="text-embedding-3-small",
        )
    except Exception as e:
        logger.warning(f"[VECTORDB] Embedding function unavailable: {e}")
        return None


def store_call_embedding(
    call_id: str,
    transcript: str,
    summary: str,
    metadata: dict,
) -> bool:
    """
    Store a processed call's embedding in ChromaDB for future RAG retrieval.

    Args:
        call_id: Unique call identifier
        transcript: Normalized transcript text
        summary: Generated summary text
        metadata: Dict with category, resolution_status, overall_score, llm_name

    Returns:
        True if stored successfully
    """
    collection = _get_collection()
    embed_fn = _get_embed_fn()
    if collection is None or embed_fn is None:
        return False

    try:
        doc = f"{summary}\n\nTranscript: {transcript[:600]}"
        embedding = embed_fn([doc])[0]

        clean_meta = {
            "call_id": call_id,
            "category": str(metadata.get("category", "unknown")),
            "resolution_status": str(metadata.get("resolution_status", "unknown")),
            "overall_score": float(metadata.get("overall_score", 0.0)),
            "llm_name": str(metadata.get("llm_name", "unknown")),
            "summary_snippet": summary[:400],
        }

        collection.upsert(
            ids=[call_id],
            embeddings=[embedding],
            documents=[doc],
            metadatas=[clean_meta],
        )
        logger.info(f"[VECTORDB] Stored embedding for call {call_id}")
        return True
    except Exception as e:
        logger.warning(f"[VECTORDB] Failed to store embedding for {call_id}: {e}")
        return False


def retrieve_similar_calls(
    transcript: str,
    top_k: int = 3,
    exclude_call_id: Optional[str] = None,
) -> list[dict]:
    """
    Retrieve semantically similar past calls from ChromaDB.

    Args:
        transcript: Current transcript to find similar calls for
        top_k: Max number of results to return
        exclude_call_id: Current call ID to exclude from results

    Returns:
        List of dicts: call_id, summary_snippet, resolution_status,
                       overall_score, category, distance
    """
    collection = _get_collection()
    embed_fn = _get_embed_fn()
    if collection is None or embed_fn is None:
        return []

    try:
        count = collection.count()
        if count == 0:
            logger.info("[VECTORDB] Vector store empty — skipping RAG retrieval")
            return []

        query_embedding = embed_fn([transcript[:800]])[0]
        n = min(top_k + 1, count)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n,
            include=["metadatas", "distances"],
        )

        similar = []
        for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
            if exclude_call_id and meta.get("call_id") == exclude_call_id:
                continue
            similar.append({
                "call_id": meta.get("call_id"),
                "summary_snippet": meta.get("summary_snippet", ""),
                "resolution_status": meta.get("resolution_status"),
                "overall_score": meta.get("overall_score"),
                "category": meta.get("category"),
                "distance": round(dist, 3),
            })
            if len(similar) >= top_k:
                break

        logger.info(f"[VECTORDB] Retrieved {len(similar)} similar call(s)")
        return similar
    except Exception as e:
        logger.warning(f"[VECTORDB] Retrieval failed: {e}")
        return []


def get_vector_store_stats() -> dict:
    """Return stats about the vector store for UI display."""
    collection = _get_collection()
    if collection is None:
        return {"available": False, "count": 0}
    try:
        return {"available": True, "count": collection.count()}
    except Exception:
        return {"available": False, "count": 0}
