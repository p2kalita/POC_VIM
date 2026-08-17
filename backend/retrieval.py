"""
retrieval.py — Semantic retrieval from FAISS index
Queries the invoice vector store and returns relevant context chunks.
"""

import os
import logging
from typing import Optional

import numpy as np
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_EMBEDDING_MODEL: str = os.getenv(
    "GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001"
)

TOP_K = 5


def _embed_query(query: str) -> np.ndarray:
    """Embed a user query and L2-normalise for cosine similarity."""
    import google.generativeai as genai  # type: ignore

    genai.configure(api_key=GEMINI_API_KEY)
    result = genai.embed_content(
        model=GEMINI_EMBEDDING_MODEL,
        content=query,
        task_type="RETRIEVAL_QUERY",
    )
    vec = np.array(result["embedding"], dtype=np.float32).reshape(1, -1)
    norm = np.linalg.norm(vec)
    return vec / (norm if norm > 0 else 1e-10)


def _load_store():
    """Import and return (index, metadata) from the FAISS store."""
    # Import here to avoid circular import with ingest.py
    from ingest import _load_store as _ls
    return _ls()


def retrieve_context(
    query: str,
    top_k: int = TOP_K,
    filter_doc_id: Optional[str] = None,
) -> str:
    """
    Embed the query, search FAISS, and return formatted context string.

    Args:
        query:          The user's question.
        top_k:          Number of chunks to retrieve.
        filter_doc_id:  If set, restrict search to a specific invoice.

    Returns:
        Formatted string with relevant invoice excerpts and source info.
    """
    index, metadata = _load_store()

    if index is None or not metadata:
        return ""

    query_vec = _embed_query(query)

    if filter_doc_id:
        # Filter metadata to target doc only
        target_metas = [m for m in metadata if m.get("doc_id") == filter_doc_id]
        if not target_metas:
            return ""

        # Get original positions of these chunks in the index
        target_positions = [i for i, m in enumerate(metadata)
                            if m.get("doc_id") == filter_doc_id]

        # Rebuild a temporary sub-index for the filtered chunks
        import faiss  # type: ignore

        sub_vecs_list = []
        for i, m in enumerate(metadata):
            if m.get("doc_id") == filter_doc_id:
                sub_vecs_list.append(i)

        if not sub_vecs_list:
            return ""

        # Reconstruct vectors from the index for the filtered subset
        dim = index.d
        all_vecs = np.zeros((index.ntotal, dim), dtype=np.float32)
        index.reconstruct_n(0, index.ntotal, all_vecs)

        sub_vecs = all_vecs[sub_vecs_list]
        sub_index = faiss.IndexFlatIP(dim)
        sub_index.add(sub_vecs)

        k = min(top_k, len(sub_vecs_list))
        scores, sub_ids = sub_index.search(query_vec, k)

        hits = [
            (target_metas[sid], float(scores[0][j]))
            for j, sid in enumerate(sub_ids[0])
            if sid >= 0
        ]
    else:
        k = min(top_k, index.ntotal)
        scores, ids = index.search(query_vec, k)
        hits = [
            (metadata[idx], float(scores[0][i]))
            for i, idx in enumerate(ids[0])
            if idx >= 0 and idx < len(metadata)
        ]

    if not hits:
        return ""

    # Format context blocks
    parts: list[str] = []
    for meta, score in hits:
        filename   = meta.get("filename", "unknown")
        page       = meta.get("page", 1)
        relevance  = round(score * 100, 1)
        text       = meta.get("text", "")
        parts.append(
            f"--- [{filename} | Page {page} | Relevance: {relevance}%] ---\n{text}"
        )

    return "\n\n".join(parts)
