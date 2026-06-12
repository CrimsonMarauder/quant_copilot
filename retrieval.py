"""
retrieval.py
------------
Thin orchestration layer between the agent and the vector database. It exposes a
single tool, retrieve_context(), that the agent can call to pull grounding text.
The vector store is built once (lazily) and reused.
"""

from __future__ import annotations
from vectorstore import LocalEmbedder, get_store
from build_index import build_if_empty

_store = None


def _get():
    global _store
    if _store is None:
        _store = get_store(LocalEmbedder())
        build_if_empty(_store)   # auto-build the index on first use
    return _store


def prewarm():
    """Build/load the vector DB ahead of time (called by the app at startup)."""
    return _get()


def retrieve_context(query: str, scope: str = "all") -> dict:
    """Retrieve grounding text from the knowledge base (company facts and method notes).

    Call this before explaining WHY a pair moves together (use scope="company") or
    before explaining a statistical method like cointegration or half-life
    (use scope="methodology"). Ground every explanation in what this returns.

    Args:
        query: What you want context about, e.g. "Visa Mastercard payment network overlap"
            or "what does the cointegration half-life mean".
        scope: One of "company", "methodology", or "all".

    Returns:
        Dict with the most relevant retrieved chunks and their source metadata.
    """
    where = None
    if scope == "company":
        where = {"type": "company"}
    elif scope == "methodology":
        where = {"type": "knowledge"}
    hits = _get().query(query, where=where)
    return {
        "query": query,
        "scope": scope,
        "chunks": [{"text": h["text"], "source": h["metadata"], "score": h["score"]} for h in hits],
    }
