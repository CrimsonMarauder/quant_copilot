"""
vectorstore.py
--------------
The retrieval engine. This is the real "vector database" layer.

  * Embeddings: each chunk of text is turned into a vector with Gemini.
  * Storage: vectors live in ChromaDB, a persistent on-disk vector database,
    with metadata (ticker, sector, chunk type) so we can FILTER as well as search.
  * Hybrid retrieval: we combine semantic similarity (vectors) with a keyword
    overlap score, which is more robust than pure vector search for short queries.

If ChromaDB can't be imported in some environment, we fall back to an in-memory
NumPy store with the identical interface, so the app keeps working.
"""

from __future__ import annotations

# --- Streamlit Cloud ships an old system sqlite3; Chroma needs a newer one.
#     This swap (with pysqlite3-binary in requirements) is the standard fix. ---
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass

import os
import time
import numpy as np
from sentence_transformers import SentenceTransformer
import config


# ---------------------------------------------------------------- embeddings
class LocalEmbedder:
    """Turns text into vectors using a local sentence-transformers model running on CPU."""

    def __init__(self):
        # This will download the model weights on first run, then load from cache
        self.model = SentenceTransformer(config.EMBED_MODEL)

    def embed(self, texts: list[str]) -> list[list[float]]:
        # encode returns a numpy array, we need a list of lists of floats for Chroma
        embeddings = self.model.encode(texts)
        return embeddings.tolist()


def _keyword_score(query: str, doc: str) -> float:
    """Fraction of meaningful query words that appear in the document (0..1)."""
    qs = {w for w in query.lower().split() if len(w) > 2}
    if not qs:
        return 0.0
    ds = set(doc.lower().split())
    return len(qs & ds) / len(qs)


# ---------------------------------------------------------------- Chroma store
class ChromaStore:
    def __init__(self, embedder: LocalEmbedder):
        import chromadb
        self.embedder = embedder
        self.client = chromadb.PersistentClient(path=config.CHROMA_DIR)
        self.col = self.client.get_or_create_collection(
            name=config.COLLECTION, metadata={"hnsw:space": "cosine"}
        )

    def count(self) -> int:
        return self.col.count()

    def add(self, ids, docs, metadatas):
        embs = self.embedder.embed(docs)
        self.col.add(ids=ids, documents=docs, metadatas=metadatas, embeddings=embs)

    def query(self, text: str, k: int = config.TOP_K, where: dict | None = None):
        qv = self.embedder.embed([text])[0]
        # over-fetch, then re-rank with the hybrid score
        res = self.col.query(query_embeddings=[qv], n_results=min(k * 3, max(self.count(), 1)), where=where)
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        out = []
        for doc, meta, dist in zip(docs, metas, dists):
            sem = 1.0 - float(dist)  # cosine distance -> similarity
            kw = _keyword_score(text, doc)
            score = config.HYBRID_ALPHA * sem + (1 - config.HYBRID_ALPHA) * kw
            out.append({"text": doc, "metadata": meta, "semantic": round(sem, 3),
                        "keyword": round(kw, 3), "score": round(score, 3)})
        out.sort(key=lambda r: r["score"], reverse=True)
        return out[:k]


# ---------------------------------------------------------------- NumPy fallback
class NumpyStore:
    """Same interface as ChromaStore but keeps vectors in memory (no Chroma needed)."""

    def __init__(self, embedder: LocalEmbedder):
        self.embedder = embedder
        self.ids, self.docs, self.metas = [], [], []
        self.mat = None

    def count(self) -> int:
        return len(self.docs)

    def add(self, ids, docs, metadatas):
        embs = np.array(self.embedder.embed(docs), dtype="float32")
        embs /= (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
        self.mat = embs if self.mat is None else np.vstack([self.mat, embs])
        self.ids += list(ids); self.docs += list(docs); self.metas += list(metadatas)

    def query(self, text: str, k: int = config.TOP_K, where: dict | None = None):
        qv = np.array(self.embedder.embed([text])[0], dtype="float32")
        qv /= (np.linalg.norm(qv) + 1e-9)
        sims = self.mat @ qv
        order = np.argsort(sims)[::-1]
        out = []
        for i in order:
            meta = self.metas[i]
            if where and not all(meta.get(kk) == vv for kk, vv in where.items()):
                continue
            sem = float(sims[i]); kw = _keyword_score(text, self.docs[i])
            score = config.HYBRID_ALPHA * sem + (1 - config.HYBRID_ALPHA) * kw
            out.append({"text": self.docs[i], "metadata": meta, "semantic": round(sem, 3),
                        "keyword": round(kw, 3), "score": round(score, 3)})
            if len(out) >= k * 3:
                break
        out.sort(key=lambda r: r["score"], reverse=True)
        return out[:k]


def get_store(embedder: LocalEmbedder):
    """Return a Chroma-backed store if possible, else the NumPy fallback."""
    try:
        return ChromaStore(embedder)
    except Exception as e:
        print(f"[vectorstore] Chroma unavailable ({e}); using NumPy fallback.")
        return NumpyStore(embedder)
