"""
build_index.py
--------------
The "ingestion" stage of the architecture, kept separate from serving.

It chunks the raw data into small documents and loads them into the vector DB:
  * each company becomes FOUR chunks (business / drivers / risks / peers),
  * each methodology note becomes one chunk,
each tagged with metadata so retrieval can filter by ticker, sector or type.

Run it directly to (re)build the database:   python build_index.py
The app also calls build_if_empty() automatically on first launch.
"""

from __future__ import annotations
import json
import os
from vectorstore import LocalEmbedder, get_store


def _load(path: str):
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, path), "r", encoding="utf-8") as f:
        return json.load(f)


def _chunks():
    ids, docs, metas = [], [], []

    companies = _load("companies.json")
    for c in companies:
        for facet in ("business", "drivers", "risks", "peers"):
            ids.append(f"{c['ticker']}-{facet}")
            docs.append(f"{c['name']} ({c['ticker']}) — {facet}: {c[facet]}")
            metas.append({"type": "company", "ticker": c["ticker"],
                          "sector": c["sector"], "facet": facet})

    for k in _load("knowledge.json"):
        ids.append(f"kb-{k['id']}")
        docs.append(f"{k['topic']}: {k['text']}")
        metas.append({"type": "knowledge", "ticker": "", "sector": "", "facet": k["topic"]})

    return ids, docs, metas


def build(store=None, force: bool = False):
    if store is None:
        store = get_store(LocalEmbedder())
    ids, docs, metas = _chunks()
    if force or store.count() == 0:
        # add in batches to be gentle on the embedding API
        B = 20
        for i in range(0, len(docs), B):
            store.add(ids[i:i + B], docs[i:i + B], metas[i:i + B])
    return store


def build_if_empty(store):
    if store.count() == 0:
        build(store)
    return store


if __name__ == "__main__":
    s = build(force=True)
    print(f"Indexed {s.count()} chunks into the vector store.")
