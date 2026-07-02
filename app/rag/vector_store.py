"""
vector_store.py — Chroma vector store setup.
Embeds paper chunks using sentence-transformers (local, free).
Persists to disk in ./chroma_db/.
"""
from __future__ import annotations

import os
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "chroma_db")
COLLECTION_NAME = "research_papers"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Singletons (lazy-loaded)
# ---------------------------------------------------------------------------
_client: Optional[chromadb.PersistentClient] = None
_collection = None
_embedder: Optional[SentenceTransformer] = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedder


def _get_collection():
    global _client, _collection
    if _client is None:
        _client = chromadb.PersistentClient(path=os.path.abspath(CHROMA_PATH))
    if _collection is None:
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clear_collection() -> None:
    """Delete and recreate the collection (called before each new digest session)."""
    global _client, _collection
    if _client is None:
        _client = chromadb.PersistentClient(path=os.path.abspath(CHROMA_PATH))
    try:
        _client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    _collection = _client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def embed_papers(papers: List[Dict[str, Any]]) -> None:
    """
    Embed each paper (abstract + summary + critique) and upsert into Chroma.
    Each paper becomes one document chunk.
    """
    collection = _get_collection()
    embedder = _get_embedder()

    ids, docs, metas, embeddings = [], [], [], []

    for i, paper in enumerate(papers):
        # Compose a rich text blob for embedding
        critique = paper.get("critique") or {}
        text = (
            f"Title: {paper['title']}\n"
            f"Authors: {', '.join(paper.get('authors', []))}\n"
            f"Published: {paper.get('published', '')}\n\n"
            f"Abstract: {paper.get('abstract', paper.get('summary', ''))}\n\n"
            f"Summary: {paper.get('summary', '')}\n\n"
            f"Limitations: {critique.get('limitations', '')}\n"
            f"Target Reader: {critique.get('target_reader', '')}\n"
            f"Relevance Score: {critique.get('relevance_score', 'N/A')}"
        )
        emb = embedder.encode(text, normalize_embeddings=True).tolist()
        doc_id = f"paper_{i}"
        ids.append(doc_id)
        docs.append(text)
        metas.append(
            {
                "title": paper["title"],
                "link": paper.get("link", ""),
                "relevance_score": str(critique.get("relevance_score", 0)),
            }
        )
        embeddings.append(emb)

    if ids:
        collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)


def query_similar(question: str, top_k: int = 4) -> List[Dict[str, Any]]:
    """
    Retrieve top-k chunks relevant to the question.
    Returns list of dicts with 'document', 'metadata', 'distance'.
    """
    collection = _get_collection()
    embedder = _get_embedder()

    count = collection.count()
    if count == 0:
        return []

    emb = embedder.encode(question, normalize_embeddings=True).tolist()
    results = collection.query(
        query_embeddings=[emb],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )

    output = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(docs, metas, dists):
        output.append({"document": doc, "metadata": meta, "distance": dist})

    return output
