"""
retriever.py — RAG Retrieval Module
=====================================
Responsibilities:
  1. Accept a plain-text user query
  2. Embed the query with the SAME model used during ingestion
  3. Run cosine-similarity search against ChromaDB
  4. Return the top-k most relevant text chunks + metadata

This module is intentionally decoupled from:
  - Any LLM / answer generation
  - Any UI layer
  - Any ingestion logic
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import List, cast

from openai import AzureOpenAI
import chromadb
from chromadb.config import Settings
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Shared configuration (must match ingest.py) ───────────────────────────────
VECTOR_DB_PATH       = Path(__file__).parent.parent / "vector_store"
COLLECTION_NAME      = "rag_documents"
EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
DEFAULT_TOP_K        = 5


# ── Result container ──────────────────────────────────────────────────────────
@dataclass
class RetrievedChunk:
    """A single chunk returned from the vector store."""
    text     : str
    source   : str          = "unknown"
    chunk_idx: int          = 0
    score    : float        = 0.0          # cosine distance (lower = more similar)
    metadata : dict         = field(default_factory=dict)

    @property
    def relevance(self) -> float:
        """Convert cosine *distance* → relevance score [0, 1]."""
        return round(1.0 - self.score, 4)


# ── Lazy-loaded singletons (initialised once per process) ─────────────────────
_embed_client : AzureOpenAI        | None = None
_collection   : chromadb.Collection | None = None


def _get_embed_client() -> AzureOpenAI:
    global _embed_client
    if _embed_client is None:
        ep  = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        key = os.getenv("AZURE_OPENAI_API_KEY", "")
        ver = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        if not ep or not key:
            raise RuntimeError("Azure OpenAI credentials not set in .env")
        print(f"[Retriever] Using Azure embedding model: {EMBEDDING_DEPLOYMENT}")
        _embed_client = AzureOpenAI(azure_endpoint=ep, api_key=key, api_version=ver)
    return _embed_client


def _get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(
            path=str(VECTOR_DB_PATH),
            settings=Settings(anonymized_telemetry=False),
        )
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


# ── Public API ────────────────────────────────────────────────────────────────
def retrieve(query: str, top_k: int = DEFAULT_TOP_K) -> List[RetrievedChunk]:
    """
    Search the vector store for chunks most relevant to *query*.

    How it works:
    ┌──────────────────────────────────────────────────────────────┐
    │  query (string)                                              │
    │    │                                                         │
    │    ▼                                                         │
    │  AzureOpenAI.embeddings.create()  →  query_vector (1536-dim)│
    │    │                                                         │
    │    ▼                                                         │
    │  ChromaDB.query()  ──cosine sim──►  top-k chunk vectors     │
    │    │                                                         │
    │    ▼                                                         │
    │  List[RetrievedChunk]  (text + metadata + score)            │
    └──────────────────────────────────────────────────────────────┘
    """
    if not query.strip():
        return []

    client     = _get_embed_client()
    collection = _get_collection()

    # Check there is anything to search
    count = collection.count()
    if count == 0:
        print("[Retriever] ⚠  Vector store is empty — run ingest.py first.")
        return []

    # Embed the user query via Azure
    response      = client.embeddings.create(input=[query], model=EMBEDDING_DEPLOYMENT)
    raw_embedding = response.data[0].embedding   # List[float] — native ChromaDB type

    # Query ChromaDB — ask for more than top_k in case some are filtered later
    results = collection.query(
        query_embeddings=[raw_embedding],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )

    # Unpack the nested lists ChromaDB returns (one list per query)
    docs  = (results["documents"]  or [[]])[0]
    metas = (results["metadatas"]  or [[]])[0]
    dists = (results["distances"]  or [[]])[0]

    chunks: List[RetrievedChunk] = []
    for doc, meta, dist in zip(docs, metas, dists):
        if meta is None:
            continue
        chunks.append(
            RetrievedChunk(
                text      = doc,
                source    = str(meta.get("source", "unknown")),
                chunk_idx = cast(int, meta.get("chunk_index", 0)),
                score     = round(float(dist), 6),
                metadata  = dict(meta),
            )
        )

    return chunks


def format_context(chunks: List[RetrievedChunk], separator: str = "\n\n---\n\n") -> str:
    """
    Merge retrieved chunks into a single context string for the LLM prompt.
    Each chunk is prefixed with its source filename for transparency.
    """
    parts = [f"[Source: {c.source}]\n{c.text}" for c in chunks]
    return separator.join(parts)


# ── Quick smoke-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) or "What is this document about?"
    print(f"\n🔍  Query: {query}\n")
    results = retrieve(query)
    for i, chunk in enumerate(results, 1):
        print(f"── Result {i} (relevance={chunk.relevance}) ──────────────────")
        print(f"Source : {chunk.source}  [chunk #{chunk.chunk_idx}]")
        print(chunk.text[:300])
        print()
