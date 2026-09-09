"""
store.py — ChromaDB collection management.

Owns exactly one rule: every uploaded document gets its own fresh,
uniquely-named collection, and the previous collection (if any) is deleted
first. This is what guarantees a new upload can never accidentally return
chunks left over from a document uploaded earlier in the same session —
there is no shared "documents" collection that things silently accumulate
into.

Uses an in-memory ChromaDB client (EphemeralClient): nothing is written to
disk and nothing requires a cloud account. Data lives only as long as the
Streamlit process is running.
"""

import uuid

import chromadb
from chromadb.api.models.Collection import Collection


def get_client() -> chromadb.ClientAPI:
    """One in-memory Chroma client per app process, created once and stashed
    in st.session_state by llm.py."""
    return chromadb.EphemeralClient()


def create_collection_for_upload(
    client: chromadb.ClientAPI,
    previous_collection_name: str | None = None,
) -> Collection:
    """Create a new, uniquely-named collection for a freshly uploaded
    document. If a previous collection name is passed in (i.e. this session
    already had a document indexed), it's deleted first.

    Called once per successful upload, from llm.py.
    """
    if previous_collection_name:
        try:
            client.delete_collection(previous_collection_name)
        except Exception:
            pass  # already gone, or never existed — either way, nothing to clean up

    collection_name = f"doc_{uuid.uuid4().hex}"
    return client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def add_chunks(collection: Collection, chunks, embeddings: list[list[float]]) -> None:
    """Add a document's chunks and their embeddings to its collection.

    Each chunk's source file and location are stored as metadata — this is
    what retrieve.py reads back to attach a citation to a retrieved chunk.
    """
    collection.add(
        ids=[f"chunk_{c.chunk_index}" for c in chunks],
        embeddings=embeddings,
        documents=[c.text for c in chunks],
        metadatas=[
            {"source_file": c.source_file, "location": c.location}
            for c in chunks
        ],
    )
