"""
retrieve.py — Top-k retrieval by cosine similarity.

Embeds the user's query with the same model used for the document's chunks
(embed.py), asks the document's ChromaDB collection for the k nearest
chunks, and returns them as RetrievedChunk objects carrying the similarity
score plus the source_file/location metadata that store.py attached at
index time. llm.py renders these directly in the "retrieved chunks" panel,
and generate.py consumes the same list to build the prompt and citations.
"""

from dataclasses import dataclass

from chromadb.api.models.Collection import Collection

from embed import embed_query

DEFAULT_TOP_K = 4


@dataclass
class RetrievedChunk:
    text: str
    source_file: str
    location: str
    similarity: float  # cosine similarity in [-1, 1]; 1.0 = same direction as the query vector


def retrieve(collection: Collection, query: str, top_k: int = DEFAULT_TOP_K) -> list[RetrievedChunk]:
    """Return the top_k chunks most similar to `query`, ranked highest
    similarity first.

    Caps n_results at the collection's actual chunk count so a small demo
    document (fewer chunks than top_k) doesn't raise an error.
    """
    query_embedding = embed_query(query)
    n_results = min(top_k, collection.count())
    if n_results == 0:
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    retrieved = []
    for doc, meta, distance in zip(documents, metadatas, distances):
        # In Chroma's cosine space, distance == 1 - cosine_similarity.
        similarity = 1 - distance
        retrieved.append(RetrievedChunk(
            text=doc,
            source_file=meta["source_file"],
            location=meta["location"],
            similarity=similarity,
        ))
    return retrieved
