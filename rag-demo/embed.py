"""
embed.py — Embedding generation via a local sentence-transformers model.

Runs entirely on-device (CPU is fine for demo-sized documents): no
embedding API, no extra API key beyond the LLM call in generate.py. Trades
a one-time model download (~80MB, cached after first run) for zero
per-query cost and one less third-party dependency to explain live.

Both document chunks (embed.py called from app.py at ingest time) and user
queries (called from retrieve.py) go through embed_texts(), using the same
model — that's required for cosine similarity between them to mean anything.
"""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, good enough for demo-scale documents


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Load the model once per process and reuse it — loading it is the
    slow part (disk + memory), encoding individual batches is fast."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of strings, returning one vector per input string.

    Vectors are L2-normalized so that cosine similarity (used in store.py /
    retrieve.py) reduces to a plain dot product, which is what ChromaDB's
    "cosine" space computes internally.
    """
    if not texts:
        return []
    model = _get_model()
    vectors = model.encode(texts, normalize_embeddings=True)
    return vectors.tolist()


def embed_query(query: str) -> list[float]:
    """Convenience wrapper for embedding a single user query."""
    return embed_texts([query])[0]
