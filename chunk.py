"""
chunk.py — Fixed-size chunking with overlap.

Splits each extracted Segment (see ingest.py) into overlapping, fixed-size
chunks, and stamps every chunk with the source filename and a precise
location (e.g. "page 2, words 341-390"). That (source_file, location) pair is
carried all the way through embed.py -> store.py -> retrieve.py and is what
generate.py turns into a numbered citation in the final answer — this file
is where that traceability starts.

NOTE ON "TOKENS": for this demo, one "token" = one whitespace-separated
word, not a real model tokenizer unit (e.g. BPE). That's a deliberate
simplification — it avoids pulling in a tokenizer library just to size
chunks, and it keeps the chunking logic something you can point at and
explain in one sentence during a live walkthrough. It slightly undercounts
true LLM tokens (~0.75 words per token on average for English), which is
fine here since we're not trying to hit an exact context-window budget.
"""

from dataclasses import dataclass

DEFAULT_CHUNK_SIZE = 500   # words per chunk
DEFAULT_OVERLAP = 50       # words shared between consecutive chunks


@dataclass
class Chunk:
    text: str
    source_file: str
    location: str      # e.g. "page 2, words 341-390" or "file, words 1-500" (1-based, inclusive)
    chunk_index: int    # 0-based position of this chunk within the document, used as its storage id


def chunk_segments(
    segments,
    source_file: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Slide a fixed-size window over each segment's words, advancing by
    (chunk_size - overlap) words each step, so consecutive chunks share
    `overlap` words. This is the simplest chunking strategy that still
    protects against splitting an answer's supporting sentence across a
    chunk boundary — see README.md for why these particular defaults.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    chunk_index = 0
    step = chunk_size - overlap

    for segment in segments:
        words = segment.text.split()
        if not words:
            continue

        start = 0
        while start < len(words):
            window = words[start:start + chunk_size]
            chunk_text = " ".join(window)
            # 1-based, inclusive word range (not the 0-based start offset alone) so a
            # citation identifies the chunk's full span, e.g. "words 1-243" for a
            # short document that fits in a single chunk, rather than just "word 0"
            # (easily misread as "zero words" instead of "starts at word index 0").
            location = f"{segment.location}, words {start + 1}-{start + len(window)}"

            chunks.append(Chunk(
                text=chunk_text,
                source_file=source_file,
                location=location,
                chunk_index=chunk_index,
            ))
            chunk_index += 1

            if start + chunk_size >= len(words):
                break  # last window already reached the end of the segment
            start += step

    return chunks
