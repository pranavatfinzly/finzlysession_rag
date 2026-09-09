"""
llm.py — Streamlit UI.

Wires the pipeline stages together for a live demo:
  upload -> ingest.extract_text -> chunk.chunk_segments -> embed.embed_texts
  -> store.add_chunks -> (on query) retrieve.retrieve -> generate.generate_answer

No pipeline logic lives here — every non-trivial step is a call into the
matching module, so each module can be read (and explained) on its own.
"""

import truststore

truststore.inject_into_ssl()  # use the OS trust store for all SSL, not just certifi's
# bundled CAs — fixes SSL: CERTIFICATE_VERIFY_FAILED / "couldn't connect to
# huggingface.co" on corporate networks that do TLS inspection with an
# internal root CA. Must run before any HTTP client (huggingface_hub, requests)
# builds its first SSL context, so this import stays first in the file.

import logging

from streamlit.watcher import local_sources_watcher  # noqa: F401

# transformers registers lazy-loaded modules for vision models (SAM, YOLOS,
# Qwen2-VL, etc.) this app never uses. Streamlit's dev-mode file watcher probes
# every loaded module's __path__ to decide what to watch for auto-reload, which
# triggers those modules' lazy imports and fails with "No module named
# 'torchvision'" (transformers/embed.py never installs torchvision — it's a
# text-only pipeline). Streamlit already catches that exception per-module
# (streamlit/watcher/local_sources_watcher.py, get_module_paths) so it can't
# crash a script run; this only silences the resulting WARNING-level log spam.
# It does NOT affect real app errors, which Streamlit renders in the browser
# via a different code path (the ScriptRunner), not this logger.
#
# The explicit import above (before setLevel) is required: Streamlit's
# get_logger() resets a logger's level to Streamlit's global default the
# first time that logger name is requested, which happens lazily inside
# Streamlit's own runtime — an override set before that first request would
# otherwise get silently clobbered.
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # picks up GROQ_API_KEY from a .env file, if present, before generate.py reads it

from ingest import (
    validate_file,
    extract_text,
    UnsupportedFileTypeError,
    FileTooLargeError,
)
from chunk import chunk_segments, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP
from embed import embed_texts
from store import get_client, create_collection_for_upload, add_chunks
from retrieve import retrieve, DEFAULT_TOP_K
from generate import generate_answer, GroqAPIError

st.set_page_config(page_title="Chat With Your Documents (RAG Demo)", layout="wide")

# --- session state -----------------------------------------------------
if "chroma_client" not in st.session_state:
    st.session_state.chroma_client = get_client()
if "collection" not in st.session_state:
    st.session_state.collection = None
if "collection_name" not in st.session_state:
    st.session_state.collection_name = None
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0
if "processed_filename" not in st.session_state:
    st.session_state.processed_filename = None

st.title("Chat With Your Documents")
st.caption(
    "A from-scratch Retrieval-Augmented Generation pipeline — upload a document, "
    "watch it get chunked and embedded, then ask questions and see exactly which "
    "chunks the answer came from."
)

with st.sidebar:
    st.header("Chunking settings")
    chunk_size = st.number_input(
        "Chunk size (words)", min_value=50, max_value=2000,
        value=DEFAULT_CHUNK_SIZE, step=50,
    )
    overlap = st.number_input(
        "Overlap (words)", min_value=0, max_value=int(chunk_size) - 1,
        value=min(DEFAULT_OVERLAP, int(chunk_size) - 1), step=10,
    )
    st.header("Retrieval settings")
    top_k = st.number_input(
        "Top-k chunks", min_value=1, max_value=20, value=DEFAULT_TOP_K, step=1,
    )

# --- step 1: upload ------------------------------------------------------
st.subheader("1. Upload a document")
st.info(
    "**Demo only** — please don't upload confidential documents. Uploaded "
    "text is sent to a third-party API (Groq) to generate answers.",
    icon="⚠️",
)

uploaded_file = st.file_uploader(
    "Choose a PDF, Markdown, or text file",
    type=["pdf", "md", "txt"],
    accept_multiple_files=False,
)

if uploaded_file is not None and uploaded_file.name != st.session_state.processed_filename:
    file_bytes = uploaded_file.getvalue()

    try:
        validate_file(uploaded_file.name, len(file_bytes))
    except (UnsupportedFileTypeError, FileTooLargeError) as e:
        st.error(str(e))
        st.stop()

    with st.spinner("Extracting text, chunking, embedding, and indexing..."):
        segments = extract_text(uploaded_file.name, file_bytes)
        if not segments:
            st.error(
                "No extractable text found in this file (e.g. a scanned PDF "
                "with no text layer isn't supported in this demo)."
            )
            st.stop()

        chunks = chunk_segments(
            segments,
            source_file=uploaded_file.name,
            chunk_size=int(chunk_size),
            overlap=int(overlap),
        )
        if not chunks:
            st.error("Document produced no chunks — the file may be empty.")
            st.stop()

        embeddings = embed_texts([c.text for c in chunks])

        collection = create_collection_for_upload(
            st.session_state.chroma_client,
            previous_collection_name=st.session_state.collection_name,
        )
        add_chunks(collection, chunks, embeddings)

        st.session_state.collection = collection
        st.session_state.collection_name = collection.name
        st.session_state.chunk_count = len(chunks)
        st.session_state.processed_filename = uploaded_file.name

if st.session_state.collection is not None:
    st.success(
        f"Indexed **{st.session_state.processed_filename}** into "
        f"**{st.session_state.chunk_count} chunks** "
        f"(collection `{st.session_state.collection_name}`)."
    )

# --- step 2: ask ----------------------------------------------------------
st.subheader("2. Ask a question")

question = st.text_input("Your question about the uploaded document")
ask_clicked = st.button("Ask", disabled=st.session_state.collection is None)

if ask_clicked and not question.strip():
    st.warning("Enter a question first.")

elif ask_clicked and question.strip():
    with st.spinner("Retrieving relevant chunks..."):
        retrieved_chunks = retrieve(st.session_state.collection, question, top_k=int(top_k))

    with st.expander(f"Retrieved {len(retrieved_chunks)} chunks (similarity scores)", expanded=True):
        if not retrieved_chunks:
            st.write("No chunks retrieved.")
        for i, chunk in enumerate(retrieved_chunks, start=1):
            st.markdown(
                f"**[{i}] {chunk.source_file} — {chunk.location}**  "
                f"·  cosine similarity: `{chunk.similarity:.4f}`"
            )
            preview = chunk.text[:500] + ("..." if len(chunk.text) > 500 else "")
            st.text(preview)
            st.divider()

    with st.spinner("Generating answer..."):
        try:
            result = generate_answer(question, retrieved_chunks)
        except GroqAPIError as e:
            st.error(f"LLM call failed: {e}")
            st.stop()

    st.subheader("Answer")
    st.markdown(result["answer"])

    if result["citations"]:
        st.markdown("**Sources**")
        for c in result["citations"]:
            st.markdown(
                f"[{c['tag']}] {c['source_file']} — {c['location']} "
                f"(cosine similarity: {c['similarity']:.4f})"
            )
    else:
        st.caption(
            "No sources shown — the model didn't cite any retrieved chunk, "
            "meaning nothing relevant enough was found in the document."
        )
