# Chat With Your Documents — RAG Demo

A minimal, dependency-light Retrieval-Augmented Generation (RAG) pipeline
built for a live internal training session. Every pipeline stage is a
separate, independently-readable file — no LangChain/LlamaIndex, so the
whole thing can be read and explained line by line.

## Architecture

```
Upload (app.py)
   |
   v
ingest.py    -> detect file type, extract raw text (per page for PDF)
   |
   v
chunk.py     -> split into overlapping fixed-size chunks, tag with source + location
   |
   v
embed.py     -> embed each chunk (local sentence-transformers model)
   |
   v
store.py     -> add chunks + embeddings to a fresh, uniquely-named ChromaDB collection
   |
   v
[ user asks a question ]
   |
   v
retrieve.py  -> embed the query, fetch top-k chunks by cosine similarity
   |
   v
generate.py  -> build a grounded prompt, call Groq's LLM, format citations
   |
   v
app.py       -> render retrieved chunks, answer, and citations
```

`app.py` contains **only** Streamlit UI code — it imports and calls the
other five modules but has no pipeline logic of its own.

## Setup

Use a project-local virtual environment — don't install into the global/system
Python. `requirements.txt` is fully pinned to versions verified to work
together (see "Environment notes" below), so a fresh `.venv` reproduces the
same working setup every time.

Windows (CMD):

```bat
cd finzlysession_rag
py -3.14 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

macOS/Linux:

```bash
cd finzlysession_rag
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

`requirements.txt` includes `pip-system-certs` on Windows only (via an
environment marker). It patches Python's SSL to trust the same certificate
store Windows/browsers already do — needed on corporate machines that do TLS
inspection with an internal root CA, which otherwise breaks
`sentence-transformers`'/`huggingface_hub`'s HTTPS calls with
`SSL: CERTIFICATE_VERIFY_FAILED` / "unable to get local issuer certificate".
No code changes needed; harmless no-op on macOS/Linux.

Set `GROQ_API_KEY` either as a real environment variable, or in a `.env`
file (loaded automatically by `app.py` via `python-dotenv` — never commit
this file, it's already in `.gitignore`):

```bash
# .env
GROQ_API_KEY=your-key-here
```

```bat
REM Windows CMD
set GROQ_API_KEY=your-key-here
.venv\Scripts\python.exe -m streamlit run app.py
```

```bash
# macOS/Linux
export GROQ_API_KEY=your-key-here
.venv/bin/python -m streamlit run app.py
```

The first run downloads the `all-MiniLM-L6-v2` embedding model
(~80 MB) — do this once *before* going live, not mid-demo:

```bat
.venv\Scripts\python.exe -c "from embed import embed_texts; embed_texts(['warm up'])"
```

## Environment notes

- **Python 3.14 is supported for this dependency stack** — verified directly:
  `torch==2.14.0` and `sentence-transformers==6.0.1` both ship official
  `cp314` wheels, and the full ingest→chunk→embed→store→retrieve pipeline
  (via `streamlit.testing.v1.AppTest` and a real `streamlit run`) was run
  end-to-end with zero fatal exceptions. There's no need to fall back to
  Python 3.12 for this project.
- **`torchvision` is not a dependency of this app and is not installed.**
  `embed.py` only loads a text embedding model (`all-MiniLM-L6-v2`) — no
  image processing anywhere in this codebase. Newer `transformers` releases
  lazily register image-model processors (nougat, sam, qwen2_vl, etc.) that
  import `torchvision` on first touch; since this app never touches those
  code paths, `torchvision` is never needed. If a future
  `transformers`/`sentence-transformers` upgrade causes Streamlit's
  local-file watcher to log `ModuleNotFoundError: No module named
  'torchvision'` while scanning loaded modules, that is a caught, non-fatal
  warning from an unrelated vision code path — not something this app's
  runtime depends on. Do not "fix" it by installing `torchvision`; if the
  noise is undesirable, it's cosmetic only.

**Which LLM model:** `generate.py`'s `GROQ_MODEL` is currently set to
`openai/gpt-oss-120b`. Groq periodically deprecates older models (this repo
originally pointed at `llama-3.3-70b-versatile`, which now 404s with
`model_not_found`) — if you hit that error, list what's currently live with:

```bash
curl -s https://api.groq.com/openai/v1/models -H "Authorization: Bearer $GROQ_API_KEY" | grep '"id"'
```

and update `GROQ_MODEL` in `generate.py` accordingly. Worth a quick check
before a live demo, since it's the one dependency outside this repo's control.

## Using it

1. Upload a `.pdf`, `.md`, or `.txt` file (5 MB max). Three synthetic
   documents are in `sample_docs/` as a backup in case live Wi-Fi or a
   participant's file misbehaves.
2. Wait for the "Indexed into N chunks" confirmation.
3. Ask a question. You'll see, in order: the retrieved chunks with
   similarity scores (expandable panel), then the generated answer with
   numbered citations pointing back to source file + location.

Chunk size, overlap, and top-k are adjustable in the sidebar if you want
to demonstrate how they change retrieval quality live.

## Chunking strategy — what to say out loud

**Defaults: 500 words per chunk, 50-word overlap (10%).**

- **Why fixed-size word chunking, not sentence/paragraph-aware chunking?**
  It's the simplest strategy to explain in one sentence and to point at in
  code (`chunk.py`'s sliding window), which matters more than optimality
  for a teaching demo. Production RAG systems often chunk on semantic
  boundaries (paragraphs, headings, sentence groups) instead — worth
  mentioning as a "next step" if someone asks.

- **Why 500 words?** Large enough that most chunks contain a complete
  thought or a self-contained Q&A pair (look at `sample_docs/product_faq.txt`
  — most Q&A pairs are well under 500 words), small enough that 4 retrieved
  chunks (the default top-k) comfortably fit in the LLM's context window
  alongside the system prompt and question, with room to spare. Chunks
  that are too large dilute the embedding (a chunk covering five unrelated
  topics gets a "smeared" embedding that matches everything a little and
  nothing well); chunks that are too small lose context (splitting an
  answer's justification from its claim).

- **Why 50-word overlap (10%)?** Overlap exists to stop a sentence that
  answers the question from being *split in half* right at a chunk
  boundary — without overlap, a fact camped right on the boundary loses
  half its context in each chunk's embedding and may not retrieve well in
  either. 10% is a common rule-of-thumb starting point: enough to catch
  boundary-spanning content, not so much that you're storing (and paying
  embedding compute for) mostly-duplicate text. There's no universally
  "correct" number — it's a knob to tune against your actual documents and
  is exposed in the sidebar for exactly that reason.

- **One honest caveat to mention live:** "token" in this code means
  whitespace-separated word, not a real LLM tokenizer unit. That's a
  deliberate simplification to avoid pulling in a tokenizer library just
  to size chunks — it under-counts true tokens slightly (~0.75 words per
  token for English) but doesn't change any of the reasoning above.

## Tracing a citation back to its source

This is the part worth stepping through live, in `generate.py`:

1. `retrieve.py` returns a list of `RetrievedChunk`, ranked by similarity,
   each already carrying `source_file` and `location` (set back in
   `chunk.py` at chunking time, stored as metadata in `store.py`).
2. `generate.py`'s `_format_context()` function numbers them `[1]`, `[2]`,
   ... in the exact order they'll appear in the prompt, and — in the same
   loop — builds a `citations` list using that identical numbering. This
   is the single place the citation tag gets attached to a chunk.
3. The system prompt instructs the model to cite using those `[n]` tags
   inline in its answer. The model **never invents** a source file or
   location string itself — it only ever echoes back a bracket number we
   already gave it.
4. `generate_answer()` returns `{"answer": ..., "citations": [...]}`
   untouched, and `app.py` renders the citations list under "Sources"
   using the same tags that appear in the answer text.

To trace any citation during the demo: find `[n]` in the model's answer,
then look up tag `n` in the `citations` list printed under "Sources" —
that dict's `source_file` and `location` came straight from `chunk.py`,
with zero opportunity for the LLM to alter it.

## Design choices / guardrails

- **No LangChain/LlamaIndex.** Every stage is plain Python so it's
  inspectable and explainable without a framework's abstractions in the way.
- **Local embeddings (sentence-transformers), remote generation (Groq).**
  Keeps the paid-API surface to just the LLM call, consistent with the
  earlier chatbot demo's pattern of reading `GROQ_API_KEY` from the
  environment.
- **Fresh collection per upload.** `store.create_collection_for_upload()`
  deletes the previous collection (if any) and creates a new
  uniquely-named one on every upload, so a new document's answers can
  never be contaminated by a previous upload's chunks in the same session.
- **In-memory ChromaDB (`EphemeralClient`).** No files written to disk, no
  cloud account, nothing to clean up after the demo — data disappears
  when the process exits.
- **Format allowlist, not a denylist.** `.pdf` / `.md` / `.txt` only; any
  other extension is rejected with an explicit error message rather than
  silently ignored or mis-parsed.
- **5 MB upload cap**, enforced in `ingest.validate_file()`, to keep
  ingestion fast enough for a live audience.
- **On-screen data-handling notice** next to the upload widget: this is a
  demo, don't upload confidential documents, uploaded text is sent to a
  third-party API.

## Known limitations (worth naming live if asked)

- PDF extraction is text-layer only — a scanned PDF with no text layer
  will show "no extractable text found."
- Single document per session — uploading a new file replaces the
  previous one; there's no multi-document corpus mode.
- Chunking is unaware of sentence/paragraph boundaries, so a chunk can
  start or end mid-sentence (the overlap mitigates but doesn't eliminate this).
