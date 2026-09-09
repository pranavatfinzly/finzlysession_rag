# Chat With Your Documents — RAG Demo

A minimal Retrieval-Augmented Generation (RAG) pipeline: upload a document,
ask questions, and see exactly which chunks each answer came from.

No LangChain/LlamaIndex — every stage is its own small, readable Python file.

## Architecture

```
Upload (llm.py)
   |
   v
ingest.py    -> detect file type, extract raw text
   |
   v
chunk.py     -> split into overlapping fixed-size chunks
   |
   v
embed.py     -> embed each chunk (local sentence-transformers model)
   |
   v
store.py     -> store chunks + embeddings in ChromaDB
   |
   v
[ you ask a question ]
   |
   v
retrieve.py  -> embed the query, fetch top-k similar chunks
   |
   v
generate.py  -> build a grounded prompt, call Groq's LLM, format citations
   |
   v
llm.py       -> render retrieved chunks, answer, and citations
```

## Setup

```bash
cd finzlysession_rag
python -m venv .venv

# Windows
.venv\Scripts\python.exe -m pip install -r requirements.txt
# macOS/Linux
.venv/bin/python -m pip install -r requirements.txt
```

Get a free Groq API key at [console.groq.com](https://console.groq.com), then
create a `.env` file in the project root:

```
GROQ_API_KEY=your-key-here
```

Run the app:

```bash
# Windows
.venv\Scripts\python.exe -m streamlit run llm.py
# macOS/Linux
.venv/bin/python -m streamlit run llm.py
```

The first run downloads the embedding model (~80 MB), so the first upload
takes a little longer than later ones.

## Using it

1. Upload a `.pdf`, `.md`, or `.txt` file (5 MB max). Sample docs are in
   `sample_docs/` if you don't have your own file handy.
2. Wait for the "Indexed into N chunks" confirmation.
3. Ask a question — you'll see the retrieved chunks (with similarity
   scores), then the generated answer with numbered citations pointing
   back to source file + location.

Chunk size, overlap, and top-k are adjustable in the sidebar — try changing
them and re-asking the same question to see how retrieval quality shifts.

> **Note:** This is a demo. Don't upload confidential documents — extracted
> text is sent to Groq (a third-party API) to generate answers.

## Design choices

- **No framework** — every stage is plain Python, easy to step through.
- **Local embeddings, remote generation** — only the LLM call leaves your machine.
- **Fresh collection per upload** — a new document never mixes with a previous one.
- **In-memory ChromaDB** — nothing persists to disk.
- **Citations are traceable** — the model only ever echoes back a `[n]` tag
  it was given; `source_file`/`location` come straight from `chunk.py`, never
  from the LLM.

## Known limitations

- PDF extraction is text-layer only (no OCR for scanned PDFs).
- Single document per session — no multi-document corpus mode.
- Chunking doesn't respect sentence/paragraph boundaries.

## Ideas to try next

- Swap fixed-size chunking for a semantic/paragraph-aware strategy.
- Add a re-ranking step between `retrieve.py` and `generate.py`.
- Support more file types (`.docx`, `.html`).
- Persist ChromaDB to disk for a multi-document corpus across sessions.
