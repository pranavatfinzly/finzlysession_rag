"""
generate.py — Prompt construction, LLM call, and citation formatting.

Builds a context-grounded prompt from the chunks retrieve.py found, calls
Groq's OpenAI-compatible chat completions endpoint over raw HTTP (requests,
no SDK), and hands back the answer plus a citation list keyed by the same
[n] tags the model was told to cite.

*** THIS IS THE FILE TO TRACE FOR CITATIONS ***
The attachment happens in two places that must stay in sync:
  1. _format_context() numbers each retrieved chunk [1], [2], ... in the
     text the model reads, AND builds a `citations` list with that same
     numbering — so citation tag N always refers to the Nth chunk in
     both places.
  2. generate_answer() returns that same `citations` list untouched
     alongside the model's answer, so app.py can render "Sources" using
     the exact tags the model was instructed to cite in its prose.
The model never invents source_file/location strings itself — it only ever
emits the bracket number, and we already know what that number maps to.
"""

import os
import re

import requests

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "openai/gpt-oss-120b"

SYSTEM_PROMPT = (
    "You are a document Q&A assistant. Answer the user's question using ONLY "
    "the context excerpts provided below. Each excerpt is labeled with a "
    "citation tag like [1], [2], etc.\n\n"
    "Rules:\n"
    "- If the answer is present in the context, answer it and cite the "
    "excerpt number(s) you used inline, like this: \"The system supports X [1].\"\n"
    "- If the context does not contain the answer, respond exactly: "
    "\"I don't know based on the provided document.\" Do not fall back on "
    "your own general knowledge to fill gaps.\n"
    "- Only cite excerpt numbers that were actually given to you."
)


class GroqAPIError(Exception):
    """Raised for a missing API key or a non-200 response from Groq."""


def _format_context(retrieved_chunks) -> tuple[str, list[dict]]:
    """Number each retrieved chunk [1], [2], ... in document order (already
    ranked by similarity by retrieve.py) and build:

      - `context`: the text block injected into the prompt, each excerpt
        prefixed with its [n] tag and its source_file/location.
      - `citations`: a list of dicts with that same tag plus the metadata,
        for the UI to render — independent of anything the model outputs.

    The tag number is the ONLY link between the model's citation and a real
    chunk; everything else (source_file, location, similarity) comes from
    our own retrieval metadata, never from the LLM.
    """
    context_lines = []
    citations = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        context_lines.append(
            f"[{i}] (source: {chunk.source_file}, {chunk.location})\n{chunk.text}"
        )
        citations.append({
            "tag": i,
            "source_file": chunk.source_file,
            "location": chunk.location,
            "similarity": chunk.similarity,
        })
    context = "\n\n".join(context_lines)
    return context, citations


def build_prompt(question: str, retrieved_chunks) -> tuple[list[dict], list[dict]]:
    """Build the chat messages list and the citations list for one query."""
    context, citations = _format_context(retrieved_chunks)
    user_message = f"Context excerpts:\n\n{context}\n\nQuestion: {question}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    return messages, citations


def call_groq(messages: list[dict]) -> str:
    """Raw HTTP POST to Groq's OpenAI-compatible chat completions endpoint.

    Reads GROQ_API_KEY from the environment (same pattern as the earlier
    chatbot project) rather than accepting it as a parameter, so callers
    never have to thread a secret through the pipeline.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise GroqAPIError(
            "GROQ_API_KEY environment variable is not set. "
            "Set it before starting the app: export GROQ_API_KEY=your-key-here"
        )

    response = requests.post(
        GROQ_API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": GROQ_MODEL,
            "messages": messages,
            "temperature": 0.0,  # deterministic-ish answers are easier to reason about live
        },
        timeout=30,
    )
    if response.status_code != 200:
        raise GroqAPIError(f"Groq API error {response.status_code}: {response.text}")

    data = response.json()
    return data["choices"][0]["message"]["content"]


def _extract_cited_tags(answer: str) -> set[int]:
    """Parse which [n] tags the model actually used in its answer text.

    Retrieval always returns its top-k nearest chunks even when none of them
    are actually relevant (nearest-neighbor search has no "no match" case —
    see retrieve.py), so `citations` from build_prompt() includes every
    retrieved chunk regardless of relevance. Without this filter, a refusal
    like "I don't know based on the provided document." would still show
    all of those unrelated chunks as if they were its sources.
    """
    tags = set()
    for group in re.findall(r"\[([\d,\s]+)\]", answer):
        for part in group.split(","):
            part = part.strip()
            if part.isdigit():
                tags.add(int(part))
    return tags


def generate_answer(question: str, retrieved_chunks) -> dict:
    """Full generation step for one query: build the prompt, call the LLM,
    and return {"answer": str, "citations": list[dict]} for app.py to render.

    `citations` only includes chunks the model actually cited inline, not
    every chunk retrieve.py handed it — see _extract_cited_tags().
    """
    messages, citations = build_prompt(question, retrieved_chunks)
    answer = call_groq(messages)
    cited_tags = _extract_cited_tags(answer)
    cited_citations = [c for c in citations if c["tag"] in cited_tags]
    return {"answer": answer, "citations": cited_citations}
