"""
ingest.py — File type detection and text extraction.

Turns an uploaded file's raw bytes into a list of Segments: chunks of raw
text paired with a human-readable location label ("page 3", or "file" for
plain text). This is the first point where we start tracking *where* text
came from — chunk.py builds on these labels to produce citation locations,
and generate.py surfaces those citations in the final answer.

Supported types: .pdf (text layer only, no OCR), .md, .txt. Anything else
is rejected with a clear error rather than silently skipped or mis-parsed.
"""

from dataclasses import dataclass
from io import BytesIO

from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt"}
MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB — kept small so ingestion is instant during a live demo


class UnsupportedFileTypeError(Exception):
    """Raised when the uploaded file's extension isn't one we support."""


class FileTooLargeError(Exception):
    """Raised when the uploaded file exceeds MAX_FILE_SIZE_BYTES."""


@dataclass
class Segment:
    """A unit of extracted text plus where it came from in the source file."""
    text: str
    location: str  # e.g. "page 1" (PDF) or "file" (md/txt)


def _get_extension(filename: str) -> str:
    if "." not in filename:
        return ""
    return "." + filename.rsplit(".", 1)[-1].lower()


def validate_file(filename: str, size_bytes: int) -> str:
    """Check extension and size before doing any real work.

    Returns the lowercase extension on success; raises UnsupportedFileTypeError
    or FileTooLargeError (both plain Exceptions with a user-facing message)
    otherwise. Called by app.py immediately after upload, before extraction.
    """
    ext = _get_extension(filename)
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"'{filename}' is not a supported file type. "
            f"Please upload a .pdf, .md, or .txt file."
        )
    if size_bytes > MAX_FILE_SIZE_BYTES:
        raise FileTooLargeError(
            f"'{filename}' is {size_bytes / 1024 / 1024:.1f} MB, which exceeds "
            f"the {MAX_FILE_SIZE_BYTES / 1024 / 1024:.0f} MB limit for this demo."
        )
    return ext


def extract_text(filename: str, file_bytes: bytes) -> list[Segment]:
    """Dispatch to the right extractor based on file extension.

    Assumes validate_file() has already been called — this raises again
    for safety, but app.py should never reach here with a bad extension.
    """
    ext = _get_extension(filename)
    if ext == ".pdf":
        return _extract_pdf(file_bytes)
    elif ext in (".md", ".txt"):
        return _extract_plain_text(file_bytes)
    raise UnsupportedFileTypeError(f"No extractor registered for '{ext}'.")


def _extract_pdf(file_bytes: bytes) -> list[Segment]:
    """Extract text only — no OCR. One Segment per page, so a citation can
    point at "page 3" rather than just "somewhere in this PDF".

    A scanned page with no embedded text layer yields an empty string and is
    dropped; that's expected for this demo's scope (text-only extraction).
    """
    reader = PdfReader(BytesIO(file_bytes))
    segments = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            segments.append(Segment(text=text, location=f"page {page_number}"))
    return segments


def _extract_plain_text(file_bytes: bytes) -> list[Segment]:
    """.md and .txt are treated identically: decode as UTF-8 and return as a
    single Segment covering the whole file. Finer-grained location (word
    offset) is added later, in chunk.py, once the text is split up.
    """
    text = file_bytes.decode("utf-8", errors="replace")
    return [Segment(text=text, location="file")]
