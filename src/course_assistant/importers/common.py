"""Shared pieces for the importers: errors, supported formats, and the chunking helpers."""
from __future__ import annotations

import re
from typing import Any, Iterable

from .. import locations

MAX_FILE_BYTES = 500 * 1024 * 1024
MIN_WORDS = 80
MAX_WORDS = 350
TIME_WINDOW_SECONDS = 90

SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".pdf": "PDF",
    ".pptx": "PowerPoint",
    ".docx": "Word document",
    ".epub": "EPUB book",
    ".md": "Markdown",
    ".txt": "Plain text",
    ".html": "Web page",
    ".htm": "Web page",
    ".vtt": "Captions (WebVTT)",
    ".srt": "Captions (SRT)",
    ".mp3": "Audio",
    ".m4a": "Audio",
    ".wav": "Audio",
    ".mp4": "Video",
    ".mov": "Video",
    ".webm": "Video",
    ".png": "Photo",
    ".jpg": "Photo",
    ".jpeg": "Photo",
    ".heic": "Photo",
}
AUDIO_VIDEO = {".mp3", ".m4a", ".wav", ".mp4", ".mov", ".webm"}
IMAGES = {".png", ".jpg", ".jpeg", ".heic"}
CAPTIONS = {".vtt", ".srt"}
HTML_EXT = {".html", ".htm"}

MIME = {
    ".pdf": "application/pdf",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".epub": "application/epub+zip",
    ".md": "text/markdown", ".txt": "text/plain", ".html": "text/html", ".htm": "text/html",
    ".vtt": "text/vtt", ".srt": "application/x-subrip",
    ".mp3": "audio/mpeg", ".m4a": "audio/mp4", ".wav": "audio/wav",
    ".mp4": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm",
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".heic": "image/heic",
}


class ImportError_(Exception):  # noqa: N801 (name fixed by the design doc)
    """The message is shown to the student as is."""


def words(text: str) -> int:
    return len(text.split())


def slugify(text: str, fallback: str = "section") -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:60] or fallback


def unique_slug(text: str, seen: set[str], fallback: str = "section") -> str:
    base = slugify(text, fallback)
    slug, n = base, 2
    while slug in seen:
        slug = f"{base}-{n}"
        n += 1
    seen.add(slug)
    return slug


def split_text(text: str, max_words: int = MAX_WORDS) -> list[str]:
    """Split text into pieces of at most max_words, on paragraph then sentence then word boundaries."""
    text = text.strip()
    if not text:
        return []
    if words(text) <= max_words:
        return [text]
    units: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if words(para) <= max_words:
            units.append(para)
            continue
        for sent in re.split(r"(?<=[.!?])\s+|\n", para):
            if words(sent) <= max_words:
                units.append(sent)
            else:
                toks = sent.split()
                for i in range(0, len(toks), max_words):
                    units.append(" ".join(toks[i:i + max_words]))
    pieces: list[str] = []
    cur: list[str] = []
    n = 0
    for u in units:
        uw = words(u)
        if cur and n + uw > max_words:
            pieces.append("\n\n".join(cur))
            cur, n = [], 0
        cur.append(u)
        n += uw
    if cur:
        pieces.append("\n\n".join(cur))
    return pieces


def chunk(text: str, loc_kind: str, loc_start: Any, *, heading: str = "", loc_end: Any = "") -> dict[str, Any]:
    return {"text": text, "loc_kind": loc_kind, "loc_start": loc_start, "loc_end": loc_end, "heading": heading,
            "loc_label": locations.label(loc_kind, loc_start, heading)}


def chunks_for(text: str, loc_kind: str, loc_start: Any, *, heading: str = "") -> list[dict[str, Any]]:
    """One location, possibly several chunks (same loc_start) when the text is long."""
    return [chunk(p, loc_kind, loc_start, heading=heading) for p in split_text(text)]


def section_chunks(sections: Iterable[tuple[str, str, str]]) -> list[dict[str, Any]]:
    """sections: (anchor, heading, body). Tiny sections are folded into the next one so chunks stay useful."""
    out: list[dict[str, Any]] = []
    pending: tuple[str, str, str] | None = None  # anchor, heading, body of a section that is too small alone
    for anchor, heading, body in sections:
        body = body.strip()
        if not body and not heading:
            continue
        if pending:
            p_anchor, p_heading, p_body = pending
            if p_heading:
                joined = f"{heading}\n{body}".strip() if heading else body
                body = f"{p_body}\n\n{joined}".strip()
                anchor, heading = p_anchor, p_heading
            else:  # a headingless preamble joins the next section and takes its name
                body = f"{p_body}\n\n{body}".strip()
            pending = None
        if words(body) < 25:
            pending = (anchor, heading, body)
            continue
        out.extend(chunks_for(body, "section", anchor, heading=heading))
    if pending and pending[2]:
        out.extend(chunks_for(pending[2], "section", pending[0], heading=pending[1]))
    return out


def chunk_cues(cues: list[tuple[float, float, str]], window: float = TIME_WINDOW_SECONDS,
               max_words: int = MAX_WORDS) -> list[dict[str, Any]]:
    """Group caption cues (start, end, text) into chunks of about a minute and a half. loc_start is in seconds."""
    out: list[dict[str, Any]] = []
    cur: list[str] = []
    start = end = 0.0
    n = 0

    def flush() -> None:
        if cur:
            s = int(start) if float(start).is_integer() else round(start, 1)
            e = int(end) if float(end).is_integer() else round(end, 1)
            out.append(chunk(" ".join(cur), "time", s, loc_end=e))

    for s, e, text in cues:
        text = " ".join(text.split())
        if not text:
            continue
        if cur and (s - start >= window or n + words(text) > max_words):
            flush()
            cur, n = [], 0
        if not cur:
            start = s
        cur.append(text)
        n += words(text)
        end = max(e, s)
    flush()
    return out
