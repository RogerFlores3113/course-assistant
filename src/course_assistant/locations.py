"""Where a citation points, and the in-app link that opens it.

Every chunk of every material has a location: a page (PDF, Word export, EPUB page), a slide (PowerPoint), a time
in seconds (recordings, from the transcript), a section anchor (web snapshots, Markdown, Word, EPUB), or a line.
A citation link is always /read/<material_id>?at=<chunk_id>#c<chunk_id>; the reader (web.py) opens the material at
that spot: the PDF at the page, the recording at the time, the snapshot at the section, and highlights the passage.
"""
from __future__ import annotations

from typing import Any


def href_for(chunk: dict[str, Any]) -> str:
    """The in-app link for a chunk row (needs id and material_id)."""
    return f"/read/{int(chunk['material_id'])}?at={int(chunk['id'])}#c{int(chunk['id'])}"


def time_label(seconds: float) -> str:
    """12:05 or 1:02:05."""
    s = int(round(max(0.0, float(seconds))))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def label(loc_kind: str, loc_start: Any, heading: str = "") -> str:
    """What a person reads for a location: "p. 12", "slide 4", "12:05", or the section heading."""
    if loc_kind == "page":
        return f"p. {loc_start}"
    if loc_kind == "slide":
        return f"slide {loc_start}"
    if loc_kind == "time":
        return time_label(float(loc_start))
    if loc_kind == "line":
        return f"line {loc_start}"
    return heading or str(loc_start)


def cite_text(chunk: dict[str, Any]) -> str:
    """'Lecture 3 slides, slide 4' for source lists and exports."""
    title = chunk.get("material_title") or chunk.get("title") or "Material"
    return f"{title}, {chunk.get('loc_label') or label(chunk.get('loc_kind', ''), chunk.get('loc_start', ''))}"
