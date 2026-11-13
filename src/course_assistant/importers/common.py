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


# ------------------------------------------------------------------ code-safe block splitting
_FENCE_OPEN = re.compile(r"^(\s*)(`{3,}|~{3,})(.*)$")
HARD_CODE_WORDS = 1500       # a single fenced block longer than this may be split at a blank line (fence reopened)
INTRO_WORDS = 90             # a paragraph this short right before a code block stays with it

_JUNK_MARKERS = ("PLUTO_PROJECT_TOML_CONTENTS", "PLUTO_MANIFEST_TOML_CONTENTS", "This file is machine-generated",
                 "# This file is machine generated", "lockfileVersion", "\"lockfileVersion\"")
_LOCK_LINE = re.compile(r"^\s*(git-tree-sha1|uuid|version|resolved|integrity|checksum|sha\d+|hash)\s*[=:]", re.I)
_BASE64_LINE = re.compile(r"^[A-Za-z0-9+/=_-]{200,}$")


class Block:
    """A run of lines: either a fenced code block (kept verbatim) or a paragraph. Line numbers are 1-based."""

    __slots__ = ("kind", "text", "start", "end")

    def __init__(self, kind: str, text: str, start: int, end: int):
        self.kind, self.text, self.start, self.end = kind, text, start, end

    @property
    def words(self) -> int:
        return len(self.text.split())


def is_junk_block(text: str) -> bool:
    """Machine-generated blobs that only waste search and prompt space: Pluto manifests, lockfiles, base64."""
    if any(m in text for m in _JUNK_MARKERS):
        return True
    if "base64," in text and re.search(r"data:[\w/+.-]+;base64,[A-Za-z0-9+/=]{100,}", text):
        return True
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if not lines:
        return False
    if any(_BASE64_LINE.match(ln.strip()) for ln in lines):
        return True
    lock = sum(1 for ln in lines if _LOCK_LINE.match(ln))
    return len(lines) >= 12 and lock / len(lines) >= 0.4


def parse_blocks(text: str, first_line: int = 1) -> list[Block]:
    """Split text into fenced code blocks (verbatim, never split on blank lines) and paragraphs."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    blocks: list[Block] = []
    para: list[str] = []
    para_start = 0

    def flush_para(end: int) -> None:
        nonlocal para
        if para:
            blocks.append(Block("para", "\n".join(para).rstrip(), para_start, end))
            para = []

    i = 0
    while i < len(lines):
        line = lines[i]
        m = _FENCE_OPEN.match(line)
        if m:
            flush_para(first_line + i - 1)
            fence, indent = m.group(2), m.group(1)
            j = i + 1
            while j < len(lines):
                mc = re.match(r"^\s*(" + re.escape(fence[0]) + r"{%d,})\s*$" % len(fence), lines[j])
                if mc:
                    break
                j += 1
            closed = j < len(lines)
            body = lines[i:j + 1] if closed else lines[i:] + [indent + fence]
            blocks.append(Block("code", "\n".join(body), first_line + i, first_line + min(j, len(lines) - 1)))
            i = j + 1
            continue
        if line.strip():
            if not para:
                para_start = first_line + i
            para.append(line)
        else:
            flush_para(first_line + i - 1)
        i += 1
    flush_para(first_line + len(lines) - 1)
    return [b for b in blocks if b.text.strip()]


def _split_big_code(b: Block) -> list[Block]:
    """Only for a fenced block over HARD_CODE_WORDS: cut at a blank line and give each part its own fence."""
    lines = b.text.split("\n")
    open_line, close_line, body = lines[0], lines[-1], lines[1:-1]
    parts: list[list[str]] = [[]]
    n = 0
    for ln in body:
        parts[-1].append(ln)
        n += len(ln.split())
        if n >= HARD_CODE_WORDS // 2 and not ln.strip():
            parts.append([])
            n = 0
    return [Block("code", "\n".join([open_line, *p, close_line]), b.start, b.end) for p in parts if "".join(p).strip()]


def _sentence_groups(para: str, max_words: int) -> list[str]:
    """Pack the sentences (and lines) of one long paragraph into pieces of at most max_words. A piece never ends
    mid-sentence unless a single sentence is longer than max_words. Original line breaks are kept."""
    parts = re.split(r"((?<=[.!?])[ \t]*\n|(?<=[.!?])[ \t]+|\n)", para)
    units: list[tuple[str, str]] = []  # (sentence, separator that followed it)
    for k in range(0, len(parts), 2):
        sent = parts[k]
        sep = parts[k + 1] if k + 1 < len(parts) else ""
        if not sent.strip():
            continue
        if words(sent) > max_words:
            toks = sent.split()
            chunks_ = [" ".join(toks[i:i + max_words]) for i in range(0, len(toks), max_words)]
            units.extend((c, " ") for c in chunks_[:-1])
            units.append((chunks_[-1], sep))
        else:
            units.append((sent, sep))
    out: list[str] = []
    cur = ""
    n = 0
    for sent, sep in units:
        sw = words(sent)
        if cur and n + sw > max_words:
            out.append(cur.rstrip())
            cur, n = "", 0
        cur += sent + (sep if "\n" in sep else " ")
        n += sw
    if cur.strip():
        out.append(cur.rstrip())
    return out


def pack_blocks(blocks: list[Block], max_words: int = MAX_WORDS) -> list[tuple[str, int, int]]:
    """Group blocks into chunks of about max_words. Returns (text, first_line, last_line).
    Code blocks are never cut. A short paragraph that introduces a code block stays with it. Paragraphs are
    cut between sentences only when one paragraph alone is longer than max_words."""
    blocks = [b for b in blocks if not is_junk_block(b.text)]
    units: list[list[Block]] = []
    for b in blocks:
        if b.kind == "code" and b.words > HARD_CODE_WORDS:
            for part in _split_big_code(b):
                units.append([part])
            continue
        if b.kind == "para" and b.words > max_words:
            for sent_group in _sentence_groups(b.text, max_words):
                units.append([Block("para", sent_group, b.start, b.end)])
            continue
        if b.kind == "code" and units and len(units[-1]) == 1 and units[-1][0].kind == "para" \
                and units[-1][0].words <= INTRO_WORDS:
            units[-1].append(b)  # keep the explaining paragraph with its code
            continue
        units.append([b])
    out: list[tuple[str, int, int]] = []
    cur: list[Block] = []
    n = 0
    for unit in units:
        uw = sum(b.words for b in unit)
        if cur and n + uw > max_words:
            out.append(("\n\n".join(b.text for b in cur), cur[0].start, cur[-1].end))
            cur, n = [], 0
        cur.extend(unit)
        n += uw
    if cur:
        out.append(("\n\n".join(b.text for b in cur), cur[0].start, cur[-1].end))
    return out


def split_text(text: str, max_words: int = MAX_WORDS) -> list[str]:
    """Split text into pieces of about max_words, on block boundaries: fenced code is kept verbatim and whole,
    paragraphs are cut only between sentences, and machine-generated blobs (Pluto manifests, lockfiles) are dropped."""
    if not text or not text.strip():
        return []
    return [t for t, _, _ in pack_blocks(parse_blocks(text), max_words)]


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


CUE_MIN_SECONDS = 30      # do not cut before this much speech (unless the next cue starts a long pause)
CUE_TARGET_SECONDS = 45   # cut as soon as a sentence ends after this many seconds
CUE_MAX_SECONDS = 60      # cut by here, at the last sentence end if there was one
CUE_MAX_WORDS = 170
_SENT_END = re.compile(r"[.!?][\"')\]]*\s*$")


def _num(x: float) -> int | float:
    return int(x) if float(x).is_integer() else round(x, 1)


def chunk_cues(cues: list[tuple[float, float, str]], window: float = CUE_MAX_SECONDS,
               max_words: int = CUE_MAX_WORDS) -> list[dict[str, Any]]:
    """Group caption cues (start, end, text) into chunks of about 30 to 60 seconds, cut on cue boundaries at
    sentence ends where possible, so a time citation lands close to the claim. loc_start and loc_end are seconds;
    the label shows the range ("1:27 to 1:58")."""
    clean = []
    for s, e, text in cues:
        text = " ".join(text.split())
        if text:
            clean.append((float(s), max(float(e), float(s)), text))
    out: list[dict[str, Any]] = []

    def emit(group: list[tuple[float, float, str]]) -> None:
        if not group:
            return
        start, end = group[0][0], max(g[1] for g in group)
        label = locations.time_label(start)
        if end - start >= 5:
            label += f" to {locations.time_label(end)}"
        c = chunk(" ".join(g[2] for g in group), "time", _num(start), loc_end=_num(end))
        c["loc_label"] = label
        out.append(c)

    cur: list[tuple[float, float, str]] = []
    n = 0
    cut = -1  # index in cur after which a sentence ended and at least CUE_MIN_SECONDS had passed
    for cue in clean:
        s, e, text = cue
        if cur and (s - cur[0][0] >= window or n + words(text) > max_words):
            if cut >= 0:
                emit(cur[:cut + 1])
                cur = cur[cut + 1:]
            else:
                emit(cur)
                cur = []
            n = sum(words(c[2]) for c in cur)
            cut = -1
            if cur:  # the carried cues may already be long enough to cut again
                for k, c in enumerate(cur):
                    if c[1] - cur[0][0] >= CUE_MIN_SECONDS and _SENT_END.search(c[2]):
                        cut = k
        cur.append(cue)
        n += words(text)
        if e - cur[0][0] >= CUE_TARGET_SECONDS and _SENT_END.search(text):
            emit(cur)  # long enough and the sentence is finished: cut here
            cur, n, cut = [], 0, -1
        elif e - cur[0][0] >= CUE_MIN_SECONDS and _SENT_END.search(text):
            cut = len(cur) - 1
    emit(cur)
    return out
