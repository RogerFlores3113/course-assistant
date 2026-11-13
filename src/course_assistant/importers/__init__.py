"""Turn files, pasted text and web pages into materials with searchable, citable chunks."""
from __future__ import annotations

import hashlib
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from .. import calendar, paths
from ..store import Store
from . import extract
from .common import (AUDIO_VIDEO, CAPTIONS, HTML_EXT, IMAGES, MAX_FILE_BYTES, MIME, SUPPORTED_EXTENSIONS,
                     ImportError_, slugify)
from .htmltext import clean_html

__all__ = ["import_file", "import_text", "import_url", "reimport", "detect_kind", "SUPPORTED_EXTENSIONS",
           "ImportError_"]

_SYLLABUS_WORDS = ("office hours", "grading", "midterm", "final exam", "prerequisite", "late policy",
                   "academic integrity", "course schedule", "learning outcomes", "course description",
                   "teaching assistant", "grade breakdown", "attendance policy", "textbook", "required reading")


# ------------------------------------------------------------------ kind detection
def detect_kind(filename: str, text_sample: str = "") -> str:
    """Guess a material kind from the file name, then from the first words of its text."""
    name = Path(filename or "").name.lower()
    stem, ext = Path(name).stem, Path(name).suffix
    tokens = [t for t in re.split(r"[^a-z0-9]+", stem) if t]
    joined = " ".join(tokens)

    def has(*words: str) -> bool:
        return any(w in tokens or w in joined for w in words)

    def hw_name() -> bool:
        return (bool(re.search(r"\b(hw|pset|ps|hmwk)\s*\d*\b", joined)) and not has("slides")) or has(
            "homework", "assignment", "problem set", "problemset", "worksheet", "lab")

    if has("syllabus", "course outline", "course info"):
        return "syllabus"
    if hw_name():
        return "assignment"
    if has("slides", "slide", "deck"):
        return "slides"
    if has("notes", "my notes", "cheatsheet", "cheat sheet"):
        return "notes"
    if has("lecture", "lec") or re.search(r"\blec\s*\d+", joined):
        return "recording" if ext in AUDIO_VIDEO | CAPTIONS else "slides"
    if has("textbook", "chapter", "book") or re.search(r"\bch\s*\d+", joined):
        return "textbook"
    if has("handout", "reading", "worksheet"):
        return "handout"
    sample = (text_sample or "")[:4000].lower()
    if sample:
        hits = sum(1 for w in _SYLLABUS_WORDS if w in sample)
        if hits >= 3 or ("syllabus" in sample[:300]) or (hits >= 2 and ("office hours" in sample or "grading" in sample)):
            return "syllabus"
        head = sample[:300]
        if re.search(r"\b(problem set|homework|assignment|pset)\s*#?\d*", head) and "due" in sample:
            return "assignment"
        if re.match(r"\s*(chapter|ch\.)\s*\d+", head):
            return "textbook"
        if re.match(r"\s*(lecture|lec\.)\s*\d+", head):
            return "slides"
    if ext == ".pptx":
        return "slides"
    if ext == ".epub":
        return "textbook"
    if ext in AUDIO_VIDEO or ext in CAPTIONS:
        return "recording"
    if ext in IMAGES:
        return "notes"
    return "other"


# ------------------------------------------------------------------ helpers
def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _pretty_title(name: str) -> str:
    stem = Path(name).stem
    stem = re.sub(r"[_]+", " ", stem)
    stem = re.sub(r"\s+", " ", stem).strip()
    return stem or "Untitled"


def _safe_name(name: str) -> str:
    base = Path(name).name
    stem, ext = Path(base).stem, Path(base).suffix.lower()
    return (slugify(stem, "file")[:60] or "file") + ext


def _store_path(material: dict[str, Any]) -> Path:
    return paths.files_dir() / material["filename"]


def _extract(path: Path, ext: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if ext == ".pdf":
        return extract.extract_pdf(path)
    if ext == ".pptx":
        return extract.extract_pptx(path)
    if ext == ".docx":
        return extract.extract_docx(path)
    if ext == ".epub":
        return extract.extract_epub(path)
    if ext in (".md", ".txt"):
        return extract.extract_text(path)
    if ext in HTML_EXT:
        return extract.extract_html(path)
    if ext in CAPTIONS:
        return extract.extract_captions(path)
    raise ImportError_(_unsupported(ext))


def _unsupported(ext: str) -> str:
    listing = ", ".join(sorted({e.lstrip('.') for e in SUPPORTED_EXTENSIONS}))
    what = f"{ext} files" if ext else "That kind of file"
    return f"I can't read {what} yet. I can read: {listing}. Save it as one of those (a PDF works for most things)."


def _sample(chunks: list[dict[str, Any]], n: int = 3000) -> str:
    out, size = [], 0
    for c in chunks:
        out.append(c["text"])
        size += len(c["text"])
        if size >= n:
            break
    return "\n".join(out)[:n]


def _year_hint(store: Store, material: dict[str, Any]) -> int | str | None:
    course = store.course(material["course_id"]) or {}
    for src in (course.get("term", ""), material.get("lecture_date", "")):
        if re.search(r"(?:19|20)\d{2}", src or ""):
            return src
    return None


def _process_syllabus(store: Store, material_id: int) -> int:
    m = store.material(material_id)
    if not m or m["kind"] != "syllabus":
        return 0
    text = "\n".join(c["text"] for c in store.chunks_of(material_id))
    store.clear_syllabus_events(material_id)
    n = 0
    for ev in calendar.extract_events(text, _year_hint(store, m)):
        store.add_event(m["course_id"], ev["kind"], ev["title"], ev["date"], time=ev.get("time", ""),
                        topics=ev.get("notes", ""), source_material_id=material_id, added_by="syllabus")
        n += 1
    return n



def _caption_sibling(media_path: Path) -> Path | None:
    for ext in (".vtt", ".srt"):
        p = media_path.with_suffix(ext)
        if p.exists():
            return p
    return None


# ------------------------------------------------------------------ public API
def import_file(store: Store, course_id: int, src: Path, *, kind: str | None = None, title: str | None = None,
                original_name: str | None = None, lecture_date: str = "", added_by: str = "student",
                captions: Path | None = None) -> int:
    """Save a file into the course and make it searchable. Returns the material id.
    Importing the same file again returns the existing material."""
    src = Path(src)
    if not store.course(course_id):
        raise ImportError_("That course doesn't exist any more.")
    if not src.is_file():
        raise ImportError_("I couldn't find that file. It may have been moved or deleted.")
    orig = original_name or src.name
    ext = Path(orig).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ImportError_(_unsupported(ext))
    if src.stat().st_size > MAX_FILE_BYTES:
        raise ImportError_("That file is bigger than 500 MB, which is too big to save. For a long recording, "
                           "export the audio only (an .mp3) or add a captions file (.vtt or .srt) instead.")
    if src.stat().st_size == 0:
        raise ImportError_("That file is empty, so there is nothing to save.")
    if captions is not None:
        captions = Path(captions)
        if captions.suffix.lower() not in CAPTIONS or not captions.is_file():
            raise ImportError_("The captions file must be a .vtt or .srt file.")

    work = src
    tmp_dir: Path | None = None
    try:
        if ext in HTML_EXT:  # the saved snapshot is the cleaned page
            tmp_dir = Path(tempfile.mkdtemp(prefix="ca-html-"))
            cleaned, page_title, _ = clean_html(src.read_bytes())
            work = tmp_dir / Path(orig).with_suffix(".html").name
            work.write_text(cleaned, encoding="utf-8")
            if title is None and page_title:
                title = page_title
        sha = _sha256_file(work)
        existing = store.material_by_sha(course_id, sha)
        if existing:
            return int(existing["id"])

        # extract before saving anything, so a bad file leaves nothing behind
        chunks: list[dict[str, Any]] = []
        meta: dict[str, Any] = {}
        if ext in AUDIO_VIDEO or ext in IMAGES:
            if captions is not None and ext in AUDIO_VIDEO:
                chunks, meta = extract.extract_captions(captions)
        else:
            chunks, meta = _extract(work, ext if ext not in HTML_EXT else ".html")

        final_ext = ".html" if ext in HTML_EXT else ext
        dest_dir = paths.course_files_dir(course_id)
        dest = dest_dir / f"{sha[:10]}-{_safe_name(orig)}"
        dest = dest.with_suffix(final_ext)
        shutil.copyfile(work, dest)
        if captions is not None and ext in AUDIO_VIDEO:
            shutil.copyfile(captions, dest.with_suffix(captions.suffix.lower()))
    finally:
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    if kind is None:
        kind = detect_kind(orig, _sample(chunks))
    final_title = (title or meta.get("title") or _pretty_title(orig)).strip()
    values: dict[str, Any] = {
        "filename": f"{course_id}/{dest.name}", "original_name": orig, "mime": MIME.get(ext, ""), "sha256": sha,
        "lecture_date": lecture_date, "added_by": added_by, "page_count": meta.get("page_count"),
        "duration_seconds": meta.get("duration_seconds"), "status": "ready"}
    if ext in AUDIO_VIDEO and not chunks:
        values["status"] = "needs_transcription"
        values["status_note"] = "Waiting to be transcribed on this computer."
    elif ext in IMAGES:
        values["status"] = "needs_transcription"
        values["status_note"] = "Photo of notes: waiting for its text to be read (OCR)."
    mid = store.add_material(course_id, kind, final_title, **values)
    if chunks:
        store.replace_chunks(mid, chunks)
    if values["status"] == "needs_transcription":
        store.add_job("ocr" if ext in IMAGES else "transcribe", mid, note=final_title)
    _process_syllabus(store, mid)
    return mid


def import_text(store: Store, course_id: int, title: str, text: str, *, kind: str = "notes",
                added_by: str = "student") -> int:
    """Save pasted text (notes, a syllabus pasted from a web page) as a material."""
    if not store.course(course_id):
        raise ImportError_("That course doesn't exist any more.")
    text = (text or "").strip()
    if not text:
        raise ImportError_("There is no text to save. Paste or type something first.")
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    existing = store.material_by_sha(course_id, sha)
    if existing:
        return int(existing["id"])
    chunks = extract.chunk_markdown(text)
    if not chunks:
        raise ImportError_("There is no text to save. Paste or type something first.")
    mid = store.add_material(course_id, kind, title or "Pasted text", mime="text/markdown", sha256=sha,
                             added_by=added_by, status="ready")
    store.replace_chunks(mid, chunks)
    _process_syllabus(store, mid)
    return mid


def _url_name(url: str, suffix: str) -> str:
    p = urlsplit(url)
    tail = unquote(Path(p.path).name) if p.path.strip("/") else ""
    stem = Path(tail).stem or ""
    base = f"{p.hostname or 'page'}-{stem}" if stem else (p.hostname or "page")
    return slugify(base, "page") + suffix


def import_url(store: Store, course_id: int, url: str, *, kind: str = "website", fetcher=None,
               added_by: str = "student") -> int:
    """Fetch a page politely and save an offline snapshot (cleaned HTML). A direct PDF link is imported as a PDF."""
    from ..fetcher import FetchError, get_default_fetcher

    fetcher = fetcher or get_default_fetcher()
    try:
        res = fetcher.get(url)
    except ImportError_:
        raise
    except FetchError as exc:
        raise ImportError_(str(exc)) from exc
    ctype = res.content_type
    is_pdf = "pdf" in ctype or res.content[:5] == b"%PDF-"
    is_html = "html" in ctype or (not ctype and res.text.lstrip()[:1] == "<") or ctype in ("", "text/plain") and \
        "<html" in res.text[:2000].lower()
    with tempfile.TemporaryDirectory(prefix="ca-url-") as d:
        if is_pdf:
            name = _url_name(res.final_url, ".pdf")
            f = Path(d) / name
            f.write_bytes(res.content)
            use_kind = kind
            if kind == "website":
                use_kind = detect_kind(name)
                if use_kind == "other":
                    use_kind = "handout"
            mid = import_file(store, course_id, f, kind=use_kind, original_name=name, added_by=added_by)
        elif is_html:
            name = _url_name(res.final_url, ".html")
            cleaned, page_title, _ = clean_html(res.content, base_url=res.final_url)
            f = Path(d) / name
            f.write_text(cleaned, encoding="utf-8")
            mid = import_file(store, course_id, f, kind=kind, title=page_title or None, original_name=name,
                              added_by=added_by)
        else:
            raise ImportError_("That link isn't a web page or a PDF, so I didn't save it. Download the file and add "
                               "it from your computer instead.")
    m = store.material(mid)
    if m and not m["source_url"]:
        store.update_material(mid, source_url=res.final_url)
    return mid


def reimport(store: Store, material_id: int) -> None:
    """Re-extract chunks (and syllabus dates) from the saved file, e.g. after an importer improves."""
    m = store.material(material_id)
    if not m:
        raise ImportError_("That material doesn't exist any more.")
    if not m["filename"]:
        return  # pasted text: its chunks are the only copy
    path = _store_path(m)
    if not path.is_file():
        raise ImportError_("The saved file is missing from this computer's data folder, so I can't read it again.")
    ext = path.suffix.lower()
    if ext in IMAGES:
        if not store.chunks_of(material_id) and not _has_open_job(store, material_id, "ocr"):
            store.add_job("ocr", material_id, note=m["title"])
        return
    if ext in AUDIO_VIDEO:
        sib = _caption_sibling(path)
        if sib is not None:
            chunks, meta = extract.extract_captions(sib)
            store.replace_chunks(material_id, chunks)
            store.update_material(material_id, status="ready", status_note="", duration_seconds=meta.get("duration_seconds"))
        elif not store.chunks_of(material_id) and not _has_open_job(store, material_id, "transcribe"):
            store.add_job("transcribe", material_id, note=m["title"])
        return
    chunks, meta = _extract(path, ext)
    store.replace_chunks(material_id, chunks)
    store.update_material(material_id, status="ready", status_note="", page_count=meta.get("page_count"),
                          **({"duration_seconds": meta["duration_seconds"]} if "duration_seconds" in meta else {}))
    _process_syllabus(store, material_id)


def _has_open_job(store: Store, material_id: int, kind: str) -> bool:
    return any(j["material_id"] == material_id and j["kind"] == kind and j["status"] in ("queued", "running")
               for j in store.jobs())
