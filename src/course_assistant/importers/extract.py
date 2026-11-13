"""One extractor per format. Each returns (chunks, meta) where meta may carry title, page_count, duration_seconds."""
from __future__ import annotations

import posixpath
import re
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from .common import (ImportError_, chunk, chunk_cues, chunks_for, section_chunks, split_text, unique_slug, words)
from .htmltext import clean_html, decode

Result = tuple[list[dict[str, Any]], dict[str, Any]]


# ------------------------------------------------------------------ PDF
def extract_pdf(path: Path) -> Result:
    from pypdf import PdfReader
    from pypdf.errors import PyPdfError

    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            try:
                ok = reader.decrypt("")
            except Exception:  # noqa: BLE001
                ok = 0
            if not ok:
                raise ImportError_("This PDF is password protected. Save an unlocked copy and add that instead.")
        n = len(reader.pages)
        chunks: list[dict[str, Any]] = []
        total = 0
        for i, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:  # noqa: BLE001 - one bad page should not sink the file
                text = ""
            text = re.sub(r"[ \t]+\n", "\n", text).strip()
            total += len(text.strip())
            chunks.extend(chunks_for(text, "page", i))
    except ImportError_:
        raise
    except (PyPdfError, ValueError, OSError, KeyError, TypeError) as exc:
        raise ImportError_("I couldn't open this PDF. It may be damaged. Try opening it in another app and saving "
                           "a fresh copy.") from exc
    if total < 20:
        raise ImportError_("This PDF looks scanned: it has no text I can read. Take photos of the pages and add "
                           "them as photos so they can be read with OCR, or export a text version of the file.")
    return chunks, {"page_count": n}


# ------------------------------------------------------------------ PPTX
def _shape_texts(shapes) -> list[str]:
    out: list[str] = []
    for sh in shapes:
        try:
            if getattr(sh, "shapes", None) is not None and sh.shape_type == 6:  # group
                out.extend(_shape_texts(sh.shapes))
                continue
            if getattr(sh, "has_table", False) and sh.has_table:
                for row in sh.table.rows:
                    cells = [c.text.strip() for c in row.cells if c.text.strip()]
                    if cells:
                        out.append(" | ".join(cells))
                continue
            if getattr(sh, "has_text_frame", False) and sh.has_text_frame:
                t = "\n".join(p.text.strip() for p in sh.text_frame.paragraphs if p.text.strip())
                if t:
                    out.append(t)
        except Exception:  # noqa: BLE001
            continue
    return out


def extract_pptx(path: Path) -> Result:
    from pptx import Presentation

    try:
        prs = Presentation(str(path))
    except Exception as exc:  # noqa: BLE001
        raise ImportError_("I couldn't open this PowerPoint file. It may be damaged or an old .ppt. Save it as "
                           ".pptx and try again.") from exc
    chunks: list[dict[str, Any]] = []
    for i, slide in enumerate(prs.slides, start=1):
        title = ""
        try:
            if slide.shapes.title is not None:
                title = slide.shapes.title.text.strip()
        except Exception:  # noqa: BLE001
            pass
        parts = _shape_texts(slide.shapes)
        if title and parts and parts[0].strip() == title:
            parts = parts[1:]
        body = "\n\n".join(parts)
        notes = ""
        try:
            if slide.has_notes_slide:
                notes = slide.notes_slide.notes_text_frame.text.strip()
        except Exception:  # noqa: BLE001
            pass
        text = "\n\n".join(x for x in (title, body, f"Speaker notes: {notes}" if notes else "") if x)
        chunks.extend(chunks_for(text, "slide", i, heading=title))
    if not chunks:
        raise ImportError_("I found no text on these slides. If they are pictures, export the deck to PDF or add "
                           "photos of the slides so they can be read with OCR.")
    return chunks, {"page_count": len(prs.slides)}


# ------------------------------------------------------------------ DOCX
def extract_docx(path: Path) -> Result:
    import docx

    try:
        doc = docx.Document(str(path))
    except Exception as exc:  # noqa: BLE001
        raise ImportError_("I couldn't open this Word file. It may be damaged or an old .doc. Save it as .docx "
                           "and try again.") from exc
    seen: set[str] = set()
    sections: list[tuple[str, str, str]] = []
    anchor, heading, paras = "top", "", []
    title = ""
    try:
        title = (doc.core_properties.title or "").strip()
    except Exception:  # noqa: BLE001
        pass
    for item in doc.iter_inner_content():
        if hasattr(item, "style") and hasattr(item, "text"):  # paragraph
            style = (item.style.name if item.style is not None else "") or ""
            text = item.text.strip()
            if not text:
                continue
            if style.lower().startswith("heading") or style.lower() == "title":
                sections.append((anchor, heading, "\n\n".join(paras)))
                if style.lower() == "title" and not title:
                    title = text
                heading, paras = text, []
                anchor = unique_slug(text, seen)
            else:
                paras.append(text)
        else:  # table
            for row in item.rows:
                cells: list[str] = []
                for c in row.cells:
                    t = c.text.strip()
                    if t and t not in cells:
                        cells.append(t)
                if cells:
                    paras.append(" | ".join(cells))
    sections.append((anchor, heading, "\n\n".join(paras)))
    chunks = section_chunks(sections)
    if not chunks:
        raise ImportError_("This Word file has no text in it.")
    meta: dict[str, Any] = {}
    if title:
        meta["title"] = title
    return chunks, meta


# ------------------------------------------------------------------ Markdown / text
_FENCE = re.compile(r"^\s*(```|~~~)")
_HEAD = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")


def chunk_markdown(text: str) -> list[dict[str, Any]]:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    seen: set[str] = set()
    sections: list[tuple[str, str, list[str]]] = []
    anchor, heading, buf = "top", "", []
    in_fence = False
    has_heading = False
    for line in lines:
        if _FENCE.match(line):
            in_fence = not in_fence
        m = None if in_fence else _HEAD.match(line)
        if m and m.group(2):
            sections.append((anchor, heading, buf))
            heading = m.group(2).strip()
            anchor = unique_slug(heading, seen)
            buf = []
            has_heading = True
        else:
            buf.append(line)
    sections.append((anchor, heading, buf))
    if has_heading:
        return section_chunks((a, h, "\n".join(b)) for a, h, b in sections)
    return chunk_lines(lines)


def chunk_lines(lines: list[str], target: int = 200) -> list[dict[str, Any]]:
    """Headingless text: line ranges of roughly `target` words, cut at blank lines where possible."""
    out: list[dict[str, Any]] = []
    cur: list[str] = []
    start = 1
    n = 0

    def flush(end: int) -> None:
        body = "\n".join(cur).strip()
        if body:
            for piece in split_text(body):
                out.append(chunk(piece, "line", start, loc_end=end))

    for i, line in enumerate(lines, start=1):
        if not cur:
            if not line.strip():
                start = i + 1
                continue
            start = i
        cur.append(line)
        n += words(line)
        if (n >= target and not line.strip()) or n >= 300:
            flush(i)
            cur, n = [], 0
    if cur:
        flush(len(lines))
    return out


def extract_text(path: Path) -> Result:
    text = decode(path.read_bytes())
    chunks = chunk_markdown(text)
    if not chunks:
        raise ImportError_("This file is empty, so there is nothing to save.")
    meta: dict[str, Any] = {}
    m = re.search(r"^#\s+(.+?)\s*$", text, re.M)
    if m:
        meta["title"] = m.group(1).strip()
    return chunks, meta


# ------------------------------------------------------------------ HTML
def extract_html(path: Path) -> Result:
    cleaned, title, sections = clean_html(path.read_bytes())
    chunks = section_chunks(sections)
    if not chunks:
        raise ImportError_("I found no readable text on that page. It may need JavaScript or a sign-in, which this "
                           "app does not do. Save the page as PDF from your browser and add that instead.")
    return chunks, {"title": title}


# ------------------------------------------------------------------ EPUB
def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def extract_epub(path: Path) -> Result:
    try:
        zf = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError) as exc:
        raise ImportError_("I couldn't open this EPUB. It may be damaged or copy protected (DRM).") from exc
    with zf:
        names = set(zf.namelist())
        try:
            container = ET.fromstring(zf.read("META-INF/container.xml"))
            opf_path = next(e.get("full-path") for e in container.iter() if _local(e.tag) == "rootfile")
            opf = ET.fromstring(zf.read(opf_path))
        except (KeyError, StopIteration, ET.ParseError) as exc:
            raise ImportError_("This EPUB is missing its table of contents file. It may be damaged or "
                               "copy protected (DRM).") from exc
        base = posixpath.dirname(opf_path)
        manifest: dict[str, tuple[str, str]] = {}
        title = ""
        for el in opf.iter():
            t = _local(el.tag)
            if t == "item":
                manifest[el.get("id", "")] = (el.get("href", ""), el.get("media-type", ""))
            elif t == "title" and not title and el.text:
                title = el.text.strip()
        spine = [el.get("idref", "") for el in opf.iter() if _local(el.tag) == "itemref"]
        seen: set[str] = set()
        sections: list[tuple[str, str, str]] = []
        for n, idref in enumerate(spine, start=1):
            href, media = manifest.get(idref, ("", ""))
            if not href or "html" not in media and not href.lower().endswith((".xhtml", ".html", ".htm")):
                continue
            member = posixpath.normpath(posixpath.join(base, href.split("#")[0]))
            if member not in names:
                continue
            _, ch_title, secs = clean_html(zf.read(member))
            for j, (anchor, heading, body) in enumerate(secs):
                if j == 0 and not heading:
                    if not body.strip():
                        continue
                    heading = ch_title or f"Chapter {n}"
                    anchor = f"ch{n}"
                else:
                    anchor = f"ch{n}-{unique_slug(anchor if anchor != 'top' else heading, seen)}"
                sections.append((anchor, heading, body))
    chunks = section_chunks(sections)
    if not chunks:
        raise ImportError_("I found no text in this EPUB. It may be copy protected (DRM) or made of page images.")
    meta: dict[str, Any] = {"title": title} if title else {}
    return chunks, meta


# ------------------------------------------------------------------ captions
_TS = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})")
_TAG = re.compile(r"<[^>]+>")


def _secs(m: re.Match) -> float:
    h, mi, s, ms = m.groups()
    return int(h or 0) * 3600 + int(mi) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000


def parse_captions(text: str) -> list[tuple[float, float, str]]:
    cues: list[tuple[float, float, str]] = []
    blocks = re.split(r"\n\s*\n", text.replace("\r\n", "\n").replace("\r", "\n").lstrip("﻿"))
    for block in blocks:
        lines = [ln for ln in block.split("\n") if ln.strip()]
        for i, ln in enumerate(lines):
            if "-->" in ln:
                left, _, right = ln.partition("-->")
                a, b = _TS.search(left), _TS.search(right)
                if not a or not b:
                    break
                body = " ".join(_TAG.sub("", x).strip() for x in lines[i + 1:])
                body = re.sub(r"\{[^}]*\}", "", body).strip()
                cues.append((_secs(a), _secs(b), body))
                break
    return cues


def extract_captions(path: Path) -> Result:
    cues = parse_captions(decode(path.read_bytes()))
    chunks = chunk_cues(cues)
    if not chunks:
        raise ImportError_("I couldn't find any timed text in this captions file. Use a .vtt or .srt file.")
    return chunks, {"duration_seconds": max(c[1] for c in cues)}
