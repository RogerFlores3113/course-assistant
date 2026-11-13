"""One extractor per format. Each returns (chunks, meta) where meta may carry title, page_count, duration_seconds."""
from __future__ import annotations

import posixpath
import re
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from .common import (_FENCE_OPEN, ImportError_, _sentence_groups, chunk, chunk_cues, chunks_for, pack_blocks,
                     parse_blocks, section_chunks, split_text, unique_slug, words)
from .htmltext import clean_html, decode

Result = tuple[list[dict[str, Any]], dict[str, Any]]


# ------------------------------------------------------------------ PDF
_GREEK_MATH = "\u0370-\u03ff\u1f00-\u1fff\u2190-\u22ff\u27c0-\u27ef\u2980-\u2aff"
_EMPTY_REF = re.compile(r"\s*\(\s*(?:(?:Figure|Fig\.?|Table|Map|Chart|Image|Photo|Source)\s*\d*[a-z]?\s*)?[,;:]?\s*\)", re.I)
_EMPTY_BRACKET = re.compile(r"\s*\[\s*\]")
_BULLET = re.compile(r"^\s*(?:[\u2022\u25cf\u25e6\u2023\u2043\u25aa*-]\s+|\(?\d{1,2}[.)]\s+|\(?[a-z][.)]\s+)")
_CHAR_W = {" ": 0.25, "i": 0.25, "l": 0.25, "j": 0.25, "t": 0.33, "f": 0.33, "r": 0.36, ".": 0.25, ",": 0.25,
           ":": 0.25, ";": 0.25, "'": 0.2, "m": 0.8, "w": 0.72, "$": 0.5}


def _est_width(text: str, size: float) -> float:
    return sum(_CHAR_W.get(c, 0.65 if c.isupper() else 0.5) for c in text) * size


def _is_symbol(text: str) -> bool:
    t = text.strip()
    return 0 < len(t) <= 2 and all(ord(c) > 0x2ff for c in t)


def _page_lines(page) -> list[dict[str, Any]]:
    """Lines of a page with their baseline, font size and text, in reading order. Symbols that the PDF draws in
    a separate font (Greek letters, math signs) come out of the text stream at the wrong place; they are put back
    where their x position says they belong."""
    frags: list[tuple[str, float, float, float]] = []

    def visit(text, cm, tm, font_dict, font_size):  # noqa: ANN001
        if not text or text == "\n" or not text.strip("\n"):
            return
        try:
            x = tm[4] * cm[0] + tm[5] * cm[2] + cm[4]
            y = tm[4] * cm[1] + tm[5] * cm[3] + cm[5]
            fs = abs(float(font_size or 0) * (cm[3] if cm[3] else 1)) or 10.0
        except Exception:  # noqa: BLE001
            return
        frags.append((text.replace("\n", " "), float(x), float(y), fs))

    page.extract_text(visitor_text=visit)
    lines: list[dict[str, Any]] = []
    for text, x, y, fs in frags:
        cur = lines[-1] if lines else None
        if cur is not None and abs(y - cur["y"]) <= 0.55 * max(fs, cur["fs"]):
            cur["frags"].append([text, x, fs])
            cur["fs"] = max(cur["fs"], fs)
        else:
            lines.append({"y": y, "fs": fs, "frags": [[text, x, fs]]})
    out: list[dict[str, Any]] = []
    for ln in lines:
        parts: list[list[Any]] = []  # [text, x, fs]
        for text, x, fs in ln["frags"]:
            if _is_symbol(text) and parts:
                # find the earlier fragment this glyph belongs inside: the last one that starts left of it
                for k in range(len(parts) - 1, -1, -1):
                    ptext, px, pfs = parts[k]
                    if px <= x and not _is_symbol(ptext):
                        off = x - px
                        total = _est_width(ptext, pfs)
                        if 0 < off < total * 0.97:
                            acc, pos = 0.0, len(ptext)
                            for idx, ch in enumerate(ptext):
                                w = _est_width(ch, pfs)
                                if acc + w / 2 >= off:
                                    pos = idx
                                    break
                                acc += w
                            bounds = [i for i in range(len(ptext) + 1)
                                      if i in (0, len(ptext)) or ptext[i - 1] == " " or ptext[i] == " "]
                            pos = min(bounds, key=lambda i: abs(i - pos))  # a symbol sits between words
                            sym = text.strip()
                            left, right = ptext[:pos], ptext[pos:]
                            if left[-1:].isalnum():
                                sym = " " + sym
                            if right[:1].isalnum():
                                sym += " "
                            parts[k] = [left + sym + right, px, pfs]
                            break
                        parts.append([text, x, fs])
                        break
                else:
                    parts.append([text, x, fs])
                continue
            parts.append([text, x, fs])
        raw = "".join(p[0] for p in parts)
        raw = re.sub(r"\s+([.,;:!?])", r"\1", raw)
        raw = re.sub(r"[ \t]+", " ", raw).strip()
        if raw:
            out.append({"y": ln["y"], "fs": ln["fs"], "text": raw})
    return out


_LIGATURES = {"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl", "\ufb05": "st",
              "\ufb06": "st", "\u0132": "IJ", "\u0133": "ij"}
_SUPER = str.maketrans({"\u2070": "0", "\u00b9": "1", "\u00b2": "2", "\u00b3": "3", "\u2074": "4", "\u2075": "5",
                        "\u2076": "6", "\u2077": "7", "\u2078": "8", "\u2079": "9", "\u207a": "+", "\u207b": "-",
                        "\u207f": "n", "\u2071": "i"})
_SUB = str.maketrans({"\u2080": "0", "\u2081": "1", "\u2082": "2", "\u2083": "3", "\u2084": "4", "\u2085": "5",
                      "\u2086": "6", "\u2087": "7", "\u2088": "8", "\u2089": "9", "\u208a": "+", "\u208b": "-",
                      "\u2099": "n", "\u1d62": "i", "\u2c7c": "j"})
_SUPER_RUN = re.compile("[\u2070\u00b9\u00b2\u00b3\u2074-\u207b\u207f\u2071]+")
_SUB_RUN = re.compile("[\u2080-\u208b\u2099\u1d62\u2c7c]+")


def normalize_math_chars(text: str) -> str:
    """Characters a PDF uses for typesetting, as plain ones the search and the reader understand: ligatures (fi, fl),
    soft hyphens and zero-width marks, the minus sign (U+2212) between numbers or letters, and super and subscript
    digits (x\u00b2 becomes x^2, A\u207b\u00b9 becomes A^-1, C\u2086 becomes C_6)."""
    for k, v in _LIGATURES.items():
        text = text.replace(k, v)
    text = re.sub("[\u00ad\u200b\u200c\u200d\u2060\ufeff]", "", text)
    text = text.replace("\u2009", " ").replace("\u202f", " ").replace("\u00a0", " ")
    text = re.sub(r"(?<=[\w)\]}]) \u2212 (?=[\w(\[])", " - ", text)  # a - b
    text = re.sub(r"(?<=[\w)\]}])\u2212(?=[\w(\[])", "-", text)  # x-1
    text = re.sub(r"(?<![\w)])\u2212(?=[\w(])", "-", text)  # a leading minus: -3
    text = text.replace("\u2212", "-")
    text = _SUPER_RUN.sub(lambda m: "^" + m.group(0).translate(_SUPER), text)
    text = _SUB_RUN.sub(lambda m: "_" + m.group(0).translate(_SUB), text)
    return text


def _clean_page_text(text: str) -> str:
    text = normalize_math_chars(text)
    # a symbol that landed after the punctuation that should follow it: "scalar.λ Next" -> "scalar λ. Next"
    text = re.sub(rf"(?<=[A-Za-z0-9])([.,;:])[ \t]?([{_GREEK_MATH}])(?=\s|$)", r" \2\1", text)
    text = re.sub(rf"\$([{_GREEK_MATH}]) \$", r"$\1$", text)
    text = _EMPTY_REF.sub("", text)
    text = _EMPTY_BRACKET.sub("", text)
    text = re.sub(r"[ \t]+([.,;:!?])", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text


def _reflow(lines: list[dict[str, Any]]) -> str:
    """Join hard-wrapped lines into paragraphs and undo end-of-line hyphenation. Paragraph breaks come from
    vertical gaps, short lines that end a sentence, headings and bullets."""
    if not lines:
        return ""
    lens = sorted(len(ln["text"]) for ln in lines)
    full = lens[int(0.85 * (len(lens) - 1))] or 1
    paras: list[str] = []
    cur = ""
    prev: dict[str, Any] | None = None
    for ln in lines:
        t = ln["text"]
        if prev is None:
            cur = t
            prev = ln
            continue
        gap = prev["y"] - ln["y"]
        big_gap = gap > 1.55 * max(prev["fs"], ln["fs"])
        size_jump = abs(prev["fs"] - ln["fs"]) >= 1.5
        prev_t = prev["text"]
        short_prev = len(prev_t) < 0.55 * full and not prev_t.endswith((",", ";", "-"))
        starts_item = bool(_BULLET.match(t)) and not _BULLET.match(prev_t[-1:] or "")
        code_like = ("$$" in t and "$$" not in prev_t) or t.startswith("    ")
        if not (big_gap or size_jump) and short_prev and len(t) < 0.55 * full and not starts_item and not code_like:
            cur = f"{cur}\n{t}"  # a run of short lines (equations, list rows, verse): keep the line breaks
        elif big_gap or size_jump or short_prev or starts_item or code_like:
            paras.append(cur)
            cur = t
        elif re.search(r"[^\W\d_]{2,}-$", cur) and t[:1].islower():
            cur = cur[:-1] + t  # de-hyphenate: "govern-" + "ment" -> "government"
        else:
            cur = f"{cur} {t}"
        prev = ln
    paras.append(cur)
    return "\n\n".join(p.strip() for p in paras if p.strip())


def _letters(text: str) -> int:
    return sum(ch.isalnum() for ch in text)


def _page_text(page) -> str:
    """Best text for one page. Candidate A re-places out-of-order symbols and reflows by position; candidate B
    is pypdf's plain text reflowed. A is used unless it lost content compared with B."""
    plain = ""
    try:
        plain = page.extract_text() or ""
    except Exception:  # noqa: BLE001
        plain = ""
    b_lines = [{"y": -i * 14.0, "fs": 12.0, "text": " ".join(ln.split())} for i, ln in enumerate(plain.split("\n"))
               if ln.strip()]
    b = _reflow(b_lines)
    try:
        a = _reflow(_page_lines(page))
    except Exception:  # noqa: BLE001
        a = ""
    best = a if a and _letters(a) >= 0.85 * _letters(b) else b
    return _clean_page_text(re.sub(r"[ \t]+\n", "\n", best)).strip()


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
                text = _page_text(page)
            except Exception:  # noqa: BLE001 - one bad page should not sink the file
                text = ""
            total += len(text.strip())
            chunks.extend(chunks_for(text, "page", i))
        meta: dict[str, Any] = {"page_count": n}
        try:
            t = (reader.metadata.title or "").strip() if reader.metadata else ""
        except Exception:  # noqa: BLE001
            t = ""
        if t and len(t) >= 4 and not re.search(r"\.(tex|docx?|pdf|indd)$|^untitled|^microsoft word", t, re.I):
            meta["title"] = t
    except ImportError_:
        raise
    except (PyPdfError, ValueError, OSError, KeyError, TypeError) as exc:
        raise ImportError_("I couldn't open this PDF. It may be damaged. Try opening it in another app and saving "
                           "a fresh copy.") from exc
    if total < 20:
        raise ImportError_("This PDF looks scanned: it has no text I can read. Take photos of the pages and add "
                           "them as photos so they can be read with OCR, or export a text version of the file.")
    return chunks, meta


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


def _norm_words(t: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", t.lower()))


def _new_notes(notes: str, slide_text: str) -> str:
    """The speaker notes without the sentences that are already on the slide. Notes that only repeat the slide
    come back empty, so searching a deck does not list the same passage twice."""
    slide = _norm_words(slide_text)
    keep = []
    for sent in re.split(r"(?<=[.!?:])\s+|\n+", notes):
        n = _norm_words(sent)
        if n and n not in slide:
            keep.append(sent.strip())
    return " ".join(keep).strip()


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
        notes = re.sub(r"^\s*speaker notes:\s*", "", notes, flags=re.I)  # some decks type the label into the notes
        if notes:
            notes = _new_notes(notes, f"{title} {body}")
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
_HEAD = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)


def strip_markdown(text: str) -> str:
    """A heading or title as plain text: no links, emphasis, code ticks, tags, entities or stray spaces."""
    t = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"&nbsp;", " ", t)
    t = re.sub(r"(\*\*|__|\*|`|~~)", "", t)
    t = re.sub(r"(?<!\w)_|_(?!\w)", "", t)
    return re.sub(r"\s+", " ", t).strip(" \t#:-")


def _blank_comments(text: str) -> str:
    """Drop HTML comments (provenance notes) but keep every line number where it was."""
    return _HTML_COMMENT.sub(lambda m: "\n" * m.group().count("\n"), text)


def _heading_lines(lines: list[str]) -> list[tuple[int, int, str]]:
    """(line index, level, text) of every heading that is not inside a code fence."""
    out: list[tuple[int, int, str]] = []
    fence = ""
    for i, line in enumerate(lines):
        m = _FENCE_OPEN.match(line)
        if fence:
            if re.match(r"^\s*" + re.escape(fence[0]) + r"{%d,}\s*$" % len(fence), line):
                fence = ""
            continue
        if m:
            fence = m.group(2)
            continue
        h = _HEAD.match(line)
        if h and h.group(2):
            out.append((i, len(h.group(1)), h.group(2).strip()))
    return out


def top_title(text: str) -> str:
    """The document's own title: a heading that is the first thing in it (after comments and front matter)."""
    t = _blank_comments(text.replace("\r\n", "\n"))
    t = re.sub(r"\A\s*---\n.*?\n---\n", "", t, flags=re.S)
    for line in t.split("\n"):
        if not line.strip():
            continue
        h = _HEAD.match(line)
        return strip_markdown(h.group(2)) if h and h.group(1) in ("#", "##") else ""
    return ""


def chunk_markdown(text: str) -> list[dict[str, Any]]:
    text = _blank_comments(text.replace("\r\n", "\n").replace("\r", "\n"))
    lines = text.split("\n")
    heads = _heading_lines(lines)
    if not heads:
        return chunk_lines(lines)
    seen: set[str] = set()
    sections: list[tuple[str, str, str]] = []
    prev_end, anchor, heading = 0, "top", ""
    for idx, level, htext in heads:
        sections.append((anchor, heading, "\n".join(lines[prev_end:idx])))
        heading = strip_markdown(htext) or htext
        anchor = unique_slug(heading, seen)
        prev_end = idx + 1
    sections.append((anchor, heading, "\n".join(lines[prev_end:])))
    return section_chunks(sections)


def chunk_lines(lines: list[str] | str, target: int = 250) -> list[dict[str, Any]]:
    """Headingless text: chunks of about `target` words made of whole paragraphs and whole code blocks, located by
    their line range ("lines 40 to 77")."""
    if isinstance(lines, str):
        lines = lines.split("\n")
    out: list[dict[str, Any]] = []
    for piece, start, end in pack_blocks(parse_blocks("\n".join(lines)), target):
        c = chunk(piece, "line", start, loc_end=end)
        if end > start:
            c["loc_label"] = f"lines {start} to {end}"
        out.append(c)
    return out


# ---- plain text (primary sources: speeches, letters, the Federalist papers)
_STRUCTURED_HEAD = re.compile(
    r"^(?:the\s+)?(federalist|chapter|article|letter|lecture|week|part|book|section|speech|address|act|psalm|"
    r"canto|essay|sermon|declaration|amendment|case)\b[\s.:,#-]*(?:no\.?|number|nr\.?)?[\s.:#-]*([0-9]+|[IVXLC]+\b)",
    re.I)
_BOILERPLATE = re.compile(r"project gutenberg|public domain|\bebook\b|copyright|all rights reserved|"
                          r"transcribed|^\s*\*{3}|^\s*\[|^\s*source:", re.I)
_BYLINE = re.compile(r"^(from\b|for the\b|to the\b|by\b|published\b|written\b|\(?\d{1,2}\s+\w+\s+\d{4}|"
                     r"(mon|tues|wednes|thurs|fri|satur|sun)day\b|[A-Z][A-Za-z.' -]*\bEdition\b)", re.I)
_SALUTATION = re.compile(r"^(to the|to my|dear|gentlemen|fellow)\b.*[:,]$", re.I)


def _titlecase_head(h: str) -> str:
    """"FEDERALIST. No. 1" -> "Federalist No. 1"."""
    h = re.sub(r"\s+", " ", h.replace(". No", " No")).strip(" .:")
    words_ = h.split(" ")
    out = []
    for w in words_:
        if w.isupper() and len(w) > 1 and not re.fullmatch(r"[IVXLC]+", w):
            w = w.capitalize()
        out.append(w)
    return " ".join(out)


def _plain_paragraphs(lines: list[str]) -> list[list[tuple[int, str]]]:
    """Paragraphs as lists of (line number, text). A text with a blank line after every line (double spaced)
    has its paragraph breaks at two blank lines in a row."""
    nonblank = [i for i, ln in enumerate(lines) if ln.strip()]
    alternating = sum(1 for i in nonblank if i + 2 < len(lines) and not lines[i + 1].strip() and lines[i + 2].strip())
    double = len(nonblank) >= 8 and alternating >= 0.7 * len(nonblank)
    paras: list[list[tuple[int, str]]] = []
    cur: list[tuple[int, str]] = []
    blanks = 0
    for i, ln in enumerate(lines):
        if ln.strip():
            if cur and blanks >= (2 if double else 1):
                paras.append(cur)
                cur = []
            cur.append((i + 1, ln.strip() if double else ln.rstrip()))
            blanks = 0
        else:
            blanks += 1
    if cur:
        paras.append(cur)
    # a salutation line that opens a paragraph ("To the People of the State of New York:") is its own paragraph
    for k, p in enumerate(paras[:10]):
        if len(p) > 1 and _SALUTATION.match(p[0][1].strip()):
            paras[k:k + 1] = [p[:1], p[1:]]
            break
    return paras


def _para_text(lines_: list[str]) -> str:
    """Re-wrap a hard-wrapped paragraph into one line; keep line breaks for short lines (verse, lists, notes)."""
    if len(lines_) >= 3:
        lens = sorted(len(x.strip()) for x in lines_)
        if lens[len(lens) // 2] >= 45 and lens[-1] <= 110:
            return re.sub(r"\s+", " ", " ".join(x.strip() for x in lines_)).strip()
    return "\n".join(x.rstrip() for x in lines_).strip()


def plain_title(lines: list[str]) -> tuple[str, int]:
    """(title, index of the first line after the title block). The first header line such as "FEDERALIST No. 10",
    plus its subtitle lines ("The Same Subject Continued"), skipping file boilerplate. ("", 0) when none."""
    head = lines[:60]
    for i, ln in enumerate(head):
        t = ln.strip()
        if not t or _BOILERPLATE.search(t):
            continue
        if not _STRUCTURED_HEAD.match(t):
            return "", 0
        title = _titlecase_head(t)
        parts: list[str] = []
        j = i + 1
        blanks = 0
        while j < len(head) and len(parts) < 4 and blanks < 6:
            u = head[j].strip()
            if not u:
                blanks += 1
                if parts and blanks >= 2:
                    break
                j += 1
                continue
            if _BYLINE.match(u) or (u.isupper() and len(u.split()) <= 5) or _SALUTATION.match(u) or len(u) > 120:
                break
            parts.append(u)
            blanks = 0
            j += 1
        sub = " ".join(parts).strip()
        if sub:
            title = f"{title}: {sub}"
        return title[:150], j
    return "", 0


def chunk_plain(text: str, title: str = "") -> list[dict[str, Any]]:
    """Plain text split on paragraph breaks only: every chunk is whole paragraphs (a very long paragraph is cut
    between sentences), located by line range and paragraph number: "paragraph 7, lines 99 to 121"."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    paras = _plain_paragraphs(lines)
    if not paras:
        return []
    body_from = 0  # numbering starts after a salutation, if the text has one
    for k, p in enumerate(paras[:8]):
        if len(p) == 1 and _SALUTATION.match(p[0][1].strip()):
            body_from = k + 1
            break
    items: list[dict[str, Any]] = []
    n = 0
    for k, p in enumerate(paras):
        t_ = _para_text([x[1] for x in p])
        front = k < body_from
        num = None
        if not front and words(t_) >= 4:
            n += 1
            num = n
        items.append({"text": t_, "lines": p, "start": p[0][0], "end": p[-1][0], "num": num, "front": front,
                      "words": words(t_)})
    n_numbered = sum(1 for it in items if it["num"])  # a header and one long run of lines has no paragraph numbers
    out: list[dict[str, Any]] = []

    def emit(group: list[dict[str, Any]], body: str | None = None, part: int = 0, span: tuple[int, int] | None = None,
             total: int = 0) -> None:
        if not group:
            return
        a_, b_ = span or (group[0]["start"], group[-1]["end"])
        rng = f"lines {a_} to {b_}" if b_ > a_ else f"line {a_}"
        nums = [g["num"] for g in group if g["num"]]
        if n_numbered <= 1:  # no paragraph breaks in the file: only line ranges mean anything
            head = "lines"
        elif group[0]["front"]:
            head = "header"
        elif nums:
            head = f"paragraph {nums[0]}" if nums[0] == nums[-1] else f"paragraphs {nums[0]} to {nums[-1]}"
            if part:
                head += f" (part {part} of {total})"
        else:
            head = "text"
        c = chunk(body if body is not None else "\n\n".join(g["text"] for g in group), "line", a_,
                  heading=(f"{title}, {head}" if title else head) if head != "lines" else title, loc_end=b_)
        c["loc_label"] = rng if head == "lines" else f"{head}, {rng}"
        out.append(c)

    cur: list[dict[str, Any]] = []
    cur_w = 0
    for it in items:
        if it["words"] > PLAIN_PARA_MAX:  # one huge paragraph: cut between sentences, with each part's own lines
            emit(cur)
            cur, cur_w = [], 0
            pieces = _sentence_groups(it["text"], PLAIN_TARGET)
            offs, pos = [], 0
            for ln_no, ln_txt in it["lines"]:
                offs.append((pos, ln_no))
                pos += len(ln_txt.strip()) + 1
            search = 0
            for part, piece in enumerate(pieces, start=1):
                i0 = it["text"].find(piece[:40], search)
                i0 = search if i0 < 0 else i0
                i1 = i0 + len(piece)
                search = i1
                first = next((ln for o, ln in reversed(offs) if o <= i0), it["start"])
                last = next((ln for o, ln in reversed(offs) if o <= max(i0, i1 - 1)), it["end"])
                emit([it], piece, part, (first, last), len(pieces))
            continue
        if cur and (cur_w + it["words"] > PLAIN_TARGET or cur[0]["front"] != it["front"]):
            emit(cur)
            cur, cur_w = [], 0
        cur.append(it)
        cur_w += it["words"]
    emit(cur)
    return out


PLAIN_TARGET = 260
PLAIN_PARA_MAX = 600


def looks_markdown(text: str) -> bool:
    return len(re.findall(r"^#{1,6}\s+\S", text, re.M)) >= 2 or "```" in text


def extract_text(path: Path) -> Result:
    text = decode(path.read_bytes())
    meta: dict[str, Any] = {}
    is_txt = path.suffix.lower() == ".txt" and not looks_markdown(text)
    if is_txt:
        lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        title, _ = plain_title(lines)
        chunks = chunk_plain(text, title)
        if title:
            meta["title"] = title
            meta["structured_title"] = True
    else:
        chunks = chunk_markdown(text)
        t = top_title(text)
        if t:
            meta["title"] = t
        heads = _heading_lines(_blank_comments(text).replace("\r\n", "\n").split("\n"))
        if heads:
            meta["first_heading"] = strip_markdown(heads[0][2])
    if not chunks:
        raise ImportError_("This file is empty, so there is nothing to save.")
    meta["raw_head"] = text[:800]  # includes provenance comments, which help to recognise lecture notebooks
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
