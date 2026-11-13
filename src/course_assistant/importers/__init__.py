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

__all__ = ["import_file", "import_text", "import_url", "reimport", "detect_kind", "find_duplicate", "keep_copy", "link_to_lecture",
           "SUPPORTED_EXTENSIONS", "ImportError_"]

_SYLLABUS_WORDS = ("office hours", "grading", "midterm", "final exam", "prerequisite", "late policy",
                   "academic integrity", "course schedule", "learning outcomes", "course description",
                   "teaching assistant", "grade breakdown", "attendance policy", "textbook", "required reading")
_TEXT_EXT = {".md", ".txt", ".pdf", ".docx", ".html", ".htm", ".epub"}
# a name that says nothing about the content: the first heading of the file is a better title
_GENERIC_NAMES = re.compile(r"^(syllabus|notes?|my notes|untitled|document|doc|scan|scanned|img|image|photo|picture|file|"
                            r"handout|lecture|slides?|reading|readings|download|copy|new|page|text|paste|pasted)"
                            r"([\s_-]*\(?\d+\)?)?$|^(img|scan|dsc|screenshot|image|photo)[\s_-]*\d+", re.I)


def reading_kind() -> str:
    """The kind for a primary source or reading: "reading" when the store has it, else "handout"."""
    from ..store import MATERIAL_KINDS

    return "reading" if "reading" in MATERIAL_KINDS else "handout"


def lecture_notes_kind() -> str:
    """The kind for the instructor's own lecture notes or notebooks: "lecture_notes" when the store has it, else
    "reading" (course material that is not the student's own "My notes")."""
    from ..store import MATERIAL_KINDS

    return "lecture_notes" if "lecture_notes" in MATERIAL_KINDS else "reading"


# notes the instructor wrote (a converted lecture notebook, 'lecture notes' in the header) versus the student's own
_INSTRUCTOR_CUES = re.compile(r"lecture notebook|pluto notebook|jupyter notebook|\.ipynb|lecture notes|"
                              r"<!--\s*source:|converted to markdown|lecture summaries", re.I)
_STUDENT_CUES = re.compile(r"\bmy (own )?notes\b|\bTODO\b|\bsorry\b|\bi think\b|\bprof said\b|\bidk\b|\?\?|"
                           r"\bunfinished\b|\bhalf-?done\b|\bmess\b|\bnot sure\b", re.I)


def notes_kind(filename: str, text_sample: str = "") -> str:
    """'notes' (the student's own) or the instructor's lecture notes kind, for a file that is some kind of notes."""
    name = Path(filename or "").stem.lower()
    sample = (text_sample or "")[:1500]
    if re.search(r"\bmy\b|cheat|half|todo|unfinished|mine", name) or _STUDENT_CUES.search(sample):
        return "notes"
    if _INSTRUCTOR_CUES.search(sample):
        return lecture_notes_kind()
    return "notes"


def _term_year(term: str) -> str:
    m = re.search(r"(?:19|20)\d{2}", term or "")
    return m.group() if m else ""


# ------------------------------------------------------------------ kind detection
def detect_kind(filename: str, text_sample: str = "", term: str = "") -> str:
    """Guess a material kind from the file name, then from the first words of its text. `term` ("Fall 2026")
    lets a file named for an earlier term be recognised as a past offering."""
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
    past = re.search(r"\b(fall|spring|summer|winter)\s*((?:19|20)\d{2})\b", joined)
    if past and (not term or past.group(2) != _term_year(term) or "past" in tokens):
        return "past_offering"
    if has("past offering", "previous offering", "past offerings", "old exams", "past exams"):
        return "past_offering"
    readme = tokens == ["readme"] or (tokens[-1:] == ["readme"] and not has("syllabus"))
    if readme:  # a README describes a folder; its dates and policy words do not make it a syllabus
        return "notes"
    if has("handout", "handouts"):
        return "handout"
    if hw_name():
        return "assignment"
    if has("slides", "slide", "deck"):
        return "slides"
    if has("notes", "my notes", "cheatsheet", "cheat sheet"):
        return notes_kind(filename, text_sample)
    if has("lecture", "lec") or re.search(r"\blec\s*\d+", joined):
        if ext in AUDIO_VIDEO | CAPTIONS:
            return "recording"
        return notes_kind(filename, text_sample) if ext in {".md", ".txt", ".docx", ".html", ".htm"} else "slides"
    if has("textbook", "chapter", "book") or re.search(r"\bch\s*\d+", joined):
        return "textbook"
    if has("reading"):
        return reading_kind()
    sample = (text_sample or "")[:4000].lower()
    if sample:
        hits = sum(1 for w in _SYLLABUS_WORDS if w in sample)
        if hits >= 3 or ("syllabus" in sample[:300]) or (hits >= 2 and ("office hours" in sample or "grading" in sample)):
            return "syllabus"
        head = sample[:300]
        if re.search(r"\b(problem set|homework|assignment|pset)\s*#?\d*", head) and "due" in sample:
            return "assignment"
        if re.match(r"\s*(chapter|ch\.)\s*\d+\s*(?:[:.\-\u2013\u2014](?!\d)|\n)", head):
            return "textbook"
        if re.match(r"\s*(lecture|lec\.|week)\s*\d+\s*(?:[:.\-\u2013\u2014](?!\d)|\n|$)", head):
            return notes_kind(filename, text_sample) if ext in {".md", ".txt", ".docx", ".html", ".htm"} else "slides"
        if re.search(r"(?m)^\W*#*\s*(lecture|week)\s*\d{1,2}\b", sample) and ext in _TEXT_EXT:
            return notes_kind(filename, text_sample)
        if re.search(r"lecture notebook|pluto notebook|jupyter notebook|\.ipynb|lecture notes", sample[:600]):
            return notes_kind(filename, text_sample)
        if re.match(r"\s*(federalist|the federalist)\b", sample) or re.search(
                r"(?m)^\s*(federalist|letter|speech|address)\b.{0,12}\bno\.?\s*\d+", sample[:600]):
            return reading_kind()
    if ext == ".pptx":
        return "slides"
    if ext == ".epub":
        return "textbook"
    if ext in AUDIO_VIDEO or ext in CAPTIONS:
        return "recording"
    if ext in IMAGES:
        return "notes"
    if ext in _TEXT_EXT and len((text_sample or "").split()) >= 150:
        return "notes"
    return "other"


# ------------------------------------------------------------------ helpers
def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


_NUMBERED = re.compile(r"^(week|lecture|lec|chapter|ch|hw|ps|pset|problem set|lab|ex|exam|quiz|unit|part)[\s_-]*0*(\d+)\b",
                       re.I)
_LABELS = {"ch": "Chapter", "lec": "Lecture", "ps": "Problem set", "pset": "Problem set", "hw": "HW", "ex": "Exercise"}


def _pretty_title(name: str) -> str:
    """A readable title from a file name: "week-04-seam-carving.md" -> "Week 4 seam carving"."""
    stem = Path(name).stem
    stem = re.sub(r"[_]+", " ", stem)
    stem = re.sub(r"\s+", " ", stem).strip()
    if not stem:
        return "Untitled"
    spaced = re.sub(r"(?<=[a-z0-9])[-.](?=[a-z0-9])", " ", stem)
    spaced = re.sub(r"\s+", " ", spaced).strip()
    m = _NUMBERED.match(spaced)
    if m:
        label = _LABELS.get(m.group(1).lower(), m.group(1).capitalize())
        rest = spaced[m.end():].strip(" -")
        spaced = f"{label} {int(m.group(2))}" + (f" {rest}" if rest else "")
    elif "-" in stem or " " not in stem:
        spaced = spaced[:1].upper() + spaced[1:]
    return spaced


def _clean_title(title: str) -> str:
    t = extract.strip_markdown(title or "")
    return t[:150].strip()


def _is_generic(name: str) -> bool:
    return bool(_GENERIC_NAMES.match(Path(name).stem.strip()))


def _choose_title(orig: str, meta: dict[str, Any], given: str | None) -> str:
    """Explicit title, then the document's own title, then (for a name that says nothing) its first heading,
    then the file name made readable."""
    if given and given.strip():
        return _clean_title(given) or given.strip()
    t = _clean_title(meta.get("title", ""))
    if t:
        return t
    if _is_generic(orig):
        t = _clean_title(meta.get("first_heading", ""))
        if t:
            return t
    return _pretty_title(orig)


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


# ------------------------------------------------------------------ near-duplicate materials
_SHINGLE = 5
DUP_CONTAINMENT = 0.75    # this much of the smaller text is found in the other
DUP_JACCARD = 0.5         # and they share a good part of their text both ways
DUP_PAST_OFFERING = 0.95  # two past offerings of a course (other years) must be almost identical to count
DUP_SIZE_RATIO = 0.5      # and the two are of a similar size
DUP_MIN_WORDS = 40


def _shingles(text: str) -> set[int]:
    ws = re.findall(r"[^\W_]+", re.sub(r"\$[^$]*\$", " ", text.lower()))
    if len(ws) < _SHINGLE:
        return set()
    return {hash(tuple(ws[i:i + _SHINGLE])) for i in range(len(ws) - _SHINGLE + 1)}


def _material_shingles(store: Store, material_id: int) -> set[int]:
    return _shingles("\n".join(c["text"] for c in store.chunks_of(material_id)))


def find_duplicate(store: Store, course_id: int, material_id: int) -> int | None:
    """The id of an earlier material in the course whose text is near-identical to this one (a chapter as .md and
    .pdf, a syllabus as .md and .docx), or None. If that material is itself a copy, the original is returned."""
    mine = _material_shingles(store, material_id)
    mine_kind = (store.material(material_id) or {}).get("kind", "")
    if len(mine) < DUP_MIN_WORDS - _SHINGLE:
        return None
    best: tuple[float, int] | None = None
    for other in store.materials(course_id):
        oid = int(other["id"])
        if oid >= material_id:
            continue
        theirs = _material_shingles(store, oid)
        if len(theirs) < DUP_MIN_WORDS - _SHINGLE:
            continue
        small, big = sorted((len(mine), len(theirs)))
        if small / big < DUP_SIZE_RATIO:
            continue
        score = len(mine & theirs) / small
        jaccard = len(mine & theirs) / len(mine | theirs)
        need = DUP_PAST_OFFERING if other["kind"] == "past_offering" and mine_kind == "past_offering" else DUP_CONTAINMENT
        if score >= need and jaccard >= DUP_JACCARD and (best is None or score > best[0]):
            best = (score, oid)
    if best is None:
        return None
    canon = store.material(best[1])
    return int(canon["duplicate_of"]) if canon and canon.get("duplicate_of") else best[1]


def _mark_duplicate(store: Store, material_id: int) -> int | None:
    m = store.material(material_id)
    if not m:
        return None
    dup = find_duplicate(store, m["course_id"], material_id)
    if dup != m.get("duplicate_of"):
        store.update_material(material_id, duplicate_of=dup)
    if dup:
        pair_copies(store, material_id)
    return dup


# ------------------------------------------------------------------ a file saved twice (.md and .pdf): one citation
_PAGE_SUFFIX = re.compile(r"(?:,\s*|\s*\()pp?\.\s*\d+(?:\s*(?:to|-)\s*\d+)?\)?$")
_PDF_PAGE_NOTE = re.compile(r"^p\. \d+(?:, .*)?$")


def _is_pdf(m: dict[str, Any]) -> bool:
    return str(m.get("original_name") or m.get("filename") or "").lower().endswith(".pdf")


def _plain_key(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (t or "").lower())


def pair_copies(store: Store, material_id: int) -> None:
    """When a file is saved as both a PDF and something else (a .md chapter), make the one that is searched cite like
    the pair: the PDF takes the other copy's title, a .md citation shows the PDF pages its section is on ('Shays'
    Rebellion, pp. 15 to 16'), and a PDF citation shows the section heading ('p. 16, Shays' Rebellion')."""
    m = store.material(material_id)
    if not m:
        return
    other_id = m.get("duplicate_of")
    if not other_id:
        twins = [x for x in store.materials(m["course_id"]) if x.get("duplicate_of") == material_id]
        if not twins:
            return
        other_id = twins[0]["id"]
    other = store.material(int(other_id))
    if not other or _is_pdf(m) == _is_pdf(other):
        return
    pdf, txt = (m, other) if _is_pdf(m) else (other, m)
    if pdf["title"] != txt["title"] and _plain_key(pdf["title"]) in {
            _plain_key(_pretty_title(pdf.get("original_name") or "")), _plain_key(Path(pdf.get("original_name") or "").stem)}:
        store.update_material(pdf["id"], title=txt["title"])  # a file name turned into a title: the copy's is better
    if pdf["kind"] == "notes" and txt["kind"] == lecture_notes_kind() and txt["kind"] != "notes":
        store.update_material(pdf["id"], kind=txt["kind"])  # a PDF of instructor notes is guessed as My notes; its twin knows
    searched = pdf if not pdf.get("duplicate_of") else txt
    pages = [c for c in store.chunks_of(pdf["id"]) if c["loc_kind"] == "page"]
    sections = [c for c in store.chunks_of(txt["id"]) if c["loc_kind"] == "section"]
    if not pages or not sections:
        return
    owner: dict[int, int] = {}
    for c in pages:
        for sh in _shingles(c["text"]):
            owner[sh] = -1 if owner.get(sh, int(c["loc_start"])) != int(c["loc_start"]) else int(c["loc_start"])
    page_text = {int(c["loc_start"]): c for c in pages}
    best_section: dict[int, tuple[int, str]] = {}  # page -> (hits, section label)
    ranges: dict[int, tuple[int, int]] = {}
    for sec in sections:
        shs = _shingles(sec["text"])
        counts: dict[int, int] = {}
        for sh in shs:
            pg = owner.get(sh, -1)
            if pg >= 0:
                counts[pg] = counts.get(pg, 0) + 1
        keep = [pg for pg, n in counts.items() if n >= max(3, 0.1 * len(shs))]
        if keep:
            ranges[sec["id"]] = (min(keep), max(keep))
        for pg, n in counts.items():
            if n >= 3 and n > best_section.get(pg, (0, ""))[0]:
                best_section[pg] = (n, re.sub(_PAGE_SUFFIX, "", sec["loc_label"]))
    if searched["id"] == txt["id"]:
        for sec in sections:
            rng = ranges.get(sec["id"])
            if not rng:
                continue
            base = re.sub(_PAGE_SUFFIX, "", sec["loc_label"])
            label = f"{base}, p. {rng[0]}" if rng[0] == rng[1] else f"{base}, pp. {rng[0]} to {rng[1]}"
            if label != sec["loc_label"]:
                store._exec("UPDATE chunks SET loc_label = ? WHERE id = ?", (label, sec["id"]))
    else:
        for pg, c in page_text.items():
            sec = best_section.get(pg)
            label = f"p. {pg}, {sec[1]}" if sec else f"p. {pg}"
            if label != c["loc_label"] and (_PDF_PAGE_NOTE.match(c["loc_label"]) or not c["loc_label"]):
                store._exec("UPDATE chunks SET loc_label = ? WHERE id = ?", (label, c["id"]))


def keep_copy(store: Store, material_id: int) -> int | None:
    """Search this copy instead of the file it was matched with (a PDF has page numbers, a .md has none).

    Swaps the pair: the copy becomes the searched material, the former original becomes the unsearched copy, and
    anything else that was a copy of the original now points at this one. A lecture link or date the original had
    moves over too. Returns the id of the former original, or None when the material was not a copy."""
    m = store.material(material_id)
    if not m or not m.get("duplicate_of"):
        return None
    orig_id = int(m["duplicate_of"])
    orig = store.material(orig_id)
    if not orig:
        store.update_material(material_id, duplicate_of=None)
        return None
    others = [x["id"] for x in store.materials(m["course_id"]) if x.get("duplicate_of") == orig_id and x["id"] != material_id]
    store.update_material(material_id, duplicate_of=None)
    store.update_material(orig_id, duplicate_of=material_id)
    for oid in others:
        store.update_material(oid, duplicate_of=material_id)
    moved: dict[str, Any] = {}
    if orig.get("event_id") and not m.get("event_id"):
        moved["event_id"] = orig["event_id"]
    if orig.get("lecture_date") and not m.get("lecture_date"):
        moved["lecture_date"] = orig["lecture_date"]
    if moved:
        store.update_material(material_id, **moved)
    pair_copies(store, orig_id)  # the copy now searched cites with the pair's labels
    return orig_id


# ------------------------------------------------------------------ linking a material to its lecture
_LECTURE_REF = re.compile(r"(?:lecture|lec|class|session|week|wk)[\s._-]*0*(\d{1,2})(?!\d)", re.I)
_LECTURE_PREFIX = re.compile(r"^\W*(?:lecture|lec\.?|class|session|week|wk\.?)\s*0*(\d{1,2})\b", re.I)
_TOPIC_STOP = {"intro", "introduction", "notes", "note", "lecture", "lectures", "slides", "slide", "captions", "caption",
               "summary", "summaries", "the", "and", "of", "for", "to", "in", "on", "a", "an", "week", "class", "part",
               "recording", "video", "audio", "handout", "chapter", "fall", "spring", "my", "with", "how", "is", "are"}
_LINKABLE = {"slides", "recording", "notes", "lecture_notes", "handout", "other", "website", "reading"}


def _stem(w: str) -> str:
    for suf in ("ization", "ations", "ation", "ings", "ing", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[:-len(suf)]
    return w


def _topic_tokens(text: str) -> set[str]:
    return {_stem(w) for w in re.findall(r"[a-z][a-z0-9]+", text.lower()) if w not in _TOPIC_STOP and len(w) > 2}


def link_to_lecture(store: Store, material_id: int) -> int | None:
    """Link a material to the lecture event it belongs to, when its title, file name or first heading names the
    lecture number, or names the topic of exactly one lecture. Sets materials.event_id and, when the material has
    no lecture date, lecture_date from the event. Returns the event id, or None when no single lecture fits.
    A material the student already linked by hand is left alone."""
    m = store.material(material_id)
    if not m or m.get("event_id") or m["kind"] not in _LINKABLE:
        return None
    lectures = [e for e in store.events(m["course_id"], kinds=["lecture"])]
    if not lectures:
        return None
    first = next((c for c in store.chunks_of(material_id)), None)
    heading = (first or {}).get("heading", "") if first else ""
    names = [m["title"], Path(m.get("original_name") or "").stem, heading]
    event: dict[str, Any] | None = None
    nums = {int(n) for t in names for n in _LECTURE_REF.findall(t)}
    if len(nums) == 1:
        n = nums.pop()
        hit = [e for e in lectures if (mm := _LECTURE_PREFIX.match(e["title"])) and int(mm.group(1)) == n]
        if len(hit) == 1:
            event = hit[0]
    if event is None:
        mine: set[str] = set()
        for t in names[:3]:
            mine |= _topic_tokens(t)
        if mine and any(len(t) >= 5 for t in mine):
            hits = []
            for e in lectures:
                topic = re.sub(r"^\W*(?:lecture|lec\.?|class|week)\s*\d+\s*[:.-]?\s*", "", e["title"], flags=re.I)
                if mine <= _topic_tokens(topic):
                    hits.append(e)
            if len(hits) == 1:
                event = hits[0]
    if event is None:
        return None
    values: dict[str, Any] = {"event_id": event["id"]}
    if not m.get("lecture_date"):
        values["lecture_date"] = event["date"]
    store.update_material(material_id, **values)
    return int(event["id"])


def delete_material(store: Store, material_id: int) -> int:
    """Delete a material and the calendar events its syllabus import created (events carry source_material_id and
    added_by = 'syllabus'; events the student added or edited by hand stay). Lectures linked to those events are
    unlinked first. The graded items split from it go too (items added by hand stay). Returns how many events went with it."""
    m = store.material(material_id)
    if not m:
        return 0
    gone = [e["id"] for e in store.events(m["course_id"]) if e["source_material_id"] == material_id
            and e["added_by"] == "syllabus"]
    for other in store.materials(m["course_id"]):
        if other.get("event_id") in gone:
            store.update_material(other["id"], event_id=None)
    store.clear_syllabus_events(material_id)
    for item in store.graded_items(m["course_id"]):  # the graded items made from this file go with it: re-adding it and
        if item.get("material_id") == material_id:  # marking it graded again must not list every problem twice
            store.delete_graded_item(item["id"])
    store.delete_material(material_id)
    return len(gone)


def _link_all(store: Store, course_id: int) -> int:
    n = 0
    for m in store.materials(course_id):
        if not m.get("event_id") and link_to_lecture(store, int(m["id"])):
            n += 1
    return n


def _process_syllabus(store: Store, material_id: int) -> int:
    """Put a syllabus's dates on the calendar. A copy of a syllabus already saved adds nothing, and a date that is
    already on the calendar (same day, kind and similar title) is not added again."""
    m = store.material(material_id)
    if not m or m["kind"] != "syllabus":
        return 0
    old_ids = {e["id"] for e in store.events(m["course_id"]) if e["source_material_id"] == material_id
               and e["added_by"] == "syllabus"}
    store.clear_syllabus_events(material_id)
    for other in store.materials(m["course_id"]):  # lectures linked to a date that is about to be re-read
        if other.get("event_id") in old_ids:
            store.update_material(other["id"], event_id=None)
    n = 0
    if not m.get("duplicate_of"):
        text = "\n".join(c["text"] for c in store.chunks_of(material_id))
        existing = store.events(m["course_id"])
        for ev in calendar.extract_events(text, _year_hint(store, m)):
            if any(calendar.similar_event(e, ev) for e in existing):
                continue
            store.add_event(m["course_id"], ev["kind"], ev["title"], ev["date"], time=ev.get("time", ""),
                            topics=ev.get("notes", ""), source_material_id=material_id, added_by="syllabus")
            existing.append(ev)
            n += 1
    _link_all(store, m["course_id"])
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

    course = store.course(course_id) or {}
    if kind is None:
        kind = detect_kind(orig, (meta.get("raw_head", "") + "\n" + _sample(chunks)).strip(), course.get("term", ""))
        if kind == "other" and meta.get("structured_title"):
            kind = reading_kind()  # a primary source with a header such as "FEDERALIST No. 10"
    final_title = _choose_title(orig, meta, title)
    values: dict[str, Any] = {
        "filename": f"{course_id}/{dest.name}", "original_name": orig, "mime": MIME.get(ext, ""), "sha256": sha,
        "lecture_date": lecture_date, "added_by": added_by, "page_count": meta.get("page_count"),
        "duration_seconds": meta.get("duration_seconds"), "status": "ready"}
    from .. import transcribe  # imported here: transcribe itself imports importers.common

    if ext in AUDIO_VIDEO and not chunks:
        values["status"] = "needs_transcription"
        values["status_note"] = transcribe.pending_note(store, "transcribe")
    elif ext in IMAGES:
        values["status"] = "needs_transcription"
        values["status_note"] = transcribe.pending_note(store, "ocr")
    mid = store.add_material(course_id, kind, final_title, **values)
    if chunks:
        store.replace_chunks(mid, chunks)
        _mark_duplicate(store, mid)
    if values["status"] == "needs_transcription":
        store.add_job("ocr" if ext in IMAGES else "transcribe", mid, note=final_title)
    _process_syllabus(store, mid)
    link_to_lecture(store, mid)
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
    final_title = _clean_title(title) or extract.top_title(text) or "Pasted text"
    mid = store.add_material(course_id, kind, final_title, mime="text/markdown", sha256=sha,
                             added_by=added_by, status="ready")
    store.replace_chunks(mid, chunks)
    _mark_duplicate(store, mid)
    _process_syllabus(store, mid)
    link_to_lecture(store, mid)
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
            from .. import transcribe

            store.add_job("ocr", material_id, note=m["title"])
            store.update_material(material_id, status="needs_transcription",
                                  status_note=transcribe.pending_note(store, "ocr"))
        return
    if ext in AUDIO_VIDEO:
        sib = _caption_sibling(path)
        if sib is not None:
            chunks, meta = extract.extract_captions(sib)
            store.replace_chunks(material_id, chunks)
            store.update_material(material_id, status="ready", status_note="", duration_seconds=meta.get("duration_seconds"))
            link_to_lecture(store, material_id)
        elif not store.chunks_of(material_id) and not _has_open_job(store, material_id, "transcribe"):
            from .. import transcribe

            store.add_job("transcribe", material_id, note=m["title"])
            store.update_material(material_id, status="needs_transcription",
                                  status_note=transcribe.pending_note(store, "transcribe"))
        return
    chunks, meta = _extract(path, ext)
    store.replace_chunks(material_id, chunks)
    store.update_material(material_id, status="ready", status_note="", page_count=meta.get("page_count"),
                          **({"duration_seconds": meta["duration_seconds"]} if "duration_seconds" in meta else {}))
    _mark_duplicate(store, material_id)
    _process_syllabus(store, material_id)
    link_to_lecture(store, material_id)


def _has_open_job(store: Store, material_id: int, kind: str) -> bool:
    return any(j["material_id"] == material_id and j["kind"] == kind and j["status"] in ("queued", "running")
               for j in store.jobs())
