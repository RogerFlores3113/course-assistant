"""Study tools: primer, exam guide, concept map, explain, practice, flashcards, missed, dashboard.

Each generator gathers numbered sources from the course, asks the provider, runs the reply through
grounding.parse_answer (invalid [n] dropped, uncited paragraphs labelled outside), appends the disclaimer footer and
saves a guide. Modules written by other builders (retrieval, grounding, integrity) are imported at call time.
"""
from __future__ import annotations

import importlib
import json
import re
from datetime import date as _date, datetime, timedelta, timezone
from typing import Any

from . import disclaimer, locations
from .store import Store, now_iso

DEPTHS = {"quick": "quick", "standard": "standard", "from scratch": "from scratch", "scratch": "from scratch",
          "from_scratch": "from scratch"}
GRADES = ("again", "hard", "good", "easy")
MAX_SOURCES = 28
MAX_CHARS = 1400


class NoProvider(Exception):
    """No AI is connected. The web shows 'connect an AI'; the MCP path still works."""

    def __init__(self, message: str = "Connect an AI in Settings (or use Claude Desktop through the MCP server) to use this."):
        super().__init__(message)


def _mod(name: str):
    return importlib.import_module(f"course_assistant.{name}")


def _need(provider) -> None:
    if provider is None:
        raise NoProvider()


def _today() -> str:
    return _date.today().isoformat()




# ---------------------------------------------------------------- results and wording
READY_MESSAGES = {
    "primer": "Your primer is ready.", "exam": "Your exam guide is ready.", "exam_guide": "Your exam guide is ready.",
    "map": "Your concept map is ready.", "concept_map": "Your concept map is ready.",
    "explain": "The explanation is ready.", "practice": "Your practice questions are ready.",
    "missed": "Your catch-up is ready.", "twin": "Your twin problem is ready.", "flashcards": "Your flashcards are ready.",
}


def ready_message(kind: str) -> str:
    """'Your practice questions are ready.' for a guide kind or a Study action name."""
    return READY_MESSAGES.get((kind or "").strip().lower(), "Your guide is ready.")


class FlashcardResult(list):
    """Ids of the cards made. Also says what was left out, so the web can tell the student."""

    dropped: int = 0          # cards whose cite pointed at no source
    duplicates: int = 0       # cards the student already has
    guarded: int = 0          # cards left out because they gave away graded work
    malformed: int = 0
    note: str = ""


# ---------------------------------------------------------------- sources
def _usable(m: dict[str, Any]) -> bool:
    return not m.get("duplicate_of") and not m.get("graded") and m["kind"] not in ("syllabus", "past_offering")


def _rfn(name: str, *args) -> Any:
    """A retrieval helper by name; False when this retrieval does not have it."""
    fn = getattr(_mod("retrieval"), name, None)
    return fn(*args) if callable(fn) else False


def _is_noise(text: str) -> bool:
    fn = getattr(_mod("retrieval"), "is_manifest_noise", None)
    return bool(fn and fn(text))


def _source_from_chunk(c: dict[str, Any], title: str = "", kind: str = "") -> dict[str, Any]:
    return {"chunk_id": c["id"], "material_id": c["material_id"],
            "material_title": c.get("material_title") or title, "material_kind": c.get("material_kind") or kind,
            "loc_kind": c["loc_kind"], "loc_start": c["loc_start"], "loc_label": c["loc_label"],
            "heading": c.get("heading", ""), "text": c["text"], "href": locations.href_for(c), "score": 0.0}


def _spread_pick(items: list, k: int) -> list:
    """k items spread evenly over the list, in order."""
    if k >= len(items):
        return list(items)
    if k <= 0:
        return []
    return [items[int(i * len(items) / k)] for i in range(k)]


def _material_sources(store: Store, material_id: int, limit: int = 12, *, spread: bool = False) -> list[dict[str, Any]]:
    m = store.material(material_id)
    if not m:
        return []
    chunks = [c for c in store.chunks_of(material_id) if not _is_noise(c["text"])]
    if spread:  # sampling across a file: skip bare titles and byline blocks, which only say who wrote it
        r = _mod("retrieval")
        is_title, is_head = getattr(r, "is_title_only", None), getattr(r, "is_header_chunk", None)
        is_filler = getattr(r, "is_filler_chunk", None)  # chapter intros and end-of-chapter exercise lists
        body = [c for c in chunks if not (is_title and is_title(c["text"], c.get("heading", "")))
                and not (is_head and is_head(c["text"], m.get("title", "")))
                and not (is_filler and is_filler(c["text"], c.get("heading", "")))]
        chunks = body or chunks
    chunks = _spread_pick(chunks, limit) if spread else chunks[:limit]
    return [_source_from_chunk(c, m["title"], m["kind"]) for c in chunks]


def _search(store: Store, course_id: int, query: str, limit: int = 6, **kw) -> list[dict[str, Any]]:
    if not query.strip():
        return []
    kw.setdefault("exclude_graded", True)  # graded material never reaches the model as a numbered source
    retrieval = _mod("retrieval")
    try:
        return retrieval.search(store, course_id, query, limit=limit, **kw)
    except TypeError:  # an older retrieval without exclude_graded: filter the graded materials out here
        graded = _graded_ids(store, course_id) if kw.pop("exclude_graded", False) else set()
        return [x for x in retrieval.search(store, course_id, query, limit=limit, **kw) if x.get("material_id") not in graded]


def _defining_first(sources: list[dict[str, Any]], topic: str) -> list[dict[str, Any]]:
    """Passages that define the topic come before ones that only mention it ("Explain eigenvalues" opens with the
    definition, not a chapter that uses eigenvalues). The order inside each group stays as retrieval ranked it."""
    score = getattr(_mod("retrieval"), "_definition_score", None)
    nq = getattr(_mod("retrieval"), "normalize_query", None)
    if not score or not nq or len(sources) < 2:
        return sources
    try:
        terms = [t for t in nq(topic)["core"] if t not in getattr(_mod("retrieval"), "_GENERIC_TOPIC", ())]
    except Exception:
        return sources
    if not terms:
        return sources

    def defines(x: dict[str, Any]) -> bool:
        try:
            return score(terms, {"text": x.get("text") or "", "heading": x.get("heading") or ""}) >= 1.5
        except Exception:
            return False
    flags = [defines(x) for x in sources]
    return [x for x, f in zip(sources, flags) if f] + [x for x, f in zip(sources, flags) if not f]


def _merge(*lists: list[dict[str, Any]], cap: int = MAX_SOURCES) -> list[dict[str, Any]]:
    seen: set[int] = set()
    out: list[dict[str, Any]] = []
    for lst in lists:
        for s in lst:
            if s["chunk_id"] in seen:
                continue
            seen.add(s["chunk_id"])
            out.append(s)
            if len(out) >= cap:
                return out
    return out


def _interleave(lists: list[list[dict[str, Any]]], cap: int = MAX_SOURCES) -> list[dict[str, Any]]:
    """Round-robin so every topic gets a share of the source budget."""
    out: list[list[dict[str, Any]]] = [list(x) for x in lists]
    merged: list[dict[str, Any]] = []
    while any(out) and len(merged) < cap * 3:
        for lst in out:
            if lst:
                merged.append(lst.pop(0))
    return _merge(merged, cap=cap)


def _spread_materials(store: Store, material_ids: list[int], cap: int, per: int = 3) -> list[dict[str, Any]]:
    """Every material gets a share of the budget, drawn from across the whole file, not just its start."""
    n = max(1, len(material_ids))
    per = max(1, min(per, cap // n))
    return _interleave([_material_sources(store, mid, per, spread=True) for mid in material_ids], cap=cap)


_TERM = re.compile(r"\b(fall|autumn|spring|summer|winter|fa|sp|su|wi)[\s\-_']*((?:20)?\d{2})\b", re.I)
_SEASONS = {"fa": "Fall", "autumn": "Fall", "sp": "Spring", "su": "Summer", "wi": "Winter"}


def term_of(text: str) -> str:
    """'Fall 2018' from 'fall2018-lecture-summaries.md', 'sp22' ..., or ''."""
    m = _TERM.search(text or "")
    if not m:
        return ""
    season = m.group(1).lower()
    season = _SEASONS.get(season, season.capitalize())
    year = m.group(2)
    return f"{season} {year if len(year) == 4 else '20' + year}"


def _past_label(store: Store, course_id: int, material_id: int, cache: dict) -> str:
    """'' for this term's material; else 'from a past offering, Fall 2018; may differ from this term'."""
    if material_id in cache:
        return cache[material_id]
    m = store.material(material_id) or {}
    course = store.course(course_id) or {}
    term = term_of(" ".join(str(m.get(k, "")) for k in ("title", "original_name", "source_url")))
    mine = term_of(course.get("term", ""))
    past = m.get("kind") == "past_offering" or bool(term and mine and term != mine)
    label = ""
    if past:
        label = "from a past offering" + (f", {term}" if term else "") + "; may differ from this term"
    cache[material_id] = label
    return label


def _graded_ids(store: Store, course_id: int) -> set[int]:
    return {m["id"] for m in store.materials(course_id) if m.get("graded")}


def _prepare(store: Store, course_id: int, sources: list[dict[str, Any]], keep_graded: bool = False) -> list[dict[str, Any]]:
    """Copies of the sources with past-offering labels. Passages of graded materials are dropped (they are never sent to
    the model as numbered sources), unless keep_graded: a twin's own essay-prompt sources, numbered by integrity."""
    cache: dict = {}
    out = []
    graded = set() if keep_graded else _graded_ids(store, course_id)
    for s in sources:
        if s.get("material_id") in graded:
            continue
        s = dict(s)
        label = _past_label(store, course_id, s["material_id"], cache)
        if label:
            s["past_label"] = label
        out.append(s)
    return out


def _sources_block(sources: list[dict[str, Any]]) -> str:
    if not sources:
        return "(No course materials matched. Everything you say is outside knowledge and must be labelled.)"
    parts = []
    for i, s in enumerate(sources, 1):
        head = f"[{i}] {s.get('material_title', '')} ({s.get('loc_label', '')})"
        if s.get("heading"):
            head += f" - {s['heading']}"
        if s.get("past_label"):
            head += f" [{s['past_label']}]"
        parts.append(f"{head}\n{(s.get('text') or '')[:MAX_CHARS]}")
    return "\n\n".join(parts)


def _guide_sources(sources: list[dict[str, Any]], used: list[int]) -> list[dict[str, Any]]:
    out = []
    for n in sorted({int(u) for u in used}):
        if 1 <= n <= len(sources):
            s = sources[n - 1]
            out.append({"n": n, "chunk_id": s["chunk_id"], "material_id": s["material_id"], "label": s["loc_label"],
                        "title": s["material_title"], "href": s["href"]})
    return out


FALLBACK_FORMAT_RULES = ("Put ALL math, including vectors and matrices, inside $...$ or $$...$$. Put code in backticks. "
                         "Square brackets are only for citations like [2], outside math and code.")

SYSTEM = """You are Course Assistant, a patient study helper for a college student. Rules you always follow:
- Ground everything in the numbered sources the student's own course materials provide. Cite the source number in square brackets like [2] right after each claim. Cite only numbers that exist.
- If the sources do not cover something and you add it from general knowledge, put it inside <outside>...</outside> tags so the app can label it "Not from your materials". Never present outside knowledge as if it came from the sources.
- Never invent page numbers, quotes, dates or deadlines. If the materials do not say, say so.
- A source marked "from a past offering" comes from an earlier term. Say so when you use it ("in Fall 2018, ...") and never present it as this term's policy, dates or exam format.
- Use plain, friendly language a student can read quickly. Define jargon the first time you use it.
{subject_line}
- Use Markdown with short headings, bullets and numbered lists."""

_MATH_LINE = "- This is often STEM material: keep units and steps explicit."
_TEXT_LINE = ("- This is often reading and argument based material: name the author or document, say what it claims, "
              "and keep any quotation short and exact.")
_MATH_SIGNS = re.compile(r"\$[^$\n]{2,}\$|\\frac|\\begin|\bsqrt\b|\^\d|[=\u2248]\s*-?\d|\b\d+(?:\.\d+)?\s*(?:mol|mmol|g|mg|kg|ml|l|m/s|hz|ev)\b",
                         re.I)


def has_math(store: Store, course_id: int) -> bool:
    """The course's materials have math in them: a graded numeric problem, or several passages with equations, numbers
    with units or LaTeX. Humanities courses say 'readings and arguments' in the prompts; STEM courses say numbers."""
    try:
        if any((g.get("kind") or "problem") in ("problem", "proof") and (g.get("body") or "").strip()
               and re.search(r"\d", g.get("body") or "") for g in store.graded_items(course_id)):
            return True
        seen = 0
        for m in store.materials(course_id):
            if not _usable(m):
                continue
            for c in store.chunks_of(m["id"])[:60]:
                if _MATH_SIGNS.search(c.get("text") or ""):
                    seen += 1
                    if seen >= 4:
                        return True
    except Exception:
        return True  # unknown: keep the STEM wording that older versions used
    return False


FALLBACK_INTEGRITY = ("Academic integrity: never solve a problem the student says is graded. Work an example with "
                      "different numbers instead.")


def _system(store: Store, course_id: int) -> str:
    """The study system prompt: base rules, the shared format rules, and the course's integrity block."""
    parts = [SYSTEM.format(subject_line=_MATH_LINE if has_math(store, course_id) else _TEXT_LINE),
             getattr(_mod("grounding"), "FORMAT_RULES", FALLBACK_FORMAT_RULES)]
    block = getattr(_mod("integrity"), "prompt_block", None)
    if not block:
        parts.append(FALLBACK_INTEGRITY)
    else:
        try:
            text = block(store, course_id)
        except Exception:
            text = ""
        if text and text.strip():
            parts.append(text.strip())
    return "\n\n".join(parts)


def _route_graded(store: Store, course_id: int, provider, question: str, graded: dict[str, Any]) -> int:
    """Answer a request that matched graded work with integrity.twin (a twin, or coaching for an essay or proof) and
    return the guide id."""
    ans = _mod("integrity").twin(store, course_id, question, provider, graded)
    gid = getattr(ans, "guide_id", None)
    if not gid:
        mode = getattr(ans, "mode", "") or "concepts"
        gid = store.add_guide(course_id, "twin", f"Twin problem: {graded.get('title', '')}"[:90],
                              _clean_body(getattr(ans, "body_md", "")) + "\n\n" + disclaimer.FOOTER,
                              data={"blocks": _block_dicts(getattr(ans, "blocks", None)), "mode": mode,
                                    "graded_item_id": graded.get("id")},
                              question=question, model=getattr(provider, "model", "") or "")
    return int(gid)


def _guard(store: Store, course_id: int, provider, raw: str, question: str) -> tuple[str, int | None]:
    """Integrity post-check of a reply. Returns (reply without marker, id of a twin/concepts guide if routed)."""
    integ = _mod("integrity")
    item_id = None
    reader = getattr(integ, "read_marker", None)
    if reader:
        item_id, raw = reader(raw)
    graded = None
    if item_id not in (None, ""):
        try:
            graded = store.graded_item(int(item_id))
        except (TypeError, ValueError):
            graded = None
    if graded is None and item_id == 0:  # a bare <graded/> marker: the model says graded but named no item
        graded = integ.match_graded(store, course_id, question) or next(iter(store.graded_items(course_id)), None)
    if graded:
        ans = integ.twin(store, course_id, question, provider, graded)
        gid = getattr(ans, "guide_id", None)
        if not gid:  # a twin that was not saved by its author: save it so the student gets it
            mode = getattr(ans, "mode", "") or ("twin" if getattr(ans, "twin", False) else "concepts")
            gid = store.add_guide(course_id, "twin", f"Twin problem: {graded.get('title', '')}"[:90],
                                  _clean_body(getattr(ans, "body_md", "")) + "\n\n" + disclaimer.FOOTER,
                                  data={"blocks": _block_dicts(getattr(ans, "blocks", None)), "mode": mode,
                                        "graded_item_id": graded.get("id")},
                                  question=question, model=getattr(provider, "model", "") or "")
        return raw, int(gid)
    return raw, None


_DETAILS_ONE_LINE = re.compile(r"<details>\s*<summary>(.*?)</summary>[ \t]*", re.I)


def normalize_details(text: str) -> str:
    """<details>, <summary> and </details> each on their own line, so every renderer shows a hidden solution."""
    text = _DETAILS_ONE_LINE.sub(lambda m: f"\n<details>\n<summary>{m.group(1)}</summary>\n", text or "")
    text = re.sub(r"[ \t]*</details>[ \t]*", "\n</details>\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip("\n")


# an empty stub removes only itself: the summary pattern cannot cross "</summary>" or "<details>", and a stub cut off at
# the end of the reply (no closing tag, or a half-written summary) goes too
_EMPTY_DETAILS = re.compile(
    r"<details>\s*(?:<summary>(?:(?!</summary>|<details>|</details>).)*</summary>)?\s*</details>[ \t]*"
    r"|<details>\s*(?:<summary>(?:(?!</summary>|<details>|</details>).)*(?:</summary>)?)?\s*\Z", re.S | re.I)


def drop_empty_details(text: str) -> str:
    """A "Show solution" block with nothing inside is dropped (a stub tells the student nothing). Text without one
    comes back as it was."""
    norm = normalize_details(text or "")
    out = _EMPTY_DETAILS.sub("", norm)
    if out == norm:
        return text
    return re.sub(r"\n{3,}", "\n\n", out).strip("\n")


def _clean_body(body: str) -> str:
    body = (body or "").replace(disclaimer.FOOTER, "").rstrip()
    return body


_SPLIT_QS = re.compile(r"(<details>.*?</details>|\n\s*\n)", re.S | re.I)
# a numbered question: "1.", "2)", "**3.**", "**4**.", "Question 5:" (the model often bolds its numbers)
_NUM_Q = re.compile(r"^\s*\*{0,2}(?:(?:Question|Q)\s*)?(\d+)(?:[.):]\*{0,2}|\*{1,2}[.):]?)\s")
_SCRUB_HEAD = re.compile(r"^(?:A part|\d+ parts) of this guide (?:was|were) left out because", re.I)


def _count_note(note: str, n: int) -> str:
    """The drop note counts questions and its grammar follows the count: 'A question was left out because it matched',
    '3 questions were left out because they matched'."""
    if not note or not _SCRUB_HEAD.match(note):
        return note
    one = n == 1
    note = _SCRUB_HEAD.sub(f"{n} practice question{'' if one else 's'} {'was' if one else 'were'} left out because", note)
    return re.sub(r"(left out because )(?:it|they)\b", r"\1" + ("it" if one else "they"), note, count=1)


def tidy_practice(before: str, after: str, scrubbed: int, note: str) -> tuple[str, int, str]:
    """A practice set after the graded-work scrub: a numbered question whose solution was cut is dropped too (no empty
    stub), the questions left are numbered 1, 2, 3 again, and the note counts questions, not parts: a copied question
    and its solution are one question. Returns (text, questions left out, note)."""
    if not scrubbed:
        return after, 0, note

    def paras(text: str) -> list[tuple[str, bool]]:
        segs = [x for x in _SPLIT_QS.split(text) if x.strip()]
        out = []
        for i, seg in enumerate(segs):
            nxt = segs[i + 1] if i + 1 < len(segs) else ""
            out.append((seg, nxt.lower().lstrip().startswith("<details>")))
        return out
    was = {seg.strip(): has_det for seg, has_det in paras(before) if _NUM_Q.match(seg)}
    kept: list[str] = []
    segs = paras(after)
    gone = sum(1 for q in was if q not in {seg.strip() for seg, _ in segs})
    for i, (seg, has_det) in enumerate(segs):
        if _NUM_Q.match(seg) and was.get(seg.strip()) and not has_det:
            gone += 1  # its solution was cut: the question alone is an empty stub
            continue
        kept.append(seg.strip())
    if not gone:
        # questions that are not numbered "1." (a bold or "Question 1" stem): count what disappeared by block, so one
        # dropped question with its solution never reads as "2 parts"
        a_set = {seg.strip() for seg, _ in segs}
        b_segs = paras(before)
        lost_q = sum(1 for seg, has_det in b_segs if not seg.lower().lstrip().startswith("<details>") and seg.strip() not in a_set
                     and has_det)
        lost_d = sum(1 for seg, _ in b_segs if seg.lower().lstrip().startswith("<details>") and seg.strip() not in a_set)
        loose = lost_q or (lost_d if not any(not seg.lower().lstrip().startswith("<details>") and seg.strip() not in a_set
                                             for seg, _ in b_segs) else 0)
        if not loose or not note or not _SCRUB_HEAD.match(note):
            return after, scrubbed, note
        return after, loose, _count_note(note, loose)
    n = 0
    renum = []
    for seg in kept:
        m = _NUM_Q.match(seg)
        if m and not seg.lower().startswith("<details>"):
            n += 1
            seg = seg[:m.start(1)] + str(n) + seg[m.end(1):]  # the model's own bold or "Question" wording stays
        renum.append(seg)
    return "\n\n".join(renum), gone, _count_note(note, gone)


def _compose(store: Store, course_id: int, provider, task: str, sources: list[dict[str, Any]], *, question: str = "",
             max_tokens: int = 4000, practice: bool = False) -> dict[str, Any]:
    """Ask, guard, parse. Returns {routed, blocks, body, used, outside, not_covered, sources}."""
    sources = _prepare(store, course_id, sources)
    user = f"{task}\n\nSOURCES\n\n{_sources_block(sources)}"
    raw = provider.complete(_system(store, course_id), [{"role": "user", "content": user}], max_tokens=max_tokens)
    raw, routed = _guard(store, course_id, provider, raw, question or task[:300])
    if routed:
        return {"routed": routed}
    integ = _mod("integrity")
    scrubbed, scrub_note, scrub_ids = 0, "", []
    if practice:
        raw = drop_empty_details(raw)
    before = raw
    if getattr(integ, "scrub_parts", None):  # drop only the parts the item's key flags, and say which item matched
        with _mod("essay").reply_sources(sources):  # fx13: a cited textbook fact the chunk says is no essay prose
            raw, scrubbed, scrub_note = integ.scrub_parts(store, course_id, raw, question or task[:300])
        if practice:
            raw, scrubbed, scrub_note = tidy_practice(before, raw, scrubbed, scrub_note)
        scrub_ids = list(getattr(integ, "LAST_SCRUB_ITEMS", []) or []) if scrubbed else []
    elif getattr(integ, "scrub_markdown", None):
        raw, scrubbed = integ.scrub_markdown(store, course_id, raw, question or task[:300])
    held = bool(scrubbed) and not re.sub(r"^[ \t]*#{1,6}[ \t].*$|<[^>]*>", "", raw, flags=re.M).strip()
    if held:  # every part was held back: the page says so (once); it never says the AI gave no answer
        return {"routed": None, "blocks": [], "body": "", "used": [], "outside": False, "not_covered": False,
                "sources": sources, "scrubbed": scrubbed, "scrub_note": scrub_note, "scrub_items": scrub_ids, "held": True}
    # the banners (not covered, no passages cited) are worked out from what remains, never from the note
    ans = _mod("grounding").parse_answer(normalize_details(raw), sources)
    return {"routed": None, "blocks": _block_dicts(ans.blocks), "body": _clean_body(ans.body_md),
            "used": list(ans.used or []), "outside": bool(getattr(ans, "outside", False)),
            "not_covered": bool(getattr(ans, "not_covered", False)), "sources": sources, "scrubbed": scrubbed,
            "scrub_note": scrub_note, "scrub_items": scrub_ids}


def _generate(store: Store, course_id: int, provider, kind: str, title: str, task: str, sources: list[dict[str, Any]],
              *, question: str = "", event_id: int | None = None, data: dict[str, Any] | None = None,
              max_tokens: int = 4000, extra_note: str = "") -> int:
    r = _compose(store, course_id, provider, task, sources, question=question, max_tokens=max_tokens)
    if r["routed"]:
        return r["routed"]
    body, blocks = r["body"], list(r["blocks"])
    extra_note = " ".join(x for x in (extra_note, r.get("scrub_note", "")) if x)
    if extra_note:
        body = f"> {extra_note}\n\n{body}"
        blocks.insert(0, {"kind": "text", "md": extra_note, "cites": []})
    body = f"{body}\n\n{disclaimer.FOOTER}"
    d = dict(data or {})
    d.update({"outside": r["outside"], "not_covered": r["not_covered"], "mode": "answer", "blocks": blocks,
              "scrubbed": r.get("scrubbed", 0)})
    if r.get("held"):
        d["held_back"] = True
    if r.get("scrub_items") and _scrub_offer(store, r):
        d["graded_offers"] = _scrub_offer(store, r)  # the button names the item the note names
    return store.add_guide(course_id, kind, title, body, sources=_guide_sources(r["sources"], r["used"]), data=d,
                           question=question, event_id=event_id, model=getattr(provider, "model", "") or "")


def _block_dicts(blocks) -> list[dict[str, Any]]:
    """Saved form of answer blocks (grounding.Block objects or dicts): [{kind, md, cites}]."""
    out = []
    for b in blocks or []:
        get = (lambda k, d=None: b.get(k, d)) if isinstance(b, dict) else (lambda k, d=None: getattr(b, k, d))
        out.append({"kind": get("kind", "text"), "md": get("md", ""), "cites": list(get("cites", []) or [])})
    return out


def _course_name(store: Store, course_id: int) -> str:
    c = store.course(course_id) or {}
    return " ".join(x for x in (c.get("code", ""), c.get("name", "")) if x) or "the course"


# ---------------------------------------------------------------- JSON helpers
def _extract_json(raw: str, want=dict):
    """First JSON value of type `want` in the text. Returns (value, text_without_json) or (None, raw)."""
    text = raw or ""
    opener = "{" if want is dict else "["
    dec = json.JSONDecoder()
    for m in re.finditer(re.escape(opener), text):
        try:
            val, end = dec.raw_decode(text, m.start())
        except ValueError:
            continue
        if isinstance(val, want):
            rest = (text[:m.start()] + text[end:]).replace("```json", "").replace("```", "").strip()
            return val, rest
    return None, text


def _ask_json(provider, system: str, user: str, want, max_tokens: int = 4000):
    """Ask, parse JSON of type `want` (dict or list); retry once with a correction message."""
    msgs = [{"role": "user", "content": user}]
    raw = provider.complete(system, msgs, max_tokens=max_tokens)
    val, rest = _extract_json(raw, want)
    if isinstance(val, want):
        return val, rest
    msgs += [{"role": "assistant", "content": raw},
             {"role": "user", "content": "That was not valid JSON in the shape requested. Reply again with the JSON "
                                         "only (no code fences, no commentary), exactly in the shape described."}]
    raw = provider.complete(system, msgs, max_tokens=max_tokens)
    val, rest = _extract_json(raw, want)
    if isinstance(val, want):
        return val, rest
    raise ValueError("The AI did not return usable JSON. Try again.")


# ---------------------------------------------------------------- events and lectures
_DAYS = r"(?:mon|tue|wed|thu|fri|sat|sun)[a-z]*"
_DANGLING = (" on", " to", " of", " for", " the", " and", " in", " at", " by")


def is_junk_event(e: dict[str, Any]) -> bool:
    """Rows a syllabus import leaves behind: 'to', 'Wed Sep 23, Fri Sep 25, Mon', a sentence, 'There is no class on'."""
    t = (e.get("title") or "").strip()
    if len(re.sub(r"[^A-Za-z0-9]", "", t)) < 3:
        return True
    low = t.lower().strip(" .:;,-|")
    if low in {"to", "and", "or", "the", "tbd", "tba", "no class", "none", "n/a"}:
        return True
    if re.match(rf"^{_DAYS}\b[^,]*,\s*{_DAYS}\b", low) or re.match(r"^there is no\b", low):
        return True
    if len(t.split()) > 14 or low.endswith(_DANGLING):
        return True
    return False


def clean_title(e: dict[str, Any]) -> str:
    """'17 | Lecture 20' -> 'Lecture 20'."""
    t = re.sub(r"^\s*\d+\s*\|\s*", "", (e.get("title") or "").strip())
    return t.strip()


def event_label(e: dict[str, Any]) -> str:
    """What a picker shows: the topic when there is one, otherwise the cleaned title."""
    title, topics = clean_title(e), (e.get("topics") or "").strip()
    if topics and (is_junk_event(dict(e, title=title)) or topics.lower() not in title.lower()):
        short = topics if len(topics) <= 60 else topics[:57].rstrip() + "..."
        if is_junk_event(dict(e, title=title)):
            return short
        return f"{title}: {short}"
    return title


def study_form_options(store: Store, course_id: int) -> dict[str, Any]:
    """Choices for the Study tab: lectures (next and past), exams, and materials to pick from."""
    today = _today()
    evs = [e for e in store.events(course_id) if not is_junk_event(e)]

    def row(e):
        return {"id": e["id"], "date": e["date"], "label": event_label(e), "kind": e["kind"]}

    return {"next_lectures": [row(e) for e in evs if e["kind"] == "lecture" and e["date"] >= today],
            "past_lectures": [row(e) for e in evs if e["kind"] == "lecture" and e["date"] <= today][-20:],
            "exams": [row(e) for e in evs if e["kind"] in ("exam", "quiz")],
            "materials": [{"id": m["id"], "title": m["title"], "kind": m["kind"]} for m in store.materials(course_id)
                          if not m.get("duplicate_of") and not m.get("graded") and m["kind"] not in ("syllabus", "past_offering")]}


def _pick_event(store: Store, course_id: int, event_id: int | None, kind: str) -> dict[str, Any]:
    if event_id is not None:
        e = store.event(event_id)
        if not e or e["course_id"] != course_id:
            raise ValueError(f"No such event {event_id} in this course.")
        return e
    nxt = [e for e in store.events(course_id, start=_today(), kinds=[kind]) if not is_junk_event(e)]
    if not nxt:
        raise ValueError(f"No upcoming {kind} on this course's calendar. Add one first.")
    return nxt[0]


_GENERIC = {"lecture", "lectures", "week", "class", "slides", "notes", "introduction", "intro", "chapter", "part",
            "review", "overview", "captions", "recording", "video", "transcript"}


def _lecture_number(ev: dict[str, Any]) -> str | None:
    for text in (clean_title(ev), ev.get("topics") or ""):
        m = re.search(r"(?:lecture|lec\.?|week|class|session)\s*#?\s*0*(\d+)", text, re.I)
        if m:
            return m.group(1)
    return None


def _name_has_number(m: dict[str, Any], num: str) -> bool:
    names = " ".join(str(m.get(k, "")) for k in ("title", "original_name", "filename")).lower()
    return bool(re.search(rf"(?<![a-z0-9])(?:lecture|lec|l|week|wk|class|session)[\s_\-#.]*0*{num}(?!\d)", names))


_MONTH_NUM = {n: i for i, names in enumerate(
    ("jan january", "feb february", "mar march", "apr april", "may", "jun june", "jul july", "aug august",
     "sep sept september", "oct october", "nov november", "dec december"), 1) for n in names.split()}
_NAME_MD = re.compile(r"(?<![a-z0-9])([a-z]{3,9})\.?[\s_\-]*(\d{1,2})(?:st|nd|rd|th)?(?![0-9])", re.I)
_NUM_MD = re.compile(r"(?<![0-9])(?:(20\d{2})[-/.])?(\d{1,2})[-/.](\d{1,2})(?![0-9])")


def _name_dates(m: dict[str, Any]) -> set[tuple[int | None, int, int]]:
    """(year or None, month, day) for every date written in a material's title or file name: 'Oct 6', 'October 6th',
    '10/6', '2026-10-06'. A bare number after a month word counts only when the word is a month ('Oct 6 lecture')."""
    names = " ".join(str(m.get(k, "")) for k in ("title", "original_name", "filename"))
    out: set[tuple[int | None, int, int]] = set()
    for a in _NAME_MD.finditer(names):
        mon = _MONTH_NUM.get(a.group(1).lower())
        if mon and 1 <= int(a.group(2)) <= 31:
            out.add((None, mon, int(a.group(2))))
    for a in _NUM_MD.finditer(names):
        mon, day = int(a.group(2)), int(a.group(3))
        if a.group(1) and 1 <= mon <= 12 and 1 <= day <= 31:
            out.add((int(a.group(1)), mon, day))
        elif not a.group(1) and 1 <= mon <= 12 and 1 <= day <= 31:
            out.add((None, mon, day))
    return out


def _name_has_date(m: dict[str, Any], ev: dict[str, Any]) -> bool:
    """The material's name carries the lecture's own month and day (and year, when it writes one)."""
    d = str(ev.get("date") or "")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", d):
        return False
    y, mo, da = int(d[:4]), int(d[5:7]), int(d[8:10])
    return any(mm == mo and dd == da and yy in (None, y) for yy, mm, dd in _name_dates(m))


def _topic_words(ev: dict[str, Any]) -> set[str]:
    text = f"{clean_title(ev)} {ev.get('topics') or ''}".lower()
    return {w for w in re.findall(r"[a-z]{4,}", text) if w not in _GENERIC}


def _name_tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]{2,}", re.sub(r"\.[a-z0-9]{2,4}$", "", (text or "").lower().replace("_", " ")))
            if w not in _GENERIC and not w.isdigit()}


def _words_match(a: str, b: str) -> bool:
    """'intro' and 'introduction', 'transpose' and 'transposes': the same word, one a prefix or plural of the other."""
    if a == b:
        return True
    short, long = sorted((a, b), key=len)
    return len(short) >= 5 and long.startswith(short) and len(long) - len(short) <= 8


def _common_tokens(mats: list[dict[str, Any]]) -> frozenset[str]:
    """Words most of the course's files share (the course code, 'chem', 'lab'): they say nothing about one lecture."""
    n = len(mats)
    counts: dict[str, int] = {}
    for m in mats:
        for w in _name_tokens(m.get("title", "")) | _name_tokens(str(m.get("original_name") or "")):
            counts[w] = counts.get(w, 0) + 1
    return frozenset(w for w, c in counts.items() if n >= 4 and c >= max(3, 0.3 * n))


def _title_score(m: dict[str, Any], ev: dict[str, Any], common: frozenset[str] = frozenset()) -> int:
    """How many words of the lecture's title and topics the material's title or file name carries."""
    mat = (_name_tokens(f"{m.get('title', '')}") | _name_tokens(str(m.get("original_name") or ""))) - common
    lec = {w for w in re.findall(r"[a-z0-9]{2,}", f"{clean_title(ev)} {ev.get('topics') or ''}".lower())
           if w not in _GENERIC and not w.isdigit() and w not in _FILLER_WORDS and w not in {"strang", "lecture"}}
    return sum(1 for w in mat if any(_words_match(w, x) for x in lec))


_NOTE_KINDS = ("notes", "lecture_notes", "slides", "recording", "handout", "other", "reading")


def link_suggestions(store: Store, course_id: int, events: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Saved notes that are not linked to any lecture but whose title or file name looks like one:
    [{material, event, score, why}]. A material is suggested for its best lecture only, and not when two tie."""
    evs = events if events is not None else [e for e in store.events(course_id, kinds=["lecture"]) if not is_junk_event(e)]
    out = []
    common = _common_tokens([m for m in store.materials(course_id) if _usable(m)])
    for m in store.materials(course_id):
        if not _usable(m) or m["kind"] not in _NOTE_KINDS or m.get("event_id") or m.get("lecture_date"):
            continue
        if _is_readme(m):
            continue

        def score(e):  # a lecture number in the name is a strong match; otherwise 2 or more shared content words
            num = _lecture_number(e)
            if num is not None and _name_has_number(m, num):
                return 100
            if _name_has_date(m, e):
                return 90
            sc = _title_score(m, e, common)
            return sc if sc >= 2 else 0

        ranked = sorted(((score(e), e) for e in evs), key=lambda x: -x[0])
        if not ranked or ranked[0][0] < 1 or (len(ranked) > 1 and ranked[1][0] == ranked[0][0]):
            continue
        sc, e = ranked[0]
        out.append({"material": m, "event": e, "score": sc,
                    "why": f"'{m['title']}' looks like {event_label(e)}"})
    return out


def _is_readme(m: dict[str, Any]) -> bool:
    """A README is a file listing, not notes for one lecture: never suggested or matched."""
    name = re.sub(r"\.[a-z0-9]{2,4}$", "", str(m.get("original_name") or m.get("filename") or "").lower().rsplit("/", 1)[-1])
    return name.strip() in ("readme", "read me", "read_me", "read-me")


def _is_recording(m: dict[str, Any]) -> bool:
    names = " ".join(str(m.get(k, "")) for k in ("title", "original_name")).lower()
    return m["kind"] == "recording" or "caption" in names or "transcript" in names


def lecture_materials(store: Store, course_id: int, ev: dict[str, Any]) -> list[tuple[dict[str, Any], str]]:
    """Saved materials for one lecture, with why each was chosen.

    Order: linked to the event, then dated the same day, then matched by lecture number, title or topic. A recording
    or captions file that matches is always included."""
    mats = [m for m in store.materials(course_id) if _usable(m)]
    common = _common_tokens(mats)
    out: list[tuple[dict[str, Any], str]] = []
    seen: set[int] = set()

    def add(m, why):
        if m["id"] not in seen:
            seen.add(m["id"])
            out.append((m, why))

    for m in mats:
        if m.get("event_id") == ev["id"]:
            add(m, "linked to this lecture")
    for m in mats:
        if m.get("lecture_date") and m["lecture_date"] == ev["date"]:
            add(m, f"dated {ev['date']}")
    num, words = _lecture_number(ev), _topic_words(ev)
    for m in mats:
        if m["id"] in seen or (out and not _is_recording(m)) or _is_readme(m):
            continue
        if m.get("event_id") or (m.get("lecture_date") and m["lecture_date"] != ev["date"]):
            continue  # belongs to some other lecture
        if num is not None and _name_has_number(m, num):
            add(m, "matched by lecture number in its name")
            continue
        if _name_has_date(m, ev):
            add(m, "matched by date in its name")
            continue
        title_words = {w for w in re.findall(r"[a-z]{4,}", f"{m['title']} {m.get('original_name', '')}".lower())
                       if w not in _GENERIC}
        if num is None or not re.search(r"(?:lecture|lec|week)[\s_\-#.]*\d+", m["title"].lower()):
            if len(words & title_words) >= 2 or (len(words) == 1 and words <= title_words and len(next(iter(words))) >= 6) \
                    or _title_score(m, ev, common) >= 2:
                add(m, "matched by title")
    return out


def _syllabus_rows(store: Store, course_id: int, ev: dict[str, Any], query: str, n: int = 3) -> list[dict[str, Any]]:
    """The syllabus passages that are about this lecture (its row, its reading), not the catalog blurb or course policies."""
    num, words = _lecture_number(ev), _topic_words(ev)
    out = []
    for x in _search(store, course_id, query, 6, kinds=["syllabus"]):
        if _rfn("is_header_chunk", x["text"], x["material_title"]) or _rfn("is_title_only", x["text"], x.get("heading", "")):
            continue
        text = x["text"].lower()
        schedule = bool(re.search(r"schedule|reading|topics|calendar|outline", x.get("heading", ""), re.I))
        about = (num is not None and re.search(rf"(?:lecture|lec|week)[\s_\-#.]*0*{num}\b", text)) or \
                ev["date"] in text or (schedule and len(words & set(re.findall(r"[a-z]{4,}", text))) >= 2)
        if about:
            out.append(x)
    return out[:n]


def _lecture_sources(store: Store, course_id: int, ev: dict[str, Any], cap: int = 20, *, primer: bool = False):
    """(sources, used materials [(title, why)], note) for a lecture."""
    found = lecture_materials(store, course_id, ev)
    recs = [m for m, _ in found if _is_recording(m)]
    others = [m for m, _ in found if not _is_recording(m)]
    ordered = [m["id"] for m in recs + others]
    own = _spread_materials(store, ordered, cap=max(4, cap - 8), per=8) if ordered else []
    query = " ".join(x for x in (clean_title(ev), ev.get("topics", "")) if x)
    if primer:  # a primer is built before the lecture: the schedule's row and reading, then notes about the same topic
        syl = _syllabus_rows(store, course_id, ev, query)
        topic_q = " ".join(x for x in (re.sub(r"^\s*(?:lecture|lec\.?|week|class|session)\s*#?\d+\s*[:.\-]*\s*", "", clean_title(ev), flags=re.I),
                                       re.sub(r"\b[A-Z][a-z]+ \d+(?:\.\d+)?(?:[-,]\s*\d+(?:\.\d+)?)*", "", ev.get("topics", ""))) if x) or query
        related = [x for x in _search(store, course_id, topic_q, 14, exclude_graded=True)
                   if x["material_kind"] not in ("past_offering", "assignment", "syllabus")
                   and not _rfn("is_scaffold_chunk", x["text"], x.get("heading", ""))]
        # padding: a passage that shares only 'matrix' or 'equations' with the topic says nothing about it
        tell_fn = getattr(_mod("retrieval"), "telling_terms", None)
        tell = tell_fn(store, course_id, topic_q) if tell_fn else []
        if tell:
            related = [x for x in related if _mod("retrieval").mentions_any(x, tell)]
        if len(own) >= 4:  # the lecture's own notes carry the primer: a few related passages, not a second lecture
            related = related[:3]
        sources = _merge(own, syl, related, cap=cap)
    else:
        syl = _search(store, course_id, query, 3, kinds=["syllabus"])
        found_hits = _search(store, course_id, query, 10, exclude_graded=True)
        sources = _merge(own, found_hits, syl, cap=cap)
    used = [(m["title"], why) for m, why in found]
    if used:
        note = "Used for this lecture: " + "; ".join(f"{t} ({w})" for t, w in used) + "."
    elif primer:
        note = ("Nothing is linked to this lecture yet, so this primer is built from the syllabus reading for it and "
                "your related notes. Link its notes or slides to the lecture in Materials for a closer match.")
    else:
        note = ("No saved material is linked to this lecture, so this uses a topic search of everything you saved. "
                "Link a material to the lecture in Materials for a better result.")
    return sources, used, note


# ---------------------------------------------------------------- primer
def primer(store: Store, course_id: int, provider, *, event_id: int | None = None) -> int:
    _need(provider)
    ev = _pick_event(store, course_id, event_id, "lecture")
    sources, used, note = _lecture_sources(store, course_id, ev, primer=True)
    title = clean_title(ev) or ev["title"]
    task = (f"Write a short pre-lecture primer for {_course_name(store, course_id)}, lecture \"{title}\" on "
            f"{ev['date']}. Topics and reading listed: {ev.get('topics') or 'not listed'}. Use these sections: ## The big picture "
            "(a short, broad overview), ## Key terms (each with a one-line definition), ## What to watch for in lecture, "
            "## 3 questions to keep in mind. Keep it to what a student can read in five minutes. Stay broad: this is "
            "before the lecture, so do not go deep."
            + ("" if used else " No notes or slides are linked to this lecture, so build the primer from the syllabus "
               "reading and the related notes in the sources, call it a primer built from the schedule, and say plainly "
               "what the sources do not cover."))
    return _generate(store, course_id, provider, "primer", f"Primer: {title}", task, sources,
                     question=title, event_id=ev["id"], max_tokens=3000, extra_note="" if used else note,
                     data={"used_materials": [{"title": t, "why": w} for t, w in used]})


# ---------------------------------------------------------------- exam guide
def _window(store: Store, course_id: int, ev: dict[str, Any]):
    exams = [e for e in store.events(course_id, kinds=["exam"]) if e["date"] < ev["date"] and not is_junk_event(e)]
    prev = exams[-1]["date"] if exams else ""
    lectures = [e for e in store.events(course_id, kinds=["lecture"])
                if not is_junk_event(e) and e["date"] <= ev["date"] and (not prev or e["date"] > prev)]
    return prev, lectures


_LOGISTICS = re.compile(r"\b(room|building|hall|location|meets?|bring|goggles|safety|due at|submit|report due|"
                        r"pre-?lab|lab (?:date|section)|schedule)\b", re.I)
_LEAD_WEAK = re.compile(r"^\s*(\[(?:image|figure)|\||>|check your learning|solution\b|example\b|\*\*example)", re.I)


def _is_logistics(c: dict[str, Any]) -> bool:
    """A lab's date and room line, a list of due dates: not something to study."""
    text = (c.get("text") or "").strip()
    return len(text) < 260 and bool(_LOGISTICS.search(text)) and bool(_rfn("has_date", text) or re.search(r"\broom\b", text, re.I))


def _usable_chunks(store: Store, m: dict[str, Any]) -> list[dict[str, Any]]:
    return [c for c in store.chunks_of(m["id"]) if not _is_noise(c["text"])
            and not _rfn("is_scaffold_chunk", c["text"], c.get("heading", ""))
            and not _rfn("is_title_only", c["text"], c.get("heading", ""))
            and not _rfn("is_header_chunk", c["text"], m.get("title", ""))
            and not _is_logistics(c) and not re.match(r"\s*\*\*Problem\.?\*\*", c["text"])]


def _stem_set(text: str) -> set[str]:
    return {w[:6] for w in re.findall(r"[a-z]{4,}", (text or "").lower()) if w not in _GENERIC and w not in _FILLER_WORDS}


_STEM_ENDS = ("ies", "ing", "ed", "es", "s")


def _whole_stem(word: str) -> str:
    """A word cut to its stem by plain suffixes only ('quantum' and 'quantity' stay different words; 'eigenvalues' and
    'eigenvalue' meet)."""
    w = word.lower()
    for end in _STEM_ENDS:
        if w.endswith(end) and len(w) - len(end) >= 4:
            w = w[: -len(end)] + ("y" if end == "ies" else "")
            break
    return w[:-1] if w.endswith("e") and len(w) > 4 else w  # 'space' and 'spaces' meet


def _same_word(a: str, b: str) -> bool:
    """The same word or a plain inflection of it: equal stems, or one stem is the other with up to three more letters
    ('empirical' and 'empirically'). Never a shared five-letter start ('quantum' and 'quantity')."""
    a, b = _whole_stem(a), _whole_stem(b)
    if a == b:
        return True
    lo, hi = sorted((a, b), key=len)
    return len(lo) >= 5 and hi.startswith(lo) and len(hi) - len(lo) <= 3


def _content_words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z]{4,}", (text or "").lower()) if w not in _GENERIC and w not in _FILLER_WORDS]


def _topic_in_text(topic: str, text: str) -> bool:
    """Every content word of the typed topic appears in the text as a whole word (or a plain inflection of it)."""
    want = _content_words(topic)
    have = set(_content_words(text))
    return bool(want) and all(any(_same_word(w, h) for h in have) for w in want)


def _best_in_section(group: list[dict[str, Any]], heading: str) -> dict[str, Any]:
    """The passage that opens a topic: prefers one that defines (bold term, 'is', 'defined') over a figure, a table
    or a worked solution."""
    hw = _stem_set(heading)
    if re.match(r"\s*key equations\s*$", heading, re.I):
        return dict(group[0], _defn=0.5)  # what follows a Key Equations table is often the exercises that come next

    def score(c: dict[str, Any]) -> float:
        t = c["text"]
        sc = 0.0
        sc += 2.0 if "**" in t[:600] else 0.0
        sc += 1.0 if re.search(r"\b(?:is|are) (?:defined|called|the|a|an)\b|\bdefin(?:ed|ition)\b|\brefers? to\b", t[:500], re.I) else 0.0
        low = t.lower()
        sc += min(2.0, sum(low.count(w) for w in re.findall(r"[a-z]{5,}", heading.lower())) / 3)  # a passage about its heading
        sc += 0.5 * len(hw & _stem_set(t[:500]))
        sc -= 3.0 if _LEAD_WEAK.match(t) else 0.0
        return sc
    best = group[0]
    for c in group[1:4]:  # the opening few passages of the section only
        if score(c) > score(best) + 0.4:
            best = c
    best = dict(best)
    best["_defn"] = score(best)
    return best


def _topic_passages(store: Store, mats: list[dict[str, Any]], topic_text: str, cap: int, course_topics: str = "") -> list[dict[str, Any]]:
    """One passage per topic heading of each material, in reading order. Headings that the exam's lectures name come
    first when there are more headings than slots. Lead-in stubs, chapter intros, end-of-chapter exercises, glossaries
    and summaries are left out."""
    want, want2 = _stem_set(topic_text), _stem_set(course_topics)
    per_mat: list[tuple[dict[str, Any], list[tuple[int, dict[str, Any]]]]] = []
    plain: list[dict[str, Any]] = []
    for m in mats:
        chunks = _usable_chunks(store, m)
        groups: dict[str, list[dict[str, Any]]] = {}
        for c in chunks:
            groups.setdefault(c.get("heading") or "", []).append(c)
        if len([h for h in groups if h]) < 4:
            plain.append(m)
            continue
        picks = []
        for i, (h, g) in enumerate(groups.items()):
            if re.match(r"\s*key equations\s*$", h, re.I) and not g[0]["text"].lstrip().startswith("|"):
                continue
            hs = _stem_set(h)
            if sum(len(c["text"]) for c in g) < 600 and not (hs & want):
                continue  # a lead-in stub that only introduces the subsections below it
            hit = 3 * len(hs & want) + 2 * len(hs & want2)  # this exam's lectures first, then the course's topic list
            picks.append((i, hit, _best_in_section(g, h), m, sum(len(c["text"]) for c in g)))
        per_mat.append((m, picks))
    total = sum(len(p) for _, p in per_mat) + len(plain) * 3
    lists = []
    for m, picks in per_mat:
        room = max(2, round(cap * len(picks) / max(1, total))) if total > cap else len(picks)
        if len(picks) > room:  # more topics than slots: the exam's own topics first, then the meatiest sections
            keep = sorted(picks, key=lambda p: (-p[1], -p[2].get("_defn", 0), -p[4]))[:room]
            picks = sorted(keep, key=lambda p: p[0])
        lists.append([_source_from_chunk(c, m["title"], m["kind"]) for _, _, c, m, _n in picks])
    share = max(3, (cap - sum(len(x) for x in lists)) // max(1, len(plain))) if plain else 0
    for m in plain:
        chunks = _usable_chunks(store, m)
        lists.append([_source_from_chunk(c, m["title"], m["kind"]) for c in _spread_pick(chunks, share)])
    return _interleave(lists, cap=cap)


def _relevant_materials(store: Store, mats: list[dict[str, Any]], topic_text: str) -> list[dict[str, Any]]:
    """With nothing linked or chosen, the materials whose title or headings name the exam's topics; the opening
    chapter and the rest only when nothing does."""
    want = _stem_set(topic_text)
    if not want:
        return mats
    scored = []
    for m in mats:
        words = _stem_set(m["title"]) | _stem_set(" ".join(c.get("heading", "") for c in store.chunks_of(m["id"])))
        scored.append((len(words & want), m))
    best = max((n for n, _ in scored), default=0)
    hit = [m for n, m in scored if n >= max(2, 0.5 * best)]  # one shared word ('chemical') is not a match
    rest = [m for m in mats if _rfn("title_doc_id", m["title"]) != ("chapter", 1)]  # the opening chapter is filler here
    hit = [m for m in hit if m in rest]
    return hit or rest or mats


def _material_text(store: Store, m: dict[str, Any]) -> str:
    return " ".join([m.get("title") or "", *(f"{c.get('heading', '')} {c['text']}" for c in store.chunks_of(m["id"]))])


def _unmatched_topics(store: Store, mats: list[dict[str, Any]], topics: list[str], skip_text: str = "") -> list[str]:
    """The typed topics that none of these materials (title, headings and passages) covers by whole words. A topic a
    lecture without materials already names is said by that note, so it is skipped here."""
    texts = [_material_text(store, m) for m in mats]
    out = []
    for t in topics:
        if any(_topic_in_text(t, x) for x in texts) or (skip_text and _topic_in_text(t, skip_text)):
            continue
        out.append(t)
    return out


def _materials_per_topic(store: Store, mats: list[dict[str, Any]], topics: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    """Each typed topic keeps its own best-matching materials, scored against that topic alone, so a chapter that
    matches another topic more strongly never pushes it out. Returns the materials in course order and the topics
    that matched nothing."""
    words = {m["id"]: _stem_set(m["title"]) | _stem_set(" ".join(c.get("heading", "") for c in store.chunks_of(m["id"])))
             for m in mats}
    rest = [m for m in mats if _rfn("title_doc_id", m["title"]) != ("chapter", 1)]  # the opening chapter is filler here
    keep: dict[int, dict[str, Any]] = {}
    unmatched: list[str] = []
    for t in topics:
        want = _stem_set(t)
        need = min(2, len(want))  # one shared word of a longer topic ('chemical') is not a match
        scored = [(len(words[m["id"]] & want), m) for m in rest]
        best = max((n for n, _ in scored), default=0)
        got = [m for n, m in scored if n >= max(need, 1) and n >= 0.5 * best] if want else []
        if want and not got:  # no title or heading names it: a material whose passages cover all its words does
            got = [m for m in rest if _topic_in_text(t, _material_text(store, m))]  # whole words, never a five-letter start
        if not got:
            unmatched.append(t)
        for m in got:
            keep[m["id"]] = m
    return [m for m in mats if m["id"] in keep], unmatched


def _exam_sections(math: bool, plan: str) -> str:
    """The section list of the exam guide prompt. Courses with math get formulas and new numbers; readings and
    arguments get key terms, claims and short-answer or essay prompts."""
    if math:
        return ("Sections: ## What's covered (by topic, with citations), ## Key formulas and definitions, ## Common "
                "mistakes, ## Practice set (4 to 6 short questions; put each answer in a block with <details> on its own "
                "line, then <summary>Answer</summary> on the next line, the answer, then </details> on its own line; make "
                "up new numbers and never copy a graded problem; if a graded problem would be the natural question, write "
                f"a twin with different numbers instead), {plan}.")
    return ("Sections: ## What's covered (by topic, with citations), ## Key terms, people and events (with citations), "
            "## Main arguments and how they connect, ## Common mistakes, ## Practice set (4 to 6 short-answer or "
            "essay-planning questions about the readings; put each model answer in a block with <details> on its own "
            "line, then <summary>Answer</summary> on the next line, the answer built from the sources with citations, "
            "then </details> on its own line; never copy a graded assignment or write the thesis or any paragraph of "
            f"graded writing; if a graded assignment would be the natural question, ask a different question on the same "
            f"reading instead), {plan}.")


def exam_guide(store: Store, course_id: int, provider, *, event_id: int | None = None,
               material_ids: list[int] | None = None) -> int:
    """Study guide for one exam. Topics come from the saved materials and syllabus rows between the previous exam and
    this one. Lectures in that window with no saved materials are named. `material_ids` limits it to chosen materials."""
    _need(provider)
    ev = _pick_event(store, course_id, event_id, "exam")
    prev, lectures = _window(store, course_id, ev)
    # what the student typed as the exam's own topics ("Covers") decides the scope: lectures in the window that do not
    # touch those topics are not part of this exam, so they are neither used nor warned about
    typed = " ".join(str(ev.get("topics") or "").split())
    typed_stems = _stem_set(typed)
    typed_parts = [t.strip(" .") for t in re.split(r"[,;\n]", typed) if t.strip(" .")] if typed else []
    if typed_stems:
        lectures = [l for l in lectures if _stem_set(f"{clean_title(l)} {l.get('topics', '')}") & typed_stems]
    chosen: list[dict[str, Any]] = []
    if material_ids:
        for mid in material_ids:
            m = store.material(int(mid))
            if m and m["course_id"] == course_id:
                chosen.append(m)
        if not chosen:
            raise ValueError("None of the chosen materials are in this course.")
    missing: list[str] = []
    in_window: dict[int, dict[str, Any]] = {}
    for lec in lectures:
        found = lecture_materials(store, course_id, lec)
        if not found:
            missing.append(f"{event_label(lec)} ({lec['date']})")
        for m, _ in found:
            in_window[m["id"]] = m
    for m in store.materials(course_id):  # dated materials in the window that no lecture event claims
        d = m.get("lecture_date") or ""
        if _usable(m) and d and d <= ev["date"] and (not prev or d > prev):
            in_window.setdefault(m["id"], m)
    suggestions = [x for x in link_suggestions(store, course_id, lectures) if x["material"]["id"] not in in_window]
    link_note = ""
    if suggestions and not chosen:  # unlinked notes whose title names a lecture of the window: used, and said so
        for x in suggestions:
            in_window.setdefault(x["material"]["id"], x["material"])
        gone = {x["event"]["id"] for x in suggestions}
        missing = [f"{event_label(l)} ({l['date']})" for l in lectures
                   if l["id"] not in gone and not lecture_materials(store, course_id, l)]
        more = len(suggestions) - 4
        link_note = ("Matched by title, not linked: " + "; ".join(x["why"] for x in suggestions[:4]) +
                     (f"; and {more} more" if more > 0 else "") + ". Link them to their lectures in Materials to make this certain.")
    topics = [" ".join(x for x in (clean_title(e), e.get("topics", "")) if x) for e in lectures]
    topics_text = [t for t in topics if t]
    if typed_parts:
        topics_text = typed_parts + [t for t in topics_text if not (_stem_set(t) <= typed_stems)]
    topics_text = topics_text or [" ".join(x for x in (ev["title"], ev.get("topics", "")) if x)]
    scope_note = ""
    no_match: list[str] = []
    graded_ids = {i.get("material_id") for i in store.graded_items(course_id) if i.get("material_id")}
    graded_chosen = [m for m in chosen if m.get("graded") or m["id"] in graded_ids]
    if chosen:
        mats = chosen
        scope_note = "Limited to the materials you chose."
        if graded_chosen:  # graded work is never a source, so a ticked handout adds nothing: say so
            names = ", ".join(m["title"] for m in graded_chosen[:3])
            scope_note += (f" {names} {'is' if len(graded_chosen) == 1 else 'are'} graded work, so "
                           f"{'it adds' if len(graded_chosen) == 1 else 'they add'} nothing to this guide and "
                           f"{'was' if len(graded_chosen) == 1 else 'were'} not used.")
            mats = [m for m in chosen if m not in graded_chosen] or chosen
    elif in_window:
        mats = list(in_window.values())
        scope_note = f"Built from the materials saved for {'the lectures since ' + prev if prev else 'the course so far'}."
    else:
        usable = [m for m in store.materials(course_id) if _usable(m)]
        if typed_parts:
            mats, unmatched = _materials_per_topic(store, usable, typed_parts)
            mats = mats or _relevant_materials(store, usable, " ".join(topics_text))
            no_match = unmatched
            found = [t for t in typed_parts if t not in unmatched]
            scope_note = (("Built from the saved materials that match the topics you typed for this exam: " +
                           ", ".join(found) + ". ") if found else "") + \
                         ("Nothing in your saved materials matched: " + ", ".join(unmatched) + "." if unmatched else "")
        else:
            mats = _relevant_materials(store, usable, " ".join(topics_text))
        scope_note = (scope_note if typed_parts else
                      "No material is linked to the lectures before this exam, so this covers all your saved materials.")
    if typed_parts and (chosen or in_window):  # every branch names a typed topic that nothing it uses covers
        gone = _unmatched_topics(store, mats, typed_parts, " ".join(f"{clean_title(l)} {l.get('topics', '')}" for l in lectures
                                                                     if f"{event_label(l)} ({l['date']})" in missing))
        no_match = gone
        if gone:
            scope_note = (scope_note + " " if scope_note else "") + \
                f"Nothing in the materials used for this guide matched: {', '.join(gone)}."
    syllabus_rows = [f"{e['date']}: {event_label(e)}" for e in lectures if e.get("topics")]
    course_topics = " ".join(f"{clean_title(e)} {e.get('topics', '')}" for e in store.events(course_id, kinds=["lecture"]))
    mids = [m["id"] for m in mats]
    sections = _topic_passages(store, mats, " ".join(topics_text), MAX_SOURCES - 2, course_topics) if mats else []
    per_topic = [[x for x in _search(store, course_id, t, 5, exclude_graded=True, **({"material_ids": mids} if mids else {}))
                  if not _rfn("is_scaffold_chunk", x["text"], x.get("heading", "")) and not _is_logistics(x)][:3]
                 for t in [*topics_text, *[m["title"] for m in mats]]]
    # the syllabus adds this exam's own row (what it covers, when), never grading or catalog text, and not when the
    # student limited the guide to chosen materials
    syl = []
    if not chosen:
        toks = re.findall(r"[a-z0-9]+", ev["title"].lower())
        syl = [x for x in _search(store, course_id, " ".join([ev["title"], *topics_text[:6]]), 4, kinds=["syllabus"])
               if toks and all(re.search(rf"\b{re.escape(t)}\b", (x["text"] + " " + x.get("heading", "")).lower()) for t in toks) and not _rfn("is_header_chunk", x["text"], x["material_title"])
               and not re.match(r"^\W*(grading|catalog|course description|academic integrity)", x.get("heading", ""), re.I)][:2]
    have = {(x["material_id"], x.get("heading", "")) for x in sections if x.get("heading")}
    fill = [x for lst in per_topic for x in lst if (x["material_id"], x.get("heading", "")) not in have]
    sources = _merge(sections, _interleave([fill[i::1] for i in range(1)], cap=MAX_SOURCES - 2), syl, cap=MAX_SOURCES)
    today = _today()
    past = False
    try:
        delta = (_date.fromisoformat(ev["date"]) - _date.fromisoformat(today)).days
        past, days = delta < 0, max(0, delta)
    except ValueError:
        days = 0
    plan_days = [(_date.fromisoformat(today) + timedelta(days=i)).isoformat() for i in range(min(days, 14) + 1)] \
        if re.match(r"\d{4}-\d{2}-\d{2}$", ev["date"]) and not past else []
    graded = _mod("integrity").match_graded(store, course_id, " ".join(topics_text))
    math = has_math(store, course_id)
    notes = [scope_note, link_note]
    if past:
        notes.append(f"This exam was on {ev['date']}, so this is a review of what it covered.")
    if missing:
        notes.append("No saved materials for: " + "; ".join(missing) + ". Add them to cover those topics.")
    if graded:
        notes.append(("Practice questions below are original twins with different numbers. "
                      if math or (graded.get("kind") or "problem") != "essay" else
                      "Practice questions below are original, not copies of your graded work. ") +
                     "Questions you marked as graded are not solved here.")
    when = (f"this exam was on {ev['date']}, so it is already over (today is {today})" if past
            else f"{days} days from today, {today}")
    plan = ("## Review plan (the exam is over: three short bullets on what to revisit; no day-by-day schedule)" if past else
            "## Study plan (one line per day from these dates: " + f"{', '.join(plan_days) or ev['date']}; spread the topics, put review and rest near the end)")
    task = (f"Write an exam study guide for {_course_name(store, course_id)}: \"{ev['title']}\" on {ev['date']} "
            f"({when}). Window: {'since the previous exam on ' + prev if prev else 'the start of the course'}.\n"
            f"Materials used: {', '.join(m['title'] for m in mats) or 'none'}.\n" +
            ("Topics this exam covers (the student's own list):\n- " if typed_parts else "Lecture topics in the window:\n- ") +
            "\n- ".join(topics_text) +
            ("\nSyllabus rows:\n- " + "\n- ".join(syllabus_rows) if syllabus_rows else "") +
            (("\nThese typed topics matched NOTHING in the saved materials, so say plainly that the guide cannot cover "
              "them and do not invent their content:\n- " + "\n- ".join(no_match)) if no_match else "") +
            (("\nThese lectures have NO saved materials, so say plainly that the guide cannot cover them and do not "
              "invent their content:\n- " + "\n- ".join(missing)) if missing else "") +
            "\n\n" + _exam_sections(math, plan))
    return _generate(store, course_id, provider, "exam_guide", f"Exam guide: {ev['title']}", task, sources,
                     question=ev["title"], event_id=ev["id"],
                     data={"topics": topics, "previous_exam": prev, "missing_lectures": missing, "past": past,
                           "link_suggestions": [{"material_id": x["material"]["id"], "event_id": x["event"]["id"],
                                                 "why": x["why"]} for x in suggestions],
                           "materials": [m["title"] for m in mats], "syllabus_rows": syllabus_rows},
                     max_tokens=5000, extra_note=" ".join(n for n in notes if n))


# ---------------------------------------------------------------- concept map
def _valid_cites(raw, n_sources: int) -> list[int]:
    out: list[int] = []
    for c in raw or []:
        try:
            ci = int(c)
        except (TypeError, ValueError):
            continue
        if 1 <= ci <= n_sources and ci not in out:
            out.append(ci)
    return out


def _dropped_cite(raw, n_sources: int) -> bool:
    """True when a raw cite list held a number that points at no source (it was dropped)."""
    for c in raw or []:
        try:
            ci = int(c)
        except (TypeError, ValueError):
            return True
        if not 1 <= ci <= n_sources:
            return True
    return False


def _clean_graph(graph: dict[str, Any], n_sources: int) -> dict[str, Any]:
    nodes, ids = [], set()
    for n in graph.get("nodes") or []:
        if not isinstance(n, dict) or not str(n.get("id", "")).strip() or not str(n.get("label", "")).strip():
            continue
        nid = str(n["id"]).strip()
        if nid in ids:
            continue
        ids.add(nid)
        node = {"id": nid, "label": str(n["label"]).strip(), "cites": _valid_cites(n.get("cites"), n_sources)}
        if _dropped_cite(n.get("cites"), n_sources):
            node["cites_removed"] = True
        nodes.append(node)
    edges = []
    for e in graph.get("edges") or []:
        if not isinstance(e, dict):
            continue
        a, b = str(e.get("a", "")).strip(), str(e.get("b", "")).strip()
        if a in ids and b in ids and a != b:
            edge = {"a": a, "b": b, "label": str(e.get("label", "")).strip(),
                    "cites": _valid_cites(e.get("cites"), n_sources)}  # out-of-range cites are dropped
            if _dropped_cite(e.get("cites"), n_sources):
                edge["cites_removed"] = True
            edges.append(edge)
    return {"nodes": nodes, "edges": edges}


_FILLER_WORDS = frozenset("""about after again also because before being between could does during each from have having
into more most other over same should since some such than that their them then there these they this those through
under until very were what when where which while will with would your""".split())


def _vocab(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{5,}", (text or "").lower()) if w not in _FILLER_WORDS}


def _scoped_sources(store: Store, pool: list[dict[str, Any]], scope: str, cap: int = MAX_SOURCES) -> list[dict[str, Any]]:
    """Sources for a map scoped to named documents ('Federalist No. 10 and 51 and Chapters 7 and 8'): every named
    document gets its share of the slots, taken from its body passages that share the most vocabulary with the other
    named documents (and with any topic words in the scope), so 'The status of women' does not crowd out Fed 51."""
    r = _mod("retrieval")
    if not hasattr(r, "query_doc_ids") or not hasattr(r, "is_scaffold_chunk"):
        return []
    ids = r.query_doc_ids(scope)
    named = [m for m in pool if r.title_doc_id(m["title"]) in ids] if ids else []
    if len(named) < 2:
        return []
    texts: dict[int, list[dict[str, Any]]] = {}
    for m in named:
        chunks = [c for c in store.chunks_of(m["id"]) if not _is_noise(c["text"])
                  and not _rfn("is_scaffold_chunk", c["text"], c.get("heading", ""))
                  and not _rfn("is_title_only", c["text"], c.get("heading", ""))
                  and not _rfn("is_header_chunk", c["text"], m.get("title", ""))]
        texts[m["id"]] = chunks or [c for c in store.chunks_of(m["id"]) if not _is_noise(c["text"])]
    df: dict[str, int] = {}
    for m in named:
        for w in set().union(*(_vocab(c["text"] + " " + c.get("heading", "")) for c in texts[m["id"]])) if texts[m["id"]] else set():
            df[w] = df.get(w, 0) + 1
    topic = _vocab(re.sub(r"\b(federalist|chapters?|papers?|no|and)\b|\d+", " ", scope.lower()))
    picks: dict[int, list[dict[str, Any]]] = {}
    for m in named:
        def score(c, m=m):
            words = _vocab(c["text"] + " " + c.get("heading", ""))
            shared = sum(1 for w in words if df.get(w, 0) >= 2)
            return (shared + 3 * len(words & topic)) / (len(words) ** 0.5 + 1)
        ranked = sorted(texts[m["id"]], key=score, reverse=True)
        picks[m["id"]] = ranked
    quota = {m["id"]: 0 for m in named}
    left = cap
    while left > 0 and any(quota[i] < len(picks[i]) for i in quota):  # one slot at a time, so a short paper gives its unused share away
        for i in quota:
            if left > 0 and quota[i] < len(picks[i]):
                quota[i] += 1
                left -= 1
    lists = []
    for m in named:
        chosen = sorted(picks[m["id"]][:quota[m["id"]]], key=lambda c: c["ord"] if "ord" in c else c["id"])
        lists.append([_source_from_chunk(c, m["title"], m["kind"]) for c in chosen])
    return _interleave(lists, cap=cap)


def _opening(sentence: str, words: int = 5) -> str:
    """The first words of a removed sentence, only when it has no number or math in it (a flagged line could be the answer)."""
    plain = re.sub(r"\[\d[\d,;\s\-\u2013]*\]", "", sentence).strip()
    if re.search(r"[0-9=$\\]", plain):
        return ""
    return " ".join(plain.split()[:words])


def scrub_summary(integ, store, course_id: int, text: str, scope: str = "") -> str:
    """A model-written summary with the sentences that a graded-work check flags taken out one by one: the safe
    sentences stay, and the note says which sentence went (its place and its first words). A summary that is only
    flagged as a whole paragraph is dropped whole, as before. Without the check, the text is returned as it was."""
    scrub = getattr(integ, "scrub_parts", None)
    if not scrub or not text.strip():
        old = getattr(integ, "scrub_markdown", None)
        return old(store, course_id, text, scope)[0] if old and text.strip() else text
    return scrub_sentences(lambda t: scrub(store, course_id, t, scope), text, _mod("grounding")._sentences)


def scrub_sentences(judge, text: str, split) -> str:
    """judge(text) -> (cleaned, parts dropped, note). Sentence-level removal on top of it."""
    cleaned, n, note = judge(text)
    if not n:
        return text
    kept_paras, gone = [], []
    for para in re.split(r"\n\s*\n", text.strip()):
        if not para.strip():
            continue
        if not judge(para)[1]:
            kept_paras.append(para)
            continue
        sents = split(para) if "\n" not in para.strip() else [para]
        flagged = [bool(judge(x)[1]) for x in sents]
        if not any(flagged):
            flagged = [True] * len(sents)  # only the paragraph as a whole matched
        kept = [x for x, f in zip(sents, flagged) if not f]
        if kept:
            kept_paras.append(" ".join(kept))
        gone += [(k + 1, len(sents), x) for k, (x, f) in enumerate(zip(sents, flagged)) if f]
    where = []
    for k, total, x in gone:
        start = _opening(x)
        where.append(f"sentence {k} of {total}" + (f', which begins "{start} ..."' if start else ""))
    detail = " Left out of the summary: " + "; ".join(where) + "." if where else ""
    out = f"_{note.rstrip('.')}.{detail}_" if note else ""
    return (out + "\n\n" + "\n\n".join(kept_paras)).strip()


def concept_map(store: Store, course_id: int, provider, *, scope: str = "", material_ids: list[int] | None = None) -> int:
    _need(provider)
    scope = scope.strip()
    pool = [m for m in store.materials(course_id) if _usable(m)]
    if material_ids:
        pool = [m for m in store.materials(course_id) if m["id"] in {int(i) for i in material_ids}]
    scoped = _scoped_sources(store, pool, scope) if scope else []
    if scoped:
        sources = scoped
    elif scope:
        mine = [m["id"] for m in pool if scope.casefold() in m["title"].casefold()]
        sources = _merge(_spread_materials(store, mine, 8, per=4) if mine else [],
                         _search(store, course_id, scope, MAX_SOURCES, exclude_graded=True,
                                 **({"material_ids": [m["id"] for m in pool]} if material_ids else {})),
                         cap=MAX_SOURCES)
    else:
        # every material takes part, primary sources and 'other' files included, sampled across each whole file
        sources = _spread_materials(store, [m["id"] for m in pool], MAX_SOURCES, per=3)
    user = (f"Build a concept map of {scope or 'the main ideas of ' + _course_name(store, course_id)} from these "
            "sources. Reply with ONE JSON object, then a blank line, then a short summary (3 to 5 sentences) with [n] "
            "citations. JSON shape: {\"nodes\":[{\"id\":\"short_id\",\"label\":\"Concept\",\"cites\":[1,2]}],"
            "\"edges\":[{\"a\":\"id1\",\"b\":\"id2\",\"label\":\"relationship, e.g. 'leads to'\",\"cites\":[3]}]}. "
            "Use 6 to 20 nodes. Every node needs at least one cite from the numbered sources. Every edge needs at least "
            "one cite: the source that says the two ideas are linked that way. Cover every kind of source, readings and "
            "primary documents included. Edges only between node ids you defined.\n\n"
            f"SOURCES\n\n{_sources_block(_prepare(store, course_id, sources))}")
    sources = _prepare(store, course_id, sources)
    graph, rest = _ask_json(provider, _system(store, course_id), user, dict)
    graph = _clean_graph(graph, len(sources))
    integ = _mod("integrity")
    leak = getattr(integ, "key_leak_any", None) or getattr(integ, "key_leak", None)
    if leak:  # a node or link that states a graded answer is dropped, the rest of the map stays
        def clean(t: str) -> bool:
            try:
                return not leak(store, course_id, t, None, {"named": True, "small": False})
            except TypeError:
                return not leak(store, course_id, t)
        graph["nodes"] = [n for n in graph["nodes"] if clean(n["label"])]
        ids = {n["id"] for n in graph["nodes"]}
        graph["edges"] = [e for e in graph["edges"] if e["a"] in ids and e["b"] in ids and clean(e["label"])]
        with _mod("essay").reply_sources(sources):
            rest = scrub_summary(integ, store, course_id, rest or "", scope)
    if not graph["nodes"]:
        raise ValueError("The AI returned a concept map with no usable nodes. Try again.")
    ans = _mod("grounding").parse_answer(rest or "", sources)
    summary = _clean_body(ans.body_md) or "Concepts in this map: " + ", ".join(n["label"] for n in graph["nodes"]) + "."
    labels = {n["id"]: n["label"] for n in graph["nodes"]}
    lines = [f"- **{n['label']}** " + "".join(f"[{c}]" for c in n["cites"]) for n in graph["nodes"]]
    lines += [f"- {labels[e['a']]} {e['label'] or 'relates to'} {labels[e['b']]} " + "".join(f"[{c}]" for c in e["cites"])
              for e in graph["edges"]]
    parsed = _mod("grounding").parse_answer("## Concepts and links\n\n" + "\n".join(l.rstrip() for l in lines), sources)
    used = sorted({c for n in graph["nodes"] for c in n["cites"]} | {c for e in graph["edges"] for c in e["cites"]}
                  | set(ans.used or []))
    body = f"{summary}\n\n{_clean_body(parsed.body_md)}\n\n{disclaimer.FOOTER}"
    graph = dict(graph, blocks=_block_dicts(list(ans.blocks) + list(parsed.blocks)), mode="answer",
                 not_covered=bool(getattr(ans, "not_covered", False)))
    return store.add_guide(course_id, "concept_map", f"Concept map: {scope}" if scope else "Concept map", body,
                           sources=_guide_sources(sources, used), data=graph, question=scope,
                           model=getattr(provider, "model", "") or "")


# ---------------------------------------------------------------- explain
def explain(store: Store, course_id: int, provider, topic: str, *, depth: str = "standard") -> int:
    _need(provider)
    depth = DEPTHS.get((depth or "standard").strip().lower(), "standard")
    guide = {"quick": "Give a quick explanation: 1 short paragraph and at most 4 bullets.",
             "standard": "Give a standard explanation: what it is, why it matters, how it works with one worked "
                         "example, and the usual pitfalls.",
             "from scratch": "Assume the student has never seen this. Define every term the first time you use it. "
                             "Use short sentences. Start from the prerequisites and build up in small steps with plain "
                             "words. Work exactly one example, taken from the sources when they have one. Connect it "
                             "back to how the course uses it. End with a section '## Check your understanding' that "
                             "has 2 questions for the student to try, with no answers."}[depth]
    hit = _mod("integrity").match_graded(store, course_id, topic)
    if hit and hit.get("id") and hit.get("kind") != "essay" and not getattr(_mod("integrity"), "names_item", lambda t, i: True)(topic, hit):
        hit = None  # a bare topic never turns into a twin
    if not hit:  # the request names a graded essay by its title ("Essay 2: The Federalist Essay"): its own prompt and rubric
        hit = _named_essay(store, course_id, topic)
    if hit:  # a graded problem pasted into Explain: a twin (or coaching), decided before the model sees it
        return _route_graded(store, course_id, provider, topic, hit)
    sources = _defining_first(_search(store, course_id, topic, 24, exclude_graded=True), topic)[:10]
    task = f"Explain \"{topic}\" for a student in {_course_name(store, course_id)}. Depth: {depth}. {guide}"
    return _generate(store, course_id, provider, "explain", f"Explain ({depth}): {topic}", task, sources,
                     question=topic, data={"depth": depth}, max_tokens=3500)


def _gview(store: Store, course_id: int) -> list[dict[str, Any]]:
    gv = getattr(_mod("integrity"), "graded_view", None)
    return gv(store, course_id) if gv else [dict(g, kind=g.get("kind") or "problem") for g in store.graded_items(course_id)]


def _named_essay(store: Store, course_id: int, topic: str) -> dict[str, Any] | None:
    """A graded essay or short-answer item whose title the request names, in full or by its 'Essay 2' prefix."""
    q = f" {_norm(topic)} "
    for it in _gview(store, course_id):
        title = (it.get("title") or "").strip()
        if it.get("kind") != "essay" or not title or not (it.get("body") or "").strip():
            continue
        full = _norm(title)
        prefix = _norm(title.split(":")[0]) if ":" in title else ""
        if (full and f" {full} " in q) or (prefix and re.search(r"\d", prefix) and f" {prefix} " in q):
            return dict(it, source="material" if it.get("material_id") else "item")
    return None


# ---------------------------------------------------------------- practice
_STOP_WORDS = {"the", "and", "with", "from", "that", "this", "problem", "question", "about", "help", "please",
               "practice", "solve", "find", "using", "what", "each", "have"}


def _graded_overlaps(store: Store, course_id: int, topic: str, limit: int = 3) -> list[dict[str, Any]]:
    """Graded items the request NAMES (their number or label, their distinctive numbers or words): the matcher's pick,
    then any item the topic names. A bare topic like "LU factorization" names none. See _graded_offers."""
    integ = _mod("integrity")
    names_item = getattr(integ, "names_item", lambda text, item: True)
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(g):
        if not g:
            return
        key = str(g.get("id") or g.get("title"))
        if key not in seen:
            seen.add(key)
            out.append(g)

    hit = integ.match_graded(store, course_id, topic)
    if hit and (not hit.get("id") or names_item(topic, hit) or hit.get("kind") == "essay"):
        add(hit)
    for it in store.graded_items(course_id):
        if (it.get("body") or "").strip() and names_item(topic, it):
            add(dict(it, source="item"))
    return out[:limit]


def _distinct_words(text: str) -> set[str]:
    """Concept words of a request, with acronyms (LU, QR) counted as words: stems of 5 letters, no generic words."""
    integ = _mod("integrity")
    full = getattr(integ, "content_tokens", lambda t: set())(text or "")
    words = {w[:5] for w in full if w not in _STOP_WORDS and w not in _TOPIC_GENERIC}
    return words | {a.lower() for a in re.findall(r"\b[A-Z]{2,4}\b", text or "")}


def _scrub_offer(store: Store, normal: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The button for the item the scrub note names (the same item, never another one)."""
    for iid in (normal or {}).get("scrub_items") or []:
        it = store.graded_item(iid)
        if it and (it.get("kind") or "problem") == "problem" and it.get("label"):
            return [{"id": it["id"], "label": it["label"], "title": it.get("title", "")}]
    return []


def _graded_offers(store: Store, course_id: int, topic: str, limit: int = 1) -> list[dict[str, Any]]:
    """At most one numeric graded item the topic only overlaps by words, offered as a button, never twinned unasked. It
    takes two or more distinctive content words that the request and the item share."""
    named = {str(g.get("id")) for g in _graded_overlaps(store, course_id, topic)}
    want = _distinct_words(topic)
    best: tuple[int, dict[str, Any]] | None = None
    for it in store.graded_items(course_id):
        if str(it["id"]) in named or (it.get("kind") or "problem") != "problem" or not (it.get("body") or "").strip() \
                or not it.get("label"):
            continue
        shared = len(want & _distinct_words(f"{it.get('title', '')} {it.get('label', '')} {it.get('body', '')}"))
        if shared >= 2 and (best is None or shared > best[0]):
            best = (shared, {"id": it["id"], "label": it["label"], "title": it.get("title", "")})
    return [best[1]] if best and limit >= 1 else []


_TOPIC_GENERIC = {"solution", "system", "number", "equation", "problem", "problems", "question", "questions", "points",
                  "using", "value", "values", "write", "explain", "describe", "determine", "step", "steps", "course",
                  "chapter", "lecture", "which", "graded", "show"}
_CITE_MARK = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def _shift_block(b: dict[str, Any], offset: int) -> dict[str, Any]:
    """Renumber a block's own citations so they point into a merged source list."""
    cites = [int(c) for c in b.get("cites") or []]
    if not cites or not offset:
        return dict(b)
    allowed = set(cites)

    def fix(m):
        nums = [int(x) for x in re.split(r"\s*,\s*", m.group(1))]
        if all(n in allowed for n in nums):
            return "[" + ", ".join(str(n + offset) for n in nums) + "]"
        return m.group(0)

    return {"kind": b.get("kind", "text"), "md": _CITE_MARK.sub(fix, b.get("md", "")),
            "cites": [c + offset for c in cites]}


def _hide_twin_solution(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Everything after the "Worked solution of the twin" label goes inside one <details> block, collapsed."""
    idx = next((i for i, b in enumerate(blocks) if "Worked solution of the twin" in (b.get("md") or "")), None)
    if idx is None or idx + 1 >= len(blocks):
        return blocks
    rest = blocks[idx + 1:]
    inner = "\n\n".join(b.get("md", "") for b in rest)
    cites = sorted({int(c) for b in rest for c in b.get("cites") or []})
    hidden = {"kind": "text", "cites": cites, "md": f"<details>\n<summary>Show solution</summary>\n\n{inner}\n\n</details>"}
    return blocks[: idx + 1] + [hidden]


def _render_blocks(blocks: list[dict[str, Any]]) -> str:
    label = getattr(_mod("grounding"), "OUTSIDE_LABEL", "Not from your materials")
    out, prev = [], False
    for b in blocks:
        if b.get("kind") == "outside":
            lines = (b.get("md") or "").splitlines() or [""]
            if not prev:
                lines[0] = f"**{label}:** {lines[0]}"
            out.append("\n".join(f"> {ln}" for ln in lines))
            prev = True
        else:
            out.append(b.get("md", ""))
            prev = False
    return "\n\n".join(out)


def practice(store: Store, course_id: int, provider, topic: str, *, count: int = 5) -> int:
    """N practice questions. Where the topic overlaps graded work, some of the N are twins of it (never the original)."""
    _need(provider)
    count = max(1, min(int(count), 20))
    overlaps = _graded_overlaps(store, course_id, topic, limit=min(3, count))
    twins: list[dict[str, Any]] = []
    for g in overlaps:
        ans = _mod("integrity").twin(store, course_id, topic, provider, g)
        tdata: dict[str, Any] = {}
        gid = getattr(ans, "guide_id", None)
        if gid:
            tg = store.guide(gid)
            tdata = (tg or {}).get("data") or {}
            store.delete_guide(gid)  # its content is folded into this practice set
        blocks = _block_dicts(getattr(ans, "blocks", None)) or [
            {"kind": "text", "md": _clean_body(getattr(ans, "body_md", "")), "cites": []}]
        twins.append({"graded": g, "blocks": blocks, "sources": list(getattr(ans, "sources", None) or []),
                      "mode": tdata.get("mode") or "twin", "title": g.get("title", "a graded item")})
    n_normal = count  # twins and coaching notes are extras: they never count toward the requested number
    normal = None
    if n_normal > 0:
        sources = _defining_first(_search(store, course_id, topic, 24, exclude_graded=True), topic)[:10]
        if has_math(store, course_id):
            task = (f"Write {n_normal} practice questions on \"{topic}\" for {_course_name(store, course_id)}, easiest to "
                    "hardest, in the style of the materials. For each, number it and state the question, then give the "
                    "worked solution step by step with citations, inside a block laid out exactly like this, with "
                    "<details> alone on its line and <summary> alone on the next line:\n<details>\n"
                    "<summary>Show solution</summary>\n\nsteps and final answer here\n\n</details>\n"
                    "Never show the solution outside the details block. Invent your own numbers; do not copy a question "
                    "from the materials.")
        else:
            task = (f"Write {n_normal} practice questions on \"{topic}\" for {_course_name(store, course_id)}, easiest to "
                    "hardest, about the readings and arguments in the materials (short-answer or source-analysis questions; "
                    "no calculations). For each, number it and state the question, then give a model answer built from "
                    "the sources, with citations, inside a block laid out exactly like this, with <details> alone on its "
                    "line and <summary> alone on the next line:\n<details>\n<summary>Show answer</summary>\n\n"
                    "the answer here\n\n</details>\n"
                    "Never show the answer outside the details block. Do not write a thesis or any paragraph of an "
                    "essay the student must submit, and do not copy a question from the materials.")
        normal = _compose(store, course_id, provider, task, sources, question=topic, max_tokens=5000, practice=True)
        if normal["routed"] and not twins:
            return normal["routed"]
        if normal["routed"]:
            normal = None
    if normal and normal.get("scrub_note"):  # name the item that matched, once, ahead of the questions
        note = normal["scrub_note"]
        normal = dict(normal, blocks=[{"kind": "text", "md": note, "cites": []}] + list(normal["blocks"]),
                      body=f"> {note}\n\n{normal['body']}")
    if not twins:
        body = f"{normal['body']}\n\n{disclaimer.FOOTER}"
        data = {"twin": False, "count": count, "mode": "answer", "blocks": normal["blocks"],
                "outside": normal["outside"], "not_covered": normal["not_covered"], "twins": [],
                "held_back": bool(normal.get("held")),
                "graded_offers": _scrub_offer(store, normal) or _graded_offers(store, course_id, topic)}
        return store.add_guide(course_id, "practice", f"Practice: {topic}", body,
                               sources=_guide_sources(normal["sources"], normal["used"]), data=data, question=topic,
                               model=getattr(provider, "model", "") or "")
    blocks: list[dict[str, Any]] = []
    all_sources: list[dict[str, Any]] = []
    used: list[int] = []
    real = [t for t in twins if t["mode"] == "twin"]
    coach = [t for t in twins if t["mode"] != "twin"]
    titles = ", ".join(t["title"] for t in twins)
    parts = [f"This set has {count} practice question{'s' if count != 1 else ''}."]
    if real:
        parts.append(f"In addition, {len(real)} {'is a twin' if len(real) == 1 else 'are twins'} of your graded work "
                     f"({', '.join(t['title'] for t in real)}), with different numbers, so your original stays yours to do. "
                     "Twins do not count toward the number of questions.")
    if coach:
        parts.append(f"{len(coach)} {'is coaching' if len(coach) == 1 else 'are coaching notes'} for graded work "
                     f"({', '.join(t['title'] for t in coach)}): concepts and an approach, no twin problem, because it is "
                     "an essay, a proof or could not be turned into a safe twin.")
    blocks.append({"kind": "text", "cites": [], "md": " ".join(parts)})
    if normal:
        all_sources = list(normal["sources"])
        used = list(normal["used"])
        blocks += normal["blocks"]
    for t in twins:
        offset = len(all_sources)
        tsrc = [dict(s) for s in t["sources"]]
        shifted = _hide_twin_solution([_shift_block(b, offset) for b in t["blocks"]]) if t["mode"] == "twin" else \
            [_shift_block(b, offset) for b in t["blocks"]]
        all_sources += _prepare(store, course_id, tsrc, keep_graded=True)
        blocks.append({"kind": "text", "cites": [], "md": (f"## Twin of graded work: {t['title']}" if t["mode"] == "twin"
                                                           else f"## Coaching for graded work: {t['title']}")})
        blocks += shifted
        used += [c + offset for b in t["blocks"] if b.get("kind") == "text" for c in b.get("cites") or []]
    verify = getattr(_mod("grounding"), "verify_quotes", None)
    Block = getattr(_mod("grounding"), "Block", None)
    if verify and Block:
        objs = [Block(b["kind"], b["md"], list(b["cites"])) for b in blocks]
        verify(objs, all_sources)
        blocks = _block_dicts(objs)
    offers = _scrub_offer(store, normal) or _graded_offers(store, course_id, topic)
    if not offers:  # "Press Practice below" is only true when the page has the button
        for b in blocks:
            b["md"] = (b.get("md") or "").replace("Press Practice below or ask again", "Ask again")
    body = f"{_render_blocks(blocks)}\n\n{disclaimer.FOOTER}"
    only_twins = normal is None
    mode = twins[0]["mode"] if only_twins and len({t['mode'] for t in twins}) == 1 else "answer"  # twin only if one was made
    data = {"twin": bool(real), "count": count, "mode": mode, "blocks": blocks, "outside": any(b["kind"] == "outside" for b in blocks),
            "not_covered": bool(normal and normal["not_covered"]), "graded": titles,
            "twins": [{"graded": t["title"], "mode": t["mode"]} for t in twins],
            "graded_offers": offers}
    return store.add_guide(course_id, "practice", f"Practice (twin): {topic}" if only_twins and real else f"Practice: {topic}", body,
                           sources=_guide_sources(all_sources, sorted(set(used))), data=data, question=topic,
                           model=getattr(provider, "model", "") or "")


# ---------------------------------------------------------------- flashcards
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").casefold()).strip()


def _section_chunks(store: Store, chunk_id: int, cap: int = 4) -> list[dict[str, Any]]:
    """The chunk and the chunks after it that share its heading (the rest of its section)."""
    ch = store.chunk(chunk_id)
    if not ch:
        return []
    rows = store.chunks_of(ch["material_id"])
    idx = next((i for i, r in enumerate(rows) if r["id"] == chunk_id), None)
    if idx is None:
        return [ch]
    out = [rows[idx]]
    for r in rows[idx + 1: idx + cap]:
        if ch.get("heading") and r.get("heading") == ch["heading"]:
            out.append(r)
        else:
            break
    return out


def _card_gives_away(store: Store, course_id: int, front: str, back: str, source_text: str = "") -> bool:
    integ = _mod("integrity")
    if hasattr(integ, "graded_view") and _mod("essay").card_hit(store, course_id, front, back, source_text):
        return True  # an essay thesis, position or conclusion
    card = getattr(integ, "check_card", None)
    if card:
        return bool(card(store, course_id, front, back))
    reply = getattr(integ, "check_reply", None)
    return bool(reply and reply(store, course_id, front, f"{front}\n{back}"))


def _card_worthy(sources: list[dict[str, Any]], *, topic_search: bool = False) -> list[dict[str, Any]]:
    """Flashcards are made from what a book or lecture teaches: not exercise lists, glossaries, chapter summaries,
    introductions, title blocks, and (for a topic search) not the syllabus or an earlier term's schedule. When nothing
    else is left the original list stays, so a thin material still makes cards."""
    r = _mod("retrieval")
    scaffold, title_only = getattr(r, "is_scaffold_chunk", None), getattr(r, "is_title_only", None)
    admin = getattr(r, "_is_admin_chunk", None)

    def worthy(x):
        if scaffold and scaffold(x["text"], x.get("heading", "")):
            return False
        if title_only and title_only(x["text"], x.get("heading", "")):
            return False
        if topic_search and (x.get("material_kind") in ("syllabus", "past_offering") or (admin and admin(x))):
            return False
        return True
    kept = [x for x in sources if worthy(x)]
    return kept or sources


def card_source_label(title: str, loc: str = "") -> str:
    """'Title, place' for a card's source line. A place that only repeats the title (a section named like its
    material) is left out, and a label saved earlier as 'Title, Title' is read back as 'Title'."""
    title, loc = (title or "").strip(), (loc or "").strip()
    if not title and "," in loc:
        title, loc = (x.strip() for x in loc.split(",", 1))
    if "," in title and not loc:  # an old saved label, 'Title, Title'
        a, b = (x.strip() for x in title.rsplit(",", 1))
        if a.lower() == b.lower():
            return a
        return title
    if not loc or loc.lower() == title.lower():
        return title
    return f"{title}, {loc}"


def make_flashcards(store: Store, course_id: int, provider, *, material_id: int | None = None, topic: str = "",
                    count: int = 12) -> FlashcardResult:
    _need(provider)
    count = max(1, min(int(count), 50))
    concept_only = False
    if material_id is not None:
        m = store.material(material_id)
        if not m or m["course_id"] != course_id:
            raise ValueError("That material is not in this course.")
        if m.get("graded"):
            if not any(g.get("material_id") == m["id"] for g in _gview(store, course_id)):
                raise ValueError("That material is marked as graded work, so the app will not make flashcards from it. "
                                 "Pick another material or type a topic.")
            concept_only = True  # graded work: cards on concepts, requirements and the rubric only, no key values
        mat_sources = _card_worthy(_material_sources(store, material_id, 10_000))  # the whole material, minus scaffolding
        if topic.strip():
            hits = _card_worthy(_search(store, course_id, topic, MAX_SOURCES, material_ids=[material_id]))
            hit_ids = {s["chunk_id"] for s in hits}
            rest = [s for s in mat_sources if s["chunk_id"] not in hit_ids]
            sources = _merge(hits, _spread_pick(rest, max(0, MAX_SOURCES - len(hits))), cap=MAX_SOURCES)
            order = {x["chunk_id"]: i for i, x in enumerate(mat_sources)}
            sources.sort(key=lambda s: order.get(s["chunk_id"], len(order)))
        else:
            sources = _spread_pick(mat_sources, MAX_SOURCES)
        deck = m["title"]
    else:
        if not topic.strip():
            raise ValueError("Give a material or a topic.")
        hits = _search(store, course_id, topic, 14, exclude_graded=True)
        extra = []
        for h in hits[:8]:  # whole sections, not only the section's first chunk
            for c in _section_chunks(store, h["chunk_id"])[1:]:
                mm = store.material(c["material_id"]) or {}
                extra.append(_source_from_chunk(c, mm.get("title", ""), mm.get("kind", "")))
        sources = _card_worthy(_merge(hits, extra, cap=MAX_SOURCES * 2), topic_search=True)[:MAX_SOURCES]
        deck = topic.strip()
    sources = _prepare(store, course_id, sources, keep_graded=concept_only)
    only = (" These sources are graded work: make cards about concepts, key terms, methods, requirements and what the "
            "rubric rewards only. Never write a thesis, a position, an argument or a conclusion for an essay, and never "
            "give a number, matrix, vector, formula or other result that answers one of its problems." if concept_only else "")
    math = has_math(store, course_id)
    cover = ("the main concepts, definitions, formulas and procedures" if math else
             "the key terms, people, events, ideas and arguments in the readings")
    user = (f"Make up to {count} study flashcards from these sources. Cover {cover}, and spread the cards across the "
            "sources: at most 2 cards from one source. Reply with ONLY a "
            "JSON list: [{\"front\":\"question or term\",\"back\":\"short answer\",\"cite\":1}]. One idea per card, "
            f"backs under 40 words, {'math as $...$, ' if math else ''}cite is the single source number the answer comes from. Skip anything "
            f"the sources do not support.{only}\n\nSOURCES\n\n{_sources_block(sources)}")
    cards, _ = _ask_json(provider, _system(store, course_id), user, list)
    res = FlashcardResult()
    seen = {_norm(c["front"]) for c in store.flashcards(course_id)}
    per_source: dict[int, int] = {}
    ok: list[tuple[str, str, int]] = []
    for c in cards:
        if not isinstance(c, dict):
            res.malformed += 1
            continue
        front, back = str(c.get("front", "")).strip(), str(c.get("back", "")).strip()
        try:
            cite = int(c.get("cite"))
        except (TypeError, ValueError):
            res.dropped += 1
            continue
        if not front or not back:
            res.malformed += 1
        elif not (1 <= cite <= len(sources)):
            res.dropped += 1
        elif _norm(front) in seen:
            res.duplicates += 1
        elif _card_gives_away(store, course_id, front, back, sources[cite - 1].get("text", "")):
            res.guarded += 1  # a card whose back works a graded problem
        else:
            seen.add(_norm(front))
            ok.append((front, back, cite))
    first, rest = [], []
    for t in ok:  # at most 2 cards per source first, the overflow only if the count is not reached
        if per_source.get(t[2], 0) < 2:
            per_source[t[2]] = per_source.get(t[2], 0) + 1
            first.append(t)
        else:
            rest.append(t)
    for front, back, cite in (first + rest)[:count]:
        s = sources[cite - 1]
        res.append(store.add_flashcard(course_id, front, back, deck=deck, chunk_id=s["chunk_id"],
                                       source_label=card_source_label(s["material_title"], s["loc_label"])))
    bits = [f"Made {len(res)} card{'s' if len(res) != 1 else ''}."]
    if res.dropped:
        bits.append(f"{res.dropped} card{'s were' if res.dropped != 1 else ' was'} dropped because "
                    f"{'their' if res.dropped != 1 else 'its'} source number did not match any passage.")
    if res.duplicates:
        bits.append(f"{res.duplicates} you already have {'were' if res.duplicates != 1 else 'was'} skipped.")
    if res.guarded:
        bits.append(f"{res.guarded} card{'s' if res.guarded != 1 else ''} left out because "
                    f"{'they' if res.guarded != 1 else 'it'} would give away graded work.")
    res.note = " ".join(bits)
    return res


def _exam_date_for(store: Store, course_id: int) -> str:
    nxt = [e for e in store.events(course_id, start=_today(), kinds=["exam"]) if not is_junk_event(e)]
    return nxt[0]["date"] if nxt else ""


def cram_queue(store: Store, course_id: int, *, material_id: int | None = None, limit: int = 50) -> list[dict[str, Any]]:
    """Cards for a cram session: every card, weakest first, ignoring due dates."""
    cards = store.flashcards(course_id)
    if material_id is not None:
        chunk_ids = {c["id"] for c in store.chunks_of(material_id)}
        title = (store.material(material_id) or {}).get("title", "")
        cards = [c for c in cards if c.get("chunk_id") in chunk_ids or (title and c.get("deck") == title)]
    cards.sort(key=lambda c: (float(c["ease"]), float(c["interval_days"]), int(c["reps"]), c["id"]))
    return cards[:max(0, int(limit))]


def review_flashcard(store: Store, card_id: int, grade: str, *, cram: bool = False,
                     exam_date: str | None = None) -> dict[str, Any]:
    grade = (grade or "").strip().lower()
    if grade not in GRADES:
        raise ValueError(f"grade must be one of {', '.join(GRADES)}")
    card = store.flashcard(card_id)
    if not card:
        raise KeyError(card_id)
    ease, interval, reps = float(card["ease"]), float(card["interval_days"]), int(card["reps"])
    now = datetime.now(timezone.utc).replace(microsecond=0)
    if grade == "again":
        ease, reps, interval = max(1.3, ease - 0.2), 0, 0.0
        due = now + timedelta(minutes=10)
    else:
        if grade == "hard":
            ease = max(1.3, ease - 0.15)
            interval = 1.0 if reps == 0 else max(1.0, interval * 1.2)
        elif grade == "good":
            interval = 1.0 if reps == 0 else 3.0 if reps == 1 else interval * ease
        else:
            ease += 0.15
            interval = 4.0 if reps == 0 else max(interval * ease * 1.3, 6.0 if reps == 1 else 0)
        reps += 1
        interval = round(interval, 2)
        # before an exam, a card must come back before it: in cram mode always, otherwise within the last 7 days
        exam = exam_date if exam_date is not None else _exam_date_for(store, card["course_id"])
        if exam:
            try:
                left = (_date.fromisoformat(exam) - _date.fromisoformat(_today())).days
            except ValueError:
                left = None
            if left is not None and left >= 0 and (cram or left <= 7):
                interval = round(min(interval, left / 2 if left >= 1 else 0.25), 2)
        due = now + timedelta(days=interval)
    store.update_flashcard(card_id, ease=round(ease, 3), interval_days=interval, reps=reps, due_at=due.isoformat())
    return store.flashcard(card_id)  # type: ignore[return-value]


# ---------------------------------------------------------------- missed lecture
def missed(store: Store, course_id: int, provider, *, event_id: int | None = None, date: str = "") -> int:
    _need(provider)
    ev = None
    if event_id is not None:
        ev = store.event(event_id)
        if not ev or ev["course_id"] != course_id:
            raise ValueError(f"No such event {event_id} in this course.")
        date = ev["date"]
    elif not date:
        raise ValueError("Give an event or a date.")
    else:
        same = [e for e in store.events(course_id, start=date, end=date, kinds=["lecture"]) if not is_junk_event(e)]
        ev = same[0] if same else None
    if ev:
        sources, used, note = _lecture_sources(store, course_id, ev, cap=MAX_SOURCES)
    else:
        mats = [m for m in store.materials(course_id) if _usable(m) and m["lecture_date"] == date]
        sources = _spread_materials(store, [m["id"] for m in mats], MAX_SOURCES - 4, per=10) if mats else []
        used = [(m["title"], f"dated {date}") for m in mats]
        note = ("Used: " + "; ".join(t for t, _ in used) + ".") if used else \
            "No saved material is dated that day, so this has little to go on."
    ids = [m["id"] for m in store.materials(course_id) if m["title"] in {t for t, _ in used}]
    announce = _search(store, course_id, "announcement deadline due reading assignment exam", 4,
                       material_ids=ids) if ids else []
    sources = _merge(sources[:MAX_SOURCES - 4], announce, sources[MAX_SOURCES - 4:], cap=MAX_SOURCES)
    title = clean_title(ev) if ev else date
    task = (f"A student skipped the {date} lecture{' (' + title + ')' if ev else ''} of {_course_name(store, course_id)}. "
            "Catch them up. Sections: ## Main points (what was taught, with citations), ## Announcements and deadlines "
            "(only what the sources actually say; if none, say none were found in the materials), ## What to read or "
            "do next. Be brief and concrete. If a recording or captions of the lecture are among the sources, base the "
            "main points on them first.")
    return _generate(store, course_id, provider, "missed", f"What you missed: {title}", task, sources, question=date,
                     event_id=ev["id"] if ev else None, max_tokens=3000, extra_note=note,
                     data={"used_materials": [{"title": t, "why": w} for t, w in used]})


# ---------------------------------------------------------------- dashboard
def upcoming(store: Store, days: int = 14) -> list[dict[str, Any]]:
    today = _date.today()
    end = today + timedelta(days=max(0, int(days)))
    rows = []
    guides_cache: dict[int, set[tuple[str, int]]] = {}
    due_cache: dict[int, int] = {}
    for e in store.events(start=today.isoformat(), end=end.isoformat()):
        cid = e["course_id"]
        if cid not in guides_cache:
            guides_cache[cid] = {(g["kind"], g["event_id"]) for g in store.guides(cid) if g["event_id"] is not None}
            due_cache[cid] = len(store.flashcards(cid, due_only=True))
        try:
            until = (_date.fromisoformat(e["date"]) - today).days
        except ValueError:
            continue
        action = ""
        if e["kind"] == "lecture" and ("primer", e["id"]) not in guides_cache[cid]:
            action = "make a primer"
        elif e["kind"] == "exam" and ("exam_guide", e["id"]) not in guides_cache[cid]:
            action = "make an exam guide"
        elif due_cache[cid]:
            action = f"{due_cache[cid]} flashcard{'' if due_cache[cid] == 1 else 's'} due"
        rows.append({"event_id": e["id"], "course_id": cid, "course_name": e["course_name"],
                     "course_code": e["course_code"], "course_color": e["course_color"], "kind": e["kind"],
                     "title": e["title"], "date": e["date"], "time": e["time"], "days_until": until,
                     "flashcards_due": due_cache[cid], "action": action})
    rows.sort(key=lambda r: (r["date"], r["time"] or "99:99", r["event_id"]))
    return rows
