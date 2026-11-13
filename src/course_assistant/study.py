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


# ---------------------------------------------------------------- sources
def _source_from_chunk(c: dict[str, Any], title: str = "", kind: str = "") -> dict[str, Any]:
    return {"chunk_id": c["id"], "material_id": c["material_id"],
            "material_title": c.get("material_title") or title, "material_kind": c.get("material_kind") or kind,
            "loc_kind": c["loc_kind"], "loc_start": c["loc_start"], "loc_label": c["loc_label"],
            "heading": c.get("heading", ""), "text": c["text"], "href": locations.href_for(c), "score": 0.0}


def _material_sources(store: Store, material_id: int, limit: int = 12) -> list[dict[str, Any]]:
    m = store.material(material_id)
    if not m:
        return []
    return [_source_from_chunk(c, m["title"], m["kind"]) for c in store.chunks_of(material_id)[:limit]]


def _search(store: Store, course_id: int, query: str, limit: int = 6, **kw) -> list[dict[str, Any]]:
    if not query.strip():
        return []
    return _mod("retrieval").search(store, course_id, query, limit=limit, **kw)


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


def _sources_block(sources: list[dict[str, Any]]) -> str:
    if not sources:
        return "(No course materials matched. Everything you say is outside knowledge and must be labelled.)"
    parts = []
    for i, s in enumerate(sources, 1):
        head = f"[{i}] {s.get('material_title', '')} ({s.get('loc_label', '')})"
        if s.get("heading"):
            head += f" - {s['heading']}"
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


SYSTEM = """You are Course Assistant, a patient study helper for a college student. Rules you always follow:
- Ground everything in the numbered sources the student's own course materials provide. Cite the source number in square brackets like [2] right after each claim. Cite only numbers that exist.
- If the sources do not cover something and you add it from general knowledge, put it inside <outside>...</outside> tags so the app can label it "Not from your materials". Never present outside knowledge as if it came from the sources.
- Never invent page numbers, quotes, dates or deadlines. If the materials do not say, say so.
- Use plain, friendly language a student can read quickly. Define jargon the first time you use it.
- This is often STEM material: write math in LaTeX style with $...$ for inline and $$...$$ for display, and keep units and steps explicit.
- Use Markdown with short headings, bullets and numbered lists.
- Academic integrity: never solve a problem the student says is graded. Work an example with different numbers instead."""


def _clean_body(body: str) -> str:
    body = (body or "").replace(disclaimer.FOOTER, "").rstrip()
    return body


def _generate(store: Store, course_id: int, provider, kind: str, title: str, task: str, sources: list[dict[str, Any]],
              *, question: str = "", event_id: int | None = None, data: dict[str, Any] | None = None,
              max_tokens: int = 4000, extra_note: str = "") -> int:
    user = f"{task}\n\nSOURCES\n\n{_sources_block(sources)}"
    raw = provider.complete(SYSTEM, [{"role": "user", "content": user}], max_tokens=max_tokens)
    ans = _mod("grounding").parse_answer(raw, sources)
    body = _clean_body(ans.body_md)
    blocks = list(ans.blocks)
    if extra_note:
        body = f"> {extra_note}\n\n{body}"
        blocks.insert(0, {"kind": "text", "md": extra_note, "cites": []})
    body = f"{body}\n\n{disclaimer.FOOTER}"
    d = dict(data or {})
    d["outside"] = bool(getattr(ans, "outside", False))
    d["blocks"] = _block_dicts(blocks)
    return store.add_guide(course_id, kind, title, body, sources=_guide_sources(sources, list(ans.used or [])), data=d,
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


# ---------------------------------------------------------------- events
def _pick_event(store: Store, course_id: int, event_id: int | None, kind: str) -> dict[str, Any]:
    if event_id is not None:
        e = store.event(event_id)
        if not e or e["course_id"] != course_id:
            raise ValueError(f"No such event {event_id} in this course.")
        return e
    nxt = store.events(course_id, start=_today(), kinds=[kind])
    if not nxt:
        raise ValueError(f"No upcoming {kind} on this course's calendar. Add one first.")
    return nxt[0]


def _event_sources(store: Store, course_id: int, ev: dict[str, Any], cap: int = 20) -> list[dict[str, Any]]:
    query = " ".join(x for x in (ev["title"], ev.get("topics", "")) if x)
    dated = []
    for m in store.materials(course_id):
        if m["lecture_date"] == ev["date"]:
            dated += _material_sources(store, m["id"], 8)
    syl = _search(store, course_id, query, 3, kinds=["syllabus"])
    found = _search(store, course_id, query, 10, exclude_graded=True)
    return _merge(dated, found, syl, cap=cap)


# ---------------------------------------------------------------- primer
def primer(store: Store, course_id: int, provider, *, event_id: int | None = None) -> int:
    _need(provider)
    ev = _pick_event(store, course_id, event_id, "lecture")
    sources = _event_sources(store, course_id, ev)
    task = (f"Write a short pre-lecture primer for {_course_name(store, course_id)}, lecture \"{ev['title']}\" on "
            f"{ev['date']}. Topics listed: {ev.get('topics') or 'not listed'}. Use these sections: ## The big picture "
            "(a short, broad overview), ## Key terms (each with a one-line definition), ## What to watch for in lecture, "
            "## 3 questions to keep in mind. Keep it to what a student can read in five minutes. Stay broad: this is "
            "before the lecture, so do not go deep.")
    return _generate(store, course_id, provider, "primer", f"Primer: {ev['title']}", task, sources,
                     question=ev["title"], event_id=ev["id"], max_tokens=3000)


# ---------------------------------------------------------------- exam guide
def exam_guide(store: Store, course_id: int, provider, *, event_id: int | None = None) -> int:
    _need(provider)
    ev = _pick_event(store, course_id, event_id, "exam")
    exams = [e for e in store.events(course_id, kinds=["exam"]) if e["date"] < ev["date"]]
    prev = exams[-1]["date"] if exams else ""
    lectures = [e for e in store.events(course_id, kinds=["lecture"])
                if e["date"] <= ev["date"] and (not prev or e["date"] > prev)]
    topics = [" ".join(x for x in (e["title"], e.get("topics", "")) if x) for e in lectures]
    if not topics:
        topics = [" ".join(x for x in (ev["title"], ev.get("topics", "")) if x)]
    per_topic = [_search(store, course_id, t, 4, exclude_graded=True) for t in topics]
    syl = _search(store, course_id, ev["title"], 3, kinds=["syllabus"])
    sources = _merge(_interleave(per_topic, cap=MAX_SOURCES - 3), syl)
    today = _today()
    try:
        days = max(0, (_date.fromisoformat(ev["date"]) - _date.fromisoformat(today)).days)
    except ValueError:
        days = 0
    plan_days = [(_date.fromisoformat(today) + timedelta(days=i)).isoformat() for i in range(min(days, 14) + 1)] \
        if days >= 0 and re.match(r"\d{4}-\d{2}-\d{2}$", ev["date"]) else []
    graded = _mod("integrity").match_graded(store, course_id, " ".join(topics))
    note = ("Practice questions below are original twins with different numbers. Questions you marked as graded are "
            "not solved here.") if graded else ""
    task = (f"Write an exam study guide for {_course_name(store, course_id)}: \"{ev['title']}\" on {ev['date']} "
            f"({days} days from today, {today}). Lectures covered since "
            f"{'the previous exam on ' + prev if prev else 'the start of the course'}:\n- " + "\n- ".join(topics) +
            "\n\nSections: ## What's covered (by topic, with citations), ## Key formulas and definitions, ## Common "
            "mistakes, ## Practice set (4 to 6 short questions; give answers hidden in <details><summary>Answer"
            "</summary>...</details>; make up new numbers and never copy a graded problem; if a graded problem would be "
            "the natural question, write a twin with different numbers instead), ## Study plan (one line per day from "
            f"these dates: {', '.join(plan_days) or ev['date']}; spread the topics, put review and rest near the end).")
    return _generate(store, course_id, provider, "exam_guide", f"Exam guide: {ev['title']}", task, sources,
                     question=ev["title"], event_id=ev["id"], data={"topics": topics, "previous_exam": prev},
                     max_tokens=5000, extra_note=note)


# ---------------------------------------------------------------- concept map
def _clean_graph(graph: dict[str, Any], n_sources: int) -> dict[str, Any]:
    nodes, ids = [], set()
    for n in graph.get("nodes") or []:
        if not isinstance(n, dict) or not str(n.get("id", "")).strip() or not str(n.get("label", "")).strip():
            continue
        nid = str(n["id"]).strip()
        if nid in ids:
            continue
        cites = []
        for c in n.get("cites") or []:
            try:
                ci = int(c)
            except (TypeError, ValueError):
                continue
            if 1 <= ci <= n_sources and ci not in cites:
                cites.append(ci)
        ids.add(nid)
        nodes.append({"id": nid, "label": str(n["label"]).strip(), "cites": cites})
    edges = []
    for e in graph.get("edges") or []:
        if not isinstance(e, dict):
            continue
        a, b = str(e.get("a", "")).strip(), str(e.get("b", "")).strip()
        if a in ids and b in ids and a != b:
            edges.append({"a": a, "b": b, "label": str(e.get("label", "")).strip()})
    return {"nodes": nodes, "edges": edges}


def concept_map(store: Store, course_id: int, provider, *, scope: str = "") -> int:
    _need(provider)
    scope = scope.strip()
    if scope:
        sources = _search(store, course_id, scope, MAX_SOURCES, exclude_graded=True)
    else:
        sources = []
        for m in store.materials(course_id):
            if m["kind"] in ("slides", "handout", "notes", "textbook", "recording"):
                sources = _merge(sources, _material_sources(store, m["id"], 3))
    user = (f"Build a concept map of {scope or 'the main ideas of ' + _course_name(store, course_id)} from these "
            "sources. Reply with ONE JSON object, then a blank line, then a short summary (3 to 5 sentences) with [n] "
            "citations. JSON shape: {\"nodes\":[{\"id\":\"short_id\",\"label\":\"Concept\",\"cites\":[1,2]}],"
            "\"edges\":[{\"a\":\"id1\",\"b\":\"id2\",\"label\":\"relationship, e.g. 'leads to'\"}]}. Use 6 to 20 nodes. "
            "Every node needs at least one cite from the numbered sources. Edges only between node ids you defined.\n\n"
            f"SOURCES\n\n{_sources_block(sources)}")
    graph, rest = _ask_json(provider, SYSTEM, user, dict)
    graph = _clean_graph(graph, len(sources))
    if not graph["nodes"]:
        raise ValueError("The AI returned a concept map with no usable nodes. Try again.")
    ans = _mod("grounding").parse_answer(rest or "", sources)
    summary = _clean_body(ans.body_md) or "Concepts in this map: " + ", ".join(n["label"] for n in graph["nodes"]) + "."
    labels = {n["id"]: n["label"] for n in graph["nodes"]}
    lines = [f"- **{n['label']}** " + "".join(f"[{c}]" for c in n["cites"]) for n in graph["nodes"]]
    lines += [f"- {labels[e['a']]} {e['label'] or 'relates to'} {labels[e['b']]}" for e in graph["edges"]]
    parsed = _mod("grounding").parse_answer("## Concepts and links\n\n" + "\n".join(lines), sources)
    used = sorted({c for n in graph["nodes"] for c in n["cites"]} | set(ans.used or []))
    body = f"{summary}\n\n{_clean_body(parsed.body_md)}\n\n{disclaimer.FOOTER}"
    graph = dict(graph, blocks=_block_dicts(list(ans.blocks) + list(parsed.blocks)))
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
             "from scratch": "Assume the student has never seen this. Start from the prerequisites, build up in small "
                             "steps with plain words and an analogy or two, include a simple worked example, then "
                             "connect it back to how the course uses it."}[depth]
    sources = _search(store, course_id, topic, 10, exclude_graded=True)
    task = f"Explain \"{topic}\" for a student in {_course_name(store, course_id)}. Depth: {depth}. {guide}"
    return _generate(store, course_id, provider, "explain", f"Explain ({depth}): {topic}", task, sources,
                     question=topic, data={"depth": depth}, max_tokens=3500)


# ---------------------------------------------------------------- practice
def practice(store: Store, course_id: int, provider, topic: str, *, count: int = 5) -> int:
    _need(provider)
    count = max(1, min(int(count), 20))
    graded = _mod("integrity").match_graded(store, course_id, topic)
    if graded:
        ans = _mod("integrity").twin(store, course_id, topic, provider, graded)
        body = (f"> This topic matches graded work ({graded.get('title', 'a graded item')}). I wrote a twin with "
                "different numbers and left your original for you.\n\n" + _clean_body(ans.body_md) +
                f"\n\n{disclaimer.FOOTER}")
        used = list(ans.used or [])
        srcs = [{"n": i, "chunk_id": s["chunk_id"], "material_id": s["material_id"], "label": s["loc_label"],
                 "title": s["material_title"], "href": s["href"]}
                for i, s in enumerate(ans.sources or [], 1) if i in used]
        note = {"kind": "text", "cites": [],
                "md": f"This topic matches graded work ({graded.get('title', 'a graded item')}). "
                      "I wrote a twin with different numbers and left your original for you."}
        return store.add_guide(course_id, "practice", f"Practice (twin): {topic}", body, sources=srcs,
                               data={"blocks": _block_dicts([note, *ans.blocks]), "twin": True, "graded": graded.get("title", ""), "twin_guide_id": ans.guide_id},
                               question=topic, model=getattr(provider, "model", "") or "")
    sources = _search(store, course_id, topic, 10, exclude_graded=True)
    task = (f"Write {count} practice questions on \"{topic}\" for {_course_name(store, course_id)}, easiest to hardest, "
            "in the style of the materials. For each, number it, state the question, then give the worked solution "
            "step by step with citations inside a block exactly like:\n<details><summary>Show solution</summary>\n\n"
            "steps and final answer here\n\n</details>\nNever show the solution outside the details block. Invent your "
            "own numbers; do not copy a question from the materials.")
    return _generate(store, course_id, provider, "practice", f"Practice: {topic}", task, sources, question=topic,
                     data={"twin": False, "count": count}, max_tokens=5000)


# ---------------------------------------------------------------- flashcards
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").casefold()).strip()


def make_flashcards(store: Store, course_id: int, provider, *, material_id: int | None = None, topic: str = "",
                    count: int = 12) -> list[int]:
    _need(provider)
    count = max(1, min(int(count), 50))
    if material_id is not None:
        m = store.material(material_id)
        if not m or m["course_id"] != course_id:
            raise ValueError("That material is not in this course.")
        sources = _material_sources(store, material_id, MAX_SOURCES)
        deck = m["title"]
    else:
        if not topic.strip():
            raise ValueError("Give a material or a topic.")
        sources = _search(store, course_id, topic, MAX_SOURCES, exclude_graded=True)
        deck = topic.strip()
    user = (f"Make up to {count} study flashcards from these sources. Reply with ONLY a JSON list: "
            "[{\"front\":\"question or term\",\"back\":\"short answer\",\"cite\":1}]. One idea per card, backs under "
            "40 words, math as $...$, cite is the single source number the answer comes from. Skip anything the sources "
            "do not support.\n\n" f"SOURCES\n\n{_sources_block(sources)}")
    cards, _ = _ask_json(provider, SYSTEM, user, list)
    seen = {_norm(c["front"]) for c in store.flashcards(course_id)}
    ids: list[int] = []
    for c in cards:
        if len(ids) >= count:
            break
        if not isinstance(c, dict):
            continue
        front, back = str(c.get("front", "")).strip(), str(c.get("back", "")).strip()
        try:
            cite = int(c.get("cite"))
        except (TypeError, ValueError):
            continue
        if not front or not back or not (1 <= cite <= len(sources)) or _norm(front) in seen:
            continue
        seen.add(_norm(front))
        s = sources[cite - 1]
        ids.append(store.add_flashcard(course_id, front, back, deck=deck, chunk_id=s["chunk_id"],
                                       source_label=f"{s['material_title']}, {s['loc_label']}"))
    return ids


def review_flashcard(store: Store, card_id: int, grade: str) -> dict[str, Any]:
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
        same = [e for e in store.events(course_id, start=date, end=date, kinds=["lecture"])]
        ev = same[0] if same else None
    mats = [m for m in store.materials(course_id) if m["lecture_date"] == date]
    dated: list[dict[str, Any]] = []
    for m in mats:
        dated += _material_sources(store, m["id"], MAX_SOURCES)
    extra = []
    if ev:
        extra = _search(store, course_id, " ".join(x for x in (ev["title"], ev.get("topics", "")) if x), 8,
                        exclude_graded=True)
    announce = _search(store, course_id, "announcement deadline due reading assignment exam", 4,
                       material_ids=[m["id"] for m in mats] or None) if mats else []
    sources = _merge(dated[:MAX_SOURCES - 8], announce, extra)
    title = ev["title"] if ev else date
    task = (f"A student skipped the {date} lecture{' (' + title + ')' if ev else ''} of {_course_name(store, course_id)}. "
            "Catch them up. Sections: ## Main points (what was taught, with citations), ## Announcements and deadlines "
            "(only what the sources actually say; if none, say none were found in the materials), ## What to read or "
            "do next. Be brief and concrete.")
    return _generate(store, course_id, provider, "missed", f"What you missed: {title}", task, sources,
                     question=date, event_id=ev["id"] if ev else None, max_tokens=3000)


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
