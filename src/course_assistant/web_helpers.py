"""Small helpers for the web pages: dates, course colours, answer views, concept map layout."""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Any

from . import disclaimer
import itertools

from .web_md import clean_label, count_marks, render, shallowest_heading, source_map, split_outside

# Accent colours: each passes AA as a bar or dot on the page background and as text on its tinted card.
PALETTE = {
    "blue": "#2f5fd0", "green": "#1f7a45", "purple": "#7a3fc0", "orange": "#b85a14", "teal": "#0b7a7a",
    "pink": "#b8337a", "red": "#c0332b", "gold": "#8a6a00",
}
PALETTE_NAMES = list(PALETTE)
HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")
MODELS = [("claude-opus-5-5", "Claude Opus 5.5 (best answers, the default)"),
          ("claude-sonnet-5-5", "Claude Sonnet 5.5 (faster, lower cost)"),
          ("claude-haiku-5-5", "Claude Haiku 5.5 (fastest, lowest cost)")]
THEMES = ("system", "light", "dark")
TABS = (("materials", "Materials"), ("calendar", "Calendar"), ("study", "Study"), ("guides", "Guides"),
        ("ask", "Ask"), ("offerings", "Past offerings"))
DEPTHS = (("quick", "Quick"), ("standard", "Standard"), ("from scratch", "From scratch"))


def course_color(course: dict[str, Any] | None) -> str:
    """The course's accent as a hex colour (a palette name, a hex value, or a stable pick from its id)."""
    c = ((course or {}).get("color") or "").strip()
    if c in PALETTE:
        return PALETTE[c]
    if HEX_RE.match(c):
        return c
    return PALETTE[PALETTE_NAMES[int((course or {}).get("id") or 0) % len(PALETTE_NAMES)]]


# ---------------------------------------------------------------- contrast
LIGHT_SURFACES = ("#ffffff", "#efefeb", "#f7f7f5")  # --surface, --surface-2, --bg (light theme)
DARK_SURFACES = ("#161615", "#1f1f1d", "#2a2a27")    # --bg, --surface, --surface-2 (dark theme)


def _rgb(hex_: str) -> tuple[int, int, int]:
    h = hex_.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def luminance(hex_: str) -> float:
    def chan(v: int) -> float:
        c = v / 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (chan(v) for v in _rgb(hex_))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: str, b: str) -> float:
    la, lb = luminance(a), luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def _mix(hex_: str, target: str, t: float) -> str:
    a, b = _rgb(hex_), _rgb(target)
    return "#%02x%02x%02x" % tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))


def ensure_contrast(hex_: str, surfaces: tuple[str, ...], ratio: float, toward: str) -> str:
    """The colour closest to hex_ (mixing toward black or white) that reaches `ratio` on every surface."""
    for step in range(0, 51):
        c = _mix(hex_, toward, step / 50)
        if all(contrast(c, bg) >= ratio for bg in surfaces):
            return c
    return toward


def accent_colors(hex_: str) -> dict[str, str]:
    """A course accent in four roles: the bar or dot (3:1 against each theme's surfaces, WCAG 1.4.11) and the
    text colour (4.5:1, WCAG 1.4.3), each for the light and for the dark theme."""
    return {"bar": ensure_contrast(hex_, LIGHT_SURFACES, 3.0, "#000000"),
            "bar_dark": ensure_contrast(hex_, DARK_SURFACES, 3.0, "#ffffff"),
            "text": ensure_contrast(hex_, LIGHT_SURFACES, 4.5, "#000000"),
            "text_dark": ensure_contrast(hex_, DARK_SURFACES, 4.5, "#ffffff")}


def accent_style(hex_: str) -> str:
    """Inline custom properties for an element that carries a course's accent colour."""
    c = accent_colors(hex_ if HEX_RE.match(hex_ or "") else PALETTE["blue"])
    return f"--accent-c: {c['bar']}; --accent-c-d: {c['bar_dark']}; --accent-t: {c['text']}; --accent-t-d: {c['text_dark']}"


def clean_color(value: str) -> str:
    value = (value or "").strip()
    return value if value in PALETTE or HEX_RE.match(value) else ""


def parse_day(value: str) -> date | None:
    try:
        return datetime.strptime((value or "")[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def day_label(value: str, today: date | None = None) -> str:
    d = parse_day(value)
    if not d:
        return value or ""
    today = today or date.today()
    delta = (d - today).days
    base = f"{d.strftime('%a')} {d.strftime('%b')} {d.day}"
    if delta == 0:
        return f"Today, {base}"
    if delta == 1:
        return f"Tomorrow, {base}"
    if 1 < delta < 7:
        return f"{base} (in {delta} days)"
    return base


def week_start(d: date) -> date:
    return d - timedelta(days=d.weekday())


def week_label(start: date) -> str:
    end = start + timedelta(days=6)
    return f"Week of {start.strftime('%b')} {start.day} to {end.strftime('%b')} {end.day}"


def time_label(value: str) -> str:
    try:
        t = datetime.strptime(value, "%H:%M")
    except ValueError:
        return value or ""
    return f"{t.hour % 12 or 12}:{t.minute:02d} {'AM' if t.hour < 12 else 'PM'}"


# ---------------------------------------------------------------- calendar and guides lists
_EVENT_LEAD = re.compile(r"^\s*\d{1,3}\s*[|:]\s*")


def clean_event_title(title: str) -> str:
    """Syllabus tables leak their first column into titles ('17 | Lecture 20'). Show the readable part."""
    t = re.sub(r"[*_`]+", "", title or "")
    t = re.sub(r"\s*\|\s*$", "", t)
    stripped = _EVENT_LEAD.sub("", t)
    if stripped.strip():
        t = stripped
    t = re.sub(r"\s*\|\s*", " - ", t)
    t = re.sub(r"\s+", " ", t).strip(" -:")
    return t or (title or "").strip()


def merge_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The same event listed twice (two copies of a syllabus) shows once, with a count. The first copy keeps its
    id so edit and delete still work; the others are listed in `copies`."""
    out: list[dict[str, Any]] = []
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for e in events:
        key = (e.get("date", ""), clean_event_title(e.get("title", "")).lower())
        row = by_key.get(key)
        if row is None:
            row = {**e, "title": clean_event_title(e.get("title", "")), "copies": [], "listed": 1}
            by_key[key] = row
            out.append(row)
            continue
        row["listed"] += 1
        row["copies"].append(e["id"])
        if row.get("kind") == "other" and e.get("kind") not in (None, "other"):
            row["kind"] = e["kind"]
        for field in ("time", "topics"):
            if not row.get(field) and e.get(field):
                row[field] = e[field]
    return out


GUIDE_ORDER = ("primer", "exam_guide", "concept_map", "explain", "practice", "missed")
ASK_KINDS = ("answer", "twin")


def group_guides(guides: list[dict[str, Any]]) -> dict[str, Any]:
    """Guides grouped by kind (newest first inside a group), with Q&A answers kept apart as 'ask history'.
    Titles that repeat inside a group get their date (and time, when the date repeats too)."""
    def label(g: dict[str, Any], dup_titles: dict[str, int], by_date: dict[tuple[str, str], int]) -> str:
        title = (g.get("title") or "Untitled").strip()
        if dup_titles.get(title, 0) < 2:
            return title
        day = (g.get("created_at") or "")[:10]
        if by_date.get((title, day), 0) > 1:
            return f"{title} ({day} {(g.get('created_at') or '')[11:16]})"
        return f"{title} ({day})"

    def prepare(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        dup: dict[str, int] = {}
        by_date: dict[tuple[str, str], int] = {}
        for g in items:
            t = (g.get("title") or "Untitled").strip()
            dup[t] = dup.get(t, 0) + 1
            by_date[(t, (g.get("created_at") or "")[:10])] = by_date.get((t, (g.get("created_at") or "")[:10]), 0) + 1
        return [{**g, "display_title": label(g, dup, by_date)} for g in items]

    ordered = sorted(guides, key=lambda g: (g.get("created_at") or "", g.get("id") or 0), reverse=True)
    groups: list[dict[str, Any]] = []
    kinds = [k for k in GUIDE_ORDER if any(g["kind"] == k for g in ordered)]
    kinds += sorted({g["kind"] for g in ordered} - set(GUIDE_ORDER) - set(ASK_KINDS))
    for k in kinds:
        groups.append({"kind": k, "items": prepare([g for g in ordered if g["kind"] == k])})
    ask = prepare([g for g in ordered if g["kind"] in ASK_KINDS])
    return {"groups": groups, "ask_history": ask, "total": len(ordered)}


def chunk_range_label(chunk: dict[str, Any]) -> str:
    """The location label for the reader. A time chunk shows its span, "1:35 to 3:10"; others show loc_label."""
    from . import locations
    label = chunk.get("loc_label") or ""
    if chunk.get("loc_kind") == "time":
        try:
            start, end = float(chunk["loc_start"]), float(chunk.get("loc_end") or "")
        except (TypeError, ValueError):
            return label
        if end > start:
            return f"{locations.time_label(start)} to {locations.time_label(end)}"
    return label


NOT_COVERED_NOTE = "Your saved materials don't cover this."
NOT_COVERED_EXTRA = " Parts below come from general knowledge."
_UID = itertools.count(1)
_LATEX_WORDS = {"lambda": "lambda", "Lambda": "Lambda", "alpha": "alpha", "beta": "beta", "gamma": "gamma",
                "Gamma": "Gamma", "delta": "delta", "Delta": "Delta", "mu": "mu", "sigma": "sigma", "Sigma": "Sigma",
                "pi": "pi", "theta": "theta", "epsilon": "epsilon", "phi": "phi", "omega": "omega", "rho": "rho",
                "cdot": "*", "times": "x", "infty": "infinity", "leq": "<=", "geq": ">=", "neq": "!=", "to": "->",
                "rightarrow": "->", "sum": "sum", "int": "integral", "sqrt": "sqrt", "det": "det", "in": "in"}


def plain_math(text: str) -> str:
    """A concept label spelled without LaTeX, for places that cannot render it (SVG text): "$A = X\\Lambda X^{-1}$"
    becomes "A = X Lambda X^-1"."""
    t = (text or "").replace("$", "")
    t = re.sub(r"\\(?:mathbf|mathrm|text|mathcal|operatorname)\{([^{}]*)\}", r"\1", t)
    t = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", t)
    t = re.sub(r"\\([A-Za-z]+)", lambda m: f" {_LATEX_WORDS.get(m.group(1), m.group(1))} ", t)
    t = re.sub(r"([_^])\{([^{}]*)\}", r"\1\2", t)
    t = t.replace("{", "").replace("}", "").replace("\\", " ")
    return re.sub(r"\s+", " ", t).strip()


def enrich_sources(store: Any, sources: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Saved source rows with a "past" label ("Past offering: Fall 2018") added for passages of a material saved as a
    past offering. Rows already carrying one are left alone; a missing material or store changes nothing."""
    rows = [dict(s) for s in (sources or [])]
    if store is None:
        return rows
    from . import offerings
    cache: dict[int, str] = {}
    for r in rows:
        if r.get("past") or not r.get("material_id"):
            continue
        mid = int(r["material_id"])
        if mid not in cache:
            try:
                cache[mid] = offerings.source_label(store.material(mid)) or ""
            except Exception:  # noqa: BLE001 - a label is a hint, never a reason to fail a page
                cache[mid] = ""
        if cache[mid]:
            r["past"] = cache[mid]
    return rows


_MISSING_RE = re.compile(r"No saved materials for: (.+?)\. (Add them to cover those topics\.)", re.S)


def _missing_as_list(md: str) -> str:
    """"No saved materials for: a; b; c. Add them ..." as an intro line plus one bullet per lecture."""
    def sub(m: re.Match) -> str:
        items = [x.strip() for x in m.group(1).split(";") if x.strip()]
        if len(items) < 2:
            return m.group(0)
        return "No saved materials for these lectures. " + m.group(2) + "\n\n" + "\n".join(f"- {x}" for x in items) + "\n"
    return _MISSING_RE.sub(sub, md)


_CONCEPT_LISTS_RE = re.compile(r"^#{1,6}\s*Concepts and links\s*#*\s*$", re.I | re.M)


def _merge_sources(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per distinct title and location, in source-number order. A row holds every number it stands for, so
    two cited passages from the same section show once with [2][5]."""
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for s in sorted(items, key=lambda x: x["n"]):
        title = s.get("title") or ""
        label = clean_label(title, s.get("label") or "")
        key = (title, label)
        row = rows.get(key)
        if row is None:
            rows[key] = {**s, "label": label, "ns": [{"n": s["n"], "href": s.get("href", "")}]}
        else:
            row["ns"].append({"n": s["n"], "href": s.get("href", "")})
    return list(rows.values())


ROUTING_FALLBACK = {"twin": "This looks like your graded work, so here is a similar problem with different numbers.",
                    "concepts": "This looks like your graded work, so this covers the ideas behind it."}


PARTLY_COVERED_NOTE = "Part of your question is not covered by your saved materials."
_INTRO_NOTE = re.compile(r"This was marked graded \(|This looks like your own draft for ")


def routing_note_src(data: dict[str, Any]) -> str:
    return str((data or {}).get("routing_note") or "")


def _list_only(md: str) -> bool:
    lines = [ln for ln in md.splitlines() if ln.strip()]
    return bool(lines) and all(re.match(r"^\s*(?:[-*+]|\d+[.)])\s+", ln) or ln.startswith((" ", "\t")) for ln in lines) \
        and bool(re.match(r"^\s*(?:[-*+]|\d+[.)])\s+", lines[0]))


def _merge_list_blocks(raw: list[dict[str, str]]) -> list[dict[str, str]]:
    """Neighbouring parts of one kind that are each only a list (an answer cut item by item so uncited ones could
    be labelled) go back to one list, so a screen reader hears one list and numbering carries on."""
    out: list[dict[str, str]] = []
    for b in raw:
        if out and out[-1]["kind"] == b["kind"] and _list_only(out[-1]["md"]) and _list_only(b["md"]):
            out[-1] = {"kind": b["kind"], "md": out[-1]["md"].rstrip() + "\n" + b["md"].strip()}
        else:
            out.append(dict(b))
    return out


def _head_twin(md: str) -> str:
    """The bold lines "Twin problem" and "Worked solution of the twin" become headings, so heading navigation finds them."""
    return re.sub(r"(?m)^\*\*(Twin problem|Worked solution of the twin)\*\*\s*$", r"## \1", md)


def _offers_of(data: dict[str, Any], course_id: int | None) -> list[dict[str, Any]]:
    """The one practice button a page offers (data["graded_offers"], at most one): 'Practice on a problem like Problem 3'."""
    if not course_id:
        return []
    out = [o for o in (data or {}).get("graded_offers") or [] if isinstance(o, dict) and o.get("label")]
    return out[:1]


UNUSED_SOURCES_MAX = 5


def grounding_topic(text: str) -> list[str]:
    from . import grounding
    return grounding.topic_stems(text)


MAP_BRANCH_DEPTH = 2   # a concept map text list nests at most this many branch levels; deeper concepts are a flat list
UNUSED_SHARE = 0.5   # an "also searched" passage must score at least this share of the best passage found


def _fair_unused(smap: dict[int, dict[str, Any]], cited: set[int], question: str = "") -> list[dict[str, Any]]:
    """The passages that were searched but not cited, worth listing: at most UNUSED_SOURCES_MAX, only those scoring
    within a fair share of the top hit when scores were saved. Without scores the first few stay."""
    rest = [smap[k] for k in sorted(smap) if k not in cited]
    scores = [float(s["score"]) for s in smap.values() if isinstance(s.get("score"), (int, float))]
    if scores and max(scores) > 0:
        floor = max(scores) * UNUSED_SHARE
        rest = [s for s in rest if not isinstance(s.get("score"), (int, float)) or float(s["score"]) >= floor]
    q = set(grounding_topic(question))
    if q:  # a passage that shares no real word with the question is noise (rows saved without words stay)
        need = 2 if len(q) >= 4 else 1
        rest = [s for s in rest if not s.get("words") or len(q & set(str(s["words"]).split())) >= need]
    return rest[:UNUSED_SOURCES_MAX]


def _join_headings(raw: list[dict[str, str]]) -> list[dict[str, str]]:
    """A block that is only a heading goes into the block after it, so no group on the page holds a heading and nothing else."""
    out: list[dict[str, str]] = []
    i = 0
    while i < len(raw):
        b = raw[i]
        lines = [ln for ln in b["md"].splitlines() if ln.strip()]
        if b["kind"] == "text" and len(lines) == 1 and lines[0].lstrip().startswith("#") and i + 1 < len(raw) \
                and raw[i + 1]["kind"] == "text":
            out.append({"kind": "text", "md": b["md"].rstrip() + "\n\n" + raw[i + 1]["md"].lstrip()})
            i += 2
            continue
        out.append(b)
        i += 1
    return out


_HEADING_MAX = 80


def answer_heading(question: str, data: dict[str, Any] | None) -> str:
    """The short title for an answer: "Feedback on your draft for Essay 1" for draft feedback, else the question's first
    line cut to about 80 characters at a word."""
    data = data or {}
    if data.get("feedback"):
        name = str(data.get("graded_title") or "").strip()
        return f"Feedback on your draft for {name}" if name else "Feedback on your draft"
    q = " ".join(str(question or "").split())
    if len(q) <= _HEADING_MAX:
        return q
    cut = q[:_HEADING_MAX - 3].rsplit(" ", 1)[0].rstrip(" ,;:-") or q[:_HEADING_MAX - 3]
    return cut + "..."


def _drop_repeated_note(raw: list[dict[str, str]], note: str) -> list[dict[str, str]]:
    """The routing sentence is said once, in the page's note: the same sentence at the top of the first block goes."""
    if not note or not raw or raw[0]["kind"] != "text":
        return raw
    md = raw[0]["md"]
    for sent in (x.strip() for x in re.split(r"(?<=[.!?])\s+", note.strip()) if len(x.strip()) > 20):
        md = re.sub(r"_?\*{0,2}" + re.escape(sent) + r"\*{0,2}_?[ \t]*", "", md, count=1, flags=re.I)
    md = md.strip()
    if md == raw[0]["md"].strip():
        return raw
    return ([{"kind": "text", "md": md}] if md else []) + raw[1:]


def answer_view(body_md: str, sources: list[dict[str, Any]] | None, *, blocks: list[Any] | None = None,
                twin: bool = False, ai_app: bool = False, mode: str | None = None,
                data: dict[str, Any] | None = None, base_level: int = 4, store: Any = None,
                course_id: int | None = None, question: str = "", tree_shown: bool = False,
                ask_normally: bool = True) -> dict[str, Any]:
    """Everything the answer partial needs, from live grounding output or from a saved guide/chat message.

    Banners come only from mode ("twin", "concepts"); a saved guide with no mode falls back to its kind (twin).
    A twin or concepts page shows its routing sentence (data["routing_note"]) and an "Ask normally" button instead of
    the not-covered banner. The not-covered banner shows only when nothing is cited or most of the answer is outside.
    Sources: view["sources"] holds only the ones the text cites, one row per distinct passage, in number order, each
    with its [n] marks and, for a past offering, its "Past offering: Fall 2018" tag; view["unused_sources"] holds the
    rest (a collapsed "Also searched (not cited)" list). tree_shown drops the flat "Concepts and links" lists of a
    concept map, since the tree on the page reads the same graph.
    base_level is the heading level the shallowest heading of the text gets (the container's heading level + 1)."""
    from . import grounding

    sources = enrich_sources(store, sources)
    smap = source_map(sources)
    raw: list[dict[str, str]] = []
    for b in blocks or []:
        kind = getattr(b, "kind", None) or (b.get("kind") if isinstance(b, dict) else "text")
        md = getattr(b, "md", None) or (b.get("md") if isinstance(b, dict) else "")
        if md and str(md).strip():
            raw.append({"kind": "outside" if kind == "outside" else "text", "md": str(md)})
    if not raw:
        raw = split_outside(body_md or "")
    data = data or {}
    if tree_shown:
        for i, b in enumerate(raw):  # the flat list is the same graph the tree shows
            if _CONCEPT_LISTS_RE.search(b["md"]):
                head = _CONCEPT_LISTS_RE.search(b["md"])
                keep = b["md"][:head.start()].strip()
                raw = raw[:i] + ([{"kind": b["kind"], "md": keep}] if keep else [])
                break
    if mode in ("twin", "concepts"):
        is_twin, is_concepts = mode == "twin", mode == "concepts"
    else:
        is_twin, is_concepts = bool(twin), False
    routed = is_twin or is_concepts
    dropped_note = ""
    if routed:  # the page says why once (the routing note); the intro block that said it again goes
        for i, b in enumerate(raw[:1]):
            m_intro = _INTRO_NOTE.match(b["md"].strip())
            if b["kind"] == "text" and m_intro:
                sents = re.split(r"(?<=[.!?])\s+", b["md"].strip())
                dropped_note = " ".join(x for x in sents[1:] if x.lower() not in (routing_note_src(data) or "").lower())
                raw = raw[1:]
    if routed:
        raw = _drop_repeated_note(raw, routing_note_src(data))
    feedback = bool(data.get("feedback"))
    worked = grounding.worked_flags([b["md"] for b in raw])
    for b, in_worked in zip(raw, worked):
        if b["kind"] == "text":
            b["md"] = grounding.unmark_gaps(b["md"])  # "the sources do not say X" is never an outside-knowledge tag
        elif grounding.is_non_claim(b["md"], calc=is_twin or in_worked, feedback=feedback):
            b["kind"] = "text"  # advice, questions, honest gaps and a worked example's own arithmetic are not claims
    raw = _merge_list_blocks(raw)
    for b in raw:
        if b["kind"] == "text":
            b["md"] = _missing_as_list(b["md"])
    if is_twin:
        for b in raw:
            b["md"] = _head_twin(b["md"])
    raw = _join_headings(raw)
    top = shallowest_heading(*[b["md"] for b in raw])
    uid = next(_UID)
    out_blocks: list[dict[str, Any]] = []
    numbering: dict[str, Any] = {}
    for b in raw:
        html = render(b["md"], smap, base_level=base_level, shallowest=top, numbering=numbering)
        if b["kind"] == "outside" and out_blocks and out_blocks[-1]["kind"] == "outside":
            out_blocks[-1]["html"] = type(html)(f"{out_blocks[-1]['html']}{html}")  # one box for a run of outside parts
        else:
            out_blocks.append({"kind": b["kind"], "html": html})
    n_out = 0
    for b in out_blocks:
        if b["kind"] == "outside":
            n_out += 1
            b["id"] = f"out-{uid}-{n_out}"
    cited = {k for b in raw if b["kind"] == "text" for k in grounding.cited_numbers(b["md"])}
    if tree_shown:  # the map's own nodes and links cite sources too
        for key in ("nodes", "edges"):
            for item in data.get(key) or []:
                cited |= {c for c in _cite_list(item.get("cites") if isinstance(item, dict) else None)}
    removed = sum(count_marks(b["md"])[0] for b in raw)
    unverified = sum(count_marks(b["md"])[1] for b in raw)
    gb = [grounding.Block(b["kind"], b["md"], grounding.cited_numbers(b["md"]) if b["kind"] == "text" else [])
          for b in raw]
    any_cited = any(b.cites for b in gb)
    # the saved flag counts only when nothing is cited; a cited answer that ends "your materials don't list more" is covered
    not_covered = bool(raw) and (grounding.is_not_covered(gb, len(sources)) or (bool(data.get("not_covered")) and not any_cited))
    has_outside = any(b["kind"] == "outside" for b in raw)
    partly = any_cited and any(b["kind"] == "text" and (grounding.has_uncited_gap(b["md"])
                                                         or (not grounding.cited_numbers(b["md"]) and grounding.says_gap(b["md"])))
                               for b in raw)
    routing_note = str(data.get("routing_note") or "").strip() if routed else ""
    if routed and not routing_note:
        routing_note = ROUTING_FALLBACK["twin" if is_twin else "concepts"]
    routing_note = (routing_note + " " + dropped_note).strip() if routed and dropped_note else routing_note
    parts = []
    if removed:
        parts.append(f"{removed} citation{'s' if removed != 1 else ''} removed")
    if unverified:
        parts.append(f"{unverified} quote{'s' if unverified != 1 else ''} not found")
    ask_normal = None
    if routed and ask_normally and course_id and (question or data.get("question")):
        ask_normal = {"course_id": int(course_id), "question": question or str(data.get("question"))}
    return {"blocks": out_blocks,
            "sources": _merge_sources([smap[k] for k in sorted(smap) if k in cited]),
            "unused_sources": _merge_sources(_fair_unused(smap, cited, question or str(data.get("question") or ""))),
            "twin": is_twin, "concepts": is_concepts,
            "routing_note": routing_note, "ask_normal": ask_normal,
            "offers": _offers_of(data, course_id), "offer_course_id": course_id,
            "outside": has_outside, "not_covered": not_covered and not routed,
            "partly_covered": bool(partly and not routed and not not_covered),  # one banner, not two
            "heading": answer_heading(question or str(data.get("question") or ""), data), "partly_covered_note": PARTLY_COVERED_NOTE,
            "not_covered_note": NOT_COVERED_NOTE + (NOT_COVERED_EXTRA if has_outside else ""),
            "removed_citations": removed, "unverified_quotes": unverified, "base_level": base_level,
            "check_line": ("Check: " + ", ".join(parts)) if parts else "",
            "footer": disclaimer.AI_APP_FOOTER if ai_app else disclaimer.FOOTER}


def guide_view(guide: dict[str, Any], base_level: int = 2, *, store: Any = None, tree_shown: bool = False,
               ask_normally: bool = True) -> dict[str, Any]:
    data = guide.get("data") if isinstance(guide.get("data"), dict) else {}
    saved = data.get("blocks") if isinstance(data.get("blocks"), list) else None
    mode = data.get("mode") if data.get("mode") in ("answer", "twin", "concepts") else None
    view = answer_view(guide.get("body_md", ""), guide.get("sources"), blocks=saved,
                       twin=guide.get("kind") == "twin", ai_app=guide.get("created_by") == "ai_app",
                       mode=mode, data=data, base_level=base_level, store=store,
                       course_id=guide.get("course_id"), question=guide.get("question") or "", tree_shown=tree_shown,
                       ask_normally=ask_normally)
    if guide.get("kind") not in ("answer", "twin"):  # a primer or a guide was not asked as a question
        what = {"primer": "this primer", "missed": "this catch-up", "exam_guide": "this guide"}.get(guide.get("kind"), "this guide")
        view["partly_covered_note"] = f"Part of {what} is not covered by your saved materials."
    return view


def _wrap(text: str, width: int) -> list[str]:
    """Word-wrap without ever cutting words short (a very long word is split)."""
    lines: list[str] = []
    cur = ""
    for word in text.split():
        while len(word) > width:
            if cur:
                lines.append(cur)
                cur = ""
            lines.append(word[:width])
            word = word[width:]
        if cur and len(cur) + 1 + len(word) > width:
            lines.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    if cur:
        lines.append(cur)
    return lines or [""]


_CITE_IN_LABEL = re.compile(r"\s*\[(\d{1,3})\]")


def _cite_list(value: Any) -> list[int]:
    if isinstance(value, int) and not isinstance(value, bool):
        return [value]
    return [c for c in (value or []) if isinstance(c, int) and not isinstance(c, bool)] if isinstance(value, list) else []


def concept_graph(data: dict[str, Any]) -> dict[str, Any]:
    """A concept map for the page ({nodes:[{id,label,cites}], edges:[{a,b,label,cites}]}).

    Returns a text tree (every node once, each edge once with its own cites), a list of cross links (the edges the
    tree could not show), and a layered SVG layout with arrows, edge labels and full, wrapped node labels."""
    nodes = {str(n.get("id")): {"id": str(n.get("id")), "label": str(n.get("label") or n.get("id")),
                                "cites": _cite_list(n.get("cites")),
                                "removed": bool(n.get("cites_removed") or n.get("removed"))}
             for n in (data.get("nodes") or []) if isinstance(n, dict) and n.get("id") is not None}
    edges: list[dict[str, Any]] = []
    for e in (data.get("edges") or []):
        if not isinstance(e, dict) or str(e.get("a")) not in nodes or str(e.get("b")) not in nodes:
            continue
        label = str(e.get("label") or "")
        cites = _cite_list(e.get("cites")) + _cite_list(e.get("cite")) + [int(x) for x in _CITE_IN_LABEL.findall(label)]
        edges.append({"a": str(e["a"]), "b": str(e["b"]), "label": _CITE_IN_LABEL.sub("", label).strip(),
                      "cites": sorted(set(cites)), "removed": bool(e.get("cites_removed") or e.get("removed"))})
    children: dict[str, list[int]] = {k: [] for k in nodes}
    has_parent: set[str] = set()
    for i, e in enumerate(edges):
        children[e["a"]].append(i)
        if e["a"] != e["b"]:
            has_parent.add(e["b"])
    roots = [k for k in nodes if k not in has_parent] or list(nodes)[:1]
    depth: dict[str, int] = {r: 0 for r in roots}
    queue = list(roots)
    while queue:
        cur = queue.pop(0)
        for i in children[cur]:
            ch = edges[i]["b"]
            if ch not in depth:
                depth[ch] = depth[cur] + 1
                queue.append(ch)
    for k in nodes:  # unreachable (cycles): put them after the rest
        depth.setdefault(k, max(depth.values(), default=0) + 1)
    seen: set[str] = set()
    tree_edges: set[int] = set()

    def tree(node: str, via: dict[str, Any] | None) -> dict[str, Any]:
        seen.add(node)
        kids = []
        for i in children[node]:
            ch = edges[i]["b"]
            if ch not in seen:
                tree_edges.add(i)
                kids.append(tree(ch, edges[i]))
        return {**nodes[node], "via": (via or {}).get("label", ""), "via_cites": (via or {}).get("cites", []),
                "has_via": via is not None, "via_removed": (via or {}).get("removed", False), "children": kids}

    forest = [tree(r, None) for r in roots if r not in seen]
    forest += [tree(k, None) for k in nodes if k not in seen]

    def flatten(n: dict[str, Any], level: int = 1) -> None:
        """A concept with one link to the next, and that one to the next, is a chain: it reads as a flat numbered
        sequence, not as lists nested one inside the other. Only a real branch nests."""
        chain, cur = [], n
        while len(cur["children"]) == 1:
            cur = cur["children"][0]
            chain.append(cur)
        n["chain"], n["branches"], n["more"] = chain, cur["children"], []
        if level >= MAP_BRANCH_DEPTH and n["branches"]:
            # deep in a branching map: the rest of this subtree is one flat list, not more nested lists
            def walk(b: dict[str, Any]) -> None:
                n["more"].append(b)
                for c in b["children"]:
                    walk(c)
            for b in n["branches"]:
                walk(b)
            n["branches"] = []
            return
        for b in n["branches"]:
            flatten(b, level + 1)
    for root in forest:
        flatten(root)
    cross = [{"a": nodes[e["a"]]["label"], "b": nodes[e["b"]]["label"], "label": e["label"], "cites": e["cites"],
              "removed": e["removed"]}
             for i, e in enumerate(edges) if i not in tree_edges]

    # layout: columns by depth, boxes sized to their full wrapped label
    box_w, gap_x, gap_y, pad, line_h, wrap_at = 190, 150, 18, 14, 15, 26
    cols: dict[int, list[str]] = {}
    for k in nodes:
        cols.setdefault(depth[k], []).append(k)
    boxes: dict[str, dict[str, Any]] = {}
    col_h: dict[int, int] = {}
    for d, ids in cols.items():
        y = pad
        for k in ids:
            lines = _wrap(plain_math(nodes[k]["label"]), wrap_at)
            h = len(lines) * line_h + 18
            boxes[k] = {"x": pad + d * (box_w + gap_x), "y": y, "w": box_w, "h": h, "lines": lines,
                        "full": plain_math(nodes[k]["label"]), "ty": y + 9 + line_h - 3}
            y += h + gap_y
        col_h[d] = y
    width = pad * 2 + (max(cols, default=0) + 1) * box_w + max(cols, default=0) * gap_x
    height = max(col_h.values(), default=pad) + pad - gap_y
    shown_edges = []
    for e in edges:
        if e["a"] == e["b"]:
            continue
        a, b = boxes[e["a"]], boxes[e["b"]]
        if b["x"] > a["x"]:
            x1, y1, x2, y2 = a["x"] + a["w"], a["y"] + a["h"] // 2, b["x"], b["y"] + b["h"] // 2
        elif b["x"] < a["x"]:
            x1, y1, x2, y2 = a["x"], a["y"] + a["h"] // 2, b["x"] + b["w"], b["y"] + b["h"] // 2
        elif b["y"] > a["y"]:
            x1, y1, x2, y2 = a["x"] + a["w"] // 2, a["y"] + a["h"], b["x"] + b["w"] // 2, b["y"]
        else:
            x1, y1, x2, y2 = a["x"] + a["w"] // 2, a["y"], b["x"] + b["w"] // 2, b["y"] + b["h"]
        lab = _wrap(plain_math(e["label"]), 22) if e["label"] else []
        shown_edges.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "lines": lab,
                            "lx": (x1 + x2) // 2, "ly": (y1 + y2) // 2 - (len(lab) - 1) * 6 - 3})
    n_nodes = len(nodes)
    aria = (f"Concept map with {n_nodes} concept{'s' if n_nodes != 1 else ''} and {len(edges)} "
            f"link{'s' if len(edges) != 1 else ''}. The same concepts and links are in the text list above the diagram.")
    return {"forest": forest, "cross": cross, "boxes": list(boxes.values()), "edges": shown_edges,
            "width": width, "height": height, "empty": not nodes, "aria": aria}


# ---------------------------------------------------------------- search snippets
def snippet_text(text: str) -> str:
    """A materials-search snippet as plain text: links keep their words (a link cut off by the snippet's width is
    cut cleanly), emphasis, code ticks, math dollars and heading marks go, and a snippet that starts or ends in the
    middle of a word drops that word so it begins and ends on a whole one."""
    t = str(text or "")
    t = re.sub(r"!\[([^\]]*)\]\([^)\s]*\)?", r"\1", t)
    t = re.sub(r"\[([^\]]+)\]\([^)\s]*(?:\)|\.\.\.|$)", r"\1", t)   # a link, closed or cut off in its address
    t = re.sub(r"\[([^\]]+)\](?=\s|$)", r"\1", t)
    t = re.sub(r"\bhttps?://\S+", "", t)
    t = re.sub(r"(\*{1,3}|_{1,3})(?=\S)([^*_\n]+?)(?<=\S)\1", r"\2", t)
    t = re.sub(r"(?m)^\s*#{1,6}\s+", "", t)
    t = re.sub(r"[`$]", "", t)
    t = re.sub(r"(?<![\w*])\*+|\*+(?![\w*])", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    lead, trail = t.startswith("..."), t.endswith("...")
    core = t[3:].strip() if lead else t
    core = core[:-3].strip() if trail else core
    if lead and " " in core and not core[:1].isupper():
        core = core.split(" ", 1)[1]
    if trail and " " in core and not re.search(r"[.!?]$", core):
        core = core.rsplit(" ", 1)[0]
    return ("..." if lead else "") + core.strip() + ("..." if trail else "")
