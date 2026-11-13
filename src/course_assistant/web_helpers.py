"""Small helpers for the web pages: dates, course colours, answer views, concept map layout."""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Any

from . import disclaimer
from .web_md import render, source_map, split_outside

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


def answer_view(body_md: str, sources: list[dict[str, Any]] | None, *, blocks: list[Any] | None = None,
                twin: bool = False, ai_app: bool = False) -> dict[str, Any]:
    """Everything the answer partial needs, from live grounding output or from a saved guide/chat message."""
    sources = list(sources or [])
    smap = source_map(sources)
    raw: list[dict[str, str]] = []
    for b in blocks or []:
        kind = getattr(b, "kind", None) or (b.get("kind") if isinstance(b, dict) else "text")
        md = getattr(b, "md", None) or (b.get("md") if isinstance(b, dict) else "")
        if md and str(md).strip():
            raw.append({"kind": "outside" if kind == "outside" else "text", "md": str(md)})
    if not raw:
        raw = split_outside(body_md or "")
    out_blocks = [{"kind": b["kind"], "html": render(b["md"], smap)} for b in raw]
    return {"blocks": out_blocks, "sources": [smap[k] for k in sorted(smap)], "twin": twin,
            "outside": any(b["kind"] == "outside" for b in raw),
            "footer": disclaimer.AI_APP_FOOTER if ai_app else disclaimer.FOOTER}


def guide_view(guide: dict[str, Any]) -> dict[str, Any]:
    data = guide.get("data") if isinstance(guide.get("data"), dict) else {}
    saved = data.get("blocks") if isinstance(data.get("blocks"), list) else None
    return answer_view(guide.get("body_md", ""), guide.get("sources"), blocks=saved,
                       twin=guide.get("kind") == "twin", ai_app=guide.get("created_by") == "ai_app")


def concept_graph(data: dict[str, Any]) -> dict[str, Any]:
    """Nested list tree and a layered SVG layout for a concept map ({nodes:[{id,label,cites}], edges:[{a,b,label}]})."""
    nodes = {str(n.get("id")): {"id": str(n.get("id")), "label": str(n.get("label") or n.get("id")),
                                "cites": [c for c in (n.get("cites") or []) if isinstance(c, int)]}
             for n in (data.get("nodes") or []) if isinstance(n, dict) and n.get("id") is not None}
    edges = [(str(e.get("a")), str(e.get("b")), str(e.get("label") or ""))
             for e in (data.get("edges") or []) if isinstance(e, dict) and str(e.get("a")) in nodes and str(e.get("b")) in nodes]
    children: dict[str, list[tuple[str, str]]] = {k: [] for k in nodes}
    has_parent: set[str] = set()
    for a, b, lab in edges:
        children[a].append((b, lab))
        has_parent.add(b)
    roots = [k for k in nodes if k not in has_parent] or list(nodes)[:1]
    # depth = shortest distance from any root, for the SVG columns
    depth: dict[str, int] = {r: 0 for r in roots}
    queue = list(roots)
    while queue:
        cur = queue.pop(0)
        for ch, _ in children[cur]:
            if ch not in depth:
                depth[ch] = depth[cur] + 1
                queue.append(ch)
    for k in nodes:  # unreachable (cycles): put them after the rest
        depth.setdefault(k, max(depth.values(), default=0) + 1)
    seen: set[str] = set()

    def tree(node: str, via: str) -> dict[str, Any]:
        seen.add(node)
        kids = [tree(ch, lab) for ch, lab in children[node] if ch not in seen]
        return {**nodes[node], "via": via, "children": kids}

    forest = [tree(r, "") for r in roots if r not in seen]
    forest += [tree(k, "") for k in nodes if k not in seen]
    # layout
    cols: dict[int, list[str]] = {}
    for k in nodes:
        cols.setdefault(depth[k], []).append(k)
    box_w, box_h, gap_x, gap_y, pad = 170, 40, 60, 18, 12
    pos: dict[str, tuple[int, int]] = {}
    for d, ids in cols.items():
        for row, k in enumerate(ids):
            pos[k] = (pad + d * (box_w + gap_x), pad + row * (box_h + gap_y))
    width = pad * 2 + (max(cols, default=0) + 1) * box_w + max(cols, default=0) * gap_x
    height = pad * 2 + max((len(v) for v in cols.values()), default=1) * (box_h + gap_y)
    shown = []
    for k, (x, y) in pos.items():
        label = nodes[k]["label"]
        shown.append({"x": x, "y": y, "w": box_w, "h": box_h, "label": label if len(label) <= 26 else label[:25] + "...",
                      "full": label})
    lines = [{"x1": pos[a][0] + box_w, "y1": pos[a][1] + box_h // 2, "x2": pos[b][0], "y2": pos[b][1] + box_h // 2}
             for a, b, _ in edges if a in pos and b in pos and a != b]
    return {"forest": forest, "boxes": shown, "lines": lines, "width": width, "height": height,
            "empty": not nodes}
