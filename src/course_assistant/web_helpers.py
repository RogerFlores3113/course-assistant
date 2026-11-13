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
