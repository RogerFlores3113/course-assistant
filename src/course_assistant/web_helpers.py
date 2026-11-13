"""Small helpers for the web pages: dates, course colours, answer views, concept map layout."""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Any

from . import disclaimer
from .web_md import count_marks, render, source_map, split_outside

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


NOT_COVERED_NOTE = "Your saved materials don't cover this well. Parts below come from general knowledge."


def answer_view(body_md: str, sources: list[dict[str, Any]] | None, *, blocks: list[Any] | None = None,
                twin: bool = False, ai_app: bool = False, mode: str | None = None,
                data: dict[str, Any] | None = None) -> dict[str, Any]:
    """Everything the answer partial needs, from live grounding output or from a saved guide/chat message.

    Sources: view["sources"] holds only the ones the text cites (they are listed under "Sources");
    view["unused_sources"] holds the rest (a collapsed "Also searched (not cited)" list)."""
    from . import grounding

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
    cited = {k for b in raw if b["kind"] == "text" for k in grounding.cited_numbers(b["md"])}
    removed = sum(count_marks(b["md"])[0] for b in raw)
    unverified = sum(count_marks(b["md"])[1] for b in raw)
    gb = [grounding.Block(b["kind"], b["md"], grounding.cited_numbers(b["md"]) if b["kind"] == "text" else [])
          for b in raw]
    data = data or {}
    not_covered = bool(raw) and (bool(data.get("not_covered")) or grounding.is_not_covered(gb, len(sources)))
    return {"blocks": out_blocks,
            "sources": [smap[k] for k in sorted(smap) if k in cited],
            "unused_sources": [smap[k] for k in sorted(smap) if k not in cited],
            "twin": twin or mode == "twin", "concepts": mode == "concepts",
            "outside": any(b["kind"] == "outside" for b in raw), "not_covered": not_covered,
            "not_covered_note": NOT_COVERED_NOTE, "removed_citations": removed, "unverified_quotes": unverified,
            "footer": disclaimer.AI_APP_FOOTER if ai_app else disclaimer.FOOTER}


def guide_view(guide: dict[str, Any]) -> dict[str, Any]:
    data = guide.get("data") if isinstance(guide.get("data"), dict) else {}
    saved = data.get("blocks") if isinstance(data.get("blocks"), list) else None
    mode = data.get("mode") if data.get("mode") in ("answer", "twin", "concepts") else None
    return answer_view(guide.get("body_md", ""), guide.get("sources"), blocks=saved,
                       twin=guide.get("kind") == "twin", ai_app=guide.get("created_by") == "ai_app",
                       mode=mode, data=data)


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
                                "cites": _cite_list(n.get("cites"))}
             for n in (data.get("nodes") or []) if isinstance(n, dict) and n.get("id") is not None}
    edges: list[dict[str, Any]] = []
    for e in (data.get("edges") or []):
        if not isinstance(e, dict) or str(e.get("a")) not in nodes or str(e.get("b")) not in nodes:
            continue
        label = str(e.get("label") or "")
        cites = _cite_list(e.get("cites")) + _cite_list(e.get("cite")) + [int(x) for x in _CITE_IN_LABEL.findall(label)]
        edges.append({"a": str(e["a"]), "b": str(e["b"]), "label": _CITE_IN_LABEL.sub("", label).strip(),
                      "cites": sorted(set(cites))})
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
                "has_via": via is not None, "children": kids}

    forest = [tree(r, None) for r in roots if r not in seen]
    forest += [tree(k, None) for k in nodes if k not in seen]
    cross = [{"a": nodes[e["a"]]["label"], "b": nodes[e["b"]]["label"], "label": e["label"], "cites": e["cites"]}
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
            lines = _wrap(nodes[k]["label"], wrap_at)
            h = len(lines) * line_h + 18
            boxes[k] = {"x": pad + d * (box_w + gap_x), "y": y, "w": box_w, "h": h, "lines": lines,
                        "full": nodes[k]["label"], "ty": y + 9 + line_h - 3}
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
        lab = _wrap(e["label"], 22) if e["label"] else []
        shown_edges.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "lines": lab,
                            "lx": (x1 + x2) // 2, "ly": (y1 + y2) // 2 - (len(lab) - 1) * 6 - 3})
    n_nodes = len(nodes)
    aria = (f"Concept map with {n_nodes} concept{'s' if n_nodes != 1 else ''} and {len(edges)} "
            f"link{'s' if len(edges) != 1 else ''}. The same concepts and links are in the text list above the diagram.")
    return {"forest": forest, "cross": cross, "boxes": list(boxes.values()), "edges": shown_edges,
            "width": width, "height": height, "empty": not nodes, "aria": aria}
