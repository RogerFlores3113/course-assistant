"""A small, safe markdown renderer for AI answers and guides.

Everything is escaped first. Only these things become markup: headings, paragraphs, bullet and numbered lists
(nested by indent), block quotes, fenced and inline code, bold, italic, details/summary, links that point into the
app (/read/...) or to http(s) pages, math ($...$ kept as text in a span with class "math") and citation chips [n].
"""
from __future__ import annotations

import re
from html import escape
from typing import Any

from markupsafe import Markup

OUTSIDE_RE = re.compile(r"<outside>(.*?)</outside>", re.S | re.I)
FENCE_RE = re.compile(r"^\s*```")
HEAD_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
ITEM_RE = re.compile(r"^(\s*)([-*+]|\d+[.)])\s+(.*)$")
SUMMARY_RE = re.compile(r"^\s*<summary>(.*?)</summary>\s*$", re.I)
SAFE_URL_RE = re.compile(r"^(/read/\d+[^\s\"'<>]*|https?://[^\s\"'<>]+)$")


def safe_url(url: str) -> str | None:
    url = (url or "").strip()
    return url if SAFE_URL_RE.match(url) else None


def source_map(sources: list[dict[str, Any]] | None) -> dict[int, dict[str, Any]]:
    """Number -> {title, label, href}. Accepts retrieval Sources and saved guide sources."""
    out: dict[int, dict[str, Any]] = {}
    for i, s in enumerate(sources or [], start=1):
        try:
            n = int(s.get("n") or i)
        except (TypeError, ValueError):
            n = i
        out[n] = {"n": n, "title": s.get("title") or s.get("material_title") or "Material",
                  "label": s.get("label") or s.get("loc_label") or "", "href": s.get("href") or ""}
    return out


def inline(text: str, sources: dict[int, dict[str, Any]] | None = None) -> str:
    sources = sources or {}
    stash: list[str] = []

    def hold(html: str) -> str:
        stash.append(html)
        return f"\x00{len(stash) - 1}\x00"

    text = text.replace("\x00", "")
    text = re.sub(r"`([^`\n]+)`", lambda m: hold(f"<code>{escape(m.group(1))}</code>"), text)
    text = re.sub(r"(?<![\w$\\])\$(?=\S)([^$\n]+?)(?<=\S)\$(?![\w$])",
                  lambda m: hold(f'<span class="math">${escape(m.group(1))}$</span>'), text)

    def link(m: re.Match) -> str:
        url = safe_url(m.group(2))
        label = escape(m.group(1))
        if not url:
            return hold(label)
        ext = ' rel="noopener noreferrer"' if url.startswith("http") else ""
        return hold(f'<a href="{escape(url)}"{ext}>{label}</a>')

    text = re.sub(r"\[([^\]\n]+)\]\(([^)\s]+)\)", link, text)

    def cite(m: re.Match) -> str:
        n = int(m.group(1))
        s = sources.get(n)
        if not s or not s.get("href"):
            return m.group(0)
        aria = escape(f"Source {n}: {s['title']}" + (f", {s['label']}" if s["label"] else ""))
        return hold(f'<a class="cite" href="{escape(s["href"])}" aria-label="{aria}">[{n}]</a>')

    text = re.sub(r"\[(\d{1,3})\]", cite, text)
    text = escape(text, quote=False)
    text = re.sub(r"\*\*(?=\S)(.+?)(?<=\S)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<![\w*])\*(?=\S)([^*\n]+?)(?<=\S)\*(?![\w*])", r"<em>\1</em>", text)
    text = re.sub(r"(?<![\w_])_(?=\S)([^_\n]+?)(?<=\S)_(?![\w_])", r"<em>\1</em>", text)
    for _ in range(3):  # a stash entry can hold another (a link label with code)
        text = re.sub(r"\x00(\d+)\x00", lambda m: stash[int(m.group(1))], text)
    return text


def _list(lines: list[str], i: int, sources) -> tuple[str, int]:
    """Render the list starting at lines[i]; returns html and the next line index."""
    first = ITEM_RE.match(lines[i])
    base = len(first.group(1).expandtabs(4))
    ordered = first.group(2)[0].isdigit()
    tag = "ol" if ordered else "ul"
    out = [f"<{tag}>"]
    while i < len(lines):
        m = ITEM_RE.match(lines[i])
        if not m:
            break
        indent = len(m.group(1).expandtabs(4))
        if indent < base:
            break
        if indent > base:  # nested list inside the previous item
            sub, i = _list(lines, i, sources)
            if out[-1].endswith("</li>"):
                out[-1] = out[-1][:-5] + sub + "</li>"
            else:
                out.append(f"<li>{sub}</li>")
            continue
        out.append(f"<li>{inline(m.group(3), sources)}</li>")
        i += 1
    out.append(f"</{tag}>")
    return "".join(out), i


def _blocks(lines: list[str], sources, depth: int = 0) -> str:
    out: list[str] = []
    i = 0
    para: list[str] = []

    def flush() -> None:
        if para:
            out.append(f"<p>{inline(' '.join(p.strip() for p in para), sources)}</p>")
            para.clear()

    while i < len(lines):
        line = lines[i]
        if FENCE_RE.match(line):
            flush()
            i += 1
            code = []
            while i < len(lines) and not FENCE_RE.match(lines[i]):
                code.append(lines[i])
                i += 1
            i += 1
            out.append(f"<pre><code>{escape(chr(10).join(code))}</code></pre>")
            continue
        if line.strip().lower() == "<details>" and depth < 4:
            flush()
            i += 1
            inner, nest = [], 1
            while i < len(lines):
                low = lines[i].strip().lower()
                if low == "<details>":
                    nest += 1
                elif low == "</details>":
                    nest -= 1
                    if nest == 0:
                        break
                inner.append(lines[i])
                i += 1
            i += 1
            summary = "More"
            if inner and (sm := SUMMARY_RE.match(inner[0])):
                summary = sm.group(1)
                inner = inner[1:]
            out.append(f"<details><summary>{inline(summary, sources)}</summary>{_blocks(inner, sources, depth + 1)}</details>")
            continue
        if not line.strip():
            flush()
            i += 1
            continue
        if (h := HEAD_RE.match(line)):
            flush()
            level = min(6, len(h.group(1)) + 2)  # page already has h1 and h2
            out.append(f"<h{level}>{inline(h.group(2), sources)}</h{level}>")
            i += 1
            continue
        if ITEM_RE.match(line):
            flush()
            html, i = _list(lines, i, sources)
            out.append(html)
            continue
        if line.lstrip().startswith(">"):
            flush()
            quote = []
            while i < len(lines) and lines[i].lstrip().startswith(">"):
                quote.append(lines[i].lstrip()[1:].lstrip())
                i += 1
            out.append(f"<blockquote>{_blocks(quote, sources, depth + 1)}</blockquote>")
            continue
        if re.match(r"^\s*(-{3,}|\*{3,})\s*$", line):
            flush()
            out.append("<hr>")
            i += 1
            continue
        para.append(line)
        i += 1
    flush()
    return "".join(out)


def render(md: str, sources: list[dict[str, Any]] | dict[int, dict[str, Any]] | None = None) -> Markup:
    smap = sources if isinstance(sources, dict) else source_map(sources)
    return Markup(_blocks((md or "").replace("\r\n", "\n").split("\n"), smap))


def split_outside(md: str) -> list[dict[str, str]]:
    """Cut stored markdown into [{kind: text|outside, md}] at <outside>...</outside> markers."""
    blocks: list[dict[str, str]] = []
    pos = 0
    for m in OUTSIDE_RE.finditer(md or ""):
        if md[pos:m.start()].strip():
            blocks.append({"kind": "text", "md": md[pos:m.start()].strip()})
        if m.group(1).strip():
            blocks.append({"kind": "outside", "md": m.group(1).strip()})
        pos = m.end()
    if (md or "")[pos:].strip():
        blocks.append({"kind": "text", "md": md[pos:].strip()})
    return blocks
