"""A small, safe markdown renderer for AI answers, guides and the reader.

Everything is escaped first. Only these things become markup: headings (# is an h2 and ## an h3 under the page's h1;
heading_offset moves them), paragraphs, bullet and numbered lists (nested by indent), pipe tables, block quotes,
fenced code (indentation kept, class "language-x") and inline code, bold, italic, details/summary (one line or
several), links that point into the app (/read/...) or to http(s) pages, math and citation chips [n].

Math: $...$ and $$...$$ are kept as plain escaped text inside span.math / div.math-block, untouched by bold and
italic rules, and static/content.js turns them into KaTeX (with MathML for screen readers). Without JS the LaTeX
shows as text. Markers written by grounding become: a "(citation removed)" note where an out-of-range [n] was
dropped, and a flag on a quotation the cited source does not contain.
"""
from __future__ import annotations

import re
from html import escape
from typing import Any

from markupsafe import Markup

from .grounding import (OUTSIDE_CLOSE, OUTSIDE_OPEN, QUOTE_CLOSE, QUOTE_FLAG_TEXT, QUOTE_OPEN, REMOVED_MARK,
                        strip_past_words)

REMOVED_LABEL = "a citation here pointed to a source that doesn't exist and was removed"
OUTSIDE_RE = re.compile(r"<outside>(.*?)</outside>", re.S | re.I)
FENCE_RE = re.compile(r"^(\s*)(```|~~~)\s*([\w+#.-]*)[^\n]*$")
HEAD_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
ITEM_RE = re.compile(r"^(\s*)([-*+]|\d+[.)])\s+(.*)$")
SUMMARY_RE = re.compile(r"^\s*<summary>(.*?)</summary>\s*$", re.I)
DETAILS_OPEN_RE = re.compile(r"^\s*<details(?:\s[^>]*)?>\s*$", re.I)
DETAILS_CLOSE_RE = re.compile(r"^\s*</details>\s*$", re.I)
TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)*\|?\s*$")
SAFE_URL_RE = re.compile(r"^(/read/\d+[^\s\"'<>]*|https?://[^\s\"'<>]+)$")
_TAG_SPLIT = re.compile(r"(?i)\s*(<details(?:\s[^>]*)?>|</details>|<summary>.*?</summary>)\s*")


def safe_url(url: str) -> str | None:
    url = (url or "").strip()
    return url if SAFE_URL_RE.match(url) else None


def clean_label(title: str, label: str) -> str:
    """The location label to show after a title: empty when the title already says it ("Diagonalization" and
    "Diagonalization"). One rule for the citation chips and the Sources list so both read the same."""
    label, title = (label or "").strip(), (title or "").strip()
    if label and (label == title or title.endswith(", " + label) or title.endswith(": " + label)):
        return ""
    return label


def source_map(sources: list[dict[str, Any]] | None) -> dict[int, dict[str, Any]]:
    """Number -> {title, label, href}. Accepts retrieval Sources and saved guide sources."""
    out: dict[int, dict[str, Any]] = {}
    for i, s in enumerate(sources or [], start=1):
        try:
            n = int(s.get("n") or i)
        except (TypeError, ValueError):
            n = i
        title = s.get("title") or s.get("material_title") or "Material"
        has_tag = bool(s.get("past"))
        title = strip_past_words(title, has_tag)
        label = strip_past_words(s.get("label") or s.get("loc_label") or "", has_tag)  # the label may repeat the title and "(past offering)"
        out[n] = {"n": n, "title": title, "label": clean_label(title, label),
                  "href": s.get("href") or "", "past": s.get("past") or "", "score": s.get("score"),
                  "words": s.get("words") or ""}
    return out


def inline(text: str, sources: dict[int, dict[str, Any]] | None = None) -> str:
    sources = sources or {}
    stash: list[str] = []

    def hold(html: str) -> str:
        stash.append(html)
        return f"\x00{len(stash) - 1}\x00"

    text = text.replace("\x00", "")
    text = text.replace(REMOVED_MARK, hold(
        f'<span class="cite-removed" title="{escape(REMOVED_LABEL)}">(citation removed)</span>'))
    text = text.replace(OUTSIDE_OPEN, hold('<span class="outside-inline">'))
    text = text.replace(OUTSIDE_CLOSE, hold(
        '</span><span class="oi-tag" title="Not from your materials">outside'
        '<span class="visually-hidden"> (not from your materials)</span></span>'))
    text = text.replace(QUOTE_OPEN, hold('<mark class="quote-unverified">'))
    text = text.replace(QUOTE_CLOSE, hold(f'</mark> <span class="quote-flag">({QUOTE_FLAG_TEXT})</span>'))
    text = re.sub(r"`([^`\n]+)`", lambda m: hold(f"<code>{escape(m.group(1))}</code>"), text)
    text = re.sub(r"\$\$(.+?)\$\$", lambda m: hold(f'<span class="math">$${escape(m.group(1))}$$</span>'), text)
    # sub/superscript-only math inside a word, as in the formula C$_6$H$_{12}$O$_6$
    text = re.sub(r"\$([_^][^$\n]*)\$", lambda m: hold(f'<span class="math">${escape(m.group(1))}$</span>'), text)
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
        before = m.string[m.start() - 1] if m.start() else ""
        after = m.string[m.end()] if m.end() < len(m.string) else ""
        if before == "[" or after == "]":  # part of a matrix or list such as [[1],[2]], not a citation
            return m.group(0)
        if re.search(r"[=≈∈]\s*$", m.string[max(0, m.start() - 6):m.start()]):  # x = [2]: a value
            return m.group(0)
        n = int(m.group(1))
        s = sources.get(n)
        if not s or not s.get("href"):
            return m.group(0)
        aria = escape(f"Source {n}: {s['title']}" + (f", {s['label']}" if s["label"] else "")
                      + (f", {s['past']}" if s.get("past") else ""))
        return hold(f'<a class="cite" href="{escape(s["href"])}" aria-label="{aria}">[{n}]</a>')

    text = re.sub(r"\[(\d{1,3})\]", cite, text)
    text = re.sub(r"\\([*_`$\[\]#])", lambda m: hold(escape(m.group(1))), text)  # \* and friends are literal
    text = escape(text, quote=False)
    text = re.sub(r"\*\*\*(?=\S)(.+?)(?<=\S)\*\*\*", r"<strong><em>\1</em></strong>", text)
    text = re.sub(r"\*\*(?=\S)(.+?)(?<=\S)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<![\w])__(?=\S)(.+?)(?<=\S)__(?![\w])", r"<strong>\1</strong>", text)
    # textbook variables written *C*1*V*1: one or two letters between asterisks, digits may touch either side
    text = re.sub(r"(?<![A-Za-z_*])\*([A-Za-z]{1,2})\*(?![A-Za-z*])", r"<em>\1</em>", text)
    # a closing * may touch a digit or letter: *V*1 is V followed by 1 (textbook variable names)
    text = re.sub(r"(?<![\w*])\*(?=[^\s*])([^*\n]+?)(?<=[^\s*])\*(?!\*)", r"<em>\1</em>", text)
    text = re.sub(r"(?<![\w_])_(?=\S)([^_\n]+?)(?<=\S)_(?![\w_])", r"<em>\1</em>", text)
    for _ in range(3):  # a stash entry can hold another (a link label with code)
        text = re.sub(r"\x00(\d+)\x00", lambda m: stash[int(m.group(1))], text)
    return text


def _next_item(lines: list[str], i: int) -> int | None:
    """Index of the next non-blank line after i when only blank lines separate it from i, else None."""
    j = i
    while j < len(lines) and not lines[j].strip():
        j += 1
    return j if j > i and j < len(lines) else None


def _list(lines: list[str], i: int, sources, state: dict | None = None) -> tuple[str, int]:
    """Render the list starting at lines[i]; returns html and the next line index.
    Items of one kind separated by blank lines are one loose list. A numbered list keeps the number it starts at
    (start="5") and, when it starts at 1 right after another numbered list (a details block in between), carries
    the running count on through state["next"]."""
    first = ITEM_RE.match(lines[i])
    base = len(first.group(1).expandtabs(4))
    ordered = first.group(2)[0].isdigit()
    tag = "ol" if ordered else "ul"
    start_no = 1
    if ordered:
        start_no = int(re.match(r"\d+", first.group(2)).group(0))
        if state is not None and start_no == 1 and state.get("next"):
            start_no = state["next"]
    out = [f"<{tag}" + (f' start="{start_no}"' if ordered and start_no != 1 else "") + ">"]
    count = 0
    while i < len(lines):
        m = ITEM_RE.match(lines[i])
        if not m:
            nxt = _next_item(lines, i) if not lines[i].strip() else None
            nm = ITEM_RE.match(lines[nxt]) if nxt is not None else None
            if nm and len(nm.group(1).expandtabs(4)) >= base and \
                    (len(nm.group(1).expandtabs(4)) > base or nm.group(2)[0].isdigit() == ordered):
                i = nxt  # blank lines between items: still one list
                continue
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
        count += 1
        i += 1
    out.append(f"</{tag}>")
    if state is not None:
        state["next"] = (start_no + count) if ordered else None
    return "".join(out), i


def _split_row(line: str) -> list[str]:
    """Cells of a pipe-table row. A | inside $...$ or `...` stays in its cell."""
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|") and not line.endswith("\\|"):
        line = line[:-1]
    cells, cur = [], ""
    for part in re.split(r"(?<!\\)\|", line):
        cur = f"{cur}|{part}" if cur else part
        if cur.count("$") % 2 == 0 and cur.count("`") % 2 == 0:
            cells.append(cur.strip().replace("\\|", "|"))
            cur = ""
    if cur:
        cells.append(cur.strip())
    return cells


def _table(lines: list[str], i: int, sources) -> tuple[str, int]:
    head = _split_row(lines[i])
    seps = _split_row(lines[i + 1])
    align = ["center" if s.startswith(":") and s.endswith(":") else "right" if s.endswith(":") else "" for s in seps]
    i += 2
    rows = []
    while i < len(lines) and lines[i].strip() and "|" in lines[i]:
        rows.append(_split_row(lines[i]))
        i += 1

    def cell(tag: str, text: str, k: int) -> str:
        a = f' style="text-align:{align[k]}"' if k < len(align) and align[k] else ""
        scope = ' scope="col"' if tag == "th" else ""
        return f"<{tag}{scope}{a}>{inline(text, sources)}</{tag}>"

    th = "".join(cell("th", c, k) for k, c in enumerate(head))
    body = "".join("<tr>" + "".join(cell("td", c, k) for k, c in enumerate(r)) + "</tr>" for r in rows)
    return (f'<div class="table-wrap" tabindex="0" role="region" aria-label="Table, scrolls sideways if wide">'
            f"<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>"), i


_STEP_RE = re.compile(r"^\s*(?:\*\*|__)?Step\s+\d+\s*(?:[.:)]|\*\*)", re.I)


def _is_mathy(line: str) -> bool:
    return "=" in line or "$" in line


def _paragraph(lines: list[str], sources) -> str:
    """Lines of one paragraph. Prose lines join with spaces; a run of equations or math lines keeps its line breaks."""
    lines = [ln.strip() for ln in lines]
    mathy = sum(1 for ln in lines if _is_mathy(ln))
    if len(lines) > 1 and mathy >= 2 and mathy * 10 >= len(lines) * 6:
        return "<p>" + "<br>".join(inline(ln, sources) for ln in lines) + "</p>"
    return f"<p>{inline(' '.join(lines), sources)}</p>"


def _normalize(lines: list[str]) -> list[str]:
    """Put <details>, <summary> and </details> on lines of their own (outside code) so one-line forms work."""
    out: list[str] = []
    fence: str | None = None
    for line in lines:
        fm = FENCE_RE.match(line)
        if fm and (fence is None or fm.group(2) == fence):
            fence = None if fence else fm.group(2)
            out.append(line)
        elif fence is None and "`" not in line and re.search(r"(?i)</?(?:details|summary)\b", line):
            parts = [p for p in _TAG_SPLIT.split(line)]
            out.extend(p for p in parts if p.strip())
        else:
            out.append(line)
    return out


def _blocks(lines: list[str], sources, depth: int = 0, offset: int = 1, state: dict | None = None) -> str:
    if state is None:
        state = {}
    out: list[str] = []
    i = 0
    para: list[str] = []

    def flush() -> None:
        if para:
            out.append(_paragraph(para, sources))
            para.clear()
            state["next"] = None

    while i < len(lines):
        line = lines[i]
        if line.strip() and not ITEM_RE.match(line) and not DETAILS_OPEN_RE.match(line) \
                and not DETAILS_CLOSE_RE.match(line) and not SUMMARY_RE.match(line):
            state["next"] = None  # anything but a list or a details block ends the numbering run
        fm = FENCE_RE.match(line)
        if fm:
            flush()
            indent, mark, lang = len(fm.group(1)), fm.group(2), fm.group(3)
            i += 1
            code = []
            while i < len(lines) and not (FENCE_RE.match(lines[i]) and FENCE_RE.match(lines[i]).group(2) == mark
                                          and not FENCE_RE.match(lines[i]).group(3)):
                code.append(lines[i][indent:] if lines[i][:indent].strip() == "" else lines[i])
                i += 1
            i += 1
            lang = re.sub(r"[^\w+#.-]", "", lang)
            cls = f' class="language-{escape(lang)}"' if lang else ""
            name = f"{lang} code" if lang else "Code"
            out.append(f'<pre tabindex="0" role="group" aria-label="{escape(name)}"{(" data-lang=" + chr(34) + escape(lang) + chr(34)) if lang else ""}>'
                       f"<code{cls}>{escape(chr(10).join(code))}</code></pre>")
            continue
        if DETAILS_OPEN_RE.match(line) and depth < 4:
            flush()
            i += 1
            inner, nest = [], 1
            while i < len(lines):
                if DETAILS_OPEN_RE.match(lines[i]):
                    nest += 1
                elif DETAILS_CLOSE_RE.match(lines[i]):
                    nest -= 1
                    if nest == 0:
                        break
                inner.append(lines[i])
                i += 1
            i += 1
            while inner and not inner[0].strip():
                inner = inner[1:]
            summary = "More"
            if inner and (sm := SUMMARY_RE.match(inner[0])):
                summary = sm.group(1)
                inner = inner[1:]
            body_html = _blocks(inner, sources, depth + 1, offset)
            if not body_html.strip():  # a "Show solution" with nothing inside is not shown
                continue
            out.append(f"<details><summary>{inline(summary, sources)}</summary>{body_html}</details>")
            continue
        if DETAILS_CLOSE_RE.match(line) or SUMMARY_RE.match(line):  # stray tag from the model: drop it
            flush()
            i += 1
            continue
        if not line.strip():
            flush()
            i += 1
            continue
        st = line.strip()
        if st.startswith("$$") and (st.count("$$") >= 2 and st.endswith("$$") and len(st) > 4 or st.count("$$") == 1):
            flush()
            block = [st]
            i += 1
            if st.count("$$") == 1:
                while i < len(lines):
                    block.append(lines[i].strip())
                    done = "$$" in lines[i]
                    i += 1
                    if done:
                        break
            out.append(f'<div class="math-block">{escape(chr(10).join(block))}</div>')
            continue
        if (h := HEAD_RE.match(line)):
            flush()
            level = max(1, min(6, len(h.group(1)) + offset))
            out.append(f"<h{level}>{inline(h.group(2), sources)}</h{level}>")
            i += 1
            continue
        if "|" in line and i + 1 < len(lines) and TABLE_SEP_RE.match(lines[i + 1]) and "|" in lines[i + 1] \
                and (len(_split_row(line)) >= 2 or (line.strip().startswith("|") and line.strip().endswith("|")
                                                    and lines[i + 1].strip().startswith("|")
                                                    and lines[i + 1].strip().endswith("|"))):
            flush()
            html, i = _table(lines, i, sources)
            out.append(html)
            continue
        if ITEM_RE.match(line):
            flush()
            html, i = _list(lines, i, sources, state)
            out.append(html)
            continue
        if line.lstrip().startswith(">"):
            flush()
            quote = []
            while i < len(lines) and lines[i].lstrip().startswith(">"):
                quote.append(lines[i].lstrip()[1:].lstrip())
                i += 1
            out.append(f"<blockquote>{_blocks(quote, sources, depth + 1, offset)}</blockquote>")
            continue
        if re.match(r"^\s*(-{3,}|\*{3,})\s*$", line):
            flush()
            out.append("<hr>")
            i += 1
            continue
        if para and _STEP_RE.match(line):  # "Step 1." / "Step 2." on single newlines are separate paragraphs
            flush()
        para.append(line)
        i += 1
    flush()
    return "".join(out)


def shallowest_heading(*texts: str) -> int | None:
    """The smallest heading level (1 for #) used outside code in any of the texts, or None when there are none."""
    best: int | None = None
    for md in texts:
        fence: str | None = None
        for line in (md or "").replace("\r\n", "\n").split("\n"):
            fm = FENCE_RE.match(line)
            if fm and (fence is None or fm.group(2) == fence):
                fence = None if fence else fm.group(2)
                continue
            if fence is None and (h := HEAD_RE.match(line)):
                best = len(h.group(1)) if best is None else min(best, len(h.group(1)))
    return best


def render(md: str, sources: list[dict[str, Any]] | dict[int, dict[str, Any]] | None = None,
           *, heading_offset: int = 1, base_level: int | None = None, shallowest: int | None = None,
           numbering: dict | None = None) -> Markup:
    """Markdown to safe HTML.

    With base_level the shallowest heading in the text becomes that level, and the others keep their distance
    from it: render(md, base_level=2) under a page h1 gives h2, h3, ... whether the model wrote # or ##.
    Pass shallowest (from shallowest_heading over every part of one answer) when md is only one piece of it.
    Without base_level, heading_offset moves every heading down by that many levels (# is h2 at offset 1).
    numbering is a dict shared between render calls for the parts of one answer, so a numbered list that starts
    at 1 right after another one (a part split off between them) goes on counting."""
    smap = sources if isinstance(sources, dict) else source_map(sources)
    if base_level is not None:
        top = shallowest if shallowest is not None else shallowest_heading(md)
        heading_offset = base_level - (top or 1)
    lines = _normalize((md or "").replace("\r\n", "\n").split("\n"))
    return Markup(_blocks(lines, smap, 0, heading_offset, numbering))


def count_marks(md: str) -> tuple[int, int]:
    """(removed citations, unverified quotes) marked in stored markdown."""
    return (md or "").count(REMOVED_MARK), (md or "").count(QUOTE_OPEN)


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


# ---------------------------------------------------------------- textbook equations (the reader)
_T = (r"\(?(?:\d+(?:\.\d+)?|[A-Za-z]\d{1,2}(?:[A-Za-z]\d{0,2}){0,3}|(?=[A-Za-z]*[A-Z])[A-Za-z]{1,3}|[a-z]{1,2})\)?")
_EQ = re.compile(rf"(?<![\w$\\.])({_T}(?:\s*[=+\-*/\u00d7\u00b7^]\s*{_T})+)(?![\w$])")


def _tex(expr: str) -> str:
    t = expr.strip()
    t = re.sub(r"\(([^()]+)\)\s*/\s*\(([^()]+)\)", r"\\frac{\1}{\2}", t)
    t = re.sub(r"(?<![\w{\\])([A-Za-z0-9]+)\s*/\s*([A-Za-z0-9]+)(?![\w}])", r"\\frac{\1}{\2}", t)
    t = re.sub(r"([A-Za-z])(\d+)", r"\1_{\2}", t)
    return t.replace("*", r" \cdot ").replace("\u00d7", r" \times ").replace("\u00b7", r" \cdot ")


# Raw TeX that reached the text without dollar signs (slide text: \hat{x} = (A^TA)^{-1}A^Tb). A run is a stretch of
# words that hold a backslash command, a power or a subscript, with the lone letters, digits and signs between them.
_TEX_STRONG = re.compile(r"\\[A-Za-z|]+|[A-Za-z0-9)}]\^|[A-Za-z0-9)}]_[{A-Za-z0-9]|\^\{|\^[-0-9A-Za-z]")
_TEX_WEAK = re.compile(r"^[(\[]?[A-Za-z0-9]?[)\]]?[=+\-*/<>,.;]?$|^[=+\-*/<>]$|^\(?[A-Za-z]\)?[,.;]?$|^[0-9]+(?:\.[0-9]+)?$")


def _balanced(expr: str) -> bool:
    depth = 0
    for ch in expr:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _tex_runs(line: str) -> str:
    """Wrap runs of raw TeX in $...$. Lines with dollar signs, code or no TeX at all are returned unchanged."""
    if "$" in line or "`" in line or not _TEX_STRONG.search(line):
        return line
    parts = re.split(r"(\s+)", line)
    toks = [(k, t) for k, t in enumerate(parts) if t and not t.isspace()]
    kinds = []
    for _, t in toks:
        core = t.rstrip(".,;:?!")
        if re.fullmatch(r"\d{1,2}[.)]", t):
            kinds.append("word")  # "2." in running text numbers the next step, it is not part of a formula
            continue
        kinds.append("strong" if _TEX_STRONG.search(core) and not re.search(r"[A-Za-z]{4,}(?<!\\)\b(?![{^_])", re.sub(r"\\[A-Za-z]+|\{[^}]*\}", "", core)) else
                     "weak" if _TEX_WEAK.match(core) else "word")
    if toks and re.fullmatch(r"(?:[A-Za-z]|\d{1,2})[.)]", toks[0][1]) and len(toks) > 1:
        kinds[0] = "word"  # a list letter ("A. ...") is not part of the formula after it
    out_parts = list(parts)
    i = 0
    while i < len(toks):
        if kinds[i] == "word":
            i += 1
            continue
        j = i
        while j < len(toks) and kinds[j] != "word":
            j += 1
        run = list(range(i, j))
        def bare(r: int) -> bool:  # an operator or number alone at the edge of a run is not part of the formula
            return re.fullmatch(r"[=+\-*/<>,.;]+", toks[r][1].rstrip(".,;:") or "-") is not None \
                or re.fullmatch(r"[A-Za-z][.:?!]", toks[r][1]) is not None  # "A." ends the sentence before the formula
        while run and kinds[run[0]] == "weak" and bare(run[0]):
            run.pop(0)
        while run and kinds[run[-1]] == "weak" and bare(run[-1]):
            run.pop()
        if run and any(kinds[r] == "strong" for r in run):
            first, last = toks[run[0]][0], toks[run[-1]][0]
            text = "".join(parts[first:last + 1])
            trail = ""
            while text and text[-1] in ".,;:?!":
                trail = text[-1] + trail
                text = text[:-1]
            if _balanced(text):
                out_parts[first] = "$" + text + "$" + trail
                for k in range(first + 1, last + 1):
                    out_parts[k] = ""
        i = j
    return "".join(out_parts)


# Stoichiometry written flat in textbook text: "(2.5 g H)/(10.0 g compound) x 100%=25%", "4.7 g K ((mol K)/(39.10 g K)) = 0.12 mol K".
_FRAC_Q = re.compile(r"\(\s*(\([^()]+\))\s*/\s*(\([^()]+\))\s*\)|(\([^()]+\))\s*/\s*(\([^()]+\))")
_NUM_Q = re.compile(r"^(-?\d[\d,]*(?:\.\d+)?)\s*(.*)$")


def _q_tex(part: str) -> str:
    """A quantity such as "2.5 g H" or "mol K" as TeX: the number, a space, the unit as text."""
    part = part.strip().replace("%", r"\%")
    m = _NUM_Q.match(part)
    if m:
        return f"{m.group(1)}\\ \\text{{{m.group(2)}}}" if m.group(2) else m.group(1)
    return f"\\text{{{part}}}"


def _frac_tex(m: re.Match) -> str:
    a, b = (m.group(1) or m.group(3)), (m.group(2) or m.group(4))
    return f"\\frac{{{_q_tex(a[1:-1])}}}{{{_q_tex(b[1:-1])}}}"


def _stoich_line(line: str) -> str:
    """A line with a "(a)/(b)" fraction of quantities. An equation line becomes one math span; prose keeps its words and
    only the fraction is typeset."""
    st = line.strip()
    if "$" in st or "`" in st or not _FRAC_Q.search(st) or st.startswith(("#", "|", ">")):
        return line
    if not any(re.search(r"\s", g[1:-1]) for m in _FRAC_Q.finditer(st) for g in m.groups() if g and g.startswith("(")):
        return line  # (C1V1)/(C2) and (a)/(b) are algebra, handled by the equation rule
    lead = line[:len(line) - len(line.lstrip())]
    pieces = re.split(r"(?<=%)\s+(?=%\s*[A-Za-z]+\s*=)", st)
    if len(pieces) > 1:  # "% C= (7.34 g C)/(12.04 g compound) x 100%=61.0% % H= ..." is several equations on one line
        return lead + " ".join(_stoich_line(x).strip() for x in pieces)
    outside = _FRAC_Q.sub(" ", st)
    words = re.findall(r"\b[a-z]{3,}\b", outside)
    if "=" not in st or len(words) > 3 or re.search(r"[.!?]\s+[A-Z]", st):
        return lead + _FRAC_Q.sub(lambda m: f"${_frac_tex(m)}$", st)
    fracs: list[str] = []

    def hold(m: re.Match) -> str:
        fracs.append(_frac_tex(m))
        return f"\x02{len(fracs) - 1}\x02"
    t = _FRAC_Q.sub(hold, st.replace("\x02", ""))
    t = re.sub(r"^%\s*([A-Za-z]+)\s*=", r"\\%\\ \\text{\1} =", t)
    t = re.sub(r"\s*[\u00d7]\s*|(?<=\d)\s+x\s+(?=\d)|(?<=\x02)\s*x\s*(?=\d)", r" \\times ", t)
    t = re.sub(r"(?<![\\\w{.])(\d[\d,]*(?:\.\d+)?)%", r"\1\\%", t)
    t = re.sub(r"(?<![\\\w{.])(\d[\d,]*(?:\.\d+)?)\s+([A-Za-z]+(?:\s+[A-Za-z]+){0,2})(?![\w}])",
               lambda m: f"{m.group(1)}\\ \\text{{{m.group(2)}}}", t)
    t = re.sub(r"(?<=[A-Za-z}])\s+(?=\x02)", r" \\cdot ", t)
    t = re.sub(r"\x02(\d+)\x02", lambda m: fracs[int(m.group(1))], t)
    return f"{lead}${t.strip()}$"


def _close_paren(t: str, i: int) -> int:
    depth = 0
    for k in range(i, len(t)):
        depth += (t[k] == "(") - (t[k] == ")")
        if depth == 0:
            return k
    return -1


def _tex_words(t: str) -> str:
    """Plain words as \\text, with an equals sign or a times sign left as math."""
    out = []
    for part in re.split(r"(=|\u00d7)", t):
        if part == "=":
            out.append("=")
        elif part == "\u00d7":
            out.append("\\times")
        elif part.strip():
            out.append("\\text{" + re.sub(r"([%&#_$])", r"\\\1", part.strip()) + "}")
    return " ".join(out)


def _nested_frac_tex(t: str) -> str:
    """"(a (b)/(c))/(d) = n" as nested \\frac with the words as \\text. Parentheses that are not a fraction stay text."""
    out, buf, i = [], "", 0
    while i < len(t):
        if t[i] == "(":
            j = _close_paren(t, i)
            m = re.match(r"\s*/\s*\(", t[j + 1:]) if j > 0 else None
            if m:
                k = j + m.end()
                e = _close_paren(t, k)
                if e > 0:
                    out.append(_tex_words(buf))
                    buf = ""
                    out.append("\\frac{" + _nested_frac_tex(t[i + 1:j]) + "}{" + _nested_frac_tex(t[k + 1:e]) + "}")
                    i = e + 1
                    continue
        buf += t[i]
        i += 1
    out.append(_tex_words(buf))
    return " ".join(x for x in out if x)


def _nested_equation(line: str) -> str:
    """A flat textbook equation whose fraction holds parentheses inside parentheses, as one math span."""
    st = line.strip()
    if "=" not in st or re.search(r"[.!?]\s+[A-Z]", st) or st.startswith(("#", "|", ">")):
        return line
    depth = peak = 0
    for ch in st:
        depth += (ch == "(") - (ch == ")")
        peak = max(peak, depth)
    if peak < 2 or not re.search(r"\)\s*/\s*\(", st):
        return line
    if st.count("(") != st.count(")"):
        return line
    tex = _nested_frac_tex(st)
    if "\\frac" not in tex:
        return line
    return line[:len(line) - len(line.lstrip())] + f"${tex}$"


_WHOLE_EQ = re.compile(rf"^({_T}\s*=\s*{_T}(?:\s*[+\-*/\u00d7\u00b7]\s*{_T})*)[.,;]?$")


def _table_math_line(line: str) -> str:
    """A pipe-table row: each cell gets the same equation treatment as a line of text."""
    cells = _split_row(line)
    if not cells or all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
        return line
    out = []
    for c in cells:
        t = _math_line(c) if c and "|" not in c else c
        out.append(t.replace("|", "\\|"))
    return "| " + " | ".join(out) + " |"


def _math_line(line: str) -> str:
    if line.lstrip().startswith("|") and line.rstrip().endswith("|") and "$" not in line:
        return _table_math_line(line)
    if "$" in line or "`" in line or line.lstrip().startswith(("#", "|", ">")):
        return line
    if (w := _WHOLE_EQ.match(line.strip())) and not line.strip().startswith(("#", ">")):
        return line[:len(line) - len(line.lstrip())] + f"${_tex(w.group(1))}$"
    if "=" in line:  # *C*1*V*1 = *C*2*V*2: the asterisks were italics for variable names
        line = re.sub(r"(?<![A-Za-z_*])\*([A-Za-z]{1,2})\*(?![A-Za-z*])", r"\1", line)
        line = re.sub(r"\*\*([A-Za-z]{1,2})\*(?=\d)", r"\1", line)  # **n*1: bold marks glued to a variable name
        if line.count("*") == 1:  # one asterisk left with nothing to close it
            line = re.sub(r"\*(?=\w)", "", line)

    if "\\" in line:  # raw TeX with commands: wrap the whole run, so "=\|Ax\|^2" is not cut in two
        whole = _tex_runs(line)
        if whole != line:
            return whole
    stoich = _stoich_line(line)
    if stoich != line:
        return stoich
    nested = _nested_equation(line)
    if nested != line:
        return nested

    def sub(m: re.Match) -> str:
        e = m.group(1)
        if "=" not in e or not re.search(r"/|[A-Za-z]\d|\^", e) or not _balanced(e):
            return e
        return f"${_tex(e)}$"
    line = _EQ.sub(sub, line)
    return _tex_runs(line)


def mathify(md: str) -> str:
    """Recognisable equations in flat textbook text ("V2= (C1V1)/(C2)") become $...$ so the page can typeset them.
    Only a run of short variable names, numbers and operators with an equals sign and a fraction, a subscripted
    variable or a power is touched; code fences, code spans and text that already has math are left alone."""
    out: list[str] = []
    fence: str | None = None
    for line in (md or "").replace("\r\n", "\n").split("\n"):
        fm = FENCE_RE.match(line)
        if fm and (fence is None or fm.group(2) == fence):
            fence = None if fence else fm.group(2)
            out.append(line)
        else:
            out.append(line if fence else _math_line(line))
    return "\n".join(out)


_EMPTY_PARENS = re.compile(r"[ \t]*\(\s*\)")


def _drop_figure_artifacts(md: str) -> str:
    """The importer leaves "()" where a figure reference was ("Thomas Jefferson ()"); the reader hides them."""
    out: list[str] = []
    fence: str | None = None
    for line in (md or "").replace("\r\n", "\n").split("\n"):
        fm = FENCE_RE.match(line)
        if fm and (fence is None or fm.group(2) == fence):
            fence = None if fence else fm.group(2)
            out.append(line)
        else:
            out.append(line if fence or "`" in line else _EMPTY_PARENS.sub("", line))
    return "\n".join(out)


def render_reader(md: str, *, base_level: int = 3, math: bool = True) -> Markup:
    """The textbook reader's rendering: markdown emphasis plus recognisable equations as math."""
    return render(mathify(_drop_figure_artifacts(md)) if math else md, base_level=base_level)
