"""HTML cleaning for offline snapshots (and EPUB chapters), using only the standard library.

The cleaner drops scripts, styles, forms and other active content, keeps headings and readable structure,
gives every heading an anchor id, and records the readable text by section.
"""
from __future__ import annotations

import html
from html.parser import HTMLParser

from .common import unique_slug

DROP = {"script", "style", "noscript", "iframe", "object", "embed", "svg", "canvas", "form", "template", "head",
        "button", "select", "textarea", "input", "audio", "video", "link", "meta", "base", "nav", "footer", "aside"}
VOID = {"br", "hr", "img", "meta", "link", "input", "base", "col", "wbr", "source", "area", "embed", "track"}
KEEP = {"h1", "h2", "h3", "h4", "h5", "h6", "p", "ul", "ol", "li", "table", "thead", "tbody", "tr", "th", "td",
        "a", "em", "strong", "b", "i", "code", "pre", "blockquote", "br", "hr", "sup", "sub", "dl", "dt", "dd",
        "figure", "figcaption", "caption", "div", "span", "section", "article", "main", "header", "u", "small"}
BLOCK = {"p", "li", "tr", "div", "section", "article", "main", "header", "blockquote", "pre", "dt", "dd", "figcaption",
         "caption", "table", "ul", "ol", "hr", "br", "figure", "dl"}
HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
SAFE_HREF = ("http://", "https://", "mailto:", "#")


class Cleaner(HTMLParser):
    def __init__(self, base_url: str = ""):
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.out: list[str] = []
        self.title = ""
        self._in_title = False
        self._skip = 0
        self._stack: list[str] = []
        self._buf: list[str] = []
        self._heading: str | None = None  # tag while inside a heading
        self._heading_text: list[str] = []
        self._heading_anchor = ""
        self.seen: set[str] = set()
        # sections: [anchor, heading, [paragraphs]]
        self.sections: list[list] = [["", "", []]]

    # --- text blocks
    def _flush(self) -> None:
        text = " ".join("".join(self._buf).split())
        self._buf = []
        if text and not self._heading:
            self.sections[-1][2].append(text)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "title":
            self._in_title = True
            return
        if self._skip:
            if tag in DROP and tag not in VOID:
                self._skip += 1
            return
        if tag in DROP:
            if tag not in VOID:
                self._skip = 1
            return
        a = {k.lower(): (v or "") for k, v in attrs}
        if tag == "img":
            alt = a.get("alt", "").strip()
            if alt:
                self.out.append(f"<span>[image: {html.escape(alt)}]</span>")
                self._buf.append(f"[image: {alt}]")
            return
        if tag in BLOCK:
            self._flush()
        if tag not in KEEP:
            return
        attr_s = ""
        if tag in HEADINGS:
            self._flush()
            anchor = a.get("id") or ""
            anchor = unique_slug(anchor, self.seen) if anchor else ""
            self._heading, self._heading_text = tag, []
            self._heading_anchor = anchor
            self.out.append(f"<{tag}@@H@@>")  # id filled in when the heading text is known
            self._stack.append(tag)
            return
        if tag == "a":
            href = a.get("href", "").strip()
            if href.lower().startswith(SAFE_HREF):
                attr_s = f' href="{html.escape(href, quote=True)}"'
            elif href and self.base_url and not href.lower().startswith(("javascript:", "data:", "vbscript:")):
                from urllib.parse import urljoin
                joined = urljoin(self.base_url, href)
                if joined.lower().startswith(("http://", "https://")):
                    attr_s = f' href="{html.escape(joined, quote=True)}"'
        elif tag in ("td", "th"):
            for k in ("colspan", "rowspan"):
                if a.get(k, "").isdigit():
                    attr_s += f' {k}="{a[k]}"'
        elif a.get("id"):
            attr_s = f' id="{html.escape(a["id"], quote=True)}"'
        if tag in ("td", "th"):
            self._buf.append(" ")
        self.out.append(f"<{tag}{attr_s}>")
        if tag not in VOID:
            self._stack.append(tag)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "title":
            self._in_title = False
            return
        if self._skip:
            if tag in DROP:
                self._skip -= 1
            return
        if tag in HEADINGS and self._heading == tag:
            text = " ".join("".join(self._heading_text).split())
            anchor = self._heading_anchor or unique_slug(text, self.seen, fallback="section")
            for i in range(len(self.out) - 1, -1, -1):
                if self.out[i] == f"<{tag}@@H@@>":
                    self.out[i] = f'<{tag} id="{html.escape(anchor, quote=True)}">'
                    break
            self.out.append(f"</{tag}>")
            self._heading = None
            if text:
                self.sections.append([anchor, text, []])
            if tag in self._stack:
                self._stack.remove(tag)
            return
        if tag in self._stack:
            while self._stack:
                top = self._stack.pop()
                if top == tag:
                    break
            self.out.append(f"</{tag}>")
        if tag in BLOCK or tag in ("td", "th"):
            self._flush()

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
            return
        if self._skip:
            return
        self.out.append(html.escape(data, quote=False))
        if self._heading:
            self._heading_text.append(data)
        else:
            self._buf.append(data)

    def close(self) -> None:
        super().close()
        self._flush()
        while self._stack:
            self.out.append(f"</{self._stack.pop()}>")

    # --- results
    def cleaned_html(self, title: str = "") -> str:
        body = "".join(self.out)
        t = html.escape(title or self.title.strip() or "Saved page")
        return ("<!doctype html>\n<html><head><meta charset=\"utf-8\"><title>" + t + "</title></head>\n<body>\n"
                + body + "\n</body></html>\n")

    def section_list(self) -> list[tuple[str, str, str]]:
        out = []
        for anchor, heading, paras in self.sections:
            out.append((anchor or "top", heading, "\n\n".join(paras)))
        return out


def decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1", errors="replace")


def clean_html(raw: str | bytes, base_url: str = "") -> tuple[str, str, list[tuple[str, str, str]]]:
    """Returns (cleaned html, page title, sections as (anchor, heading, text))."""
    text = decode(raw) if isinstance(raw, bytes) else raw
    c = Cleaner(base_url)
    c.feed(text)
    c.close()
    title = " ".join(c.title.split())
    if not title:
        for _, heading, _ in c.section_list():
            if heading:
                title = heading
                break
    return c.cleaned_html(title), title, c.section_list()
