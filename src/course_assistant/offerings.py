"""Past-offering discovery: find earlier runs of the same course at the same university.

There is no web search API here. "Search" means three things the student controls: pasting links, crawling outward
from the course website (or a department course page the student gives), and trying a few likely archive
addresses under the course website. Every page is fetched through fetcher.PoliteFetcher (robots.txt, a slow rate
per site, size caps, no cookies, no credentials). Nothing here signs in anywhere. Candidates are only
suggestions: a page is saved into the course (import_url, kind past_offering) when the student picks it.
"""
from __future__ import annotations

import importlib
import re
from collections import deque
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urldefrag, urlsplit

from .store import Store

MAX_PASTED_URLS = 20
DEFAULT_MAX_PAGES = 20

_SEASONS = {"spring": "Spring", "sp": "Spring", "summer": "Summer", "su": "Summer", "sm": "Summer",
            "fall": "Fall", "fa": "Fall", "autumn": "Fall", "winter": "Winter", "wi": "Winter"}
_TERM_RE = re.compile(r"(?<![a-z])(spring|summer|autumn|fall|winter|sp|su|sm|fa|wi)[\s_\-./']*((?:19|20)?\d{2})(?!\d)",
                      re.I)
_YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
_ARCHIVE_RE = re.compile(r"\b(archives?|previous|past|older|prior|earlier|history|offerings?|semesters?|terms?)\b", re.I)
_SKIP_EXT = (".pdf", ".ppt", ".pptx", ".doc", ".docx", ".zip", ".png", ".jpg", ".jpeg", ".gif", ".mp4", ".mp3",
             ".mov", ".webm", ".css", ".js", ".ico", ".svg", ".tar", ".gz", ".ipynb", ".py", ".java", ".c", ".txt")
_ARCHIVE_GUESSES = ("archive/", "archives/", "previous/", "past/", "old/", "semesters/")


def guess_term(text: str) -> str:
    """'Fall 2023' from 'fall2023', 'sp22' or '/cs101/fa-2021/'; a bare year if that is all there is; else ''."""
    m = _TERM_RE.search(text or "")
    if m:
        season = _SEASONS[m.group(1).lower()]
        yr = m.group(2)
        if len(yr) == 2:
            yr = "20" + yr
        return f"{season} {yr}"
    y = _YEAR_RE.search(text or "")
    return y.group(1) if y else ""


def looks_like_offering(text: str) -> bool:
    return bool(guess_term(text) or _ARCHIVE_RE.search(text or ""))


class _Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[tuple[str, str]] = []
        self.title = ""
        self._href: str | None = None
        self._text: list[str] = []
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag == "a":
            self._href = dict(attrs).get("href") or None
            self._text = []
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._href is not None:
            self.links.append((self._href, " ".join("".join(self._text).split())))
            self._href = None
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        if self._href is not None:
            self._text.append(data)


def parse_page(html: str) -> tuple[str, list[tuple[str, str]]]:
    p = _Links()
    try:
        p.feed(html)
        p.close()
    except Exception:  # broken markup: keep whatever was read
        pass
    return " ".join(p.title.split()), p.links


def clean_url(url: str) -> str:
    """Validate a pasted address: http(s) with a host, fragment removed. Raises ValueError with a plain message."""
    u = (url or "").strip()
    parts = urlsplit(u)
    if parts.scheme.lower() not in ("http", "https") or not parts.hostname:
        raise ValueError(f"'{u[:80]}' is not a web address (it must start with http:// or https://).")
    if parts.username or parts.password:
        raise ValueError("Web addresses with a user name or password are not accepted.")
    return urldefrag(u)[0]


def _mod(name: str):
    return importlib.import_module(f"{__package__ or 'course_assistant'}.{name}")


def _page_text(res: Any) -> str:
    text = getattr(res, "text", None)
    if isinstance(text, str):
        return text
    body = getattr(res, "content", None) or getattr(res, "body", None) or b""
    return body.decode("utf-8", "replace") if isinstance(body, (bytes, bytearray)) else str(body)


def _is_blocked(exc: Exception) -> bool:
    return type(exc).__name__ == "FetchBlocked"


def _get_fetcher(fetcher: Any = None) -> Any:
    if fetcher is not None:
        return fetcher
    fetchmod = _mod('fetcher')
    global _SHARED
    if _SHARED is None:  # one per process so the per-host rate limit holds across calls
        _SHARED = fetchmod.PoliteFetcher()
    return _SHARED


_SHARED: Any = None


def candidates_from_urls(store: Store, course_id: int, urls: list[str], *, fetcher: Any = None,
                         found_by: str = "paste") -> list[dict[str, Any]]:
    """Validate pasted links, read each page's title politely, and store them as candidates (nothing is imported)."""
    if store.course(course_id) is None:
        raise ValueError(f"No course with id {course_id}.")
    cleaned: list[str] = []
    for u in urls:
        c = clean_url(u)
        if c not in cleaned:
            cleaned.append(c)
    if not cleaned:
        raise ValueError("Give at least one link.")
    if len(cleaned) > MAX_PASTED_URLS:
        raise ValueError(f"Give at most {MAX_PASTED_URLS} links at a time.")
    f = _get_fetcher(fetcher)
    out = []
    for u in cleaned:
        title, status, note = "", "found", ""
        try:
            res = f.get(u)
            title, _ = parse_page(_page_text(res))
        except Exception as exc:
            if _is_blocked(exc):
                status, note = "blocked", "The site's robots.txt asks programs not to read this page."
            else:
                note = f"Could not read the page to get its title: {str(exc)[:120]}"
        cid = store.add_offering_candidate(course_id, u, title=title, term_label=guess_term(f"{title} {u}"),
                                           found_by=found_by, note=note, status=status)
        out.append(store.offering_candidate(cid))
    return out  # type: ignore[return-value]


def crawl(store: Store, course_id: int, seed_url: str, *, max_pages: int = DEFAULT_MAX_PAGES,
          fetcher: Any = None, found_by: str = "crawl") -> list[dict[str, Any]]:
    """Walk outward from seed_url on the same host and under the same path prefix, following links whose text or
    address looks like a term, a year or an archive. Returns the candidates stored by this run."""
    if store.course(course_id) is None:
        raise ValueError(f"No course with id {course_id}.")
    seed = clean_url(seed_url)
    sp = urlsplit(seed)
    host = (sp.hostname or "").lower()
    prefix = sp.path if sp.path.endswith("/") else sp.path.rsplit("/", 1)[0] + "/"
    max_pages = max(1, min(int(max_pages), 50))
    f = _get_fetcher(fetcher)

    def in_scope(u: str) -> bool:
        p = urlsplit(u)
        return (p.scheme in ("http", "https") and (p.hostname or "").lower() == host
                and p.path.startswith(prefix) and not p.path.lower().endswith(_SKIP_EXT))

    queue: deque[tuple[str, str]] = deque([(seed, "")])
    seen = {seed}
    fetched = 0
    found: list[dict[str, Any]] = []
    while queue and fetched < max_pages:
        url, link_text = queue.popleft()
        fetched += 1
        title = ""
        try:
            res = f.get(url)
            html = _page_text(res)
        except Exception as exc:
            if _is_blocked(exc) and url != seed and guess_term(f"{link_text} {url}"):
                cid = store.add_offering_candidate(course_id, url, title=link_text,
                                                   term_label=guess_term(f"{link_text} {url}"), found_by=found_by,
                                                   status="blocked", note="robots.txt asks programs not to read this page.")
                found.append(store.offering_candidate(cid))  # type: ignore[arg-type]
            continue
        title, links = parse_page(html)
        term = guess_term(f"{title} {link_text} {url}")
        if url != seed and term:
            cid = store.add_offering_candidate(course_id, url, title=title or link_text, term_label=term,
                                               found_by=found_by)
            found.append(store.offering_candidate(cid))  # type: ignore[arg-type]
        for href, text in links:
            if not href or href.startswith(("mailto:", "javascript:", "tel:", "#")):
                continue
            target = urldefrag(urljoin(url, href))[0]
            if target in seen or not in_scope(target):
                continue
            if not looks_like_offering(f"{text} {urlsplit(target).path}"):
                continue
            seen.add(target)
            queue.append((target, text))
    return found


def guess_course_urls(course: dict[str, Any]) -> list[str]:
    """Likely archive pages under the course website, if it is known. Guesses only: a missing page is skipped."""
    site = (course.get("website_url") or "").strip()
    if not site:
        return []
    try:
        site = clean_url(site)
    except ValueError:
        return []
    last = site.rsplit("/", 1)[-1]
    if site.endswith("/"):
        base = site
    elif "." in last and urlsplit(site).path not in ("", "/"):
        base = site.rsplit("/", 1)[0] + "/"  # a file such as index.html: look beside it
    else:
        base = site + "/"
    return [site] + [urljoin(base, g) for g in _ARCHIVE_GUESSES]


def list_candidates(store: Store, course_id: int, status: str | None = None) -> list[dict[str, Any]]:
    return store.offering_candidates(course_id, status)


def pick(store: Store, course_id: int, candidate_id: int, *, fetcher: Any = None, added_by: str = "ai_app") -> int:
    """Save the page of one candidate into the course as a past offering. Returns the material id."""
    cand = store.offering_candidate(int(candidate_id))
    if cand is None or int(cand["course_id"]) != int(course_id):
        raise ValueError(f"Course {course_id} has no past-offering candidate {candidate_id}.")
    if cand["status"] == "saved" and cand.get("material_id"):
        return int(cand["material_id"])
    importers = _mod('importers')
    f = _get_fetcher(fetcher) if fetcher is not None else None
    try:
        mid = importers.import_url(store, course_id, cand["url"], kind="past_offering", fetcher=f, added_by=added_by)
    except Exception as exc:
        if _is_blocked(exc):
            store.update_offering_candidate(cand["id"], status="blocked", note="robots.txt asks programs not to read this page.")
        raise
    store.update_offering_candidate(cand["id"], status="saved", material_id=mid)
    return int(mid)
