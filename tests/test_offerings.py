import sys
import types

import pytest

from course_assistant import offerings
from course_assistant.store import Store


class Res:
    def __init__(self, text):
        self.text = text


class FetchBlocked(Exception):
    pass


class FakeFetcher:
    def __init__(self, pages, blocked=()):
        self.pages = pages
        self.blocked = set(blocked)
        self.calls = []

    def get(self, url):
        self.calls.append(url)
        if url in self.blocked:
            raise FetchBlocked(url)
        if url not in self.pages:
            raise RuntimeError("404")
        return Res(self.pages[url])


SITE = "https://cs.example.edu/courses/cs101/"
PAGES = {
    SITE: '<title>CS 101</title><a href="fall2023/">Fall 2023</a> <a href="sp22/">Spring 2022</a>'
          '<a href="archive.html">Previous offerings</a><a href="https://other.edu/cs101/fall2019/">Fall 2019</a>'
          '<a href="/courses/cs202/fall2023/">Fall 2023 (other course)</a><a href="homework/">Homework</a>'
          '<a href="mailto:a@b.c">mail</a>',
    SITE + "fall2023/": "<title>CS 101 Fall 2023</title><p>old</p>",
    SITE + "sp22/": "<title>CS 101, Spring 2022</title>",
    SITE + "archive.html": '<title>Archive</title><a href="2021/">2021</a><a href="fa19/">fa19</a>',
    SITE + "2021/": "<title>CS 101 2021</title>",
    SITE + "fa19/": "<title>Old</title>",
}


def make():
    s = Store(":memory:")
    cid = s.add_course("CS 101", university="Example U", website_url=SITE)
    return s, cid


def test_guess_term():
    assert offerings.guess_term("fall2023") == "Fall 2023"
    assert offerings.guess_term("/x/sp22/") == "Spring 2022"
    assert offerings.guess_term("Fall 2019") == "Fall 2019"
    assert offerings.guess_term("archive 2021") == "2021"
    assert offerings.guess_term("Homework 3") == ""


def test_crawl_stays_on_host_and_prefix_and_finds_terms():
    s, cid = make()
    f = FakeFetcher(PAGES)
    found = offerings.crawl(s, cid, SITE, fetcher=f)
    urls = {c["url"] for c in found}
    assert urls == {SITE + "fall2023/", SITE + "sp22/", SITE + "2021/", SITE + "fa19/"}
    assert not any("other.edu" in u or "cs202" in u or "homework" in u for u in f.calls)
    terms = {c["term_label"] for c in found}
    assert {"Fall 2023", "Spring 2022", "2021", "Fall 2019"} <= terms
    assert all(c["found_by"] == "crawl" for c in found)
    assert len(s.offering_candidates(cid)) == 4


def test_crawl_respects_max_pages_and_blocked():
    s, cid = make()
    f = FakeFetcher(PAGES)
    offerings.crawl(s, cid, SITE, max_pages=2, fetcher=f)
    assert len(f.calls) == 2
    s2, cid2 = make()
    f2 = FakeFetcher(PAGES, blocked={SITE + "sp22/"})
    found = offerings.crawl(s2, cid2, SITE, fetcher=f2)
    blocked = [c for c in found if c["status"] == "blocked"]
    assert [c["url"] for c in blocked] == [SITE + "sp22/"]


def test_candidates_from_urls_validates_and_titles():
    s, cid = make()
    f = FakeFetcher({"https://x.edu/a": "<title>CS 101 Fall 2020</title>", "https://x.edu/b": ""}, blocked={"https://x.edu/b"})
    out = offerings.candidates_from_urls(s, cid, ["https://x.edu/a#top", "https://x.edu/b", "https://x.edu/a"], fetcher=f)
    assert len(out) == 2
    assert out[0]["title"] == "CS 101 Fall 2020" and out[0]["term_label"] == "Fall 2020"
    assert out[1]["status"] == "blocked"
    for bad in ["ftp://x.edu/a", "javascript:alert(1)", "https://user:pw@x.edu/", "not a url"]:
        with pytest.raises(ValueError):
            offerings.candidates_from_urls(s, cid, [bad], fetcher=f)


def test_guess_course_urls():
    assert offerings.guess_course_urls({"website_url": ""}) == []
    g = offerings.guess_course_urls({"website_url": SITE})
    assert g[0] == SITE and SITE + "archive/" in g
    g2 = offerings.guess_course_urls({"website_url": SITE + "index.html"})
    assert SITE + "archive/" in g2


def test_pick_imports_as_past_offering(monkeypatch):
    s, cid = make()
    calls = {}

    def import_url(store, course_id, url, *, kind, fetcher=None, added_by="student"):
        calls.update(url=url, kind=kind, added_by=added_by)
        return store.add_material(course_id, kind, "Old page", source_url=url)

    fake = types.SimpleNamespace(import_url=import_url)
    monkeypatch.setitem(sys.modules, "course_assistant.importers", fake)
    cand = s.add_offering_candidate(cid, "https://x.edu/a")
    mid = offerings.pick(s, cid, cand)
    assert calls["kind"] == "past_offering" and calls["added_by"] == "ai_app"
    assert s.offering_candidate(cand)["status"] == "saved"
    assert s.offering_candidate(cand)["material_id"] == mid
    assert offerings.pick(s, cid, cand) == mid
    other = s.add_course("Other")
    with pytest.raises(ValueError):
        offerings.pick(s, other, cand)
