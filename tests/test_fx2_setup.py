"""Cycle 2 setup and onboarding fixes (fx2-setup)."""
import sys
import types

import httpx
import pytest

from course_assistant import connect, offerings, paths
from course_assistant.store import Store


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    h.mkdir()
    for var in ("HOME", "USERPROFILE"):
        monkeypatch.setenv(var, str(h))
    monkeypatch.setenv("APPDATA", str(h / "Roaming"))
    monkeypatch.setenv("LOCALAPPDATA", str(h / "Local"))
    monkeypatch.setenv("CODEX_HOME", str(h / ".codex"))
    monkeypatch.setenv("PATH", str(tmp_path / "emptybin"))
    monkeypatch.setenv(paths.ENV_DATA_DIR, str(tmp_path / "data"))
    return h


def test_problem1_connect_all_writes_nothing_without_installed_apps(home):
    results = connect.connect("all")
    skills = results[-1]
    assert skills["targets"] == {} and set(skills["skipped"]) == {"Claude Code", "Codex"}
    assert list(home.iterdir()) == []
    assert connect.write_plan("all")["writes"] == []


def test_problem1_only_installed_clients_get_skills(home):
    (home / ".codex").mkdir()
    plan = connect.write_plan("all", home=home)
    assert {w["app"] for w in plan["writes"]} == {"Codex"} and "Claude Desktop" in plan["skipped"]
    results = connect.connect("all", home=home)
    assert (home / ".agents" / "skills" / "course-assistant").is_dir()
    assert not (home / ".claude").exists()
    assert (home / ".codex" / "config.toml").exists()


def test_problem1_detects_binary_on_path(home, tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    exe = bindir / "claude"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    monkeypatch.setenv("PATH", str(bindir))
    assert connect.detect_clients(home)["Claude Code"] is True
    assert connect.detect_clients(home)["Codex"] is False


def _app(tmp_path):
    from types import SimpleNamespace

    from fastapi.testclient import TestClient

    from course_assistant import disclaimer, web

    store = Store(":memory:")
    store.set_setting(disclaimer.ACK_SETTING, "2026-01-01T00:00:00+00:00")
    app = web.create_app(store, key="k" * 40, provider_factory=lambda: None)
    c = TestClient(app, base_url="http://127.0.0.1:8766")
    c.cookies.set(web.COOKIE, "k" * 40)
    return store, c


def test_problem1_web_connect_page_lists_paths_before_the_button(home, tmp_path):
    (home / ".codex").mkdir()
    store, c = _app(tmp_path)
    html = c.get("/connect").text
    assert str(home / ".codex" / "config.toml") in html and str(home / ".agents" / "skills") in html
    assert "Nothing, because" in html  # Claude Desktop is not installed: nothing will be written for it


def test_problem2_setup_study_ask_say_what_each_option_unlocks(home, tmp_path):
    store, c = _app(tmp_path)
    cid = store.add_course("Math")
    for url in (f"/courses/{cid}/study", f"/courses/{cid}/ask"):
        t = c.get(url).text
        assert "API key or local model:" in t and "these buttons work" in t
        assert "you ask in Claude Desktop or Codex; the buttons here stay off" in t
        assert "/settings" in t and "switched off" in t
    setup = c.get("/setup").text
    assert "I don&#39;t have any of these" in setup or "I don't have any of these" in setup
    assert "spending limit" in setup and "$" not in setup.split("I don")[-1].split("Skip for now")[0]


def test_problem3_unreachable_site_says_so(tmp_path):
    store = Store(tmp_path / "w.db")
    cid = store.add_course("Math", website_url="https://math.example.edu/course/")

    class Down:
        def get(self, url):
            raise OSError("connection refused")

    rep = offerings.find_past_offerings_report(store, cid, "", fetcher=Down())
    assert rep["found"] == [] and rep["unreachable"]
    assert "couldn't reach math.example.edu" in rep["message"]


def test_problem3_skipped_links_are_listed_with_reasons(tmp_path):
    store = Store(tmp_path / "w.db")
    cid = store.add_course("Math")

    class OK:
        def get(self, url):
            return types.SimpleNamespace(text="<title>Fall 2018</title>")

    text = "https://a.edu/f18\nhttps://a.edu/f18\nftp://a.edu/old.zip"
    rep = offerings.find_past_offerings_report(store, cid, text, fetcher=OK())
    assert len(rep["found"]) == 1
    reasons = {x["url"]: x["reason"] for x in rep["skipped"]}
    assert "more than once" in reasons["https://a.edu/f18"]
    assert "ftp://" in reasons["ftp://a.edu/old.zip"]
    lines = offerings.report_lines(rep)
    assert any("ftp://a.edu/old.zip" in l for l in lines)
    again = offerings.find_past_offerings_report(store, cid, "https://a.edu/f18", fetcher=OK())
    assert again["found"] == [] and "already on the list" in again["skipped"][0]["reason"]


def test_problem3_web_paste_and_search_report_the_reasons(home, tmp_path, monkeypatch):
    store, c = _app(tmp_path)
    cid = store.add_course("Math", website_url="https://math.example.edu/c/")
    ok = types.SimpleNamespace(get=lambda url: types.SimpleNamespace(text="<title>Fall 2018</title>"))
    monkeypatch.setattr(offerings, "_SHARED", ok)
    r = c.post(f"/courses/{cid}/offerings/paste", data={"links": "https://a.edu/x\nftp://a.edu/y"}, follow_redirects=True)
    assert "Added 1 link" in r.text and "Skipped ftp://a.edu/y" in r.text

    class Down:
        def get(self, url):
            raise OSError("no route")

    monkeypatch.setattr(offerings, "_SHARED", Down())
    r = c.post(f"/courses/{cid}/offerings/search", follow_redirects=True)
    assert "couldn&#39;t reach math.example.edu" in r.text and "Nothing new turned up" not in r.text


def test_problem3_source_label():
    assert offerings.source_label({"kind": "past_offering", "title": "Exam 1, Fall 2018"}) == "Past offering: Fall 2018"
    assert offerings.source_label({"kind": "past_offering", "title": "Old notes"}) == "Past offering"
    assert offerings.source_label({"kind": "notes", "title": "Fall 2018"}) is None
    assert offerings.source_label(None) is None


def _add(store, cid, kind, title, name):
    return store.add_material(cid, kind, title, original_name=name, filename=f"{cid}/{name}")


def test_problem4_materials_show_file_type_group_and_search(home, tmp_path):
    store, c = _app(tmp_path)
    cid = store.add_course("Math")
    _add(store, cid, "syllabus", "Syllabus", "syllabus.docx")
    _add(store, cid, "syllabus", "Syllabus", "syllabus.md")
    _add(store, cid, "assignment", "Problem Set 2", "ps2.pdf")
    _add(store, cid, "slides", "Lecture 1", "l1.pptx")
    t = c.get(f"/courses/{cid}/materials").text
    assert "Word: syllabus.docx" in t and "Markdown: syllabus.md" in t and "PDF: ps2.pdf" in t
    assert t.index("Syllabus</h4>") < t.index("Assignment</h4>") or "kind-group" in t
    assert 'role="search"' in t and 'name="q"' in t
    f = c.get(f"/courses/{cid}/materials?q=ps2").text
    assert "Problem Set 2" in f and "syllabus.docx" not in f
    none = c.get(f"/courses/{cid}/materials?q=zzzz").text
    assert "No materials match" in none


def test_problem4_mark_graded_only_for_assignment_like_kinds(home, tmp_path):
    store, c = _app(tmp_path)
    cid = store.add_course("Math")
    for kind in ("syllabus", "slides", "reading", "assignment", "handout"):
        _add(store, cid, kind, f"Item {kind}", f"{kind}.md")
    t = c.get(f"/courses/{cid}/materials").text
    assert t.count("Mark as graded<span") == 2
    rows = {k: r for k in ("syllabus", "slides", "reading", "assignment", "handout") for r in t.split('<li class="row-item">') if f"Item {k}<" in r}
    assert all(("Mark as graded" in rows[k]) == (k in ("assignment", "handout")) for k in rows)
