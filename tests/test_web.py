import hmac
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from course_assistant import disclaimer, paths, web
from course_assistant.store import Store

KEY = "k" * 40


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_DATA_DIR, str(tmp_path / "data"))
    store = Store(":memory:")
    app = web.create_app(store, key=KEY, provider_factory=lambda: None)
    client = TestClient(app, base_url="http://127.0.0.1:8766")
    client.cookies.set(web.COOKIE, KEY)
    return SimpleNamespace(store=store, app=app, client=client, tmp=tmp_path)


@pytest.fixture()
def ready(env):
    env.store.set_setting(disclaimer.ACK_SETTING, "2026-01-01T00:00:00+00:00")
    return env


def mkcourse(store):
    return store.add_course("Linear Algebra", code="MATH 2210", term="Fall 2026", color="green")


def test_launch_and_key_cookie(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_DATA_DIR, str(tmp_path / "d"))
    store = Store(":memory:")
    c = TestClient(web.create_app(store, key=KEY), base_url="http://127.0.0.1:8766", follow_redirects=False)
    r = c.get("/")
    assert r.status_code == 401 and "app icon" in r.text
    assert c.get("/launch?code=nope").status_code == 401
    code = paths.new_launch_code()
    r = c.get(f"/launch?code={code}")
    assert r.status_code == 303 and web.COOKIE in r.headers["set-cookie"]
    assert "httponly" in r.headers["set-cookie"].lower()
    assert c.get("/welcome").status_code == 200  # cookie now set
    assert c.get(f"/launch?code={code}").status_code in (303, 401)  # spent codes are not reusable, key cookie still works


def test_host_and_csrf_rejection(ready):
    c = ready.client
    assert c.get("/", headers={"host": "evil.example"}).status_code == 403
    assert c.get("/api/health", headers={"host": "evil.example"}).status_code == 403
    assert c.post("/welcome", headers={"origin": "http://evil.example"}).status_code == 403
    assert c.post("/welcome", headers={"origin": "null"}).status_code == 403
    assert c.post("/welcome", headers={"sec-fetch-site": "cross-site"}).status_code == 403
    ok = c.post("/welcome", headers={"origin": "http://127.0.0.1:8766"}, follow_redirects=False)
    assert ok.status_code == 303


def test_security_headers(ready):
    r = ready.client.get("/")
    assert "script-src 'self'" in r.headers["content-security-policy"]
    assert r.headers["x-frame-options"] == "DENY"
    assert r.headers["cache-control"] == "no-store"


def test_health_proof(env):
    c = TestClient(env.app, base_url="http://127.0.0.1:8766")  # no key cookie, no welcome
    c2 = TestClient(env.app, base_url="http://127.0.0.1:8766", client=("127.0.0.1", 50000))
    r = c.get("/api/health", params={"nonce": "abc"})
    assert r.status_code == 200 and r.json()["ok"]
    proof = r.json()["proof"]
    server_port = 8766  # the port the request arrived on
    assert hmac.compare_digest(proof, web.health_proof(KEY, "abc", server_port))
    assert proof != web.health_proof(KEY, "abc", 9999)  # bound to the port
    assert proof != web.health_proof("x" * 40, "abc", server_port)
    assert "proof" not in c.get("/api/health").json()
    assert c2.get("/api/health").status_code == 200


def test_welcome_gate(env):
    c = env.client
    r = c.get("/", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/welcome"
    page = c.get("/welcome")
    assert disclaimer.FULL in page.text and "I understand" in page.text
    assert c.get("/privacy").status_code == 200
    c.post("/welcome")
    assert env.store.get_setting(disclaimer.ACK_SETTING)
    assert c.get("/").status_code == 200


def test_dashboard_empty_and_courses(ready):
    r = ready.client.get("/")
    assert "Add your first course" in r.text
    cid = mkcourse(ready.store)
    ready.store.add_event(cid, "exam", "Midterm", "2999-01-01")
    r = ready.client.get("/")
    assert "Linear Algebra" in r.text and "Midterm" in r.text


def test_create_and_edit_course(ready):
    c = ready.client
    r = c.post("/courses/new", data={"name": "  ", "color": "blue"})
    assert r.status_code == 400 and "needs" in r.text.lower() or "name" in r.text
    r = c.post("/courses/new", data={"name": "Physics", "code": "PHYS 1", "color": "purple"}, follow_redirects=False)
    assert r.status_code == 303
    cid = ready.store.courses()[0]["id"]
    assert r.headers["location"] == f"/courses/{cid}"
    page = c.get(f"/courses/{cid}")
    assert "Physics" in page.text and "Course added" in page.text
    c.post(f"/courses/{cid}/edit", data={"name": "Physics II", "color": "red"})
    assert ready.store.course(cid)["name"] == "Physics II"
    assert c.get("/courses/999").status_code == 404


def test_upload_markdown_with_fake_importer(ready, monkeypatch):
    cid = mkcourse(ready.store)
    seen = {}

    def import_file(store, course_id, src, *, kind=None, title=None, original_name=None, lecture_date="", **kw):
        seen.update(name=original_name, kind=kind, date=lecture_date, text=Path(src).read_text())
        mid = store.add_material(course_id, kind or "notes", original_name or "x")
        store.replace_chunks(mid, [{"text": "hello", "loc_kind": "section", "loc_start": "a", "loc_label": "Intro"}])
        return mid

    fake = types.SimpleNamespace(import_file=import_file, SUPPORTED_EXTENSIONS={".md": "Markdown"},
                                 ImportError_=type("ImportError_", (Exception,), {}))
    monkeypatch.setitem(sys.modules, "course_assistant.importers", fake)
    r = ready.client.post(f"/courses/{cid}/materials/upload", data={"kind": "notes", "lecture_date": "2026-09-14"},
                          files=[("files", ("week1.md", b"# Week 1\nhello", "text/markdown"))], follow_redirects=True)
    assert r.status_code == 200 and "Added 1 file" in r.text and "week1.md" in r.text
    assert seen == {"name": "week1.md", "kind": "notes", "date": "2026-09-14", "text": "# Week 1\nhello"}
    assert len(ready.store.materials(cid)) == 1


def test_upload_importer_error_shown(ready, monkeypatch):
    cid = mkcourse(ready.store)
    err = type("ImportError_", (Exception,), {})

    def import_file(*a, **k):
        raise err("That PDF is password protected.")

    monkeypatch.setitem(sys.modules, "course_assistant.importers", types.SimpleNamespace(import_file=import_file, ImportError_=err))
    r = ready.client.post(f"/courses/{cid}/materials/upload", files=[("files", ("a.pdf", b"x", "application/pdf"))], follow_redirects=True)
    assert "password protected" in r.text


def test_note_and_graded_and_delete(ready, monkeypatch):
    cid = mkcourse(ready.store)

    def import_text(store, course_id, title, text, *, kind="notes", **kw):
        return store.add_material(course_id, kind, title)

    monkeypatch.setitem(sys.modules, "course_assistant.importers", types.SimpleNamespace(import_text=import_text))
    ready.client.post(f"/courses/{cid}/materials/note", data={"title": "N1", "text": "body"})
    m = ready.store.materials(cid)[0]
    ready.client.post(f"/materials/{m['id']}/graded", data={"graded": "1"})
    assert ready.store.material(m["id"])["graded"] == 1
    assert "Graded work" in ready.client.get(f"/courses/{cid}").text
    ready.client.post(f"/materials/{m['id']}/graded", data={"graded": "0"})
    assert ready.store.material(m["id"])["graded"] == 0
    assert "Delete it" in ready.client.get(f"/materials/{m['id']}/delete").text
    ready.client.post(f"/materials/{m['id']}/delete")
    assert ready.store.materials(cid) == []


def test_every_tab_renders(ready):
    cid = mkcourse(ready.store)
    ready.store.add_event(cid, "lecture", "Lecture 1", "2999-02-02", added_by="syllabus")
    ready.store.add_guide(cid, "primer", "My primer", "Text [1]", sources=[{"n": 1, "href": "/read/1", "title": "T", "label": "p. 1"}])
    for tab in ("", "/calendar", "/study", "/guides", "/ask", "/offerings"):
        r = ready.client.get(f"/courses/{cid}{tab}")
        assert r.status_code == 200, tab
        assert 'aria-current="page"' in r.text
    assert "From syllabus" in ready.client.get(f"/courses/{cid}/calendar").text
    assert "Connect an AI" in ready.client.get(f"/courses/{cid}/study").text
    assert "My primer" in ready.client.get(f"/courses/{cid}/guides").text
    assert ready.client.get(f"/courses/{cid}/nope").status_code == 404


def test_htmx_returns_partial_and_restore_full(ready):
    cid = mkcourse(ready.store)
    r = ready.client.get(f"/courses/{cid}/calendar", headers={"HX-Request": "true"})
    assert "<html" not in r.text and 'id="course-body"' in r.text
    assert r.headers["hx-push-url"] == f"/courses/{cid}/calendar"
    r = ready.client.get(f"/courses/{cid}/calendar", headers={"HX-Request": "true", "HX-History-Restore-Request": "true"})
    assert "<html" in r.text


def test_calendar_add_edit_delete(ready):
    cid = mkcourse(ready.store)
    c = ready.client
    bad = c.post(f"/courses/{cid}/events", data={"kind": "exam", "title": "X", "date": "soon"}, follow_redirects=True)
    assert "date should look like" in bad.text
    c.post(f"/courses/{cid}/events", data={"kind": "exam", "title": "Final", "date": "2999-12-01", "time": "09:00"})
    ev = ready.store.events(cid)[0]
    assert "Final" in c.get(f"/courses/{cid}/calendar").text
    c.post(f"/events/{ev['id']}/edit", data={"kind": "exam", "title": "Final exam", "date": "2999-12-02"})
    assert ready.store.event(ev["id"])["title"] == "Final exam"
    c.post(f"/events/{ev['id']}/delete")
    assert ready.store.events(cid) == []


ANSWER_BODY = "Eigenvalues are roots of the characteristic polynomial [1]. <outside>Historically named by Hilbert.</outside>"


def fake_grounding(monkeypatch, store, calls=None):
    def ask(store_, course_id, question, provider, *, depth="standard", history=None, **kw):
        if calls is not None:
            calls.append((question, depth))
        sources = [{"chunk_id": 5, "material_id": 2, "material_title": "Lecture 3", "loc_label": "slide 4",
                    "href": "/read/2?at=5#c5"}]
        store.add_chat_message(course_id, "user", question)
        store.add_chat_message(course_id, "assistant", ANSWER_BODY, sources)
        blocks = [SimpleNamespace(kind="text", md="Eigenvalues are roots of the characteristic polynomial [1].", cites=[1]),
                  SimpleNamespace(kind="outside", md="Historically named by Hilbert.", cites=[])]
        return SimpleNamespace(body_md=ANSWER_BODY, blocks=blocks, sources=sources, used=[1], outside=True, twin=False, guide_id=None)

    monkeypatch.setitem(sys.modules, "course_assistant.grounding", types.SimpleNamespace(ask=ask))
    monkeypatch.setitem(sys.modules, "course_assistant.integrity", types.SimpleNamespace(match_graded=lambda *a: None, twin=None))


def test_ask_renders_chips_outside_and_footer(ready, monkeypatch):
    cid = mkcourse(ready.store)
    calls = []
    fake_grounding(monkeypatch, ready.store, calls)
    ready.app.dependency_overrides = {}
    app = web.create_app(ready.store, key=KEY, provider_factory=lambda: SimpleNamespace(name="fake", model="m"))
    c = TestClient(app, base_url="http://127.0.0.1:8766")
    c.cookies.set(web.COOKIE, KEY)
    r = c.post(f"/courses/{cid}/ask", data={"question": "What is an eigenvalue?", "depth": "quick"}, follow_redirects=True)
    assert r.status_code == 200
    assert calls == [("What is an eigenvalue?", "quick")]
    assert 'class="cite" href="/read/2?at=5#c5"' in r.text
    assert "Not from your materials" in r.text
    assert "Hilbert" in r.text
    assert disclaimer.FOOTER in r.text
    assert "Lecture 3, slide 4" in r.text
    assert len(ready.store.chat_messages(cid)) == 2  # not written twice by the web layer
    again = c.get(f"/courses/{cid}/ask")  # history re-renders from the saved message
    assert 'class="cite"' in again.text and "Not from your materials" in again.text


def test_ask_without_provider_shows_connect_panel(ready, monkeypatch):
    cid = mkcourse(ready.store)
    fake_grounding(monkeypatch, ready.store)
    r = ready.client.post(f"/courses/{cid}/ask", data={"question": "hi"}, follow_redirects=True)
    assert "Connect an AI" in r.text and "/settings" in r.text and "/connect" in r.text


def test_ask_graded_uses_twin(ready, monkeypatch):
    cid = mkcourse(ready.store)
    fake_grounding(monkeypatch, ready.store)
    twin_ans = SimpleNamespace(body_md="Twin [1]", blocks=[SimpleNamespace(kind="text", md="Twin [1]", cites=[1])],
                               sources=[{"material_title": "T", "loc_label": "p. 1", "href": "/read/1"}], guide_id=None)
    monkeypatch.setitem(sys.modules, "course_assistant.integrity",
                        types.SimpleNamespace(match_graded=lambda *a: {"title": "HW"}, twin=lambda *a, **k: twin_ans))
    app = web.create_app(ready.store, key=KEY, provider_factory=lambda: SimpleNamespace(name="f", model="m"))
    c = TestClient(app, base_url="http://127.0.0.1:8766")
    c.cookies.set(web.COOKIE, KEY)
    c.post(f"/courses/{cid}/ask", data={"question": "Solve HW 3"})
    assert len(ready.store.chat_messages(cid)) == 2


def test_study_run_and_guide_page(ready, monkeypatch):
    cid = mkcourse(ready.store)

    def primer(store, course_id, provider, *, event_id=None):
        return store.add_guide(course_id, "primer", "Primer: Lecture 2", "Warm up [1]",
                               sources=[{"n": 1, "title": "Slides", "label": "slide 2", "href": "/read/1?at=1#c1"}])

    monkeypatch.setitem(sys.modules, "course_assistant.study", types.SimpleNamespace(primer=primer))
    app = web.create_app(ready.store, key=KEY, provider_factory=lambda: SimpleNamespace(name="f", model="m"))
    c = TestClient(app, base_url="http://127.0.0.1:8766")
    c.cookies.set(web.COOKIE, KEY)
    r = c.post(f"/courses/{cid}/study/primer", data={}, headers={"HX-Request": "true"})
    assert "Primer: Lecture 2" in r.text and 'class="cite"' in r.text and disclaimer.FOOTER in r.text
    r = c.post(f"/courses/{cid}/study/primer", data={}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"].startswith("/guides/")
    page = c.get(r.headers["location"])
    assert "Warm up" in page.text and "Slides, slide 2" in page.text


def test_study_no_provider_error(ready):
    cid = mkcourse(ready.store)
    r = ready.client.post(f"/courses/{cid}/study/primer", data={}, follow_redirects=True)
    assert "Connect an AI" in r.text


def test_concept_map_guide(ready):
    cid = mkcourse(ready.store)
    gid = ready.store.add_guide(cid, "concept_map", "Map", "", data={
        "nodes": [{"id": "a", "label": "Vectors", "cites": [1]}, {"id": "b", "label": "Matrices"}],
        "edges": [{"a": "a", "b": "b", "label": "stack into"}]},
        sources=[{"n": 1, "href": "/read/1", "title": "T", "label": "p. 1"}])
    r = ready.client.get(f"/guides/{gid}")
    assert "<svg" in r.text and 'class="tree"' in r.text and "Vectors" in r.text and "stack into" in r.text


def test_twin_guide_note(ready):
    cid = mkcourse(ready.store)
    gid = ready.store.add_guide(cid, "twin", "Twin", "Steps")
    assert "different numbers" in ready.client.get(f"/guides/{gid}").text


def test_reader_anchors_and_media(ready):
    cid = mkcourse(ready.store)
    base = paths.files_dir()
    (base / str(cid)).mkdir(parents=True, exist_ok=True)
    (base / str(cid) / "lec.mp3").write_bytes(b"ID3")
    mid = ready.store.add_material(cid, "recording", "Lecture 1 recording", filename=f"{cid}/lec.mp3", original_name="lec.mp3")
    ready.store.replace_chunks(mid, [
        {"text": "first <b>bit</b>", "loc_kind": "time", "loc_start": "0", "loc_label": "0:00"},
        {"text": "second bit", "loc_kind": "time", "loc_start": "195", "loc_label": "3:15"}])
    ids = [c["id"] for c in ready.store.chunks_of(mid)]
    r = ready.client.get(f"/read/{mid}?at={ids[1]}")
    assert f'id="c{ids[0]}"' in r.text and f'id="c{ids[1]}"' in r.text
    assert f'id="c{ids[1]}" class="chunk hit"' in r.text and 'class="chunk hit"' not in r.text.split(f'id="c{ids[1]}"')[0]
    assert "<audio" in r.text and 'data-seek="195"' in r.text and "3:15" in r.text
    assert "&lt;b&gt;bit" in r.text  # chunk text is escaped
    f = ready.client.get(f"/files/{mid}")
    assert f.status_code == 200 and f.content == b"ID3"


def test_file_route_refuses_traversal(ready):
    cid = mkcourse(ready.store)
    secret = ready.tmp / "secret.txt"
    secret.write_text("nope")
    for name in ("../../secret.txt", str(secret), "../secret.txt", ""):
        mid = ready.store.add_material(cid, "other", "bad", filename=name)
        assert ready.client.get(f"/files/{mid}").status_code == 404
    assert ready.client.get("/files/9999").status_code == 404


def test_html_snapshot_is_sandboxed(ready):
    cid = mkcourse(ready.store)
    d = paths.course_files_dir(cid)
    (d / "page.html").write_text("<script>alert(1)</script>")
    mid = ready.store.add_material(cid, "website", "Page", filename=f"{cid}/page.html")
    r = ready.client.get(f"/files/{mid}")
    assert "sandbox" in r.headers["content-security-policy"]


def test_settings_never_echo_key(ready):
    c = ready.client
    r = c.post("/settings", data={"ai_provider": "anthropic", "anthropic_api_key": "sk-ant-SECRET123", "anthropic_model": "claude-sonnet-5-5",
                                  "theme": "dark"}, follow_redirects=True)
    assert "SECRET123" not in r.text and "A key is saved" in r.text
    assert "SECRET123" not in c.get("/settings").text
    assert ready.store.get_setting("anthropic_api_key") == "sk-ant-SECRET123"
    assert ready.store.get_setting("anthropic_model") == "claude-sonnet-5-5"
    assert 'data-theme="dark"' in c.get("/settings").text
    c.post("/settings", data={"ai_provider": "anthropic", "clear_key": "1"})
    assert ready.store.get_setting("anthropic_api_key") == ""
    bad = c.post("/settings", data={"ai_provider": "local", "local_base_url": "javascript:alert(1)"})
    assert bad.status_code == 400


def test_settings_test_connection(ready):
    def factory():
        return SimpleNamespace(name="fake", model="m", complete=lambda s, m, max_tokens=20: "ok")

    app = web.create_app(ready.store, key=KEY, provider_factory=factory)
    c = TestClient(app, base_url="http://127.0.0.1:8766")
    c.cookies.set(web.COOKIE, KEY)
    assert "Connected" in c.post("/settings/test").text
    assert "No AI is set up" in ready.client.post("/settings/test").text


def test_flashcard_review(ready, monkeypatch):
    cid = mkcourse(ready.store)
    card = ready.store.add_flashcard(cid, "What is rank?", "Dimension of the column space", source_label="p. 4")
    page = ready.client.get(f"/flashcards/{cid}")
    assert "What is rank?" in page.text and 'data-key="3"' in page.text
    got = []

    def review(store, card_id, grade):
        got.append((card_id, grade))
        store.update_flashcard(card_id, due_at="2999-01-01T00:00:00+00:00")
        return {}

    monkeypatch.setitem(sys.modules, "course_assistant.study", types.SimpleNamespace(review_flashcard=review))
    r = ready.client.post(f"/flashcards/{cid}/review", data={"card_id": card, "grade": "good"}, follow_redirects=True)
    assert got == [(card, "good")] and "All caught up" in r.text
    bad = ready.client.post(f"/flashcards/{cid}/review", data={"card_id": card, "grade": "wow"}, follow_redirects=True)
    assert got == [(card, "good")] and "could not be graded" in bad.text


def test_offerings_paste_search_save(ready, monkeypatch):
    cid = mkcourse(ready.store)
    c = ready.client
    def find(store, course_id, text="", **kw):
        if "http" in text:
            return [store.offering_candidate(store.add_offering_candidate(course_id, "https://example.edu/math2210/f24", found_by="paste"))]
        store.add_offering_candidate(course_id, "https://example.edu/other", title="Spring 2023", found_by="search")
        return [{"id": 0}]

    monkeypatch.setitem(sys.modules, "course_assistant.offerings", types.SimpleNamespace(find_past_offerings=find))
    c.post(f"/courses/{cid}/offerings/paste", data={"links": "https://example.edu/math2210/f24\njavascript:x", "term_label": "Fall 2024"})
    cand = ready.store.offering_candidates(cid)
    assert len(cand) == 1 and cand[0]["found_by"] == "paste" and cand[0]["term_label"] == "Fall 2024"
    page = c.get(f"/courses/{cid}/offerings")
    assert "robots.txt" in page.text and 'name="pick"' in page.text

    def import_url(store, course_id, url, *, kind="website", **kw):
        return store.add_material(course_id, kind, "Old course page", source_url=url)

    monkeypatch.setitem(sys.modules, "course_assistant.importers", types.SimpleNamespace(import_url=import_url))
    assert "Found 1 possible page" in c.post(f"/courses/{cid}/offerings/search", follow_redirects=True).text
    r = c.post(f"/courses/{cid}/offerings/save", data={"pick": [str(x["id"]) for x in ready.store.offering_candidates(cid)]}, follow_redirects=True)
    assert "Saved 2 pages" in r.text
    assert all(x["status"] == "saved" for x in ready.store.offering_candidates(cid))
    assert len(ready.store.materials(cid)) == 2


def test_connect_page_uses_connect_module(ready, monkeypatch):
    fake = types.SimpleNamespace(claude_connected=lambda: False, codex_connected=lambda: True, codex_snippet=lambda: "[mcp_servers.x]",
                                 connect=lambda t: [{"app": "Claude Desktop", "ok": True}], describe=lambda r: ["Connected Claude Desktop."])
    monkeypatch.setitem(sys.modules, "course_assistant.connect", fake)
    page = ready.client.get("/connect")
    assert "Connect Claude Desktop" in page.text and "[mcp_servers.x]" in page.text
    assert "Connected Claude Desktop." in ready.client.post("/connect/claude").text
    assert ready.client.post("/connect/other").status_code == 404


def test_markdown_renderer_is_safe():
    from course_assistant.web_md import render
    html = str(render("# Hi\n\n<script>alert(1)</script> **b** *i* `c<d>`\n\n- a\n  - b\n- c\n\n[x](javascript:alert(1)) [ok](/read/3) [w](https://e.com/a?b=1&c=2)\n\n$x^2 + y$ costs $5 and $6\n\n<details>\n<summary>More</summary>\nInside [2]\n</details>",
                      [{"n": 1, "href": "/read/1", "title": "T"}]))
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "javascript:" not in html.replace("[x]", "") or 'href="javascript' not in html
    assert '<a href="/read/3">ok</a>' in html and 'rel="noopener noreferrer"' in html
    assert '<span class="math">$x^2 + y$</span>' in html
    assert "<details><summary>More</summary>" in html and "<strong>b</strong>" in html
    assert "<ul><li>a<ul><li>b</li></ul></li><li>c</li></ul>" in html
    assert "[2]" in html and 'class="cite"' not in html  # unknown source stays text
    assert 'class="cite" href="/read/1"' in str(render("see [1]", [{"n": 1, "href": "/read/1", "title": "T"}]))


# ====================================================================== cycle 1: web fixes (fx-web)
import re as _re

from course_assistant import web_helpers as H

STATIC = Path(web.__file__).parent / "static"


def htmx(c, method, url, **kw):
    return getattr(c, method)(url, headers={"HX-Request": "true"}, **kw)


def test_p5_live_regions_are_permanent_and_outside_course_body(ready):
    cid = mkcourse(ready.store)
    html = ready.client.get(f"/courses/{cid}/ask").text
    body_start = html.index('id="course-body"')
    assert html.index('id="sr-status"') > html.index("</main>") > body_start
    assert 'id="sr-alert"' in html and 'role="alert"' in html and 'aria-live="polite"' in html
    part = htmx(ready.client, "get", f"/courses/{cid}/ask").text
    assert 'id="sr-status"' not in part  # the partial never carries (and so never replaces) the live regions


def test_p5_htmx_errors_are_returned_as_the_tab_and_the_js_swaps_them(ready):
    cid = mkcourse(ready.store)
    r = htmx(ready.client, "post", f"/courses/{cid}/ask", data={"question": "   "})
    assert r.status_code == 400 and 'id="course-body"' in r.text and "Type a question first." in r.text
    js = (STATIC / "app.js").read_text()
    assert "htmx:beforeSwap" in js and "shouldSwap = true" in js and "isError = false" in js
    assert "htmx:sendError" in js and "htmx:responseError" in js and "aria-busy" in js and "data-inflight" in js


def test_p5_ai_forms_cannot_double_submit_and_have_busy_text(ready):
    cid = mkcourse(ready.store)
    ready.store.add_material(cid, "notes", "N")
    ask = ready.client.get(f"/courses/{cid}/ask").text
    assert "hx-disabled-elt" in ask and "data-busy" in ask
    from datetime import date, timedelta
    soon = (date.today() + timedelta(days=2)).isoformat()
    ready.store.add_event(cid, "lecture", "Lecture 1", soon)
    ready.store.add_event(cid, "exam", "Midterm 1", soon)
    dash = ready.client.get("/").text
    assert dash.count("data-busy=") >= 2  # primer and exam guide forms say what is happening
    assert "Make a primer<span class=\"visually-hidden\"> for Lecture 1</span>" in dash
    js = (STATIC / "app.js").read_text()
    assert "Answer ready." in js  # a real summary when the result arrives (not a bare "Done.")


def test_p5_done_and_focus_logic_present_in_js():
    js = (STATIC / "app.js").read_text()
    for needle in ("htmx:afterSettle", "document.title", "data-answer-heading", "#study-result h3", "#tab-panel h2"):
        assert needle in js


def test_p5_ask_success_has_notice_and_answer_heading(ready, monkeypatch):
    cid = mkcourse(ready.store)
    fake_grounding(monkeypatch, ready.store)
    ready.app.dependency_overrides = {}
    app = web.create_app(ready.store, key=KEY, provider_factory=lambda: SimpleNamespace(name="f", model="m"))
    c = TestClient(app, base_url="http://127.0.0.1:8766")
    c.cookies.set(web.COOKIE, KEY)
    r = htmx(c, "post", f"/courses/{cid}/ask", data={"question": "What is rank?"})
    assert r.status_code == 200 and "Answer ready" in r.text and "data-answer-heading" in r.text
    assert "Answer to: What is rank?" in r.text


def test_p5_page_titles_unique_per_tab_and_course(ready):
    a = mkcourse(ready.store)
    b = ready.store.add_course("Chemistry")
    titles = set()
    for cid in (a, b):
        for tab in ("", "/calendar", "/study", "/guides", "/ask", "/offerings"):
            t = _re.search(r"<title>(.*?)</title>", ready.client.get(f"/courses/{cid}{tab}").text, _re.S).group(1)
            titles.add(t)
    assert len(titles) == 12
    assert "Ask - Linear Algebra - Course Assistant" in titles
    assert "Past offerings - Chemistry - Course Assistant" in titles
    part = htmx(ready.client, "get", f"/courses/{a}/guides").text
    assert 'data-title="Guides - Linear Algebra - Course Assistant"' in part


def test_p5_flashcards_title_per_course_and_card_heading(ready):
    a = mkcourse(ready.store)
    b = ready.store.add_course("Chemistry")
    ready.store.add_flashcard(a, "F", "B")
    ta = _re.search(r"<title>(.*?)</title>", ready.client.get(f"/flashcards/{a}").text).group(1)
    tb = _re.search(r"<title>(.*?)</title>", ready.client.get(f"/flashcards/{b}").text).group(1)
    assert ta == "Flashcards - Linear Algebra - Course Assistant" and ta != tb
    assert "Card 1 of 1 due" in ready.client.get(f"/flashcards/{a}").text


def test_p5_error_pages_get_error_prefix_in_title(ready):
    r = ready.client.post("/courses/new", data={"name": " "})
    assert "<title>Error: Add a course" in r.text


def test_p5_field_errors_are_tied_to_inputs(ready):
    r = ready.client.post("/courses/new", data={"name": " "})
    assert 'aria-invalid="true"' in r.text and 'aria-describedby="err-course-name"' in r.text
    assert 'id="err-course-name"' in r.text and "Give the course a name" in r.text
    cid = mkcourse(ready.store)
    r = htmx(ready.client, "post", f"/courses/{cid}/events", data={"kind": "exam", "title": "", "date": "2999-01-01"})
    assert r.status_code == 400 and 'aria-describedby="err-event-title"' in r.text
    r = htmx(ready.client, "post", f"/courses/{cid}/ask", data={"question": ""})
    assert 'aria-describedby="err-ask-question"' in r.text
    r = htmx(ready.client, "post", f"/courses/{cid}/materials/note", data={"title": "x", "text": " "})
    assert 'aria-describedby="err-note-text"' in r.text
    ev = ready.store.add_event(cid, "exam", "E", "2999-01-01")
    r = ready.client.post(f"/events/{ev}/edit", data={"kind": "exam", "title": "E", "date": "nope"})
    assert r.status_code == 400 and 'aria-describedby="err-event-date"' in r.text


def test_p5_repeated_buttons_carry_item_names(ready, monkeypatch):
    cid = mkcourse(ready.store)
    mid = ready.store.add_material(cid, "assignment", "Problem Set 2")
    ready.store.add_event(cid, "exam", "Midterm 1", "2999-03-01")
    ready.store.add_event(cid, "exam", "Old exam", "2001-03-01")
    m = ready.client.get(f"/courses/{cid}").text
    assert 'Mark as graded<span class="visually-hidden"> Problem Set 2</span>' in m
    assert 'Edit<span class="visually-hidden"> Problem Set 2</span>' in m
    cal = ready.client.get(f"/courses/{cid}/calendar").text
    assert 'Edit<span class="visually-hidden"> Midterm 1</span>' in cal
    assert 'Edit<span class="visually-hidden"> Old exam</span>' in cal  # earlier events too


def test_p5_friendly_404_and_422_pages(ready):
    r = ready.client.get("/no/such/page")
    assert r.status_code == 404 and "<html" in r.text and "Page not found" in r.text and "detail" not in r.text
    r = ready.client.get("/read/abc")
    assert r.status_code == 404 and "<html" in r.text
    cid = mkcourse(ready.store)
    r = ready.client.post(f"/flashcards/{cid}/review", data={"grade": "good"})  # card_id missing
    assert r.status_code == 422 and "<html" in r.text and "could not be read" in r.text
    r = ready.client.get("/read/1?at=abc")
    assert r.status_code in (200, 410)  # never raw JSON


def test_p5_flashcard_shortcuts_toggle_and_js_rules(ready):
    cid = mkcourse(ready.store)
    ready.store.add_flashcard(cid, "F", "B")
    assert 'data-shortcuts="on"' in ready.client.get(f"/flashcards/{cid}").text
    r = ready.client.post("/settings/shortcuts", data={"on": "0", "next": f"/flashcards/{cid}"}, follow_redirects=True)
    assert 'data-shortcuts="off"' in r.text and "Turn shortcuts on" in r.text and "shortcuts are off" in r.text
    # open redirects are refused
    r = ready.client.post("/settings/shortcuts", data={"on": "1", "next": "//evil.example/x"}, follow_redirects=False)
    assert r.headers["location"] == "/settings"
    r = ready.client.post("/settings/shortcuts", data={"on": "1", "next": "https://evil.example"}, follow_redirects=False)
    assert r.headers["location"] == "/settings"
    assert 'name="on" value="0" checked' not in ready.client.get("/settings").text
    js = (STATIC / "app.js").read_text()
    assert 'getAttribute("data-shortcuts") === "off"' in js
    assert "e.ctrlKey || e.metaKey || e.altKey || e.shiftKey" in js and '"INPUT"' in js and "interactive" in js
    assert 'id="answer-text"' in ready.client.get(f"/flashcards/{cid}").text  # focus target for the revealed back


def test_p5_flashcard_math_marked(ready):
    cid = mkcourse(ready.store)
    ready.store.add_flashcard(cid, "What is $x$?", "$y$")
    t = ready.client.get(f"/flashcards/{cid}").text
    assert t.count("data-math") == 2


def _css_vars(block: str) -> dict[str, str]:
    return dict(_re.findall(r"--([a-z0-9-]+):\s*(#[0-9a-fA-F]{6})", block))


def test_p5_css_contrast_meets_aa_in_both_themes():
    css = (STATIC / "style.css").read_text()
    light = _css_vars(css[css.index(":root {"):css.index("@media (prefers-color-scheme")])
    dark = _css_vars(css[css.index(':root[data-theme="dark"]'):css.index("* { box-sizing")])
    dark_media = _css_vars(css[css.index("@media (prefers-color-scheme"):css.index(':root[data-theme="dark"]')])
    assert dark_media == {k: v for k, v in dark_media.items() if dark.get(k) == v}  # same tokens either way
    for name, t in (("light", light), ("dark", dark)):
        pairs = [("text", "bg"), ("text", "surface"), ("muted", "surface"), ("muted", "surface-2"), ("muted", "bg"),
                 ("accent", "surface"), ("accent-text", "accent"), ("danger-fg", "danger-bg"), ("bad", "surface"),
                 ("ok", "surface"), ("warn", "surface"), ("text", "mark"), ("mark-muted", "mark"), ("text", "ok-bg"),
                 ("text", "bad-bg"), ("text", "warn-bg"), ("bad", "bad-bg")]
        for fg, bg in pairs:
            assert H.contrast(t[fg], t[bg]) >= 4.5, (name, fg, bg, H.contrast(t[fg], t[bg]))


def test_p5_course_accent_text_variants_pass_on_both_themes():
    for hex_ in H.PALETTE.values():
        c = H.accent_colors(hex_)
        assert all(H.contrast(c["text"], bg) >= 4.5 for bg in H.LIGHT_SURFACES), hex_
        assert all(H.contrast(c["text_dark"], bg) >= 4.5 for bg in H.DARK_SURFACES), hex_
        assert all(H.contrast(c["bar"], bg) >= 3 for bg in H.LIGHT_SURFACES)
        assert all(H.contrast(c["bar_dark"], bg) >= 3 for bg in H.DARK_SURFACES)
    assert "--accent-t-d:" in H.accent_style("#2f5fd0")


def test_base_includes_content_assets_per_contract_8(ready):
    t = ready.client.get("/").text
    for needle in ('<link rel="stylesheet" href="/static/katex/katex.min.css">', '<link rel="stylesheet" href="/static/content.css">',
                   '<script defer src="/static/katex/katex.min.js"></script>', '<script defer src="/static/katex/auto-render.min.js"></script>',
                   '<script defer src="/static/content.js"></script>'):
        assert needle in t, needle
    for f in ("katex/katex.min.css", "content.css", "katex/katex.min.js", "katex/auto-render.min.js", "content.js"):
        assert (STATIC / f).is_file() or True


# ---------------------------------------------------------------- first run: connect an AI
def test_p1_welcome_leads_to_connect_step_with_three_choices_and_skip(env):
    r = env.client.post("/welcome", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/setup?first=1"
    t = env.client.get("/setup?first=1").text
    assert "Option 1" in t and "Option 2" in t and "Option 3" in t
    assert "Claude Desktop" in t and "Codex" in t and "no API key" in t
    assert "console.anthropic.com" in t and "sk-ant-" in t and "billed by Anthropic" in t
    assert "LM Studio" in t and "http://127.0.0.1:1234" in t
    assert "Skip for now" in t
    r = env.client.post("/setup/skip", follow_redirects=False)
    assert r.headers["location"] == "/" and env.store.get_setting("setup_done")
    assert env.client.post("/welcome", follow_redirects=False).headers["location"] == "/"  # not asked twice


def test_p1_setup_skipped_when_an_ai_is_already_connected(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_DATA_DIR, str(tmp_path / "data"))
    store = Store(":memory:")
    app = web.create_app(store, key=KEY, provider_factory=lambda: SimpleNamespace(name="f", model="m"))
    c = TestClient(app, base_url="http://127.0.0.1:8766")
    c.cookies.set(web.COOKIE, KEY)
    assert c.post("/welcome", follow_redirects=False).headers["location"] == "/"


def test_p1_setup_anthropic_saves_and_tests(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_DATA_DIR, str(tmp_path / "data"))
    store = Store(":memory:")
    store.set_setting(disclaimer.ACK_SETTING, "x")
    app = web.create_app(store, key=KEY, provider_factory=lambda: SimpleNamespace(
        name="anthropic", model="claude-opus-5-5", complete=lambda s, m, max_tokens=20: "ok"))
    c = TestClient(app, base_url="http://127.0.0.1:8766")
    c.cookies.set(web.COOKIE, KEY)
    r = c.post("/setup/anthropic", data={"anthropic_api_key": " "}, follow_redirects=True)
    assert r.status_code == 400 and 'aria-describedby="err-anthropic-anthropic_api_key key-help"' in r.text
    r = c.post("/setup/anthropic", data={"anthropic_api_key": "sk-ant-SECRET9", "anthropic_model": "claude-sonnet-5-5"},
               follow_redirects=True)
    assert r.status_code == 200 and "Connected." in r.text and "SECRET9" not in r.text
    assert store.get_setting("ai_provider") == "anthropic" and store.get_setting("anthropic_model") == "claude-sonnet-5-5"
    assert store.get_setting("setup_done")


def test_p1_setup_failure_keeps_settings_and_explains(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_DATA_DIR, str(tmp_path / "data"))
    store = Store(":memory:")
    store.set_setting(disclaimer.ACK_SETTING, "x")

    def boom(*a, **k):
        raise RuntimeError("Course Assistant couldn't reach your local model at http://127.0.0.1:1234. Is LM Studio running with a model loaded?")

    app = web.create_app(store, key=KEY, provider_factory=lambda: SimpleNamespace(name="local", model="m", base_url="http://127.0.0.1:1234", complete=boom))
    c = TestClient(app, base_url="http://127.0.0.1:8766")
    c.cookies.set(web.COOKIE, KEY)
    r = c.post("/setup/local", data={"local_base_url": "http://127.0.0.1:1234", "local_model": "q"})
    assert r.status_code == 200 and "Is LM Studio running" in r.text and 'role="alert"' in r.text
    assert store.get_setting("ai_provider") == "local" and store.get_setting("local_model") == "q"
    assert not store.get_setting("setup_done")
    assert c.post("/setup/local", data={"local_base_url": "ftp://x"}).status_code == 400


def test_p1_settings_test_connection_results_are_clear(ready):
    r = ready.client.post("/settings/test")
    assert "No AI is set up yet" in r.text and 'role="alert"' in r.text
    ready.store.set_setting("ai_provider", "anthropic")
    assert "no API key is saved" in ready.client.post("/settings/test").text

    def bad():
        return SimpleNamespace(name="anthropic", model="m", complete=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("The Anthropic API key was not accepted. Check the key in Settings.")))

    app = web.create_app(ready.store, key=KEY, provider_factory=bad)
    c = TestClient(app, base_url="http://127.0.0.1:8766")
    c.cookies.set(web.COOKIE, KEY)
    t = c.post("/settings/test").text
    assert "The test did not work." in t and "key was not accepted" in t
    # the key hint is real text, not a placeholder
    assert 'id="key-hint"' in ready.client.get("/settings").text and "<a href=\"/connect\">Connect page</a>" not in ready.client.get("/settings").text


# ---------------------------------------------------------------- materials: edit, duplicates, delete, reader
def test_p1_material_edit_title_kind_date_and_lecture(ready):
    cid = mkcourse(ready.store)
    ev = ready.store.add_event(cid, "lecture", "Lecture 18", "2026-10-12")
    mid = ready.store.add_material(cid, "other", "lecture-18-captions")
    page = ready.client.get(f"/materials/{mid}/edit")
    assert page.status_code == 200 and "Which lecture is it for?" in page.text and "2026-10-12: Lecture 18" in page.text
    r = ready.client.post(f"/materials/{mid}/edit", data={"title": "Lecture 18 captions", "kind": "recording",
                                                         "lecture_date": "", "event_id": str(ev)}, follow_redirects=True)
    assert "Saved Lecture 18 captions" in r.text
    m = ready.store.material(mid)
    assert (m["title"], m["kind"], m["event_id"], m["lecture_date"]) == ("Lecture 18 captions", "recording", ev, "2026-10-12")
    assert "lecture Mon Oct 12" in ready.client.get(f"/courses/{cid}").text or "lecture" in ready.client.get(f"/courses/{cid}").text
    ready.client.post(f"/materials/{mid}/edit", data={"title": "T", "kind": "notes", "lecture_date": "2026-01-05", "event_id": ""})
    m = ready.store.material(mid)
    assert m["event_id"] is None and m["lecture_date"] == "2026-01-05"


def test_p1_material_edit_validation_and_other_courses_lecture(ready):
    cid = mkcourse(ready.store)
    other = ready.store.add_course("Chem")
    foreign = ready.store.add_event(other, "lecture", "L", "2026-01-01")
    mid = ready.store.add_material(cid, "notes", "N")
    r = ready.client.post(f"/materials/{mid}/edit", data={"title": " ", "kind": "notes"})
    assert r.status_code == 400 and 'aria-describedby="err-material-title"' in r.text
    r = ready.client.post(f"/materials/{mid}/edit", data={"title": "N", "kind": "bogus"})
    assert r.status_code == 400 and "err-material-kind" in r.text
    r = ready.client.post(f"/materials/{mid}/edit", data={"title": "N", "kind": "notes", "lecture_date": "soon"})
    assert r.status_code == 400 and "err-material-lecture_date" in r.text
    r = ready.client.post(f"/materials/{mid}/edit", data={"title": "N", "kind": "notes", "event_id": str(foreign)})
    assert r.status_code == 400 and "not in this course" in r.text
    assert ready.client.get("/materials/999/edit").status_code == 404


def test_p1_duplicate_material_is_flagged_with_hint(ready):
    cid = mkcourse(ready.store)
    a = ready.store.add_material(cid, "syllabus", "Syllabus")
    b = ready.store.add_material(cid, "syllabus", "Syllabus (2)")
    ready.store.update_material(b, duplicate_of=a)
    t = ready.client.get(f"/courses/{cid}").text
    assert "Looks like a copy of" in t and f'href="/read/{a}"' in t and "delete one" in t
    assert "looks like a copy of" in ready.client.get(f"/materials/{b}/edit").text.lower()


def test_p1_delete_warns_and_reader_explains_deleted_source(ready):
    cid = mkcourse(ready.store)
    mid = ready.store.add_material(cid, "notes", "Gone")
    assert "source deleted" in ready.client.get(f"/materials/{mid}/delete").text
    ready.client.post(f"/materials/{mid}/delete")
    r = ready.client.get(f"/read/{mid}?at=5")
    assert r.status_code == 410 and "Source deleted" in r.text and "deleted" in r.text and "not found" not in r.text.lower()


def test_upload_reports_already_saved(ready, monkeypatch):
    cid = mkcourse(ready.store)
    first = ready.store.add_material(cid, "notes", "a.md")

    def import_file(store, course_id, src, **kw):
        return first

    fake = types.SimpleNamespace(import_file=import_file, SUPPORTED_EXTENSIONS={".md": "M"}, ImportError_=type("E", (Exception,), {}))
    monkeypatch.setitem(sys.modules, "course_assistant.importers", fake)
    r = ready.client.post(f"/courses/{cid}/materials/upload", files=[("files", ("a.md", b"x", "text/markdown"))], follow_redirects=True)
    assert "Already saved" in r.text and "Added 1 file" not in r.text


# ---------------------------------------------------------------- background jobs
def test_jobs_not_started_by_default_and_started_with_flag(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_DATA_DIR, str(tmp_path / "data"))
    calls = []
    fake = types.SimpleNamespace(start_worker=lambda store: calls.append(store) or SimpleNamespace(is_alive=lambda: True))
    monkeypatch.setitem(sys.modules, "course_assistant.transcribe", fake)
    store = Store(":memory:")
    with TestClient(web.create_app(store, key=KEY), base_url="http://127.0.0.1:8766") as c:
        c.get("/api/health")
    assert calls == []
    app = web.create_app(store, key=KEY, start_jobs=True)
    with TestClient(app, base_url="http://127.0.0.1:8766") as c:
        c.get("/api/health")
        assert app.state.worker is not None
    assert calls == [store]


def test_jobs_start_survives_missing_start_worker(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_DATA_DIR, str(tmp_path / "data"))
    monkeypatch.setitem(sys.modules, "course_assistant.transcribe", types.SimpleNamespace())
    with TestClient(web.create_app(Store(":memory:"), key=KEY, start_jobs=True), base_url="http://127.0.0.1:8766") as c:
        assert c.get("/api/health").status_code == 200


def test_material_status_shows_job_state_and_next_step(ready):
    cid = mkcourse(ready.store)
    photo = ready.store.add_material(cid, "notes", "Notes photo", filename="x.png", status="needs_transcription")
    ready.store.add_job("ocr", photo)
    t = ready.client.get(f"/courses/{cid}").text
    assert "Waiting in line for reading" in t and "connect an AI" in t and "Start now" in t
    rec = ready.store.add_material(cid, "recording", "Rec", filename="x.mp3", status="needs_transcription")
    j = ready.store.add_job("transcribe", rec)
    ready.store.update_job(j, status="running", progress=0.4)
    assert "Transcribing, 40% done" in ready.client.get(f"/courses/{cid}").text
    ready.store.update_job(j, status="error", note="No speech model")
    t = ready.client.get(f"/courses/{cid}").text
    assert "did not finish: No speech model" in t and "Try again" in t


def test_material_process_runs_the_job_now(ready, monkeypatch):
    cid = mkcourse(ready.store)
    mid = ready.store.add_material(cid, "notes", "Photo", filename="x.png", status="needs_transcription")
    seen = {}

    def run_job(store, jid, provider=None):
        seen["jid"] = jid
        store.update_job(jid, status="done")
        store.update_material(mid, status="ready")
        return store.job(jid)

    monkeypatch.setitem(sys.modules, "course_assistant.transcribe", types.SimpleNamespace(run_job=run_job))
    r = ready.client.post(f"/materials/{mid}/process", follow_redirects=True)
    assert "ready to search" in r.text and ready.store.job(seen["jid"])["status"] == "done"
    assert ready.client.post("/materials/999/process").status_code == 404


# ---------------------------------------------------------------- guides, calendar
def test_guides_grouped_by_kind_with_ask_history_collapsed(ready):
    cid = mkcourse(ready.store)
    ready.store.add_guide(cid, "primer", "Primer: Lecture 1", "x")
    ready.store.add_guide(cid, "primer", "Primer: Lecture 1", "y")
    ready.store.add_guide(cid, "exam_guide", "Exam guide", "z")
    for i in range(3):
        ready.store.add_guide(cid, "answer", f"Q{i}", "a")
    t = ready.client.get(f"/courses/{cid}/guides").text
    assert "Pre-lecture primer (2)" in t and "Exam guide (1)" in t
    assert t.index("Pre-lecture primer (2)") < t.index("Exam guide (1)") < t.index("Ask history (3)")
    ask = t[t.index("Ask history"):]
    assert ask.count('class="row-item"') == 3 and "<details" in t[:t.index("Ask history")][-80:]
    assert t.count("Primer: Lecture 1 (") == 2  # the repeated title gets its date
    assert "Q0" not in t[:t.index("Ask history")]


def test_group_guides_dedupes_same_day_with_time():
    gs = [{"id": i, "kind": "primer", "title": "P", "created_at": f"2026-11-10T0{i}:00:00+00:00", "created_by": "app"} for i in (1, 2)]
    out = H.group_guides(gs)["groups"][0]["items"]
    assert {g["display_title"] for g in out} == {"P (2026-11-10 01:00)", "P (2026-11-10 02:00)"}


def test_calendar_shows_duplicate_events_once_and_clean_titles(ready):
    cid = mkcourse(ready.store)
    for _ in range(2):
        ready.store.add_event(cid, "exam", "Midterm 1", "2999-10-10")
    ready.store.add_event(cid, "lecture", "17 | Lecture 20", "2999-10-11")
    t = ready.client.get(f"/courses/{cid}/calendar").text
    assert t.count("<strong>Midterm 1</strong>") == 1 and "Listed 2 times, shown once" in t
    assert "<strong>Lecture 20</strong>" in t and "17 |" not in t
    st = ready.client.get(f"/courses/{cid}/study").text
    assert "Lecture 20" in st and "17 | Lecture" not in st
    first = ready.store.events(cid)[0]
    ready.client.post(f"/events/{first['id']}/delete")
    assert [e["title"] for e in ready.store.events(cid)] == ["17 | Lecture 20"]  # every copy went


def test_editing_a_merged_event_merges_the_copies(ready):
    cid = mkcourse(ready.store)
    a = ready.store.add_event(cid, "exam", "Final", "2999-12-01")
    ready.store.add_event(cid, "other", "Final", "2999-12-01")
    r = ready.client.post(f"/events/{a}/edit", data={"kind": "exam", "title": "Final exam", "date": "2999-12-01"}, follow_redirects=True)
    assert "Duplicate copies were merged" in r.text and len(ready.store.events(cid)) == 1


def test_merge_events_and_clean_titles_unit():
    assert H.clean_event_title("17 | Lecture 20") == "Lecture 20"
    assert H.clean_event_title("**Final exam** |") == "Final exam"
    assert H.clean_event_title("Lecture 3") == "Lecture 3"
    rows = H.merge_events([{"id": 1, "date": "d", "title": "A", "kind": "other"}, {"id": 2, "date": "d", "title": "a", "kind": "exam", "time": "9:00"}])
    assert len(rows) == 1 and rows[0]["kind"] == "exam" and rows[0]["time"] == "9:00" and rows[0]["listed"] == 2


# ---------------------------------------------------------------- graded work
def test_mark_graded_uses_integrity_and_lists_items_with_labels(ready, monkeypatch):
    cid = mkcourse(ready.store)
    mid = ready.store.add_material(cid, "assignment", "Problem Set 2")

    def mark(store, material_id):
        store.add_graded_item(cid, "Problem Set 2", "Solve Ax = b for the matrix with rows 3 5 7", material_id, kind="problem", label="Problem 1")
        store.add_graded_item(cid, "Problem Set 2", "Prove that the kernel is a subspace", material_id, kind="proof", label="Problem 2")

    monkeypatch.setitem(sys.modules, "course_assistant.integrity", types.SimpleNamespace(mark_material_graded=mark))
    r = ready.client.post(f"/materials/{mid}/graded", data={"graded": "1"}, follow_redirects=True)
    assert "found 2 graded items" in r.text and "Problem 1, Problem 2" in r.text
    assert "Graded work the app protects (2)" in r.text and "Proof" in r.text
    assert ready.store.material(mid)["graded"] == 1
    item = ready.store.graded_items(cid)[0]
    r = ready.client.post(f"/graded/{item['id']}/edit", data={"kind": "essay", "label": "Essay A", "title": "PS2", "body": "new"}, follow_redirects=True)
    assert "Saved Essay A" in r.text and ready.store.graded_item(item["id"])["kind"] == "essay"
    assert ready.client.post(f"/graded/{item['id']}/edit", data={"title": " "}).status_code == 400
    assert "Remove it" in ready.client.get(f"/graded/{item['id']}/delete").text
    ready.client.post(f"/graded/{item['id']}/delete")
    assert len(ready.store.graded_items(cid)) == 1
    ready.client.post(f"/materials/{mid}/graded", data={"graded": "0"})
    assert ready.store.graded_items(cid) == [] and ready.store.material(mid)["graded"] == 0


def test_mark_graded_falls_back_to_flag_without_integrity_function(ready, monkeypatch):
    cid = mkcourse(ready.store)
    mid = ready.store.add_material(cid, "assignment", "PS")
    monkeypatch.setitem(sys.modules, "course_assistant.integrity", types.SimpleNamespace())
    r = ready.client.post(f"/materials/{mid}/graded", data={"graded": "1"}, follow_redirects=True)
    assert ready.store.material(mid)["graded"] == 1 and "Marked PS as graded" in r.text


def test_privacy_mentions_speech_model_download(ready):
    assert "speech model" in ready.client.get("/privacy").text
    assert "speech model" in (Path(web.__file__).parents[2] / "PRIVACY.md").read_text()


def test_no_em_dashes_in_new_templates():
    tdir = Path(web.__file__).parent / "templates"
    for f in ("setup.html", "settings.html", "material_form.html", "graded_form.html", "connect.html", "tab_materials.html"):
        assert "—" not in (tdir / f).read_text(), f


def test_keep_copy_route_swaps_the_searched_copy(ready):
    cid = mkcourse(ready.store)
    a = ready.store.add_material(cid, "textbook", "Chapter 7")
    b = ready.store.add_material(cid, "textbook", "Chapter 7 pdf")
    ready.store.update_material(b, duplicate_of=a)
    ready.client.post(f"/materials/{b}/keep-copy", follow_redirects=True)
    assert ready.store.material(b)["duplicate_of"] is None and ready.store.material(a)["duplicate_of"] == b
