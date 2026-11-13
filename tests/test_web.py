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
    c.post(f"/courses/{cid}/offerings/paste", data={"links": "https://example.edu/math2210/f24\njavascript:x", "term_label": "Fall 2024"})
    cand = ready.store.offering_candidates(cid)
    assert len(cand) == 1 and cand[0]["found_by"] == "paste"
    page = c.get(f"/courses/{cid}/offerings")
    assert "robots.txt" in page.text and 'name="pick"' in page.text

    def import_url(store, course_id, url, *, kind="website", **kw):
        return store.add_material(course_id, kind, "Old course page", source_url=url)

    def find(store, course_id):
        store.add_offering_candidate(course_id, "https://example.edu/other", title="Spring 2023", found_by="search")

    monkeypatch.setitem(sys.modules, "course_assistant.importers", types.SimpleNamespace(import_url=import_url))
    monkeypatch.setitem(sys.modules, "course_assistant.offerings", types.SimpleNamespace(find_past_offerings=find))
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
