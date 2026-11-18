import re
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from course_assistant import disclaimer, paths, web
from course_assistant.store import Store

KEY = "k" * 40
HOST = "http://127.0.0.1:8766"
STATIC = Path(__file__).parent.parent / "src" / "course_assistant" / "static"


@pytest.fixture()
def ready(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_DATA_DIR, str(tmp_path / "data"))
    store = Store(":memory:")
    store.set_setting(disclaimer.ACK_SETTING, "2026-01-01T00:00:00+00:00")

    def boom(*a, **k):
        raise AssertionError("the model must not be called")
    app = web.create_app(store, key=KEY, provider_factory=lambda: SimpleNamespace(name="f", model="m", complete=boom))
    c = TestClient(app, base_url=HOST)
    c.cookies.set(web.COOKIE, KEY)
    return SimpleNamespace(store=store, client=c)


def day(n):
    return (date.today() + timedelta(days=n)).isoformat()


def test_type_tokens_used_for_base_sizes():
    css = (STATIC / "style.css").read_text()
    head = css.split("Cycle 4")[0]
    for bad in ("font-size: 24px", "font-size: 18px", "font-size: 16px", "font: 15px", "font-size: 13px", "font-size: 12px",
                "font-size: 17px"):
        assert bad not in head, bad
    assert "h1 { font-size: var(--fs-2xl)" in css and "--lh-body" in css


def test_ask_history_has_jump_link_and_collapses_older_turns(ready):
    s, c = ready.store, ready.client
    cid = s.add_course("Chem")
    for i in range(3):
        s.add_chat_message(cid, "user", f"Question number {i}")
        s.add_chat_message(cid, "assistant", f"Answer number {i}")
    t = c.get(f"/courses/{cid}/ask").text
    assert 'href="#latest-answer"' in t and 'id="latest-answer"' in t and "Jump to latest answer" in t
    assert '<details class="older-turns"' in t
    older, latest = t.split("</details>", 1)
    assert "Question number 0" in older and "Question number 1" in older and "Question number 2" not in older
    assert "Question number 2" in latest and "Answer number 2" in latest
    assert "data-jump-latest" in t and "data-jump-latest" in (STATIC / "app.js").read_text()


def test_ask_single_turn_has_no_jump_link(ready):
    s, c = ready.store, ready.client
    cid = s.add_course("Chem")
    s.add_chat_message(cid, "user", "Q")
    s.add_chat_message(cid, "assistant", "A")
    t = c.get(f"/courses/{cid}/ask").text
    assert "Jump to latest" not in t and "older-turns" not in t


def test_dashboard_ask_link_prefills_and_focuses(ready):
    s, c = ready.store, ready.client
    cid = s.add_course("Chem")
    s.add_event(cid, "exam", "Exam 1: Atoms & bonds", day(3))
    d = c.get("/").text
    assert f'/courses/{cid}/ask?about=Exam%201%3A%20Atoms%20%26%20bonds#ask-question' in d
    t = c.get(f"/courses/{cid}/ask?about=Exam%201%3A%20Atoms%20%26%20bonds").text
    assert re.search(r'<textarea id="ask-question"[^>]*autofocus[^>]*>About Exam 1: Atoms &amp; bonds: </textarea>', t)
    assert "data-caret-end" in t and "data-caret-end" in (STATIC / "app.js").read_text()
    # no 'about': an empty box
    assert "></textarea>" in c.get(f"/courses/{cid}/ask").text


def test_delete_card_confirmation_is_plain_text(ready):
    s, c = ready.store, ready.client
    cid = s.add_course("Chem")
    card = s.add_flashcard(cid, r"What is a steady state of a Markov matrix $A$? Use $\frac{1}{2}\lambda_{1}$", "a")
    t = c.get(f"/flashcards/card/{card}/delete").text
    assert "$" not in t.split("<main")[1] and "\\frac" not in t and "data-math" not in t
    assert "steady state of a Markov matrix A" in t


def test_link_suggestion_buttons_and_route(ready):
    s, c = ready.store, ready.client
    cid = s.add_course("Chem")
    ev = s.add_event(cid, "lecture", "Lecture 2: Moles", day(-5))
    s.add_event(cid, "lecture", "Lecture 3: Solutions", day(-2))
    mid = s.add_material(cid, "notes", "Moles notes")
    s.replace_chunks(mid, [{"text": "moles", "loc_kind": "section", "loc_start": "1", "loc_label": "S1", "heading": "Moles"}])
    t = c.get(f"/courses/{cid}/materials").text
    if "Link it?" in t:
        assert f'action="/materials/{mid}/link"' in t and "Link to " in t
    r = c.post(f"/materials/{mid}/link", data={"event_id": ev}, follow_redirects=False)
    assert r.status_code == 303
    m = s.material(mid)
    assert m["event_id"] == ev and m["lecture_date"] == day(-5)
    assert "Linked to" not in "" and "Link it?" not in c.get(f"/courses/{cid}/materials").text


def test_link_route_rejects_other_course_event(ready):
    s, c = ready.store, ready.client
    a, b = s.add_course("A"), s.add_course("B")
    ev = s.add_event(b, "lecture", "Lecture 1", day(-1))
    mid = s.add_material(a, "notes", "N")
    r = c.post(f"/materials/{mid}/link", data={"event_id": ev}, follow_redirects=False)
    assert s.material(mid).get("event_id") is None
    assert r.status_code in (303, 400)
    assert c.post("/materials/9999/link", data={"event_id": ev}).status_code == 404


def test_link_buttons_render_with_study_suggestion(ready):
    s, c = ready.store, ready.client
    cid = s.add_course("Chem")
    ev = s.add_event(cid, "lecture", "Lecture 2: Moles and molar mass", day(-5))
    s.add_event(cid, "lecture", "Lecture 3: Solutions", day(-2))
    mid = s.add_material(cid, "notes", "Moles and molar mass review")
    s.replace_chunks(mid, [{"text": "moles", "loc_kind": "section", "loc_start": "1", "loc_label": "S1", "heading": "Moles"}])
    t = c.get(f"/courses/{cid}/materials").text
    assert "Link it?" in t and f'action="/materials/{mid}/link"' in t and "Link to Lecture 2" in t
    assert f'name="event_id" value="{ev}"' in t
    c.post(f"/materials/{mid}/link", data={"event_id": ev})
    assert "Link it?" not in c.get(f"/courses/{cid}/materials").text


def test_answer_check_note_on_ask_page(ready):
    import json
    s, c = ready.store, ready.client
    cid = s.add_course("Chem")
    assert "Answer check:" not in c.get(f"/courses/{cid}/ask").text
    ids = [s.add_graded_item(cid, f"Problem {i + 1}", "Find x.", kind="problem") for i in range(3)]
    t = c.get(f"/courses/{cid}/ask").text
    assert "Answer check: couldn&#39;t be made for 3 graded items" in t or "Answer check: off for 3 graded items" in t
    for i in ids:
        s.update_graded_item(i, answer_key=json.dumps(["x = 2", "y = 3"]))
    t = c.get(f"/courses/{cid}/ask").text
    assert "Answer check: on for 3 graded items" in t
