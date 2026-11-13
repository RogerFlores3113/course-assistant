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


def mk(store):
    return store.add_course("Chemistry", code="CHEM 1", term="Fall", color="green")


def test_end_states_are_focusable_and_announced(ready):
    s, c = ready.store, ready.client
    cid = mk(s)
    t = c.get(f"/flashcards/{cid}").text
    assert 'data-end-state' in t and 'id="end-h" tabindex="-1"' in t
    mid = s.add_material(cid, "reading", "Ch 1")
    s.replace_chunks(mid, [{"text": "x", "loc_kind": "section", "loc_start": "1", "loc_label": "S1", "heading": "Molarity"}])
    ch = s.chunks_of(mid)[0]["id"]
    a = s.add_flashcard(cid, "Q", "A", chunk_id=ch)
    r = c.post(f"/flashcards/{cid}/review", data={"card_id": a, "grade": "again", "cram": "1"}, follow_redirects=False)
    t = c.get(r.headers["location"]).text
    assert 'data-end-state' in t and 'id="round-h" tabindex="-1"' in t
    js = (STATIC / "app.js").read_text()
    assert "[data-end-state]" in js and "focusEl(h)" in js


def test_no_duplicate_aria_describedby_on_lecture_pickers(ready):
    s, c = ready.store, ready.client
    cid = mk(s)
    s.add_event(cid, "lecture", "Lecture 1: Atoms", day(-7))
    mid = s.add_material(cid, "notes", "N")
    for url in (f"/courses/{cid}", f"/materials/{mid}/edit"):
        t = c.get(url).text
        for tag in re.findall(r"<select[^>]*>", t):
            assert tag.count("aria-describedby") <= 1, tag
    assert 'aria-describedby="up-lec-help"' in c.get(f"/courses/{cid}").text


def test_delete_card_confirmation_is_plain_text(ready):
    s, c = ready.store, ready.client
    cid = mk(s)
    card = s.add_flashcard(cid, "What does the trace of $A$ equal?", "The sum of the diagonal.")
    t = c.get(f"/flashcards/card/{card}/delete").text
    assert "<p>" in t and "$A$" not in t and "trace of A equal" in t
    long_card = s.add_flashcard(cid, "x" * 115 + " $a + b$ end", "B")
    t2 = c.get(f"/flashcards/card/{long_card}/delete").text
    assert "$a" not in t2  # a cut formula is dropped, never shown half open


def test_dashboard_ask_link_focuses_question_box(ready):
    s, c = ready.store, ready.client
    cid = mk(s)
    s.add_event(cid, "lecture", "Lecture 4: Gases", day(2))
    s.add_event(cid, "other", "No class (Thanksgiving)", day(3))
    t = c.get("/").text
    assert f'href="/courses/{cid}/ask?about=Lecture%204%3A%20Gases#ask-question"' in t
    assert t.count("#ask-question") == 1  # none for a non-class event
    assert 'id="ask-question"' in c.get(f"/courses/{cid}/ask").text


def test_catch_me_up_picker_lists_unreviewed_past_lectures_newest_first(ready):
    s, c = ready.store, ready.client
    cid = mk(s)
    e1 = s.add_event(cid, "lecture", "Lecture 1: Atoms", day(-21))
    e2 = s.add_event(cid, "lecture", "Lecture 2: Bonds", day(-14))
    e3 = s.add_event(cid, "lecture", "Lecture 3: Gases", day(-7))
    s.add_event(cid, "lecture", "Lecture 4: Future", day(7))
    s.add_guide(cid, "missed", "What you missed: Gases", "text", event_id=e3)
    t = c.get("/").text
    picker = t.split("Lecture to catch up on")[1].split("</select>")[0]
    assert [int(o) for o in re.findall(r'<option value="(\d+)">', picker)] == [e2, e1]  # newest first, e3 already done
    assert "Lecture 4: Future" not in picker


def test_weak_topic_uses_section_heading(ready):
    s, c = ready.store, ready.client
    cid = mk(s)
    mid = s.add_material(cid, "reading", "Chapter 1")
    s.replace_chunks(mid, [{"text": "x", "loc_kind": "section", "loc_start": "1", "loc_label": "S", "heading": "Molarity"}])
    ch = s.chunks_of(mid)[0]["id"]
    a = s.add_flashcard(cid, "Qa", "Aa", chunk_id=ch)
    r = c.post(f"/flashcards/{cid}/review", data={"card_id": a, "grade": "again", "cram": "1"}, follow_redirects=False)
    t = c.get(r.headers["location"]).text
    assert "Molarity (Chapter 1): 1 card" in t


def test_photo_status_polls_while_running_and_stops_when_done(ready):
    s, c = ready.store, ready.client
    cid = mk(s)
    mid = s.add_material(cid, "notes", "Photo", filename="p.jpg", original_name="p.jpg", status="needs_transcription")
    job = s.add_job("ocr", mid)
    s.update_job(job, status="running")
    t = c.get(f"/materials/{mid}/status").text
    assert f'hx-get="/materials/{mid}/status"' in t and 'hx-trigger="every 4s"' in t and "Reading the photo" in t
    assert 'hx-get="/materials/' in c.get(f"/courses/{cid}").text
    s.update_job(job, status="done")
    s.update_material(mid, status="ready")
    done = c.get(f"/materials/{mid}/status").text
    assert "hx-get" not in done and "Photo read" in done
    assert c.get("/materials/9999/status").status_code == 404
    js = (STATIC / "app.js").read_text()
    assert 'hasAttribute("data-poll")' in js


def test_busy_text_names_graded_checks(ready):
    s, c = ready.store, ready.client
    cid = mk(s)
    ask = c.get(f"/courses/{cid}/ask").text
    study = c.get(f"/courses/{cid}/study").text
    assert "Checking this against your graded work..." in ask
    assert study.count("Checking this against your graded work...") >= 3


def test_app_js_announces_partly_covered_answers():
    js = (STATIC / "app.js").read_text()
    assert "[data-partly-covered]" in js and "Part of your question is not covered." in js


def test_css_has_type_scale_buttons_and_print():
    css = (STATIC / "style.css").read_text()
    for needle in ("--fs-md", "--space-4", ".btn.secondary", ".btn.danger", "@media print", "[aria-invalid=\"true\"]"):
        assert needle in css
