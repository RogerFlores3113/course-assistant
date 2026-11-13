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
    calls = []

    def boom(*a, **k):
        calls.append(1)
        raise AssertionError("the model must not be called")
    app = web.create_app(store, key=KEY, provider_factory=lambda: SimpleNamespace(name="f", model="m", complete=boom))
    c = TestClient(app, base_url=HOST)
    c.cookies.set(web.COOKIE, KEY)
    return SimpleNamespace(store=store, client=c, calls=calls)


def day(n):
    return (date.today() + timedelta(days=n)).isoformat()


def course(store):
    return store.add_course("Chemistry", code="CHEM 1", term="Fall", color="green")


def with_material(store, cid, title="Chapter 1"):
    mid = store.add_material(cid, "reading", title)
    store.replace_chunks(mid, [{"text": "Molarity is moles per litre.", "loc_kind": "section", "loc_start": "1",
                                "loc_label": "Section 1"}])
    return mid, store.chunks_of(mid)[0]["id"]


# ---- 1. cram loop
def test_cram_round_end_shows_counts_redo_and_weak_topics(ready):
    s, c = ready.store, ready.client
    cid = course(s)
    _, chunk = with_material(s, cid, "Chapter 1")
    a = s.add_flashcard(cid, "Qa", "Aa", chunk_id=chunk)
    b = s.add_flashcard(cid, "Qb", "Ab", chunk_id=chunk)
    d = s.add_flashcard(cid, "Qc", "Ac", deck="Typed deck")
    r = c.post(f"/flashcards/{cid}/review", data={"card_id": a, "grade": "again", "cram": "1", "done": "", "missed": ""},
               follow_redirects=False)
    assert "missed=" + str(a) in r.headers["location"]
    url = r.headers["location"]
    r = c.post(f"/flashcards/{cid}/review", data={"card_id": b, "grade": "good", "cram": "1", "done": str(a), "missed": str(a)},
               follow_redirects=False)
    r = c.post(f"/flashcards/{cid}/review", data={"card_id": d, "grade": "hard", "cram": "1", "done": f"{a},{b}",
                                                  "missed": str(a)}, follow_redirects=False)
    t = c.get(r.headers["location"]).text
    assert "1 right, 2 to review, 3 in total" in t
    assert "Redo the ones I missed (2)" in t
    assert "Chapter 1: 1 card" in t and "Typed deck: 1 card" in t
    m = re.search(r'href="(/flashcards/\d+\?cram=1&amp;only=[\d,]+)"', t)
    assert m
    redo = c.get(m.group(1).replace("&amp;", "&")).text
    assert "Card 1 of 2" in redo and "Qb" not in redo


def test_flashcard_from_label_links_to_the_passage(ready):
    s, c = ready.store, ready.client
    cid = course(s)
    mid, chunk = with_material(s, cid)
    s.add_flashcard(cid, "Q", "A", chunk_id=chunk, source_label="Chapter 1, Section 1")
    t = c.get(f"/flashcards/{cid}").text
    assert f'href="/read/{mid}?at={chunk}#c{chunk}"' in t


def test_flashcard_edit_and_delete_with_confirm_page(ready):
    s, c = ready.store, ready.client
    cid = course(s)
    card = s.add_flashcard(cid, "Old front", "Old back")
    assert "Edit this card" in c.get(f"/flashcards/{cid}").text
    assert "Old front" in c.get(f"/flashcards/card/{card}/edit").text
    bad = c.post(f"/flashcards/card/{card}/edit", data={"front": " ", "back": "x"})
    assert bad.status_code == 400 and 'aria-invalid="true"' in bad.text
    ok = c.post(f"/flashcards/card/{card}/edit", data={"front": "New front", "back": "New back"}, follow_redirects=False)
    assert ok.status_code == 303 and s.flashcard(card)["front"] == "New front"
    conf = c.get(f"/flashcards/card/{card}/delete")
    assert "Delete this flashcard?" in conf.text and s.flashcard(card)
    c.post(f"/flashcards/card/{card}/delete", follow_redirects=False)
    assert s.flashcard(card) is None


def test_material_filter_with_no_cards_does_not_say_all_caught_up(ready):
    s, c = ready.store, ready.client
    cid = course(s)
    mid, _ = with_material(s, cid)
    s.add_flashcard(cid, "Q", "A", deck="Elsewhere")
    t = c.get(f"/flashcards/{cid}?material={mid}").text
    assert "No cards for this material yet" in t and "All caught up" not in t


# ---- 2. exam countdown
def test_dashboard_and_study_tab_show_exam_countdown_and_prefill(ready):
    s, c = ready.store, ready.client
    cid = course(s)
    eid = s.add_event(cid, "exam", "Midterm 1", day(5))
    s.add_event(cid, "exam", "Final", day(40))
    dash = c.get("/").text
    assert "Exam in 5 days" in dash and "Midterm 1" in dash and f"/courses/{cid}/study?exam={eid}" in dash
    study = c.get(f"/courses/{cid}/study").text
    assert "Exam in 5 days" in study and "Start an exam guide" in study
    pre = c.get(f"/courses/{cid}/study?exam={eid}").text
    assert re.search(rf'<option value="{eid}"\s+selected', pre)


def test_no_exam_offers_add_your_exam_date(ready):
    s, c = ready.store, ready.client
    cid = course(s)
    assert "Add your exam date" in c.get("/").text
    assert "Add your exam date" in c.get(f"/courses/{cid}/study").text


# ---- 3. upload lecture picker
def test_upload_has_lecture_picker_and_sets_event_id(ready):
    s, c = ready.store, ready.client
    cid = course(s)
    e1 = s.add_event(cid, "lecture", "Lecture 1: Atoms", day(-14))
    e2 = s.add_event(cid, "lecture", "Lecture 2: Bonds", day(-7))
    t = c.get(f"/courses/{cid}").text
    assert 'name="event_id"' in t and "No particular lecture" in t
    assert t.index("Lecture 2: Bonds") < t.index("Lecture 1: Atoms")  # newest first
    r = c.post(f"/courses/{cid}/materials/upload", data={"event_id": str(e1), "kind": "notes"},
               files={"files": ("atoms.md", b"# Atoms\n\nAn atom has a nucleus.\n", "text/markdown")}, follow_redirects=False)
    assert r.status_code == 303
    m = s.materials(cid)[0]
    assert m["event_id"] == e1 and m["lecture_date"] == day(-14)
    bad = c.post(f"/courses/{cid}/materials/upload", data={"event_id": "9999"},
                 files={"files": ("x.md", b"# X\n\ntext here.\n", "text/markdown")}, follow_redirects=False)
    assert len(s.materials(cid)) == 1


# ---- 4. catch me up
def test_dashboard_catch_me_up_for_latest_past_lecture(ready):
    s, c = ready.store, ready.client
    cid = course(s)
    s.add_event(cid, "lecture", "Lecture 1: Atoms", day(-14))
    e2 = s.add_event(cid, "lecture", "Lecture 2: Bonds", day(-7))
    s.add_event(cid, "lecture", "Lecture 3: Future", day(7))
    t = c.get("/").text
    assert "Catch me up" in t and f'action="/courses/{cid}/study/missed"' in t
    assert f'name="event_id" value="{e2}"' in t


def test_catch_up_with_nothing_saved_says_so_before_calling_the_model(ready):
    s, c = ready.store, ready.client
    cid = course(s)
    e = s.add_event(cid, "lecture", "Lecture 2: Bonds", day(-7))
    r = c.post(f"/courses/{cid}/study/missed", data={"event_id": str(e)}, follow_redirects=True)
    assert "Nothing is saved for" in r.text
    assert ready.calls == [] and s.guides(cid) == []


# ---- 5. accessibility
def test_setup_skip_link_first_and_skip_for_now_not_at_the_bottom(ready):
    t = ready.client.get("/setup?first=1").text
    body = t[t.index("<body>"):]
    first_focusable = re.search(r"<(a|button|input|select|textarea)\b", body)
    assert 'class="skip-link"' in body[:first_focusable.end() + 40]
    assert body.index("Skip for now") < body.index('id="opt-a"')
    assert '<p class="muted small"></p>' not in t


def test_copy_note_names_the_original_once(ready):
    s, c = ready.store, ready.client
    cid = course(s)
    a = s.add_material(cid, "reading", "Diagonalization")
    b = s.add_material(cid, "reading", "Diagonalization copy")
    s.update_material(b, duplicate_of=a)
    t = c.get(f"/courses/{cid}").text
    assert "Looks like a copy" not in t
    assert t.count("Copy of ") == 1


def test_app_js_restores_focus_to_submit_button_after_failed_request():
    js = (STATIC / "app.js").read_text()
    after_request = js.split('"htmx:afterRequest"')[1].split("});")[0]
    assert "btn.disabled = false" in after_request and "btn.focus()" in after_request
    assert "pressed = document.activeElement" in js
