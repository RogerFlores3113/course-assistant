"""Wiring: study options, cram mode, fetch block reasons, duplicates and lecture links, worker, MCP results."""
import sys
import types
from datetime import date, timedelta
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from course_assistant import disclaimer, mcp_server, offerings, paths, web
from course_assistant.fetcher import FetchBlocked
from course_assistant.store import MATERIAL_KINDS, Store

KEY = "k" * 40
HOST = "http://127.0.0.1:8766"


@pytest.fixture()
def ready(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_DATA_DIR, str(tmp_path / "data"))
    store = Store(":memory:")
    store.set_setting(disclaimer.ACK_SETTING, "2026-01-01T00:00:00+00:00")
    app = web.create_app(store, key=KEY, provider_factory=lambda: SimpleNamespace(name="f", model="m"))
    c = TestClient(app, base_url=HOST)
    c.cookies.set(web.COOKIE, KEY)
    return SimpleNamespace(store=store, client=c, app=app)


def course(store):
    return store.add_course("Chemistry", code="CHEM 1", term="Fall", color="green")


def day(n):
    return (date.today() + timedelta(days=n)).isoformat()


# ---- study tab
def test_study_tab_shows_material_pickers(ready):
    cid = course(ready.store)
    ready.store.add_material(cid, "slides", "Atoms slides")
    ready.store.add_event(cid, "exam", "Midterm", day(10))
    t = ready.client.get(f"/courses/{cid}/study").text
    assert 'name="material_ids"' in t and 'name="material_id"' in t and "Atoms slides" in t


def test_study_form_passes_material_ids_and_note(ready, monkeypatch):
    cid = course(ready.store)
    seen = {}

    def exam_guide(store, course_id, provider, *, event_id=None, material_ids=None):
        seen["exam"] = material_ids
        return store.add_guide(course_id, "exam", "Exam guide", "Body")

    def make_flashcards(store, course_id, provider, *, material_id=None, topic="", count=12):
        seen["fc"] = material_id
        res = type("R", (list,), {"note": "Made 1 card. 2 you already have were skipped."})([1])
        return res

    def practice(store, course_id, provider, topic, count=5):
        return store.add_guide(course_id, "twin", "Twin", "Body")

    mod = types.SimpleNamespace(exam_guide=exam_guide, make_flashcards=make_flashcards, practice=practice,
                                ready_message=lambda k: f"READY {k}")
    monkeypatch.setitem(sys.modules, "course_assistant.study", mod)
    h = {"HX-Request": "true"}
    r = ready.client.post(f"/courses/{cid}/study/exam", data={"material_ids": ["3", "4"]}, headers=h)
    assert seen["exam"] == [3, 4] and "READY exam" in r.text
    r = ready.client.post(f"/courses/{cid}/study/flashcards", data={"material_id": "7"}, headers=h)
    assert seen["fc"] == 7 and "2 you already have were skipped" in r.text
    r = ready.client.post(f"/courses/{cid}/study/practice", data={"topic": "x"}, headers=h)
    assert "READY twin" in r.text  # picked from the saved guide's kind, not the action


# ---- cram mode
def cards(store, cid, n=3):
    return [store.add_flashcard(cid, f"Q{i}", f"A{i}") for i in range(n)]


def test_cram_page_toggle_and_material_filter(ready):
    cid = course(ready.store)
    ids = cards(ready.store, cid)
    ready.store.update_flashcard(ids[0], due_at="2999-01-01T00:00:00+00:00")  # not due
    ready.store.add_material(cid, "slides", "Atoms slides")
    normal = ready.client.get(f"/flashcards/{cid}").text
    assert "Cram mode" in normal and "data-math" in normal and 'id="reveal"' in normal
    cram = ready.client.get(f"/flashcards/{cid}?cram=1").text
    assert "Cram mode: on" in cram and 'name="cram" value="1"' in cram and "Card 1 of 3" in cram
    assert "Atoms slides" in cram  # material filter


def test_cram_review_passes_cram_and_exam_date(ready, monkeypatch):
    cid = course(ready.store)
    ids = cards(ready.store, cid)
    ready.store.add_event(cid, "exam", "Midterm", day(3))
    calls = []
    real = __import__("course_assistant.study", fromlist=["x"])
    monkeypatch.setattr(real, "review_flashcard", lambda store, card_id, grade, **kw: calls.append((card_id, grade, kw)))
    r = ready.client.post(f"/flashcards/{cid}/review", data={"card_id": ids[0], "grade": "good", "cram": "1", "done": ""},
                          follow_redirects=False)
    assert calls[0][2]["cram"] is True and calls[0][2]["exam_date"] == day(3)
    assert r.headers["location"] == f"/flashcards/{cid}?cram=1&done={ids[0]}"
    nxt = ready.client.get(r.headers["location"]).text
    assert "Card 2 of 3" in nxt and "Q0" not in nxt
    # a normal review far from the exam passes no exam_date and no cram
    calls.clear()
    ready.store._exec("DELETE FROM events")
    ready.store.add_event(cid, "exam", "Final", day(40))
    ready.client.post(f"/flashcards/{cid}/review", data={"card_id": ids[1], "grade": "good"}, follow_redirects=False)
    assert calls[0][2] == {}


def test_real_cram_caps_interval(ready):
    cid = course(ready.store)
    (a,) = cards(ready.store, cid, 1)
    ready.store.add_event(cid, "exam", "Midterm", day(4))
    ready.client.post(f"/flashcards/{cid}/review", data={"card_id": a, "grade": "easy", "cram": "1"}, follow_redirects=False)
    assert ready.store.flashcard(a)["interval_days"] <= 2


def test_cram_material_filter_ignores_other_course(ready):
    cid = course(ready.store)
    other = ready.store.add_course("Other")
    m = ready.store.add_material(other, "slides", "Other slides")
    cards(ready.store, cid)
    assert ready.client.get(f"/flashcards/{cid}?cram=1&material={m}").status_code == 200


# ---- fetch errors
class Blocker:
    def __init__(self, reason):
        self.reason = reason

    def get(self, url):
        raise FetchBlocked("That address is on your own network, so it was not opened." if self.reason == "private"
                           else "robots says no", self.reason)

    def why_blocked(self, url):
        return (self.reason, "why text")


def test_offerings_say_the_right_reason(ready):
    cid = course(ready.store)
    (cand,) = offerings.candidates_from_urls(ready.store, cid, ["http://192.168.0.5/page"], fetcher=Blocker("private"))
    assert cand["status"] == "blocked" and "robots" not in cand["note"] and "network" in cand["note"]


def test_offering_note_falls_back_to_why_blocked():
    class E(FetchBlocked):
        pass
    note = offerings._block_note(E("", "private"), Blocker("private"), "http://x")
    assert note == "why text"
    assert "private" in offerings._block_note(E("", "private"), None, "")
    assert "robots.txt" in offerings._block_note(E("", "robots"), None, "")


def test_web_offering_save_uses_exception_text(ready, monkeypatch):
    cid = course(ready.store)
    c = ready.store.add_offering_candidate(cid, "http://10.0.0.1/x", title="t")

    def import_url(store, course_id, url, **kw):
        raise FetchBlocked("That address is on your own network, so it was not opened.", "private")

    monkeypatch.setitem(sys.modules, "course_assistant.importers", types.SimpleNamespace(import_url=import_url))
    r = ready.client.post(f"/courses/{cid}/offerings/save", data={"pick": str(c)}, follow_redirects=True)
    assert "your own network" in r.text and "robots.txt asks" not in r.text


# ---- duplicates and lecture links
def test_materials_tab_copy_and_lecture_link(ready):
    st = ready.store
    cid = course(st)
    ev = st.add_event(cid, "lecture", "Lecture 3 | Gases", day(-3))
    orig = st.add_material(cid, "slides", "Gases")
    dup = st.add_material(cid, "slides", "Gases again")
    st.update_material(dup, duplicate_of=orig)
    st.update_material(orig, event_id=ev)
    t = ready.client.get(f"/courses/{cid}").text
    assert "Copy of" in t and "Not a copy" in t and "Linked to Lecture 3" in t
    r = ready.client.post(f"/materials/{dup}/not-copy", follow_redirects=True)
    assert st.material(dup)["duplicate_of"] is None
    assert "Not a copy" not in r.text


# ---- worker
def test_worker_started_once_and_stopped(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_DATA_DIR, str(tmp_path / "data"))
    events = []

    class W:
        def stop(self):
            events.append("stop")

        def is_alive(self):
            return True

    mod = types.SimpleNamespace(start_worker=lambda store: events.append("start") or W())
    monkeypatch.setitem(sys.modules, "course_assistant.transcribe", mod)
    app = web.create_app(Store(":memory:"), key=KEY, start_jobs=True)
    with TestClient(app, base_url=HOST) as c:
        c.get("/api/health")
        c.get("/api/health")
        assert events == ["start"]
    assert events == ["start", "stop"] and app.state.worker is None


# ---- kinds
def test_reading_kind_everywhere(ready):
    assert "reading" in MATERIAL_KINDS
    cid = course(ready.store)
    t = ready.client.get(f"/courses/{cid}").text
    assert 'value="reading"' in t
    from course_assistant import cli
    ns = cli.build_parser().parse_args(["import", "Chem", "f.pdf", "--kind", "reading"])
    assert ns.kind == "reading"


# ---- MCP
def test_mcp_note_reports_copy_and_lecture(monkeypatch):
    st = Store(":memory:")
    mcp_server.set_store(st)
    cid = st.add_course("Chem")
    st.add_event(cid, "lecture", "Lecture 20: Gases", day(1))
    body = " ".join(f"Gas law fact number {i} relates pressure to volume at temperature {i * 7}." for i in range(12))
    mcp_server.add_note(cid, "Lecture 20 notes", body)
    out = mcp_server.add_note(cid, "Lecture 20 notes copy", body + " extra")
    assert "Linked to the lecture Lecture 20" in out
    assert "looks like a copy of 'Lecture 20 notes'" in out
    ups = mcp_server.list_upcoming(days=5)
    assert "Lecture 20: Gases" in ups
