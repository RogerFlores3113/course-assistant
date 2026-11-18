"""Cycle 7 source selection, import report and UI regressions. All text is original (kiln, glaze and grid-puzzle lessons)."""
import pytest

from course_assistant import retrieval, study
from course_assistant.store import Store
from test_fx6_sources import FILLER, _client, mat


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("COURSE_ASSISTANT_DATA_DIR", str(tmp_path / "data"))
    s = Store(":memory:")
    return s, s.add_course("Grid Puzzles", term="Fall 2026"), tmp_path


def _week(s, cid):
    mat(s, cid, "recording", "Week 5 captions", [
        ("", "Let's set up a puzzle. Take the grid with rows four, one and two, three. " + FILLER),
        ("", "Its shifts come from the corner total. Corners add to seven and multiply to ten. " + FILLER),
        ("", "So the shifts are five and two. Check: seven again and ten again. " + FILLER),
        ("", "Now a side comment about the kiln schedule. " + FILLER),
        ("", "Now we want the pair for shift five. Subtract five from the corners: rows negative one, one and two, negative two. " + FILLER),
        ("", "The first row says negative a plus b is zero, so a equals b. Take a equals one and b equals one. The pair is one, one. " + FILLER),
        ("", "For shift two the pair is one, negative two. " + FILLER),
    ])
    mat(s, cid, "lecture_notes", "Cycle notes", [
        ("Cycle pairs", "A cycle grid has a pair for shift one: the pair is the steady state. Every pair of a cycle grid shows the long run. "
                        "Find the pair by solving. The pair, the pair, the pair. " + FILLER),
        ("Cycle pairs 2", "To find the pair of a cycle grid, find the shift and then the pair. Pairs of cycle grids sum to one. " + FILLER)])


@pytest.mark.parametrize("q", ["How do I find the pairs of [[4,1],[2,3]]?",
                               "How do I find the pair of the grid with rows 4 1 and 2 3?"])
def test_the_caption_that_sets_up_the_matrix_and_its_walkthrough_beat_a_wordier_chunk(env, q):
    s, cid, _ = env
    _week(s, cid)
    top = retrieval.search(s, cid, q, limit=6)
    assert top[0]["material_title"] == "Week 5 captions" and "rows four, one and two, three" in top[0]["text"]
    assert any("The pair is one, one" in x["text"] for x in top[:3]), [x["text"][:30] for x in top]


def test_a_matrix_typed_in_words_is_not_split_into_clauses_but_a_document_list_still_is():
    q = "How do I find the pair of the grid with rows 4 1 and 2 3?"
    assert retrieval.split_clauses(q) == [q]
    q2 = "rows four, one and two, three: what is the pair"
    assert len(retrieval.split_clauses(q2)) == 1
    assert len(retrieval.split_clauses("Federalist No. 10 and 51 and Chapters 7 and 8")) >= 2


def test_a_question_without_a_matrix_is_not_pulled_to_number_chunks(env):
    s, cid, _ = env
    _week(s, cid)
    top = retrieval.search(s, cid, "what is a cycle grid", limit=3)
    assert top[0]["material_title"] == "Cycle notes"


# ---- import report on a plain post
def _two(n=2, pad=""):
    return [("files", (f"kiln-notes-{i}{pad}.md", (b"# Kiln\n\nCones mark the end of a firing, number %d. " % i) * 4, "text/markdown")) for i in range(n)]


def test_the_import_report_shows_after_a_plain_post_and_redirect(env):
    s, cid, _ = env
    c = _client(s)
    r = c.post(f"/courses/{cid}/materials/upload", data={}, files=_two(), follow_redirects=True)
    assert r.status_code == 200 and "What happened to each file" in r.text
    assert "kiln-notes-0.md: " in r.text and "kiln-notes-1.md: " in r.text
    again = c.get(f"/courses/{cid}")
    assert "What happened to each file" not in again.text  # shown once


def test_a_long_import_report_fits_the_cookie_and_says_how_many_more(env):
    s, cid, _ = env
    c = _client(s)
    r = c.post(f"/courses/{cid}/materials/upload", data={}, files=_two(30, '-' + 'glaze' * 12), follow_redirects=False)
    assert r.status_code == 303 and len(r.headers["set-cookie"]) < 4000
    page = c.get(f"/courses/{cid}")
    assert "What happened to each file" in page.text and "listed on the Materials page" in page.text
    assert page.text.count("kiln-notes-") >= 5


def test_the_htmx_import_report_still_works(env):
    s, cid, _ = env
    r = _client(s).post(f"/courses/{cid}/materials/upload", data={}, headers={"HX-Request": "true"}, files=_two())
    assert "What happened to each file" in r.text


# ---- Add exam
def test_add_exam_link_lands_on_the_title_field(env):
    s, cid, _ = env
    c = _client(s)
    r = c.get(f"/courses/{cid}/calendar?add=exam")
    assert 'data-focus-title' in r.text and 'autofocus' in r.text and 'id="add-event" tabindex="-1"' in r.text
    assert "Add an event, type Exam" in r.text
    plain = c.get(f"/courses/{cid}/calendar")
    assert "data-focus-title" not in plain.text and "Focus is on the Title field" not in plain.text


def test_app_js_moves_focus_to_the_title_and_names_code_blocks():
    from pathlib import Path
    js = (Path(retrieval.__file__).parent / "static" / "app.js").read_text()
    assert "[data-focus-title]" in js
    assert "block \" + (i + 1) + \" of \"" in js and "Copy this code\"" not in js.replace('"Copy this code".', "")


def test_dashboard_offers_another_exam_only_when_the_next_one_is_far_off(env):
    from datetime import date, timedelta
    s, cid, _ = env
    c = _client(s)
    s.add_event(cid, "exam", "Final", (date.today() + timedelta(days=40)).isoformat())
    assert "Add another exam" in c.get("/").text
    s2 = Store(":memory:")
    cid2 = s2.add_course("Near")
    s2.add_event(cid2, "exam", "Exam 3", (date.today() + timedelta(days=3)).isoformat())
    assert "Add another exam" not in _client(s2).get("/").text


# ---- flashcard source line
def test_a_card_source_does_not_repeat_the_title():
    f = study.card_source_label
    assert f("Glaze basics", "Glaze basics") == "Glaze basics"
    assert f("Glaze basics, Glaze basics") == "Glaze basics"
    assert f("Glaze basics", "p. 4") == "Glaze basics, p. 4"
    assert f("Chapter 3, Kilns") == "Chapter 3, Kilns"  # a real place is kept


def test_a_card_page_shows_the_source_once(env):
    s, cid, _ = env
    m = mat(s, cid, "notes", "Glaze basics", [("Glaze basics", "Glaze is a glassy coat. " + FILLER)])
    ch = s.chunks_of(m)[0]["id"]
    s.add_flashcard(cid, "What is glaze?", "A glassy coat", chunk_id=ch, source_label="Glaze basics, Glaze basics")
    r = _client(s).get(f"/flashcards/{cid}")
    assert "Glaze basics, Glaze basics" not in r.text and "From:" in r.text
