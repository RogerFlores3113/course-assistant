"""Cycle 6 source-selection regressions: spelled-out numbers in captions, primer sources and the Used materials line,
dates in note titles, flashcard sources, the tail of search results, the per-file import report.
All text here is original (made-up pottery, kiln and grid-puzzle lessons)."""
import pytest

from course_assistant import importers, retrieval, study
from course_assistant.store import Store
from course_assistant.web_helpers import guide_view


class Prov:
    name, model = "fake", "fake-1"

    def __init__(self, reply="Fine. [1]"):
        self.reply, self.prompts = reply, []

    def complete(self, system, messages, *, max_tokens=4000):
        self.prompts.append(messages[-1]["content"])
        return self.reply


def mat(s, cid, kind, title, chunks, **kw):
    mid = s.add_material(cid, kind, title, **kw)
    s.replace_chunks(mid, [{"text": t, "loc_kind": "section", "loc_start": str(i), "loc_label": h or title, "heading": h}
                           for i, (h, t) in enumerate(chunks)])
    return mid


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("COURSE_ASSISTANT_DATA_DIR", str(tmp_path / "data"))
    s = Store(":memory:")
    return s, s.add_course("Grid Puzzles", term="Fall 2026"), tmp_path


FILLER = "The class kept going with the usual chatter about clay, glaze, shelves and the studio schedule. " * 3


def _captions(s, cid):
    return mat(s, cid, "recording", "Week 5 captions", [
        ("", "Let's set up a puzzle. Take the grid with rows four, one and two, three. " + FILLER),
        ("", "Now the scores. Add the corners and you get seven, multiply them and you get ten. " + FILLER),
        ("", "Now the second score. Take the shift equals two. Solve two a plus b equals zero. Take a equals one, then b is "
             "negative two. So the pair is one, negative two. Check it by running the grid. " + FILLER),
        ("", "Next week we will glaze the bowls and talk about the kiln. " + FILLER),
    ])


# ---- 1. spelled-out numbers in captions
@pytest.mark.parametrize("q", [
    "pair for shift 2 of the grid with rows 4 1 and 2 3",
    "In the lecture example, what is the pair for shift 2 of [[4,1],[2,3]]",
    "pair for shift two of the grid with rows four, one and two, three",
    "so the pair is one, negative two? how did they get that",
])
def test_caption_with_spoken_numbers_is_found_from_digits_or_words(env, q):
    s, cid, _ = env
    _captions(s, cid)
    mat(s, cid, "lecture_notes", "Grid notes", [("Grids", "A grid is a table of numbers. " + FILLER)] * 1)
    out = retrieval.search(s, cid, q, limit=6, exclude_graded=True)
    assert any("a equals one, then b is negative two" in x["text"] for x in out), [x["text"][:40] for x in out]


def test_a_lead_in_clause_does_not_become_its_own_search():
    assert retrieval.split_clauses("In the lecture example, what is the pair for shift 2 of [[4,1],[2,3]]") == [
        "In the lecture example, what is the pair for shift 2 of [[4,1],[2,3]]"]


def test_spoken_number_runs_count_but_a_lone_one_does_not():
    assert retrieval._spoken_numbers("rows four, one and two, three") == ["4", "1", "2", "3"]
    assert retrieval._spoken_numbers("the pair one, negative two") == ["1", "2"]
    assert retrieval._spoken_numbers("one of the best ideas") == []


# ---- 2. primers
def _lecture(s, cid, title, topics, date="2026-11-19"):
    return s.add_event(cid, "lecture", title, date, topics=topics)


def test_primer_with_nothing_linked_drops_passages_that_only_share_a_common_word(env):
    s, cid, _ = env
    mat(s, cid, "syllabus", "Syllabus", [("Lecture schedule", "Nov 19 | Lecture 21: Kiln firing curves and cone temperatures | ch 6\n" + FILLER)])
    mat(s, cid, "lecture_notes", "Glaze notes", [
        ("Glaze basics", "Glaze is a glassy coat. In passing, the kiln matters later in the term. " + FILLER),
        ("Wheel throwing", "Centering the clay on the wheel takes patience and wet hands. " + FILLER),
        ("Kiln firing curves", "A firing curve is the schedule of kiln temperature over hours; cone temperatures mark the "
                               "end of a firing. " + FILLER)])
    for i in range(12):  # a course is bigger than three chunks: a word in every passage says little, a rare one a lot
        mat(s, cid, "lecture_notes", f"Other notes {i}", [("Misc", f"Notes number {i} about shelves, sinks and studio rules. " + FILLER)])
    eid = _lecture(s, cid, "Lecture 21: Kiln firing curves and cone temperatures", "ch 6")
    ev = s.event(eid)
    sources, used, note = study._lecture_sources(s, cid, ev, primer=True)
    heads = [x["heading"] for x in sources]
    assert "Kiln firing curves" in heads and "Wheel throwing" not in heads and "Glaze basics" not in heads
    assert not used


def test_primer_with_own_notes_adds_only_a_few_related_passages(env):
    s, cid, _ = env
    own = mat(s, cid, "lecture_notes", "Lecture 8 notes: Slip casting", [(f"Part {i}", f"Slip casting part {i} uses plaster molds. " + FILLER) for i in range(6)])
    for i in range(6):
        mat(s, cid, "reading", f"Reading {i}", [("Slip casting", f"Slip casting reading {i}: plaster molds absorb water from the slip. " + FILLER)])
    eid = _lecture(s, cid, "Lecture 8: Slip casting", "plaster molds")
    s.update_material(own, event_id=eid)
    sources, used, _ = study._lecture_sources(s, cid, s.event(eid), primer=True)
    assert used and len([x for x in sources if x["material_id"] != own]) <= 3


def _client(s):
    from fastapi.testclient import TestClient
    from course_assistant import disclaimer, web
    s.set_setting(disclaimer.ACK_SETTING, "2026-01-01T00:00:00+00:00")
    c = TestClient(web.create_app(s, key="k" * 40, provider_factory=lambda: None), base_url="http://127.0.0.1:8766")
    c.cookies.set(web.COOKIE, "k" * 40)
    return c


def test_primer_page_lists_used_materials(env):
    s, cid, _ = env
    guide = {"id": 1, "kind": "primer", "course_id": cid, "title": "Primer: Slip casting", "body_md": "Plaster molds absorb water.",
             "sources": [], "data": {"used_materials": [{"title": "Lecture 8 notes", "why": "linked to this lecture"}]},
             "question": "Lecture 8", "created_at": "2026-11-13T00:00:00"}
    c = _client(s)
    gid = s.add_guide(cid, "primer", "Primer: Slip casting", "Plaster molds absorb water.", sources=[],
                      data=guide["data"], question="Lecture 8")
    r = c.get(f"/guides/{gid}")
    assert r.status_code == 200 and "Used materials:" in r.text and "Lecture 8 notes (linked to this lecture)" in r.text
    gid2 = s.add_guide(cid, "primer", "Primer: Glaze", "Glaze is a glassy coat.", sources=[], data={"used_materials": []}, question="Lecture 9")
    r = c.get(f"/guides/{gid2}")
    assert "Used materials:" in r.text and "none linked" in r.text


# ---- 3. dates in note titles
def test_a_note_titled_with_the_lectures_date_is_suggested_for_it(env):
    s, cid, _ = env
    _lecture(s, cid, "Lecture 7: Trimming bowls", "foot rings", date="2026-10-06")
    _lecture(s, cid, "Lecture 8: Handles", "pulling handles", date="2026-10-08")
    a = mat(s, cid, "notes", "Oct 6 lecture", [("", "Trimmed the bowls today. " + FILLER)])
    b = mat(s, cid, "notes", "My notes 10/8", [("", "Handles are pulled. " + FILLER)])
    c = mat(s, cid, "notes", "Notes from October 9th", [("", "No lecture that day. " + FILLER)])
    d = mat(s, cid, "notes", "Week 3 notes", [("", "Nothing dated. " + FILLER)])
    got = {x["material"]["id"]: x["event"]["title"] for x in study.link_suggestions(s, cid)}
    assert got.get(a) == "Lecture 7: Trimming bowls" and got.get(b) == "Lecture 8: Handles"
    assert c not in got and d not in got


def test_a_dated_title_with_the_wrong_year_is_not_a_match(env):
    s, cid, _ = env
    _lecture(s, cid, "Lecture 7: Trimming bowls", "foot rings", date="2026-10-06")
    a = mat(s, cid, "notes", "2025-10-06 notes", [("", "Old term. " + FILLER)])
    assert a not in {x["material"]["id"] for x in study.link_suggestions(s, cid)}


def test_lecture_materials_matches_a_date_in_the_name(env):
    s, cid, _ = env
    eid = _lecture(s, cid, "Lecture 7: Trimming bowls", "foot rings", date="2026-10-06")
    a = mat(s, cid, "notes", "Oct 6 lecture", [("", "Trimmed. " + FILLER)])
    assert [(m["id"], why) for m, why in study.lecture_materials(s, cid, s.event(eid))] == [(a, "matched by date in its name")]


# ---- 4. flashcard sources
def test_flashcard_sources_skip_exercises_glossary_and_summaries(env):
    s, cid, _ = env
    m = mat(s, cid, "textbook", "Chapter 2: Clay bodies", [
        ("Introduction", "In this chapter you will learn about clay bodies. " + FILLER),
        ("Plasticity", "Plasticity is how far wet clay bends before it cracks. " + FILLER),
        ("Shrinkage", "Clay shrinks as it dries and again when it is fired. " + FILLER),
        ("Key Equations", "Problem. How much does a 20 cm pot shrink at 12 percent?\n\nSolution. 2.4 cm\n\nProblem. And at 8 percent?\n\nSolution. 1.6 cm"),
        ("Glossary", "plasticity: how far wet clay bends. " + FILLER),
        ("Section Summary", "Clay bodies differ in plasticity and shrinkage. " + FILLER),
        ("Chapter 2 Exercises", "1. Define plasticity. 2. Name two clay bodies. " + FILLER),
    ])
    p = Prov('[{"front": "What is plasticity?", "back": "How far wet clay bends before it cracks", "cite": 1}]')
    study.make_flashcards(s, cid, p, material_id=m)
    sent = p.prompts[-1]
    assert "Plasticity is how far" in sent and "Clay shrinks as it dries" in sent
    for junk in ("Glossary", "Section Summary", "Chapter 2 Exercises", "Solution. 2.4 cm", "learn about clay bodies"):
        assert junk not in sent, junk


def test_flashcards_by_topic_skip_the_syllabus_and_past_schedules(env):
    s, cid, _ = env
    mat(s, cid, "textbook", "Clay book", [("Plasticity", "Plasticity is how far wet clay bends before it cracks. " + FILLER),
                                         ("Wedging", "Wedging removes air from clay and makes plasticity even. " + FILLER),
                                         ("Slaking", "Slaking soaks dry scrap clay so its plasticity returns. " + FILLER)])
    mat(s, cid, "syllabus", "Syllabus", [("Schedule", "Week 4: plasticity of clay bodies. " + FILLER)])
    p = Prov('[{"front": "What is plasticity?", "back": "How far wet clay bends", "cite": 1}]')
    study.make_flashcards(s, cid, p, topic="plasticity of clay")
    assert "Week 4: plasticity" not in p.prompts[-1] and "Plasticity is how far" in p.prompts[-1]


def test_a_thin_material_still_makes_cards_when_everything_is_scaffolding(env):
    s, cid, _ = env
    m = mat(s, cid, "textbook", "Glossary only", [("Glossary", "plasticity: how far wet clay bends. " + FILLER)])
    p = Prov('[{"front": "plasticity", "back": "how far wet clay bends", "cite": 1}]')
    study.make_flashcards(s, cid, p, material_id=m)
    assert "plasticity: how far wet clay bends" in p.prompts[-1]


# ---- 5. the tail of a search
def test_exercise_lists_under_another_heading_and_summaries_leave_the_tail(env):
    s, cid, _ = env
    mat(s, cid, "textbook", "Clay book", [
        ("Plasticity", "Plasticity describes how wet clay bends. Plasticity depends on particle size. " + FILLER),
        ("Plasticity and water", "Plasticity rises with water content up to a point. " + FILLER),
        ("Plasticity tests", "A coil test checks plasticity by bending a coil around a finger. " + FILLER),
        ("Aging clay", "Aged clay has better plasticity because bacteria make it slick. " + FILLER),
        ("Key Equations", "Problem. What is the plasticity index of a clay with limit 30 and 18?\n\nSolution. 12\n\nProblem. Again with 25 and 15?\n\nSolution. 10"),
        ("Section Summary", "Plasticity is a key property of clay. " + FILLER),
    ])
    out = retrieval.search(s, cid, "How is plasticity measured?", limit=8, exclude_graded=True)
    heads = [x["heading"] for x in out]
    assert heads[:1] and "Key Equations" not in heads and "Section Summary" not in heads


def test_cited_recall_is_kept_when_the_question_asks_for_exercises_or_a_summary(env):
    s, cid, _ = env
    mat(s, cid, "textbook", "Clay book", [
        ("Plasticity", "Plasticity describes how wet clay bends. " + FILLER), ("Aging", "Aged clay has better plasticity. " + FILLER),
        ("Tests", "A coil test checks plasticity. " + FILLER), ("Water", "Water raises plasticity. " + FILLER),
        ("Section Summary", "Plasticity is a key property of clay. " + FILLER),
        ("Key Equations", "Problem. What is the plasticity index of 30 and 18?\n\nSolution. 12\n\nProblem. And 25 and 15?\n\nSolution. 10")])
    heads = [x["heading"] for x in retrieval.search(s, cid, "Give me practice problems on plasticity", limit=8, exclude_graded=True)]
    assert "Key Equations" in heads
    heads = [x["heading"] for x in retrieval.search(s, cid, "Summarize plasticity", limit=8, exclude_graded=True)]
    assert "Section Summary" in heads


def test_search_tail_drops_passages_far_behind_the_best_one(env):
    s, cid, _ = env
    mat(s, cid, "syllabus", "Syllabus", [("Problem set due dates", "Problem set due dates: Set 1 is due Sep 20, set 2 is due Oct 4, set 3 is due Oct 18. Late problem sets lose points. " + FILLER),
                                        ("Problem set policy", "Problem sets are posted on Thursdays. " + FILLER)])
    mat(s, cid, "lecture_notes", "Odd", [("Other", "The room has a poster about the set of shelves. " + FILLER)])
    out = retrieval.search(s, cid, "When is the problem set due?", limit=8, exclude_graded=True)
    assert out[0]["heading"].startswith("Problem set") and all(x["score"] >= 0.25 * out[0]["score"] for x in out[3:])


def test_topic_sources_stop_at_the_items_own_topic(env):
    s, cid, _ = env
    mat(s, cid, "lecture_notes", "Notes", [
        ("Glaze thickness", "Glaze thickness is measured by dipping time. Glaze thickness changes the finish of a glaze. " + FILLER),
        ("Glaze thickness tests", "To test glaze thickness scratch the surface after drying. " + FILLER),
        ("Dipping glaze", "Dip the pot for three seconds so glaze thickness stays even. " + FILLER),
        ("Kiln shelves", "Shelves hold pots in the kiln; a glaze that runs fuses the pot to the shelf. " + FILLER),
        ("Studio rules", "Clean the sink after glaze work. " + FILLER)])
    gid = s.add_graded_item(cid, "Problem 1", "Problem 1\nHow does glaze thickness change with dipping time?", None, kind="problem", label="Problem 1")
    out = retrieval.topic_sources(s, s.graded_item(gid), k=6)
    heads = [x["heading"] for x in out]
    assert heads[0].startswith("Glaze thickness") and "Studio rules" not in heads


# ---- 6. the import report
def test_import_report_says_kind_chunks_and_what_was_left_out(env, tmp_path):
    s, cid, tmp = env
    p = tmp / "week3-notes.md"
    p.write_text("# Week 3 notes\n\nCentering clay on the wheel takes practice and wet hands. " * 3)
    rep = importers.import_file_report(s, cid, p)
    assert rep["name"] == "week3-notes.md" and rep["kind_how"] == "guessed" and rep["chunks"] >= 1 and rep["skipped"] == []
    assert "guessed" in rep["line"] and "passage" in rep["line"]
    again = importers.import_file_report(s, cid, p)
    assert again["already_saved"] and again["material_id"] == rep["material_id"]
    assert any("Already saved" in x for x in again["skipped"]) and "Left out" in again["line"]
    q = tmp / "other.md"
    q.write_text("# Reading\n\nGlaze is a glassy coat on a pot. " * 3)
    chosen = importers.import_file_report(s, cid, q, kind="reading")
    assert chosen["kind"] == "reading" and chosen["kind_how"] == "chosen" and "(chosen)" in chosen["line"]


def test_import_report_notes_blank_slides_and_pages(env, tmp_path):
    from course_assistant.importers import extract
    assert extract._skip_note("page", [2, 5]) == ["2 pages with no readable text (pages 2, 5)"]
    assert extract._skip_note("slide", [3]) == ["1 slide with no readable text (slide 3)"]
    assert extract._skip_note("page", []) == []


def test_import_file_still_returns_the_material_id(env, tmp_path):
    s, cid, tmp = env
    p = tmp / "a.md"
    p.write_text("# A\n\nSome text about the kiln and cones. " * 3)
    mid = importers.import_file(s, cid, p)
    assert isinstance(mid, int) and s.material(mid)["id"] == mid


def test_upload_page_shows_one_line_per_file(env):
    s, cid, _ = env
    c = _client(s)
    r = c.post(f"/courses/{cid}/materials/upload", data={}, headers={"HX-Request": "true"},
               files=[("files", ("kiln-notes.md", b"# Kiln\n\nCones mark the end of a firing. " * 4, "text/markdown")),
                      ("files", ("empty-ish.md", b"# Two\n\nGlaze is a glassy coat on a pot. " * 4, "text/markdown"))])
    assert r.status_code == 200 and "What happened to each file" in r.text
    assert "kiln-notes.md: " in r.text and "(guessed)" in r.text and "passage" in r.text
