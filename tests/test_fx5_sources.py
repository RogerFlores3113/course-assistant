"""Cycle 5 source-selection regressions: topic_sources, comparison questions, link suggestions, exam guide notes,
delete cascade, primer wording. All text here is original (a made-up woodworking and brewing course)."""
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
    return s, s.add_course("Joinery", term="Fall 2026"), tmp_path


BODY = "Joints hold boards together when glue, grain direction and clamping pressure all agree. " * 4


def _joinery(s, cid):
    lec = mat(s, cid, "lecture_notes", "Lecture notes: joints", [
        ("Dovetail joints", "A dovetail joint locks two boards with interlocking wedge shaped pins and tails. " + BODY),
        ("Mortise and tenon", "A mortise is a pocket and the tenon is the tongue that fits it. " + BODY),
        ("Big boards on the shop floor", "```python\nboard = load('oak')\nboard.cut(3)\nprint(board)\n```\n"),
        ("Key Concepts and Summary", "Summary of joints: dovetails and tenons. " + BODY),
    ])
    mat(s, cid, "syllabus", "JOIN 101 syllabus", [("Schedule", "Week 3: dovetail joints. " + BODY)])
    mat(s, cid, "handout", "Bench handout", [("Dovetail joints", "Handout dovetail joints. " + BODY)])
    return lec


def _item(s, cid, body, **kw):
    gid = s.add_graded_item(cid, "Problem 1", body, kw.get("material_id"), kind="problem", label="Problem 1")
    return s.graded_item(gid)


def test_topic_sources_follow_the_items_own_topic_and_skip_code_summaries_and_syllabus(env):
    s, cid, _ = env
    _joinery(s, cid)
    it = _item(s, cid, "Problem 1 (10 points)\nWhich dovetail joint locks the two boards, and why do the wedge shaped pins matter?")
    out = retrieval.topic_sources(s, it, k=6)
    assert out and out[0]["heading"] == "Dovetail joints" and out[0]["material_kind"] == "lecture_notes"
    for r in out:
        assert r["material_kind"] not in ("syllabus", "handout", "past_offering", "assignment")
        assert r["heading"] not in ("Key Concepts and Summary", "Big boards on the shop floor")
        assert not r["text"].startswith("```")


def test_topic_sources_use_synonyms_and_notation_hints(env):
    s, cid, _ = env
    mat(s, cid, "textbook", "Brewing", [
        ("Row swaps", "When a pivot is zero the brewer swaps two rows of the recipe table before continuing. " + BODY),
        ("Molarity", "Molarity is the moles of solute per liter of solution, written as mol per L. " + BODY),
        ("Empirical formulas", "Divide each mole count by the smallest to get the formula. " + BODY),
        ("Key Concepts and Summary", "Empirical formulas and molarity in a nutshell. " + BODY),
    ])
    swap = retrieval.topic_sources(s, _item(s, cid, "What value forces a row exchange at the second pivot?"))
    assert swap and swap[0]["heading"] == "Row swaps"
    molar = retrieval.topic_sources(s, _item(s, cid, "How many grams of salt make 250 mL of a 0.35 M solution?"))
    assert molar and molar[0]["heading"] == "Molarity"
    assert all(r["heading"] != "Key Concepts and Summary" for r in molar)


def test_topic_sources_empty_when_nothing_on_topic_and_never_graded_material(env):
    s, cid, _ = env
    mat(s, cid, "textbook", "Tools", [("Clamps", "Clamps hold boards while glue dries. " + BODY)], graded=1)
    assert retrieval.topic_sources(s, _item(s, cid, "Explain how a clamp holds boards while glue dries.")) == []
    assert retrieval.topic_sources(s, {"course_id": cid, "body": ""}) == []


# ---- comparison questions
def _terms(s, cid):
    mat(s, cid, "syllabus", "JOIN 101 syllabus Fall 2026", [
        ("Exams", "Midterm on October 8 and final on December 11. Both cover joints."),
        ("Policies", "Late essays lose points each day and need footnotes in the house style."),
        ("Schedule", "Week 1 sanding. Week 2 gluing. Newton style iteration is not covered. " + BODY)])
    mat(s, cid, "past_offering", "JOIN 101 Fall 2025 summary", [
        ("Exam 1 (Oct 2)", "Exam 1 covered sanding and gluing on October 2. " + BODY),
        ("Exam 2 (Nov 6)", "Exam 2 covered dovetails on November 6. " + BODY),
        ("Exam 3 (Dec 4)", "Exam 3 covered tenons on December 4. " + BODY),
        ("Lecture 12 (Oct 5)", "Newton style root finding for plane angles was shown. " + BODY),
        ("Lecture 25 (Nov 9)", "Newton style iteration again for chisels. " + BODY)])
    mat(s, cid, "assignment", "Essay 1: Joints", [("Essay 1: Joints", "Write about joints in five pages. " + BODY)])


def test_comparison_question_returns_both_sides(env):
    s, cid, _ = env
    _terms(s, cid)
    r = retrieval.search(s, cid, "What were the exam dates last year, and why do they differ from this term?",
                         limit=8, exclude_graded=True)
    kinds = {(x["material_kind"], x["heading"][:6]) for x in r}
    assert any(k == "syllabus" and h == "Exams" for k, h in kinds)
    assert any(k == "past_offering" and h.startswith("Exam") for k, h in kinds)
    assert not any("Newton" in x["text"] and x["heading"].startswith("Lecture") for x in r[:4])


def test_named_past_term_comparison_reaches_every_exam_and_this_terms_policies(env):
    s, cid, _ = env
    _terms(s, cid)
    r = retrieval.search(s, cid, "What were the exam dates in the Fall 2025 offering, and how do they compare with the "
                         "syllabus exams for Fall 2026?", limit=8, exclude_graded=True)
    heads = [x["heading"] for x in r[:8]]
    assert all(any(h.startswith(f"Exam {n}") for h in heads) for n in (1, 2, 3))
    assert any(x["material_kind"] == "syllabus" and x["heading"] == "Exams" for x in r)
    r = retrieval.search(s, cid, "How was the essay different in Fall 2025 from this term?", limit=8, exclude_graded=True)
    assert any(x["heading"] == "Policies" for x in r)
    assert any(x["material_kind"] == "assignment" for x in r)


def test_plain_question_is_not_treated_as_comparison(env):
    s, cid, _ = env
    assert not retrieval.is_comparison("What is a dovetail joint?")
    assert not retrieval.is_comparison("Fall 2025 exam dates")
    assert retrieval.is_comparison("exam dates last year vs this term")


# ---- link suggestions and README
def _lectures(s, cid):
    s.add_event(cid, "lecture", "Lecture 3: Dovetail joints and glue", "2026-09-10", topics="Dovetails, glue lines")
    s.add_event(cid, "lecture", "Lecture 21: Finishing oils and varnish", "2026-11-05", topics="Oil finishes, varnish coats")
    s.add_event(cid, "lecture", "Lecture 4: Mortise and tenon fit", "2026-09-12", topics="Mortise, tenon, fit")


def test_link_suggestions_need_a_number_or_two_shared_words_and_skip_the_readme(env):
    s, cid, _ = env
    _lectures(s, cid)
    mat(s, cid, "notes", "Dovetail joints and glue lines", [("a", BODY)])       # two shared words: suggested
    mat(s, cid, "notes", "Notes for lecture 4", [("a", BODY)])                   # a lecture number: suggested
    mat(s, cid, "notes", "Glue", [("a", BODY)])                                  # one shared word: not suggested
    mat(s, cid, "notes", "Joinery sample course", [("a", "Files: dovetail, glue, varnish. " + BODY)],
        original_name="README.md")                                                # the README: never
    got = {x["material"]["title"]: x["event"]["title"] for x in study.link_suggestions(s, cid)}
    assert got["Dovetail joints and glue lines"].startswith("Lecture 3")
    assert got["Notes for lecture 4"].startswith("Lecture 4")
    assert "Glue" not in got and "Joinery sample course" not in got


def test_matrix_style_one_word_titles_are_not_matched_to_an_unrelated_lecture(env):
    s, cid, _ = env
    _lectures(s, cid)
    mat(s, cid, "notes", "Finishing perspectives on joints", [("a", BODY)])      # one word with Lecture 21
    assert not any("Lecture 21" in x["why"] for x in study.link_suggestions(s, cid))


def test_readme_never_ranks_as_a_search_source(env):
    s, cid, _ = env
    mat(s, cid, "notes", "Joinery sample course", [("a", "Dovetail dovetail dovetail files listed here. " + BODY)],
        original_name="README.md")
    mat(s, cid, "notes", "Dovetail notes", [("Dovetails", "Dovetail pins and tails. " + BODY)])
    r = retrieval.search(s, cid, "dovetail", limit=8)
    assert r and all(x["material_title"] != "Joinery sample course" for x in r)


# ---- exam guide, graded handout
def test_exam_guide_says_a_ticked_graded_handout_adds_nothing(env):
    s, cid, _ = env
    s.add_event(cid, "lecture", "Lecture 1: Dovetail joints", "2026-09-01", topics="Dovetails")
    ex = s.add_event(cid, "exam", "Midterm", "2026-10-08")
    notes = mat(s, cid, "lecture_notes", "Dovetail notes", [("Dovetail joints", "Dovetail pins and tails. " + BODY)])
    lab = mat(s, cid, "handout", "Lab 3 handout", [("Steps", "Cut the pins first. " + BODY)], graded=1)
    p = Prov()
    gid = study.exam_guide(s, cid, p, event_id=ex, material_ids=[notes, lab])
    g = s.guide(gid)
    assert "Lab 3 handout is graded work, so it adds nothing to this guide and was not used." in str(g)
    assert "Materials used: Dovetail notes.\n" in p.prompts[-1]


# ---- delete cascade
def test_deleting_a_material_deletes_its_graded_items_not_hand_added_ones(env):
    s, cid, tmp = env
    p = tmp / "ps.md"
    p.write_text("# Problem Set 1\n\n## Problem 1 (10 points)\nHow many dovetail pins fit a board 12 cm wide?\n\n"
                 "## Problem 2 (10 points)\nWhy does grain direction matter in a glued joint?\n")
    mid = importers.import_file(s, cid, p)
    from course_assistant import integrity
    ids = integrity.mark_material_graded(s, mid)
    assert len(ids) >= 2
    s.add_graded_item(cid, "Extra by hand", "Explain glue lines.")
    importers.delete_material(s, mid)
    left = [i["title"] for i in s.graded_items(cid)]
    assert left == ["Extra by hand"]
    mid2 = importers.import_file(s, cid, p)
    assert len(integrity.mark_material_graded(s, mid2)) == len(ids)
    assert len(s.graded_items(cid)) == len(ids) + 1


# ---- primer wording
def test_primer_view_never_says_your_question(env):
    s, cid, _ = env
    guide = {"kind": "primer", "course_id": cid, "body_md": "Dovetails lock boards [1]. The sources do not cover finishing.",
             "sources": [{"n": 1, "chunk_id": 1, "material_id": 1, "material_title": "Notes", "loc_label": "a",
                          "heading": "h", "text": "Dovetails lock boards.", "href": "/read/1"}],
             "data": {}, "question": "Lecture 21"}
    v = guide_view(guide)
    assert "question" not in v["partly_covered_note"]
    assert "primer" in v["partly_covered_note"]
