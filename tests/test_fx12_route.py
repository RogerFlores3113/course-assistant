"""Cycle 12 routing and context (fx12-route). Fixtures are original text. Each leak test sits beside an honest look-alike."""
import json

import pytest

from course_assistant import grounding, importers, integrity, study, web_helpers
from course_assistant.providers import FakeProvider
from course_assistant.store import Store

EIG = "Problem 5 (10 points). Find the eigenvalues of the matrix with rows (2, 3) and (3, -6)."
EIG_KEY = ["lambda1 = 3", "lambda2 = -7"]
WORDP = "Problem 7 (10 points). Explain in two sentences why a determinant of zero means the rows are dependent."
NOTES = """# Lecture 18: Eigenvalues

## Main idea

Eigenvalues are the numbers that tell you how a matrix stretches a direction. The matrix with rows 4 1 and 2 3 has eigenvalues 5 and 2.

## Method

To find the eigenvalues of a matrix, solve det(A - lambda I) = 0 and read off the roots. An eigenvector is only stretched.
"""


def eworld(tmp_path):
    s = Store(":memory:")
    c = s.add_course("Course")
    ids = {"eig": s.add_graded_item(c, "Set 2: Problem 5", EIG, None, kind="problem", label="Problem 5",
                                    answer_key=json.dumps(EIG_KEY))}
    f = tmp_path / "lecture-18.md"
    f.write_text(NOTES)
    importers.import_file(s, c, f)
    return s, c, ids


LECTURE_SUMMARY = ("## Main points\n\n- Eigenvalues are the topic the course builds on [1].\n"
                   "- Find the eigenvalues of a matrix by solving det(A - lambda I) = 0 [1].\n"
                   "- The matrix with rows 4 1 and 2 3 has eigenvalues 5 and 2 [1].\n")


COPY_TEXTS = ["Find the eigenvalues of the matrix with rows (2, 3) and (3, -6).",
              "1. Find the eigenvalues of the matrix with rows (2, 3) and (3, -6) and report both."]
HONEST_TEXTS = [
    "Eigenvalues are the topic the course builds on, and to find the eigenvalues of a matrix you solve a determinant.",
    "Find the eigenvalues of the matrix with rows (6, 1) and (2, 5).",
    "The matrix with rows 4 1 and 2 3 has eigenvalues 5 and 2, found from the characteristic equation.",
    "To find the eigenvalues of a matrix with rows of numbers, subtract lambda from the diagonal and take the determinant.",
]


@pytest.mark.parametrize("t", COPY_TEXTS)
def test_d1_a_restated_item_is_still_a_copy(t):
    it = {"kind": "problem", "body": EIG, "id": 1}
    assert integrity.copies_item(t, it, rows=True)


@pytest.mark.parametrize("t", HONEST_TEXTS)
def test_d1_topic_vocabulary_is_never_a_copy(t):
    it = {"kind": "problem", "body": EIG, "id": 1}
    assert not integrity.copies_item(t, it, rows=True)
    assert not integrity.copies_item(t, it)


def test_d1_explain_on_a_shared_topic_is_kept(tmp_path):
    s, c, ids = eworld(tmp_path)
    gid = study.explain(s, c, FakeProvider(LECTURE_SUMMARY), "eigenvalues")
    g = s.guide(gid)
    assert "Main points" in g["body_md"] or "Eigenvalues are the topic" in g["body_md"]
    assert not g["data"].get("scrubbed")


def test_d1_a_fully_held_back_answer_says_so_once_and_truthfully(tmp_path):
    s, c, ids = eworld(tmp_path)
    copy = "Find the eigenvalues of the matrix with rows (2, 3) and (3, -6)."
    gid = study.explain(s, c, FakeProvider(copy), "eigenvalues")
    g = s.guide(gid)
    assert g["data"].get("held_back") and g["data"].get("scrubbed")
    view = web_helpers.guide_view(g, store=s)
    assert not view["not_covered"]
    text = " ".join(b["html"] for b in view["blocks"])
    assert "did not return an answer" not in text.lower()
    assert text.lower().count("held back") == 1
    assert "Problem 5" in text


# ------------------------------------------------------------------ decision 8 (context half)
from fx9_world import world as w9


def _sticky(s, ids, label="Problem 1"):
    return dict(s.graded_item(ids[label]), kind="problem", reason="sticky", source="item", score=1.0)


OWN = "Solve A x = b for A = [[1,1],[0,2]] and b = (3, 4) using back substitution"


def test_d8_unrelated_self_contained_problem_after_a_twin_is_not_judged_against_it():
    s, c, ids = w9(essay=False)
    assert not integrity.check_reply(s, c, OWN, "x2 = 2 and x1 = 1 [1]", force_item=_sticky(s, ids))


@pytest.mark.parametrize("q,r", [
    ("what is x1?", "x1 = 1 [1]"),                       # names one of the item's unknowns
    ("Problem 1: what is x1?", "x1 = 1 [1]"),
    (OWN, "The solution is x = (1, 1, 2) [1]."),         # the item's own whole answer is a leak in any context
])
def test_d8_a_question_that_refers_to_the_item_is_still_judged(q, r):
    s, c, ids = w9(essay=False)
    assert integrity.check_reply(s, c, q, r, force_item=_sticky(s, ids))


# ------------------------------------------------------------------ decision 9: routing precision
from fx9_world import world as w9b  # noqa: E402

SET3 = """# Problem Set 3

## Problem 1

How many grams of NaCl are needed to make 0.350 mol of solution?

## Problem 2

Find the empirical formula of a compound that is 40.0% carbon, 6.7% hydrogen and 53.3% oxygen.

## Problem 3

Explain why a limiting reagent stops a reaction.
"""


def set_world(tmp_path):
    s = Store(":memory:")
    c = s.add_course("Chem")
    f = tmp_path / "problem-set-3.md"
    f.write_text(SET3)
    mid = importers.import_file(s, c, f)
    integrity.mark_material_graded(s, mid)
    return s, c


@pytest.mark.parametrize("q", ["Is Problem 2 hard?", "How long is Problem 2?", "Is problem 2 harder than problem 1?",
                               "How long will Problem 2 take?", "Is Problem 2 doable in an hour?"])
def test_d9_size_questions_are_not_check_asks_and_do_not_route(q):
    s, c, ids = w9b()
    assert not integrity.is_check_ask(q)
    assert integrity.match_graded(s, c, q) is None


@pytest.mark.parametrize("q", ["Problem 2 is hard, how do I start?", "Is it hard to solve for d in Problem 2?",
                               "Is Problem 2 about determinants?"])
def test_d9_help_with_the_item_still_routes(q):
    s, c, ids = w9b()
    m = integrity.match_graded(s, c, q)
    assert m and m["id"] == ids["Problem 2"]


def test_d9_the_nacl_problem_is_one_item_not_the_set(tmp_path):
    s, c = set_world(tmp_path)
    m = integrity.match_graded(s, c, "Walk me through the NaCl problem on the problem set")
    assert m and m.get("id") and "NaCl" in m["body"] and not m.get("members")


def test_d9_a_set_ask_that_two_items_share_still_touches_the_whole_set(tmp_path):
    s, c = set_world(tmp_path)
    m = integrity.match_graded(s, c, "I'm stuck on the problem set about NaCl and carbon")
    assert m and len(m.get("members") or []) == 2


def test_d9_handout_note_needs_a_content_match_that_is_not_a_definition_question(tmp_path):
    import fx11_world
    s, c, mid = fx11_world.handout_only_questions(tmp_path)
    leak = "How do I compute the molarity of a brine sample and what happens to conductivity when it is diluted?"
    assert integrity.handout_sources(s, c, leak)[1] >= 1
    for q in ["How is the molarity of a brine sample defined?", "What is the molarity of brine and how do I compute it?"]:
        assert integrity.handout_sources(s, c, q)[1] == 0, q


def test_d9_handout_note_never_beside_a_source_row_of_that_handout(tmp_path):
    import fx11_world
    s, c, mid = fx11_world.handout_only_questions(tmp_path)
    q = "How do I compute the molarity of a brine sample and what happens to conductivity when it is diluted?"
    assert integrity.handout_sources(s, c, q)[1] >= 1
    assert integrity.handout_sources(s, c, q, listed={mid})[1] == 0
    named = "What does the lab handout say about the molarity of the brine sample?"
    assert integrity.handout_sources(s, c, named)[1] >= 1
    assert integrity.handout_sources(s, c, named, listed={mid})[1] == 0


# ------------------------------------------------------------------ decision 9: terse finished answers, banner agrees with fallback
SHORT6 = ("Problem 6 (6 points). Explain the difference between a solute and a solvent, give an example of each in a named "
          "solution, and state how concentration relates to the amount of solute.")
TERSE6 = ("A solvent is the part that dissolves things and the solute is what gets dissolved [1]. In a salt solution, water is the "
          "solvent and salt is the solute [1]. Concentration rises as the amount of solute rises in a fixed volume [1].")
COACH6 = ("Name what each word means in your own words. Then choose a solution you can name as your example and ask yourself what "
          "happens to concentration when you add more solute.")
TWO6 = ("A solvent is the part that dissolves things and the solute is what gets dissolved [1]. Salt and sugar are common solutes "
        "in water [1].")


def test_d9_terse_complete_answer_is_finished_and_honest_look_alikes_are_not():
    assert integrity.finished_answer(SHORT6, TERSE6)
    assert not integrity.finished_answer(SHORT6, COACH6)
    assert not integrity.finished_answer(SHORT6, TWO6)  # two of the three parts


def _short_world(tmp_path):
    s = Store(":memory:")
    c = s.add_course("Chem")
    s.add_graded_item(c, "Set 6: Problem 6", SHORT6, None, kind="essay", label="Problem 6")
    f = tmp_path / "ch.md"
    f.write_text("# Solutions\n\n## Parts\n\nA solvent dissolves a solute to make a solution. Concentration measures solute per volume.\n")
    importers.import_file(s, c, f)
    return s, c


def test_d9_terse_cited_answer_is_replaced_and_the_banner_agrees_with_the_fallback(tmp_path):
    s, c = _short_world(tmp_path)
    a = grounding.ask(s, c, SHORT6, FakeProvider(lambda system, messages: TERSE6 if "GRADED TASK" in messages[0]["content"] else "ok"))
    assert a.mode == "concepts"
    assert "salt solution" not in a.body_md
    assert "I can't write this answer" in a.body_md
    note = (a.routing_note or "").lower()
    assert "concepts and an approach" not in note and "held" in note
    assert "here are the concepts" not in a.body_md.lower()


def test_d9_coaching_reply_for_the_same_item_keeps_the_concepts_banner(tmp_path):
    s, c = _short_world(tmp_path)
    a = grounding.ask(s, c, SHORT6, FakeProvider(lambda system, messages: COACH6 if "GRADED TASK" in messages[0]["content"] else "ok"))
    assert a.mode == "concepts" and "ask yourself" in a.body_md.lower()
    assert "concepts and an approach" in (a.routing_note or "")
