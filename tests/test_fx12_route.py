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


# ------------------------------------------------------------------ decision 9: proofs without a set opening
P4 = ("Problem 4 (10 points). Prove that if a linear system Ax = b has two different solutions, then it has infinitely "
      "many solutions.")
PROOFS = [
    "If Ax = b has two solutions x and y, then A(x - y) = 0, and x + t(x - y) solves Ax = b for every t, so there are infinitely many.",
    "Two solutions x and y give A(x-y) = 0; adding any multiple of x-y to x keeps Ax = b, so there are infinitely many solutions.",
    "Take the midpoint of two solutions; it solves the system too, and repeating this with the new pair gives infinitely many solutions.",
    "Every point on the line through the two solutions x and y is a solution, since A(x + t(y - x)) = b, so the system has infinitely many solutions.",
    "The difference of two solutions is a nonzero vector sent to zero, and any multiple of it can be added to a solution, which gives infinitely many solutions.",
]
HONEST = [
    "To show infinitely many solutions, a good method is to build new solutions from the ones you are given.",
    "Try subtracting the two solutions and see what A does to the difference.",
    "The difference of two solutions of Ax = b is a solution of Ax = 0.",
    "A system with exactly one solution has a pivot in every column.",
    "Infinitely many solutions happen when there is a free variable.",
]


@pytest.mark.parametrize("r", PROOFS)
def test_d9_a_two_solution_proof_is_finished_whatever_its_opening(r):
    it = {"kind": "proof", "body": P4}
    assert integrity._leaks_proof(it, r, "")


@pytest.mark.parametrize("r", HONEST)
def test_d9_honest_method_sentences_without_the_items_steps_pass(r):
    it = {"kind": "proof", "body": P4}
    assert not integrity._leaks_proof(it, r, "")


# ------------------------------------------------------------------ decision 9: equation-line systems read by their rows
SYS_A = ("Problem 1 (10 points). Solve the system 2x + 5y + z = 0, 4x + 11y + z = 3, -2x - 2y - 5z = 11 by elimination and "
         "report (x, y, z).")
SYS_A_KEY = ["x = -3/2", "y = 1", "z = -2", "x = (-3/2, 1, -2)", "U = [[2, 5, 1], [0, 1, -1], [0, 0, -1]]", "c = (0, 3, 2)",
             "l21 = 2", "l31 = -1", "l32 = 2"]
SYS_B = "Problem 3 (10 points). Solve the system x + y + z = 4, 2x + 3y + z = 9, x + y + 3z = 8 for (x, y, z)."
SYS_B_KEY = ["x = (1, 1, 2)", "x1 = 1", "x2 = 1", "x3 = 2", "c = (4, 5, 4)"]


def sys_world():
    s = Store(":memory:")
    c = s.add_course("LA")
    ib = s.add_graded_item(c, "Set: Problem 3", SYS_B, None, kind="problem", label="Problem 3", answer_key=json.dumps(SYS_B_KEY))
    ia = s.add_graded_item(c, "Set: Problem 1", SYS_A, None, kind="problem", label="Problem 1", answer_key=json.dumps(SYS_A_KEY))
    return s, c, ia, ib


SYS_LEAKS = ["The reduced system is 2x + 5y + z = 0 / y - z = 3 / -z = 2 [1]", "2x + 5y + z = 0\ny - z = 3\n-z = 2",
             "Row 3 reads -z = 2.", "-z = 2", "The last equation becomes -z = 2."]
SYS_HONEST = ["Elimination ends with an upper triangular system, then you back substitute.",
              "A system such as x + y = 9 / y = 4 is already triangular, so start from the last equation.",
              "For example 3x + y = 5 and -z = 7 are equations; the sign of the pivot matters."]


@pytest.mark.parametrize("r", SYS_LEAKS)
def test_d9_equation_lines_are_read_by_their_rows_and_blamed_on_the_matching_item(r):
    s, c, ia, ib = sys_world()
    st = dict(s.graded_item(ia), kind="problem", reason="sticky", source="item", score=1.0)
    assert integrity.check_reply(s, c, "what does the last row say?", r, force_item=st)["id"] == ia
    g = integrity.check_guide(s, c, "Notes", r)
    assert g and g["id"] == ia


@pytest.mark.parametrize("r", SYS_HONEST)
def test_d9_triangular_system_look_alikes_pass(r):
    s, c, ia, ib = sys_world()
    st = dict(s.graded_item(ia), kind="problem", reason="sticky", source="item", score=1.0)
    assert integrity.check_reply(s, c, "how does elimination end?", r, force_item=st) is None
    assert integrity.check_guide(s, c, "Notes", r) is None


# ------------------------------------------------------------------ decision 9: part references
NEW_PART_FORMS = ["hw 3a", "HW3a", "hw3 b", "3 b what is x", "3-b", "homework 3b", "pset 3 b", "Problem three b", "problem 3 b"]


@pytest.mark.parametrize("q", NEW_PART_FORMS)
def test_d9_part_forms_with_spaces_hyphens_and_hw_prefixes_resolve(q):
    s, c, ids = w9b(essay=False)
    s.add_graded_item(c, "Problem Set 4: Problem 3(b)", "Problem 3(b). Say what a pivot is.", None, kind="problem", label="Problem 3(b)")
    pairs = integrity._prob_pairs(q)
    want = "a" if q.lower().replace(" ", "").endswith("3a") else "b"
    assert ("3", want) in pairs, (q, pairs)


@pytest.mark.parametrize("q,num", [("number 1", "1"), ("the third problem", "3"), ("What is the fourth question about?", "4"),
                                   ("Problem set 2 number 3", "3"), ("ps2 1", "1"), ("pset 2 2", "2")])
def test_d9_number_ordinal_and_set_forms_name_a_problem(q, num):
    assert num in integrity._prob_refs(q), q


@pytest.mark.parametrize("q", ["I need 5g of salt for 3d printing", "hw 3 is due Friday", "ps2 10 points late", "pset 2 3 hours",
                               "She has a 3 a.m. shift", "the third term of the series",
                               "phone number five", "Homework 2 is hard"])
def test_d9_part_form_look_alikes_name_no_problem(q):
    pairs = integrity._prob_pairs(q)
    assert not [x for x in pairs if x[1]] and not pairs, (q, pairs)


def test_d9_ps2_1_resolves_to_the_item_with_a_label(tmp_path):
    s, c = set_world(tmp_path)
    for t in s.graded_items(c):
        pass
    m = integrity.match_graded(s, c, "ps3 1")
    assert m and m.get("id") and "NaCl" in m["body"]
    assert m.get("label") or m.get("title")
