"""Hotfix pass after the cycle 7 panel: numeric leak shapes and sticky over-blocking, each regression next to its look-alike."""
import json

import pytest

from course_assistant import grounding, integrity, mcp_server
from course_assistant.providers import FakeProvider
from test_fx7_route import held, make, model

Q_AUG = "What does the augmented matrix look like at the end of elimination in problem 2?"
PRIMED = ("What is d in problem 1?", "Think about which row exchange the pivot forces.")


def item(s, c, label, needle):
    return next(i for i in integrity.graded_view(s, c) if i["label"] == label and needle in i["body"])


def sticky_chat(reply_for, opener=PRIMED[0], opener_reply=PRIMED[1], retry=None):
    """A chat touched by Problem 1; later questions get reply_for(question)."""
    s, c, m, ids = make()

    def f(system, messages):
        if "<twin_problem>" in system and "Reply in exactly this format" in system:
            from test_fx3_integrity import TWIN_OK
            return TWIN_OK
        if "THE STUDENT'S GRADED TASK" in messages[0]["content"]:
            return "Review how the method works."
        t = messages[-1]["content"]
        if "Answer again" in t:
            return retry or "Here is the method in words: eliminate downward, then back-substitute."
        if opener in t:
            return opener_reply
        return reply_for(t)
    p = FakeProvider(f)
    grounding.ask(s, c, opener, p)
    return s, c, p


# ------------------------------------------------------------------ 1. a name prefix on the first row of a matrix
LEAKY_X = ["Result:\n\n```\nX = -2 3\n     1 0\n    -1 1\n```\n[1]", "X = [-2 3\n 1 0\n -1 1]", "X = [-2 3 / 1 0 / -1 1] [1]",
           "X = -2 3 / 1 0 / -1 1 [1]"]
OTHER_X = "Result:\n\n```\nX = 4 7\n    2 9\n    5 8\n```\n[1]"


@pytest.mark.parametrize("text", LEAKY_X)
def test_name_prefixed_matrix_is_caught_in_ask_save_guide_and_cards(text):
    s, c, m, ids = make()
    a = grounding.ask(s, c, Q_AUG, FakeProvider(model({"augmented matrix": text})), route=False)
    assert held(a) and "-2 3" not in a.body_md
    mcp_server.set_store(s)
    try:
        assert mcp_server.save_guide(c, "answer", "Notes", text, []).startswith("NOT SAVED")
        assert mcp_server.add_flashcards(c, [{"front": "Result", "back": text}]).startswith("NOT ADDED")
    finally:
        mcp_server.set_store(None)


def test_name_prefixed_matrix_look_alike_with_other_numbers_passes():
    s, c, m, ids = make()
    a = grounding.ask(s, c, Q_AUG, FakeProvider(model({"augmented matrix": OTHER_X})), route=False)
    assert a.mode == "answer" and "X = 4 7" in a.body_md
    mcp_server.set_store(s)
    try:
        assert mcp_server.save_guide(c, "answer", "Notes", OTHER_X, []).startswith("Saved")
    finally:
        mcp_server.set_store(None)


# ------------------------------------------------------------------ 2 and 3a. a citation marker is not a number
def test_which_lectures_cover_question_is_not_refused_in_a_touched_chat():
    s, c, m, ids = make()
    ans = "LU factorization is introduced in the inverses and LU notes [1]."
    p = FakeProvider(model({"augmented matrix": "x1 = (-2, 1, -1) and x2 = (3, 0, 1)", "cover": ans, "defined": ans}))
    held_first = grounding.ask(s, c, Q_AUG, p)
    assert held(held_first)
    for q in ("Which lectures cover LU?", "Which weeks cover inverses?", "How is the LU factorization defined?"):
        a = grounding.ask(s, c, q, p)
        assert a.mode == "answer" and "introduced in the inverses" in a.body_md, q


def test_citation_marker_never_matches_a_key_value_but_a_real_value_still_does():
    s, c, m, ids = make()
    sticky = dict(item(s, c, "Problem 2", "augmented"), reason="sticky")
    q = "Which lectures cover LU?"
    assert integrity.check_reply(s, c, q, "LU is introduced in the inverses notes [1].", force_item=sticky) is None
    assert integrity.check_reply(s, c, q, "LU is introduced in the inverses notes [1, 2].", force_item=sticky) is None
    assert integrity.check_reply(s, c, q, "The LU notes say x1 = (-2, 1, -1) [1].", force_item=sticky) is not None


# ------------------------------------------------------------------ 3. the student's own equation, logistics counts, a quoted example
def test_students_own_equation_restated_is_not_a_leak():
    q = "Give me a tiny elimination example like 2x + y = 5 and x - y = 1"
    s, c, p = sticky_chat(lambda t: "Take 2x + y = 5 and x - y = 1. Adding the two equations eliminates y, which leaves 3x = 6.")
    a = grounding.ask(s, c, q, p)
    assert a.mode == "answer" and "2x + y = 5" in a.body_md and "reworded" not in a.body_md


def test_a_value_the_student_did_not_type_is_still_a_leak():
    q = "Give me a tiny elimination example like 2x + y = 5 and x - y = 1"
    s, c, p = sticky_chat(lambda t: "Take 2x + y = 5. For your own system d = 11 and z = -2.")
    a = grounding.ask(s, c, q, p)
    assert "d = 11" not in a.body_md


@pytest.mark.parametrize("reply", ["There are 11 problem sets and the lowest is dropped [1].",
                                   "Each problem set is worth 11 points, and eleven problem sets count [1].",
                                   "The course has eleven problem sets, the exam is worth 10 points and Midterm 1 is worth 11 points [1]."])
def test_logistics_counts_are_not_key_values(reply):
    s, c, p = sticky_chat(lambda t: reply)
    a = grounding.ask(s, c, "When is Midterm 1 and how many problem sets are dropped?", p)
    assert a.mode == "answer" and "problem set" in a.body_md


def test_a_chemistry_value_that_is_a_count_in_prose_still_counts_when_bound_to_the_name():
    s, c, p = sticky_chat(lambda t: "As a parameter, d = 11 makes the third pivot vanish.")
    a = grounding.ask(s, c, "How many problem sets are dropped?", p)
    assert "d = 11" not in a.body_md


LECTURE = {"text": "Worked example. Factor the matrix with rows 4 3 5, 12 10 22 and 8 9 30. The multipliers are 3 and 2, and "
                   "after the third step U ends in 7. We get L with 3, 2 and 7 below the diagonal.",
           "heading": "Lecture 6", "material_id": 99}


def test_worked_example_quoted_from_a_cited_lecture_is_fine_and_a_copy_of_the_item_is_not():
    s, c, m, ids = make()
    sticky = dict(item(s, c, "Problem 3", "forward"), reason="sticky")
    q = "Walk me through one tiny elimination example"
    reply = "Take the lecture's matrix: the multipliers are 3 and 2, and U ends in 7 [1]."
    assert integrity.check_reply(s, c, q, reply, force_item=sticky, sources=[LECTURE]) is None
    # numbers that are not in the cited passage are the student's problem leaking
    bad = "Take the lecture's matrix: the multipliers are 3 and 2, and U ends in 7 and 5 [1]. So x = (2, 1, -2)."
    assert integrity.check_reply(s, c, q, bad, force_item=sticky, sources=[LECTURE]) is not None
    # the cited passage is the graded problem itself
    own = {"text": item(s, c, "Problem 3", "forward")["body"], "heading": "PS2", "material_id": 1}
    assert integrity.check_reply(s, c, q, "The answer is x = (2, 1, -2) and y = (2, 6, -10) [1].", force_item=sticky,
                                 sources=[own]) is not None


def test_reworded_retry_says_it_was_reworded():
    s, c, p = sticky_chat(lambda t: "Here d = 11 and z = -2.")
    a = grounding.ask(s, c, "Give me an elimination example", p)
    assert a.mode == "answer" and "d = 11" not in a.body_md
    assert "I reworded this answer to stay clear of your graded Problem 1" in a.body_md
    s2, c2, p2 = sticky_chat(lambda t: "Elimination subtracts a multiple of one row from another [1].")
    b = grounding.ask(s2, c2, "Give me an elimination example", p2)
    assert "reworded" not in b.body_md


def test_textbook_molarity_answer_with_a_citation_is_not_blocked_by_the_chemistry_key():
    s, c, m, ids = make()
    p2 = item(s, c, "Problem 2", "empirical")
    s.update_graded_item(p2["id"], answer_key=json.dumps(["C = 1", "H = 1.33", "O = 1", "C3H4O3"]))
    textbook = "How many grams of KCl are in 500.0 mL of 0.200 M solution? It is a textbook practice problem."
    reply = "Moles are 0.5000 L times 0.200 M = 0.100 mol, and 0.100 mol times 74.55 g/mol gives 7.46 g of KCl [1]."
    pr = FakeProvider(model({"chemistry pset": "Think about the mole ratios.", "KCl": reply}))
    grounding.ask(s, c, "I need help with problem 2 of the chemistry pset", pr)
    a = grounding.ask(s, c, textbook, pr)
    assert a.mode == "answer" and "7.46 g" in a.body_md


# ------------------------------------------------------------------ 4. a finished proof with no end marker
PROOF = ("Suppose x and y are two different solutions of Ax = b, so Ax = b and Ay = b. Consider z = x + t(y - x) for any real "
         "number t. Then Az = Ax + t(Ay - Ax) = b + t(b - b) = b. Hence z is a solution for every t. Since y differs from x, "
         "each t gives a different z, so there are infinitely many solutions. Therefore Ax = b never has exactly two solutions.")
HINT = ("A system Ax = b has no solution, one solution or infinitely many. Think about what happens when you have two "
        "solutions and combine them. Which combination of x and y might also solve it? Try it with a small example.")
LECTURE_PROOF = ("Suppose v and w are eigenvectors of B for the same eigenvalue. Then Bv = kv and Bw = kw. Hence B(v + w) = "
                 "kv + kw = k(v + w). Therefore v + w is an eigenvector for k too, so the eigenspace is closed under addition.")


def test_finished_proof_without_an_end_marker_is_caught_in_plain_ask():
    s, c, m, ids = make()
    q = "why can't Ax=b have exactly two solutions"
    a = grounding.ask(s, c, q, FakeProvider(model(default=PROOF)), route=False)
    assert held(a) and "Consider z = x + t(y - x)" not in a.body_md
    b = grounding.ask(s, c, q, FakeProvider(model(default=HINT)), route=False)
    assert b.mode == "answer" and "Think about" in b.body_md


def test_a_proof_of_something_else_and_a_short_explanation_are_not_caught():
    s, c, m, ids = make()
    a = grounding.ask(s, c, "Why is the sum of two eigenvectors for one eigenvalue an eigenvector?",
                      FakeProvider(model(default=LECTURE_PROOF)), route=False)
    assert a.mode == "answer"
    assert not integrity.proof_shaped(HINT)
    assert integrity.proof_shaped(PROOF)


# ------------------------------------------------------------------ 5. a negated mention does not name the set
@pytest.mark.parametrize("q", ["How many grams of KCl are in 500.0 mL of 0.200 M solution? It is a textbook problem, not the pset.",
                               "Molarity of a solution: this isn't homework, how many grams of KCl for 500.0 mL of 0.200 M?",
                               "How many grams of KCl for 500.0 mL of 0.200 M? Not from the problem set, just practice."])
def test_negated_set_mention_does_not_route_to_the_set(q):
    s, c, m, ids = make()
    assert integrity.match_graded(s, c, q) is None


def test_a_real_set_mention_still_routes():
    s, c, m, ids = make()
    got = integrity.match_graded(s, c, "I'm stuck on the molarity question from my problem set")
    assert got is not None
    assert integrity.drop_negated_sets("I am not sure how the pset works") == "I am not sure how the pset works"


# ------------------------------------------------------------------ 6. cards and renderings
def test_cards_naming_the_unknown_with_a_key_value_back_are_refused():
    s, c, m, ids = make()
    s.update_graded_item(item(s, c, "Problem 3", "forward")["id"], answer_key=json.dumps(["x1 = 2", "x2 = 1", "x3 = -2"]))
    s.update_graded_item(item(s, c, "Problem 1", "dy")["id"], answer_key=json.dumps(["d = 10", "d = 11"]))
    mcp_server.set_store(s)
    try:
        for f, b in (("x for problem 3 -> ", "(2, 1, -2)"), ("Singular d", "11"), ("Singular d in the pset", "11")):
            assert mcp_server.add_flashcards(c, [{"front": f, "back": b}]).startswith("NOT ADDED"), f
        for f, b in (("What does a pivot do?", "It divides the rows below"), ("Vitamin C per serving", "11 mg")):
            assert mcp_server.add_flashcards(c, [{"front": f, "back": b}]).startswith("Added"), f
    finally:
        mcp_server.set_store(None)


GENERIC = "Walk me through Gaussian elimination with a parameter d in the coefficient matrix and the solution"


@pytest.mark.parametrize("reply", ["x = −3⁄2, y = 1, z = −2", "$x=-\\dfrac32$, $y=1$, $z=-2$",
                                   "| var | value |\n|---|---|\n| x | -3/2 |\n| y | 1 |\n| z | -2 |",
                                   "| x | y | z |\n|---|---|---|\n| -3/2 | 1 | -2 |",
                                   "$x=-\\frac{3}{2}$, $y=1$ and $z=-2$"])
def test_alternate_renderings_of_the_solution_are_caught(reply):
    s, c, m, ids = make()
    a = grounding.ask(s, c, GENERIC, FakeProvider(model({"parameter": reply})))
    assert held(a)


def test_look_alike_tables_and_fractions_pass():
    s, c, m, ids = make()
    for reply in ("| var | value |\n|---|---|\n| x | 3/4 |\n| y | 5 |\n| z | 8 |", "$x=-\\dfrac32$ is the slope of the line [1]",
                  "Gaussian elimination subtracts a multiple of one row from another [1]."):
        a = grounding.ask(s, c, GENERIC, FakeProvider(model({"parameter": reply})))
        assert a.mode == "answer", reply


def test_two_bound_values_of_the_parameter_are_a_leak_one_unrelated_value_is_not():
    s, c, m, ids = make()
    s.update_graded_item(item(s, c, "Problem 1", "dy")["id"], answer_key=json.dumps(["d = 10", "d = 11", "z = -2"]))
    bad = "With a parameter d: d = 10 gives a zero pivot after the exchange and d = 11 makes the system singular."
    a = grounding.ask(s, c, GENERIC, FakeProvider(model({"parameter": bad})))
    assert held(a)
    ok = "With a parameter d, try d = 4 and d = 5 in your own notes to see how the pivot changes [1]."
    b = grounding.ask(s, c, GENERIC, FakeProvider(model({"parameter": ok})))
    assert b.mode == "answer"
