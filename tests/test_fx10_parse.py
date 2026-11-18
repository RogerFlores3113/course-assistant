"""Cycle 10 number reading: each panel leak gets a regression test and an honest look-alike next to it.
Every fixture is original text."""
import json

import pytest

from course_assistant import integrity
from course_assistant.store import Store

LU = ("Problem 3 (15 points)\n\nFactor the matrix A with rows 3 2 1, 6 -2 5 and -3 4 -3 as LU, then solve A x = b "
      "for b = (7, 14, -5). Report L, U, y and x.")
LU_KEYS = {
    "matrix form": ["L = [[1,0,0],[2,1,0],[-1,-1,1]]", "U = [[3,2,1],[0,-6,3],[0,0,1]]", "y = (7, 0, 2)", "x = (1, 1, 2)"],
    "named entries": ["l21 = 2", "l31 = -1", "l32 = -1", "u11 = 3", "u12 = 2", "u13 = 1", "u22 = -6", "u23 = 3", "u33 = 1",
                      "y = (7, 0, 2)", "x = (1, 1, 2)"],
}
EIG = "Problem 4 (10 points)\n\nFind the eigenvalues and the eigenvectors of the matrix with rows 4 1 and 2 3."
EIG_KEYS = {
    "numbered": ["lambda1 = 5", "lambda2 = 2", "v1 = (1, 1)", "v2 = (1, -2)"],
    "plain": ["lambda = 5", "lambda = 2"],
    "pair": ["eigenvalues = (5, 2)"],
}
TOUCHED = {"sticky": True, "short": True}


def make(*specs):
    s = Store(":memory:")
    c = s.add_course("Course")
    for body, key in specs:
        s.add_graded_item(c, body.split("\n")[0], body, None, kind="problem", label=body.split(" (")[0],
                          answer_key=json.dumps(key) if key else "")
    return s, c


def item(s, n=1):
    return integrity.graded_view(s, 1)[n - 1]


def forced(s, n=1):
    return dict(item(s, n), reason="sticky", score=1.0)


def leaks(s, text, n=1, **ctx):
    return bool(integrity.key_leak(text, item(s, n), ctx or dict(TOUCHED)))


# --------------------------------------------------- decision 1: named unknowns as a set, prose lists of intermediate values
SET_LEAKS = ["z = 2, then y = 1, then x = 1", "x is 1 and z is 2", "Bottom up: z = 2; y = 1; x = 1.", "z=2, y=1, x=1",
             "You get z = 2, so y = 1, and finally x = 1 [1]", "y equals 1 and x equals 1"]
SET_OK = ["z = 4, then y = 3, then x = 7", "x = 1", "x = 1, y = 5", "z = 9 and y = 1",
          "Substitute z back into the second row to get y, then x."]


@pytest.mark.parametrize("kind", list(LU_KEYS))
@pytest.mark.parametrize("text", SET_LEAKS)
def test_p1_named_unknowns_in_any_order_are_a_hit_while_sticky(kind, text):
    s, _ = make((LU, LU_KEYS[kind]))
    assert leaks(s, text), text
    assert integrity.check_reply(s, 1, "How does back substitution work?", text, force_item=forced(s)), text


@pytest.mark.parametrize("kind", list(LU_KEYS))
@pytest.mark.parametrize("text", SET_OK)
def test_p1_honest_back_substitution_with_other_numbers_passes(kind, text):
    s, _ = make((LU, LU_KEYS[kind]))
    assert not leaks(s, text), text
    assert not integrity.check_reply(s, 1, "How does back substitution work?", text, force_item=forced(s)), text


def test_p1_named_set_needs_a_touched_context():
    s, _ = make((LU, LU_KEYS["matrix form"]))
    assert not leaks(s, "z = 2, then y = 1, then x = 1", sticky=False, short=True)


PROSE_LISTS = ["The pivots are 3, -6 and 1.", "The pivots are 3, -6 and 1 [1]", "the multipliers are 2, -1, -1",
               "The multipliers are 2, -1 and -1 [1]", "Multipliers: 2, -1, -1"]
PROSE_OK = ["The pivots are 4, 9 and 5.", "The multipliers are 3, 2, 7.", "You need one pivot per row.",
            "The multipliers are the numbers you subtract with, one per entry below a pivot."]


@pytest.mark.parametrize("kind", list(LU_KEYS))
@pytest.mark.parametrize("text", PROSE_LISTS)
def test_p1_a_prose_list_naming_the_quantity_counts_like_a_vector(kind, text):
    s, _ = make((LU, LU_KEYS[kind]))
    assert leaks(s, text, sticky=False, short=True), text
    assert integrity.check_reply(s, 1, "What are the pivots and multipliers in general?", text, force_item=forced(s)), text


@pytest.mark.parametrize("kind", list(LU_KEYS))
@pytest.mark.parametrize("text", PROSE_OK)
def test_p1_a_prose_list_of_other_values_passes(kind, text):
    s, _ = make((LU, LU_KEYS[kind]))
    assert not leaks(s, text), text


def test_p1_the_same_run_without_the_quantity_word_is_not_a_hit_outside_a_touched_chat():
    s, _ = make((LU, LU_KEYS["matrix form"]))
    assert not leaks(s, "In the lecture example the numbers 2, -1, -1 appear in a different matrix.", sticky=False, short=True)


# --------------------------------------------------- decision 2: prefixes and counts
HONEST_PREFIX = [
    "The lecture notes give 3 examples of elimination [1]",
    "The lecture gives two worked examples [1]",
    "The notes give 2 examples [1]",
    "The syllabus lists Monday 6 to 7 pm [1], but gives no room.",
    "Office hours are Tuesday 2 to 3 pm [1], but gives no location",
    "The exam is on Oct 1 [1], but gives no room",
    "Multiplying out becomes lambda^2 - 7 lambda + 10 [3]",
    "Answer: the lecture covers 3 topics [2]",
    "The result is that the quiz has 2 questions [1]",
]
HONEST_QUESTIONS = ["What do I do next?", "Which room is the TA session in?", "How many examples does the lecture give?",
                    "Which day are office hours?"]


@pytest.mark.parametrize("kind", list(LU_KEYS))
@pytest.mark.parametrize("q", HONEST_QUESTIONS)
@pytest.mark.parametrize("text", HONEST_PREFIX)
def test_p3_honest_gives_becomes_answer_sentences_pass_in_a_touched_chat(kind, q, text):
    s, _ = make((LU, LU_KEYS[kind]))
    srcs = [{"text": "x"}] * 3
    assert not integrity.check_reply(s, 1, q, text, force_item=forced(s), sources=srcs), (q, text)


PREFIX_LEAKS = ["gives 1 1 2", "The solution gives x = (1, 1, 2)", "Answer: 1, 1, 2", "Result: (1, 1, 2)", "x becomes 1 1 2",
                "So U gives\n3 2 1\n0 -6 3\n0 0 1", "The result is U = [[3,2,1],[0,-6,3],[0,0,1]]", "=> [1 1 2]"]


@pytest.mark.parametrize("text", PREFIX_LEAKS)
def test_p3_prefixes_before_a_vector_or_matrix_still_leak(text):
    s, _ = make((LU, LU_KEYS["matrix form"]))
    assert leaks(s, text), text


def test_p3_a_cite_marker_followed_by_a_comma_is_not_a_value():
    s, _ = make((LU, LU_KEYS["named entries"]))
    t = "The syllabus lists the room [1], but gives no time."
    assert integrity._reply_numbers(t) == []
    assert not integrity.check_reply(s, 1, "Which room is the TA session in?", t, force_item=forced(s), sources=[{"text": "x"}] * 2)


def test_p3_a_matrix_row_in_brackets_is_still_read():
    assert integrity._reply_numbers("x = [1], y = [1], z = [2]") == ["1", "1", "2"]
    assert integrity._reply_numbers("rows [1, 2], [3]") == ["1", "2", "3"]


def test_p3_which_points_at_a_value_only_when_it_names_one():
    s, _ = make((LU, LU_KEYS["matrix form"]))
    assert not integrity.question_refs(item(s), "Which room is the TA session in?")
    assert not integrity.question_refs(item(s), "Which day are office hours?")
    assert integrity.question_refs(item(s), "Which entry of x is the last one?")
    assert integrity.question_refs(item(s), "Which pivot is negative?")


def test_p3_counts_of_examples_hours_and_questions_are_prose():
    for t in ["The notes give 3 examples.", "Allow 2 hours.", "There are 2 questions on it.", "It has two worked examples."]:
        assert integrity.blank_counts(t).strip().split()[-1] != "" and not any(ch.isdigit() for ch in integrity.blank_counts(t)), t
    # a value stated for a name is not a count
    assert "2" in integrity.blank_counts("z = 2")
    assert "5.11" in integrity.blank_counts("Weigh out 5.11 g of salt in 2 steps")


# --------------------------------------------------- decision 3: small keys in a touched or named context
EIG_LEAKS = ["The eigenvalues are 5 and 2.", "lambda = 5", "Answer: lambda = 5 and lambda = 2", "eigenvalue is 5",
             "The eigenvalues of A are 5 and 2.", "The characteristic polynomial gives eigenvalues 5 and 2 [3]",
             "Eigenvalues: 5, 2", "The larger eigenvalue is 5", "lambda1 = 5 and lambda2 = 2"]
EIG_OK = ["The eigenvalues are 3 and 4.", "lambda = 7", "Each eigenvalue lambda satisfies det(A - lambda I) = 0.",
          "The trace is the sum of the eigenvalues.", "An eigenvalue can be zero only when the matrix is singular."]


@pytest.mark.parametrize("kind", list(EIG_KEYS))
@pytest.mark.parametrize("text", EIG_LEAKS)
def test_p5_small_key_bound_to_the_items_symbol_leaks_in_a_touched_chat(kind, text):
    s, _ = make((EIG, EIG_KEYS[kind]))
    assert leaks(s, text), text
    assert leaks(s, text, named=True), text
    assert integrity.check_reply(s, 1, "What do I do after the characteristic polynomial?", text, force_item=forced(s)), text


@pytest.mark.parametrize("kind", list(EIG_KEYS))
@pytest.mark.parametrize("text", EIG_OK)
def test_p5_honest_eigenvalue_talk_passes(kind, text):
    s, _ = make((EIG, EIG_KEYS[kind]))
    assert not leaks(s, text), text
    assert not leaks(s, text, named=True), text


def test_p5_untouched_contexts_keep_the_small_number_rule():
    s, _ = make((EIG, EIG_KEYS["numbered"]))
    assert not leaks(s, "The eigenvalues are 5 and 2.", sticky=False, short=True)
    assert not leaks(s, "lambda = 5", sticky=False, short=True)


def test_p5_a_lecture_example_whose_numbers_are_in_the_cited_chunk_is_data():
    s, _ = make((EIG, EIG_KEYS["numbered"]))
    text = "The lecture gives eigenvalues 5 and 2 for its example matrix."
    assert leaks(s, text)
    assert not leaks(s, text, sticky=True, short=True, data=["5", "2"])
    assert not leaks(s, "lambda = 5", sticky=True, short=True, data=["5", "2"])
    # the data exemption is per value: a different key value is still a leak
    assert leaks(s, "lambda = 5 and lambda = 2", sticky=True, short=True, data=["5"])


# --------------------------------------------------- cards and guides go through the same readers
CARD_LEAKS = [("What are the pivots of the LU factorization of Problem 3?", "3, -6 and 1."),
              ("What are the multipliers in Problem 3?", "2, -1, -1."),
              ("What are the eigenvalues in Problem 4?", "5 and 2."),
              ("Back substitution in Problem 3", "z = 2, then y = 1, then x = 1")]
CARD_OK = [("What is the pivot rule?", "Pivots are the leading nonzero entries."),
           ("What is an eigenvalue?", "A scalar lambda with A v = lambda v."),
           ("Back substitution order", "Solve the last unknown first, then work up.")]


@pytest.mark.parametrize("front,back", CARD_LEAKS)
def test_cards_with_pivots_multipliers_eigenvalues_or_a_named_set_are_refused(front, back):
    s, _ = make((LU, LU_KEYS["named entries"]), (EIG, EIG_KEYS["numbered"]))
    assert integrity.check_card(s, 1, front, back), (front, back)


@pytest.mark.parametrize("front,back", CARD_OK)
def test_honest_method_cards_are_kept(front, back):
    s, _ = make((LU, LU_KEYS["named entries"]), (EIG, EIG_KEYS["numbered"]))
    assert not integrity.check_card(s, 1, front, back), (front, back)


def test_guides_with_unnamed_multipliers_or_eigenvalues_are_refused_and_method_guides_kept():
    s, _ = make((LU, LU_KEYS["named entries"]), (EIG, EIG_KEYS["numbered"]))
    assert integrity.check_guide(s, 1, "Problem 3 notes", "The multipliers are 2, -1, -1. [1]")
    assert integrity.check_guide(s, 1, "Problem 4 notes", "The eigenvalues are 5 and 2. [1]")
    assert not integrity.check_guide(s, 1, "Elimination", "Use multipliers to clear entries below each pivot. [1]")
