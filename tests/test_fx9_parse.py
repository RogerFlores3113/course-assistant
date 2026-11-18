"""Cycle 9 number, unit and matrix reading: one regression test per panel leak, each with its honest look-alike.
Every fixture is original text."""
import json

import pytest

from course_assistant import integrity
from course_assistant.store import Store

LU = ("Problem 3 (15 points)\n\nFactor the matrix A with rows 3 2 1, 6 -2 5 and -3 4 -3 as LU, then solve A x = b "
      "for b = (7, 14, -5). Report L, U, y and x.")
LU_KEY = ["L = [[1,0,0],[2,1,0],[-1,-1,1]]", "U = [[3,2,1],[0,-6,3],[0,0,1]]", "y = (7, 0, 2)", "x = (1, 1, 2)"]
SYS = ("Problem 1 (15 points)\n\nSolve x + 2y + z = 1, 2x + 5y - z = 12, -x + y + 2z = -7 by elimination and back "
       "substitution. Report the solution.")
SYS_KEY = ["x = 2", "y = 1", "z = -3"]
SOLVE = ("Problem 2 (10 points)\n\nSolve A X = B where A has rows 1 1 1, 0 1 1 and 0 0 1 and B has rows 0 1, 3 -1 and 2 -1.")
SOLVE_KEY = ["X = [[-3,2],[1,0],[2,-1]]"]
DIL = ("Pre-lab question 2 (5 points)\n\nDilute 25.00 mL of 0.100 M stock to 250.0 mL (dilution A), then 25.00 mL of A to "
       "250.0 mL (dilution B). Give the molarity of each.")
DIL_KEY = ["cA = 0.0100 M", "cB = 0.00100 M"]
HYD = ("Problem 5 (10 points)\n\nFind the molar mass of copper(II) sulfate pentahydrate, then the mass needed to make "
       "500.0 mL of a 0.1000 M solution.")
HYD_KEY = ["M = 249.68 g/mol", "m = 12.48 g"]
GUIDE = {"named": True, "guide": True, "small": True}


def make(*specs):
    s = Store(":memory:")
    c = s.add_course("Course")
    for body, key in specs:
        s.add_graded_item(c, body.split("\n")[0], body, None, kind="problem", label=body.split(" (")[0],
                          answer_key=json.dumps(key) if key else "")
    return s, c


def it(s, n=1):
    return integrity.graded_view(s, 1)[n - 1]


def sticky(s, n=1):
    return dict(it(s, n), reason="sticky", score=1.0)


def kl(s, text, n=1, **ctx):
    return integrity.key_leak(text, it(s, n), ctx)


# ------------------------------------------------------------------ decision 2: counts of things are not values
SYS_CTX = {"sticky": True, "short": True, "question_refs": True}
PROSE_COUNTS = [
    "After elimination the last equation has only one unknown, so you can solve it directly.",
    "The last equation has 1 unknown.",
    "There is exactly one solution when the pivots are all nonzero.",
    "Elimination works one pivot at a time.",
    "You can end up with a single free variable.",
    "You are left with two equations after the first step.",
    "A system like this has 1 solution when every column has a pivot.",
    "Each row needs one pivot, and you go one at a time.",
    "Back substitution fills in one unknown per equation.",
]


@pytest.mark.parametrize("text", PROSE_COUNTS)
def test_p1_prose_counts_are_not_key_values_while_sticky(text):
    s, c = make((SYS, SYS_KEY))
    assert not kl(s, text, **SYS_CTX), text
    assert not integrity.check_reply(s, 1, "Why does back substitution start at the bottom row?", text + " [3]",
                                     force_item=sticky(s)), text


BOUND_ONES = ["y = 1", "y is one", "y equals 1", "So y = 1 and x = 2.", "The value of y is 1.", "It is 1.", "1."]


@pytest.mark.parametrize("text", BOUND_ONES)
def test_p1_a_value_bound_to_a_name_or_standing_alone_still_leaks_while_sticky(text):
    s, c = make((SYS, SYS_KEY))
    assert kl(s, text, **SYS_CTX), text


def test_p1_counts_next_to_a_real_value_do_not_hide_it():
    s, c = make((SYS, SYS_KEY))
    assert kl(s, "Each row has one pivot, so z = -3.", **SYS_CTX)
    assert kl(s, "With one unknown left, y = 1.", **SYS_CTX)


def test_p1_the_own_example_ends_in_one_but_the_retry_wording_passes():
    s, c = make((SYS, SYS_KEY))
    assert not integrity.check_reply(s, 1, "How do I back substitute?", "Work upward one pivot at a time, solving for "
                                     "one unknown in each equation [2].", force_item=sticky(s))


# ------------------------------------------------------------------ decision 1: an unnamed key vector in a touched context
VEC_FORMS = [
    "(1, 1, 2)", "[1, 1, 2]", "1, 1, 2", "1 1 2", "[1; 1; 2]", "(1;1;2)", "<1,1,2>", "⟨1, 1, 2⟩",
    "(x1, x2, x3) = (1, 1, 2)", "The solution vector is (1, 1, 2).", "Solving gives (1, 1, 2) as the solution.",
    "Answer: 1 1 2", "x is the vector (1, 1, 2)", "one, one, two", "The entries are 1, 1 and 2.",
    "{\"x\": [1, 1, 2]}", "Result: (1, 1, 2)",
]


@pytest.mark.parametrize("text", VEC_FORMS)
def test_p6_unnamed_key_vector_is_a_hit_while_the_chat_is_touched(text):
    s, c = make((LU, LU_KEY))
    assert kl(s, text, sticky=True, short=True), text
    assert integrity.check_reply(s, 1, "what was x again?", text, force_item=sticky(s)), text
    assert kl(s, text, named=True), text


@pytest.mark.parametrize("text", VEC_FORMS)
def test_p6_the_same_vector_in_a_fresh_chat_stays_a_plain_example(text):
    s, c = make((LU, LU_KEY))
    assert not kl(s, text), text
    assert not integrity.check_reply(s, 1, "what is an example of a solution vector?", text), text


TOUCHED_FINE = [
    "(2, 1, 1)", "[1, 2, 1]", "1, 2, 3", "The rank of [[1,2],[2,4]] is 1.", "A has b = (7, 14, -5) on the right.",
    "Take the vector (1, 0, 0) and (0, 1, 0).", "x has three entries, one per column.", "(1, 1)",
]


@pytest.mark.parametrize("text", TOUCHED_FINE)
def test_p6_other_vectors_and_the_problems_own_data_pass_in_a_touched_chat(text):
    s, c = make((LU, LU_KEY))
    assert not kl(s, text, sticky=True, short=True), text
    assert not integrity.check_reply(s, 1, "Why does LU need row exchanges sometimes?", text, force_item=sticky(s)), text


@pytest.mark.parametrize("text", ["The solution is (1, 1, 2). [1]", "The answer vector is 1, 1, 2.", "Answer: [1; 1; 2]",
                                  "x -> (1, 1, 2)", "Problem 3 answer vector\n(1, 1, 2)"])
def test_p6_a_saved_guide_or_card_that_names_the_answer_refuses_the_bare_vector(text):
    s, c = make((LU, LU_KEY))
    assert integrity.check_guide(s, 1, "Notes on LU", text) or integrity.check_card(s, 1, "Problem 3 answer vector", text), text
    assert kl(s, text, **GUIDE) or "x ->" in text, text


@pytest.mark.parametrize("text", ["A basis such as (1, 1, 2) spans a line.", "Try (1, 1, 2) as a test vector for the dot product.",
                                  "Lecture 4: the vectors (1, 1, 2), (0, 1, 0) and (3, 0, 1) are independent."])
def test_p6_a_saved_guide_that_only_uses_the_numbers_as_an_example_is_kept(text):
    s, c = make((LU, LU_KEY))
    assert not kl(s, text, **GUIDE), text


def test_p6_cards_front_names_the_answer_or_not():
    s, c = make((LU, LU_KEY))
    assert integrity.check_card(s, 1, "Problem 3 answer vector", "(1, 1, 2)")
    assert integrity.check_card(s, 1, "Problem 3 solution", "[1; 1; 2]")
    assert integrity.check_card(s, 1, "Problem 3 answer vector", "1 1 2")
    assert not integrity.check_card(s, 1, "What is a unit vector?", "A vector of length 1, like (1, 0, 0).")
    assert not integrity.check_card(s, 1, "Is (1, 1, 2) a unit vector?", "No, its length is the square root of 6.")


# ------------------------------------------------------------------ decision 3: prefixes and plain factors
U_ROWS = [
    "U = 3 2 1 / 0 -6 3 / 0 0 1", "U = [[3,2,1],[0,-6,3],[0,0,1]]", "U = np.array([[3,2,1],[0,-6,3],[0,0,1]])",
    "```\n3 2 1\n0 -6 3\n0 0 1\n```", "[3 2 1; 0 -6 3; 0 0 1]", "Result: U = 3 2 1 / 0 -6 3 / 0 0 1",
    "\\begin{bmatrix}3&2&1\\\\0&-6&3\\\\0&0&1\\end{bmatrix}", "U -> 3 2 1 / 0 -6 3 / 0 0 1", "Answer: U = 3 2 1\n0 -6 3\n0 0 1",
    "0 -6 3 / 0 0 1", "[[0,-6,3],[0,0,1]]", "```\n0 -6 3\n0 0 1\n```", "Final: 3 2 1 / 0 -6 3 / 0 0 1",
    "So U = [3 2 1; 0 -6 3; 0 0 1]", "U gives 3 2 1 / 0 -6 3 / 0 0 1", "=> 3 2 1 / 0 -6 3 / 0 0 1",
]


@pytest.mark.parametrize("text", U_ROWS)
def test_p3_plain_u_rows_match_even_when_row_one_is_a_given(text):
    s, c = make((LU, LU_KEY))
    assert kl(s, text, **GUIDE), text
    assert kl(s, text, named=True), text


U_FINE = [
    "3 2 1 / 6 -2 5 / -3 4 -3",                      # the problem's own matrix A
    "A = 3 2 1 / 6 -2 5 / -3 4 -3",
    "0 -6 3",                                         # one row alone
    "[[0, 0, 1]]",
    "3 2 1 / 0 -5 2 / 0 0 4",                         # another matrix with the same first row
    "```\n3 2 1\n0 -4 7\n0 0 2\n```",
    "E = 1 0 0 / -2 1 0 / 0 0 1",                     # an elementary matrix
]


@pytest.mark.parametrize("text", U_FINE)
def test_p3_a_matrix_that_is_not_u_passes(text):
    s, c = make((LU, LU_KEY))
    assert not kl(s, text, **GUIDE), text
    assert not kl(s, text, named=True), text


X_PREFIX = [
    "Result: X = -3 2 / 1 0 / 2 -1", "Result:\nX = -3 2\n1 0\n2 -1", "```\nResult: X = -3 2 / 1 0 / 2 -1\n```",
    "X -> -3 2 / 1 0 / 2 -1", "Answer: -3 2\n1 0\n2 -1", "So X = [[-3,2],[1,0],[2,-1]]", "=> -3 2 / 1 0 / 2 -1",
    "-> -3 2\n1 0\n2 -1", "Final: -3 2 / 1 0 / 2 -1", "X gives -3 2 / 1 0 / 2 -1", "X becomes -3 2 / 1 0 / 2 -1",
    "After row reduction the right block becomes -3 2 / 1 0 / 2 -1", "Final answer: X = -3 2\n1 0\n2 -1",
    "Result: X = np.array([[-3, 2], [1, 0], [2, -1]])", "Solution:\n-3 2\n1 0\n2 -1",
]


@pytest.mark.parametrize("text", X_PREFIX)
def test_p3_result_answer_and_arrow_prefixes_do_not_hide_a_matrix(text):
    s, c = make((SOLVE, SOLVE_KEY))
    assert kl(s, text, **GUIDE), text
    assert integrity.check_reply(s, 1, "How do I solve A X = B?", text, force_item=sticky(s)), text


X_FINE = [
    "Result: X = 5 4 / 3 7 / 2 8", "Answer: 5 4\n3 7\n2 8", "X -> 1 1 1 / 0 1 1 / 0 0 1",
    "The result is the matrix A with rows 1 1 1, 0 1 1 and 0 0 1.", "Result: the pivots are 1, 1 and 1.",
    "Answer: the system has one solution.",
]


@pytest.mark.parametrize("text", X_FINE)
def test_p3_a_prefix_in_front_of_another_matrix_or_a_given_passes(text):
    s, c = make((SOLVE, SOLVE_KEY))
    assert not kl(s, text, **GUIDE), text
    assert not kl(s, text, named=True), text


# ------------------------------------------------------------------ decision 8: significant figures and reference data
DIL_FINE = [
    "A 0.010 M solution has 0.010 mol of solute per liter, ten times weaker than 0.10 M.",
    "What is the difference between a 0.10 M and a 0.010 M solution?",
    "0.1 M versus 0.01 M", "A 0.010 M solution is ten times weaker than a 0.10 M one.", "0.0010 M is weaker still.",
]
DIL_LEAKS = ["cA = 0.0100 M", "The first dilution comes out to 0.0100 M.", "cA = 0.01 M", "Dilution B is 0.00100 M.",
             "A is 0.0100 M and B is 0.00100 M"]


@pytest.mark.parametrize("text", DIL_FINE)
def test_p2_a_coarser_value_never_matches_a_finer_key_by_rounding(text):
    s, c = make((DIL, DIL_KEY))
    assert not kl(s, text, **GUIDE), text
    assert not kl(s, text, named=True), text
    assert not integrity.check_card(s, 1, "Concentration", text)


@pytest.mark.parametrize("text", DIL_LEAKS)
def test_p2_the_exact_key_value_or_a_bound_name_still_leaks(text):
    s, c = make((DIL, DIL_KEY))
    assert kl(s, text, **GUIDE) or kl(s, text, named=True), text


def test_p2_the_number_of_figures_does_not_open_a_whole_number_or_a_precise_value():
    s, c = make((HYD, HYD_KEY))
    assert kl(s, "You need 12.48 g.", named=True)
    assert kl(s, "You need 12.5 g.", named=True)  # three figures, rounded the way a student would
    assert not kl(s, "You need about 12 g.", named=True)


def test_p2_a_molar_mass_the_materials_give_as_a_constant_is_reference_data():
    s, c = make((HYD, HYD_KEY))
    text = "The molar mass of CuSO4 5H2O is 249.68 g/mol."
    assert kl(s, text, **GUIDE)  # with no word from the materials the number is a key value
    assert not kl(s, text, data=["249.68"], **GUIDE)
    assert not kl(s, text, data=("249.68 g/mol",), named=True)
    assert kl(s, "m = 12.48 g, using 249.68 g/mol", data=["249.68"], **GUIDE)  # the answer next to it still leaks


def test_p6_angle_bracket_vector_that_names_x_is_a_hit_in_a_fresh_chat_too():
    s, c = make((LU, LU_KEY))
    assert kl(s, "x = <1, 1, 2>")
    assert kl(s, "x = ⟨1, 1, 2⟩")
    assert not kl(s, "v = <1, 3, 2>")
