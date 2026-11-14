"""Cycle 7 number, unit and matrix reading: one regression test per panel leak, each with its honest look-alike.
Every fixture is original text."""
import json

import pytest

from course_assistant import grounding, integrity, mcp_server
from course_assistant.providers import FakeProvider
from course_assistant.store import Store
from test_fx5_numeric import model

AUG = ("Problem 1 (15 points)\n\nRow reduce [A | B] where A has rows 3 1 4, 2 0 -1 and 1 5 2 and B has the columns "
       "(7, 3, -2) and (4, 0, 6). Report the solution columns x1 and x2.")
AUG_KEY = ["x1 = (6, -4, 9)", "x2 = (-5, 8, 3)"]
TRI = "Problem 2 (10 points)\n\nSolve the triangular system 2x + y + 3z = -6, 2y - z = 8, -z = 2 by back substitution."
TRI_KEY = ["z = -2", "y = 3", "x = -3/2"]
SWAP = ("Problem 3 (10 points)\n\nFind the value of d that forces a row exchange in the matrix with rows 1 2 3, 2 4 d "
        "and 1 0 1.")
SWAP_KEY = ["d = 11"]
LOWER = ("Problem 4 (10 points)\n\nFactor the 3 by 3 matrix with rows 1 2 0, 2 5 3 and 0 4 9 as LU and report L.")
LOWER_KEY = ["L = [[1,0,0],[2,1,0],[0,2,1]]"]
HYD = ("Problem 5 (10 points)\n\nFind the molar mass of copper(II) sulfate pentahydrate, then the mass needed to make "
       "500.0 mL of a 0.1000 M solution.")
HYD_KEY = ["M = 249.68 g/mol", "m = 12.48 g"]
MARK = "Report question 3 (5 points)\n\nAdd water to the 100.0 mL mark, then one more dropper, and record the volume."
DIL = ("Pre-lab question 2 (5 points)\n\nDilute 25.00 mL of 0.100 M stock to 250.0 mL (dilution A), then 25.00 mL of A to "
       "250.0 mL (dilution B). Give the molarity of each.")
DIL_KEY = ["cA = 0.0100 M", "cB = 0.00100 M"]
SALT = ("Problem 6 (5 points)\n\nHow many grams of NaCl (molar mass 58.44 g/mol) are in 100.0 mL of a 0.8750 M solution?")
SALT_KEY = ["m = 5.11 g"]


def make(*specs):
    s = Store(":memory:")
    c = s.add_course("Course")
    for body, key in specs:
        s.add_graded_item(c, body.split("\n")[0], body, None, kind="problem", label=body.split(" (")[0],
                          answer_key=json.dumps(key) if key else "")
    return s, c


def it(s, n):
    return integrity.graded_view(s, 1)[n - 1]


def leak(s, n, text, **ctx):
    return integrity.key_leak(text, it(s, n), {"named": True, **ctx})


# ------------------------------------------------------------------ p3: matrices in every common form
AUG_LEAKS = [
    "\\left[\\begin{array}{ccc|cc} 1&0&0&6&-5 \\\\ 0&1&0&-4&8 \\\\ 0&0&1&9&3 \\end{array}\\right]",
    "$$\\left[\\begin{array}{ccc|cc}\n1 & 0 & 0 & 6 & -5 \\\\\n0 & 1 & 0 & -4 & 8 \\\\\n0 & 0 & 1 & 9 & 3\n\\end{array}\\right]$$",
    "\\begin{array}{cc} 6 & -5 \\\\ -4 & 8 \\\\ 9 & 3 \\end{array}",
    "| 1 | 0 | 0 | 6 | -5 |\n|---|---|---|---|---|\n| 0 | 1 | 0 | -4 | 8 |\n| 0 | 0 | 1 | 9 | 3 |",
    "| x1 | x2 |\n|----|----|\n| 6 | -5 |\n| -4 | 8 |\n| 9 | 3 |",
    "6 -5 / -4 8 / 9 3",
    "X = 6 -5 / -4 8 / 9 3",
    "```\nX = 6 -5 / -4 8 / 9 3\n```",
    "```\n6 -5\n-4 8\n9 3\n```",
    "6 -5\n-4 8\n9 3",
    "The solutions side by side: [[6, -5],[-4, 8],[9, 3]] [1]",
    "Final [[1, 0, 0, 6, -5], [0, 1, 0, -4, 8], [0, 0, 1, 9, 3]] [1]",
    "[1] [[6, -5], [-4, 8], [9, 3]]",
    "[[6, -5], [-4, 8], [9, 3]][1]",
]


@pytest.mark.parametrize("text", AUG_LEAKS)
def test_p3_augmented_and_solution_matrices_are_leaks_in_every_form(text):
    s, c = make((AUG, AUG_KEY))
    assert leak(s, 1, text), text


AUG_FINE = [
    "\\left[\\begin{array}{ccc|cc} 1&0&0&3&1 \\\\ 0&1&0&2&7 \\\\ 0&0&1&5&4 \\end{array}\\right]",
    "| 1 | 0 | 0 | 3 | 1 |\n|---|---|---|---|---|\n| 0 | 1 | 0 | 2 | 7 |\n| 0 | 0 | 1 | 5 | 4 |",
    "| name | score |\n|---|---|\n| quiz | 12 |\n| exam | 40 |",
    "3 1 / 2 7 / 5 4",
    "```\n3 1\n2 7\n5 4\n```",
    "A different system: [[3, 1], [2, 7], [5, 4]] [1]",
    "The identity block [[1,0,0],[0,1,0],[0,0,1]] stays on the left [1].",
    "Gauss-Jordan ends with the identity on the left, as in [1] and [2].",
]


@pytest.mark.parametrize("text", AUG_FINE)
def test_p3_other_matrices_in_the_same_forms_are_fine(text):
    s, c = make((AUG, AUG_KEY))
    assert leak(s, 1, text) == [], text


TRI_BAR = ["\\left[\\begin{array}{ccc|c} 2 & 1 & 3 & -6 \\\\ 0 & 2 & -1 & 8 \\\\ 0 & 0 & -1 & 2 \\end{array}\\right]",
           "| 2 | 1 | 3 | -6 |\n|---|---|---|---|\n| 0 | 2 | -1 | 8 |\n| 0 | 0 | -1 | 2 |",
           "2 1 3 -6 / 0 2 -1 8 / 0 0 -1 2",
           "[2 1 3 | -6; 0 2 -1 | 8; 0 0 -1 | 2] [1]"]


@pytest.mark.parametrize("text", TRI_BAR)
def test_p3_triangular_system_in_bar_form_is_a_leak(text):
    s, c = make((TRI, TRI_KEY))
    assert leak(s, 1, text), text


def test_p3_triangular_look_alike_with_other_numbers_is_fine():
    s, c = make((TRI, TRI_KEY))
    assert leak(s, 1, "\\left[\\begin{array}{ccc|c} 2 & 1 & 3 & 1 \\\\ 0 & 2 & 1 & 5 \\\\ 0 & 0 & 1 & 2 \\end{array}\\right]") == []


def test_p3_a_citation_marker_does_not_hide_a_matrix_but_a_real_column_still_reads():
    h = integrity._Hay("[[6, -5], [-4, 8], [9, 3]] [1]")
    assert any(seq == ["6", "-4", "9"] for _, seq in h.seqs)
    col = integrity._Hay("[[1],[2],[3]]")  # a column vector keeps its single-entry rows
    assert any(seq == ["1", "2", "3"] for _, seq in col.seqs)


def test_p3_matrix_rows_reads_arrays_with_a_bar():
    rows = integrity.matrix_rows("\\begin{array}{cc|c} 3 & 1 & 4 \\\\ 2 & 0 & -1 \\end{array}")
    assert rows == [["3", "1", "4"], ["2", "0", "-1"]]


def test_p3_matrix_leaks_are_refused_by_the_mcp_tools():
    s, c = make((AUG, AUG_KEY))
    mcp_server.set_store(s)
    try:
        for body in ("The solutions side by side: [[6, -5],[-4, 8],[9, 3]] [1]",
                     "| 6 | -5 |\n|---|---|\n| -4 | 8 |\n| 9 | 3 |", "Problem 1 ends with 6 -5 / -4 8 / 9 3"):
            assert mcp_server.save_guide(c, "answer", "Notes", body, []).startswith("NOT SAVED"), body
        assert mcp_server.add_flashcards(c, [{"front": "Problem 1: X?", "back": "[[6, -5],[-4, 8],[9, 3]] [1]"}]).startswith("NOT ADDED")
        assert mcp_server.save_guide(c, "answer", "Notes", "Gauss-Jordan leaves [I | X]; the right block is X [1].", []).startswith("Saved")
    finally:
        mcp_server.set_store(None)


# ------------------------------------------------------------------ p3: small whole numbers that are not values
@pytest.mark.parametrize("text", [
    "Problem Set 2 is due Thursday, Sep 17, 2026 at 11:59 pm and is worth 60 points.",
    "Problem sets are due Thursdays at 11:59 pm.", "Lectures 10 and 11 cover inverses.", "Lecture 11 covers inverses.",
    "Office hours are 10 to 11 am.", "Office hours are 10-11 am on weeks 10 and 11.",
    "The quiz is worth 10 points and the exam 11 points.", "Read chapter 11, pages 10 to 11, before the 11th.",
    "1. Read chapter 3\n11. Check the syllabus", "Meet at 11 am.",
])
def test_p3_clock_times_dates_and_lecture_numbers_are_not_key_values(text):
    s, c = make((SWAP, SWAP_KEY))
    assert leak(s, 1, text, sticky=True, short=True, question_refs=True) == [], text
    assert leak(s, 1, text, small=True) == [], text


@pytest.mark.parametrize("text", ["d = 11", "Then d is eleven.", "The matrix needs a row exchange when d = 11 at 11:59 pm."])
def test_p3_the_real_value_still_counts_next_to_those_numbers(text):
    s, c = make((SWAP, SWAP_KEY))
    assert leak(s, 1, text, sticky=True), text


def test_p3_a_twin_column_of_zeros_ones_and_twos_is_not_a_hit_but_a_distinctive_row_is():
    s, c = make((LOWER, LOWER_KEY))
    # column 2 of this new matrix is (2, 1, 0), which equals row 2 of L
    assert leak(s, 1, "A new system A has rows (1,2,0), (0,1,1), (1,0,1).") == []
    assert leak(s, 1, "A new system with rows (3,2,0), (0,1,1), (1,0,1).") == []
    assert leak(s, 1, "L = [[1,0,0],[2,1,0],[0,2,1]]")  # the whole matrix is still a leak
    assert leak(s, 1, "L has rows (1,0,0), (2,1,0), (0,2,1)")


# ------------------------------------------------------------------ p2: precision, constants, percent contexts, units
@pytest.mark.parametrize("t,k,hit", [
    ("250", "249.68", False), ("2.5", "2.497", False), ("250.0", "249.68", False), ("249.7", "249.68", True),
    ("249.68", "249.68", True), ("2.50", "2.497", True), ("5.1", "5.11", False), ("5.110", "5.11", True),
    ("-1.5", "-3/2", True), ("5", "5.0", True), ("0.12", "0.1234", False), ("0.123", "0.1234", True),
])
def test_p2_a_reply_number_matches_only_at_three_figures_or_exactly(t, k, hit):
    assert integrity.num_eq(t, k) is hit


def test_p2_a_bare_250_does_not_match_a_molar_mass_key_but_249_7_does():
    s, c = make((HYD, HYD_KEY))
    for q in ("What is 250 mL in liters?", "How do I prepare 250 mL of solution in a volumetric flask?", "That is 2.5 g."):
        assert integrity.key_leak(q, it(s, 1), {"lone": True}) == [], q
    assert integrity.key_leak("The molar mass is 249.7 g/mol.", it(s, 1), {"lone": True})


def test_p2_percent_and_sample_hundreds_are_not_values():
    s, c = make((MARK, None), (DIL, DIL_KEY), (SALT, SALT_KEY))
    for front, back in [("What is percent yield?", "Actual over theoretical, times 100%"),
                        ("What is mass percent?", "(solute / solution) x 100%"),
                        ("First step of an empirical formula?", "Assume a 100 g sample"),
                        ("Why can a percent yield be below 100%?", "Side reactions use some of the reactant."),
                        ("Percent composition formula", "mass of element / total mass x 100")]:
        assert integrity.check_card(s, 1, front, back) is None, (front, back)
    assert integrity.check_guide(s, 1, "Notes", "Percent composition is mass of element over total mass, times 100.") is None
    assert integrity.numbers("a 100 g sample, x 100 and 100%") == []
    assert integrity.numbers("the 100.0 mL mark") == ["100"]


def test_p2_the_given_hundred_still_counts_when_it_is_the_problem():
    s, c = make((MARK, None))
    assert integrity.check_card(s, 1, "Report question 3", "Add water to the 100.0 mL mark, then one more dropper")


def test_p2_a_given_constant_is_not_a_key_value():
    s, c = make((SALT, SALT_KEY + ["M = 58.44 g/mol"]))
    assert leak(s, 1, "NaCl has a molar mass of 58.44 g/mol.") == []
    assert leak(s, 1, "Use R = 8.314 and pi = 3.14159.") == []
    s2, c2 = make((HYD, ["R = 8.314 J/mol K", "m = 12.48 g"]))
    assert leak(s2, 1, "The gas constant is 8.314.") == []


def test_p2_latex_unit_wrappers_are_stripped():
    s, c = make((DIL, DIL_KEY))
    for text in ("Dilution A: $(0.100)(25.00)/250.0 = 0.0100\\ M$. Dilution B: $(0.0100)(25.00)/250.0 = 0.00100\\ M$",
                 "A is $0.0100\\,\\text{M}$ and B is $0.00100\\,\\text{M}$.", "$c_A = 0.0100~\\mathrm{M}$, $c_B = 0.00100~\\mathrm{M}$",
                 "A: $0.0100\\ \\text{M}$, B: $0.00100\\ \\text{M}$"):
        assert leak(s, 1, text), text
        assert integrity.scrub_parts(s, 1, text, "serial dilution")[1] == 1, text
    honest = "Dilution A: $(0.200)(10.00)/100.0 = 0.0200\\ M$. Dilution B: $(0.0200)(10.00)/100.0 = 0.00200\\ M$"
    assert leak(s, 1, honest) == []
    assert integrity.scrub_parts(s, 1, honest, "serial dilution")[1] == 0
    assert integrity._norm_hay("$5.11\\,\\text{g}$").split() == ["5.11", "g"]


# ------------------------------------------------------------------ p6: fractions, spelled numbers, units, bound names
@pytest.mark.parametrize("text", ["x = -3/2.", "y = 3, x = -3/2.", "So x = -3/2\nand done", "x = -3/2 ", "x = (-3/2)", "x = -3/2;",
                                  "x is minus three halves", "x equals negative three halves.", "x = -1.5.", "x = -1.5"])
def test_p6_sentence_final_and_spelled_fractions_read_as_values(text):
    s, c = make((TRI, ["x = -3/2"]))
    assert leak(s, 1, text), text
    assert integrity.numbers("x = -3/2.") == ["-3/2"]


def test_p6_minus_three_halves_look_alikes_are_fine():
    s, c = make((TRI, ["x = -3/2"]))
    assert leak(s, 1, "x = 3/2.") == []
    assert leak(s, 1, "x is three halves") == []
    assert leak(s, 1, "x = -1/2.") == []
    assert integrity.numbers("two thirds and one half") == ["2/3", "1/2"]


def test_p6_three_bound_key_entries_are_a_leak_even_when_small():
    s, c = make((TRI, TRI_KEY))
    full = "Back substitution gives z = -2, then y = 3, then x = -3/2."
    assert leak(s, 1, full)
    assert leak(s, 1, "z = -2, y = 3, x = -3/2.", named=False, small=False)  # not named: three bindings alone decide
    assert leak(s, 1, "z = -2 and y = 3.", named=False, small=False) == []  # two small ones in text not about the item
    assert leak(s, 1, "For a different system z = 4, y = 1, x = 5.", named=False, small=False) == []


def test_p6_spelled_units_convert_like_symbols():
    s, c = make((SALT, SALT_KEY))
    for text in ("That is 5110 milligrams of NaCl.", "That is 5110 mg of NaCl.", "It is 0.00511 kilograms of NaCl.", "5.11 grams of NaCl"):
        assert leak(s, 1, text), text
    for text in ("That is 5000 milligrams of NaCl.", "That is 5110 milliliters of water."):
        assert leak(s, 1, text) == [], text
    mcp_server.set_store(s)
    try:
        assert mcp_server.save_guide(1, "answer", "Notes", "That is 5110 milligrams of NaCl", []).startswith("NOT SAVED")
        assert mcp_server.save_guide(1, "answer", "Notes", "A pinch is about 500 milligrams of salt.", []).startswith("Saved")
    finally:
        mcp_server.set_store(None)


# ------------------------------------------------------------------ the same leaks and look-alikes through Ask
def test_ask_normally_catches_the_array_with_a_bar_and_a_trailing_citation_but_not_a_different_matrix():
    s, c = make((AUG, AUG_KEY))
    q = "What does the augmented matrix look like at the end of elimination in problem 1?"
    for leaky in ("It ends as \\left[\\begin{array}{ccc|cc} 1&0&0&6&-5 \\\\ 0&1&0&-4&8 \\\\ 0&0&1&9&3 \\end{array}\\right].",
                  "The solutions are [[6, -5], [-4, 8], [9, 3]] [1]."):
        a = grounding.ask(s, c, q, FakeProvider(model(default=leaky)), route=False)
        assert a.mode in ("twin", "concepts") and "6&-5" not in a.body_md and "[[6, -5]" not in a.body_md
    fine = "A different example ends as \\left[\\begin{array}{cc|c} 1&0&3 \\\\ 0&1&2 \\end{array}\\right]."
    assert grounding.ask(s, c, q, FakeProvider(model(default=fine)), route=False).mode == "answer"


def test_ask_a_due_date_in_a_touched_chat_is_not_refused_but_the_value_is():
    s, c = make((SWAP, SWAP_KEY))
    p = FakeProvider(model({"due": "Problem Set 2 is due Thursday, Sep 17, 2026 at 11:59 pm and is worth 60 points.",
                            "which d": "d = 11"}))
    grounding.ask(s, c, "Problem 3: " + SWAP.split("\n\n")[1], p)
    a = grounding.ask(s, c, "When is Problem Set 2 due and how many points is it worth?", p)
    assert a.mode == "answer" and "11:59" in a.body_md
    assert grounding.ask(s, c, "which d forces the swap?", p).mode in ("twin", "concepts")


def test_ask_a_sentence_final_fraction_is_caught():
    s, c = make((TRI, TRI_KEY))
    p = FakeProvider(model({"Problem 2": "Back substitution gives z = -2, then y = 3, and finally x = -3/2."}))
    a = grounding.ask(s, c, "Problem 2: " + TRI.split("\n\n")[1], p)
    assert a.mode in ("twin", "concepts") and "-3/2" not in a.body_md


# ------------------------------------------------------------------ p3: restating the problem's own data is not a leak
GIVEN_LU = ("Problem 7 (10 points)\n\nFactor A with rows (4, 2, 6), (8, 7, 19) and (-4, -6, -4) as LU.")
GIVEN_LU_KEY = ["L = [[1,0,0],[2,1,0],[-1,-2,1]]", "U = [[4,2,6],[0,3,7],[0,0,5]]"]


def test_p3_restating_the_givens_is_not_a_leak_but_the_factors_are():
    s, c = make((GIVEN_LU, GIVEN_LU_KEY))
    # row 1 of A is also row 1 of U
    assert leak(s, 1, "Problem 7 gives A with rows (4, 2, 6), (8, 7, 19) and (-4, -6, -4); factor it as LU.") == []
    assert leak(s, 1, "Problem 7 gives A with rows 4 2 6, 8 7 19 and -4 -6 -4.") == []
    assert leak(s, 1, "U = [[4,2,6],[0,3,7],[0,0,5]]")
    assert leak(s, 1, "L has rows (1,0,0), (2,1,0), (-1,-2,1)")
    mcp_server.set_store(s)
    try:
        assert mcp_server.save_guide(1, "answer", "Notes", "Problem 7: U has rows (4,2,6), (0,3,7), (0,0,5).", []).startswith("NOT SAVED")
    finally:
        mcp_server.set_store(None)


def test_p6_spelled_units_also_match_without_a_key_and_a_deadline_guide_saves():
    s, c = make((SALT, None), (SWAP, SWAP_KEY))
    assert integrity.check_guide(s, 1, "Notes", "That is 5110 milligrams of NaCl.")
    assert integrity.check_guide(s, 1, "Notes", "A pinch is 500 milligrams of salt.") is None
    mcp_server.set_store(s)
    try:
        assert mcp_server.save_guide(1, "answer", "Deadlines", "Problem Set 2 is due Thursday, Sep 17, 2026 at 11:59 pm and is "
                                     "worth 60 points. Lectures 10 and 11 are on inverses; office hours are 10 to 11 am.", []).startswith("Saved")
        assert mcp_server.save_guide(1, "answer", "Notes", "For Problem 3 the answer is d = 11.", []).startswith("NOT SAVED")
    finally:
        mcp_server.set_store(None)
