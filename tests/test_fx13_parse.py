"""Cycle 13 parse fixes: decision 2 (key name formats), 3 (shapes of a finished answer), 4 (unit-aware laundering).
Each panel repro sits next to its honest look-alike. Fixtures are original (fx13_world)."""
import json

import pytest

import fx13_world as W
from course_assistant import integrity, mcp_server
from course_assistant.store import Store


@pytest.fixture
def w(tmp_path):
    x = W.world(tmp_path)
    mcp_server.set_store(x["s"])
    yield x
    mcp_server.set_store(None)


def item(x, label):
    return next(i for i in integrity.graded_view(x["s"], x["c"]) if i["label"] == label)


def leaks(x, label, reply, mode="sticky"):
    """Is the reply caught for the item? sticky: the chat touched it; named: the question names it."""
    it, s, c = item(x, label), x["s"], x["c"]
    if mode == "sticky":
        return bool(integrity.check_reply(s, c, "what next?", reply, force_item=dict(it, reason="sticky")))
    return bool(integrity.check_reply(s, c, f"how do I start {label}?", reply))


def save(x, body, chunk):
    return mcp_server.save_guide(x["c"], "explain", "Notes", body, [x["chunks"][chunk][0]])


# ------------------------------------------------------------ decision 4: the unit must agree with the cited chunk
LAUNDER = ["You need 132.6 g of pigment. [1]", "mass = 132.6 g [1]", "mass = 132.6 g. [1]", "about 132.6 g [1]",
           "The mass is 132.6 grams. [1]", "You need 11.27 mol of binder. [1]", "binder = 11.27 mol [1]"]


@pytest.mark.parametrize("body", LAUNDER)
def test_d4_a_pounds_weight_or_a_clock_time_does_not_license_grams_or_moles(w, body):
    assert save(w, body, "decoy").startswith("NOT SAVED"), body


@pytest.mark.parametrize("body", LAUNDER)
def test_d4_the_chunk_that_shows_the_same_quantity_still_licenses_it(w, body):
    assert save(w, body, "unit").startswith("Saved"), body


def test_d4_shown_in_units():
    pounds, grams = "The crate weighs 132.6 pounds.", "The batch needed 132.6 g of pigment."
    assert not integrity.shown_in("It is 132.6 g [1]", [pounds])
    assert integrity.shown_in("It is 132.6 g [1]", [grams])
    assert integrity.shown_in("It is 132.6 grams [1]", [grams])  # grams and g are one unit
    assert not integrity.shown_in("It is 0.0850 mol [1]", ["It starts at 0.0850 past nine."])
    assert integrity.shown_in("A 0.0850 mol/L solution [1]", ["A 0.0850 M solution."])  # mol/L and M are one unit
    assert integrity.shown_in("The molar mass is 58.44 g [1]", ["The molar mass is 58.44 g/mol."])
    # a unitless chunk number never licenses a value the reply gives with a unit; the other way round is the old rule
    assert not integrity.shown_in("It is 132.6 g [1]", ["The reading was 132.6."])
    assert integrity.shown_in("The reading was 132.6 [1]", [grams])


# ------------------------------------------------------------ decision 2: every common way a key names its values
KEYS = {
    "lambda1": ["lambda1 = 7", "lambda2 = 4"], "lambda 1": ["lambda 1 = 7", "lambda 2 = 4"], "greek": ["λ1 = 7", "λ2 = 4"],
    "greek sub": ["λ₁ = 7", "λ₂ = 4"], "eigenvalue 1": ["eigenvalue 1 = 7", "eigenvalue 2 = 4"],
    "l1": ["l1 = 7", "l2 = 4"], "e1": ["e1 = 7", "e2 = 4"], "x_1": ["x_1 = 7", "x_2 = 4"], "lam1": ["lam1 = 7", "lam2 = 4"],
    "bare": ["7", "4"], "csv": ["7, 4"], "eigenvalues colon": ["eigenvalues: 7 and 4"], "eigenvalues eq": ["eigenvalues = 7, 4"],
    "ordinal": ["first eigenvalue = 7", "second eigenvalue = 4"], "lambda eq": ["lambda = 7", "lambda = 4"],
}
STATED = ["The eigenvalues are 7 and 4.", "The roots are 4 and 7.", "eigenvalues: 7, 4", "lambda 1 = 7 and lambda 2 = 4",
          "The eigenvalues are seven and four.", "One eigenvalue is 4."]
UNLIKE = ["The eigenvalues of the lecture matrix are 10 and 5.", "An eigenvalue is a scalar lambda with Av = lambda v.",
          "The eigenvalues are 3 and 8.", "How many eigenvalues? A 2 by 2 matrix has 2."]


def keyed(tmp_path, key):
    s = Store(":memory:")
    c = s.add_course("C")
    s.add_graded_item(c, "Set 1: Problem 2", W.ROOTS, None, kind="problem", label="Problem 2", answer_key=json.dumps(key))
    return {"s": s, "c": c}


@pytest.mark.parametrize("name", list(KEYS))
def test_d2_the_plain_answer_is_a_hit_whatever_the_key_names_its_values(tmp_path, name):
    x = keyed(tmp_path, KEYS[name])
    for reply in STATED:
        for mode in ("sticky", "named"):
            assert leaks(x, "Problem 2", reply, mode), (name, reply, mode)
    for reply in UNLIKE:  # the honest look-alikes: other numbers, a definition
        for mode in ("sticky", "named"):
            assert not leaks(x, "Problem 2", reply, mode), (name, reply, mode)


def test_d2_key_names_are_read_in_one_shape():
    names = {"λ1 = 7": "lambda1", "lambda 1 = 7": "lambda1", "x_1 = 7": "x1", "eigenvalue 2 = 4": "eigenvalue2",
             "second eigenvalue = 4": "eigenvalue2", "L = [[1, 0], [2, 1]]": "l", "mass = 5.11 g": "mass"}
    for raw, nm in names.items():
        assert integrity._KeyVal(raw).name == nm, raw
    # the index belongs to the name: it is never read as the value
    assert integrity._KeyVal("lambda 1 = 7").nums == ["7"]


def test_d2_a_key_without_a_family_word_in_the_problem_is_left_alone(tmp_path):
    s = Store(":memory:")
    c = s.add_course("C")
    s.add_graded_item(c, "Set 1: Problem 9", "Problem 9 (5 points). How many marbles are in the jar after the swap?", None,
                      kind="problem", label="Problem 9", answer_key=json.dumps(["7", "4"]))
    x = {"s": s, "c": c}
    assert not leaks(x, "Problem 9", "The eigenvalues are 7 and 4.", "sticky")


# ------------------------------------------------------------ decision 3: the shapes of a finished answer
HIT_ROOTS = ["lambda^2 - 11 lambda + 28 = 0", "x^2 - 11x + 28 = 0", "(lambda - 7)(lambda - 4) = 0", "(λ-4)(λ-7)=0",
             "That gives $\\lambda^2 - 11\\lambda + 28 = 0$", "The characteristic equation is λ² − 11λ + 28 = 0.",
             "det(A - lambda I) = (6 - lambda)(5 - lambda) - 2, which expands to lambda^2 - 11 lambda + 28. That factors as (lambda - 7)(lambda - 4) = 0",
             "Q: What are the eigenvalues of A?\nA: 7 and 4"]
LIKE_ROOTS = ["lambda^2 - 15 lambda + 50 = 0", "(lambda - 10)(lambda - 5) = 0", "x^2 - 5x + 6 = 0",
              "The characteristic polynomial is lambda^2 - tr(A) lambda + det(A).",
              "Q: How many eigenvalues does a 2 by 2 matrix have?\nA: 2", "Q: What is an eigenvalue?\nA: A scalar lambda with Av = lambda v."]


@pytest.mark.parametrize("mode", ["sticky", "named"])
def test_d3_characteristic_polynomial_and_its_factors(w, mode):
    for r in HIT_ROOTS:
        assert leaks(w, "Problem 2", r, mode), r
    for r in LIKE_ROOTS:
        assert not leaks(w, "Problem 2", r, mode), r


@pytest.mark.parametrize("mode", ["sticky", "named"])
def test_d3_elimination_row_steps(w, mode):
    for r in ["Row 2 becomes (0, -7, -9) after subtracting 3 times row 1.", "Row 3 becomes (0, 14, 23) after subtracting 2 times row 1.",
              "R3 -> (0, 14, 23)", "Subtract 3 times row 1 from row 2, so the new row 2 is 0 -7 -9."]:
        assert leaks(w, "Problem 3", r, mode), r
    for r in ["Row 2 becomes (0, -5, -3) after subtracting 3 times row 1.", "The second row of the lecture matrix becomes (0, 7, 7).",
              "For row 2 take (6, 5, 9).", "Row 1 stays (2, 4, 6)."]:  # a lecture's numbers, a given row
        assert not leaks(w, "Problem 3", r, mode), r


@pytest.mark.parametrize("mode", ["sticky", "named"])
def test_d3_vectors_tables_arrows_and_question_answer_pairs(w, mode):
    for r in ["The vectors are (1,1) and (1,-2).", "| 7 | (1,1) |\n| 4 | (1,-2) |", "- 7 -> (1,1)\n- 4 -> (1,-2)",
              "7: (1, 1)\n4: (1, -2)", "Q: What are the eigenvectors?\nA: (1,1) and (1,-2)"]:
        assert leaks(w, "Problem 5", r, mode), r
    for r in ["The vectors are (3, 2) and (1, -4).", "| 4 | (2,3) |", "- 9 -> (2, 5)", "Q: What is an eigenvector?\nA: A nonzero vector v with Av = lambda v."]:
        assert not leaks(w, "Problem 5", r, mode), r


@pytest.mark.parametrize("mode", ["sticky", "named"])
def test_d3_a_multiplication_line_of_the_given_values(w, mode):
    for r in ["Compute 0.350 x 0.2500 x 58.44.", "Multiply 0.350 mol/L by 0.2500 L by 58.44 g/mol", "$0.350 \\times 0.2500 \\times 58.44$",
              "0.350 * 0.2500 * 58.44 [1]", "Compute 0.425 x 312.0 to get the mass."]:
        assert leaks(w, "Problem 4" if "0.350" in r else "Problem 1", r, mode), r
    for r in ["Compute 0.350 x 0.2500.", "Compute 0.410 x 0.3000 x 58.44.", "Multiply the rate by the coverage by the mass.",
              "The room is 3 by 3 and 4 x 5."]:
        assert not leaks(w, "Problem 4", r, mode), r
    assert not leaks(w, "Problem 1", "Compute 0.500 x 312.0 for a different glaze.", mode)


def test_d3_save_guide_refuses_the_shapes_and_keeps_the_lecture_case(w):
    for body in ["The polynomial is lambda^2 - 11 lambda + 28 = 0 [1]", "Row 2 becomes (0, -7, -9) after subtracting 3 times row 1 [1]",
                 "Compute 0.350 x 0.2500 x 58.44 [1]"]:
        assert save(w, body, "lecture-9").startswith("NOT SAVED"), body
    # the lecture's own worked case, cited to the chunk that prints it
    assert save(w, "The characteristic polynomial is lambda^2 - 15 lambda + 50 = 0, which factors as (lambda - 10)(lambda - 5) = 0 [1]",
                "lecture-9").startswith("Saved")


def test_d1_terse_label_values_are_hits_once_the_item_is_the_context(w):
    for r in ["mass = 132.6 g", "You need 132.6 g of pigment.", "132.6 g", "It works out to about 132.6 grams.", "Roughly 133 g.",
              "Roughly 132.6 g [9]."]:
        assert leaks(w, "Problem 1", r), r
    for r in ["u22 = -7", "l31 = 2", "u33 = 5", "l32 = -2"]:
        assert leaks(w, "Problem 3", r), r
    for r in ["Roughly 140 g.", "mass = 99.9 g", "It weighs 132.6 pounds.", "u22 = 4"]:
        assert not leaks(w, "Problem 1" if "mass" in r or "g." in r or "pounds" in r else "Problem 3", r), r


def test_d3_a_cited_lecture_paraphrase_with_a_looser_verb_saves(w):
    # the lecture prints "the eigenvalues are 10 and 5"; 'come out as' says the same thing
    assert save(w, "In the lecture, the eigenvalues come out as 10 and 5 [1].", "lecture-9").startswith("Saved")
    # the same verb for this item's own values, cited to a chunk that prints other numbers, is not licensed
    assert save(w, "The eigenvalues come out as 7 and 4 [1].", "lecture-9").startswith("NOT SAVED")
