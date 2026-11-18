"""Hotfix pass 2 (numeric): bracketed vectors are not citation markers, long name prefixes, plain-row matrices in the MCP
tools, triangular systems, cited proofs while sticky and two over-blocks. Each leak has its honest look-alike beside it."""
import json

import pytest

from course_assistant import integrity, mcp_server
from test_fx7_route import make

GENERIC = "Can you explain Gaussian elimination and triangular systems with an example?"

LU7 = ("Problem 7\n\nLet A be the 3 by 3 matrix with rows (2, 4, 1), (4, 3, 4), (-2, 1, 0), and let b = (8, 10, 4).\n\n"
       "(a) Find the factorization A = LU. (b) Solve Ax = b with forward and back substitution.")
EL8 = ("Problem 8\n\nConsider elimination on the system\n\n2x + y + z = 8\n4x + dy + 3z = 21\n3y + 4z = 23\n\n"
       "(a) Which d forces a row exchange, and what triangular system results for that d?")
KEY7 = ["L = [[1,0,0],[2,1,0],[-1,-1,1]]", "U = [[2,4,1],[0,-5,2],[0,0,3]]", "y = (8, -6, 6)", "x = (-1, 2, 2)"]
KEY8 = ["d = 2", "u11 = 2", "u12 = 1", "u13 = 1", "c1 = 8", "u22 = 3", "u23 = 4", "c2 = 23", "u33 = 1", "c3 = 5",
        "x = 1", "y = 1", "z = 5"]


def world(extra=False, p3=None, p2=None):
    s, c, m, ids = make()
    ps2 = m["problem-set-2"]
    if p3 is not None:
        s.update_graded_item(ids[(ps2, "Problem 3")], answer_key=json.dumps(p3))
    if p2 is not None:
        s.update_graded_item(ids[(ps2, "Problem 2")], answer_key=json.dumps(p2))
    if extra:
        s.add_graded_item(c, "Problem 7", LU7, None, kind="problem", label="Problem 7", answer_key=json.dumps(KEY7))
        s.add_graded_item(c, "Problem 8", EL8, None, kind="problem", label="Problem 8", answer_key=json.dumps(KEY8))
    return s, c, m, ids


def reply_hit(s, c, text, q=GENERIC, **kw):
    return integrity.check_reply(s, c, q, text, **kw)


def mcp_refused(s, c, text):
    mcp_server.set_store(s)
    try:
        g = mcp_server.save_guide(c, "answer", "Notes", text, []).startswith("NOT SAVED")
        f = mcp_server.add_flashcards(c, [{"front": "Result", "back": text}]).startswith("NOT ADDED")
        return g, f
    finally:
        mcp_server.set_store(None)


# ------------------------------------------------------------------ 1. a bracketed vector is not a citation marker
@pytest.mark.parametrize("text", ["x = [1, 1, 2]", "The solution is x = [1, 1, 2].", "x = [1;1;2]", "x = [1, 1, 2] [1]",
                                  "x = [1, 1, 2] and see [1, 2]"])
def test_bracketed_key_vector_is_caught_everywhere(text):
    s, c, m, ids = world(p3=["x = (1, 1, 2)"])
    assert reply_hit(s, c, text)["label"] == "Problem 3"
    assert reply_hit(s, c, text, sources=[{"text": "a"}, {"text": "b"}])["label"] == "Problem 3"
    assert mcp_refused(s, c, text) == (True, True)


def test_second_problems_bracketed_vector_is_caught():
    s, c, m, ids = world()
    assert reply_hit(s, c, "x2 = [3, 0, 1]")["label"] == "Problem 2"


def test_what_is_x_again_in_a_touched_chat_catches_the_bracketed_vector():
    s, c, m, ids = world(p3=["x = (1, 1, 2)"])
    sticky = dict(next(i for i in integrity.graded_view(s, c) if i["label"] == "Problem 3"), reason="sticky")
    assert reply_hit(s, c, "x = [1, 1, 2]", "what is x again?", force_item=sticky)
    assert reply_hit(s, c, "The solution vector is [1, 1, 2].", "what is x again?", force_item=sticky)


@pytest.mark.parametrize("text", ["Elimination is introduced in the notes [1].", "Both lectures cover the method [1, 2].",
                                  "See the notes [1] and the examples [2] for triangular systems."])
def test_real_citation_markers_still_strip_and_never_leak(text):
    s, c, m, ids = world(p3=["x = (1, 1, 2)"])
    sticky = dict(next(i for i in integrity.graded_view(s, c) if i["label"] == "Problem 3"), reason="sticky")
    src = [{"text": "a"}, {"text": "b"}]
    assert reply_hit(s, c, text, "Which lectures cover elimination?", force_item=sticky, sources=src) is None
    assert mcp_refused(s, c, text) == (False, False)


def test_marker_strip_rules():
    strip = integrity._CITE_STRIP.sub
    assert strip("", "a [1] b [1, 2] c [3-4]") == "a  b  c "
    assert strip("", "x = [1, 2]") == "x = [1, 2]"
    assert strip("", "x: [1]") == "x: [1]"
    assert strip("", "see [1, 1, 2]") == "see [1, 1, 2]"
    assert strip("", "see [3]", 2) == "see [3]"  # not a source of this reply
    assert strip("", "see [2]", 2) == "see "


# ------------------------------------------------------------------ 2. long name prefixes and the colon form
XMAT = ["X = [[-2, 3], [1, 0], [-1, 1]]"]
PREFIXED = ["result = -2 3\n1 0\n-1 1", "```\nanswer = -2 3\n         1 0\n        -1 1\n```", "solution = -2 3 / 1 0 / -1 1",
            "```\nX_sol = -2 3\n        1 0\n       -1 1\n```", "X: -2 3 / 1 0 / -1 1", "X: -2 3\n1 0\n-1 1",
            "```\nX: -2 3\n   1 0\n  -1 1\n```", "solution = [-2 3\n 1 0\n -1 1]"]


@pytest.mark.parametrize("text", PREFIXED)
def test_long_name_and_colon_prefixed_matrix_is_caught(text):
    s, c, m, ids = world(p2=XMAT)
    assert reply_hit(s, c, text)["label"] == "Problem 2"
    assert mcp_refused(s, c, text) == (True, True)


@pytest.mark.parametrize("text", ["result = 4 7\n2 9\n5 8", "The matrix X: 4 7 / 2 9 / 5 8", "step: 3 1\n2 5\n7 7"])
def test_prefixed_matrix_with_other_numbers_passes(text):
    s, c, m, ids = world(p2=XMAT)
    assert reply_hit(s, c, text) is None
    assert mcp_refused(s, c, text) == (False, False)


# ------------------------------------------------------------------ 3. plain rows after a name, in the MCP tools too
@pytest.mark.parametrize("text", ["U = 2 4 1 / 0 -5 2 / 0 0 3", "U = 2 4 1\n0 -5 2\n0 0 3", "U = [2 4 1; 0 -5 2; 0 0 3]",
                                  "```\nU = 2 4 1\n    0 -5 2\n    0 0 3\n```"])
def test_u_with_a_restated_first_row_is_caught_in_ask_and_mcp(text):
    s, c, m, ids = world(extra=True)
    assert reply_hit(s, c, text)["label"] == "Problem 7"
    assert mcp_refused(s, c, text) == (True, True)


@pytest.mark.parametrize("text", ["A = 2 4 1 / 4 3 4 / -2 1 0", "Problem 7 gives A with rows (2,4,1), (4,3,4), (-2,1,0) and asks for LU",
                                  "U = 2 4 1 / 0 3 6 / 0 0 9"])
def test_restating_the_given_matrix_or_other_numbers_is_fine(text):
    s, c, m, ids = world(extra=True)
    assert reply_hit(s, c, text) is None
    assert mcp_refused(s, c, text) == (False, False)


# ------------------------------------------------------------------ 4. a triangular system for an elimination problem
SYSTEMS = ["2 1 1 | 8 / 0 3 4 | 23 / 0 0 1 | 5", "T = 2 1 1 8 / 0 3 4 23 / 0 0 1 5", "0 0 1 | 5", "The last row is 0 0 1 | 5",
           "[[2,1,1,8],[0,3,4,23],[0,0,1,5]]", "2 1 1 8\n0 3 4 23\n0 0 1 5", "T: 2 1 1 | 8 / 0 3 4 | 23 / 0 0 1 | 5",
           "\\left[\\begin{array}{ccc|c} 2 & 1 & 1 & 8 \\\\ 0 & 3 & 4 & 23 \\\\ 0 & 0 & 1 & 5 \\end{array}\\right]",
           "```\n[ 2 1 1 | 8 ]\n[ 0 3 4 | 23 ]\n[ 0 0 1 | 5 ]\n```"]


@pytest.mark.parametrize("text", SYSTEMS)
def test_triangular_system_rows_are_caught_in_ask_and_mcp(text):
    s, c, m, ids = world(extra=True)
    assert reply_hit(s, c, text)["label"] == "Problem 8"
    assert mcp_refused(s, c, text) == (True, True)


@pytest.mark.parametrize("text", ["2 1 1 | 8 / 0 3 4 | 23", "0 3 4 | 23", "The bottom row is 0 0 1 | 7", "0 0 1 | 6 / 0 2 | 4"])
def test_the_systems_own_given_rows_or_other_rows_pass(text):
    s, c, m, ids = world(extra=True)
    assert reply_hit(s, c, text) is None
    assert mcp_refused(s, c, text) == (False, False)


# ------------------------------------------------------------------ 5. a cited finished proof while sticky
PROOF = ("Suppose a system of linear equations Ax = b has two solutions x and y with y different from x [1]. "
         "Then A(x + t(y - x)) = b for every number t, because A x = b and A y = b [1]. "
         "Since y differs from x, each t gives another solution, so there are more solutions than two. "
         "Hence the number of solutions of Ax = b is never exactly two, and this is true for every system. "
         "Therefore the number of solutions is never exactly two [1].")


def test_cited_finished_proof_of_a_graded_proof_item_is_caught_while_sticky():
    s, c, m, ids = world()
    p4 = dict(next(i for i in integrity.graded_view(s, c) if i["label"] == "Problem 4"), reason="sticky")
    src = [{"text": "A linear system has zero, one or infinitely many solutions, never exactly 2."}]
    assert reply_hit(s, c, PROOF, "Prove that a system has zero, one or infinitely many solutions.", force_item=p4,
                     sources=src)


def test_cited_numeric_lecture_example_is_still_quoted_while_sticky():
    s, c, m, ids = world()
    p1 = dict(next(i for i in integrity.graded_view(s, c) if i["label"] == "Problem 1"), reason="sticky")
    src = [{"text": "Worked example: 5x + 7y = 53 and 2x + y = 12 give x = 3 and y = 5 by elimination."}]
    text = "In the lecture example, 5x + 7y = 53 and 2x + y = 12 give x = 3 and y = 5 [1]."
    assert reply_hit(s, c, "Show me a worked elimination example", force_item=p1, sources=src) is None, text


# ------------------------------------------------------------------ 6. over-blocks
A2_TEXTS = ["Problem 2 gives A with rows (1,6,-3), (-2,3,4), (1,0,-2) and asks you to eliminate",
            "A = 1 6 -3 / -2 3 4 / 1 0 -2", "```\nA = 1 6 -3\n    -2 3 4\n     1 0 -2\n```"]


@pytest.mark.parametrize("text", A2_TEXTS)
def test_restating_problem_2s_matrix_is_not_a_leak_of_problem_3(text):
    s, c, m, ids = world(p3=["y = (-3, 4, -2)"])  # a key vector that happens to be a column of Problem 2's matrix
    assert reply_hit(s, c, text) is None
    assert mcp_refused(s, c, text) == (False, False)


def test_stating_problem_3s_own_key_vector_is_still_a_leak():
    s, c, m, ids = world(p3=["y = (-3, 4, -2)"])
    assert reply_hit(s, c, "y = (-3, 4, -2)")["label"] == "Problem 3"
    assert mcp_refused(s, c, "y = (-3, 4, -2)") == (True, True)


@pytest.mark.parametrize("text", ["A has rows (1,0,2), (0,3,1), (2,1,0) and b = (1,2,3)",
                                  "Practice: A has rows (2,1,0), (1,2,1), (0,1,2) and b = (1,0,1)"])
def test_a_twin_with_small_integers_is_not_a_leak_of_a_keyed_item(text):
    s, c, m, ids = world(extra=True)
    assert reply_hit(s, c, text) is None
    assert mcp_refused(s, c, text) == (False, False)


def test_a_key_matrix_with_two_of_its_rows_still_counts():
    s, c, m, ids = world(extra=True)
    assert reply_hit(s, c, "rows (0, -5, 2) and (0, 0, 3)")["label"] == "Problem 7"
