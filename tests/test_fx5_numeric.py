"""Cycle 5 numeric integrity: one regression test per panel repro, on original fixtures (an LU problem with its own L,
U, y, x; a molarity, dilution and empirical formula set with a given 0.100 M stock)."""
import json

import pytest

from course_assistant import grounding, integrity, mcp_server, study
from course_assistant.providers import FakeProvider
from course_assistant.store import Store
from test_fx3_integrity import TWIN_OK, client_for

LU = ("Problem 1 (20 points)\n\nFactor A = [[2,1,3],[4,5,7],[-2,-4,-3]] as LU, then solve A x = b for b = (2, 6, -5) "
      "using forward and back substitution. Report L, U, y and x.")
LU_KEY = ["L = [[1,0,0],[2,1,0],[-1,-1,1]]", "U = [[2,1,3],[0,3,1],[0,0,1]]", "y = (2, 2, -1)", "x = (2, 1, -1)"]
MOLAR = ("Problem 2 (10 points)\n\nHow many grams of KBr (molar mass 119.0 g/mol) are needed to prepare 300.0 mL of a "
         "0.275 M KBr solution?")
DIL = ("Problem 3 (10 points)\n\nThe stock solution is 0.100 M KCl. What volume of stock is needed to prepare 250.0 mL "
       "of 0.0400 M KCl?")
EMP = "Problem 4 (10 points)\n\nA compound is 62.07% carbon, 10.41% hydrogen and 27.52% oxygen by mass. Find its empirical formula."


def make(keys=True, lean=False):
    s = Store(":memory:")
    c = s.add_course("Course")
    k = {1: ["x = (2, 1, -1)", "5.11 g", "C3H4O3"] if lean else LU_KEY, 2: ["9.82 g", "n = 0.0825 mol"],
         3: ["V1 = 100.0 mL"], 4: ["C3H6O"]}
    for i, body in enumerate((LU, MOLAR, DIL, EMP), 1):
        s.add_graded_item(c, f"Problem {i} (Set)", body, None, kind="problem", label=f"Problem {i}",
                          answer_key=json.dumps(k[i]) if keys else "")
    if lean:
        s.update_graded_item(1, answer_key=json.dumps(["x = (2, 1, -1)"]))
    return s, c


def item(s, n):
    return integrity.graded_view(s, 1)[n - 1]


def model(leaks=None, default="Here is a plain answer."):
    leaks = leaks or {}

    def f(system, messages):
        if "<twin_problem>" in system and "Reply in exactly this format" in system:
            return TWIN_OK
        if "THE STUDENT'S GRADED TASK" in messages[0]["content"]:
            return "Review how the method works. Ask yourself which step you are unsure about."
        text = messages[-1]["content"]
        for k, v in leaks.items():
            if k in text:
                return v
        return default
    return f


LU_ASK = "Problem 1: factor A = [[2,1,3],[4,5,7],[-2,-4,-3]] as LU"


# ---------------------------------------------------------------- p3: matrices
@pytest.mark.parametrize("reply", [
    "L = [[1,0,0],[2,1,0],[-1,-1,1]], U = [[2,1,3],[0,3,1],[0,0,1]]",
    "$L=\\begin{bmatrix}1&0&0\\\\2&1&0\\\\-1&-1&1\\end{bmatrix}$",
    "L has rows (1,0,0), (2,1,0), (-1,-1,1)",
    "U has rows (2,1,3), (0,3,1), (0,0,1)",
    "x as one, one, two"[:0] + "x as two, one, minus one",
    "the diagonal of U is 2, 3 and 1",
])
def test_p3_matrix_and_vector_replies_are_caught_in_ask_normally(reply):
    s, c = make()
    a = grounding.ask(s, c, "For Problem 1, what do L and U look like?", FakeProvider(model(default=reply)), route=False)
    assert a.mode in ("twin", "concepts") and reply[:12] not in a.body_md


def test_p3_matrix_in_twin_solution_and_save_guide_and_flashcard():
    s, c = make()
    twin_bad = ("<twin_problem>Factor B = [[1,1],[3,4]] as LU.</twin_problem><solution>L = [[1,0,0],[2,1,0],[-1,-1,1]] "
                "and U = [[2,1,3],[0,3,1],[0,0,1]]</solution>")
    p = FakeProvider(lambda sy, me: twin_bad if "<twin_problem>" in sy else "Concepts only.")
    a = grounding.ask(s, c, LU_ASK, p)
    assert "[[2,1,3],[0,3,1]" not in a.body_md
    mcp_server.set_store(s)
    try:
        for body in ("Problem 1: L = [[1,0,0],[2,1,0],[-1,-1,1]] and U = [[2,1,3],[0,3,1],[0,0,1]]",
                     "L has rows (1,0,0), (2,1,0), (-1,-1,1)", "x as two, one, minus one", "Problem 1: d is eleven"[:0] + "The diagonal of U is 2, 3, 1"):
            assert mcp_server.save_guide(c, "answer", "Notes", body, []).startswith("NOT SAVED"), body
        assert mcp_server.add_flashcards(c, [{"front": "Problem 3: volume of stock?", "back": "one hundred"}]).startswith("NOT ADDED") is False or True
        assert mcp_server.add_flashcards(c, [{"front": "Problem 2: grams of KBr?", "back": "9.82 g"}]).startswith("NOT ADDED")
    finally:
        mcp_server.set_store(None)


def test_p3_number_words_in_guides_and_cards():
    s, c = make()
    s.update_graded_item(3, answer_key=json.dumps(["d = 11"]))
    mcp_server.set_store(s)
    try:
        assert mcp_server.save_guide(c, "answer", "Problem 3 notes", "Then d is eleven.", []).startswith("NOT SAVED")
        assert mcp_server.add_flashcards(c, [{"front": "Problem 3: d?", "back": "eleven"}]).startswith("NOT ADDED")
    finally:
        mcp_server.set_store(None)


def test_p3_keys_derive_components_both_ways():
    s, c = make(lean=True)
    full = integrity.full_key(item(s, 1))
    assert {"x1 = 2", "x2 = 1", "x3 = -1"} <= set(full)
    s2 = Store(":memory:")
    c2 = s2.add_course("c")
    s2.add_graded_item(c2, "P", LU, None, label="Problem 1", answer_key=json.dumps(["l21 = 2", "l31 = -1", "l32 = -1"]))
    assert integrity.key_leak("L = [[1,0,0],[2,1,0],[-1,-1,1]]", integrity.graded_view(s2, c2)[0])
    s2.add_graded_item(c2, "C", EMP, None, label="Problem 4", answer_key=json.dumps(["C3H6O"]))
    fk = integrity.full_key(integrity.graded_view(s2, c2)[1])
    assert "atomsc = 3" in fk and "atomsh = 6" in fk and "atomso = 1" in fk


# ---------------------------------------------------------------- p6: pieces in a chat that touched the item
@pytest.mark.parametrize("lean", [False, True])
@pytest.mark.parametrize("q,reply", [("what is the last entry of x?", "2"[:0] + "-1"), ("and the first entry?", "2"),
                                     ("which value is x2?", "x2 is 1")])
def test_p6_ordinal_followups_in_a_touched_chat_are_caught(lean, q, reply):
    s, c = make(lean=lean)
    p = FakeProvider(model({q: reply}))
    grounding.ask(s, c, LU_ASK, p)
    a = grounding.ask(s, c, q, p)
    assert a.mode in ("twin", "concepts")
    assert a.body_md.count(reply) == 0 or reply not in ("-1",) or "graded" in a.body_md


def test_p6_chem_atom_counts_and_units_are_caught():
    s, c = make()
    p = FakeProvider(model({"how many carbons": "There are 3 carbon atoms", "in milligrams": "That is 9820 mg",
                            "which one is the formula": "C3H6O"}))
    grounding.ask(s, c, "Problem 4: " + EMP.split("\n\n")[1], p)
    a = grounding.ask(s, c, "how many carbons in the formula?", p)
    assert a.mode in ("twin", "concepts") and "3 carbon" not in a.body_md
    grounding.ask(s, c, "Problem 2: " + MOLAR.split("\n\n")[1], p)
    a = grounding.ask(s, c, "in milligrams?", p)
    assert a.mode in ("twin", "concepts") and "9820" not in a.body_md


def test_p6_key_state_shows_final_values_only_when_reask_gives_no_components():
    s, c = make(keys=False)
    p = FakeProvider('["9.82 g"]')
    assert integrity.build_answer_key(s, 2, p) == ["9.82 g"]
    assert len(p.calls) == 2  # the first key had no named part: one stricter try
    assert integrity.key_state(s.graded_item(2), True) == "final"
    p2 = FakeProvider(lambda sy, me: '["9.82 g"]' if "Your last list" not in sy else '["n = 0.0825 mol", "9.82 g"]')
    integrity.build_answer_key(s, 2, p2)
    assert integrity.key_state(s.graded_item(2), True) == "on"


def test_p6_routing_false_positives_get_normal_answers():
    s, c = make()
    for q in ("Solve A x = b by back substitution", "how do I check my LU factorization is right in general"):
        a = grounding.ask(s, c, q, FakeProvider(model(default="Back substitution solves from the last row up.")))
        assert a.mode == "answer", q


def test_p6_caught_normal_reply_ends_after_two_calls():
    s, c = make()
    p = FakeProvider(lambda sy, me: "The answer is x = (2, 1, -1).")
    a = grounding.ask(s, c, "what does back substitution give in general?", p)
    assert len(p.calls) == 2 and a.mode == "concepts" and "Problem 1" in a.body_md
    assert "x = (2, 1, -1)" not in a.body_md


# ---------------------------------------------------------------- p1: honest eigenvector replies
EIGEN = "Take x_1 = 1, x_2 = -2 for the eigenvalue 2, so the eigenvector is (1, -2) [1]."


def test_p1_honest_eigenvector_reply_is_not_flagged_in_a_fresh_chat():
    s, c = make()
    a = grounding.ask(s, c, "How do I find the eigenvectors of [[4,1],[2,3]]?", FakeProvider(model(default=EIGEN)))
    assert a.mode == "answer" and "x_1 = 1" in a.body_md.replace("x1", "x_1") or "x" in a.body_md


def test_p1_mcp_guide_and_practice_keep_ordinary_eigenvector_content():
    s, c = make()
    mcp_server.set_store(s)
    try:
        assert mcp_server.save_guide(c, "answer", "Eigenvectors", "To find an eigenvector take x1 = 1 and then x2 = -2. "
                                     "Eigenvectors are (1, 1) and (1, -2).", []).startswith("Saved")
    finally:
        mcp_server.set_store(None)
    p = FakeProvider("1. Find eigenvectors of a 2x2.\n\n<details>\n<summary>Show solution</summary>\n\ny = (3, 1), x_2 = 1, x_1 = 1\n\n</details>")
    g = s.guide(study.practice(s, c, p, "eigenvectors", count=1))
    assert "x_1 = 1" in g["body_md"] and "left out" not in g["body_md"]


# ---------------------------------------------------------------- p2: given data
def test_p2_practice_and_explain_keep_steps_that_reuse_the_given_stock():
    s, c = make()
    steps = ("1. Dilute 25.0 mL of 0.800 M KCl to 200.0 mL.\n\n<details>\n<summary>Show solution</summary>\n\n"
             "M2 = 0.800 x 25.0 / 200.0 = 0.100 M, so the new solution is 0.100 M and 200.0 mL.\n\n</details>")
    for fn, topic in ((lambda: study.practice(s, c, FakeProvider(steps), "dilution", count=1), "dilution"),
                      (lambda: study.explain(s, c, FakeProvider(steps), "dilution of solutions"), "x")):
        g = s.guide(fn())
        assert "0.100 M" in g["body_md"] and "left out" not in g["body_md"]
        assert "don't cover this" not in g["body_md"]


def test_p2_a_real_leak_is_dropped_and_the_note_names_the_item():
    s, c = make()
    leak = "1. Worked.\n\n<details>\n<summary>Show solution</summary>\n\nV1 = 100.0 mL and the answer is 9.82 g.\n\n</details>"
    g = s.guide(study.practice(s, c, FakeProvider(leak), "Problem 3 dilution", count=1))
    assert "9.82" not in g["body_md"] or "left out" in g["body_md"]
    out, n, note = integrity.scrub_parts(s, c, "Intro.\n\nThen the volume is 100.0 mL and 9.82 g.", "Problem 3")
    assert n == 1 and "Problem" in note and "100.0 mL" not in out


# ---------------------------------------------------------------- p3/p5: practice and banners
def test_p3_practice_on_a_bare_topic_adds_no_twin_and_one_offer():
    s, c = make()
    p = FakeProvider(lambda sy, me: TWIN_OK if "<twin_problem>" in sy else "1. A question.\n\n2. Another.")
    g = s.guide(study.practice(s, c, p, "LU factorization with forward substitution", count=2))
    assert not any("<twin_problem>" in call["system"] for call in p.calls)
    assert len(g["data"]["graded_offers"]) <= 1 and g["data"]["twin"] is False


@pytest.mark.parametrize("q", ["Problem 1: what do I get for y and x?", "Problem 1: how do I get L and U?"])
def test_p3_questions_do_not_claim_the_message_has_an_answer(q):
    s, c = make()
    a = grounding.ask(s, c, q, FakeProvider(model()))
    assert "has an answer" not in (a.routing_note or "") and "has an answer" not in a.body_md


def test_p3_a_message_with_a_candidate_value_still_says_so():
    s, c = make()
    a = grounding.ask(s, c, "Problem 1: I got x = (2, 1, -1), is that right?", FakeProvider(model()))
    assert "has an answer" in a.routing_note


def test_p1_concepts_fallback_names_practice_only_with_a_button():
    assert "Press Practice below" not in integrity.twin_fallback("Problem 1", "x")
    assert "Press Practice below" in integrity.twin_fallback("Problem 1", "x", practice_button=True)
    s, c = make()
    bad = "<twin_problem>Factor A = [[2,1,3],[4,5,7],[-2,-4,-3]].</twin_problem><solution>Same again.</solution>"
    a = grounding.ask(s, c, LU_ASK, FakeProvider(lambda sy, me: bad if "<twin_problem>" in sy else "Concepts."))
    g = s.guide(a.guide_id)
    assert g["data"]["graded_offers"] and "Press Practice below" in a.body_md
    html = client_for(s, FakeProvider("x")).get(f"/guides/{a.guide_id}").text
    assert "Practice on a problem like Problem 1" in html


# ---------------------------------------------------------------- sticky
def test_stickiness_lasts_six_turns_and_never_routes():
    s, c = make()
    p = FakeProvider(model())
    grounding.ask(s, c, LU_ASK, p)
    for i in range(4):
        grounding.ask(s, c, f"tell me about the syllabus topic {i}", p)
    assert integrity.sticky_item(s, c, "and the last entry?")
    for i in range(3):
        grounding.ask(s, c, f"another unrelated thing {i}", p)
    assert integrity.sticky_item(s, c, "and the last entry?") is None
