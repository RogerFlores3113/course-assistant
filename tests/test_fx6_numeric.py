"""Cycle 6 numeric integrity: one regression test per panel repro, on original fixtures."""
import json

import pytest

from course_assistant import grounding, integrity, mcp_server, study
from course_assistant.providers import FakeProvider
from course_assistant.store import Store
from test_fx3_integrity import TWIN_OK, client_for
from test_fx5_numeric import DIL, EMP, LU, LU_ASK, LU_KEY, MOLAR, item, make, model

LONG = ("Problem 6 (20 points)\n\nConsider the 3 by 3 matrix A with rows 2 1 1, 4 -6 0 and -2 7 2, and the right-hand side "
        "b = (5, -2, 9). Use elimination without row exchanges to find the factors L and U of the matrix, recording each "
        "multiplier you use. Then solve the lower triangular system for y by forward substitution and the upper triangular "
        "system for x by back substitution. Report the multipliers, L, U, y and x.")
PROOF = ("Problem 5 (10 points)\n\nProve that if elimination on an invertible matrix A needs no row exchanges then A equals LU, "
         "and explain why each step of the elimination holds.")


def make_proof():
    s, c = make()
    s.add_graded_item(c, "Problem 5 (Set)", LONG.replace("Problem 6", "Problem 5"), None, kind="problem", label="Problem 5",
                      answer_key=json.dumps(["l21 = 2", "l31 = -1", "l32 = 1"]))
    s.add_graded_item(c, "Problem 6 (Set)", PROOF.replace("Problem 5", "Problem 6"), None, kind="proof", label="Problem 6")
    return s, c


# ---------------------------------------------------------------- p6: a pasted problem is not a draft of the proof
@pytest.mark.parametrize("paste", [
    LONG.split("\n\n", 1)[1],
    "I need help with this: " + LONG.split("\n\n", 1)[1],
    "Use elimination without row exchanges to find the factors L and U of the matrix with rows 2 1 1, 4 -6 0 and -2 7 2. "
    "Record each multiplier you use. Then solve the lower triangular system for y by forward substitution and the upper "
    "system by back substitution.",
])
def test_p6_pasted_numeric_problem_matches_the_numeric_item_not_the_proof(paste):
    s, c = make_proof()
    m = integrity.match_graded(s, c, paste)
    assert m and m["id"] == 5 and m.get("intent") != "feedback"


def test_p6_pasted_problem_chat_stays_guarded_on_the_right_item():
    s, c = make_proof()
    p = FakeProvider(model({"third": "2"}))
    grounding.ask(s, c, LONG.split("\n\n", 1)[1], p)
    assert integrity.sticky_item(s, c, "third?")["id"] == 5


def test_p6_a_real_proof_draft_still_gets_feedback():
    s, c = make_proof()
    draft = ("Here is my draft for the proof: I think that if elimination needs no row exchanges then every pivot is nonzero, "
             "so I can undo each step. My argument is that the multipliers fill the entries of L. Can you give me feedback on it?")
    m = integrity.match_graded(s, c, draft)
    assert m and m["id"] == 6 and m.get("intent") == "feedback"


# ---------------------------------------------------------------- p6: bare ordinal and "and <name>?" follow-ups
@pytest.mark.parametrize("q,reply", [
    ("third?", "-1"), ("first?", "2"), ("last?", "-1"), ("and row 3?", "Minus one times row 1, so you add it once."),
    ("and x?", "x = (2, 1, -1)"), ("and x3?", "x3 = -1"), ("and the one below?", "-1"),
    ("what's the multiplier in row 2?", "The multiplier for row 2 is 2"),
])
@pytest.mark.parametrize("lean", [False, True])
def test_p6_bare_followups_in_a_touched_chat_are_caught(q, reply, lean):
    s, c = make(lean=lean)
    p = FakeProvider(model({q: reply}))
    grounding.ask(s, c, LU_ASK, p)
    a = grounding.ask(s, c, q, p)
    assert a.mode in ("twin", "concepts"), (q, a.body_md)
    assert reply not in a.body_md


def test_p6_bare_followups_in_an_unrelated_chat_are_fine():
    s, c = make()
    a = grounding.ask(s, c, "third?", FakeProvider(model({"third": "2"})))
    assert a.mode == "answer"


def test_p6_named_entry_retouches_after_the_window():
    s, c = make()
    p = FakeProvider(model({"l31": "l31 = -1", "last entry of x": "-1"}))
    grounding.ask(s, c, LU_ASK, p)
    for i in range(8):
        grounding.ask(s, c, f"another unrelated thing {i}", p)
    assert integrity.sticky_item(s, c, "and the last entry?") is None
    assert integrity.sticky_item(s, c, "what was l31 again?")["id"] == 1
    a = grounding.ask(s, c, "what was l31 again?", p)
    assert a.mode in ("twin", "concepts") and "l31 = -1" not in a.body_md


def test_p6_chemistry_intermediates_on_a_key_with_them_are_caught():
    s, c = make()
    s.update_graded_item(4, answer_key=json.dumps(["C3H6O", "moles C = 3.407 mol", "ratio = 1.33", "multiplier = 3"]))
    p = FakeProvider(model({"moles of carbon": "That is 3.407 mol of carbon", "ratio of H to C": "The ratio is 1.33",
                            "what do I multiply by": "Multiply by 3"}))
    grounding.ask(s, c, "Problem 4: " + EMP.split("\n\n")[1], p)
    for q, bad in (("moles of carbon?", "3.407"), ("ratio of H to C?", "1.33"), ("what do I multiply by?", "Multiply by 3")):
        a = grounding.ask(s, c, q, p)
        assert a.mode in ("twin", "concepts") and bad not in a.body_md, q


def test_p6_answer_key_prompt_asks_for_intermediates():
    s, c = make(keys=False)
    p = FakeProvider('["moles C = 3.407 mol", "ratio = 1.33", "multiplier = 3", "C3H6O"]')
    integrity.build_answer_key(s, 4, p)
    first = p.calls[0]["system"].lower()
    assert all(w in first for w in ("moles", "ratio", "multiplier"))


# ---------------------------------------------------------------- p2: round given values and honest chemistry
LAB = ("Pre-lab question 1\n\nYou will prepare a 0.100 M NaCl stock solution. How many grams of NaCl (molar mass 58.44 g/mol) "
       "are needed for 100.0 mL of this solution? Report your answer in grams.")


def make_lab():
    s, c = make()
    s.add_graded_item(c, "Pre-lab questions 1", LAB, None, kind="problem", label="Pre-lab question 1",
                      answer_key=json.dumps(["n = 0.0100 mol", "m = 0.584 g", "V1 = 100.0 mL"]))
    return s, c


@pytest.mark.parametrize("q", ["What is the pH of a 0.10 M acetic acid solution?", "What is the pH of 0.10 M HCl?",
                               "What is a 0.10 M solution?", "What does a 0.100 M solution mean?",
                               "Assume a 100 g sample so percentages become grams"])
def test_p2_a_bare_round_value_never_routes_to_a_graded_item(q):
    s, c = make_lab()
    assert integrity.match_graded(s, c, q) is None


def test_p2_pH_question_answers_normally_and_its_honest_reply_is_not_rejected():
    s, c = make_lab()
    reply = ("For a 0.10 M weak acid with Ka about 1.8e-5, [H+] is roughly the square root of Ka times 0.10, which is "
             "0.0013 M, so the pH is about 2.87. A 0.10 M solution is 0.10 mol per liter [1].")
    p = FakeProvider(model(default=reply))
    for route in (True, False):
        a = grounding.ask(s, c, "What is the pH of a 0.10 M acetic acid solution?", p, route=route)
        assert a.mode == "answer" and "2.87" in a.body_md, route
    assert len(p.calls) == 2  # one call each, no regeneration


def test_p2_units_are_aware():
    s, c = make_lab()
    it = item(s, 5)
    assert integrity.key_leak("Assume a 100 g sample.", it, {"named": True}) == []
    assert integrity.key_leak("That is 0.0100 mol.", it, {"named": True})
    assert integrity.key_leak("That is 0.0100 g.", it, {"named": True}) == []
    assert integrity.key_leak("The solution is 0.10 M.", it, {"named": True}) == []
    assert integrity.key_leak("the mass is 0.584 g", it, {"named": True})


@pytest.mark.parametrize("key", [["x"], ["linear = yes"], ["yes", "no"]])
def test_p2_one_word_keys_are_invalid(key):
    s, c = make()
    s.update_graded_item(3, answer_key=json.dumps(key))
    assert integrity.item_key(item(s, 3)) == []
    assert integrity.key_state(item(s, 3), True) == "failed"
    mcp_server.set_store(s)
    try:
        assert "nothing was stored" in _fail_text(lambda: mcp_server.set_answer_key(3, key))
    finally:
        mcp_server.set_store(None)
    s2, c2 = make(keys=False)
    assert integrity.build_answer_key(s2, 3, FakeProvider(json.dumps(key))) == []
    assert integrity.key_state(s2.graded_item(3), True) == "failed"


def _fail_text(fn):
    try:
        return str(fn())
    except Exception as e:  # the MCP layer raises a tool error
        return str(e)


# ---------------------------------------------------------------- p3: multi right-hand sides
MULTI = ("Problem 7 (15 points)\n\nUse Gauss-Jordan elimination on the augmented matrix of A with rows 2 1 -1, 1 3 2 and 1 0 1 "
         "and the two right-hand sides b1 = (1, 2, 3) and b2 = (0, 5, 4). Report the solutions x1 and x2.")
MULTI_KEY = ["x1 = (-2, 1, -1)", "x2 = (3, 0, 1)"]
TRI = ("Problem 8 (15 points)\n\nReduce the system 2x + 5y + z = 0, 4x + 11y + z = 3, -2x - 6y - 3z = 2 to triangular form and solve it.")
TRI_KEY = ["x = (-1, 4, -18)"]


def make_multi():
    s, c = make()
    s.add_graded_item(c, "Problem 7 (Set)", MULTI, None, kind="problem", label="Problem 7", answer_key=json.dumps(MULTI_KEY))
    s.add_graded_item(c, "Problem 8 (Set)", TRI, None, kind="problem", label="Problem 8", answer_key=json.dumps(TRI_KEY))
    return s, c


MULTI_LEAKS = [
    "[[-2, 3], [1, 0], [-1, 1]]",
    "$X=\\begin{bmatrix}-2&3\\\\1&0\\\\-1&1\\end{bmatrix}$",
    "[[1,0,0,-2,3],[0,1,0,1,0],[0,0,1,-1,1]]",
    "\\begin{bmatrix}1&0&0&|&-2&3\\\\0&1&0&|&1&0\\\\0&0&1&|&-1&1\\end{bmatrix}",
    "[[-2, 1, -1], [3, 0, 1]]",
    "the columns are (-2, 1, -1) and (3, 0, 1)",
    "-2 1 -1 and 3 0 1",
    "[[2,5,1,0],[0,1,-1,3],[0,0,-18,-18+0]]"[:0] + "[[2,5,1,0],[0,1,-1,22],[0,0,1,-18]]",
]


@pytest.mark.parametrize("text", MULTI_LEAKS[:7])
def test_p3_multi_rhs_solution_forms_are_leaks(text):
    s, c = make_multi()
    assert integrity.key_leak(text, item(s, 5), {"named": True}), text


def test_p3_triangular_system_as_augmented_matrix_is_a_leak():
    s, c = make_multi()
    # 2x + 5y + z = 0, y - z = 22, z = -18 solves to x = (-1, 4, -18) 
    assert integrity.key_leak("[[2,5,1,0],[0,1,-1,22],[0,0,1,-18]]", item(s, 6), {"named": True})
    assert integrity.key_leak("[[2,5,1,0],[0,1,-1,22],[0,0,1,-18]]", item(s, 6), {}) 


def test_p3_a_system_that_is_not_the_answer_is_fine():
    s, c = make_multi()
    assert integrity.key_leak("[[2,5,1,0],[4,11,1,3],[-2,-6,-3,2]]", item(s, 6), {}) == []
    assert integrity.key_leak("[[1,0],[0,1]]", item(s, 5), {"named": True}) == []


def test_p3_flat_entry_lists_for_l_and_u():
    s, c = make()
    assert integrity.key_leak("1 0 0 2 1 0 -1 -1 1 and 2 1 3 0 3 1 0 0 1", item(s, 1), {"named": True})
    assert integrity.key_leak("U is 2 1 3 0 3 1 0 0 1", item(s, 1), {"named": True})


def test_p3_ask_normally_twins_guides_and_cards_catch_multi_rhs():
    s, c = make_multi()
    q = "What does the augmented matrix look like at the end of elimination in problem 7?"
    a = grounding.ask(s, c, q, FakeProvider(model(default="It ends as [[1,0,0,-2,3],[0,1,0,1,0],[0,0,1,-1,1]].")), route=False)
    assert a.mode in ("twin", "concepts") and "[[1,0,0,-2,3]" not in a.body_md
    twin_bad = ("<twin_problem>Use Gauss-Jordan elimination on A with rows 1 2 3, 2 5 3 and 1 0 8 and b1 = (1, 0, 4).</twin_problem>"
                "<solution>The answer is [[-2, 3], [1, 0], [-1, 1]]</solution>")
    p = FakeProvider(lambda sy, me: twin_bad if "<twin_problem>" in sy else "Concepts only.")
    a = grounding.ask(s, c, "Problem 7: use Gauss-Jordan elimination on the augmented matrix with rows 2 1 -1, 1 3 2 and 1 0 1", p)
    assert "[[-2, 3]" not in a.body_md
    mcp_server.set_store(s)
    try:
        for body in ("Problem 7 solutions side by side: [[-2, 3],[1, 0],[-1, 1]]", MULTI_LEAKS[2], MULTI_LEAKS[1],
                     "Problem 8 as an augmented system: [[2,5,1,0],[0,1,-1,22],[0,0,1,-18]]"):
            assert mcp_server.save_guide(c, "answer", "Notes", body, []).startswith("NOT SAVED"), body
        for back in ("[[-2, 3], [1, 0], [-1, 1]]", "[[1,0,0,-2,3],[0,1,0,1,0],[0,0,1,-1,1]]"):
            assert mcp_server.add_flashcards(c, [{"front": "Problem 7: the solutions?", "back": back}]).startswith("NOT ADDED"), back
    finally:
        mcp_server.set_store(None)


# ---------------------------------------------------------------- p1: a loose "matrix with rows" is not the graded matrix
@pytest.mark.parametrize("q", ["What is the eigenvector for eigenvalue 2 of the matrix with rows 4 1 and 2 3?",
                               "what is the determinant of the matrix with rows 4 1 and 2 3",
                               "Find the inverse of the matrix with rows 4 1 and 2 3, step by step"])
def test_p1_plain_words_small_matrix_does_not_route_to_a_graded_item(q):
    s, c = make_proof()
    assert integrity.match_graded(s, c, q) is None
    a = grounding.ask(s, c, q, FakeProvider(model(default="Subtract the eigenvalue from the diagonal and solve.")))
    assert a.mode == "answer"


def test_p1_the_same_matrix_in_plain_words_still_routes():
    s, c = make_proof()
    m = integrity.match_graded(s, c, "Explain how to do elimination on the 3 by 3 matrix with rows 2 1 1, 4 -6 0 and -2 7 2")
    assert m and m["id"] == 5


# ---------------------------------------------------------------- MCP: definitions, small integers, arrow cards
SA9 = ("Problem 9 (5 points)\n\nIn two or three sentences, explain why molarity is defined with the volume of the final "
       "solution rather than the volume of solvent added.")


def make_mcp():
    s, c = make_lab()
    s.add_graded_item(c, "Problem 9 (Set)", SA9, None, kind="essay", label="Problem 9")
    s.add_graded_item(c, "Problem 10 (Set)", "Problem 10\n\nFind the value of d that forces a row exchange in the matrix with "
                      "rows 1 2 3, 2 4 d and 1 0 1, then solve.", None, kind="problem", label="Problem 10",
                      answer_key=json.dumps(["d = 11", "x = (11, 4, 2)"]))
    mcp_server.set_store(s)
    return s, c


@pytest.mark.parametrize("body", [
    "Molarity is moles of solute per liter of solution. A mole is 6.022e23 particles.",
    "Assume a 100 g sample so percentages become grams, then convert each mass to moles.",
    "Percent composition is the mass of each element divided by the total mass, times 100.",
    "Lectures 10 and 11 cover the four fundamental subspaces and orthogonality.",
    "The exam is worth 11 points and the quiz 10 points.",
])
def test_mcp_plain_definitions_and_small_integers_save(body):
    s, c = make_mcp()
    try:
        assert mcp_server.save_guide(c, "answer", "Notes", body, []).startswith("Saved"), body
    finally:
        mcp_server.set_store(None)


def test_mcp_refusals_still_name_the_right_item():
    s, c = make_mcp()
    try:
        r = mcp_server.save_guide(c, "answer", "Notes", "Then d is eleven, so x = (11, 4, 2).", [])
        assert r.startswith("NOT SAVED") and "Problem 10" in r
        r = mcp_server.save_guide(c, "answer", "Notes", "The mass needed is 0.584 g of NaCl.", [])
        assert r.startswith("NOT SAVED") and "Pre-lab" in r
    finally:
        mcp_server.set_store(None)


@pytest.mark.parametrize("front,back", [("Problem 10: x3?", "x3 -> 2"), ("LU problem x3", "2"), ("x3 -> 2", "2"),
                                        ("l21 -> 2", "2"), ("first multiplier l21", "2"), ("d for problem 10 that forces a swap", "eleven")])
def test_mcp_arrow_and_component_cards_are_refused(front, back):
    s, c = make_mcp()
    try:
        assert mcp_server.add_flashcards(c, [{"front": front, "back": back}]).startswith("NOT ADDED"), (front, back)
    finally:
        mcp_server.set_store(None)


def test_mcp_plain_cards_are_added():
    s, c = make_mcp()
    try:
        assert mcp_server.add_flashcards(c, [{"front": "What is molarity?", "back": "Moles of solute per liter of solution"},
                                             {"front": "Define percent yield", "back": "actual / theoretical x 100"}]).startswith("Added")
    finally:
        mcp_server.set_store(None)


def test_flashcards_from_a_graded_numeric_material_are_concept_only():
    from test_fx2_integrity import course
    s, c, m = course()
    mid = m["problem-set-3"]
    cards = [{"front": "What is a limiting reagent?", "back": "The reactant that runs out first.", "cite": 1},
             {"front": "Problem 1: grams of NaCl?", "back": "5.11 g", "cite": 1}]
    p = FakeProvider(json.dumps(cards))
    s.update_graded_item(next(i["id"] for i in s.graded_items(c) if i["material_id"] == mid and i["label"] == "Problem 1"),
                         answer_key=json.dumps(["5.11 g"]))
    res = study.make_flashcards(s, c, p, material_id=mid)
    fronts = [x["front"] for x in s.flashcards(c)]
    assert "What is a limiting reagent?" in fronts and "Problem 1: grams of NaCl?" not in fronts and res.guarded == 1
    assert "Never" in p.calls[0]["messages"][0]["content"]
