"""Cycle 7 routing and context: one regression test per panel repro, each next to its honest look-alike."""
import json

import pytest

from course_assistant import grounding, integrity, mcp_server, study
from course_assistant.providers import FakeProvider
from test_fx2_integrity import course
from test_fx3_integrity import TWIN_OK

PRE2 = ("Pre-lab question 2\n\nA 10.00 mL sample of the 0.100 M stock is diluted to 100.0 mL (dilution A). Then 10.00 mL of "
        "dilution A is diluted to 100.0 mL (dilution B). What is the molarity of each dilution?")
REP3 = "Report question 3\n\nExplain why you add water to the 100.0 mL mark and then one more dropper."
ESS = ("Essay 1: Why eigenvectors matter\n\nIn about 800 words, argue which idea about eigenvalues and eigenvectors matters most. "
       "Use at least two lecture examples, such as diagonalization or Markov chains.\n\nRubric: thesis 30%, evidence 40%.")


def make(lab=False, essay=False):
    s, c, m = course()
    ids = {(i["material_id"], i["label"]): i["id"] for i in s.graded_items(c)}
    ps2, ps3 = m["problem-set-2"], m["problem-set-3"]
    keys = {(ps2, "Problem 1"): ["d = 11", "z = -2", "y = 1", "x = -3/2"],
            (ps2, "Problem 2"): ["x1 = (-2, 1, -1)", "x2 = (3, 0, 1)"],
            (ps2, "Problem 3"): ["L = [[1,0,0],[3,1,0],[0,2,1]]", "U = [[1,2,1],[0,2,-2],[0,0,5]]", "y = (2, 6, -10)",
                                 "x = (2, 1, -2)", "y2 = 6", "y3 = -10", "u22 = 2", "u33 = 5"],
            (ps3, "Problem 1"): ["5.11 g", "n = 0.0875 mol"], (ps3, "Problem 2"): ["C3H4O3"]}
    for k, v in keys.items():
        s.update_graded_item(ids[k], answer_key=json.dumps(v))
    if lab:
        s.add_graded_item(c, "Pre-lab questions 2", PRE2, None, kind="problem", label="Pre-lab question 2",
                          answer_key=json.dumps(["A = 0.0100 M", "B = 0.00100 M"]))
        s.add_graded_item(c, "Report questions 3", REP3, None, kind="problem", label="Report question 3")
    if essay:
        s.add_graded_item(c, "Essay 1: Why eigenvectors matter", ESS, None, kind="essay", label="Essay 1")
    return s, c, m, ids


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


def held(a):
    return a.mode in ("twin", "concepts")


# ------------------------------------------------------------------ p6: a set described, not numbered, touches every item
@pytest.mark.parametrize("opener", ["In the LU problem on pset 2, what is the multiplier in row 2?",
                                    "I'm stuck on the LU question from my problem set"])
def test_p6_set_level_opening_guards_components(opener):
    s, c, m, ids = make()
    p = FakeProvider(model({"u22": "u22 = 2", "y2": "y2 = 6", "u33": "u33 = 5, so U ends in 5"}))
    grounding.ask(s, c, opener, p)
    for q in ("what's u22?", "what is y2?", "and u33?"):
        a = grounding.ask(s, c, q, p)
        assert held(a), q
        assert "u22 = 2" not in a.body_md and "y2 = 6" not in a.body_md and "u33 = 5" not in a.body_md


def test_p6_set_level_first_turn_is_guarded_even_with_ask_normally():
    s, c, m, ids = make()
    p = FakeProvider(model({"u22": "The entry u22 = 2 and y2 = 6 and u33 = 5"}))
    a = grounding.ask(s, c, "In the LU problem on pset 2, what is u22, y2 and u33?", p, route=False)
    assert held(a) and "u33 = 5" not in a.body_md


def test_p6_set_level_look_alikes_are_not_touched():
    s, c, m, ids = make()
    p = FakeProvider(model({"u22": "u22 = 2"}))
    assert integrity.match_graded(s, c, "When is the problem set due?") is None
    assert integrity.match_graded(s, c, "What is the best way to study for a problem set?") is None
    a = grounding.ask(s, c, "How do I do an LU factorization in general?", p)
    assert a.mode == "answer"
    assert grounding.ask(s, c, "what's u22?", p).mode == "answer"  # nothing touched, so a plain question stays plain


# ------------------------------------------------------------------ p6 / p1: a paste goes to the item it pastes
P2_PASTE = ("Let A be the 3 by 3 matrix with rows (1, 6, -3), (-2, 3, 4), (1, 0, -2), and let b1 = (7, 3, 0) and b2 = (0, -2, 1) be "
            "column vectors. Solve both A x1 = b1 and A x2 = b2 at once by forming the augmented matrix and eliminating once.")
P3_PASTE = ("Let A be the 3 by 3 matrix with rows (1, 2, 1), (3, 8, 1), (0, 4, 1), and let b = (2, 12, 2) as a column vector. "
            "Find the factorization A = LU, then solve Ax = b.")


@pytest.mark.parametrize("q", [P2_PASTE, "Hi! " + P2_PASTE, "A=[[1,6,-3],[-2,3,4],[1,0,-2]], b1=[7,3,0], b2=[0,-2,1]"])
def test_p6_problem_2_paste_is_labelled_problem_2(q):
    s, c, m, ids = make()
    r = integrity.match_graded(s, c, q)
    assert r and r["id"] == ids[(m["problem-set-2"], "Problem 2")]


def test_p6_plain_paste_does_not_claim_an_answer_but_a_real_answer_does():
    s, c, m, ids = make()
    a = grounding.ask(s, c, P3_PASTE, FakeProvider(model()))
    assert a.routing_note and "has an answer" not in a.routing_note
    a = grounding.ask(s, c, P3_PASTE + " My answer: L = [[1,0,0],[3,1,0],[0,2,1]] and x = (2, 1, -2). Is it right?",
                      FakeProvider(model()))
    assert "has an answer" in a.routing_note


def test_p6_chatty_lead_in_plus_inline_problem_is_not_a_proof_draft():
    s, c, m, ids = make()
    lead = ("Hi there! Thanks so much for helping me, I have been stuck on this for hours and my TA is out of office, and I "
            "want to understand how to do it so please explain it to me carefully. ")
    r = integrity.match_graded(s, c, lead + "3x + 2y + z = 5, 6x + dy + 4z = 13, y + 3z = 7")
    assert r and r["id"] == ids[(m["problem-set-2"], "Problem 1")] and r.get("intent") != "feedback"


# ------------------------------------------------------------------ p1: plain-words matrix, method-only follow-up
def test_p1_matrix_with_rows_in_words_does_not_route_but_the_graded_system_does():
    s, c, m, ids = make()
    assert integrity.match_graded(s, c, "What is the eigenvector for eigenvalue 2 of the matrix with rows 4 1 and 2 3?") is None
    assert integrity.match_graded(s, c, "determinant of the matrix with rows 4 1 and 2 3") is None
    assert integrity.match_graded(s, c, "3x + 2y + z = 5, 6x + dy + 4z = 13, y + 3z = 7")


def test_p1_last_one_followup_allows_a_method_only_reply_but_not_a_value():
    for reply, ok in (("Do the first two elimination steps first, then substitute back.", True),
                      ("The last multiplier, l32, is 2.", False)):
        s, c, m, ids = make()
        p = FakeProvider(model({"last one": reply}))
        grounding.ask(s, c, "For problem 3 on pset 2 I want the multipliers l21 and l32", p)
        a = grounding.ask(s, c, "ok and the last one?", p)
        assert (a.mode == "answer") is ok, reply


# ------------------------------------------------------------------ p5: how-to is not a check, essay needs its topic words
def test_p5_how_to_check_in_a_tool_is_not_a_check_ask_or_an_essay_reply():
    s, c, m, ids = make(essay=True)
    long = ("The eigenvalues of A^4 are the fourth powers of the eigenvalues of A, so in Julia you can compare eigvals(A).^4 with "
            "eigvals(A^4). This works because diagonalization writes A = P D P^-1, so A^4 = P D^4 P^-1 and the eigenvalues of "
            "A^4 are the entries of D^4. That is why eigenvalues explain the long-run behavior of Markov chains, the main idea "
            "an argument about eigenvectors should show.")
    p = FakeProvider(model({"Julia": long, "thesis": "Think about which idea you can defend with two examples."}))
    grounding.ask(s, c, "What should my thesis be for Essay 1 about which idea matters most?", p)
    a = grounding.ask(s, c, "How do I check eigenvalues of A^4 in Julia, as in the diagonalization notes?", p)
    assert a.mode == "answer" and "fourth powers" in a.body_md
    assert not integrity.is_check_ask("How do I check eigenvalues of A^4 in Julia?")
    assert integrity.is_check_ask("I got 5.11 g for the salt problem, is that right?")
    assert integrity.is_check_ask("How do I check my answer? I got x = 3/2")


def test_p5_an_essay_item_needs_two_of_its_topic_words_to_route():
    s, c, m, ids = make(essay=True)
    assert integrity.match_graded(s, c, "Write an intro paragraph about diagonalization") is None
    r = integrity.match_graded(s, c, "Write an intro paragraph about eigenvalues and eigenvectors for my essay")
    assert r and r["kind"] == "essay"


# ------------------------------------------------------------------ p2: keyless item, percent and conversions, copies
def test_p2_percent_cards_and_guides_pass_on_a_keyless_item():
    s, c, m, ids = make(lab=True)
    mcp_server.set_store(s)
    try:
        for f, b in (("What is percent yield?", "Actual over theoretical, times 100%"),
                     ("First step of an empirical formula?", "Assume a 100 g sample"),
                     ("What is 250 mL in L?", "0.250 L")):
            assert mcp_server.add_flashcards(c, [{"front": f, "back": b}]).startswith("Added"), f
        assert mcp_server.save_guide(c, "answer", "Notes", "250 mL = 0.250 L", []).startswith("Saved")
        assert mcp_server.save_guide(c, "answer", "Notes", "Percent yield is actual over theoretical times 100.", []).startswith("Saved")
        # the look-alike: the graded question copied as a card is still refused
        assert mcp_server.add_flashcards(c, [{"front": "Report question 3", "back": "Add water to the 100.0 mL mark, then one "
                                                "more dropper"}]).startswith("NOT ADDED")
    finally:
        mcp_server.set_store(None)


def test_p2_practice_copy_of_a_graded_question_is_dropped_with_its_latex_solution():
    s, c, m, ids = make(lab=True)
    raw = ("1. A 10.00 mL sample of the 0.100 M stock is diluted to 100.0 mL (dilution A). Then 10.00 mL of dilution A is "
           "diluted to 100.0 mL (dilution B). What is the molarity of each dilution?\n\n<details>\n<summary>Show solution</summary>"
           "\n\nDilution A: $(0.100)(10.00)/100.0 = 0.0100\\ M$. Dilution B: $0.00100\\ M$.\n\n</details>\n\n"
           "2. A 5.00 mL sample of a 0.400 M stock is diluted to 50.0 mL. What is the new molarity?\n\n<details>\n"
           "<summary>Show solution</summary>\n\n$0.0400\\ M$\n\n</details>")
    text, n, note = integrity.scrub_parts(s, c, raw, "serial dilution")
    assert n == 2 and "0.0100" not in text and "10.00 mL sample" not in text
    assert "5.00 mL sample" in text and "0.0400" in text and "Pre-lab question 2" in note  # the twin with new numbers stays


def test_p6_cards_with_a_bare_value_of_an_item_are_refused_but_concept_cards_pass():
    s, c, m, ids = make(lab=True)
    mcp_server.set_store(s)
    try:
        for f, b in (("d for problem 1 that forces a swap", "eleven"),
                     ("carbon atoms in the compound of problem 2 of problem set 3", "3")):
            assert mcp_server.add_flashcards(c, [{"front": f, "back": b}]).startswith("NOT ADDED"), f
        assert mcp_server.add_flashcards(c, [{"front": "What does a pivot do?", "back": "It divides the rows below"}]).startswith("Added")
    finally:
        mcp_server.set_store(None)


# ------------------------------------------------------------------ p3: honest guides and answers
def test_p3_guide_restating_givens_saves_but_the_solution_does_not():
    s, c, m, ids = make()
    mcp_server.set_store(s)
    try:
        ok = "Problem 3 gives A with rows (1,2,1), (3,8,1), (0,4,1) and b = (2,12,2); factor A as LU first."
        assert mcp_server.save_guide(c, "answer", "Notes", ok, []).startswith("Saved")
        bad = "Problem 3: L = [[1,0,0],[3,1,0],[0,2,1]] and U = [[1,2,1],[0,2,-2],[0,0,5]]"
        assert mcp_server.save_guide(c, "answer", "Notes", bad, []).startswith("NOT SAVED")
    finally:
        mcp_server.set_store(None)


def test_p3_due_date_answer_naming_the_set_is_not_refused_in_a_touched_chat():
    ans = "Problem Set 2 is due Thursday, Sep 17, 2026 at 11:59 pm and is worth 60 points. Office hours are 10 to 11 am."
    s, c, m, ids = make()
    p = FakeProvider(model({"due": ans}))
    grounding.ask(s, c, "For problem 1 on pset 2 I want d", p)
    a = grounding.ask(s, c, "When is Problem Set 2 due?", p)
    assert a.mode == "answer" and "11:59 pm" in a.body_md


# ------------------------------------------------------------------ decision 10: failed key builds are cached
def test_failed_key_builds_are_not_retried_on_every_ask():
    s, c, m, ids = make()
    iid = ids[(m["problem-set-2"], "Problem 1")]
    s.update_graded_item(iid, answer_key="[]")  # a build that failed
    p = FakeProvider('["d = 11"]')
    integrity.ensure_keys(s, c, p)
    integrity.ensure_keys(s, c, p)
    assert s.graded_item(iid)["answer_key"] == "[]" and not p.calls
    assert integrity.build_answer_key(s, iid, p) == ["d = 11"]  # Try again still works


def test_p6_blind_practice_solution_of_a_graded_problem_is_scrubbed_but_a_new_numbers_one_stays():
    s, c, m, ids = make()
    graded = ("1. Find the d that forces a swap.\n\n<details>\n<summary>Show solution</summary>\n\nd = 11, then z = -2, y = 1, "
              "x = -3/2.\n\n</details>")
    own = ("2. Solve 5x + y = 7 and 2x + 3y = 8.\n\n<details>\n<summary>Show solution</summary>\n\nx = 13/13 = 1, y = 2.\n\n</details>")
    text, n, note = integrity.scrub_parts(s, c, graded + "\n\n" + own, "Problem 1 on pset 2: Gaussian elimination row exchange")
    assert n >= 1 and "-3/2" not in text and "Problem 1" in note
    assert "5x + y = 7" in text
