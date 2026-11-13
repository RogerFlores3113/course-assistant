"""Cycle 10 routing and context: each panel repro next to its honest look-alike. Fixtures are original (fx9_world)."""
import pytest

from course_assistant import grounding, integrity, mcp_server
from course_assistant.providers import FakeProvider
from fx9_world import world

PART_FORMS = ["Problem 1b", "Solve problem 1b", "Problem 1a: find x", "Problem 1a please", "help with 1b", "prob 1b", "q1b",
              "number 1b", "1(a)", "1b", "#1b", "Problem 1.b", "problem 1 (b)", "Problem 1(b)", "Problem 1 part b", "P1b"]


@pytest.mark.parametrize("q", PART_FORMS)
def test_d4_part_forms_resolve_to_the_item_number(q):
    s, c, ids = world()
    m = integrity.match_graded(s, c, q)
    assert m and m["id"] == ids["Problem 1"], q
    assert integrity._own_ref(q, s.graded_item(ids["Problem 1"]))


@pytest.mark.parametrize("q", ["I need 5g of salt for 3d printing", "Section 3a of the syllabus lists the late policy",
                               "The 4d model is on page 3b", "what is problem set 4 about", "a 3rd term and 2nd term"])
def test_d4_look_alikes_name_no_item(q):
    s, c, ids = world()
    assert not integrity._prob_refs(q) or integrity.match_graded(s, c, q) is None


def test_d4_parts_that_exist_as_separate_items_match_exactly():
    s, c, ids = world(essay=False)
    a = s.add_graded_item(c, "Set 7: Problem 3(a)", "Problem 3(a). Say what a pivot is.", None, kind="problem", label="Problem 3(a)")
    b = s.add_graded_item(c, "Set 7: Problem 3(b)", "Problem 3(b). Say what a multiplier is.", None, kind="problem", label="Problem 3(b)")
    assert integrity.match_graded(s, c, "help with 3b")["id"] == b
    assert integrity.match_graded(s, c, "help with 3a")["id"] == a


@pytest.mark.parametrize("reply", ["The solution is (1, 1, 2).", "x: 1 1 2", "[1; 1; 2]", "<1,1,2>", '{"x": [1, 1, 2]}',
                                   "The entries are 1, 1, 2."])
def test_d4_part_named_question_makes_the_chat_sticky_and_checked(reply):
    s, c, ids = world()
    q = "What is the solution in problem 1b?"
    s.add_chat_message(c, "user", q)
    st = integrity.sticky_item(s, c, "and what was x again?")
    assert st and st["id"] == ids["Problem 1"]
    hit = integrity.check_reply(s, c, "and what was x again?", reply, force_item=st)
    assert hit and hit["id"] == ids["Problem 1"]


def test_d4_leak_is_caught_for_a_part_named_question():
    s, c, ids = world()
    q = "What is x in problem 1b?"
    m = integrity.match_graded(s, c, q)
    assert m and m["id"] == ids["Problem 1"]
    hit = integrity.check_reply(s, c, q, "The solution is (1, 1, 2).", force_item=m)
    assert hit and hit["id"] == ids["Problem 1"]


# ------------------------------------------------------------------ decision 5: proofs
PROOFS = [
    "Suppose Ax = b has two different solutions x and y. Take z = x + t(y - x). Then Az = b for every t. Since y differs from x, there "
    "are infinitely many solutions. Hence the number of solutions is never exactly two. QED [1]",
    "Suppose Ax = b has two solutions x and y. Then Ax = b and Ay = b. So A(x+y)/2 = b, hence (x+y)/2 is a third solution.",
    "Let x and y be distinct solutions. The midpoint (x + y)/2 also solves it, so exactly two is impossible.",
    "If x and y both solve Ax = b, then so does (x+y)/2. Hence there are infinitely many solutions.",
    "Suppose Ax = b has two solutions x and y. Then A(x + y)/2 = b. The midpoint is a third solution, so exactly two is impossible. ∎",
    "Suppose Ax = b has two solutions x and y. Then A(x + y)/2 = b. The midpoint is a third solution. This completes the proof.",
]
PROOF_OK = [
    "To prove a statement like this, start by assuming the hypothesis and look for a construction that gives a third solution.",
    "You could suppose the system has two solutions and try to build another one from them. Ask yourself what combination of two "
    "solutions still solves Ax = b.",
    "Suppose a system has one solution. Then the matrix is invertible, so the solution is unique.",
    "If a system Ax = b has a unique solution, then A is invertible; otherwise there are infinitely many solutions or none [1].",
    "Linear systems have either no solution, one solution, or infinitely many solutions [1].",
]


def _p4(s, ids):
    return dict(s.graded_item(ids["Problem 4"]), kind="proof", source="item", score=1.0, reason="sticky")


@pytest.mark.parametrize("reply", PROOFS)
def test_d5_finished_short_proof_is_a_hit_with_or_without_an_end_marker(reply):
    s, c, ids = world()
    hit = integrity.check_reply(s, c, "Prove that a linear system never has exactly two solutions", reply, force_item=_p4(s, ids))
    assert hit and hit["id"] == ids["Problem 4"]
    cold = integrity.check_reply(s, c, "why not exactly two solutions?", reply)
    assert cold and cold["id"] == ids["Problem 4"]


@pytest.mark.parametrize("reply", PROOF_OK)
def test_d5_method_sentences_and_other_facts_pass(reply):
    s, c, ids = world()
    assert integrity.check_reply(s, c, "How do I start problem 4?", reply, force_item=_p4(s, ids)) is None


def test_d5_mcp_tools_refuse_the_two_sentence_midpoint_proof_and_save_the_method():
    s, c, ids = world()
    mcp_server.set_store(s)
    try:
        assert mcp_server.save_guide(c, "explain", "Notes", PROOFS[3], []).startswith("NOT SAVED")
        assert mcp_server.save_guide(c, "explain", "Notes", PROOFS[1], []).startswith("NOT SAVED")
        assert mcp_server.save_guide(c, "explain", "Notes", PROOF_OK[0], []).startswith("Saved")
        assert mcp_server.save_guide(c, "explain", "Notes", PROOF_OK[4], []).startswith("Saved")
    finally:
        mcp_server.set_store(None)
