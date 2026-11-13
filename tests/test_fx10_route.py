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
