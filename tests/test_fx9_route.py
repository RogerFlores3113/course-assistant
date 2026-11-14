"""Cycle 9 routing and context: each panel repro next to its honest look-alike. Fixtures are original (fx9_world)."""
import pytest

from course_assistant import grounding, integrity, mcp_server
from course_assistant.providers import FakeProvider
from fx9_world import ESS, world

TWIN_OK = ("<twin_problem>Solve 3x + y + z = 7, x - y + 2z = 4, 2x + 3y - z = 4 for (x, y, z).</twin_problem>"
           "<solution>Eliminate x from rows 2 and 3, then back substitute. Apply the same steps to your own problem.</solution>")


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


def touched(s, c, ids):
    """The chat touched Problem 1 (its system pasted), so replies are checked against it."""
    p1 = s.graded_item(ids["Problem 1"])
    return dict(p1, kind="problem", source="item", score=1.0, reason="sticky")


# ------------------------------------------------------------------ decision 1: unnamed key vector in a touched chat
VEC_FORMS = ["The solution vector is (1, 1, 2).", "The solution vector is [1, 1, 2].", "So we get 1, 1, 2.",
             "The answer is 1 1 2.", "Written as a column: [1; 1; 2].", "(x1, x2, x3) = (1, 1, 2)", "It comes out to <1,1,2>.",
             "The solution is the vector (1, 1, 2) [1]."]


@pytest.mark.parametrize("reply", VEC_FORMS)
def test_d1_touched_chat_bare_key_vector_is_a_hit(reply):
    s, c, ids = world()
    hit = integrity.check_reply(s, c, "what was x again?", reply, force_item=touched(s, c, ids))
    assert hit and hit["id"] == ids["Problem 1"] and hit["reason"] == "key"


@pytest.mark.parametrize("reply", ["The rank of [[1, 2], [2, 4]] is 1.", "A vector such as (1, 2, 3) has three entries.",
                                   "For the vector (2, 1, 1) the length is the square root of 6.",
                                   "You have three equations and three unknowns, so expect one solution."])
def test_d1_touched_chat_look_alikes_pass(reply):
    s, c, ids = world()
    assert integrity.check_reply(s, c, "what was x again?", reply, force_item=touched(s, c, ids)) is None


def test_d1_the_students_own_vector_echoed_back_is_not_a_leak():
    s, c, ids = world()
    q = "is the vector (1, 1, 2) a unit vector? what is its length?"
    assert integrity.check_reply(s, c, q, "The vector (1, 1, 2) has length the square root of 6.",
                                 force_item=touched(s, c, ids)) is None


def test_d1_an_untouched_chat_keeps_the_small_integer_rule():
    s, c, ids = world()
    assert integrity.check_reply(s, c, "what does a direction vector look like?",
                                 "A direction vector can be written as (1, 1, 2) or [1; 1; 2].") is None


def test_d1_named_item_in_the_question_counts_as_context():
    s, c, ids = world()
    hit = integrity.check_reply(s, c, "what is the solution to problem 1 of problem set 4?", "The solution is (1, 1, 2).")
    assert hit and hit["id"] == ids["Problem 1"]


def test_d1_ask_drops_the_bare_vector_while_sticky_and_answers_the_look_alike():
    s, c, ids = world()
    p = FakeProvider(model({"what was x": "The solution vector is (1, 1, 2).", "unit": "A unit vector has length one."}))
    grounding.ask(s, c, "Problem 1 of problem set 4: 2x + y + z = 5, x - y + 3z = 6, 3x + 2y - z = 3. How do I start?", p)
    a = grounding.ask(s, c, "what was x again?", p)
    assert held(a) and "(1, 1, 2)" not in a.body_md
    assert grounding.ask(s, c, "what makes a vector a unit vector?", p).mode == "answer"


def test_d1_save_guide_and_flashcards_refuse_the_bare_vector_when_the_text_names_a_solution():
    s, c, ids = world()
    mcp_server.set_store(s)
    try:
        for body in ("The solution is (1, 1, 2).", "The answer vector is 1, 1, 2.", "x is the vector [1; 1; 2]."):
            assert mcp_server.save_guide(c, "explain", "Notes", body, []).startswith("NOT SAVED"), body
        for f, b in (("Problem 1 answer vector", "(1, 1, 2)"), ("Problem 1 answer vector", "[1; 1; 2]"),
                     ("Problem 1 answer vector", "1 1 2"), ("Problem 1 answer vector", "<1, 1, 2>"),
                     ("x for the first problem", "one, one, two"), ("Solution of problem 1", "x = (1, 1, 2)")):
            assert mcp_server.add_flashcards(c, [{"front": f, "back": b}]).startswith("NOT ADDED"), (f, b)
        # look-alikes: other vectors, a plain vector definition, a direction vector with no solution wording
        assert mcp_server.save_guide(c, "explain", "Notes", "A vector with three entries, such as (2, 5, 4), lives in R^3.", []).startswith("Saved")
        assert mcp_server.save_guide(c, "explain", "Notes", "The solution is (4, 9, 3) for a different system.", []).startswith("Saved")
        assert mcp_server.add_flashcards(c, [{"front": "What is a unit vector?", "back": "A vector of length one"}]).startswith("Added")
        assert mcp_server.add_flashcards(c, [{"front": "Solution of Ax = 0 with a free variable?", "back": "A line through the origin"}]).startswith("Added")
    finally:
        mcp_server.set_store(None)


# ------------------------------------------------------------------ decision 4: d = 11 and a short proof in the MCP paths
@pytest.mark.parametrize("body", ["d = 11 makes the system singular", "d=11", "The singular value d = 11.", "d is eleven"])
def test_d4_two_digit_answer_bound_to_its_name_is_refused(body):
    s, c, ids = world()
    mcp_server.set_store(s)
    try:
        assert mcp_server.save_guide(c, "explain", "Notes", body, []).startswith("NOT SAVED"), body
        assert mcp_server.add_flashcards(c, [{"front": "Singular d in problem 2", "back": body}]).startswith("NOT ADDED"), body
    finally:
        mcp_server.set_store(None)


@pytest.mark.parametrize("body", ["Lectures 10 and 11 cover pivots and row exchanges.", "The review is at 11:59 pm on the 11th.",
                                  "Office hours moved to Room 11 this week.", "Section 11 of the reading explains determinants."])
def test_d4_look_alikes_with_the_number_eleven_are_saved(body):
    s, c, ids = world()
    mcp_server.set_store(s)
    try:
        assert mcp_server.save_guide(c, "explain", "Notes", body, []).startswith("Saved"), body
    finally:
        mcp_server.set_store(None)


SHORT_PROOF = ("Suppose x and y are two different solutions. Then Ax = b and Ay = b. So A(x + y) = 2b, which means (x + y)/2 "
               "is also a solution. Since it differs from both, there are at least three solutions.")


def test_d4_three_sentence_proof_is_caught_in_ask_guide_and_card():
    s, c, ids = world()
    hit = integrity.check_reply(s, c, "why does a system with two solutions have infinitely many?", SHORT_PROOF)
    assert hit and hit["id"] == ids["Problem 4"]
    mcp_server.set_store(s)
    try:
        assert mcp_server.save_guide(c, "explain", "Notes", SHORT_PROOF, []).startswith("NOT SAVED")
        assert mcp_server.add_flashcards(c, [{"front": "Two solutions?", "back": SHORT_PROOF}]).startswith("NOT ADDED")
    finally:
        mcp_server.set_store(None)
    a = grounding.ask(s, c, "why does a system with two solutions have infinitely many?",
                      FakeProvider(model(default=SHORT_PROOF)))
    assert held(a) and "2b" not in a.body_md


@pytest.mark.parametrize("body", [
    "Suppose a system has two solutions. Then their difference solves the homogeneous system. Therefore you can add multiples of it to get more solutions.",
    "Suppose you list the pivots. Then count the free variables. So the nullity equals the number of free variables.",
    "To prove a statement about solutions, suppose the opposite, then follow the consequences until you reach a contradiction."])
def test_d4_look_alike_method_sentences_are_not_a_proof(body):
    s, c, ids = world()
    assert integrity.check_reply(s, c, "how do proofs about solutions usually go?", body) is None
    mcp_server.set_store(s)
    try:
        assert mcp_server.save_guide(c, "explain", "Notes", body, []).startswith("Saved")
    finally:
        mcp_server.set_store(None)


# ------------------------------------------------------------------ decision 5: routing precision
@pytest.mark.parametrize("q", ["Problem Set 4 is due Thursday, right?", "Problem 2 is worth 15 points, right?",
                               "Problem Set 4 is due Thursday at 11:59, correct?", "Problem 2 isn't due yet, right?",
                               "The late policy is 10 percent a day, right?"])
def test_d5_logistics_ending_in_right_is_not_a_check_ask(q):
    assert not integrity.is_check_ask(q)
    s, c, ids = world()
    r = integrity.match_graded(s, c, q)
    assert r is None or not r.get("by_check")
    a = grounding.ask(s, c, q, FakeProvider(model(default="Yes, due Thursday.")))
    assert a.mode == "answer" and not getattr(a, "routing_note", "")


@pytest.mark.parametrize("q", ["I got x = (1, 1, 2) for problem 1, right?", "Is x = 3/2 correct?", "I got d = 11. Is that correct?",
                               "My answer to Problem 3 is C5H10O3, right?", "Please check my answer: z = 2"])
def test_d5_a_stated_value_with_right_is_still_a_check_ask(q):
    assert integrity.is_check_ask(q)


def test_d5_explicit_problem_number_beats_topic_overlap():
    s, c, ids = world()
    q = ("How should I structure my answer to Problem 5 on the empirical vs molecular formula? What should I include? "
         "I think the ratio is 1:2, right?")
    r = integrity.match_graded(s, c, q)
    assert r is None or r.get("id") == ids["Problem 5"]
    # the look-alike: with no problem number the topic still finds the item for a real check ask
    r = integrity.match_graded(s, c, "I got C5H10O3 for the ester empirical formula problem, is it right?")
    assert r and r.get("id") == ids["Problem 3"]


def test_d5_pasted_essay_intro_routes_to_the_essay_not_the_proof():
    s, c, ids = world()
    q = ("Here is my draft intro for my essay: 'Two different solutions along the canal show that a town has infinitely many "
         "ways to grow, and I will show this with tolls and with freight.' Can you give me feedback?")
    r = integrity.match_graded(s, c, q)
    assert r and r["id"] == ids["Essay 1"] and r.get("intent") == "feedback"
    # the look-alike: a pasted proof draft that names the proof item goes to the proof
    q2 = ("Here is my draft for Problem 4: Suppose x and y are two different solutions of Ax = b. Then A(x + y)/2 = b, so there "
          "are infinitely many. Can you give me feedback?")
    r = integrity.match_graded(s, c, q2)
    assert r and r["id"] == ids["Problem 4"] and r.get("intent") == "feedback"
    # and a proof draft with proof markers but no number still reaches the proof
    q3 = ("Can you give feedback on my proof? Suppose x and y solve Ax = b. Then A(x + y)/2 = b, "
          "so every point between them is a solution. QED.")
    r = integrity.match_graded(s, c, q3)
    assert r and r["id"] == ids["Problem 4"]


# ------------------------------------------------------------------ decision 5: graded handouts are searched outside their questions
def test_d5_graded_handout_equipment_and_safety_are_found_but_its_questions_are_not(tmp_path):
    from fx9_world import handout_world
    s, c, ids, mid = handout_world(tmp_path)

    def f(system, messages):
        return "You need two 50 mL burets, a ring stand and three 125 mL flasks [1]."
    a = grounding.ask(s, c, "What equipment do I need for the vinegar titration?", FakeProvider(f))
    assert a.mode == "answer" and any("125 mL flasks" in (x.get("text") or "") for x in a.sources)
    assert all("31.20" not in (x.get("text") or "") and "Answer in two sentences" not in (x.get("text") or "") for x in a.sources)
    assert not a.not_covered
    # the safety paragraph beside a question is found too
    a = grounding.ask(s, c, "What safety rules apply when I use sodium hydroxide?",
                      FakeProvider(model(default="Wear goggles at all times [1].")))
    assert any("goggles" in (x.get("text") or "") for x in a.sources)


def test_d5_a_hit_inside_a_graded_question_is_left_out_and_said(tmp_path):
    import fx11_world  # cycle 11: the note never sits beside an "Also searched" row of the same handout, so the questions are alone here
    s, c, mid = fx11_world.handout_only_questions(tmp_path)
    a = grounding.ask(s, c, "How do I compute the molarity of a brine sample and what happens to conductivity when it is diluted?",
                      FakeProvider(model(default="The sources do not cover it.")))
    assert all("Compute its molarity" not in (x.get("text") or "") for x in a.sources)
    assert "Part of your graded handout was not searched" in a.body_md


def test_d5_the_graded_question_pasted_still_goes_to_the_graded_route(tmp_path):
    from fx9_world import handout_world
    s, c, ids, mid = handout_world(tmp_path)
    q = ("A 25.00 mL sample of vinegar needs 31.20 mL of 0.1000 M sodium hydroxide to reach the endpoint. What is the molarity "
         "of the acetic acid in the vinegar?")
    a = grounding.ask(s, c, q, FakeProvider(model()))
    assert held(a)
    assert all("31.20" not in (x.get("text") or "") for x in a.sources)


def test_d5_no_graded_handout_no_note():
    s, c, ids = world()
    a = grounding.ask(s, c, "What equipment do I need for the lab?", FakeProvider(model(default="The materials do not cover this.")))
    assert "graded handout" not in a.body_md
