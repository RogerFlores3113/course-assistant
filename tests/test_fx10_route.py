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


# ------------------------------------------------------------------ decision 6: equation-line systems
import fx10_world as W10


@pytest.fixture
def mcp10(tmp_path):
    w = W10.world(tmp_path, lab=True)
    mcp_server.set_store(w["s"])
    yield w
    mcp_server.set_store(None)


EQ_LEAKS = ["3x + 4y + z = 15 / y + z = 3 / 3z = 6", "3x + 4y + z = 15\ny + z = 3\n3z = 6",
            "After elimination the system becomes 3x+4y+z=15, y+z=3, 3z=6."]


@pytest.mark.parametrize("text", EQ_LEAKS)
def test_d6_triangular_system_as_equation_lines_names_its_item(text):
    w = W10.world()
    s, c, ids = w["s"], w["c"], w["ids"]
    hit = integrity.check_reply(s, c, "what does the system look like after the exchange?", text)
    assert hit and hit["id"] == ids["Problem 6"]
    assert integrity.check_guide(s, c, "Notes", text)["id"] == ids["Problem 6"]


@pytest.mark.parametrize("text", EQ_LEAKS)
def test_d6_mcp_refuses_the_equation_lines(mcp10, text):
    r = mcp_server.save_guide(mcp10["c"], "explain", "Notes", text, [])
    assert r.startswith("NOT SAVED") and "Problem 6" in r


@pytest.mark.parametrize("text", ["5x + 2y + z = 9 / 3y - z = 4 / 2z = 6", "2x + y = 5\n3y = 6",
                                  "Back substitution solves the last equation first: from 4z = 8 you get z = 2.",
                                  "x = 4, y = 5, z = 9"])
def test_d6_look_alike_systems_pass(text):
    w = W10.world()
    assert integrity.check_reply(w["s"], w["c"], "how does back substitution work?", text) is None
    assert integrity.check_guide(w["s"], w["c"], "Notes", text) is None


def test_d6_equation_system_reader():
    assert integrity.equation_system("2x + 5y + z = 0 / y - z = 3 / -z = 2") == [["2", "5", "1", "0"], ["0", "1", "-1", "3"], ["0", "0", "-1", "2"]]
    assert integrity.equation_system("x + y = 3, x - y = 1") == []  # not triangular
    assert integrity.equation_system("x = 1, y = 1, z = 2") == []  # a list of values, no equation with two unknowns


# ------------------------------------------------------------------ decision 7: keyless items, data, wrong item
def _chunk(w, which):
    return [w["book_chunks"][0]] if which == "example" else [w["handout_chunks"][0]]


def test_d7_textbook_worked_example_cited_to_its_chunk_is_data_in_a_guide(mcp10):
    w = mcp10
    body = "The textbook example: a 0.250 L broth sample holds 0.0875 mol of urea, so the molarity is 0.350 M [1]"
    assert mcp_server.save_guide(w["c"], "explain", "Notes", body, _chunk(w, "example")).startswith("Saved")


def test_d7_same_numbers_with_no_cited_chunk_that_prints_them_are_checked(mcp10):
    w = mcp10
    body = "You need 0.0125 mol, which is 3.12 g of copper sulfate pentahydrate [1]"
    r = mcp_server.save_guide(w["c"], "explain", "Notes", body, _chunk(w, "example"))
    assert r.startswith("NOT SAVED") and "Pre-lab" in r


def test_d7_a_chunk_of_graded_work_never_makes_a_sentence_data(mcp10):
    w = mcp10
    body = "Prepare 250.0 mL of 0.0500 M, which needs 3.12 g [1]"
    assert mcp_server.save_guide(w["c"], "explain", "Notes", body, _chunk(w, "handout")).startswith("NOT SAVED")


def test_d7_textbook_example_card_with_its_chunk_is_saved_and_the_item_answer_is_not(mcp10):
    w = mcp10
    ok = {"front": "Molarity of 0.0875 mol urea in 0.250 L?", "back": "0.350 M", "chunk_id": w["book_chunks"][0]}
    assert mcp_server.add_flashcards(w["c"], [ok]).startswith("Added")
    bad = {"front": "Grams of copper sulfate pentahydrate for the pre-lab", "back": "3.12 g"}
    r = mcp_server.add_flashcards(w["c"], [bad])
    assert r.startswith("NOT ADDED") and "Pre-lab" in r


def test_d7_data_constants_the_handout_prints_are_free_in_cards_and_guides(mcp10):
    w = mcp10
    assert mcp_server.add_flashcards(w["c"], [{"front": "Molar mass of CuSO4 5H2O?", "back": "249.68 g/mol"}]).startswith("Added")
    assert mcp_server.save_guide(w["c"], "explain", "Notes", "The molar mass of copper sulfate pentahydrate is 249.68 g/mol.", []).startswith("Saved")
    # the key's own answer is still refused
    assert mcp_server.add_flashcards(w["c"], [{"front": "Moles of copper sulfate needed in the pre-lab", "back": "0.0125 mol"}]).startswith("NOT ADDED")


def test_d7_a_keyless_item_is_not_named_on_a_bare_molarity_sentence(mcp10):
    w = mcp10
    for body in ("For example, 0.0875 mol in 250 mL is 0.350 M", "A 0.010 M solution has 0.010 mol of solute per liter, ten times weaker than 0.10 M",
                 "2.5 g of NaCl is about 0.043 mol"):
        assert mcp_server.save_guide(w["c"], "explain", "Notes", body, []).startswith("Saved"), body


def test_d7_a_keyless_item_is_still_refused_when_the_text_is_about_it_and_names_the_right_item(mcp10):
    w = mcp10
    r = mcp_server.save_guide(w["c"], "answer", "Notes", "Plot absorbance against concentration: the slope is 0.742.", [])
    assert r.startswith("NOT SAVED") and "Report" in r and "Pre-lab" not in r


def test_d7_save_guide_refuses_unnamed_multipliers_like_flashcards(mcp10):
    w = mcp10
    assert mcp_server.save_guide(w["c"], "explain", "Notes", "The multipliers are 2, -1, 6. [1]", []).startswith("NOT SAVED")
    assert mcp_server.add_flashcards(w["c"], [{"front": "Multipliers of problem 6", "back": "2, -1, 6"}]).startswith("NOT ADDED")
    for ok in ("The multipliers are the numbers that clear each entry below a pivot.", "For another system the multipliers are 4, 7 and 3."):
        assert mcp_server.save_guide(w["c"], "explain", "Notes", ok, []).startswith("Saved"), ok


def test_d7_cited_textbook_sentence_is_not_refused_as_a_keyless_item(tmp_path):
    w = W10.world(tmp_path, lab=True, prelab_key=False)
    mcp_server.set_store(w["s"])
    try:
        for body in ("For example, 0.0875 mol in 250 mL is 0.350 M [1]",
                     "The textbook example: a 0.250 L broth sample holds 0.0875 mol of urea, so the molarity is 0.0875 mol divided by 0.250 L, which is 0.350 M [1]"):
            assert mcp_server.save_guide(w["c"], "explain", "Notes", body, [w["book_chunks"][0]]).startswith("Saved"), body
        # a number the chunk does not print is still judged: the pre-lab answer is a result, not an example
        r = mcp_server.save_guide(w["c"], "explain", "Notes", "Pre-lab 1: you need 3.12 g of copper sulfate pentahydrate [1]", [w["book_chunks"][0]])
        assert r.startswith("NOT SAVED") and "Pre-lab" in r
    finally:
        mcp_server.set_store(None)


def test_cite_comma_marker_is_stripped_and_a_vector_is_not():
    f = integrity._CITE_SOFT.sub
    assert "[1]" not in f("", "The syllabus lists Monday 6 to 7 pm [1], but gives no room.")
    assert f("", "so x = [1, 1, 2] here") == "so x = [1, 1, 2] here"
    assert f("", "A = [[1], [1], [2]]") == "A = [[1], [1], [2]]"
