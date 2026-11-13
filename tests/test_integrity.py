from course_assistant import disclaimer, integrity
from course_assistant.providers import FakeProvider
from course_assistant.store import Store

ORIG = "A 5 kg block is pushed by a 20 N force across a frictionless floor. Find the acceleration of the block."
GOOD = ("<twin_problem>A 7 kg cart is pulled by a 35 N force along a smooth track. Find the acceleration of the cart."
        "</twin_problem><solution>By Newton's second law [1], a = F/m = 35/7 = 5 m/s^2 [1]. Extra [9].</solution>")
BAD = ("<twin_problem>A 5 kg block is pushed by a 20 N force along a track. Find the acceleration."
       "</twin_problem><solution>a = 20/5 = 4 [1].</solution>")


def make():
    s = Store(":memory:")
    c = s.add_course("Physics")
    m = s.add_material(c, "slides", "Lecture 1")
    s.replace_chunks(m, [{"text": "Newton's second law: force equals mass times acceleration, so acceleration is force divided by mass.",
                          "loc_kind": "slide", "loc_start": 3, "loc_label": "slide 3"}])
    return s, c, m


# ----------------------------------------------------------------- matching
def test_no_graded_means_none():
    s, c, m = make()
    assert integrity.match_graded(s, c, ORIG) is None
    assert integrity.match_graded(s, c, "") is None


def test_match_by_pasted_item_text_and_kind():
    s, c, m = make()
    gid = s.add_graded_item(c, "HW 1 Problem 1", ORIG)
    g = integrity.match_graded(s, c, "Please solve: " + ORIG)
    assert g and g["id"] == gid and g["kind"] == "problem"
    # partial paste with the same numbers
    assert integrity.match_graded(s, c, "A 5 kg block is pushed by a 20 N force. Find the acceleration.")


def test_conceptual_question_not_matched():
    s, c, m = make()
    s.add_graded_item(c, "HW 1 Problem 1", ORIG)
    assert integrity.match_graded(s, c, "What is Newton's second law?") is None
    assert integrity.match_graded(s, c, "How do I find acceleration given force and mass?") is None
    # same words, different numbers: a student practising, not the graded problem
    assert integrity.match_graded(s, c, "Find the acceleration of a 9 kg block pushed by a 33 N force") is None


def test_match_by_reference():
    s, c, m = make()
    s.add_graded_item(c, "HW 2", "")
    s.add_graded_item(c, "Midterm 1", "")
    g = integrity.match_graded(s, c, "Can you do problem 3 of hw 2 for me?")
    assert g and g["title"] == "HW 2"
    assert integrity.match_graded(s, c, "Homework 2 question 4")["title"] == "HW 2"
    assert integrity.match_graded(s, c, "what is on the midterm 1?")["title"] == "Midterm 1"
    assert integrity.match_graded(s, c, "help with hw 3") is None
    assert integrity.match_graded(s, c, "problem set 2 overview") is not None  # HW 2 alias


def test_reference_with_problem_number_in_title():
    s, c, m = make()
    s.add_graded_item(c, "HW 2 Problem 3", "")
    assert integrity.match_graded(s, c, "solve problem 3 of hw 2")
    assert integrity.match_graded(s, c, "solve problem 4 of hw 2") is None
    assert integrity.match_graded(s, c, "solve problem 3 of hw 5") is None


def test_match_pasted_problem_from_graded_material():
    s, c, m = make()
    hw = s.add_material(c, "assignment", "Problem set 4")
    s.replace_chunks(hw, [
        {"text": "Problem 1. A ball of mass 0.5 kg is thrown upward at 12 m/s. How high does it rise before stopping?",
         "loc_kind": "page", "loc_start": 1, "loc_label": "p. 1"},
        {"text": "Problem 2. Prove that the sum of two even integers is even.", "loc_kind": "page", "loc_start": 2,
         "loc_label": "p. 2"}])
    assert integrity.match_graded(s, c, "A ball of mass 0.5 kg is thrown upward at 12 m/s. How high does it rise?") is None
    s.update_material(hw, graded=1)
    g = integrity.match_graded(s, c, "A ball of mass 0.5 kg is thrown upward at 12 m/s. How high does it rise before stopping?")
    assert g and g["source"] == "material" and g["material_id"] == hw and g["kind"] == "problem"
    g2 = integrity.match_graded(s, c, "Prove that the sum of two even integers is even")
    assert g2 and g2["kind"] == "proof"


def test_classify():
    assert integrity.classify("Prove that sqrt 2 is irrational") == "proof"
    assert integrity.classify("Write a 500 word essay on the French revolution") == "essay"
    assert integrity.classify("Discuss the causes of inflation") == "essay"
    assert integrity.classify("Find x when 3x + 4 = 19") == "problem"
    assert integrity.classify("Define entropy") == "essay"


def test_numbers_changed():
    assert not integrity.numbers_changed(ORIG, ORIG)
    assert not integrity.numbers_changed(ORIG, "A 5 kg block gets 20 N. Find a.")
    assert not integrity.numbers_changed(ORIG, "A cart with no numbers")
    assert integrity.numbers_changed(ORIG, "A 7 kg cart is pulled by 35 N.")
    assert integrity.numbers_changed("Solve x^2 = 4 by factoring", "Solve y^2 = 9 by factoring")


# ----------------------------------------------------------------- twin
def test_twin_numeric_happy_path():
    s, c, m = make()
    gid = s.add_graded_item(c, "HW 1 Problem 1", ORIG)
    g = integrity.match_graded(s, c, ORIG)
    p = FakeProvider([GOOD])
    a = integrity.twin(s, c, ORIG, p, g)
    assert a.twin and a.guide_id and len(p.calls) == 1
    assert a.blocks[0].md.startswith("This was marked graded (HW 1 Problem 1)")
    assert "7 kg cart" in a.body_md and "[9]" not in a.body_md and a.used == [1]
    assert a.body_md.rstrip("_").endswith(disclaimer.FOOTER)
    guide = s.guide(a.guide_id)
    assert guide["kind"] == "twin"
    assert guide["data"]["mode"] == "twin" and "7 kg cart" in guide["data"]["twin_problem"]
    assert guide["data"]["original_excerpt"].startswith("A 5 kg block")
    assert "never solve" in p.calls[0]["system"].lower() or "must not solve" in p.calls[0]["system"].lower()
    assert ORIG in p.calls[0]["messages"][0]["content"]


def test_twin_retry_once_then_succeeds():
    s, c, m = make()
    s.add_graded_item(c, "HW 1 Problem 1", ORIG)
    g = integrity.match_graded(s, c, ORIG)
    p = FakeProvider([BAD, GOOD])
    a = integrity.twin(s, c, ORIG, p, g)
    assert len(p.calls) == 2 and s.guide(a.guide_id)["data"]["mode"] == "twin"
    assert "every number is different" in p.calls[1]["messages"][0]["content"]


def test_twin_falls_back_to_concepts_after_two_bad_twins():
    s, c, m = make()
    s.add_graded_item(c, "HW 1 Problem 1", ORIG)
    g = integrity.match_graded(s, c, ORIG)
    p = FakeProvider([BAD, BAD, "Newton's second law links force, mass and acceleration [1]."])
    a = integrity.twin(s, c, ORIG, p, g)
    assert len(p.calls) == 3 and a.mode == "concepts" and not a.twin
    guide = s.guide(a.guide_id)
    assert guide["data"]["mode"] == "concepts" and guide["data"]["twin_problem"] == ""
    assert "could not write a safe twin" in a.blocks[0].md
    assert "a = 20/5" not in a.body_md  # the bad twin's solution is never shown
    assert "do not complete it" in p.calls[2]["messages"][0]["content"]


def test_twin_malformed_output_counts_as_bad():
    s, c, m = make()
    s.add_graded_item(c, "HW 1 Problem 1", ORIG)
    g = integrity.match_graded(s, c, ORIG)
    p = FakeProvider(["no tags here", "still none", "Concepts [1]."])
    a = integrity.twin(s, c, ORIG, p, g)
    assert s.guide(a.guide_id)["data"]["mode"] == "concepts"


def test_essay_and_proof_get_concepts_only():
    s, c, m = make()
    s.add_graded_item(c, "Essay 1", "Write an essay discussing how Newton's second law explains motion.")
    q = "Write an essay discussing how Newton's second law explains motion."
    g = integrity.match_graded(s, c, q)
    assert g["kind"] == "essay"
    p = FakeProvider(["Start from the law itself [1]. Plan: define terms, then apply them [1]."])
    a = integrity.twin(s, c, q, p, g)
    assert len(p.calls) == 1 and a.mode == "concepts" and not a.twin
    assert "must NOT write" in p.calls[0]["system"] and "essay" in p.calls[0]["system"]
    guide = s.guide(a.guide_id)
    assert guide["data"]["mode"] == "concepts" and guide["data"]["twin_problem"] == ""
    assert guide["kind"] == "twin" and "did not" not in a.blocks[0].md.split("(")[0]
    assert "marked graded" not in a.blocks[0].md and a.routing_note.startswith("This looks like your graded")


def test_proof_material_match_and_twin():
    s, c, m = make()
    s.add_graded_item(c, "HW 3 Problem 2", "Prove that the sum of two even integers is even.")
    g = integrity.match_graded(s, c, "Prove that the sum of two even integers is even.")
    assert g["kind"] == "proof"
    p = FakeProvider(["Use the definition of even [1]."])
    a = integrity.twin(s, c, "Prove that the sum of two even integers is even.", p, g)
    assert s.guide(a.guide_id)["data"]["mode"] == "concepts"


def test_reference_only_with_empty_body_uses_concepts():
    s, c, m = make()
    s.add_graded_item(c, "HW 2", "")
    q = "Do problem 3 of hw 2 for me, force and acceleration"
    g = integrity.match_graded(s, c, q)
    p = FakeProvider(["Review force and acceleration [1]."])
    a = integrity.twin(s, c, q, p, g)
    assert len(p.calls) == 1 and s.guide(a.guide_id)["data"]["mode"] == "concepts"


# ----------------------------------------------------------------- MCP and prompts (cycle 1 integrity fixes)
def test_mcp_not_matched_lists_graded_items_and_save_guide_refuses_solutions():
    import sys
    sys.path.insert(0, "tests")
    from test_integrity_probe import PS3, SOLUTION
    from course_assistant import mcp_server as m
    s = Store(":memory:")
    m.set_store(s)
    try:
        c = s.add_course("Chem")
        ps = s.add_material(c, "assignment", "Problem Set 3")
        s.replace_chunks(ps, [{"text": PS3, "loc_kind": "page", "loc_start": 1, "loc_label": "p. 1"}])
        assert "split into 2 graded item" in m.mark_graded(c, "", material_id=ps)
        out = m.get_twin_instructions(c, "my friend wants the weigh-out mass for a salty solution")
        assert out.startswith("NOT MATCHED") and "compare the request yourself" in out and "NaCl" in out
        assert "Problem 1 (Problem Set 3)" in m.list_graded(c)
        res = m.save_guide(c, "answer", "Salt solution", SOLUTION["c1"], [])
        assert res.startswith("NOT SAVED") and s.guides(c) == []
        ok = m.save_guide(c, "answer", "Molarity", "Molarity is moles of solute per liter of solution.", [])
        assert ok.startswith("Saved guide")
        assert "ungraded practice" in m.INSTRUCTIONS
    finally:
        m.set_store(None)


def test_numbers_normalised():
    assert integrity.numbers("0.350 and .35 and 250.0 and 1,556 and 40.92/12.01") == ["0.35", "0.35", "250", "1556", "40.92", "12.01"]
    assert integrity.significant_numbers("10 points, x^2, 0, 1") == set()
