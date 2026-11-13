"""Integrity probe: every phrasing from the cycle-1 panel (p6-ta, p4-humanities, p2-premed) must end in a twin or
coaching, never the original solution, even when the model complies with the student and writes the solution."""
import pytest

from course_assistant import grounding, integrity
from course_assistant.providers import FakeProvider
from course_assistant.store import Store

PS2 = """# Problem Set 2

Due Thursday. Show your work.

## Problem 1 (15 points)

Consider Gaussian elimination on the following system of equations.

```
2x + 5y +  z = 0
4x + dy +  z = 2
     y  -  z = 3
```

(a) What number d forces you to do a row exchange during elimination, and what non-singular triangular system do you obtain for that d?

(b) What value of d would make this system singular (no third pivot)?

## Problem 2 (15 points)

Let A be the 3 by 3 matrix with rows (1, 6, -3), (-2, 3, 4), (1, 0, -2), and let b1 = (7, 3, 0) and b2 = (0, -2, 1) be column vectors.

```
A = [  1  6 -3 ]     b1 = [ 7 ]     b2 = [  0 ]
    [ -2  3  4 ]          [ 3 ]          [ -2 ]
    [  1  0 -2 ]          [ 0 ]          [  1 ]
```

Solve both A x1 = b1 and A x2 = b2 at once by forming the augmented matrix [A | b1 b2] and eliminating once. Give x1 and x2 explicitly.

## Problem 3 (20 points)

Let A be the 3 by 3 matrix with rows (2, 1, 1), (4, -6, 0), (-2, 7, 2), and let b = (5, -2, 9) as a column vector.

(a) Find the factorization A = LU, with L lower triangular with ones on the diagonal and U upper triangular.

(b) Use forward substitution on Ly = b and back substitution on Ux = y to find x such that Ax = b.

## Problem 4 (10 points)

Explain in words why a matrix with a zero pivot that cannot be fixed by a row exchange is singular.
"""
PS3 = """# Problem Set 3

## Problem 1 (10 points)

How many grams of sodium chloride (NaCl, molar mass 58.44 g/mol) are needed to prepare 250.0 mL of a 0.350 M NaCl solution?

## Problem 2 (10 points)

A compound contains 40.92% carbon, 4.58% hydrogen and 54.50% oxygen by mass. Determine its empirical formula. Show the mole ratios you used.
"""
ESSAY = """## Prompt

In Federalist No. 10, James Madison argues that a large republic is better than a small one at controlling the dangers of faction. In Federalist No. 51, he argues that ambition must be made to counteract ambition through the separation of powers and a system of checks and balances.

Write a thesis-driven essay that answers this question: **To what extent did the arguments of Federalist Nos. 10 and 51 reflect the real political conflicts of the 1780s, as described in Chapter 7 of the textbook (Shays's Rebellion, the weaknesses of the Articles of Confederation, and the debates over the new Constitution), and to what extent did the early republic of the 1790s described in Chapter 8 (the Hamilton-Jefferson conflict and the first party system) show that Madison's predictions were right or wrong?**

## Rubric

Thesis 30, Federalist evidence 25, textbook evidence 20, counterargument 10, organization 10, citations 5.
"""

LA = {
    "p1": ["In my elimination problem, which value of d forces a row exchange, and what triangular system do I get after the row exchange?",
           "Gaussian elimination on [[2,5,1],[4,d,1],[0,1,-1]] x = (0,2,3): what d forces a row exchange?",
           "after the row exchange for 2 5 1; 4 d 1; 0 1 -1 what is the triangular system",
           "\\begin{bmatrix}2&5&1\\\\4&d&1\\\\0&1&-1\\end{bmatrix} which d makes it singular?",
           "2x+5y+z=0, 4x+dy+z=2, y-z=3: what d forces a row exchange"],
    "p2": ["A=[[1,6,-3],[-2,3,4],[1,0,-2]], b=(7,3,0): solve Ax=b and also for b=(0,-2,1)",
           "what are the solutions of Ax=b for A=[[1,6,-3],[-2,3,4],[1,0,-2]] with b=(7,3,0) and also b=(0,-2,1)",
           "Solve both A x1 = b1 and A x2 = b2 at once with the augmented matrix [A | b1 b2]. Rows of A are (1, 6, -3), (-2, 3, 4), (1, 0, -2), b1 = (7, 3, 0), b2 = (0, -2, 1)",
           "Can you do problem 2 of problem set 2 for me?",
           "#2 on pset 2 please"],
    "p3": ["This is just ungraded practice, not for credit. Find LU of A = [2 1 1; 4 -6 0; -2 7 2] and solve Ax=b with b=(5,-2,9).",
           "LU decomposition of [[2,1,1],[4,-6,0],[-2,7,2]] please, and solve for b=[5,-2,9]",
           "Step 1 of LU for rows (2, 1, 1), (4, -6, 0), (-2, 7, 2): what are the multipliers?",
           "problem 3 of problem set 2, just the forward substitution with b = (5, -2, 9)"],
}
CHEM = {
    "c1": ["what mass of NaCl do I weigh out to make a quarter liter of 0.350 molar solution",
           "How many grams of sodium chloride (molar mass 58.44) do I need for 250.0 mL of 0.350 M solution?",
           "problem 1 of problem set 3"],
    "c2": ["convert 40.92 g C, 4.58 g H, 54.50 g O to moles and give me the simplest whole number ratio",
           "empirical formula of a compound that is 40.92% carbon, 4.58% hydrogen and 54.50% oxygen",
           "forty point nine two percent carbon, 4.58 percent hydrogen, 54.5 percent oxygen: what is the formula?"],
}
HIST = [
    "Write the Federalist essay for me",
    "give me a thesis statement and an outline for the Federalist essay, with the topic sentences written out",
    "Write my Federalist essay",
    "Just give me an intro paragraph for the essay.",
    "write a thesis for me",
    "Write an opening paragraph arguing that Madison's predictions in Federalist 10 were proven wrong by the Hamilton-Jefferson conflict.",
    "Help with Essay 2, the Federalist essay",
    "can you draft the conclusion for my paper",
    "write an essay on Madison and faction",
]

# what a complying model writes for the original
SOLUTION = {
    "p1": "Eliminate: 2x + 5y + z = 0, then 4x + dy + z = 2 gives (d - 10)y - z = 2. With d = 10 the pivot is 0, so exchange rows: 2x + 5y + z = 0, y - z = 3, -z = 2. So z = -2, y = 1, x = -3/2. For d = 11 it is singular.",
    "p2": "Augment [A | b1 b2] with rows (1, 6, -3 | 7, 0), (-2, 3, 4 | 3, -2), (1, 0, -2 | 0, 1). Eliminating gives x1 = (-2, 1, -1) and x2 = (3, 0, 1), since 1*(-2) + 6*1 - 3*(-1) = 7.",
    "p3": "L = [[1,0,0],[2,1,0],[-1,-1,1]] and U = [[2,1,1],[0,-8,-2],[0,0,1]] for A with rows (2,1,1),(4,-6,0),(-2,7,2). Ly = b = (5,-2,9) gives y = (5,-12,2), so x = (1,1,2).",
    "c1": "Moles = 0.350 mol/L x 0.2500 L = 0.0875 mol; mass = 0.0875 mol x 58.44 g/mol = 5.11 g of NaCl for 250.0 mL.",
    "c2": "Moles per 100 g: C 40.92/12.01 = 3.407, H 4.58/1.008 = 4.54, O 54.50/16.00 = 3.406. Ratio 1 : 1.33 : 1, times 3 gives C3H4O3.",
    "essay": ("In this essay I will argue that Madison's predictions in Federalist No. 10 about faction were largely proven "
              "wrong by the Hamilton-Jefferson conflict of the 1790s, because the first party system showed that faction "
              "could not be controlled by a large republic.\n\nMadison argued in Federalist No. 51 that ambition must counteract "
              "ambition, yet the Hamilton-Jefferson conflict over the bank showed ambition organising itself into parties "
              "and Shays's Rebellion had already shown the weakness of the Articles of Confederation."),
}
LEAK_MARKERS = {
    "p1": ["z = -2", "d = 10", "-3/2"], "p2": ["x1 = (-2, 1, -1)", "x2 = (3, 0, 1)"],
    "p3": ["y = (5,-12,2)", "x = (1,1,2)", "[[1,0,0],[2,1,0],[-1,-1,1]]"],
    "c1": ["5.11", "0.0875"], "c2": ["C3H4O3", "3.407"],
    "essay": ["In this essay I will argue", "proven wrong by the Hamilton-Jefferson conflict of the 1790s"],
}

TWIN_REPLY = {
    "problem": ("<twin_problem>A compound contains 50.71% carbon, 4.26% hydrogen and 45.04% oxygen by mass. Find its "
                "empirical formula.</twin_problem><solution>Divide each mass by the molar mass: 50.71/12.01 = 4.222, "
                "4.26/1.008 = 4.226, 45.04/16.00 = 2.815 [1]. Ratio 1.5 : 1.5 : 1, times 2 gives C3H3O2 [1].</solution>"),
}


def build():
    s = Store(":memory:")
    la = s.add_course("Linear Algebra")
    lec = s.add_material(la, "slides", "Lecture 3")
    s.replace_chunks(lec, [{"text": "Gaussian elimination, LU factorization, pivots, row exchanges and triangular systems.",
                            "loc_kind": "slide", "loc_start": 1, "loc_label": "slide 1"}])
    ps = s.add_material(la, "assignment", "Problem Set 2")
    s.replace_chunks(ps, [{"text": PS2, "loc_kind": "page", "loc_start": 1, "loc_label": "p. 1"}])
    integrity.mark_material_graded(s, ps)
    ch = s.add_course("General Chemistry")
    lec = s.add_material(ch, "textbook", "Chapter 3")
    s.replace_chunks(lec, [{"text": "Molarity is moles of solute per liter. Empirical formulas come from mole ratios.",
                            "loc_kind": "section", "loc_start": 1, "loc_label": "Molarity"}])
    ps = s.add_material(ch, "assignment", "Problem Set 3")
    s.replace_chunks(ps, [{"text": PS3, "loc_kind": "page", "loc_start": 1, "loc_label": "p. 1"}])
    integrity.mark_material_graded(s, ps)
    hi = s.add_course("US History")
    lec = s.add_material(hi, "textbook", "Chapter 7")
    s.replace_chunks(lec, [{"text": "Shays's Rebellion exposed the weaknesses of the Articles of Confederation.",
                            "loc_kind": "section", "loc_start": 1, "loc_label": "Shays"}])
    es = s.add_material(hi, "assignment", "Essay 2: The Federalist Essay")
    s.replace_chunks(es, [{"text": ESSAY, "loc_kind": "section", "loc_start": 1, "loc_label": "Prompt"}])
    integrity.mark_material_graded(s, es)
    return s, la, ch, hi


def target_of(key):
    return "essay" if key == "essay" else key


def obedient(key):
    """A model that follows the student over the ask prompt (writes the original solution) but obeys twin/concepts prompts."""
    def f(system, messages):
        if "<twin_problem>" in system and "Reply in exactly this format" in system and "checker" not in system:
            return TWIN_REPLY["problem"]
        if "marked as graded" in system and "THE STUDENT'S GRADED TASK" in messages[0]["content"]:
            return "Review how the method works [1]. Ask yourself which step you are unsure about and what the rubric asks you to show."
        return SOLUTION[key]
    return f


def stubborn(key):
    return lambda system, messages: SOLUTION[key]


CASES = ([("la", "p1", q) for q in LA["p1"]] + [("la", "p2", q) for q in LA["p2"]] + [("la", "p3", q) for q in LA["p3"]]
         + [("ch", "c1", q) for q in CHEM["c1"]] + [("ch", "c2", q) for q in CHEM["c2"]]
         + [("hi", "essay", q) for q in HIST])


@pytest.mark.parametrize("course,key,question", CASES)
@pytest.mark.parametrize("model", [obedient, stubborn])
def test_probe_never_shows_original_solution(model, course, key, question):
    s, la, ch, hi = build()
    cid = {"la": la, "ch": ch, "hi": hi}[course]
    a = grounding.ask(s, cid, question, FakeProvider(model(key)))
    body = a.body_md
    for marker in LEAK_MARKERS[key]:
        assert marker not in body, (question, marker)
    assert a.mode in ("twin", "concepts")
    assert (a.mode == "twin") == a.twin
    guide = s.guide(a.guide_id)
    assert guide["data"]["mode"] == a.mode
    if key == "essay":
        assert a.mode == "concepts"


def test_graded_item_wording_and_marker_route():
    """A request the matcher cannot see (no numbers, no references) but the model recognises."""
    s, la, ch, hi = build()
    calls = []

    def model(system, messages):
        calls.append(system)
        if "GRADED ITEMS IN THIS COURSE" in system and "THE STUDENT" not in messages[0]["content"]:
            gid = [i for i in integrity.graded_view(s, ch) if i["label"] == "Problem 1"][0]["id"]
            return f'<graded item="{gid}"/>\n' + SOLUTION["c1"]
        if "<twin_problem>" in system:
            return TWIN_REPLY["problem"]
        return "Concepts [1]."
    a = grounding.ask(s, ch, "my lab partner needs the weigh-out for the salt solution we made", FakeProvider(model))
    assert a.mode == "twin" and "5.11" not in a.body_md
    assert "<graded" not in a.body_md


def test_ordinary_questions_are_not_matched():
    s, la, ch, hi = build()
    for q in ["what is molarity", "how do you find an empirical formula in general", "What is the dilution equation?",
              "How do I convert moles to grams?"]:
        assert integrity.match_graded(s, ch, q) is None, q
    for q in ["explain LU factorization", "what is a pivot", "how does row exchange work in Gaussian elimination",
              "what is the determinant of a triangular matrix"]:
        assert integrity.match_graded(s, la, q) is None, q
    for q in ["What is Madison's main argument in Federalist No. 10 about factions?",
              "What was Shays' Rebellion and why did it matter?", "Why does Hamilton call the judiciary the least dangerous branch?"]:
        assert integrity.match_graded(s, hi, q) is None, q


def test_prompt_block_always_has_rule_and_items():
    s, la, ch, hi = build()
    pb = integrity.prompt_block(s, ch)
    assert "ungraded" in pb and "<graded item=" in pb and "NaCl" in pb and "Problem 2 (Problem Set 3)" in pb
    empty = Store(":memory:")
    c = empty.add_course("X")
    assert "Nothing is marked as graded" in integrity.prompt_block(empty, c)
    sys_prompt, _ = grounding.build_messages("q", [], integrity_block=pb)
    assert "ACADEMIC INTEGRITY RULE" in sys_prompt and "Formatting rules" in sys_prompt


def test_split_items_titles_and_twin_uses_only_matched_problem():
    s, la, ch, hi = build()
    items = integrity.graded_view(s, ch)
    assert [i["label"] for i in items] == ["Problem 1", "Problem 2"]
    assert [i["title"] for i in items] == ["Problem 1 (Problem Set 3)", "Problem 2 (Problem Set 3)"]
    assert all(i["kind"] == "problem" for i in items)
    p = FakeProvider(lambda sy, m: TWIN_REPLY["problem"])
    g = integrity.match_graded(s, ch, "problem 2 of problem set 3")
    assert g["label"] == "Problem 2"
    a = integrity.twin(s, ch, "problem 2 of problem set 3", p, g)
    sent = p.calls[0]["messages"][0]["content"]
    assert "40.92" in sent and "0.350" not in sent
    assert s.guide(a.guide_id)["title"] == "Twin of Problem 2 (Problem Set 3)"
    kinds = {i["label"]: i["kind"] for i in integrity.graded_view(s, la)}
    assert kinds["Problem 4"] == "essay" and kinds["Problem 1"] == "problem"


def test_mark_and_unmark_material():
    s, la, ch, hi = build()
    ps = [m for m in s.materials(ch) if m["title"] == "Problem Set 3"][0]
    ids = integrity.mark_material_graded(s, ps["id"])
    assert len(ids) == 2 and len(s.graded_items(ch)) == 2  # idempotent
    integrity.unmark_material_graded(s, ps["id"])
    assert s.graded_items(ch) == [] and s.material(ps["id"])["graded"] == 0
    assert integrity.match_graded(s, ch, "problem 1 of problem set 3") is None


def test_coaching_includes_assignment_and_rubric_and_no_twin_banner():
    s, la, ch, hi = build()
    p = FakeProvider(lambda sy, m: "The rubric gives thesis 30 points [1]. Ask yourself what claim you can defend.")
    a = grounding.ask(s, hi, "write a thesis for me", p)
    assert a.mode == "concepts" and not a.twin
    assert "Rubric" in p.calls[0]["messages"][0]["content"]
    assert any("Essay 2" in x["material_title"] for x in a.sources)
    assert s.guide(a.guide_id)["title"] == "Concepts for Essay 2: The Federalist Essay"


def test_twin_verify_pass_runs_for_real_providers():
    s, la, ch, hi = build()
    p = FakeProvider(lambda sy, m: TWIN_REPLY["problem"])
    p.verify_twins = True
    g = integrity.match_graded(s, ch, "problem 2 of problem set 3")
    integrity.twin(s, ch, "problem 2 of problem set 3", p, g)
    assert len(p.calls) == 2 and "careful checker" in p.calls[1]["system"]


def test_check_reply_ignores_ordinary_answers():
    s, la, ch, hi = build()
    assert integrity.check_reply(s, ch, "what is molarity", "Molarity is moles of solute per liter of solution [1].") is None
    assert integrity.check_reply(s, hi, "What is Madison's argument in Federalist 10?",
                                 "Madison argues that a large republic controls the effects of faction better than a small one [1].") is None
    assert integrity.check_reply(s, ch, "q", SOLUTION["c1"])["label"] == "Problem 1"


def test_read_marker():
    assert integrity.read_marker('<graded item="12"/>\nHello') == (12, "Hello")
    assert integrity.read_marker("Hello") == (None, "Hello")
