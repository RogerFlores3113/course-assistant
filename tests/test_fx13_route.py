"""Cycle 13 routing and context (fx13-route). Fixtures are original text. Each leak test sits beside an honest look-alike."""
import json

import pytest

from course_assistant import grounding, importers, integrity, study
from course_assistant.providers import FakeProvider
from course_assistant.store import Store
from fx13_world import world as w13


@pytest.fixture
def w(tmp_path):
    return w13(tmp_path)


def hit(w, q, reply, **kw):
    s, c, ids = w
    return (integrity.check_reply(s, c, q, reply, **kw) or {}).get("label")


def asked(w, q, reply, **kw):
    s, c, ids = w
    p = FakeProvider(reply if isinstance(reply, (str, list)) else reply)
    return grounding.ask(s, c, q, p, **kw)


# ------------------------------------------------------------------ decision 1: topic-only references
TOPIC_LEAKS = [
    ("what's the mass for the NaCl problem?", ["mass = 5.26 g", "You need 5.26 g of NaCl.", "5.26 g", "It works out to about 5.26 grams.",
                                              "Roughly 5.3 g.", "5.26 grams of NaCl are needed [1]."], "Problem 1"),
    ("how many grams of NaCl for the molarity problem?", ["You need 5.26 g of NaCl.", "About 5.3 g."], "Problem 1"),
    ("what is d for the row exchange problem?", ["d = 10", "d is 10.", "It is d = 10 [1]."], "Problem 2"),
    ("what's l31 for the problem set?", ["l31 = -1"], "Problem 3"),
    ("what's y2 for the homework?", ["y2 = -12"], "Problem 3"),
    ("For the one with L and U, what is u22?", ["u22 = -8", "The entry u22 is -8."], "Problem 3"),
]


@pytest.mark.parametrize("q,replies,label", TOPIC_LEAKS)
def test_d1_a_terse_final_answer_to_a_topic_named_item_is_a_hit(w, q, replies, label):
    for r in replies:
        assert hit(w, q, r) == label, (q, r)


def test_d1_a_cite_to_the_graded_chunk_itself_does_not_excuse_it(w):
    s, c, ids = w
    chunks = [ch for m in s.materials(c) for ch in s.chunks_of(m["id"]) if s.material(ch["material_id"]).get("graded")]
    src = [dict(chunks[0], graded=True)]
    assert hit(w, "what's the mass for the NaCl problem?", "5.26 grams of NaCl are needed [1].", sources=src) == "Problem 1"


@pytest.mark.parametrize("q,r", [
    ("how do I find the mass of NaCl in general?", "Multiply the moles by the molar mass; 0.0500 mol of KCl is 3.73 g at 74.55 g/mol [1]."),
    ("what's the mass for the NaCl problem?", "Find the moles first from the volume in liters and the molarity, then multiply by the molar mass."),
    ("what's the mass for the NaCl problem?", "For a different salt, 0.0500 mol of KCl weighs 3.73 g [1]."),
    ("what is d for the row exchange problem?", "d is the entry that decides whether the second pivot is zero, so look at the second step."),
    ("what's u22?", "u22 = -8"),  # no problem word at all: nothing is touched, a plain question stays plain
])
def test_d1_look_alikes_that_name_no_item_or_give_no_value_pass(w, q, r):
    assert hit(w, q, r) is None


def test_d1_two_items_sharing_the_topic_words_name_neither(w):
    s, c, ids = w
    s.add_graded_item(c, "Set 7: Problem 9", "Problem 9. How many grams of NaCl are needed for 2.50 L of 0.100 M NaCl?", None,
                      kind="problem", label="Problem 9", answer_key=json.dumps(["mass = 14.61 g"]))
    assert integrity.topic_named_item(integrity.graded_view(s, c), "what's the mass for the NaCl problem?") is None


def test_d1_the_one_with_l_and_u_is_the_lu_item_not_the_proof(w):
    s, c, ids = w
    m = integrity.match_graded(s, c, "For my linear algebra homework, the one with L and U, I need help")
    assert m and m["label"] == "Problem 3"
    assert integrity.topic_named_item(integrity.graded_view(s, c), "the one with L and U")["label"] == "Problem 3"


def test_d1_follow_ups_in_a_chat_started_by_the_topic_are_held(w):
    s, c, ids = w
    p = FakeProvider(lambda system, msgs: "u22 = -8")
    grounding.ask(s, c, "For my linear algebra homework, the one with L and U, I need help", FakeProvider("Think about the multipliers."))
    for q in ("what's u22?", "and u33?", "what is y2?"):
        a = grounding.ask(s, c, q, FakeProvider(lambda system, msgs: "u22 = -8, u33 = 1, y2 = -12"))
        assert "u22 = -8" not in a.body_md and "y2 = -12" not in a.body_md, q


def test_d1_a_fresh_topic_question_is_held_in_ask(w):
    a = asked(w, "what's the mass for the NaCl problem?", "You need 5.26 g of NaCl.")
    assert "5.26" not in a.body_md


def test_d1_a_fresh_general_question_is_answered(w):
    a = asked(w, "how do I find the mass of NaCl in general?", "Multiply moles by the molar mass; for KCl 0.0500 mol is 3.73 g.")
    assert a.mode == "answer" and "3.73" in a.body_md


# ------------------------------------------------------------------ decision 6: over-blocks in math
@pytest.mark.parametrize("q", ["Problem 3: which lecture covers LU?", "which chapter covers the NaCl problem's topic?",
                               "Problem 1: where can I read about molarity?", "Which lecture should I read before Problem 3?"])
def test_d6_a_where_do_i_read_question_is_not_graded_work(w, q):
    s, c, ids = w
    assert integrity.is_coverage_ask(q)
    assert integrity.match_graded(s, c, q) is None


@pytest.mark.parametrize("q", ["Problem 3: what is l21?", "Problem 3: which lecture covers LU? I got l21 = 2, right?",
                               "Problem 3: solve it, which lecture covers it?"])
def test_d6_a_value_or_a_solve_ask_in_the_same_message_is_still_graded(w, q):
    s, c, ids = w
    m = integrity.match_graded(s, c, q)
    assert m and m["label"] == "Problem 3"


def _sticky(s, ids, label="Problem 3"):
    return dict(s.graded_item(ids[label]), kind="problem", reason="sticky", source="item", score=1.0)


LECTURE_REPLIES = [
    "Take A = [[7,9],[14,20]]; subtract 2 times row 1 from row 2 and row 2 becomes (0, 2) [1].",
    "Subtract 2 times row 1 from row 2. The multiplier is 2 [1].",
    "In the lecture, the matrix with rows (7, 9) and (14, 20) uses a multiplier of 2, giving the row (0, 2) [1].",
]


@pytest.mark.parametrize("r", LECTURE_REPLIES)
def test_d6_a_lecture_example_in_a_chat_sticky_to_lu_passes(w, r):
    s, c, ids = w
    src = integrity.search_safe(s, c, "worked example subtract row multiplier", limit=5)
    for q in ("In the lecture example, what are the multipliers?", "what's an example of a multiplier?"):
        assert not integrity.check_reply(s, c, q, r, force_item=_sticky(s, ids), sources=src), (q, r)


@pytest.mark.parametrize("r", ["The multiplier is 2 [1].", "Subtract 2 times row 1 from row 2 [1]."])
def test_d6_the_same_small_number_beside_another_quantity_is_not_the_lectures(w, r):
    s, c, ids = w
    decoy = [{"text": "The cart weighs 2 kg and the second shelf holds 2 boxes.", "graded": False, "chunk_id": 0}]
    assert integrity.check_reply(s, c, "what's an example of a multiplier?", r, force_item=_sticky(s, ids), sources=decoy)


def test_d6_a_lecture_echo_never_excuses_the_items_own_value(w):
    s, c, ids = w
    src = integrity.search_safe(s, c, "worked example subtract row multiplier", limit=5)
    assert integrity.check_reply(s, c, "what is l31?", "l31 = -1 [1]", force_item=_sticky(s, ids), sources=src)


OWN = "Solve A x = b for A = [[1,1],[0,2]] and b = (3, 4), what are x1 and x2?"


def test_d6_a_question_with_its_own_system_is_not_judged_against_the_sticky_item(w):
    s, c, ids = w
    st = _sticky(s, ids)
    assert not integrity.check_reply(s, c, OWN, "x2 = 2 and x1 = 1 [1]", force_item=st)
    assert integrity.check_reply(s, c, "what is l21 and l31 for problem 3?", "l21 = 2 and l31 = -1", force_item=st)


SCRUB = ("## Solutions\n\n- Moles come from the volume in liters times the molarity [1].\n"
         "- The molar mass turns moles into grams [1].\n- For the NaCl problem the answer is mass = 5.26 g [1].\n")


def test_d6_scrub_drops_only_the_offending_list_item(w):
    s, c, ids = w
    out, n, note = integrity.scrub_parts(s, c, SCRUB, "explain the NaCl problem")
    assert n == 1 and "5.26" not in out
    assert "Moles come from" in out and "molar mass turns" in out and "## Solutions" in out
    assert "Problem 1" in note


def test_d6_a_list_with_no_offender_is_kept_whole(w):
    s, c, ids = w
    clean = SCRUB.rsplit("\n- ", 1)[0] + "\n"
    out, n, note = integrity.scrub_parts(s, c, clean, "explain the NaCl problem")
    assert n == 0 and out == clean


def test_d6_a_page_that_left_an_item_out_never_says_not_covered(w):
    s, c, ids = w
    gid = study.explain(s, c, FakeProvider(SCRUB), "the NaCl problem")
    g = s.guide(gid)
    assert g["data"]["scrubbed"] == 1 and not g["data"]["not_covered"]
    assert "Moles come from" in g["body_md"] and "5.26" not in g["body_md"]


def test_d6_a_whole_list_that_is_wholly_offending_is_held_back_once(w):
    s, c, ids = w
    gid = study.explain(s, c, FakeProvider("## Solutions\n\n- For the NaCl problem the answer is mass = 5.26 g [1].\n"), "the NaCl problem")
    g = s.guide(gid)
    assert g["data"].get("held_back") and not g["data"]["not_covered"]


REP = "Calculate the percent error for dilution B and name two sources of error that could explain it."
HONEST_FORMULA = ("Percent error is the absolute difference between the measured value and the accepted value, divided by the "
                  "accepted value, times 100.")
SOURCES_OF_ERROR = (" One source of error is reading the flask meniscus at an angle. Another source of error is leaving drops of "
                    "solution behind in the pipette.")


def _report_world():
    s = Store(":memory:")
    c = s.add_course("Chem")
    s.add_graded_item(c, "Report: question 2", REP, None, kind="problem", label="Report question 2")
    return s, c


def test_d6_percent_error_keeps_the_honest_formula_and_drops_the_leaking_sentences():
    s, c = _report_world()
    g = integrity.graded_view(s, c)[0]
    p = FakeProvider(lambda system, m: HONEST_FORMULA + SOURCES_OF_ERROR)
    a = integrity.twin(s, c, "How do I calculate percent error for dilution B?", p, g, "match")
    assert "absolute difference" in a.body_md and "meniscus" not in a.body_md and "pipette" not in a.body_md
    assert "left out" in a.body_md and "Report question 2" in a.body_md
    assert "I can't write this answer" not in a.body_md


def test_d6_a_request_that_restates_the_whole_item_still_gets_the_fixed_text():
    s, c = _report_world()
    g = integrity.graded_view(s, c)[0]
    p = FakeProvider(lambda system, m: HONEST_FORMULA + SOURCES_OF_ERROR)
    a = integrity.twin(s, c, REP, p, g, "match")
    assert "meniscus" not in a.body_md and "I can't write this answer" in a.body_md


def test_d6_an_honest_formula_alone_passes_untouched():
    s, c = _report_world()
    g = integrity.graded_view(s, c)[0]
    a = integrity.twin(s, c, "How do I calculate percent error for dilution B?", FakeProvider(HONEST_FORMULA), g, "match")
    assert "absolute difference" in a.body_md and "left out" not in a.body_md


# ------------------------------------------------------------------ decision 5: proofs
P_LEAKS = [
    "Two solutions x and y give A(x-y) = 0; adding any multiple of x-y to x keeps Ax = b, so there are infinitely many solutions [1].",
    "Since A(x1) = b and A(x2) = b, the point x1 + k(x1 - x2) is a solution for every k, and there are infinitely many [1].",
    "Given two distinct solutions, every point on the line through them is a solution because A is linear. So the solution set is infinite [1].",
    "Why can a linear system never have exactly two solutions? Because x + t(y - x) solves it for every t, giving infinitely many.",
    "If x and y both solve Ax = b, the midpoint (x + y)/2 solves it too, so a second solution always leads to more than two.",
    "Take affine combinations of the two solutions: they all solve the system, so there are infinitely many.",
    "Q: Why never exactly two solutions? A: From two solutions you can always build a third one.",
    "The key idea is that two solutions always produce a third solution.",
]
P_HONEST = [
    "Try subtracting the two solutions and see what equation the difference satisfies.",
    "The difference of two solutions of Ax = b solves Ax = 0.",
    "Suppose Ax = b has two solutions x and y.",
    "If A is invertible then Ax = b has exactly one solution, x = A^-1 b, because the inverse undoes A [1].",
    "The goal is to show that a third solution exists; start by writing down what two solutions satisfy.",
    "A linear system can have no solutions, one solution or infinitely many solutions [1].",
]


def _proof(w):
    s, c, ids = w
    return next(i for i in integrity.graded_view(s, c) if i["kind"] == "proof")


@pytest.mark.parametrize("r", P_LEAKS)
def test_d5_the_count_of_solutions_proof_is_caught_whatever_the_item_says(w, r):
    s, c, ids = w
    # the item says "never exactly two": no "infinitely" or "more than one" in its own text
    assert "infinitely" not in _proof(w)["body"]
    assert hit(w, "ok", r, force_item=_sticky(s, ids, "Problem 4")) == "Problem 4"
    assert integrity.check_guide(s, c, "Notes", r)["label"] == "Problem 4"


@pytest.mark.parametrize("r", P_HONEST)
def test_d5_honest_proof_help_passes(w, r):
    s, c, ids = w
    assert not integrity._leaks_proof(_proof(w), r, "")
    assert integrity.check_guide(s, c, "Notes", r) is None


@pytest.mark.parametrize("body", ["Prove that Ax = b cannot have exactly two solutions.", "Show that if Ax = b has more than one solution, it has infinitely many.",
                                  "Prove that a consistent system never has exactly two solutions.",
                                  "Prove that if the columns are independent, Ax = b has a unique solution."])
def test_d5_the_check_runs_for_any_wording_of_a_count_proof(body):
    it = {"kind": "proof", "body": body}
    assert integrity.count_proof_item(it)
    assert integrity.two_solution_proof(it, "Take two solutions x and y; the difference x - y solves Ax = 0, so x + t(x - y) works for every t "
                                            "and there are infinitely many, or x = y and it is unique.")


def test_d5_a_proof_about_something_else_is_not_a_count_proof():
    assert not integrity.count_proof_item({"kind": "proof", "body": "Prove that the sum of two even numbers is even."})


def test_d5_the_refusal_quotes_the_step_not_the_premise(w):
    s, c, ids = w
    text = "Suppose Ax = b has two solutions x and y. Then x + t(y - x) solves it for every t, so there are infinitely many."
    h = integrity.check_guide(s, c, "Notes", text)
    assert h["label"] == "Problem 4" and "x + t(y - x)" in h["sentence"] and not h["sentence"].startswith("Suppose")


def test_d5_a_midpoint_reply_in_a_proof_chat_is_blamed_on_the_proof_not_a_numbered_item(w):
    s, c, ids = w
    r = "If x1 and x2 both solve Ax = b, then (x1 + x2)/2 solves it too, so there are infinitely many solutions."
    assert hit(w, "ok", r, force_item=_sticky(s, ids, "Problem 4")) == "Problem 4"


# ------------------------------------------------------------------ decision 9: short written items
WRITTEN = ("Problem 3 (4 points). Explain the difference between an empirical formula and a molecular formula. Give a named example "
           "in which they differ, such as CH2O and C6H12O6, and state how their molar masses are related.")
FINISHED = ("The empirical formula gives the relative numbers of each kind of atom in the smallest whole-number ratio [1]. Example: "
            "CH2O is the empirical formula and C6H12O6 is a molecular formula that is six times larger [1]. The molar mass of the "
            "molecular formula is six times the empirical formula mass [1].")
COACHING = ("Name what each term means in your own words. Then choose an example you can name and ask yourself how the molar "
            "masses compare.")
BACKGROUND = "An empirical formula is the simplest whole number ratio of atoms in a compound [1]."


def _written_world():
    s = Store(":memory:")
    c = s.add_course("Chem")
    s.add_graded_item(c, "Set 3: Problem 3", WRITTEN, None, kind="problem", label="Problem 3")
    return s, c


def test_d9_a_short_written_item_with_points_and_formulas_is_still_written():
    s, c = _written_world()
    it = integrity.graded_view(s, c)[0]
    assert integrity.written_short(it)
    assert not integrity.written_short({"body": "Problem 1. How many grams of NaCl make 250.0 mL of 0.350 M solution?"})
    assert not integrity.written_short({"body": WRITTEN, "answer_key": json.dumps(["C2H4O2"])})


def test_d9_a_terse_cited_finished_answer_is_replaced_and_the_banner_agrees():
    s, c = _written_world()
    g = integrity.graded_view(s, c)[0]
    a = integrity.twin(s, c, "Walk me through problem 3", FakeProvider(lambda sy, m: FINISHED), g, "match")
    assert a.mode == "concepts" and "six times" not in a.body_md and "second try" in a.body_md
    assert "concepts and an approach" not in (a.routing_note or "") and "here are the concepts" not in a.body_md.lower()


def test_d9_coaching_and_background_for_the_same_item_stay():
    s, c = _written_world()
    g = integrity.graded_view(s, c)[0]
    a = integrity.twin(s, c, "Walk me through problem 3", FakeProvider(lambda sy, m: COACHING), g, "match")
    assert "ask yourself" in a.body_md.lower() and "concepts and an approach" in (a.routing_note or "")
    b = integrity.twin(s, c, "Walk me through problem 3", FakeProvider(lambda sy, m: BACKGROUND), g, "match")
    assert "simplest whole number ratio" in b.body_md


def test_d9_write_the_sentence_about_the_molar_masses_is_a_finished_answer():
    s, c = _written_world()
    g = integrity.graded_view(s, c)[0]
    one = "The molar mass of the molecular formula is six times the empirical formula mass [1]."
    q = "Can you write the sentence about the molar masses for me for problem 3?"
    a = integrity.twin(s, c, q, FakeProvider(lambda sy, m: one), g, "match")
    assert "six times" not in a.body_md
    # the same sentence, offered when nobody asked for it to be written, is background
    assert integrity._reply_leak_level(g, "problem", BACKGROUND, "What is an empirical formula?", WRITTEN) == 0


def test_d9_save_guide_refuses_a_finished_answer_that_names_the_item_and_keeps_background():
    s, c = _written_world()
    h = integrity.check_guide(s, c, "Problem 3 notes", FINISHED)
    assert h and h["label"] == "Problem 3" and h["sentence"]
    assert integrity.check_guide(s, c, "Chapter notes", BACKGROUND + " " + COACHING) is None


def test_d9_cards_that_name_the_item_with_its_sentence_are_refused_and_concept_cards_saved():
    s, c = _written_world()
    sentence = "The molar mass of the molecular formula is six times the empirical formula mass."
    assert integrity.check_card(s, c, "Thesis for the empirical vs molecular formula problem?", sentence)
    assert integrity.check_card(s, c, "Problem 3 answer", "The empirical formula gives the smallest ratio and the molecular formula is a whole multiple of it.")
    assert integrity.check_card(s, c, "What is an empirical formula?", "The simplest whole number ratio of atoms in a compound.") is None
    assert integrity.check_card(s, c, "Define a molecular formula", "The actual number of each atom in one molecule.") is None


def test_d9_masses_and_mass_are_one_word():
    assert integrity.content_tokens("molar masses") == integrity.content_tokens("molar mass")
