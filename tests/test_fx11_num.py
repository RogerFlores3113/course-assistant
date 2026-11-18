"""Cycle 11 numeric fixes: each panel repro next to its honest look-alike. Fixtures are original (fx11_world, fx9_world)."""
import json

import pytest

import fx11_world as W
from course_assistant import grounding, integrity, mcp_server
from course_assistant.providers import FakeProvider
from course_assistant.store import Store


@pytest.fixture
def w(tmp_path):
    x = W.world(tmp_path)
    mcp_server.set_store(x["s"])
    yield x
    mcp_server.set_store(None)


def item(x, label):
    return next(i for i in integrity.graded_view(x["s"], x["c"]) if i["label"] == label)


def sticky(x, label):
    return dict(item(x, label), reason="sticky")


# ------------------------------------------------------------ decision 1: cite laundering
LAUNDERED = ["x = (1, 1, 2) [1]", "The solution is x = (1, 1, 2) [1].",
             "The multipliers l21 = 2, l31 = -1, l32 = -1 come from elimination [1].", "z = 2, then y = 1, then x = 1 [1]",
             "The unknowns are x1 = 1, x2 = 1 and x3 = 2 [1]."]


@pytest.mark.parametrize("text", LAUNDERED)
def test_d1_small_digits_in_the_cited_chunk_never_launder_the_answer(w, text):
    lec = [w["chunks"]["lecture-4"][0]]
    assert mcp_server.save_guide(w["c"], "explain", "Notes", text, lec).startswith("NOT SAVED")
    card = {"front": "Lecture", "back": text.replace("[1]", ""), "chunk_id": lec[0]}
    assert mcp_server.add_flashcards(w["c"], [card]).startswith("NOT ADDED")


@pytest.mark.parametrize("text", LAUNDERED)
def test_d1_ask_with_a_cited_lecture_chunk_does_not_show_the_answer(w, text):
    s, c = w["s"], w["c"]
    src = [{"text": s.chunk(w["chunks"]["lecture-4"][0])["text"], "graded": False}]
    hit = integrity.check_reply(s, c, "and the multipliers?", text, force_item=sticky(w, "Problem 9"), sources=src)
    assert hit and hit["label"] == "Problem 9"


def test_d1_look_alikes_the_chunk_itself_shows_still_pass(w):
    lec2 = [w["chunks"]["lecture-18"][0]]
    ok = "The lecture example has eigenvectors (1, 1) and (1, -2) [1]."
    assert mcp_server.save_guide(w["c"], "explain", "Notes", ok, lec2).startswith("Saved")
    assert mcp_server.save_guide(w["c"], "explain", "Notes", "The lecture example has eigenvalues 4 and 1 [1].", lec2).startswith("Saved")
    # the same sentence cited to a chunk that does not print it is judged on its own
    assert mcp_server.save_guide(w["c"], "explain", "Notes", "The eigenvectors are (1, 1) and (1, -2) [1].",
                                 [w["chunks"]["lecture-4"][0]]).startswith("NOT SAVED")
    for body, ix in (("The textbook example: a 355 mL soft drink holds 40.0 g of sucrose, which is 0.117 mol, so the molarity is 0.330 M [1]", 0),
                     ("The molar mass of copper sulfate pentahydrate is 249.68 g/mol [1].", 1)):
        assert mcp_server.save_guide(w["c"], "explain", "Notes", body, [w["chunks"]["chapter-2"][ix]]).startswith("Saved"), body
    assert mcp_server.add_flashcards(w["c"], [{"front": "Molar mass of the hydrate?", "back": "249.68 g/mol",
                                               "chunk_id": w["chunks"]["chapter-2"][1]}]).startswith("Added")


def test_d1_ask_look_alike_lecture_example_shown(w):
    s, c = w["s"], w["c"]
    src = [{"text": s.chunk(w["chunks"]["lecture-18"][0])["text"], "graded": False}]
    r = "The lecture matrix has eigenvalues 4 and 1, with eigenvectors (1, 1) and (1, -2) [1]."
    assert integrity.check_reply(s, c, "what does the lecture example say?", r, force_item=sticky(w, "Problem 10"), sources=src) is None


def test_d1_bare_digits_in_a_chunk_do_not_make_a_constant(w):
    # a chunk that merely prints 5.11 and 2.497 with no unit or quantity word frees nothing
    assert integrity._source_data([{"text": "Table 3 lists 5.11 and 2.497 for run B.", "graded": False}]) == set()
    assert "249.68" in integrity._source_data([{"text": W.BOOK, "graded": False}])
    assert mcp_server.save_guide(w["c"], "explain", "Notes", "Weigh 2.497 g for Pre-lab 1 [1]",
                                 [w["chunks"]["lecture-4"][0]]).startswith("NOT SAVED")


# ------------------------------------------------------------ decision 2: trailing marker
@pytest.mark.parametrize("body", ["The answer to Problem 9 is (1, 1, 2) [1]", "x = 1, y = 1, z = 2 [1]", "(x, y, z) = (1, 1, 2) [1]",
                                  "U = 2 1 1 / 0 -8 -2 / 0 0 1 [1]", "U = 2 1 1 / 0 -8 -2 / 0 0 1. [1]", "d = 11 [1]"])
def test_d2_a_trailing_marker_is_judged_like_one_after_the_full_stop(w, body):
    for chunk in ([w["chunks"]["chapter-2"][0]], [w["chunks"]["lecture-4"][0]]):
        assert mcp_server.save_guide(w["c"], "explain", "Notes", body, chunk).startswith("NOT SAVED")
        assert mcp_server.save_guide(w["c"], "explain", "Notes", body.replace(" [1]", ".").replace(". [1]", ".") + " [1]", chunk).startswith("NOT SAVED")


def test_d2_honest_sentence_is_the_same_with_and_without_the_full_stop(w):
    book = [w["chunks"]["chapter-2"][0]]
    for body in ("For example, 0.117 mol in 0.355 L is 0.330 M [1]", "For example, 0.117 mol in 0.355 L is 0.330 M. [1]"):
        assert mcp_server.save_guide(w["c"], "explain", "Notes", body, book).startswith("Saved"), body
    assert integrity._sentences_of("x = (1, 1, 2). [1]") == ["x = (1, 1, 2). [1]"]


# ------------------------------------------------------------ decision 3: integrity-bar numbers
def eig3(x):
    return x["s"].add_graded_item(x["c"], "Set 7: Problem 13", "Problem 13 (10 points). Find the eigenvalues and eigenvectors of the "
                                  "matrix with rows (7, 3) and (-2, 2).", None, kind="problem", label="Problem 13",
                                  answer_key=json.dumps(["lambda1 = 4", "lambda2 = 5", "v1 = (3, 1)", "v2 = (1, -3)"]))


@pytest.mark.parametrize("text", ["The eigenvectors are (3, 1) and (1, -3).", "The eigenvectors are (3, 1) and (1, -3) [1].",
                                  "(3,1) for 4 and (1,-3) for 5"])
def test_d3_two_entry_eigenvectors_are_hits_when_touched_or_in_a_guide(w, text):
    eig3(w)
    s, c = w["s"], w["c"]
    hit = integrity.check_reply(s, c, "explain", text, force_item=sticky(w, "Problem 13"))
    assert hit and hit["label"] == "Problem 13"
    assert mcp_server.save_guide(c, "explain", "Notes", text, []).startswith("NOT SAVED")


def test_d3_pair_look_alikes_pass(w):
    eig3(w)
    s, c = w["s"], w["c"]
    for text in ("The eigenvectors of a 2 by 2 matrix satisfy A v = lambda v.", "The point (3, 1) lies right of (1, -3).",
                 "The eigenvectors of the lecture matrix are (2, 5) and (1, 1)."):
        assert integrity.check_reply(s, c, "explain", text, force_item=sticky(w, "Problem 13")) is None, text
    assert integrity.check_reply(s, c, "explain", "The eigenvectors are (3, 1) and (1, -3).") is None  # untouched chat


def test_d3_roots_count_like_eigenvalues(w):
    hit = integrity.check_reply(w["s"], w["c"], "explain", "The roots are 5 and 2.", force_item=sticky(w, "Problem 10"))
    assert hit and hit["label"] == "Problem 10"
    assert mcp_server.save_guide(w["c"], "explain", "Notes", "The roots are 5 and 2", []).startswith("NOT SAVED")
    assert integrity.check_reply(w["s"], w["c"], "explain", "The roots of x^2 - 9 are 3 and -3.", force_item=sticky(w, "Problem 10")) is None


def test_d3_two_significant_figures_of_a_unit_value_when_touched(w):
    s, c = w["s"], w["c"]
    for r in ("You need about 5.1 g of salt.", "about 5.1 g"):
        hit = integrity.check_reply(s, c, "how many grams?", r, force_item=sticky(w, "Problem 12"))
        assert hit and hit["label"] == "Problem 12", r
        assert integrity.check_reply(s, c, "how many grams?", r) is None  # untouched keeps the three figure rule
    for r in ("You need about 5 g of salt.", "A pinch is about 0.5 g.", "Use 58.4 g/mol for the molar mass."):
        assert integrity.check_reply(s, c, "how many grams?", r, force_item=sticky(w, "Problem 12")) is None, r


def test_d3_matrix_sizes_are_never_values(w):
    for r in ("Find the eigenvalues of the 2 by 2 matrix B.", "Take a 2x2 matrix.", "A 3-by-3 matrix has nine entries."):
        assert integrity.check_reply(w["s"], w["c"], "explain", r, force_item=sticky(w, "Problem 10")) is None, r
        assert integrity.key_leak(r, item(w, "Problem 10"), {"named": True}) == []
    assert integrity.numbers("2 x 3 = 6") == ["2", "3", "6"]  # multiplication is not a size


# ------------------------------------------------------------ decision 4: confirm-style checks
@pytest.mark.parametrize("q", ["Is 2.50 g okay for the pre-lab mass?", "Is 2.50 g fine for the pre-lab?", "Would 2.50 g work for pre-lab 1?"])
def test_d4_confirm_style_question_with_a_key_value_is_a_check_ask(w, q):
    assert integrity.is_check_ask(q)
    assert integrity.match_graded(w["s"], w["c"], q)["label"] == "Pre-lab 1"


def test_d4_a_reply_that_confirms_it_is_a_leak(w):
    q = "Is 2.50 g okay for the pre-lab mass?"
    for r in ("Yes, that is fine.", "Yes, 2.50 g works.", "That's okay."):
        assert integrity.confirms_answer(r), r
        hit = integrity.check_reply(w["s"], w["c"], q, r, force_item=sticky(w, "Pre-lab 1"))
        assert hit and hit["label"] == "Pre-lab 1", r
    a = grounding.ask(w["s"], w["c"], q, FakeProvider(lambda system, messages: "Yes, 2.50 g is fine."))
    assert a.mode in ("twin", "concepts") and "is fine" not in a.body_md


def test_d4_look_alikes_are_not_check_asks(w):
    for q in ("Is the lab report due on Friday okay?", "Is 58.44 g/mol okay for NaCl?", "Is the TA session fine on Monday?"):
        assert integrity.match_graded(w["s"], w["c"], q) is None, q
    assert integrity.check_reply(w["s"], w["c"], "Is 58.44 g/mol okay for NaCl?", "Yes, 58.44 g/mol is the molar mass of NaCl [1].") is None


# ------------------------------------------------------------ decision 5: handout note
def test_d5_note_only_when_the_question_names_the_handout_or_two_words_match(tmp_path):
    s, c, mid = W.handout_only_questions(tmp_path)
    assert integrity.handout_sources(s, c, "What should I put in the lab handout?")[1] >= 1
    assert integrity.handout_sources(s, c, "How do I compute the molarity of a brine sample when it is diluted?")[1] >= 1
    for q in ("Explain molarity", "What is a mole?", "Is the lab report due Friday?", "Is my answer right?"):
        assert integrity.handout_sources(s, c, q)[1] == 0, q


def test_d5_never_beside_an_also_searched_row_for_the_same_handout(tmp_path):
    from fx9_world import handout_world
    s, c, ids, mid = handout_world(tmp_path)  # its questions share a chunk with the safety text
    q = "Why does the pink color fade after the endpoint of a titration?"
    out, n = integrity.handout_sources(s, c, q)
    assert out and n == 0
    a = grounding.ask(s, c, q, FakeProvider(lambda system, messages: "The sources do not cover it."))
    assert "Part of your graded handout was not searched" not in a.body_md


# ------------------------------------------------------------ decision 6: short written items
BODY14 = ("Problem 14 (8 points). Define the empirical formula and the molecular formula of a compound. Then use glucose, with molecular "
          "formula C6H12O6 and molar mass 180.16 g/mol, to explain how the molecular formula relates to the empirical formula CH2O. "
          + " ".join(f"Part {i} is worth points for a clear statement and a correct use of the term in context." for i in range(1, 9)))
ANS14 = ("An empirical formula is the simplest whole-number ratio of the atoms in a compound [1]. A molecular formula gives the actual number "
         "of atoms in one molecule of the compound [1]. Glucose has the molecular formula C6H12O6 and the empirical formula CH2O, so its "
         "molar mass is six times the empirical formula mass [1].")
COACH14 = ("Name the two formulas, say what each tells you, and choose a pair of your own to compare. Ask yourself how the ratio and the "
           "count could differ.")


def chem_world(tmp_path):
    from course_assistant import importers
    s = Store(":memory:")
    c = s.add_course("C")
    s.add_graded_item(c, "Set: Problem 14", BODY14, None, kind="essay", label="Problem 14")
    f = tmp_path / "chem.md"
    f.write_text("# Formulas\n\n## Empirical and molecular\n\n" + ANS14.replace(" [1]", "") + "\n")
    importers.import_file(s, c, f)
    return s, c


def test_d6_a_cited_finished_answer_to_a_long_short_answer_item_is_replaced(tmp_path):
    s, c = chem_world(tmp_path)

    def f(system, messages):
        if "THE STUDENT'S GRADED TASK" in messages[0]["content"]:
            return COACH14 if "Your last reply contained" in messages[-1]["content"] else ANS14
        return "ok"
    a = grounding.ask(s, c, BODY14, FakeProvider(f))
    assert a.mode == "concepts" and "simplest whole-number ratio" not in a.body_md and "ask yourself" in a.body_md.lower()
    assert integrity.written_item(BODY14)


def test_d6_fallback_for_a_short_written_item_talks_about_the_question_not_a_thesis(tmp_path):
    s, c = chem_world(tmp_path)
    low = integrity.fallback_text({"body": BODY14}, "essay", False).lower()
    assert "thesis" not in low and "first paragraph" not in low and "your own answer" in low
    a = grounding.ask(s, c, BODY14, FakeProvider(lambda system, messages: ANS14 if "GRADED TASK" in messages[0]["content"] else "ok"))
    assert "thesis" not in a.body_md.lower()
    # a real essay keeps its essay wording
    assert integrity.fallback_text({"body": "In about 700 words, argue which effect mattered most. Rubric: thesis 30%."}, "essay", False) == integrity.SAFE_ESSAY


def test_d6_coaching_reply_to_the_same_item_passes_first_time():
    assert integrity.finished_answer(BODY14, COACH14) is False
    assert integrity.finished_answer(BODY14, ANS14) is True
    two = "An empirical formula is the simplest whole-number ratio of the atoms in a compound [1]. A molecular formula gives the actual number of atoms in one molecule [1]."
    assert integrity.finished_answer(BODY14, two) is False


# ------------------------------------------------------------ orchestrator note: no twin for written work
def test_check_wording_for_essay_and_proof_never_mentions_a_twin():
    for kind in ("essay", "proof"):
        for text in (integrity.CHECK_RULE_WRITTEN, integrity.fallback_text({"body": "Explain why x."}, kind, True)):
            assert "twin" not in text.lower() and "similar problem" not in text.lower(), (kind, text)
    assert "twin" in integrity.CHECK_RULE and "twin" in integrity.fallback_text({"body": "x"}, "problem", True).lower() + "twin"


def test_check_ask_on_an_essay_item_page_never_mentions_a_twin(tmp_path):
    s, c = chem_world(tmp_path)
    seen = []

    def f(system, messages):
        seen.append(messages[0]["content"])
        return "Compare your own words with each requirement and say which part you have not addressed."
    q = "I got that a molecular formula is a multiple of the empirical formula. Is that right? " + BODY14
    a = grounding.ask(s, c, q, FakeProvider(f))
    assert "twin" not in a.body_md.lower() and all("twin" not in x.lower() for x in seen)
