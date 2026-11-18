"""Cycle 4 essay fixes: essay.py (prose detection, drafts, feedback check), Ask normally, wording, MCP gates."""
import json
from pathlib import Path

import pytest

from course_assistant import essay, grounding, importers, integrity, mcp_server, study
from course_assistant.providers import FakeProvider
from course_assistant.store import Store

FIX = Path(__file__).parent / "fixtures" / "import"
ESSAY_Q = ("Write a thesis-driven essay that answers this question: To what extent did the arguments of Federalist Nos. "
           "10 and 51 reflect the real political conflicts of the 1780s, as described in Chapter 7 of the textbook "
           "(Shays's Rebellion, the weaknesses of the Articles of Confederation, and the debates over the new "
           "Constitution), and to what extent did the early republic of the 1790s described in Chapter 8 (the "
           "Hamilton-Jefferson conflict and the first party system) show that Madison's predictions were right or wrong?")
THESIS_PAST = ("Madison's Federalist Nos. 10 and 51 reflected the crisis of the 1780s, but the party conflict between "
               "Hamilton and Jefferson in the 1790s showed that a large republic could not prevent faction from "
               "organizing itself.")
BODY = ("Body paragraph 1: Madison argued in Federalist No. 10 that a large republic would dilute faction, and Shays's "
        "Rebellion in Chapter 7 shows the Articles of Confederation were too weak to keep order.")
CONCLUSION = "In conclusion, the Founders' hope for a republic free of faction was undone by the party conflict of the 1790s."
BARE_PARA = ("Madison's predictions in Federalist No. 10 about faction were undermined by the 1790s. The Hamilton-Jefferson "
             "conflict showed that parties formed in a large republic anyway, and the first party system revealed that "
             "ambition did not cancel out. Shays's Rebellion had already shown the weakness of the Articles of Confederation.")
COACH = "Plan around the rubric [1]. Ask yourself which passage of Federalist No. 10 you would test your claim against."


def build():
    s = Store(":memory:")
    c = s.add_course("History")
    mids = {}
    for n in ("problem-set-2", "problem-set-3", "essay-assignment-federalist"):
        mids[n] = importers.import_file(s, c, FIX / f"{n}.md")
        integrity.mark_material_graded(s, mids[n])
    fed = s.add_material(c, "reading", "Federalist No. 10")
    s.replace_chunks(fed, [{"text": "Madison argues that a large republic controls the effects of faction better than a "
                                    "small one because it holds more interests and parties.", "loc_kind": "section",
                            "loc_start": 20, "loc_label": "paragraph 20"}])
    return s, c, mids


def essay_item(s, c):
    return next(i for i in integrity.graded_view(s, c) if i["kind"] == "essay" and "Federalist" in i["title"])


def blind(leak):
    """A model that ignores the rule: concepts calls get coaching, every other call gets the leak."""
    def f(system, messages):
        if "THE STUDENT'S GRADED TASK" in messages[0]["content"]:
            return COACH
        return leak
    return f


# ------------------------------------------------------------------ 1 tense and coaching words
@pytest.mark.parametrize("text", [
    'A strong thesis might look like this: "' + THESIS_PAST + '"',
    "For example: \"The 1790s revealed that Madison's large republic could not stop parties from forming around Hamilton "
    "and Jefferson.\"",
    "Madison's Federalist No. 10 was proven wrong because the Hamilton-Jefferson party conflict demonstrated that a large "
    "republic could not control faction.",
    "The Hamilton-Jefferson conflict proved that Madison's large republic would not escape faction in the 1790s.",
])
def test_past_tense_thesis_is_a_leak_even_with_could_not(text):
    s, c, m = build()
    assert essay.leaks(essay_item(s, c), text, "Just give me a sample thesis statement")


def test_coaching_is_not_a_leak():
    s, c, m = build()
    it = essay_item(s, c)
    assert essay.leaks(it, "A strong thesis must take a position on faction and the rubric rewards a clear arguable claim.") is None
    assert essay.leaks(it, COACH) is None
    assert essay.leaks(it, "The essay is due Friday and should be 1,500 words; late essays lose a third of a grade per day.") is None


def test_cited_restatement_of_a_source_is_fine():
    s, c, m = build()
    it = essay_item(s, c)
    src = [{"text": "Madison argues that a large republic controls the effects of faction better than a small one."}]
    ok = "Madison argues that a large republic controls the effects of faction better than a small one [1]."
    assert essay.leaks(it, ok, "", src) is None
    assert essay.leaks(it, THESIS_PAST + " [1]", "", src)  # a cite that does not back the claim does not help


# ------------------------------------------------------------------ 2 prose without an opening cue
@pytest.mark.parametrize("text", [BODY, CONCLUSION, BARE_PARA, BODY + "\n\n" + CONCLUSION])
def test_body_conclusion_and_plain_paragraphs_leak(text):
    s, c, m = build()
    assert essay.leaks(essay_item(s, c), text, "What did Madison say about faction and how does it fit chapters 7 and 8?")


@pytest.mark.parametrize("q", ["What did Madison say about faction and how does it fit chapters 7 and 8?",
                               "and the conclusion?"])
@pytest.mark.parametrize("leak", [BODY, CONCLUSION, BARE_PARA])
def test_ask_never_shows_essay_prose_from_a_blind_model(q, leak):
    s, c, m = build()
    a = grounding.ask(s, c, q, FakeProvider(blind(leak)))
    assert "Founders' hope" not in a.body_md and "Body paragraph 1" not in a.body_md and "undermined by the 1790s" not in a.body_md


@pytest.mark.parametrize("leak", [BODY, CONCLUSION, BARE_PARA])
def test_study_explain_and_practice_replace_essay_prose(leak):
    s, c, m = build()
    p = FakeProvider(lambda sy, me: f"## Notes\n\nFederalist 10 is about faction [1].\n\n{leak}")
    gid = study.explain(s, c, p, "Federalist 10 faction and the 1790s")
    body = s.guide(gid)["body_md"]
    assert leak.split(":")[-1].strip()[:40] not in body
    gid = study.practice(s, c, p, "Federalist 10 and 51 essay faction", count=2)
    assert leak.split(":")[-1].strip()[:40] not in s.guide(gid)["body_md"]


def test_honest_cited_answer_still_passes_in_one_call():
    s, c, m = build()
    p = FakeProvider("Madison argues that a large republic controls the effects of faction better than a small one [1].")
    a = grounding.ask(s, c, "What is Madison's argument in Federalist 10?", p)
    assert a.mode == "answer" and len(p.calls) == 1


# ------------------------------------------------------------------ 3 Ask normally
def test_ask_normally_on_the_essay_prompt_keeps_the_essay_check():
    s, c, m = build()
    essay_text = f"In this essay I will argue that {THESIS_PAST}\n\n{BODY}\n\n{CONCLUSION}"
    a = grounding.ask(s, c, ESSAY_Q, FakeProvider(blind(essay_text)), route=False)
    assert "In this essay" not in a.body_md and "Founders' hope" not in a.body_md
    assert a.mode == "concepts"


def test_ask_normally_button_says_what_stays_on():
    s, c, m = build()
    a = grounding.ask(s, c, ESSAY_Q, FakeProvider(blind("x")))
    g = s.guide(a.guide_id)
    assert g["data"]["ask_normally"] is True
    html = (Path(__file__).parents[1] / "src/course_assistant/templates/_answer.html").read_text()
    assert "graded answers and essays are still not written for you" in html


# ------------------------------------------------------------------ 4 check my draft
DRAFT = ("In Federalist No. 10, Madison argues that a large republic controls faction better, but the Hamilton-Jefferson "
         "conflict in Chapter 8 shows that parties still formed.")


@pytest.mark.parametrize("q", [
    f"Here is my first body paragraph, can you check it against the rubric? '{DRAFT}'",
    f"Check it against the rubric please. {DRAFT}",
    f"Can you give feedback on my paragraph “{DRAFT}”",
    f"Here is my body paragraph for the Federalist essay, thoughts?\n{DRAFT} Shays's Rebellion in Chapter 7 also matters here.",
    "Madison argued in Federalist No. 10 that a large republic controls faction better than a small one. The "
    "Hamilton-Jefferson conflict in Chapter 8 showed that parties formed anyway, so his prediction was partly wrong. "
    "Shays's Rebellion in Chapter 7 also showed the Articles were weak, and the debates over the Constitution matter.",
])
def test_draft_is_found_without_a_colon(q):
    s, c, m = build()
    assert essay.draft_request(q)
    g = integrity.match_graded(s, c, q)
    assert g and g["intent"] == "feedback" and "Federalist" in g["title"]


def test_draft_prefers_the_named_item_over_a_numeric_one():
    s, c, m = build()
    q = ("Here's my draft for Problem 3, can you check my draft? An empirical formula shows the simplest ratio of atoms "
         "in a compound, while a molecular formula shows the actual numbers of atoms and the molar mass of a multiple.")
    g = integrity.match_graded(s, c, q)
    assert g["label"] == "Problem 3" and g["intent"] == "feedback" and "CHEM" in g["title"]


def test_a_numeric_check_ask_is_not_a_draft():
    s, c, m = build()
    q = "Check my answer for problem 1 of problem set 3: I got 5.11 g of sodium chloride for the solution, is that right?"
    g = integrity.match_graded(s, c, q)
    assert g and g.get("intent") != "feedback" and g["kind"] == "problem"


def test_pasted_assignment_is_not_a_draft():
    s, c, m = build()
    assert essay.draft_item(f"Can you review this? {ESSAY_Q}", [essay_item(s, c)], integrity.graded_view(s, c)) is None


@pytest.mark.parametrize("reply", [
    "A stronger thesis would be: Madison's theory explained the 1780s but underestimated parties in the 1790s.",
    "Consider revising it to say that Madison correctly diagnosed Shays-era factions but missed how quickly Hamilton and "
    "Jefferson would organize parties.",
    "Parties formed around Hamilton and Jefferson, so Madison's large republic did not control faction in the 1790s.",
    'You could open with "Diagonalization turns hard matrix powers into easy scalar powers in modeling."',
])
def test_feedback_that_supplies_new_sentences_is_rejected(reply):
    s, c, m = build()
    it = essay_item(s, c)
    assert essay.feedback_leaks(DRAFT, it, reply)


def test_good_feedback_passes():
    s, c, m = build()
    it = essay_item(s, c)
    good = (f'Your paragraph says "{DRAFT}", which the rubric rewards as Federalist evidence (25 points). It does not yet '
            "say which paragraph of Federalist No. 10 you cite. Which event in Chapter 8 would a reader ask you to "
            "explain, and what counterargument would the Anti-Federalists raise?")
    assert not essay.feedback_leaks(DRAFT, it, good)


def test_draft_feedback_falls_back_to_rubric_questions():
    s, c, m = build()
    bad = "A stronger thesis would be: Madison's theory explained the 1780s but underestimated parties in the 1790s."
    p = FakeProvider(lambda sy, me: bad if "OWN" in sy else "ok")
    q = f"Check my thesis, can you give feedback on it? '{DRAFT}'"
    a = grounding.ask(s, c, q, p)
    assert "stronger thesis" not in a.body_md and "questions" in a.body_md and "feedback against the rubric" in a.body_md
    assert len([x for x in p.calls if "OWN" in x["system"]]) == 2


def test_concept_question_after_a_draft_turn_is_not_a_draft():
    s, c, m = build()
    good = f'Your draft says "{DRAFT}". Which paragraph of Federalist No. 10 do you cite, and what does the rubric reward?'
    q1 = ("Here's my draft for Problem 3, can you check my draft? An empirical formula shows the simplest ratio of atoms "
          "in a compound, while a molecular formula shows the actual numbers of atoms and the molar mass of a multiple.")
    p = FakeProvider(lambda sy, me: good if "OWN" in sy else "Divide each mass by its atomic mass, then divide by the "
                     "smallest mole value to get the empirical formula from percent composition [1].")
    grounding.ask(s, c, q1, p)
    a = grounding.ask(s, c, "How do I find an empirical formula from percent composition? Walk me through the method.", p)
    assert a.mode == "answer" and "Divide each mass" in a.body_md


def test_proof_drafts_get_method_feedback():
    s = Store(":memory:")
    c = s.add_course("Linear Algebra")
    s.add_graded_item(c, "Problem 4", "Prove that if A is invertible then A^T is invertible, and that (A^T)^-1 = (A^-1)^T.",
                      kind="proof")
    seen = []

    def f(system, messages):
        seen.append(system)
        return ('You wrote "since A is invertible, A^T is invertible" with no reason. Which property of the transpose '
                "would justify it, and is every case covered?")
    q = ("Check my proof draft for Problem 4: Let A be invertible. Since A is invertible, A^T is invertible. Then the "
         "inverse of A^T is the transpose of the inverse of A because they multiply to the identity matrix.")
    g = integrity.match_graded(s, c, q)
    assert g and g["intent"] == "feedback" and g["kind"] == "proof"
    a = integrity.twin(s, c, q, FakeProvider(f), g)
    assert "proof for work" in seen[0] and "missing step" in seen[0]
    assert s.guide(a.guide_id)["data"]["feedback"] is True and "Similar problem" not in s.guide(a.guide_id)["title"]


# ------------------------------------------------------------------ 5 wording and banners
def test_essay_pages_never_say_twin_and_carry_one_banner():
    s, c, m = build()
    a = grounding.ask(s, c, "Write my Federalist essay thesis", FakeProvider(blind("x")))
    g = s.guide(a.guide_id)
    assert g["kind"] == "twin" and g["data"]["mode"] == "concepts"
    assert not a.blocks[0].md.startswith("This was marked graded")  # the routing sentence is the only note
    assert essay.ready_message(g, "Your twin problem is ready.") == "Your concepts guide is ready."
    assert essay.guide_kind_label(g, "Worked twin problem") == "Concepts for graded work"
    from fastapi.testclient import TestClient
    from course_assistant import disclaimer, web
    s.set_setting(disclaimer.ACK_SETTING, "2026-01-01T00:00:00+00:00")
    cl = TestClient(web.create_app(s, key="k" * 40, provider_factory=lambda: None), base_url="http://127.0.0.1:8766")
    cl.cookies.set(web.COOKIE, "k" * 40)
    html = cl.get(f"/guides/{a.guide_id}").text
    assert "Worked twin problem" not in html and "Graded work: help with concepts only" not in html
    assert html.count("data-routing-note") == 1


def test_numeric_twin_keeps_its_wording():
    s, c, m = build()
    assert essay.ready_message({"kind": "twin", "data": {"mode": "twin"}}, "Your twin problem is ready.") == "Your twin problem is ready."


# ------------------------------------------------------------------ 6 routing
@pytest.mark.parametrize("q", ["Write 300 words on whether Madison's predictions came true",
                               "How would I argue that Madison was wrong in an essay?"])
def test_writing_requests_without_the_word_essay_route(q):
    assert essay.writing_request(q)
    s, c, m = build()
    g = integrity.match_graded(s, c, q)
    assert g and g["kind"] == "essay"


@pytest.mark.parametrize("q", ["When is the Federalist essay due?", "What does Federalist 10 say about faction?",
                               "Summarize Chapter 8 in 300 words"])
def test_plain_questions_are_not_writing_requests(q):
    s, c, m = build()
    assert integrity.match_graded(s, c, q) is None


# ------------------------------------------------------------------ 7 MCP
@pytest.fixture
def mcp():
    s, c, m = build()
    mcp_server.set_store(s)
    yield s, c, m
    mcp_server.set_store(None)


def fed_chunk(s, c):
    mid = next(x["id"] for x in s.materials(c) if x["title"] == "Federalist No. 10")
    return s.chunks_of(mid)[0]["id"]


@pytest.mark.parametrize("kind", ["answer", "explain", "practice", "twin"])
def test_save_guide_refuses_a_one_sentence_thesis(mcp, kind):
    s, c, m = mcp
    r = mcp_server.save_guide(c, kind, "Thesis idea", THESIS_PAST + " [1]", [fed_chunk(s, c)])
    assert r.startswith("NOT SAVED")


@pytest.mark.parametrize("body", [
    "1. Thesis: Madison was partly right.\n2. Evidence: Federalist 10.\n3. Counter: Anti-Federalists. [1]",
    "Essay 2 outline:\n- Thesis: Madison was wrong about faction.\n- Evidence: the Hamilton-Jefferson party conflict.\n- Counter: Federalist 10.",
    BODY, CONCLUSION,
])
def test_save_guide_refuses_outlines_and_paragraphs(mcp, body):
    s, c, m = mcp
    assert mcp_server.save_guide(c, "answer", "Essay 2 outline", body, [fed_chunk(s, c)]).startswith("NOT SAVED")


@pytest.mark.parametrize("body", [
    "An empirical formula gives the simplest whole-number ratio of the atoms in a compound, while a molecular formula "
    "gives the actual number of each atom. For glucose the empirical formula is CH2O and the molecular formula is "
    "C6H12O6, so the molar mass is a whole-number multiple of the empirical formula mass.",
    "Using the final volume matters because molarity is moles per litre of solution. Weigh the NaCl, dissolve it in a "
    "little water in the volumetric flask, then fill to the mark so the final solution volume is 250.0 mL.",
])
def test_save_guide_refuses_full_written_answers_to_short_answer_items(mcp, body):
    s, c, m = mcp
    assert mcp_server.save_guide(c, "answer", "Study notes", body, []).startswith("NOT SAVED")


def test_save_guide_still_saves_cited_restatements_and_coaching(mcp):
    s, c, m = mcp
    ok = "Madison argues that a large republic controls the effects of faction better than a small one [1]."
    assert mcp_server.save_guide(c, "explain", "Federalist 10", ok, [fed_chunk(s, c)]).startswith("Saved guide")
    coach = ("The rubric gives thesis 30 points. Ask yourself which passage of Federalist No. 10 you would test your claim "
             "against, and what a reader could doubt.")
    assert mcp_server.save_guide(c, "explain", "Plan the essay", coach, []).startswith("Saved guide")


def test_add_flashcards_refuses_essay_theses_and_conclusions(mcp):
    s, c, m = mcp
    for back in ("Madison's theory explained the 1780s but underestimated parties in the 1790s.", CONCLUSION, THESIS_PAST):
        r = mcp_server.add_flashcards(c, [{"front": "My thesis", "back": back}])
        assert r.startswith("NOT ADDED"), back
    ok = mcp_server.add_flashcards(c, [{"front": "What does Madison argue in Federalist 10?",
                                        "back": "A large republic controls the effects of faction better than a small one.",
                                        "chunk_id": fed_chunk(s, c)}])
    assert ok.startswith("Added")


def test_set_answer_key_refuses_essays_and_clears_with_an_empty_list(mcp):
    s, c, m = mcp
    it = essay_item(s, c)
    with pytest.raises(Exception):
        mcp_server.set_answer_key(it["id"], ["x"])
    num = next(i for i in integrity.graded_view(s, c) if i["label"] == "Problem 1" and "CHEM" in i["title"])
    mcp_server.set_answer_key(num["id"], ["5.11 g"])
    assert integrity.item_key(s.graded_item(num["id"])) == ["5.11 g"]
    assert "Cleared" in mcp_server.set_answer_key(num["id"], [])
    assert integrity.item_key(s.graded_item(num["id"])) == []
