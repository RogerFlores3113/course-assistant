"""Cycle 7 essay fixes: coaching frames never exempt a claim, rhetorical and non-Latin claims, imperative coaching, per-sentence
cited restatements, not-covered sentences, feedback post-check precision, per-kind fallback, thesis cards. Original text."""
import re

import pytest

from course_assistant import essay, grounding, integrity, mcp_server
from course_assistant.providers import FakeProvider
from course_assistant.store import Store

from test_fx5_essay import FED10, build, chunk_of, essay_item
from test_fx6_essay import la_course, la_item

ASK_Q = "Write the introduction paragraph for my Federalist essay"

FRAMED = [
    "A student should note that Madison's large republic could not prevent the parties that formed around Hamilton and "
    "Jefferson in the 1790s, so his remedy for faction was only partly successful.",
    "Consider that Madison's large republic could not prevent the parties of the 1790s, so his remedy for faction was only "
    "partly right.",
    "You should note that this essay argues that Madison's diagnosis captured the 1780s.",
    "As you can see, Madison's remedy for faction failed once Hamilton and Jefferson organized parties in the 1790s.",
    "It is worth noting that Madison's diagnosis of faction was correct for the 1780s but incomplete for the 1790s.",
    "To answer the question directly: his predictions held for the 1780s and failed for the 1790s.",
]
FRAMED_COACHING = [
    "A strong thesis takes a side on 'to what extent' the papers reflected the 1780s.",
    "You should check that your thesis takes a side.",
    "You should note that your thesis needs a claim someone could dispute.",
    "Consider that the rubric gives 30 points for a clear, arguable thesis.",
    "As you can see, the requirements ask for a counterargument and a short conclusion.",
    "A student should note that the essay needs two quoted passages from the Federalist Papers.",
]


def sample_blind(leak):
    return lambda sy, me: ("Plan around the rubric. Ask yourself which passage you would test your claim against."
                           if "THE STUDENT'S GRADED TASK" in me[0]["content"] else leak)


# ------------------------------------------------------------------ 1 coaching frames at the start of a sentence
@pytest.mark.parametrize("text", FRAMED)
def test_a_coaching_frame_never_exempts_a_claim(text):
    s, c = build()
    assert essay.level(essay_item(s, c), text) == 2, text
    assert essay.level(essay_item(s, c), text, ASK_Q, True) == 2, text


def test_the_three_panel_sentences_are_hard_hits_together():
    s, c = build()
    para = ("Consider that Madison's large republic could not prevent the parties of the 1790s, so his remedy for faction was "
            "only partly right. You should note that this essay argues that Madison's diagnosis captured the 1780s. As you can "
            "see, Shays's Rebellion supports that reading.")
    assert essay.level(essay_item(s, c), para) == 2


@pytest.mark.parametrize("text", FRAMED_COACHING)
def test_a_frame_before_real_coaching_stays_fine(text):
    s, c = build()
    assert essay.level(essay_item(s, c), text) == 0, text
    assert essay.level(essay_item(s, c), text, ASK_Q, True) == 0, text


@pytest.mark.parametrize("text", FRAMED)
def test_framed_claims_never_reach_the_page_in_chat(text):
    s, c = build()
    for route in (True, False):
        a = grounding.ask(s, c, ASK_Q, FakeProvider(sample_blind(text)), route=route)
        assert text[:40] not in re.sub(r"\s+", " ", a.body_md), (route, text)


@pytest.mark.parametrize("text", FRAMED)
def test_framed_claims_are_not_saved_by_mcp(text):
    s, c = build()
    mcp_server.set_store(s)
    try:
        assert mcp_server.save_guide(c, "answer", "Idea", text + " [1]", [chunk_of(s, c, "Federalist No. 10")]).startswith("NOT SAVED")
        assert mcp_server.add_flashcards(c, [{"front": "My claim", "back": text}]).startswith("NOT ADDED")
    finally:
        mcp_server.set_store(None)


def test_mcp_still_saves_honest_framed_coaching():
    s, c = build()
    mcp_server.set_store(s)
    try:
        body = "A strong thesis takes a side on 'to what extent' the papers reflected the 1780s. [1] You should check that your thesis takes a side."
        assert mcp_server.save_guide(c, "answer", "Method", body, [chunk_of(s, c, "Federalist No. 10")]).startswith("Saved")
    finally:
        mcp_server.set_store(None)


# ------------------------------------------------------------------ 2 questions that carry a claim
RHETORICAL = [
    "Wasn't the real lesson of the 1790s that factions organize around ideas, so Madison's remedy was incomplete?",
    "Isn't it true that Madison's diagnosis of faction was right for the 1780s but his remedy failed by 1794?",
    "Wasn't Madison wrong about the party system of the 1790s?",
]
HONEST_QUESTIONS = [
    "Which passage of Federalist No. 10 would you test your claim against?",
    "Isn't it worth checking your thesis against the rubric?",
    "Wasn't Federalist No. 10 published in 1787?",
    "Does your draft say which events from Chapter 7 you will use?",
]


@pytest.mark.parametrize("text", RHETORICAL)
def test_a_rhetorical_question_that_carries_a_claim_is_essay_prose(text):
    s, c = build()
    assert essay.level(essay_item(s, c), text, ASK_Q, True) == 2, text
    assert essay.level(essay_item(s, c), text) > 0, text


@pytest.mark.parametrize("text", HONEST_QUESTIONS)
def test_honest_questions_are_fine(text):
    s, c = build()
    assert essay.level(essay_item(s, c), text, ASK_Q, True) == 0, text


# ------------------------------------------------------------------ 3 other scripts
CJK = "麦迪逊对1780年代的派系判断是正确的，但1790年代政党的出现表明他的预测不完整。"


def test_a_chinese_claim_is_judged_like_other_non_english_text():
    s, c = build()
    assert essay.non_english(CJK)
    assert essay.level(essay_item(s, c), CJK) == 2
    assert essay.level(essay_item(s, c), "补充说明：" + CJK, ASK_Q, True) == 2
    a = grounding.ask(s, c, ASK_Q, FakeProvider(sample_blind(CJK)), route=False)
    assert "麦迪逊" not in a.body_md


def test_a_short_chinese_greeting_and_an_english_sentence_are_not_judged():
    s, c = build()
    assert not essay.non_english("你好")
    assert not essay.non_english("The assignment is due on Friday and asks for five pages with Chicago footnotes.")
    assert essay.level(essay_item(s, c), "好的，我来帮你。") == 0


# ------------------------------------------------------------------ 4 quote plus analysis
def test_a_quote_plus_analysis_sentence_is_a_position():
    s, c = build()
    t = ("'Ambition must be made to counteract ambition' captures why the 1790s rivalries stayed inside the constitutional "
         "system, even though Madison feared faction.")
    assert essay.level(essay_item(s, c), t) > 0


def test_a_plain_quote_with_attribution_is_fine():
    s, c = build()
    it = essay_item(s, c)
    assert essay.level(it, "Madison writes, \"Ambition must be made to counteract ambition\" in Federalist No. 51. [1]",
                       sources=[{"text": "Ambition must be made to counteract ambition. Madison writes in Federalist No. 51."}]) == 0
    assert essay.level(it, "The line \"Ambition must be made to counteract ambition\" is the phrase the prompt quotes.") == 0


# ------------------------------------------------------------------ 5 per-sentence restatement, not-covered
SHAYS_SOURCES = [
    {"text": "In Federalist No. 10, James Madison argues that a large republic is better than a small one at controlling the "
             "dangers of faction. Write a thesis-driven essay about Shays's Rebellion, the weaknesses of the Articles of "
             "Confederation, and the debates over the new Constitution."},
    {"text": "Farmers in the interior closed courthouses in 1786 to stop foreclosures and debt suits."},
    {"text": "Use at least three specific events or people from Chapters 7 and 8 as evidence."},
]
SHAYS = ("The prompt names Shays's Rebellion as one of the 1780s events to cover [1]. Farmers in the interior closed "
         "courthouses to stop foreclosures [2]. The requirements ask for at least three events or people from Chapters 7 "
         "and 8 [3]. The chapter does not say how much the rebellion changed any individual delegate's mind, so I would treat "
         "the link as urgency, not as a single cause.")


def test_an_honest_shays_answer_with_a_not_covered_sentence_passes():
    s, c = build()
    it = essay_item(s, c)
    assert essay.level(it, SHAYS, "How did Shays's Rebellion shape the debate over the new Constitution?", True, SHAYS_SOURCES) == 0
    assert essay.level(it, SHAYS, "", False, SHAYS_SOURCES) == 0


def test_cited_sentences_are_judged_against_their_own_chunk_not_the_paragraph():
    s, c = build()
    it = essay_item(s, c)
    cited = ("Farmers in the interior closed courthouses to stop foreclosures [2]. The prompt names Shays's Rebellion as one "
             "of the 1780s events to cover [1]. The requirements ask for at least three events or people from Chapters 7 and 8 [3].")
    uncited = "Therefore Madison's large republic proved incomplete against the parties of the 1790s, so his remedy for faction failed."
    assert essay.level(it, cited, "", True, SHAYS_SOURCES) == 0
    assert essay.level(it, cited + " " + uncited, "", True, SHAYS_SOURCES) > 0


@pytest.mark.parametrize("sent", [
    "The chapter does not say how much the rebellion changed any individual delegate's mind.",
    "The textbook does not mention what Madison thought of Shays's Rebellion in 1786.",
    "Chapter 8 does not specify how many members joined each party by 1794.",
    "The lecture slides do not cover the Anti-Federalist reply to Federalist No. 10.",
])
def test_the_chapter_does_not_say_is_not_covered_not_a_claim(sent):
    s, c = build()
    assert essay.level(essay_item(s, c), sent, "What did Madison think of Shays's Rebellion?", True) == 0, sent
    assert essay.card_hit(s, c, "Shays", sent) is None


def test_a_not_covered_lead_in_does_not_hide_a_claim_after_it():
    s, c = build()
    sent = ("The textbook does not say Madison was wrong, but his remedy for faction failed once the parties of the 1790s "
            "formed around Hamilton and Jefferson.")
    assert essay.level(essay_item(s, c), sent, ASK_Q, True) > 0


def test_an_honest_coaching_reply_passes_the_concepts_reply_in_one_call():
    s, c = build()
    it = essay_item(s, c)
    calls = []
    p = FakeProvider(lambda sy, me: calls.append(1) or SHAYS)
    blocks, used = integrity._concepts_reply(p, s, c, it, "essay", "", it["body"],
                                             "How did Shays's Rebellion shape the debate over the new Constitution?", "", 3, SHAYS_SOURCES)
    assert len(calls) == 1
    assert "does not say how much the rebellion" in " ".join(b.md for b in blocks)
    assert "did not come out right" not in " ".join(b.md for b in blocks)


def test_a_soft_leak_in_a_concepts_reply_is_still_regenerated():
    s, c = build()
    it = essay_item(s, c)
    leak = ("Madison's predictions in Federalist No. 10 about faction were undermined by the 1790s. The Hamilton-Jefferson "
            "conflict showed that parties formed in a large republic anyway.")
    calls = []
    p = FakeProvider(lambda sy, me: calls.append(1) or (leak if len(calls) == 1 else "Plan around the rubric and read the two papers first."))
    blocks, _ = integrity._concepts_reply(p, s, c, it, "essay", "", it["body"], "How do I start?", "", 0, None)
    assert len(calls) == 2 and "undermined" not in " ".join(b.md for b in blocks)


# ------------------------------------------------------------------ 6 imperative coaching
LA_Q = "What would a good thesis for Essay 1 look like?"


def test_imperative_coaching_that_names_the_rubrics_examples_is_not_argument():
    s, c = la_course()
    it = la_item(s, c)
    src = [{"text": "Essay 1 rubric: evidence from at least two lectures, for example Fibonacci numbers or Markov chains."}]
    sent = "Then pick two lecture examples, such as Fibonacci numbers or Markov chains, and ask what each one shows [1]."
    assert essay.level(it, sent, LA_Q, True, src) == 0
    assert essay.level(it, sent, LA_Q, True) == 0
    assert essay.level(it, "First, choose the idea you would defend and list the lecture examples that show it.", LA_Q, True) == 0


def test_an_imperative_that_carries_a_claim_is_still_judged():
    s, c = build()
    it = essay_item(s, c)
    assert essay.level(it, "Pick the view that Madison was wrong about the parties of the 1790s and his remedy failed.", ASK_Q, True) > 0


# ------------------------------------------------------------------ 7 feedback post-check
LA_DRAFT = ("Eigenvectors matter because a repeated process settles down when the dominant eigenvalue is small, and "
            "the Fibonacci numbers and Markov chains from lecture both show it.")
FB_LEAKS = [
    "Your thesis: eigenvalue magnitudes decide whether a repeated process settles down or blows up, so eigenvectors tell you "
    "the long-run behavior of the system.",
    "Isn't the real point that the dominant eigenvalue decides what happens to a repeated process in the long run?",
    "What if the real point is that the dominant eigenvalue decides what happens to a repeated process in the long run?",
    "You could say that the dominant eigenvalue decides what happens to any repeated process, which gives you the long-run story.",
]
FB_FINE = [
    "Which sentence of your draft is the claim, and could a reader disagree with it?",
    "You wrote \"settles down when the dominant eigenvalue is small\". Where do the lecture notes say that [1]?",
    "The rubric rewards a clear, arguable thesis. Does yours take a side?",
    "Where in your draft do you use two examples from lecture, as the assignment asks [1]?",
]


@pytest.mark.parametrize("reply", FB_LEAKS)
def test_feedback_rejects_a_thesis_whose_claim_contains_you_and_question_forms(reply):
    s, c = la_course()
    assert essay.feedback_leaks(LA_DRAFT, la_item(s, c), reply), reply


@pytest.mark.parametrize("reply", FB_FINE)
def test_honest_feedback_sentences_pass(reply):
    s, c = la_course()
    assert not essay.feedback_leaks(LA_DRAFT, la_item(s, c), reply), reply


CHEM = {"title": "Problem 3: Empirical and molecular formulas",
        "body": ("Explain the difference between an empirical formula and a molecular formula. Give one example where they "
                 "differ, and describe the relationship between their molar masses."), "kind": "essay"}
CHEM_DRAFT = ("An empirical formula is the simplest ratio and a molecular formula is the real count, for example glucose "
              "is CH2O and C6H12O6.")


def test_restating_the_assignments_requirements_is_not_a_new_sentence():
    reply = ("What does the assignment ask for? A difference between the two formulas, one example where they differ, and "
             "the relationship between the molar masses [1]\n\nWhich of those three does your draft cover?")
    assert not essay.feedback_leaks(CHEM_DRAFT, CHEM, reply, [{"text": CHEM["body"]}])
    assert not essay.feedback_leaks(CHEM_DRAFT, CHEM, reply)


def test_new_chemistry_content_in_feedback_is_still_a_leak():
    reply = "The molecular formula is a whole-number multiple of the empirical formula, so the molar mass of the molecule is n times larger."
    assert essay.feedback_leaks(CHEM_DRAFT, CHEM, reply)


def test_the_fallback_questions_fit_the_item_kind():
    txt = essay.rubric_questions(CHEM, "essay")
    assert "Which passage or event" not in txt and "event" not in txt.lower()
    assert "difference between an empirical formula" in txt.lower()
    es = essay.rubric_questions({"title": "Essay 1", "body": LA_BODY_FOR_FALLBACK, "kind": "essay"}, "essay")
    assert "Where in your draft does it meet this: A clear, arguable thesis" in es
    pr = essay.rubric_questions({"title": "P", "body": "Prove it.", "kind": "proof"}, "proof")
    assert "proof" in pr


LA_BODY_FOR_FALLBACK = "Write an essay.\n\n## Rubric\n\n- A clear, arguable thesis (30 points)\n- One counterargument (20 points)"


# ------------------------------------------------------------------ 8 thesis cards
def test_a_thesis_card_with_a_short_claim_is_refused_and_a_method_card_is_added():
    s, c = build()
    mcp_server.set_store(s)
    try:
        bad = mcp_server.add_flashcards(c, [{"front": "What should the thesis say?",
                                              "back": "That Madison's diagnosis fit the 1780s but his remedy failed by 1794"}])
        assert bad.startswith("NOT ADDED")
        ok = mcp_server.add_flashcards(c, [{"front": "What makes a thesis arguable?",
                                             "back": "A claim a reasonable reader could dispute, backed by evidence."}])
        assert ok.startswith("Added")
        ok2 = mcp_server.add_flashcards(c, [{"front": "What does Federalist No. 10 say about factions?",
                                              "back": "Madison says a large republic holds more interests, so a majority is harder to form."}])
        assert ok2.startswith("Added")
    finally:
        mcp_server.set_store(None)


def test_honest_attributed_background_is_fine():
    s, c = build()
    it = essay_item(s, c)
    txt = ("Federalist No. 10 argues that a large republic holds more interests and parties. Federalist No. 51 argues that "
           "ambition must be made to counteract ambition through the separation of powers.")
    assert essay.level(it, txt, "What are the papers about?", False) == 0
    mcp_server.set_store(s)
    try:
        assert mcp_server.save_guide(c, "answer", "Papers", txt + " [1]", [chunk_of(s, c, "Federalist No. 10")]).startswith("Saved")
    finally:
        mcp_server.set_store(None)


# ------------------------------------------------------------------ 9 honest look-alikes next to the leaks
@pytest.mark.parametrize("text", [
    "Note that the rubric weights evidence at 25 points.",
    "Remember that Chapter 8 covers the 1790s and the Hamilton-Jefferson conflict.",
    "Consider how Federalist No. 10 defines faction.",
    "As you can see, the assignment asks for three events from Chapters 7 and 8.",
    "It is worth noting that the paper is due Friday.",
    "Check that your thesis answers whether Madison's predictions were right or wrong.",
    "Look at how Hamilton and Jefferson disagreed in Chapter 8 and ask what that shows about parties.",
    "A thesis needs a claim, evidence and reasoning; check your claim is arguable and that each paragraph cites chapter 7 or 8.",
])
def test_honest_framed_and_imperative_coaching_is_level_zero(text):
    s, c = build()
    assert essay.level(essay_item(s, c), text, ASK_Q, True) == 0, text
