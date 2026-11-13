"""Hotfix pass hx-essay: generalised model openings, hedged feedback suggestions, frame plus tail in any order, the cited
Shays's Rebellion false block, and signpost lead-ins that are not boxed. Original text."""
import re

import pytest

from course_assistant import essay, grounding, integrity
from course_assistant.providers import FakeProvider

from test_fx5_essay import build, essay_item
from test_fx7_essay import SHAYS_SOURCES

CLAIM = "Madison's theory fit the 1780s but could not anticipate the party system of the 1790s."
ASK_Q = "Write the introduction paragraph for my Federalist essay"


def blind(leak):
    return lambda sy, me: ("Plan around the rubric. Ask yourself which passage you would test your claim against."
                           if "THE STUDENT'S GRADED TASK" in me[0]["content"] else leak)


# ------------------------------------------------------------------ 1 generalised model openings
OPENINGS = [
    "A thesis you could use: " + CLAIM, "A thesis you can use: " + CLAIM, "A claim you could make: " + CLAIM,
    "A claim to consider: " + CLAIM, "Try this: " + CLAIM, "Try something along the lines of: " + CLAIM,
    "Here is a thesis: " + CLAIM, "One position: " + CLAIM, "Something like: " + CLAIM, "Another angle worth trying: " + CLAIM,
    "Verdict: partly right.", "Short answer: half right.",
    "Student: Was Madison right about faction? Tutor: Partly. His theory fit the 1780s but missed the party system of the 1790s.",
]
HONEST_COACHING = [
    "A good thesis should take a side.",
    "Try this: write one sentence that takes a side on Madison, then test it against two passages.",
    "Here is how to write a thesis: pick a side first and make it arguable.",
    "One position to avoid is a thesis that only summarizes the papers.",
    "A thesis you could use is one that takes a side the rubric can reward.",
    "A claim to consider testing: does your thesis survive the strongest counterexample?",
    "Student: How do I start my thesis? Tutor: Pick a side first, then check it against the rubric.",
    "Verdict: your draft needs a second quoted passage.",
]


@pytest.mark.parametrize("text", OPENINGS)
def test_model_openings_before_a_claim_are_hard_hits(text):
    s, c = build()
    it = essay_item(s, c)
    assert essay.level(it, text) == 2, text
    assert essay.level(it, text, ASK_Q, True) == 2, text


@pytest.mark.parametrize("text", OPENINGS)
def test_model_openings_never_reach_the_page_in_chat(text):
    s, c = build()
    for route in (True, False):
        a = grounding.ask(s, c, ASK_Q, FakeProvider(blind(text)), route=route)
        assert CLAIM[:30] not in re.sub(r"\s+", " ", a.body_md) and "partly right" not in a.body_md.lower(), (route, text)


@pytest.mark.parametrize("text", HONEST_COACHING)
def test_coaching_about_how_to_write_one_is_not_a_hit(text):
    s, c = build()
    it = essay_item(s, c)
    assert essay.level(it, text) == 0, text
    assert essay.level(it, text, ASK_Q, True) == 0, text


# ------------------------------------------------------------------ 2 hedged suggestions in draft feedback
HEDGED = [
    "Perhaps your thesis is that " + CLAIM.rstrip("."),
    "What if your thesis said that Madison's theory fit the 1780s but missed the party system of the 1790s? Would that be arguable?",
    "Why not argue that Madison's theory fit the 1780s but missed the party system of the 1790s?",
    "Maybe you could argue that Madison's theory fit the 1780s but missed the party system of the 1790s.",
]
HEDGED_HONEST = [
    "Perhaps tighten the thesis so it names a side.",
    "What if your thesis named a side?",
    "Why not add a second quote from Federalist No. 10?",
    "Perhaps your thesis is too broad, so name the side you take.",
    "Maybe you could argue with more evidence from the second paper.",
]


@pytest.mark.parametrize("text", HEDGED)
def test_hedged_thesis_suggestions_in_feedback_are_caught(text):
    s, c = build()
    assert essay.feedback_leaks("My draft says things about faction.", essay_item(s, c), text)


@pytest.mark.parametrize("text", HEDGED_HONEST)
def test_hedged_coaching_in_feedback_is_fine(text):
    s, c = build()
    assert not essay.feedback_leaks("My draft says things about faction.", essay_item(s, c), text)


# ------------------------------------------------------------------ 3 frame plus tail, in any order
TAILED = [
    "As you can see, Madison's remedy for faction was only partly successful because the parties of the 1790s formed anyway, "
    "which a student should note.",
    "Consider that Madison's remedy for faction was only partly successful because the parties of the 1790s formed anyway, as a "
    "student should note.",
    "It is worth noting that Madison's remedy for faction was only partly successful because the parties of the 1790s formed "
    "anyway (students should note this).",
]


@pytest.mark.parametrize("text", TAILED)
def test_frame_plus_tail_is_judged_like_the_claim_alone(text):
    s, c = build()
    it = essay_item(s, c)
    alone = essay.level(it, essay.trim_tail(essay.strip_frames(text)[0]))
    assert alone >= 1
    assert essay.level(it, text) >= alone, text
    assert essay.level(it, text, ASK_Q, True) == 2, text


def test_trim_tail_keeps_the_frame_and_never_eats_it():
    t = "As you can see, parties formed anyway in the 1790s, which a student should note."
    assert essay.trim_tail(t) == "As you can see, parties formed anyway in the 1790s"
    assert essay.strip_frames(essay.trim_tail(t))[0] == "parties formed anyway in the 1790s"


def test_frame_plus_tail_on_real_coaching_stays_fine():
    s, c = build()
    t = "As you can see, the requirements ask for a counterargument, which a student should note."
    assert essay.level(essay_item(s, c), t) == 0


# ------------------------------------------------------------------ 4 the cited Shays's Rebellion answer in a touched chat
SHAYS_CITED = ("Farmers in the interior closed courthouses in 1786 to stop foreclosures and debt suits [2]. The prompt names "
               "Shays's Rebellion as one of the 1780s events to cover [1]. The requirements ask for at least three events or "
               "people from Chapters 7 and 8 [3].")


def test_an_honest_cited_answer_is_not_a_sticky_leak():
    s, c = build()
    it = essay_item(s, c)
    q = "How did Shays's Rebellion shape the debate over the new Constitution?"
    assert integrity.check_reply(s, c, q, SHAYS_CITED, force_item=it, sources=SHAYS_SOURCES) is None


def test_the_same_paragraph_with_cites_that_do_not_back_it_is_still_held():
    s, c = build()
    it = essay_item(s, c)
    q = "How did Shays's Rebellion shape the debate over the new Constitution?"
    leak = ("Madison's predictions in Federalist No. 10 about faction were undermined by the 1790s. The Hamilton-Jefferson "
            "conflict showed that parties formed in a large republic anyway, and the first party system revealed that "
            "ambition did not cancel out. Shays's Rebellion had already shown the weakness of the Articles of Confederation.")
    assert integrity.check_reply(s, c, q, leak, force_item=it, sources=SHAYS_SOURCES)
    assert integrity.check_reply(s, c, q, leak + " [1] [2] [3]", force_item=it, sources=SHAYS_SOURCES)


def test_a_cited_soft_sentence_passes_the_concepts_retry_but_an_uncited_one_does_not():
    s, c = build()
    it = essay_item(s, c)
    cited = ("Shays's Rebellion matters to the debate because the chapter ties it to urgency about the Articles [1]. Farmers in "
             "the interior closed courthouses in 1786 [2]. The requirements ask for three events from Chapters 7 and 8 [3].")
    calls = []
    p = FakeProvider(lambda sy, me: calls.append(1) or cited)
    blocks, _ = integrity._concepts_reply(p, s, c, it, "essay", "", it["body"],
                                          "How did Shays's Rebellion shape the debate?", "", 3, SHAYS_SOURCES)
    assert len(calls) == 1 and "urgency about the Articles" in " ".join(b.md for b in blocks)
    uncited = ("Madison's predictions in Federalist No. 10 about faction were undermined by the 1790s. The Hamilton-Jefferson "
               "conflict showed that parties formed in a large republic anyway.")
    calls.clear()
    p = FakeProvider(lambda sy, me: calls.append(1) or (uncited if len(calls) == 1 else "Plan around the rubric."))
    blocks, _ = integrity._concepts_reply(p, s, c, it, "essay", "", it["body"], "How do I start?", "", 0, None)
    assert len(calls) == 2 and "undermined" not in " ".join(b.md for b in blocks)


# ------------------------------------------------------------------ 5 lead-ins are not boxed as "Not from your materials"
@pytest.mark.parametrize("raw", [
    "On AI, the syllabus says you may not use it to write the essay or to produce an outline for graded work. Grammar checks "
    "are allowed [1].",
    "On AI, the syllabus says you may not use it to write the essay. Ask me if unsure. Grammar checks are allowed [1].",
    "Measured against the rubric, here is what I see. Your paragraph names a claim but gives one quote [1].",
    "Here is my read of the draft. Your paragraph names a claim [1].",
    "Grammar checks are allowed [1]. In short, first decide whether a tool is writing or only checking.",
])
def test_lead_ins_and_advice_recaps_stay_plain(raw):
    blocks, _ = grounding.parse_blocks(raw, 3)
    assert [b.kind for b in blocks] == ["text"], blocks


@pytest.mark.parametrize("raw", [
    "Madison was born in Texas in 1801. The syllabus allows AI [1].",
    "The syllabus allows AI [1]. In short, Madison was wrong about parties and the Senate.",
    "Here is what I see: the Constitution was signed in 1790 by the king of France. The syllabus allows AI [1].",
])
def test_a_real_uncited_claim_is_still_boxed(raw):
    blocks, _ = grounding.parse_blocks(raw, 3)
    assert "outside" in [b.kind for b in blocks], blocks
