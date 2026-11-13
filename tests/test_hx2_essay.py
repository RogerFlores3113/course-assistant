"""Hotfix pass hx2-essay: structural lead-ins in front of a position (Ask and draft feedback), essay thesis cards through MCP,
honest cited 'because' and 'so' sentences in a touched chat, and the concepts retry keeping the chapter sources. Original text."""
import re

import pytest

from course_assistant import essay, grounding, integrity, mcp_server
from course_assistant.providers import FakeProvider

from test_fx5_essay import build, essay_item

CLAIM = "Madison's theory fit the 1780s but could not anticipate the party system of the 1790s."
CORE = CLAIM.rstrip(".")
ASK_Q = "Write the introduction paragraph for my Federalist essay"
DRAFT = "My draft says things about faction."

# ------------------------------------------------------------------ 1 structural lead-ins
LEADS = [
    "Suppose your thesis were that " + CORE + ".",
    "If your thesis were that " + CORE + ", the essay would be arguable.",
    "Imagine your claim is that " + CORE + ".",
    "If you took the side that " + CORE + ", what would you quote?",
    "A strong thesis here would read: " + CLAIM,
    "Think of it as: " + CLAIM,
    "Pick this side: " + CLAIM,
    "Your thesis could take this form: " + CLAIM,
    "Here is how it could go: " + CLAIM,
    "In a sentence, the idea would be that " + CORE + ".",
    "Verdict: your thesis should say Madison was partly right about faction.",
    "Student: Was Madison right? Tutor: Partly. " + CLAIM,
    "Q: Was Madison right about faction? A: Partly; " + CORE + ".",
    "Q: Was Madison right about faction? A: Partly. His theory fit the 1780s but missed the party system of the 1790s.",
]
HONEST = [
    "A thesis should take a side that the rubric can reward.",
    "Suppose your thesis were too broad; how would you narrow it?",
    "If your thesis were that broad, you would need a narrower claim.",
    "Imagine your claim is vague; ask what a reader could disagree with.",
    "Think of it as: a claim plus two reasons.",
    "Pick this side of the debate only if you can quote two passages.",
    "Here is what a thesis must do: take a side the paper can defend.",
    "A strong thesis here would read as one sentence with a side and a reason.",
    "Your thesis could take a clearer form if it named the side.",
    "Verdict: your draft needs a second quoted passage.",
    "Q: How do I start my thesis? A: Pick a side and test it against two passages.",
    "Student: How do I start? Tutor: Name a side first, then check it against the rubric.",
    "The rubric says that the thesis counts for 30 points and must take a side.",
]


@pytest.mark.parametrize("text", LEADS)
def test_any_lead_in_before_a_position_is_judged_as_the_position(text):
    s, c = build()
    it = essay_item(s, c)
    assert essay.level(it, text) == 2, text
    assert essay.level(it, text, ASK_Q, True) == 2, text
    assert essay.feedback_leaks(DRAFT, it, text), text


@pytest.mark.parametrize("text", LEADS)
def test_lead_in_positions_never_reach_the_page(text):
    s, c = build()
    blind = lambda sy, me: ("Plan around the rubric. Ask yourself which passage you would test your claim against."
                            if "THE STUDENT'S GRADED TASK" in me[0]["content"] else text)
    for route in (True, False):
        a = grounding.ask(s, c, ASK_Q, FakeProvider(blind), route=route)
        assert "party system of the 1790s" not in re.sub(r"\s+", " ", a.body_md), (route, text)


@pytest.mark.parametrize("text", HONEST)
def test_lead_ins_before_coaching_stay_at_level_zero(text):
    s, c = build()
    it = essay_item(s, c)
    assert essay.level(it, text) == 0, text
    assert essay.level(it, text, ASK_Q, True) == 0, text
    assert not essay.feedback_leaks(DRAFT, it, text), text


def test_lead_core_gives_the_clause_after_the_lead_in_and_nothing_for_source_reports():
    assert essay.lead_core("Think of it as: " + CLAIM).startswith("Madison's theory fit")
    assert essay.lead_core("It seems to me that " + CLAIM).startswith("Madison's theory fit")
    assert essay.lead_core("Madison argues that a large republic controls the effects of faction.") == ""
    assert essay.lead_core("The textbook says that parties formed in the 1790s around Hamilton and Jefferson.") == ""


# ------------------------------------------------------------------ 2 MCP essay thesis cards
@pytest.fixture
def mcp():
    s, c = build()
    mcp_server.set_store(s)
    yield s, c
    mcp_server.set_store(None)


THESIS_CARDS = [
    ("Thesis for the essay", "Madison was half right"),
    ("My thesis", "Parties show Madison underestimated faction"),
    ("Position for essay 2", "Madison was partly right."),
    ("What is my thesis?", "Madison was only partly right."),
    ("Essay answer", "Partly right"),
    ("Verdict", "Madison was wrong"),
    ("Argument for Essay 2", "Half right"),
    ("How should I answer the essay?", "Say that Madison got it half right."),
]


@pytest.mark.parametrize("front,back", THESIS_CARDS)
def test_mcp_refuses_a_short_thesis_card(mcp, front, back):
    s, c = mcp
    assert mcp_server.add_flashcards(c, [{"front": front, "back": back}]).startswith("NOT ADDED"), (front, back)
    assert s.flashcards(c) == [] if hasattr(s, "flashcards") else True


@pytest.mark.parametrize("front,back", [
    ("What is a thesis?", "A claim a paper defends with evidence."),
    ("What does a thesis need?", "A side the reader could dispute."),
    ("Verdict on the rubric", "Thesis counts for 30 points."),
    ("What is a faction?", "A group united by a passion or interest against others."),
    ("Where is the essay due?", "Check the assignment page."),
])
def test_mcp_still_adds_honest_cards_next_to_a_thesis_word(mcp, front, back):
    s, c = mcp
    assert mcp_server.add_flashcards(c, [{"front": front, "back": back}]).startswith("Added"), (front, back)


# ------------------------------------------------------------------ 3 sticky chat, cited because/so sentences
SRC = [
    {"text": "Delegates argued about taxes and western land. Shays' Rebellion in 1786 made men of property fear that the "
             "republic was sliding toward anarchy, and it pushed many of them to support a stronger central government."},
    {"text": "A small society has few distinct parties, so a majority is easily formed."},
    {"text": "Madison argues that a large republic is better placed than a small one to control the effects of faction "
             "because it holds more interests and parties."},
    {"text": "Madison says the causes of faction cannot be removed without destroying liberty, so its effects must be "
             "controlled."},
]
Q = "How did Shays' Rebellion shape the debate over the new Constitution?"
HONEST_STICKY = [
    "Shays' Rebellion mattered to the constitutional debate because it made men of property fear the republic was sliding into "
    "anarchy [1].",
    "Third, he argues that a large republic is better placed to control those effects than a small one [3][3].",
    "Madison argues in Federalist No. 10 that the causes of faction cannot be removed, so its effects must be controlled [4].",
]


@pytest.mark.parametrize("raw", HONEST_STICKY)
def test_an_honest_cited_because_or_so_sentence_is_not_a_sticky_leak(raw):
    s, c = build()
    it = essay_item(s, c)
    assert integrity.check_reply(s, c, Q, raw, force_item=it, sources=SRC) is None, raw


def test_the_same_sentences_with_cites_the_chunks_do_not_back_are_still_held():
    s, c = build()
    it = essay_item(s, c)
    wrong = "Madison's predictions about faction were undermined by the 1790s because a large republic could not stop parties [2]."
    assert integrity.check_reply(s, c, Q, wrong, force_item=it, sources=SRC)
    wrong2 = "Madison's remedy for faction was only partly successful because the parties of the 1790s formed anyway [1]."
    assert integrity.check_reply(s, c, Q, wrong2, force_item=it, sources=SRC)
    uncited = HONEST_STICKY[0].replace(" [1]", "")
    assert integrity.check_reply(s, c, Q, uncited, force_item=it, sources=SRC)


def test_a_verdict_hidden_in_a_cited_clause_is_still_held():
    s, c = build()
    it = essay_item(s, c)
    raw = ("Madison's whole diagnosis of faction was undermined by Hamilton and Jefferson in the 1790s, because a large republic "
           "holds more interests and parties [3].")
    assert integrity.check_reply(s, c, Q, raw, force_item=it, sources=SRC)


# ------------------------------------------------------------------ 4 the concepts retry keeps the chapter sources
def test_the_graded_route_keeps_the_original_chapter_sources():
    s, c = build()
    it = essay_item(s, c)
    chapter = {"chunk_id": 987654, "material_id": 0, "material_title": "Chapter 7", "text": "Shays' Rebellion frightened men of "
               "property in 1786 and made them fear anarchy.", "loc_label": "pages 1 to 3", "score": 1.0}
    seen = []
    p = FakeProvider(lambda sy, me: seen.append(me[0]["content"]) or "Plan around the rubric [1].")
    integrity.twin(s, c, Q, p, dict(it, reason="sticky"), "sticky", keep_sources=[chapter])
    assert seen and "frightened men of property" in seen[0] and "SOURCES:" in seen[0]
    assert "THE STUDENT'S GRADED TASK" in seen[0]
    seen.clear()
    integrity.twin(s, c, Q, p, dict(it, reason="sticky"), "sticky")
    assert seen and "frightened men of property" not in seen[0]
