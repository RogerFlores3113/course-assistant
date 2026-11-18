"""Cycle 12 essay fixes (decisions 3, 4, 5): a pick or thesis cue plus a named candidate idea is a position, hedged opening
lead-ins and two-sided verdicts are positions, thesis flashcards are judged front and back together, and honest cited history,
source summaries and plain coaching are kept. Every leak test has an honest look-alike next to it. Original fixture text."""
import re

import pytest

from course_assistant import essay, grounding, mcp_server
from course_assistant.providers import FakeProvider

from test_fx5_essay import FED10, build, chunk_of, essay_item
from test_fx9_essay import open_item

LIB_SRC = [{"text": "Public libraries lent books for free to anyone with a card. Paperbacks sold for a few cents."}]
Q = "What thesis should I use for Essay 3?"


def lib_hit(it, text, q=Q, sources=LIB_SRC):
    return bool(essay.level(it, text, q, False, sources))


# ------------------------------------------------------------------ decision 3: a pick cue and a named idea
PICKS = [
    "Go with libraries as your thesis.",
    "You could argue for paperbacks.",
    "Tutor: Your thesis could be night schools.",
    "Your thesis could be libraries, because every later change builds on it [1].",
    "Pick libraries.",
    "I'd pick paperbacks as the idea to defend.",
    "My pick would be night schools.",
    "Defend libraries, since later changes build on it.",
    "Choose paperbacks and argue that cost mattered most.",
    "Your thesis: libraries.",
    "How about saying libraries did the most?",
    "A reader might expect you to claim paperbacks did the most.",
]


@pytest.mark.parametrize("text", PICKS)
def test_a_pick_cue_and_a_named_idea_is_a_position(text):
    s, c, it = open_item()
    assert lib_hit(it, text), text
    assert lib_hit(it, text, ""), text
    assert essay.guide_hit(s, c, "Notes", text, LIB_SRC), text


HONEST_PICKS = [
    "Public libraries lent books for free to anyone with a card. [1]",
    "The reading describes libraries first and paperbacks second. [1]",
    "Which idea do you think matters most?",
    "Which idea do you think matters most, libraries, paperbacks or night schools?",
    "Pick the idea you can defend with two readings.",
    "Pick one of the three changes and defend it with two readings.",
    "Pick libraries, paperbacks or night schools, whichever you can defend with two readings.",
    "Choose the change you find most convincing, then test it against the counterargument.",
    "Defend your pick with two readings.",
    "You could argue for whichever change the readings support best.",
    "Start with a sentence on what each change did, then pick your own example.",
    "Your thesis should take a side and cite two readings.",
    "Your thesis could be any of the three changes, as long as you can defend it.",
]


@pytest.mark.parametrize("text", HONEST_PICKS)
def test_honest_questions_lists_and_coaching_are_not_picks(text):
    s, c, it = open_item()
    assert not lib_hit(it, text), text
    assert essay.guide_hit(s, c, "Notes", text, LIB_SRC) is None, text


# ------------------------------------------------------------------ decision 3: hedged opening lead-ins
LEADS = [
    "One way to begin: Paperbacks widened access on paper but not in small towns.",
    "If I were you I'd open with: Libraries were only partly the answer.",
    "Your opening line could read: Libraries helped on paper but not in practice.",
    "I'd open with: libraries opened doors for the poor.",
    "One way to begin: libraries did the most for readers.",
]


@pytest.mark.parametrize("text", LEADS)
def test_a_hedged_opening_lead_in_before_a_claim_is_a_position(text):
    s, c, it = open_item()
    assert lib_hit(it, text), text
    assert essay.guide_hit(s, c, "Notes", text, LIB_SRC), text


HONEST_LEADS = [
    "One way to begin: ask which change you would defend.",
    "One way to begin: list what each reading says about libraries [1].",
    "Your opening line could read as a question you answer by the end.",
    "I'd open with a short question that names the three changes.",
    "One way to begin: read the prompt twice and underline the word extent.",
]


@pytest.mark.parametrize("text", HONEST_LEADS)
def test_an_opening_instruction_is_coaching(text):
    s, c, it = open_item()
    assert not lib_hit(it, text), text
    assert essay.guide_hit(s, c, "Notes", text, LIB_SRC) is None, text


# ------------------------------------------------------------------ decision 3: contrast verdicts, even cited, even with no cue
CONTRAST = [
    "Libraries were right about access but wrong about cost.",
    "Libraries worked on paper but not in practice [1].",
    "Libraries were only partly right.",
    "Tutor: Libraries were only partly right. [1]",
    "Paperbacks helped in the cities but not in small towns.",
]


@pytest.mark.parametrize("text", CONTRAST)
def test_a_two_sided_verdict_on_the_essays_question_is_a_position(text):
    s, c, it = open_item()
    assert lib_hit(it, text), text
    assert essay.guide_hit(s, c, "Notes", text, LIB_SRC), text


HONEST_CONTRAST = [
    "Public libraries lent books for free to anyone with a card [1].",
    "Paperbacks sold for a few cents [1].",
    "The reading says libraries lent books but not in every town.",
    "Ask yourself whether libraries were right about access but wrong about cost.",
    "Which change was right about access but wrong about cost?",
    "A strong thesis says what was right and what was wrong, so name both.",
    "The rubric rewards a thesis that is right about one thing but wrong about another only if you defend it.",
]


@pytest.mark.parametrize("text", HONEST_CONTRAST)
def test_facts_reading_notes_questions_and_coaching_are_not_verdicts(text):
    s, c, it = open_item()
    assert not lib_hit(it, text), text
    assert essay.guide_hit(s, c, "Notes", text, LIB_SRC) is None, text


def test_the_history_course_hedged_leads_and_verdicts_are_held_in_ask_save_guide_and_one_line_tutor_form():
    s, c = build()
    it = essay_item(s, c)
    texts = [
        "One way to begin: Madison's large republic controlled faction on paper but not in the 1790s.",
        "If I were you I'd open with: Madison was only partly right about faction.",
        "Madison was right about Shays but wrong about the parties of the 1790s. [1]",
        "Tutor: Madison was right about 1786 but wrong about the 1790s parties. [1]",
        "Go with the large republic as your thesis.",
    ]
    for t in texts:
        assert essay.level(it, t, "How should I start my Federalist essay?", False, [{"text": FED10}]), t
        assert essay.guide_hit(s, c, "Notes", t, [{"text": FED10}]), t


def test_the_history_course_honest_replies_pass():
    s, c = build()
    it = essay_item(s, c)
    honest = [
        "Madison argues in No. 10 that a large republic controls faction better than a small one. [1]",
        "The structure of No. 10 is: the problem of faction, then two cures, then the republic. [1]",
        "Which idea do you think matters most, the diagnosis or the remedy?",
        "Once you can say what he predicted about faction in No. 10, the question becomes whether the first party system fits that prediction [2].",
    ]
    for t in honest:
        assert essay.level(it, t, "How should I start my Federalist essay?", True, [{"text": FED10}, {"text": FED10}]) == 0, t


# ------------------------------------------------------------------ decision 4: thesis cards, front and back together
CARD_LEAKS = [
    ("What should Essay 3 argue?", "Libraries did the most for readers"),
    ("What should Essay 3 argue?", "Libraries."),
    ("What should Essay 3 argue?", "That libraries were only partly right"),
    ("Essay thesis: which idea is most important?", "Paperbacks."),
    ("Essay thesis: which idea is most important?", "Paperbacks"),
    ("Essay 3 main claim", "Libraries were only partly right"),
    ("Reading essay: main claim", "Paperbacks widened access on paper but not in practice."),
]


@pytest.mark.parametrize("front,back", CARD_LEAKS)
def test_a_thesis_card_is_judged_front_and_back_whatever_the_wording_and_punctuation(front, back):
    s, c, it = open_item()
    assert essay.card_hit(s, c, front, back), (front, back)


CARD_OK = [
    ("What is a thesis statement?", "A sentence that states the main claim of a paper."),
    ("What makes a thesis arguable?", "A claim a reasonable reader could dispute, backed by evidence."),
    ("Which change did Essay 3 name?", "Libraries, paperbacks, night schools"),
    ("What does the Essay 3 rubric weight most?", "The thesis, 30 points."),
    ("What is a library card?", "Permission to borrow books."),
    ("Essay 3 length?", "1000 words"),
    ("How many readings does Essay 3 need?", "Two."),
]


@pytest.mark.parametrize("front,back", CARD_OK)
def test_concept_requirement_and_option_cards_are_still_added(front, back):
    s, c, it = open_item()
    assert essay.card_hit(s, c, front, back) is None, (front, back)


def test_a_refused_batch_names_every_refused_card():
    s, c, it = open_item()
    mcp_server.set_store(s)
    try:
        r = mcp_server.add_flashcards(c, [
            {"front": "What is a library card?", "back": "Permission to borrow books."},
            {"front": "What should Essay 3 argue?", "back": "Libraries."},
            {"front": "Essay 3 main claim", "back": "Libraries were only partly right"},
        ])
    finally:
        mcp_server.set_store(None)
    assert r.startswith("NOT ADDED")
    assert "card 2" in r and "card 3" in r and "card 1 " not in r
    assert "No cards were added" in r


# ------------------------------------------------------------------ decision 5: over-blocks
@pytest.fixture
def mcp():
    s, c = build()
    mcp_server.set_store(s)
    yield s, c
    mcp_server.set_store(None)


def test_save_guide_keeps_a_cited_source_summary_and_plain_coaching(mcp):
    s, c = mcp
    cid = chunk_of(s, c, "Federalist No. 10")
    body = ("Madison argues in No. 10 that a large republic controls faction better than a small one. [1]\n\n"
            "Federalist 10 is one of the two readings the essay prompt names. [1]")
    assert mcp_server.save_guide(c, "primer", "What No. 10 says", body, [cid]).startswith("Saved guide")
    coaching = "Start with a sentence on what each reading argues, then pick your own example."
    assert mcp_server.save_guide(c, "primer", "Plan", coaching, [cid]).startswith("Saved guide")


def test_save_guide_still_refuses_a_summary_its_cited_chunk_does_not_back(mcp):
    s, c = mcp
    ch7 = chunk_of(s, c, "Chapter 7")
    r = mcp_server.save_guide(c, "primer", "Reading", "Madison argues that a large republic controls the effects of faction better than a small one [1].", [ch7])
    assert r.startswith("NOT SAVED")


def test_a_refusal_quotes_the_sentence_without_a_space_before_the_full_stop(mcp):
    s, c = mcp
    r = mcp_server.save_guide(c, "answer", "Position", "Madison was only partly right about faction [1].", [chunk_of(s, c, "Federalist No. 10")])
    assert r.startswith("NOT SAVED")
    assert " ." not in r and ' ."' not in r


HIST = ("Shays' Rebellion was an uprising of indebted farmers in Massachusetts in 1786 [1]. So it mattered because it made the "
        "Articles look too weak to keep order.")
HIST_SRC = [{"text": "Shays' Rebellion in 1786 was an uprising of indebted farmers in Massachusetts who shut the courts."}]


def test_an_uncited_causal_closer_after_cited_history_is_not_an_essay_leak():
    s, c = build()
    it = essay_item(s, c)
    assert essay.level(it, HIST, "Why did Shays' Rebellion matter?", True, HIST_SRC) == 0
    # the same history with a verdict in the closer is still held
    assert essay.level(it, HIST.replace("So it mattered because it made", "So it proved Madison was right, because it made"),
                       "Why did Shays' Rebellion matter?", True, HIST_SRC)


def test_ask_keeps_an_honest_cited_history_answer_with_a_causal_closer():
    s, c = build()
    out = grounding.ask(s, c, "Why did Shays' Rebellion matter?", FakeProvider(lambda sy, me: HIST), route=False)
    shown = re.sub(r"\s+", " ", out.body_md)
    assert "too weak to keep order" in shown
    assert "too close to your graded" not in shown


def test_the_mark_as_graded_notice_never_says_twin_problems_for_an_essay():
    assert "twin" not in essay.graded_notice_tail([{"title": "Essay 1", "body": "Write an essay arguing which idea matters most.", "kind": "essay"}])
    assert "coach" in essay.graded_notice_tail([{"title": "Essay 1", "body": "Write an essay arguing which idea matters most.", "kind": "essay"}])
    assert "twin" not in essay.graded_notice_tail([])


@pytest.mark.parametrize("text", [
    "Go with the large republic as your thesis.",
    "One way to begin: Madison's large republic controlled faction on paper but not in the 1790s.",
    "Madison was right about Shays but wrong about the parties of the 1790s [1].",
])
def test_ask_never_shows_a_pick_a_hedged_lead_or_a_verdict(text):
    s, c = build()
    COACH = "Plan around the rubric [1]. Ask yourself which passage of Federalist No. 10 you would test your claim against."

    def blind(system, messages):
        return COACH if "THE STUDENT'S GRADED TASK" in messages[0]["content"] else text

    out = grounding.ask(s, c, "How should I start my Federalist essay?", FakeProvider(blind), route=True)
    shown = re.sub(r"\s+", " ", out.body_md)
    core = re.sub(r"\s*\[\d\]", "", text.split(":")[-1]).strip()[:30]
    assert core not in shown, shown
