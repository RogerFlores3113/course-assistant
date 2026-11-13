"""Cycle 11 essay fixes (decisions 7, 8, 9): a thesis cue followed by a short clause about the essay's topic is a position
whatever its length (chat, feedback, save_guide, cards); new plain cues and suffix cues; a bare pasted draft under a header
routes to feedback and the heading never repeats it; save_guide names the sentence that triggered it, causal background
passes, and a restatement of what the assignment asks is not replaced. Every leak test has an honest look-alike.
Original fixture text."""
import pytest

import re

from course_assistant import essay, grounding, integrity, mcp_server, web_helpers
from course_assistant.providers import FakeProvider

from test_fx5_essay import build, chunk_of, essay_item
from test_fx9_essay import DRAFT, open_item
from test_fx10_essay import CH, READ

LIB_SRC = [{"text": "Public libraries lent books for free to anyone with a card. Paperbacks sold for a few cents."}]


def lib_hit(it, text, sources=None):
    return bool(essay.level(it, text, "", False, sources) or essay.feedback_leaks("My draft says libraries matter.", it, text, sources))


# ------------------------------------------------------------------ decision 7: short clauses after a thesis cue
SHORT_OPEN = [
    "Perhaps your thesis is that libraries mattered most.",
    "Perhaps your thesis is that libraries mattered most [1].",
    "Try framing it as: libraries widened access.",
    "Try framing it as: public libraries widened who could read [1].",
    "A good claim: public libraries are the key idea [1].",
    "The real point is that paperbacks changed everything [1].",
    "Here is a thesis you could use: night schools did the most.",
    "The argument I'd make is that libraries did the most.",
    "A strong essay would say that libraries did the most.",
    "That could be your thesis: libraries did the most.",
    "Does public libraries matter most? That could be your thesis.",
    "I'd say libraries did the most, and that is your thesis.",
    "Tutor: Libraries did the most for readers.",
]


@pytest.mark.parametrize("text", SHORT_OPEN)
def test_a_short_clause_after_a_thesis_cue_is_a_position(text):
    s, c, it = open_item()
    assert essay.level(it, text, "", False, None), text
    assert essay.level(it, text, "What thesis should I use for Essay 3?", False, None), text


@pytest.mark.parametrize("text", SHORT_OPEN)
def test_the_same_short_cues_are_caught_in_feedback_and_guides(text):
    s, c, it = open_item()
    assert essay.feedback_leaks("My draft says libraries matter.", it, text, LIB_SRC), text
    assert essay.guide_hit(s, c, "Notes", text, LIB_SRC), text


HONEST_OPEN = [
    "A good claim takes a side and cites two readings.",
    "A good claim: it takes a side and cites two readings.",
    "Perhaps your thesis is that it needs a clearer claim.",
    "Perhaps your thesis is that evidence matters.",
    "Here is a thesis you could use: ask yourself which change you would defend.",
    "Here is a thesis you could use: it should name one of the three changes.",
    "Try framing it as: a question you can answer with two readings.",
    "A strong essay would say what it wants the reader to believe.",
    "Which change did the most? Whichever you pick, that could be your thesis once you can defend it with two readings.",
    "Which change did the most? That could be your thesis.",
    "That could be your thesis, so test it against the rubric.",
    "Tutor: Which change do you think did the most?",
    "Tutor: The rubric weights the thesis at 30 points.",
]


@pytest.mark.parametrize("text", HONEST_OPEN)
def test_honest_coaching_about_a_claim_still_passes(text):
    s, c, it = open_item()
    assert essay.level(it, text, "", False, None) == 0, text
    assert not essay.feedback_leaks("My draft says libraries matter.", it, text, LIB_SRC), text
    assert essay.guide_hit(s, c, "Notes", text, LIB_SRC) is None, text


SHORT_FED = [
    "Perhaps your thesis is that the remedy failed.",
    "A good claim: Madison was partly right.",
    "Tutor: Madison was right about 1786 but wrong about the 1790s.",
    "Tutor: Madison was right about 1786 but wrong about the 1790s [1].",
    "Here is a thesis you could use: the remedy worked.",
    "Try framing it as: parties proved him wrong.",
    "That could be your thesis: Madison misjudged parties.",
    "The argument I'd make is that parties beat the remedy.",
]


@pytest.mark.parametrize("text", SHORT_FED)
def test_short_cues_on_a_named_history_essay(text):
    s, c = build()
    it = essay_item(s, c)
    assert essay.level(it, text, "", False, READ), text
    assert essay.feedback_leaks(DRAFT, it, text, READ), text


def test_a_single_line_tutor_answer_about_the_essay_is_prose_but_a_logistics_answer_is_not():
    s, c = build()
    it = essay_item(s, c)
    assert essay.level(it, "Tutor: Madison was right about 1786 and wrong about 1790.", "", False, None)
    for ok in ("A: Friday, in class.", "Tutor: It is due Friday at 11:59 pm.", "Tutor: Five pages, double-spaced."):
        assert essay.level(it, ok, "", False, None) == 0, ok


@pytest.fixture
def mcp():
    s, c = build()
    mcp_server.set_store(s)
    yield s, c
    mcp_server.set_store(None)


@pytest.mark.parametrize("back", ["Perhaps your thesis is that the remedy failed.",
                                  "A good claim: Madison was partly right.",
                                  "Does the remedy matter most? That could be your thesis."])
def test_cards_with_a_short_thesis_back_are_refused(mcp, back):
    s, c = mcp
    assert mcp_server.add_flashcards(c, [{"front": "What should the Federalist essay argue?", "back": back}]).startswith("NOT ADDED"), back


def test_cards_that_define_things_are_kept(mcp):
    s, c = mcp
    out = mcp_server.add_flashcards(c, [{"front": "What is a faction?", "back": "A group of citizens united by a shared passion."},
                                        {"front": "What does a good claim need?", "back": "A good claim takes a side and cites two readings."}])
    assert not out.startswith("NOT ADDED"), out


def test_save_guide_refuses_a_short_cue_and_saves_a_coaching_guide(mcp):
    s, c = mcp
    fed = chunk_of(s, c, "Federalist No. 10")
    assert mcp_server.save_guide(c, "explain", "Ideas", "Perhaps your thesis is that the remedy failed [1].", [fed]).startswith("NOT SAVED")
    ok = ("A good claim takes a side and cites two readings [1]. Ask yourself which passage of Federalist No. 10 you would test "
          "your claim against.")
    assert not mcp_server.save_guide(c, "explain", "Plan", ok, [fed]).startswith("NOT SAVED")


# ------------------------------------------------------------------ decision 8: a bare pasted draft routes to feedback
BARE = [
    "Here's what I have so far: Madison argued that a large republic would control faction better than a small one. I think the "
    "1790s parties show he was only partly right about that.",
    "Here is my thesis: Madison was only partly right because the large republic did not stop parties from forming in the 1790s.",
    "Here's what I have so far for Essay 2: Madison argued that a large republic would control faction. I think parties showed "
    "he was only partly right.",
    "My intro for Essay 2: Madison argued that a large republic controls faction but parties still formed.",
]


@pytest.mark.parametrize("q", BARE)
def test_a_header_and_a_pasted_draft_with_no_question_is_draft_feedback(q):
    s, c = build()
    assert essay.draft_request(q), q
    hit = integrity.match_graded(s, c, q)
    assert hit and hit["intent"] == "feedback" and "Federalist" in hit["title"], q
    heading = web_helpers.answer_heading(q, {})
    assert heading == essay.paste_heading(q) and len(heading.split()) <= 10 and "Madison" not in heading, heading


def test_a_ten_word_paste_with_a_draft_cue_routes():
    paste = "Madison argued a large republic tames faction, yet parties formed."
    assert len(paste.split()) == 10
    q = f"My intro so far: {paste} What would make it stronger?"
    assert essay.draft_request(q) == paste
    s, c = build()
    assert integrity.match_graded(s, c, q)["intent"] == "feedback"


@pytest.mark.parametrize("q", [
    "Here's the assignment for Essay 2: Write 1000 words arguing how far Federalist Nos. 10 and 51 matched what happened.",
    "Here is my problem: Find the eigenvalues of the matrix with rows 2 1 and 1 2 and then report them both.",
    "Here's what I have so far: Which of the three changes did the most? Why did libraries matter more than schools in 1900?",
    "Here is my thesis: still working on it",
    "Here is the syllabus: the essay is due in November and counts for a quarter of the grade in this course.",
])
def test_an_assignment_a_problem_a_list_of_questions_or_a_stub_is_not_a_draft(q):
    assert essay.draft_request(q) is None, q


def test_a_header_paste_with_a_trailing_ask_keeps_its_ask_wording():
    q = ("Here is my draft for Essay 2: Madison argued that a large republic controls faction. Parties still formed. "
         "Feedback please, and what would a good thesis claim?")
    d = essay.draft_request(q)
    assert d and d.endswith("Parties still formed.")


def test_an_essay_check_gets_no_twin_instruction_over_mcp(mcp):
    s, c = mcp
    out = mcp_server.get_twin_instructions(c, "Is this okay for Essay 2: Madison was right about faction but parties still formed "
                                            "in the 1790s?")
    assert out.startswith("GRADED") and "never write a twin" in out.lower()
    plain = mcp_server.get_twin_instructions(c, "Is my answer to Problem 2 right? x = 3")
    assert "Write a twin and tell them" not in out and plain is not None


# ------------------------------------------------------------------ decision 9: naming, background, restatements
def test_save_guide_names_the_sentence_that_triggered_it_not_the_background(mcp):
    s, c = mcp
    fed = chunk_of(s, c, "Federalist No. 10")
    body = ("Shays' Rebellion was an uprising of indebted farmers in Massachusetts. It mattered because it suggested to men of "
            "property that the republic was falling into chaos and the weak Articles could not fix it.")
    out = mcp_server.save_guide(c, "explain", "Shays notes", body, [fed])
    if out.startswith("NOT SAVED"):  # when refused, the quoted sentence is the one that judges, not the opener
        quoted = out.split("The sentence that matched:", 1)[1]
        assert "It mattered because" in quoted and "uprising of indebted farmers in Massachusetts." not in quoted
    hit = essay.guide_hit(s, c, "Shays notes", body, READ)
    assert hit is None or ("It mattered because" in hit["sentence"] and "uprising" not in hit["sentence"])


@pytest.mark.parametrize("body", [
    "Federalist No. 10 was published in 1787 under the name Publius. It was written in support of ratifying the new Constitution "
    "and appeared in New York newspapers.",
    "Shays' Rebellion was an uprising of indebted farmers in Massachusetts during 1786 and 1787 who shut the courts. It was put "
    "down by a private militia raised by merchants because the state had no money to pay an army of its own.",
])
def test_plain_background_with_a_causal_because_passes(body):
    s, c = build()
    assert essay.guide_hit(s, c, "Background", body, READ) is None


def test_background_around_a_real_verdict_is_still_refused():
    s, c = build()
    body = ("Shays' Rebellion was an uprising of indebted farmers in Massachusetts during 1786 and 1787 who shut the courts. "
            "Madison was wrong about parties because the large republic could not stop Hamilton and Jefferson from organizing "
            "supporters in the 1790s, and the faction problem returned.")
    hit = essay.guide_hit(s, c, "Background", body, READ)
    assert hit and "Madison was wrong" in hit["sentence"] and "uprising" not in hit["sentence"]


@pytest.mark.parametrize("reply", [
    "The prompt asks how far Federalist Nos. 10 and 51 matched what actually happened in the new republic.",
    "The prompt asks how far Madison was right about faction.",
    "The assignment asks you to judge how far Federalist Nos. 10 and 51 matched what happened afterwards.",
    "Essay 2 wants to know whether the remedy for faction worked in the 1790s.",
    "The prompt asks to what extent Madison's argument fit the 1780s.",
])
@pytest.mark.parametrize("q", ["Can you help me with Essay 2? I need a thesis.", ""])
def test_a_restatement_of_what_the_assignment_asks_is_not_replaced(reply, q):
    s, c = build()
    it = essay_item(s, c)
    assert essay.level(it, reply, q, False, None) == 0, reply
    assert essay.level(it, reply + " Which passage will you test it against?", q, False, None) == 0, reply


@pytest.mark.parametrize("reply", [
    "The prompt asks whether Madison was right, and I think he was only partly right about faction in the 1790s.",
    "Madison was only partly right about how well a large republic controls faction, as the 1790s parties show.",
])
def test_a_restatement_with_a_position_in_it_is_still_replaced(reply):
    s, c = build()
    assert essay.level(essay_item(s, c), reply, "", False, None) > 0, reply


# ------------------------------------------------------------------ end to end
def _blind(leak):
    coach = "Plan around the rubric [1]. Ask yourself which passage of Federalist No. 10 you would test your claim against."

    def f(system, messages):
        return coach if "THE STUDENT'S GRADED TASK" in messages[0]["content"] else leak
    return f


@pytest.mark.parametrize("text", ["Perhaps your thesis is that the remedy failed [1].", "A good claim: Madison was partly right [1].",
                                  "Here is a thesis you could use: parties proved him wrong."])
def test_ask_never_shows_a_short_thesis_cue(text):
    s, c = build()
    a = grounding.ask(s, c, "What thesis should I use for Essay 2?", FakeProvider(_blind(text)), route=True)
    shown = re.sub(r"\s+", " ", a.body_md)
    assert "remedy failed" not in shown and "partly right" not in shown and "proved him wrong" not in shown, shown


def test_ask_keeps_honest_coaching_that_restates_the_task():
    s, c = build()
    honest = ("The prompt asks how far Federalist Nos. 10 and 51 matched what actually happened in the new republic [1]. "
              "A good claim takes a side and cites two readings. Which passage would you test your claim against?")
    a = grounding.ask(s, c, "Can you help me with Essay 2? I need a thesis.", FakeProvider(_blind(honest)), route=True)
    assert "The prompt asks how far" in a.body_md
