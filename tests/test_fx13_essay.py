"""Cycle 13 essay fixes (decisions 7, 8 and the thesis and card half of 9): an example sentence on a concept request is never essay
prose; cited reading facts and historiography survive Explain and Practice; coaching after a colon is coaching; a cited pick or
bare verdict is a position; thesis cards are judged front and back. Every leak test has an honest look-alike next to it.
Original fixture text only."""
import pytest

from course_assistant import essay, integrity, mcp_server

from test_fx5_essay import FED10, build, chunk_of, essay_item
from test_fx9_essay import open_item

LIB_SRC = [{"text": "Public libraries lent books for free to anyone with a card. Paperbacks sold for a few cents."}]
TEXT_SRC = [
    {"text": "Under the Articles of Confederation, land sales failed to produce the revenue needed to deal with the nation's economic "
             "problems. Shays' Rebellion resulted in eighteen deaths overall, but to men of property it strongly suggested the republic "
             "was falling into chaos."},
    {"text": "In the 1790s Hamilton and Jefferson organized supporters into rival parties. Some historians argue that the first party "
             "system showed how a large republic could still divide into factions."},
    {"text": FED10},
]
TOPIC_Q = "Shays' Rebellion and the Articles of Confederation"


# ------------------------------------------------------------------ decision 7: 'For example' on concept requests
CONCEPT_QS = ["What is a public library? Give me an example.", "Explain what a library card is", "Practice: public libraries and paperbacks",
              "Define a paperback"]
EXAMPLES = [
    "For example, a library card lets a reader borrow books for free from any branch in the city.",
    "For example, public libraries lent books for free to anyone with a card [1].",
    "For instance, paperbacks sold for a few cents, so a reader could own a book without borrowing it [1].",
    "e.g., a branch library lends novels, magazines and picture books to anyone with a card.",
]


@pytest.mark.parametrize("q", CONCEPT_QS)
@pytest.mark.parametrize("text", EXAMPLES)
def test_an_example_sentence_never_fires_on_a_concept_request(q, text):
    s, c, it = open_item()
    assert essay.level(it, text, q, False, LIB_SRC) == 0, (q, text)
    assert essay.level(it, text, q, True, None) == 0, (q, text)
    assert essay.guide_hit(s, c, q, text, LIB_SRC) is None, (q, text)


def test_a_cited_example_is_not_the_essays_claim_even_with_no_request():
    s, c, it = open_item()
    cited = "For example, public libraries lent books for free to anyone with a card [1]."
    assert essay.level(it, cited, "", False, LIB_SRC) == 0
    assert essay.level(it, cited, "", False, None) == 0


LEAKS = [  # the same lead-in on a request to write, or wrapping a real position, is still caught
    ("Write my thesis for Essay 3.", "For example, public libraries mattered most because anyone with a card could borrow books."),
    ("Give me a model thesis for Essay 3.", "For example, you could argue that libraries opened the reading room to poor families."),
    ("What thesis should I use for Essay 3?", "For example, you might say that paperbacks did the most because they were cheap."),
    ("What is a public library? Give me an example.", "For example, you could argue that libraries opened the reading room to poor families."),
    ("What is a public library? Give me an example.", "For example, libraries were right about access but wrong about cost."),
    ("Which change did the most for ordinary readers?", "For example, public libraries mattered most because anyone could borrow books."),
]


@pytest.mark.parametrize("q,text", LEAKS)
def test_the_example_lead_in_does_not_excuse_a_real_position(q, text):
    s, c, it = open_item()
    assert essay.level(it, text, q, False, LIB_SRC) > 0, (q, text)


def test_the_essays_own_question_asked_as_a_question_is_not_a_concept_request():
    s, c, it = open_item()
    assert not essay.concept_request("Write my intro") and not essay.concept_request("What should my thesis be for Essay 3?")
    assert essay.concept_request("What is a public library?") and not essay.concept_request("")
    assert essay._asks_own_question(it, "Which change did the most for ordinary readers?")
    assert not essay._asks_own_question(it, "What is a public library?")


def test_a_concept_request_through_the_chat_gate_keeps_the_example():
    s, c, it = open_item()
    r = "A public library lends books to anyone with a card. For example, a branch library lends novels and picture books to all."
    assert not integrity.check_reply(s, c, "What is a public library? Give me an example.", r, sources=LIB_SRC)
    cited = "For example, public libraries lent books for free to anyone with a card [1]."
    assert not integrity.check_reply(s, c, "What is a public library? Give me an example.", cited, sources=LIB_SRC)


# ------------------------------------------------------------------ decision 8: cited facts on a topic the essay names
FACTS = [
    "Under the Articles of Confederation, land sales failed to produce the revenue needed to deal with the nation's economic problems [1].",
    "Shays' Rebellion resulted in eighteen deaths overall, but to men of property it strongly suggested the republic was falling into chaos [1].",
]


@pytest.mark.parametrize("text", FACTS)
def test_a_cited_textbook_fact_on_an_essay_topic_is_kept_in_explain_and_practice(text):
    s, c = build()
    it = essay_item(s, c)
    assert essay.level(it, text, TOPIC_Q, True, TEXT_SRC) == 0
    with essay.reply_sources(TEXT_SRC):
        cleaned, n, note = integrity.scrub_parts(s, c, f"## Explain\n\n{text}\n\nMore on the topic.", TOPIC_Q)
    assert n == 0 and ("land sales" if "land sales" in text else "chaos") in cleaned
    assert essay.level(it, text, TOPIC_Q, True, None) == 0  # a cite whose chunk is unknown on a concept request


def test_a_fact_its_cited_chunk_does_not_say_is_still_held():
    s, c = build()
    it = essay_item(s, c)
    wrong = "Shays' Rebellion proved that the Articles had failed, because it made men of property fear anarchy [3]."
    assert essay.level(it, wrong, TOPIC_Q, True, TEXT_SRC) > 0
    verdict = "Madison was right about 1786 but wrong about the 1790s parties [3]."
    with essay.reply_sources(TEXT_SRC):
        cleaned, n, note = integrity.scrub_parts(s, c, f"## Explain\n\n{verdict}\n\n{FACTS[0]}", TOPIC_Q)
    assert n == 1 and "wrong about the 1790s" not in cleaned and "land sales" in cleaned


def test_only_the_position_part_of_an_explain_answer_is_dropped():
    s, c = build()
    text = (f"## What happened\n\n{FACTS[0]}\n\n{FACTS[1]}\n\n## Where it leads\n\nMadison's remedy worked on paper but not in practice.")
    with essay.reply_sources(TEXT_SRC):
        cleaned, n, note = integrity.scrub_parts(s, c, text, TOPIC_Q)
    assert n == 1 and "land sales" in cleaned and "chaos" in cleaned and "on paper" not in cleaned


COACHING = [
    "Look at what the prompt asks: how far Madison's argument in Federalist No. 10 about a large republic controlling faction held up [3].",
    "Consider: Madison argues a large republic controls faction better than a small one [3].",
    "A useful question: was Madison right that a large republic would control faction [3].",
    "Start by asking: did parties form despite Madison's remedy [2].",
    "Ask yourself: which events from Chapter 8 would test Madison's prediction?",
    "One question to consider: how did Shays' Rebellion change what the delegates feared?",
    "Think about: whether the parties of the 1790s show the remedy failed.",
]


@pytest.mark.parametrize("text", COACHING)
def test_colon_coaching_followed_by_a_question_or_the_prompt_is_coaching(text):
    s, c = build()
    it = essay_item(s, c)
    assert essay.level(it, text, "What should my thesis be for Essay 2?", False, TEXT_SRC) == 0, text
    assert essay.guide_hit(s, c, "Coaching notes", text, TEXT_SRC) is None, text


COLON_LEAKS = [
    "Consider: Madison was right about faction but wrong about the parties of the 1790s [3].",
    "Look at what the prompt asks: Madison's remedy worked on paper but not in practice [3].",
    "A useful question: is it not true that Madison was wrong about the parties of the 1790s?",
    "Start by saying that Madison was only partly right about faction.",
    "Consider this thesis: Madison was half right about faction.",
]


@pytest.mark.parametrize("text", COLON_LEAKS)
def test_a_colon_lead_in_in_front_of_a_verdict_is_still_a_position(text):
    s, c = build()
    it = essay_item(s, c)
    assert essay.level(it, text, "What should my thesis be for Essay 2?", False, TEXT_SRC) > 0, text


def test_a_cited_historiographic_view_is_background_when_its_chunk_says_it():
    s, c = build()
    it = essay_item(s, c)
    ok = "Many historians would say the first party system showed how a large republic could still divide into factions [2]."
    assert essay.level(it, ok, "What should my thesis be for Essay 2?", False, TEXT_SRC) == 0
    assert essay.guide_hit(s, c, "Reading notes", ok, TEXT_SRC) is None
    assert integrity.check_reply(s, c, "What should my thesis be for Essay 2?", ok, sources=TEXT_SRC) is None


def test_historiography_is_not_a_way_to_pass_a_verdict_nobody_cited_or_the_chunk_does_not_say():
    s, c = build()
    it = essay_item(s, c)
    verdict = "Many historians would say Madison was right about faction but wrong about the parties of the 1790s"
    assert essay.level(it, verdict + ".", "What should my thesis be for Essay 2?", False, TEXT_SRC) > 0  # no cite
    assert essay.level(it, verdict + " [1].", "What should my thesis be for Essay 2?", False, TEXT_SRC) > 0  # chunk 1 does not say it


@pytest.fixture
def mcp():
    s, c = build()
    mcp_server.set_store(s)
    yield s, c
    mcp_server.set_store(None)


def test_save_guide_and_chat_agree_on_the_same_sentences(mcp):
    s, c = mcp
    ch = chunk_of(s, c, "Chapter 7")
    fed = chunk_of(s, c, "Federalist No. 10")
    for text, cites in [(COACHING[0], [fed]), (COACHING[3], [fed]), (COACHING[1], [fed])]:
        text = text.replace("[3]", "[1]").replace("[2]", "[1]")
        out = mcp_server.save_guide(c, "primer", "Coaching notes", text, cites)
        assert out.startswith("Saved guide"), (text, out)
    for text in (COLON_LEAKS[0].replace("[3]", "[1]"), "Madison was right about 1786 but wrong about the 1790s parties [1]."):
        out = mcp_server.save_guide(c, "primer", "Notes", text, [fed])
        assert out.startswith("NOT SAVED"), (text, out)
        assert integrity.check_reply(s, c, "What should my thesis be for Essay 2?", text, sources=[{"text": FED10}])


def test_an_essay_refusal_never_says_twin_and_names_every_sentence(mcp):
    s, c = mcp
    fed = chunk_of(s, c, "Federalist No. 10")
    body = ("Madison was right about 1786 but wrong about the 1790s parties [1].\n\nGo with the failure of the remedy as your thesis [1].\n\n"
            "Madison argues that a large republic controls the effects of faction better than a small one [1].")
    out = mcp_server.save_guide(c, "primer", "Essay 2 notes", body, [fed])
    assert out.startswith("NOT SAVED") and "twin" not in out.lower() and "different numbers" not in out
    assert "wrong about the 1790s" in out and "failure of the remedy" in out and "controls the effects" not in out


# ------------------------------------------------------------------ decision 9: a cited pick and a bare cited verdict
PICKS = [
    "Go with the failure of the remedy as your thesis. [1]",
    "Go with the idea that parties formed anyway as your thesis. [1]",
    "Pick the idea that parties formed anyway.",
    "Pick the idea that Madison's remedy failed in practice. [1]",
    "Choose the weakness of the Articles as your main claim.",
    "Madison was right about 1786 but wrong about the 1790s parties. [1]",
    "His theory worked on paper but not in practice. [1]",
]


@pytest.mark.parametrize("text", PICKS)
def test_a_cited_pick_or_bare_cited_verdict_is_a_position_in_ask_and_save_guide(mcp, text):
    s, c = mcp
    it = essay_item(s, c)
    fed = chunk_of(s, c, "Federalist No. 10")
    assert essay.level(it, text, "What should my thesis be for Essay 2?", False, TEXT_SRC) == 2, text
    assert integrity.check_reply(s, c, "What should my thesis be for Essay 2?", text, sources=TEXT_SRC), text
    assert mcp_server.save_guide(c, "primer", "Notes", text, [fed]).startswith("NOT SAVED"), text


HONEST_PICKS = [
    "Pick the idea that you can defend with two readings.",
    "Pick the idea that is best supported by the readings.",
    "Go with whichever idea you can defend as your thesis.",
    "Go with the claim you find most convincing as your thesis.",
    "Choose the reading you will quote first.",
    "Pick two events from Chapter 8 and say what each one shows.",
    "Madison argues that a large republic controls the effects of faction better than a small one [3].",
    "Shays' Rebellion made men of property fear anarchy [1].",
]


@pytest.mark.parametrize("text", HONEST_PICKS)
def test_pick_look_alikes_that_hand_over_nothing_pass(mcp, text):
    s, c = mcp
    it = essay_item(s, c)
    assert essay.level(it, text, "What should my thesis be for Essay 2?", False, TEXT_SRC) == 0, text


def test_a_bare_verdict_that_its_chunk_really_says_is_a_reading_note():
    s, c = build()
    it = essay_item(s, c)
    src = [{"text": "Hamilton and Jefferson organized rival parties by 1794. Madison was right about 1786 but wrong about the parties of "
                    "the 1790s, as a later textbook put it."}]
    assert essay.level(it, "Madison was right about 1786 but wrong about the parties of the 1790s [1].", "What does the textbook say?",
                       False, src) == 0


THESIS_CARDS = [
    ("Essay 2: best argument", "Faction"),
    ("Which idea should the Federalist essay center on?", "Checks and balances in No. 51"),
    ("Which argument should my Federalist essay build on?", "Parties"),
    ("Essay 2 strongest claim", "Checks and balances"),
]


@pytest.mark.parametrize("front,back", THESIS_CARDS)
def test_thesis_cards_with_a_bare_pick_are_refused(mcp, front, back):
    s, c = mcp
    out = mcp_server.add_flashcards(c, [{"front": front, "back": back}])
    assert out.startswith("NOT ADDED"), (front, back, out)


READING_CARDS = [
    ("Federalist 10 main claim", "A large republic controls the effects of faction"),
    ("Federalist No. 10 main argument", "Madison argues a large republic dilutes faction"),
    ("Chapter 7 main idea", "The Articles left the national government unable to raise revenue"),
    ("What does Federalist 10 define a faction as?", "A group united by a common passion or interest adverse to the rights of others"),
    ("What is a thesis statement?", "A sentence that states the main claim of a paper"),
    ("Essay 2 requirements", "A thesis, two Federalist passages, three events and a counterargument"),
]


@pytest.mark.parametrize("front,back", READING_CARDS)
def test_reading_fact_cards_are_saved(mcp, front, back):
    s, c = mcp
    out = mcp_server.add_flashcards(c, [{"front": front, "back": back}])
    assert out.startswith("Added"), (front, back, out)


def test_a_reading_front_does_not_launder_a_verdict(mcp):
    s, c = mcp
    out = mcp_server.add_flashcards(c, [{"front": "Federalist 10 main claim", "back": "Madison was right about faction but wrong about parties"}])
    assert out.startswith("NOT ADDED")


# ------------------------------------------------------------------ a terse cited finished answer to a short written item
SHORT_BODY = ("Short answer 2: In two or three sentences, explain how a lending library differs from a reference library, name one "
              "example of each, and say how their loan rules differ.")
SHORT_SRC = [{"text": "A lending library lets members take books home. A reference library keeps its books on the premises. The city "
                      "branch is a lending library, while the archive reading room is a reference library. Lending libraries set loan "
                      "periods and fines, but reference libraries allow no loans at all."}]


def short_item():
    from course_assistant.store import Store
    s = Store(":memory:")
    c = s.add_course("Reading History")
    s.add_graded_item(c, "Short answer 2", SHORT_BODY, None, kind="essay", label="Short answer 2")
    return s, c


def test_cited_sentences_that_together_answer_every_part_of_a_short_item_are_the_answer():
    s, c = short_item()
    leak = ("A lending library lets members take books home, while a reference library keeps its books on the premises [1]. The city "
            "branch is a lending library and the archive reading room is a reference library [1]. Lending libraries set loan periods "
            "and fines, but reference libraries allow no loans at all [1].")
    assert essay.guide_hit(s, c, "Short answer notes", leak, SHORT_SRC)
    assert essay.card_hit(s, c, "Short answer 2", leak, SHORT_SRC[0]["text"])


def test_one_cited_definition_or_a_longer_concept_guide_is_not_a_short_answer():
    s, c = short_item()
    assert essay.guide_hit(s, c, "Libraries", "A lending library lets members take books home [1].", SHORT_SRC) is None
    assert essay.guide_hit(s, c, "Libraries", "A reference library keeps its books on the premises [1]. Ask yourself what a reader "
                           "gains from each kind of library, and where you would look for a rare map.", SHORT_SRC) is None


def test_a_verdict_free_example_with_no_request_is_a_plain_sentence_and_a_verdict_example_is_not():
    s, c, it = open_item()
    assert essay.level(it, "For example, a library card lets a reader borrow books for free from any branch in the city.", "", False, None) == 0
    assert essay.level(it, "For example, public libraries mattered most because anyone could borrow books for free.", "", False, None) == 2
