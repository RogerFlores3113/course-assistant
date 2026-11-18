"""Cycle 9 essay fixes (decisions 6 and 7): a question or a cited reading note after a lead-in is never a position, and a
labelled thesis ('Thesis idea:', 'Your claim:', 'The real point is that', 'Essay 2 thesis' cards, dialogue forms) is one whatever
words it is made of. Every leak test has an honest look-alike next to it. Original fixture text."""
import pytest

from course_assistant import essay, mcp_server
from course_assistant.store import Store

from test_fx5_essay import FED10, build, chunk_of, essay_item

DRAFT = "In Federalist No. 10, Madison argues that a large republic controls faction better, but parties still formed."
CHUNK = (FED10 + " A faction is a group of citizens united by a common passion or interest adverse to the rights of others. "
         "The latent causes of faction are sown in the nature of man. The causes of faction cannot be removed, so its effects "
         "must be controlled. A republic offers the cure because it passes the public views through chosen representatives.")
SRC = [{"text": CHUNK}]
UNRELATED = [{"text": "Delegates met in taverns to argue about postage rates, canal tolls and the price of salt in the western "
                      "counties. Wagon drivers carried mail between the river towns twice a week."}]

OPEN_BODY = ("Essay 3: Which change did the most for ordinary readers?\n\nWrite 1000 words arguing which of three changes "
             "(public libraries, cheap paperbacks or night schools) did the most to widen who could read in the early 1900s. "
             "Take a side, use two readings, address one counterargument.\n\n- Thesis (30 points)\n- Evidence from the readings "
             "(40 points)\n- Counterargument (30 points)")


def open_item():
    s = Store(":memory:")
    c = s.add_course("Reading History")
    s.add_graded_item(c, "Essay 3: Which change did the most for ordinary readers?", OPEN_BODY, None, kind="essay", label="Essay 3")
    m = s.add_material(c, "reading", "Library reading")
    s.replace_chunks(m, [{"text": "Public libraries lent books for free to anyone with a card. Paperbacks sold for a few cents.",
                          "loc_kind": "section", "loc_start": 1, "loc_label": "page 1"}])
    from course_assistant import integrity
    return s, c, next(i for i in integrity.graded_view(s, c) if i["kind"] == "essay")


def hit(it, text, sources=SRC, question=""):
    return bool(essay.level(it, text, question, False, sources) or essay.feedback_leaks(DRAFT, it, text, sources))


# ------------------------------------------------------------------ 1 questions after a lead-in
QUESTIONS = [
    "Questions for your revision: Which of Madison's predictions do you think held in the 1780s, and what from Chapter 7 shows it [1]?",
    "Ask yourself: which of Madison's predictions held in the 1780s, and what from Chapter 7 shows it?",
    "Questions for your revision: which part of Madison's prediction failed in the 1790s?",
    "Ask yourself: did the 1790s parties arise because the remedy failed, or because the remedy worked and a new problem appeared? "
    "Which chapter 8 events would you cite for each? [1]",
    "One question to consider: was Madison wrong about parties, or did he never address them? What does No. 10 say about "
    "interest groups? [1]",
    "State a position you can argue against, then support it with two events from Chapter 8 and one from No. 51 [1].",
    "Did the 1790s parties arise because the remedy failed, or because it worked and a new problem appeared? [1]",
]


@pytest.mark.parametrize("text", QUESTIONS)
def test_a_question_or_a_method_sentence_after_a_lead_in_is_not_a_position(text):
    s, c = build()
    assert not hit(essay_item(s, c), text), text


@pytest.mark.parametrize("text", [
    "Questions for your revision: Madison's remedy was only partly successful.",
    "Ask yourself: Madison was wrong about the parties of the 1790s.",
    "Questions for your revision: Isn't the real point that Madison's remedy failed in the 1790s?",
    "Question to consider: wasn't Madison wrong about parties in the 1790s?",
    "Say that Madison's remedy failed once Hamilton and Jefferson organized parties in the 1790s.",
])
def test_a_claim_after_the_same_lead_in_is_still_a_position(text):
    s, c = build()
    assert hit(essay_item(s, c), text), text


# ------------------------------------------------------------------ 2 descriptive lead-ins and cited reading notes
READING = [
    "Structure: first he diagnoses faction [1]; second he rules out removing its causes [1]; third he offers the republic as cure [1].",
    "The structure is that first he diagnoses faction [1], then he rules out removing its causes [1], and finally he offers the "
    "republic as cure [1].",
    "Structure: first he diagnoses faction [1].",
    "Step 1, the diagnosis: Madison says the latent causes of faction are sown in the nature of man [1].",
    "The diagnosis: Madison says the latent causes of faction are sown in the nature of man [1].",
    "Part one, the definition: Madison says a faction is a group united by a common passion or interest adverse to the rights of others [1].",
    "Look for the turn in the argument: Madison moves from diagnosis to effects when he says the causes of faction cannot be removed [1].",
    "Third, the cure: a republic that works through chosen representatives, which he contrasts with the small society [1]. "
    "He then argues that a large republic controls the effects of faction better than a small one [1].",
]


@pytest.mark.parametrize("text", READING)
def test_a_cited_summary_of_the_reading_after_a_lead_in_is_not_a_position(text):
    s, c = build()
    it = essay_item(s, c)
    assert not hit(it, text), text
    assert essay.soft_leaks(it, text, "", SRC) == 0, text


@pytest.mark.parametrize("text", [
    "Structure: first he diagnoses faction; second he proves Madison's remedy failed in the 1790s.",
    "Step 1, the diagnosis: Madison was wrong about parties because the 1790s showed his remedy failed [1].",
    "The structure is that Madison was right about 1786 and wrong about the parties of the 1790s [1].",
])
def test_a_verdict_after_a_descriptive_lead_in_is_still_a_position(text):
    s, c = build()
    assert hit(essay_item(s, c), text), text


@pytest.mark.parametrize("text", [
    "Structure: first he diagnoses faction [1]; second he rules out removing its causes [1].",
    "Step 1, the diagnosis: Madison says the latent causes of faction are sown in the nature of man [1].",
])
def test_the_same_summary_citing_a_chunk_that_does_not_cover_it_is_still_judged(text):
    # the reading note is exempt only while the chunk it cites really covers it; with an unrelated chunk it falls back to the
    # ordinary judgement (no verdict word, so only a lexical hit on the old rule would show it) and with no cite it never gets
    # the exemption at all
    s, c = build()
    it = essay_item(s, c)
    assert not essay.lead_exempt(text, UNRELATED)
    assert essay.lead_exempt(text, SRC)
    assert not essay.lead_exempt(text.replace(" [1]", ""), SRC)


# ------------------------------------------------------------------ 3 labelled theses
LABELLED = [
    "Thesis idea: the remedy failed in practice.",
    "Thesis: the large republic did not stop parties from forming.",
    "Your thesis: the large republic did not prevent parties from forming.",
    "Your claim: the remedy only partly worked in the 1790s.",
    "Your thesis should be that the large republic did not stop parties.",
    "Your thesis should be that the remedy failed in practice.",
    "The real point is that the large republic did not prevent parties.",
    "Isn't the real point that the large republic did not prevent parties from forming?",
    "What if the real point is that the remedy only partly worked in the 1790s?",
    "Consider that Madison's large republic did not prevent the parties of the 1790s from forming, which weakens his prediction.",
    "I would say that the remedy failed in practice.",
    "You might say that the large republic did not stop parties.",
    "Your essay should say that the remedy failed in practice.",
    "**Thesis idea:** the remedy failed in practice.",
    "Tutor: The remedy failed in practice.",
]


@pytest.mark.parametrize("text", LABELLED)
def test_a_labelled_thesis_is_a_position_even_with_no_cite_and_no_name(text):
    s, c = build()
    assert hit(essay_item(s, c), text), text


@pytest.mark.parametrize("text", [
    "Thesis: the rubric weights thesis most heavily [1].",
    "Your thesis should take a side and name the evidence it rests on.",
    "Your thesis needs a claim someone could dispute.",
    "The real point of the rubric is that the paper must cite two passages.",
    "Isn't it worth checking whether your thesis takes a side?",
    "Your claim: that the paragraph needs a citation, which the rubric requires.",
    'Your thesis: "Madison argues that a large republic controls faction better." That names a source, not a side.',
    "Thesis: take a side and cite two passages.",
    "Madison's argument: factions come from human nature [1].",
    "The key point is that the rubric asks for one counterargument.",
])
def test_an_honest_sentence_that_looks_like_a_labelled_thesis_is_fine(text):
    s, c = build()
    assert not hit(essay_item(s, c), text, SRC, DRAFT), text  # the student's draft is in the question, as in Ask


def test_a_labelled_thesis_with_no_name_on_an_open_prompt():
    s, c, it = open_item()
    for t in ("Thesis idea: public libraries widened readership more than paperbacks did.",
              "Your claim: night schools reached more new readers than libraries.",
              "Your thesis should be that cheap paperbacks mattered most.",
              "The real point is that libraries reached the readers paperbacks never did."):
        assert hit(it, t, [{"text": "Public libraries lent books for free to anyone with a card."}]), t
    for t in ("Thesis: the rubric gives thirty points for a clear side.",
              "Your thesis should name which of the three changes you chose.",
              "Which of the three changes would you test first, and what would a counterexample look like?"):
        assert not hit(it, t, [{"text": "Public libraries lent books for free to anyone with a card."}]), t


# ------------------------------------------------------------------ 4 dialogue forms
@pytest.mark.parametrize("text", [
    "Student: Was Madison right? Tutor: Partly. His diagnosis fit the 1780s but the 1790s parties showed his remedy failed.",
    "Q: Was Madison right? A: Partly; his remedy failed to stop parties in the 1790s.",
    "Q: Did the remedy work? A: Mostly. It limited what any one party could do.",
    "Tutor: Madison was right about 1786 but wrong about the 1790s parties.",
])
def test_a_one_line_dialogue_is_judged_turn_by_turn(text):
    s, c = build()
    assert hit(essay_item(s, c), text), text


@pytest.mark.parametrize("text", [
    "Student: What does the rubric reward? Tutor: A thesis that takes a side and cites two passages.",
    "Q: What does the prompt ask? A: To what extent the papers reflected the 1780s.",
    "Student: Where do I start? Tutor: Pick two passages and ask what each one shows about the question.",
    "Q: When is it due? A: Friday, in class.",
])
def test_an_honest_dialogue_about_the_task_is_fine(text):
    s, c = build()
    assert not hit(essay_item(s, c), text), text


# ------------------------------------------------------------------ 5 flashcards and guides over MCP
@pytest.fixture
def mcp():
    s, c = build()
    mcp_server.set_store(s)
    yield s, c
    mcp_server.set_store(None)


@pytest.mark.parametrize("front,back", [
    ("Essay 2 thesis", "The remedy only partly worked, because parties formed anyway."),
    ("Essay 2 claim", "The large republic did not prevent parties from forming."),
    ("Essay 2 thesis idea", "Factions formed anyway after 1790."),
])
def test_a_card_that_names_the_essay_thesis_refuses_any_declarative_back(mcp, front, back):
    s, c = mcp
    assert mcp_server.add_flashcards(c, [{"front": front, "back": back}]).startswith("NOT ADDED")


@pytest.mark.parametrize("front,back", [
    ("What does the Essay 2 thesis need?", "A claim someone could dispute, backed by two passages."),
    ("Essay 2 thesis requirement", "It must take a side on to what extent the papers reflect the 1780s."),
    ("Essay 2 thesis", "Must take a side and cite two passages."),
    ("Essay 2 length", "Between 1200 and 1500 words, due Friday."),
    ("Faction", "A group united by a common passion or interest adverse to the rights of others."),
])
def test_an_honest_card_next_to_it_is_added(mcp, front, back):
    s, c = mcp
    assert not mcp_server.add_flashcards(c, [{"front": front, "back": back}]).startswith("NOT ADDED")


def test_mcp_guide_keeps_a_cited_reading_note_and_refuses_a_labelled_thesis(mcp):
    s, c = mcp
    fed = chunk_of(s, c, "Federalist No. 10")
    ok = ("Step 1, the diagnosis: Madison says a large republic controls the effects of faction better than a small one because "
          "it holds more interests and parties [1].\n\nStructure: first he compares small and large societies [1].")
    assert not mcp_server.save_guide(c, "answer", "Reading notes", ok, [fed]).startswith("NOT SAVED")
    bad = "Thesis idea: the large republic did not stop parties from forming."
    assert mcp_server.save_guide(c, "answer", "Reading notes 2", bad, [fed]).startswith("NOT SAVED")


# ------------------------------------------------------------------ 6 a quotation after a label is judged as a quotation
def test_a_quoted_thesis_that_is_not_the_students_still_leaks():
    s, c = build()
    it = essay_item(s, c)
    leak = 'Here is a thesis: "Madison was wrong about faction in the 1790s because the parties formed anyway."'
    assert essay.feedback_leaks(DRAFT, it, leak, SRC)
    assert essay.level(it, leak, "", False, SRC) == 2


# ------------------------------------------------------------------ 7 end to end: honest feedback ends with questions
def test_feedback_with_revision_questions_is_answered_in_one_call():
    from course_assistant import grounding
    from course_assistant.providers import FakeProvider
    s, c = build()
    calls = []
    reply = ('You wrote "Madison argues that a large republic controls faction better" [1], which names the source but not a '
             "side.\n\nQuestions for your revision: Which of Madison's predictions do you think held in the 1780s, and what "
             "from the chapter shows it [1]?\n\nAsk yourself: which part of the rubric does your counterargument meet [1]?")
    p = FakeProvider(lambda sy, me: calls.append(1) or reply)
    a = grounding.ask(s, c, "Here is my draft intro for Essay 2: " + DRAFT + " Can you give me feedback?", p, route=False)
    assert len(calls) == 1, calls
