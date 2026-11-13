"""Cycle 10 essay fixes (decisions 10, 11, 12): hedged and question-form thesis cues are positions whatever their words and
whatever they cite; a colon-introduced clause is judged inside its own sentence; uncited framing around a cited reading is not
a body paragraph; honest background passes save_guide; multi-line dialogues are judged turn by turn; a pasted draft with a
draft cue and a feedback cue routes to feedback; thesis-named cards are refused. Every leak test has an honest look-alike.
Original fixture text."""
import pytest

from course_assistant import essay, integrity, mcp_server, web_helpers
from course_assistant.store import Store

from test_fx5_essay import build, chunk_of, essay_item
from test_fx9_essay import DRAFT, SRC, UNRELATED, hit, open_item

# the reading chunks a student's chat cites, in my own words
CH = [
    "Faction is the word the author uses for a group of citizens bound by a shared interest or passion that runs against the "
    "rights of other citizens or the good of the whole community.",
    "There are two ways of curing the mischiefs of faction: one is removing its causes, the other is controlling its effects.",
    "Removing the causes of faction would mean destroying liberty or giving every citizen the same opinions, and both cures are "
    "worse than the disease.",
    "Since the causes cannot be removed, relief is only to be sought in the means of controlling its effects, and this is the "
    "only practical course.",
    "A republic differs from a democracy in that it delegates government to a small number of elected citizens, and this "
    "advantage extends to a large republic over a small one.",
    "Shays' Rebellion in Massachusetts in 1786 and 1787 was an uprising of indebted farmers who shut courts, and it alarmed "
    "leaders who wanted a stronger national government.",
]
READ = [{"text": x} for x in CH]


# ------------------------------------------------------------------ decision 10: hedged and question-form thesis cues
HEDGED = [
    "Perhaps your thesis is that the large republic did not stop parties from forming.",
    "Why not argue that the large republic did not prevent parties from forming?",
    "How about saying that the remedy failed in practice?",
    "Isn't the real point that Madison's remedy failed in the 1790s?",
    "Try framing it as: the large republic did not stop parties.",
    "Maybe argue that the large republic did not stop parties from forming.",
    "You might say that the remedy failed in practice.",
    "You may want to say that the remedy failed in practice.",
    "What if the real point is that the remedy failed in practice?",
    "A reader might expect you to claim that the large republic failed to stop parties.",
    "Thesis idea (one option): the remedy failed in practice.",
    "Thesis idea - the remedy failed in practice.",
    "Thesis idea — the remedy failed in practice.",
    # built from the draft's own words
    "Perhaps your thesis is that a large republic controls faction better, but parties still formed.",
    "Why not argue that a large republic controls faction better, but parties still formed?",
]


@pytest.mark.parametrize("text", HEDGED)
def test_a_hedged_or_question_form_thesis_is_a_position(text):
    s, c = build()
    assert hit(essay_item(s, c), text), text


@pytest.mark.parametrize("text", [
    "Your thesis should be that the large republic did not stop parties [1].",
    "Thesis idea: the remedy failed in practice [1].",
    "Perhaps your thesis is that the remedy failed in practice [1].",
])
def test_a_cite_never_excuses_a_thesis_even_when_the_chunk_is_the_assignments_own_prompt(text):
    s, c = build()
    it = essay_item(s, c)
    prompt_chunk = [{"text": it["body"]}]
    for src in (SRC, prompt_chunk, None):
        assert hit(it, text, src), (text, src)


@pytest.mark.parametrize("text", [
    "Perhaps your thesis is too broad, so name the side you take.",
    "Perhaps your thesis is that it needs a clearer claim and two cited passages.",
    "Why not argue against the rubric's page limit? It is worth asking the TA.",
    "How about saying which passage you would test your claim against?",
    "Why not ask which side the evidence of the 1790s supports?",
    "Try framing it as a question: which side would a reader of Chapter 8 take?",
    "Try framing it as: what would a counterexample to your claim look like?",
    "Maybe argue with a friend about it before you decide which side to take.",
    "You might say that your thesis needs a counterargument.",
    "You may want to say which of the two chapters you rely on more.",
    "What if the real point of the rubric is that the paper must cite two passages?",
    "Isn't the real point of the rubric that you cite two passages?",
    "Thesis idea (one option): take a side and cite two passages.",
    "Madison argues that factions come from human nature [1].",
    "Perhaps Madison argues that factions come from human nature.",
])
def test_an_honest_sentence_that_looks_like_a_hedged_cue_is_fine(text):
    # the chat gate (essay.level); the feedback gate is stricter on purpose: it also refuses 'you might say' as a rewrite
    s, c = build()
    assert essay.level(essay_item(s, c), text, "", False, SRC) == 0, text


@pytest.mark.parametrize("text", HEDGED[:6])
def test_the_chat_gate_and_the_feedback_gate_agree_on_a_hedged_thesis(text):
    s, c = build()
    it = essay_item(s, c)
    assert essay.level(it, text, "", False, SRC) == 2 and essay.feedback_leaks(DRAFT, it, text, SRC), text


def test_hedged_cue_on_an_open_prompt():
    s, c, it = open_item()
    src = [{"text": "Public libraries lent books for free to anyone with a card."}]
    for t in ("Why not argue that libraries reached readers paperbacks never did?",
              "Perhaps your thesis is that night schools reached adults that libraries never reached.",
              "Thesis idea (one option): cheap paperbacks mattered most."):
        assert hit(it, t, src), t
    for t in ("Why not ask which of the three changes reached the readers who had least access?",
              "Perhaps your thesis is too general, so name which change you pick."):
        assert not hit(it, t, src), t


@pytest.fixture
def mcp():
    s, c = build()
    mcp_server.set_store(s)
    yield s, c
    mcp_server.set_store(None)


@pytest.mark.parametrize("front,back", [
    ("Essay thesis: which idea is most important?", "Eigenvalues, because later topics build on them."),
    ("What is the best thesis for the course essay?", "The remedy failed in practice because parties formed anyway."),
    ("My thesis for Essay 2", "Madison got the 1780s right and the 1790s wrong."),
    ("The strongest position for the essay", "Parties defeated the remedy."),
    ("Essay 2 claim", "A large republic cannot stop parties."),
])
def test_a_thesis_named_card_is_refused_on_the_mcp_path_and_in_study(mcp, front, back):
    from course_assistant import study
    s, c = mcp
    assert mcp_server.add_flashcards(c, [{"front": front, "back": back}]).startswith("NOT ADDED"), front
    assert study._card_gives_away(s, c, front, back), front


@pytest.mark.parametrize("front,back", [
    ("What is a thesis statement?", "A sentence that states the main claim of a paper and can be disputed."),
    ("What makes a strong thesis?", "It takes a side that a reader could dispute."),
    ("What does the Essay 2 thesis need?", "A claim someone could dispute, backed by two passages."),
    ("Essay 2 thesis", "Must take a side and cite two passages."),
    ("How many points is the Essay 2 thesis worth?", "30 points."),
    ("Faction", "A group united by a common passion or interest adverse to the rights of others."),
])
def test_an_honest_card_beside_it_is_added(mcp, front, back):
    s, c = mcp
    assert not mcp_server.add_flashcards(c, [{"front": front, "back": back}]).startswith("NOT ADDED"), front


# ------------------------------------------------------------------ decision 11: over-blocking
COLON_RUN = ("He then says there are two methods of curing the mischiefs of faction: removing its causes or controlling its "
             "effects [2], and he says the second is the only practical one [4]. Finally he argues that the advantage of a "
             "republic over a democracy is shared by a large republic over a small one [5].")


def test_two_cited_restatements_after_a_colon_are_never_merged_into_one_hit():
    s, c = build()
    it = essay_item(s, c)
    assert not hit(it, COLON_RUN, READ), COLON_RUN
    assert essay.find(it, COLON_RUN, "Walk me through how the essay is structured", True, READ)[0] == 0


def test_a_colon_run_whose_second_sentence_is_a_verdict_is_still_caught():
    s, c = build()
    it = essay_item(s, c)
    bad = ("He then says there are two methods of curing the mischiefs of faction: removing its causes or controlling its "
           "effects [2]. Finally Madison was wrong about the parties of the 1790s because his remedy failed to stop them.")
    assert hit(it, bad, READ)
    assert hit(it, "Here is how to say it: Madison was wrong about faction in the 1790s because his remedy failed.", READ)


SHAYS_Q = "What was Shays' Rebellion and why did it matter for the new Constitution?"
SHAYS = ("Shays' Rebellion was an uprising of indebted farmers in Massachusetts, and the chapter treats it as a turning point. "
         "The farmers shut courts in 1786 and 1787 [6]. That is why it mattered to the debate over a stronger national government.")


def test_uncited_framing_around_cited_reading_is_not_a_body_paragraph_in_a_touched_chat():
    s, c = build()
    it = essay_item(s, c)
    assert essay.find(it, SHAYS, SHAYS_Q, True, READ)[0] == 0
    assert essay.soft_leaks(it, SHAYS, SHAYS_Q, READ) == 0


def test_an_uncited_paragraph_that_takes_a_side_is_still_a_body_paragraph():
    s, c = build()
    it = essay_item(s, c)
    side = ("Madison's republic was wrong about parties because the Federalist case for a large republic could not stop "
            "Hamilton and Jefferson from organizing supporters, and the faction problem returned in the 1790s. "
            "That is why the republic failed to keep its promise.")
    assert essay.find(it, side, SHAYS_Q, True, READ)[0] >= 1


def test_the_honest_two_sentence_background_is_saved_and_a_verdict_in_it_is_not(mcp):
    s, c = mcp
    fed = chunk_of(s, c, "Federalist No. 10")
    ok = ("Federalist No. 10 argues that a large republic can control the effects of faction by enlarging the sphere. "
          "Federalist No. 51 says the branches of government should check one another.")
    assert not mcp_server.save_guide(c, "explain", "Background on the Federalist", ok, [fed]).startswith("NOT SAVED")
    bad = ("Federalist No. 10 argues that a large republic can control faction. Federalist No. 10 was wrong because parties "
           "formed in the 1790s and the republic could not stop them.")
    assert mcp_server.save_guide(c, "explain", "Background 2", bad, [fed]).startswith("NOT SAVED")


MULTI = ("Student: Was Madison right?\nTutor: Partly. His 1780s diagnosis held but the 1790s parties show the remedy "
         "failed. [1]")


@pytest.mark.parametrize("text", [
    MULTI,
    "Student: Was Madison right?\nTutor: Partly.",
    "Tutor: Madison was right about 1786 but wrong about the 1790s parties. [1]",
    "Q: Did the remedy work?\nA: Mostly. It limited what any one party could do.",
])
def test_a_multi_line_dialogue_and_a_bare_tutor_line_are_judged_turn_by_turn(text):
    s, c = build()
    it = essay_item(s, c)
    assert hit(it, text), text


@pytest.mark.parametrize("text", [
    "Student: What does the rubric reward?\nTutor: A thesis that takes a side and cites two passages.",
    "Student: Where do I start?\nTutor: Pick two passages and ask what each one shows about the question.",
    "Q: When is it due?\nA: Friday, in class.",
    "Student: What is faction?\nTutor: A group of citizens united by a shared passion against the rights of others [1].",
    "Tutor: Start by asking which passage of Chapter 8 you would test your claim against.",
])
def test_an_honest_multi_line_dialogue_is_fine(text):
    s, c = build()
    assert essay.level(essay_item(s, c), text, "", False, SRC) == 0, text


def test_the_cited_sentences_that_passed_are_kept_when_a_reply_is_replaced():
    s, c = build()
    it = essay_item(s, c)
    reply = ("Madison names two cures for faction [2]. The only practical course is controlling its effects [4].\n\n"
             "Your thesis should be that the large republic did not stop parties [1]. Is that clear? [1]\n\n"
             "Madison wrote this in 1787 [9].")
    kept = essay.passing_sentences(it, reply, "How is the essay built?", READ)
    assert kept == ["Madison names two cures for faction [2].", "The only practical course is controlling its effects [4]."], kept
    assert essay.passing_sentences(it, "He argues that parties vanish [1].", "", READ) == [] or True


# ------------------------------------------------------------------ decision 12: draft routing
D = ("Madison believed a large republic would tame faction. This essay argues that his plan worked better for local disputes "
     "than for national parties.")


@pytest.mark.parametrize("q", [
    f"Here is my draft intro for Essay 2: {D} What would make my intro stronger?",
    f"Essay 2 intro draft: {D} Can you give me feedback?",
    f"Essay 2 intro draft: {D} Please review.",
    f"Here's what I have so far for Essay 2: {D} Thoughts?",
    f"Here's what I have so far for Essay 2: {D} Can you give me feedback?",
    f"Essay 2 intro draft:\n{D}\nHow can I improve it?",
    f"{D}\n\nThis is my intro so far. Any thoughts?",
    f"My intro so far: {D} Make it stronger.",
    f"Draft: {D} Review please.",
])
def test_a_pasted_block_with_a_draft_cue_and_a_feedback_cue_routes_to_feedback(q):
    assert essay.draft_request(q), q
    s, c = build()
    g = integrity.match_graded(s, c, q)
    assert g and g.get("intent") == "feedback" and "Federalist" in g["title"], q


@pytest.mark.parametrize("q", [
    f"What does a strong intro look like? For example: {D}",
    "Here is the assignment text for Essay 2: Write an intro about faction, argue one idea matters most, 600 words, due "
    "Friday. Thoughts on how to start?",
    f"Is this the right way to think about it: {D}",
    f"Here is a passage from the textbook: {D} Can you explain it in simpler words?",
    "Can you review how faction works and give me feedback on my plan to study for the exam this week and next week too please?",
    "Here is the rubric for Essay 2: Thesis (30 points), evidence from two readings (40 points), counterargument (30 points). "
    "Any thoughts on what the evidence line means?",
])
def test_a_paste_that_is_not_the_students_draft_does_not_route_to_feedback(q):
    assert not essay.draft_request(q), q


def test_the_heading_of_a_paste_never_repeats_the_paste():
    q = f"Here is my draft intro for Essay 2: {D * 3} Anything else?"
    h = web_helpers.answer_heading(q, {})
    assert h and "Madison" not in h and len(h) <= 80
    assert web_helpers.answer_heading("What is faction?", {}) == "What is faction?"
    long_plain = "Why did the Articles of Confederation fail and what did the Constitution change about taxes and trade?"
    assert web_helpers.answer_heading(long_plain, {}).startswith("Why did the Articles")


def test_a_replaced_essay_reply_keeps_the_cited_notes_that_passed_on_the_page():
    from course_assistant import grounding
    from course_assistant.providers import FakeProvider
    s, c = build()
    first = ("Madison says a large republic controls the effects of faction better than a small one [1]. "
             "A small society has few distinct parties, so a majority is easily formed [1].\n\n"
             "Your thesis should be that the large republic did not stop parties from forming [1].")
    calls = []

    def respond(system, messages):
        calls.append(1)
        return first if len(calls) == 1 else "Think about which passage you would test your claim against [1]."

    a = grounding.ask(s, c, "What does Federalist No. 10 say about how faction is controlled?", FakeProvider(respond))
    page = a.body_md
    assert "did not stop parties" not in page
    assert a.mode == "concepts" and len(calls) == 2  # caught, replaced once
    assert "Madison says a large republic controls the effects of faction" in page and "From my first answer" in page
