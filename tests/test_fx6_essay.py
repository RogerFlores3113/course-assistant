"""Cycle 6 essay fixes: sentence-by-sentence judging, wrapped and open-prompt positions, other languages, question-form
theses, false positives, drafts, assignment sources, splitter, wording. Original fixture text."""
import json
import re

import pytest

from course_assistant import essay, grounding, integrity, mcp_server, study
from course_assistant.providers import FakeProvider
from course_assistant.store import Store

from test_fx5_essay import FED10, build, chunk_of, essay_item

INTRO = ("Here is a draft introduction for your essay. In the 1780s, Americans watched Shays's Rebellion expose the weakness "
         "of the Articles of Confederation and wondered how a large republic could tame faction. This essay argues that "
         "Madison's diagnosis captured the conflicts of the 1780s but that the first party system of the 1790s showed his "
         "remedy was incomplete. I will draw on Chapters 7 and 8 and on Federalist Nos. 10 and 51 to make the case.")

LA_BODY = ("Essay 1: The most important idea in linear algebra\n\nWrite an essay arguing which single idea in linear algebra "
           "is most important and why.\n\n## Rubric\n\n- A clear, arguable thesis (30 points)\n- Evidence from at least two "
           "lectures (30 points)\n- One counterargument (20 points)")


def la_course():
    s = Store(":memory:")
    c = s.add_course("Linear Algebra")
    s.add_graded_item(c, "Essay 1: The most important idea in linear algebra", LA_BODY, None, kind="essay", label="Essay 1")
    lec = s.add_material(c, "notes", "Lecture 12 notes")
    s.replace_chunks(lec, [{"text": "An eigenvector of a matrix keeps its direction when the matrix acts on it.",
                            "loc_kind": "section", "loc_start": 1, "loc_label": "Eigenvectors"}])
    return s, c


def la_item(s, c):
    return next(i for i in integrity.graded_view(s, c) if i["kind"] == "essay")


# ------------------------------------------------------------------ 1 a finished intro is judged by sentence
def test_a_finished_intro_with_your_and_could_is_level_two():
    s, c = build()
    assert essay.level(essay_item(s, c), INTRO) == 2


def test_this_essay_argues_is_never_source_attribution():
    for sent in ("This essay argues that Madison's diagnosis of faction failed by the 1790s.",
                 "This paper argues that Madison's diagnosis of faction failed by the 1790s.",
                 "I argue that Madison's diagnosis of faction failed by the 1790s."):
        assert not essay.attributed(sent), sent
        s, c = build()
        assert essay.level(essay_item(s, c), sent) == 2, sent
    assert essay.attributed("Madison argues that a large republic controls the effects of faction.")


def test_a_coaching_word_in_one_sentence_never_exempts_another():
    s, c = build()
    it = essay_item(s, c)
    para = ("You should open with a claim. This essay argues that Madison's diagnosis of faction held in the 1780s but "
            "failed by the 1790s, and the first party system proves it.")
    assert essay.level(it, para) == 2
    assert essay.level(it, "A student should note that this essay argues that Madison's diagnosis failed by the 1790s.") == 0


def test_a_coaching_tail_on_a_finished_sentence_does_not_exempt_it():
    s, c = build()
    it = essay_item(s, c)
    body = ("Hamilton's financial program divided the political elite, and the resulting parties showed that Madison's "
            "large republic could not stop faction, which a student should note.")
    assert essay.level(it, body) > 0
    assert essay.level(it, body.replace(", which a student should note", " (students should note this)")) > 0
    assert essay.trim_tail("Parties formed anyway in the 1790s, which a student should note.") == "Parties formed anyway in the 1790s"


def test_ask_normally_never_shows_a_finished_intro():
    s, c = build()
    blind = lambda sy, me: ("Plan around the rubric. Ask yourself which passage you would test your claim against."
                            if "THE STUDENT'S GRADED TASK" in me[0]["content"] else INTRO)
    for route in (True, False):
        a = grounding.ask(s, c, "Write the introduction paragraph for my Federalist essay", FakeProvider(blind), route=route)
        assert "This essay argues" not in a.body_md


def test_mcp_refuses_a_finished_sentence_with_a_coaching_tail():
    s, c = build()
    mcp_server.set_store(s)
    try:
        body = ("Hamilton's financial program divided the political elite, and the resulting parties showed that Madison's "
                "large republic could not stop faction, which a student should note. [1]")
        assert mcp_server.save_guide(c, "answer", "Study note", body, [chunk_of(s, c, "Federalist No. 10")]).startswith("NOT SAVED")
    finally:
        mcp_server.set_store(None)


# ------------------------------------------------------------------ 2 wrapped positions in draft feedback
DRAFT = ("In Federalist No. 10, Madison argues that a large republic controls faction better, but the Hamilton-Jefferson "
         "conflict in Chapter 8 shows that parties still formed.")
WRAPPED = [
    "Your thesis: Madison was right about the 1780s and wrong about the 1790s.",
    "A strong essay would say that Madison's theory fit the 1780s but missed the party system.",
    "A strong thesis would say that Madison's theory fit the 1780s but missed the party system.",
    "You could argue that Madison correctly diagnosed faction in the 1780s but misjudged parties in the 1790s.",
    "A good position is that Madison diagnosed faction well but his remedy failed to stop parties by 1794.",
    "Your thesis could be built around this idea: faction in the 1780s was majority-driven, in the 1790s party-driven.",
    "One possible thesis: Madison was half right.",
    "If I were writing it, I would say that Federalist 10 explains Shays but Hamilton and Jefferson prove the theory incomplete.",
    "What if you argued that the real test of Madison's theory was the party system of the 1790s, not the mob of the 1780s?",
    "Podrias escribir que Madison acerto en los 1780s pero no en los 1790s.",
    "Madison tenia razon sobre la facción en los años 1780 pero fallo con los partidos de los 1790.",
]


@pytest.mark.parametrize("reply", WRAPPED)
def test_feedback_rejects_every_wrapped_position(reply):
    s, c = build()
    assert essay.feedback_leaks(DRAFT, essay_item(s, c), reply), reply


@pytest.mark.parametrize("reply", WRAPPED)
def test_every_wrapped_position_is_essay_prose_in_ask_and_ask_normally(reply):
    s, c = build()
    assert essay.leaks(essay_item(s, c), reply, "so what should I say in my Federalist essay?"), reply
    blind = lambda sy, me: ("Plan around the rubric. Ask yourself which passage you would test your claim against."
                            if "THE STUDENT'S GRADED TASK" in me[0]["content"] else reply)
    for route in (True, False):
        a = grounding.ask(s, c, "so what should I say in my Federalist essay?", FakeProvider(blind), route=route)
        assert reply[:35] not in re.sub(r"\s+", " ", a.body_md)


@pytest.mark.parametrize("reply", WRAPPED)
def test_mcp_refuses_wrapped_positions_in_guides_and_cards(reply):
    s, c = build()
    mcp_server.set_store(s)
    try:
        assert mcp_server.save_guide(c, "answer", "Idea", reply, [chunk_of(s, c, "Federalist No. 10")]).startswith("NOT SAVED")
        assert mcp_server.add_flashcards(c, [{"front": "My claim", "back": reply}]).startswith("NOT ADDED")
    finally:
        mcp_server.set_store(None)


@pytest.mark.parametrize("reply", [
    "Honest answer against the rubric [1]: the paragraph makes a claim but is thin on evidence.",
    "Where in your draft does it meet the requirement to address one counterargument?",
    "A strong thesis takes a side on whether Madison was right or wrong about faction.",
    "You could ask yourself which passage of Federalist No. 10 supports each paragraph of your draft.",
])
def test_feedback_that_coaches_is_still_fine(reply):
    s, c = build()
    assert not essay.feedback_leaks(DRAFT, essay_item(s, c), reply), reply


# ------------------------------------------------------------------ 3 open prompts: no proper names
OPEN = [
    "A strong essay would say that eigenvalues are the most important idea in linear algebra, because they turn hard problems easy.",
    "You could argue that diagonalization shows this best, since it reduces repeated multiplication to scalar powers.",
    "A strong thesis would be that eigenvectors turn repeated multiplication into scalar powers across many fields.",
    "Your thesis: eigenvalue magnitudes decide whether a dynamical system is stable or blows up over time.",
]


@pytest.mark.parametrize("text", OPEN)
def test_wrapped_position_on_an_open_prompt_is_a_hard_hit(text):
    s, c = la_course()
    lvl, why, _ = essay.find(la_item(s, c), text, "", True)
    assert lvl == 2, text
    assert essay.feedback_leaks("Eigenvalues matter because they simplify matrix powers.", la_item(s, c), text)


def test_open_prompt_coaching_about_the_essay_itself_is_fine():
    s, c = la_course()
    it = la_item(s, c)
    assert essay.level(it, "A strong essay would say how each lecture example supports the claim in the thesis.", "", True) == 0
    assert essay.level(it, "A strong thesis would be that it names one idea and a reason a reader could dispute.", "", True) == 0


@pytest.mark.parametrize("q", [
    "What should my thesis be for the essay on the most important idea?",
    "what should I argue in essay 1",
    "what position should I take in my essay",
    "which idea should I pick for my essay",
    "what should I say in my essay about linear algebra",
])
def test_position_questions_are_writing_requests_and_route_to_concepts(q):
    assert essay.writing_request(q), q
    s, c = la_course()
    hit = integrity.match_graded(s, c, q)
    assert hit and hit["kind"] == "essay", q


def test_a_thesis_flashcard_on_an_open_prompt_is_refused():
    s, c = la_course()
    assert essay.card_hit(s, c, "Which idea is the most important?",
                          "Eigenvalues are the most important idea because they turn hard matrix problems into easy scalar ones.")
    assert essay.card_hit(s, c, "What does the rubric reward?", "A clear, arguable thesis and evidence from two lectures.") is None
    mcp_server.set_store(s)
    try:
        r = mcp_server.add_flashcards(c, [{"front": "Which idea is the most important?",
                                           "back": "Eigenvalues are the most important idea because they turn hard matrix problems into easy scalar ones."}])
        assert r.startswith("NOT ADDED")
    finally:
        mcp_server.set_store(None)
    cards = [{"front": "Which idea is the most important?", "cite": 1,
              "back": "Eigenvalues are the most important idea because they turn hard matrix problems into easy scalar ones."},
             {"front": "What does an eigenvector keep?", "back": "Its direction.", "cite": 1}]
    res = study.make_flashcards(s, c, FakeProvider(json.dumps(cards)), topic="eigenvector direction")
    assert [x["front"] for x in s.flashcards(c)] == ["What does an eigenvector keep?"] and res.guarded == 1


def test_a_failed_essay_feedback_turn_saves_no_guide():
    s, c = la_course()
    it = dict(la_item(s, c), intent="feedback")
    leak = "Your thesis: eigenvalue magnitudes decide whether a dynamical system is stable or blows up over time."
    p = FakeProvider(lambda sy, me: leak)
    before = len(s.guides(c))
    ans = integrity.twin(s, c, "Here is my draft: Eigenvalues matter because they simplify powers. Feedback please.", p, it)
    assert essay.FEEDBACK_FAILED in ans.body_md and "Your thesis:" not in ans.body_md
    assert ans.guide_id is None and len(s.guides(c)) == before


# ------------------------------------------------------------------ 5 false positives
def test_a_thesis_method_question_is_answered_normally():
    s, c = build()
    assert not essay.writing_request("How do I write a good thesis statement in general?")
    assert integrity.match_graded(s, c, "How do I write a good thesis statement in general?") is None
    assert essay.writing_request("Write a good thesis statement for my Federalist essay")
    assert essay.writing_request("How do I write a good thesis for my Federalist essay?")


def test_an_honest_not_covered_sentence_is_not_essay_prose():
    s, c = build()
    reply = ("Last term the essay was 8 pages in MLA; this term it is 5 pages, due Nov 20, worth 25 percent. The materials I "
             "searched do not show this term's citation style in these passages.")
    assert essay.level(essay_item(s, c), reply, "How was the Federalist essay different in Fall 2023 from this term?") == 0
    assert essay.level(essay_item(s, c), "The materials I searched do not show whether Madison's remedy for faction worked "
                       "in the 1790s or the 1780s.") == 0


# ------------------------------------------------------------------ 6 drafts and the student's own quotes
def test_a_short_draft_after_my_draft_with_a_feedback_word_is_a_draft():
    q = ("Here is my draft for Essay 1: Eigenvectors matter because they make repeated matrix multiplication simple. "
         "Fibonacci and Markov are my examples. Feedback please, and what would a good thesis claim?")
    d = essay.draft_request(q)
    assert d and d.startswith("Eigenvectors matter") and d.endswith("my examples.")
    s, c = la_course()
    hit = integrity.match_graded(s, c, q)
    assert hit and hit["intent"] == "feedback"


def test_leaks_ignores_a_quote_of_the_students_own_message():
    s, c = build()
    it = essay_item(s, c)
    mine = "Madison was half right: his theory fit the 1780s but the 1790s showed parties could not be stopped."
    question = f'Is this a good thesis: "{mine}" Why or why not?'
    reply = f'You wrote "{mine}" and that names a position; which passage of Federalist No. 10 would you test it against?'
    assert essay.leaks(it, reply, question) is None
    assert essay.leaks(it, reply, "so what is a good thesis for my essay?")


# ------------------------------------------------------------------ 7 the item's prompt and rubric reach the model
def test_feedback_and_explain_get_the_hand_added_prompt_and_rubric_as_source_1():
    s, c = la_course()
    seen = []

    def f(system, messages):
        seen.append(messages[0]["content"])
        return "Your draft names no counterargument yet [1]. Where does it meet the rubric's evidence requirement [1]?"
    q = ("Here is my draft for Essay 1: Eigenvectors matter because they make repeated matrix multiplication simple. "
         "Fibonacci and Markov are my examples. Feedback please.")
    a = grounding.ask(s, c, q, FakeProvider(f))
    assert a.mode == "concepts" and "[1]" in seen[0] and "A clear, arguable thesis (30 points)" in seen[0]
    assert seen[0].index("A clear, arguable thesis") < seen[0].index("THE STUDENT'S OWN DRAFT")
    seen.clear()
    study.explain(s, c, FakeProvider(lambda sy, me: (seen.append(me[0]["content"]) or
                                                     "Plan around the rubric [1]. Ask yourself which lecture you would cite.")),
                  "Essay 1: The most important idea in linear algebra")
    assert "A clear, arguable thesis (30 points)" in "\n".join(seen)


# ------------------------------------------------------------------ 8 splitter and wording
def test_essay_splitter_keeps_abbreviations_together():
    parts = [x for x, _ in essay._spans("Madison wrote Federalist Nos. 10 and 51 with Gov. Bowdoin and Gen. Lincoln at St. Louis, "
                                           "see pp. 14 to 16. Then parties formed.") if _ == "plain"]
    assert len(parts) == 2 and parts[0].startswith("Madison") and parts[0].endswith("16.")


def test_flashcard_prompt_uses_reading_wording_without_math():
    s2 = Store(":memory:")
    h = s2.add_course("History")
    m = s2.add_material(h, "reading", "Federalist No. 10")
    s2.replace_chunks(m, [{"text": FED10, "loc_kind": "section", "loc_start": 2, "loc_label": "paragraphs 2 to 4"}])
    p = FakeProvider("[]")
    study.make_flashcards(s2, h, p, topic="faction")
    text = p.calls[0]["messages"][0]["content"]
    assert "formulas" not in text and "procedures" not in text and "math as" not in text and "arguments in the readings" in text
    s, c = build()
    p = FakeProvider("[]")
    study.make_flashcards(s, c, p, topic="Federalist faction")
    assert "formulas and procedures" in p.calls[0]["messages"][0]["content"]
