"""Cycle 5 essay fixes: one-sentence positions, source restatements (essay.supported), feedback, flashcards from a graded
essay, MCP paraphrase and refusal wording, Explain on an essay name, humanities prompt wording. Original fixture text."""
import json
import re
from pathlib import Path

import pytest

from course_assistant import essay, grounding, importers, integrity, mcp_server, study
from course_assistant.providers import FakeProvider
from course_assistant.store import Store

FIX = Path(__file__).parent / "fixtures" / "import"

FED10 = ("Madison argues that a large republic controls the effects of faction better than a small one because it holds "
         "more interests and parties. A small society has few distinct parties, so a majority is easily formed. Among "
         "the numerous classes of citizens, a combination of the majority becomes less likely.")
# a long chunk on the same period that says nothing about the sentence under test
CH7 = ("Delegates to the ratifying conventions argued for weeks about taxes, western land and the cost of the army. "
       "Many farmers in the interior were in debt and feared the courts. Merchants on the coast wanted a government that "
       "could regulate trade and pay its creditors. The convention in the largest state voted narrowly for the new "
       "frame of government after promises of amendments. Speeches praised Madison's notes and his careful reading of "
       "history, while newspapers reprinted the debates for readers in every town. A republic of such size had never "
       "been tried, and opponents said so in print. The faction in the legislature that opposed the plan met in "
       "taverns and wrote letters to neighbors in other states.")

THESIS_CITED = ("Madison's Federalist No. 10 was right about faction in the 1780s but the 1790s showed that a large republic "
                "could not stop parties from forming. [1]")
POSITIONS = [
    "Overall, Madison was half right: he diagnosed the problem of faction but not its cure.",
    "A strong essay would say that Madison's theory fit the 1780s but missed the party system, because Hamilton and "
    "Jefferson organized supporters by 1794.",
    "You could argue that Madison correctly diagnosed faction in the 1780s but misjudged parties in the 1790s.",
    "You could sharpen it, for example by arguing that Madison correctly diagnosed faction in the 1780s but misjudged "
    "parties in the 1790s.",
    "A thesis in this direction: Madison got the 1780s right but the 1790s wrong.",
    "Your thesis: Madison got the 1780s right and the 1790s wrong.",
    "Your thesis will claim that Madison's diagnosis of faction failed by 1794.",
    "Madison got faction right for 1786 (Shays) and wrong for 1794 (parties) [1].",
]
COACHING = [
    "A thesis should take a side on whether Madison's predictions about faction came true.",
    "A strong thesis takes a side on whether Madison was right or wrong about faction in the 1790s.",
    "The rubric gives 30 points for a clear, arguable thesis, so decide what you would defend about Madison.",
    "Ask yourself which passage of Federalist No. 10 you would test your claim against.",
]


def build():
    s = Store(":memory:")
    c = s.add_course("History")
    for n in ("problem-set-2", "problem-set-3", "essay-assignment-federalist"):
        mid = importers.import_file(s, c, FIX / f"{n}.md")
        integrity.mark_material_graded(s, mid)
    fed = s.add_material(c, "reading", "Federalist No. 10")
    s.replace_chunks(fed, [{"text": FED10, "loc_kind": "section", "loc_start": 2, "loc_label": "paragraphs 2 to 4"}])
    ch = s.add_material(c, "reading", "Chapter 7")
    s.replace_chunks(ch, [{"text": CH7, "loc_kind": "section", "loc_start": 1, "loc_label": "pages 1 to 3"}])
    return s, c


def essay_item(s, c):
    return next(i for i in integrity.graded_view(s, c) if i["kind"] == "essay" and "Federalist" in i["title"])


def chunk_of(s, c, title):
    mid = next(x["id"] for x in s.materials(c) if x["title"] == title)
    return s.chunks_of(mid)[0]["id"]


COACH = "Plan around the rubric [1]. Ask yourself which passage of Federalist No. 10 you would test your claim against."


def blind(leak):
    def f(system, messages):
        return COACH if "THE STUDENT'S GRADED TASK" in messages[0]["content"] else leak
    return f


# ------------------------------------------------------------------ 1 one-sentence positions
@pytest.mark.parametrize("text", POSITIONS)
@pytest.mark.parametrize("q", ["", "so what position should I take?"])
def test_one_sentence_positions_are_essay_prose(text, q):
    s, c = build()
    assert essay.leaks(essay_item(s, c), text, q), text


@pytest.mark.parametrize("text", COACHING)
def test_coaching_without_a_position_is_fine(text):
    s, c = build()
    assert essay.leaks(essay_item(s, c), text) is None, text


@pytest.mark.parametrize("text", POSITIONS)
@pytest.mark.parametrize("route", [True, False])
def test_ask_and_ask_normally_never_show_a_one_sentence_position(text, route):
    s, c = build()
    a = grounding.ask(s, c, "so what position should I take in my Federalist essay?", FakeProvider(blind(text)), route=route)
    shown = re.sub(r"\s+", " ", a.body_md)
    core = re.sub(r"\s*\[\d\]", "", text.split(":")[-1]).strip()[:40]
    assert core not in shown, shown


@pytest.mark.parametrize("text", POSITIONS)
def test_study_explain_removes_a_one_sentence_position(text):
    s, c = build()
    p = FakeProvider(lambda sy, me: f"## Notes\n\nFederalist 10 is about faction [1].\n\n{text}")
    gid = study.explain(s, c, p, "Federalist 10 faction and the 1790s")
    assert text.split(":")[-1].strip()[:30] not in s.guide(gid)["body_md"]


@pytest.fixture
def mcp():
    s, c = build()
    mcp_server.set_store(s)
    yield s, c
    mcp_server.set_store(None)


@pytest.mark.parametrize("text", POSITIONS)
def test_mcp_refuses_a_one_sentence_position_in_a_guide_and_a_card(mcp, text):
    s, c = mcp
    assert mcp_server.save_guide(c, "answer", "Position", text, [chunk_of(s, c, "Federalist No. 10")]).startswith("NOT SAVED")
    assert mcp_server.add_flashcards(c, [{"front": "My position", "back": text}]).startswith("NOT ADDED")


# ------------------------------------------------------------------ 2 source restatements need the cited sentence
def test_supported_uses_the_sentence_window_not_the_whole_chunk():
    assert essay.supported("Madison argues that a large republic controls the effects of faction better than a small one [1].", FED10)
    assert essay.supported("A small society has few distinct parties, so a majority is easily formed.", {"text": FED10})
    assert not essay.supported(THESIS_CITED, CH7)  # long on-topic chunk, but no sentence says it
    assert not essay.supported(THESIS_CITED, FED10)
    assert not essay.supported("anything", "")


def test_a_thesis_citing_a_long_unrelated_chunk_is_refused_by_mcp(mcp):
    s, c = mcp
    r = mcp_server.save_guide(c, "answer", "Thesis idea", THESIS_CITED, [chunk_of(s, c, "Chapter 7")])
    assert r.startswith("NOT SAVED")
    assert "This is the problem" not in r and "sentence is the problem" not in r


def test_background_guide_that_describes_sources_saves(mcp):
    s, c = mcp
    body = ("Federalist No. 10 argues that a large republic can control the effects of faction because it holds more "
            "interests and parties [1].\n\nMadison also says a small society has few distinct parties, so a majority is "
            "easily formed [1].\n\nFederalist No. 51 argues for the separation of powers.")
    assert mcp_server.save_guide(c, "primer", "What the sources say", body, [chunk_of(s, c, "Federalist No. 10")]).startswith("Saved guide")


def test_attribution_with_a_cite_that_does_not_back_it_is_not_cleared():
    s, c = build()
    it = essay_item(s, c)
    src = [{"text": CH7}]
    ok = "Madison argues that a large republic controls the effects of faction better than a small one [1]."
    assert essay.find(it, ok, "", True, [{"text": FED10}])[0] == 0
    assert essay.find(it, ok, "", True, src)[0] > 0  # in a guide: the cited chunk never says it


# ------------------------------------------------------------------ 3 feedback
DRAFT = ("In Federalist No. 10, Madison argues that a large republic controls faction better, but the Hamilton-Jefferson "
         "conflict in Chapter 8 shows that parties still formed.")


@pytest.mark.parametrize("reply", [
    "Honest answer against the rubric [1]: the paragraph makes a claim but is thin on evidence.",
    "Verdict: the thesis names a position, but the paragraph needs a citation for its Federalist evidence.",
])
def test_honest_feedback_that_describes_the_draft_is_allowed(reply):
    s, c = build()
    assert not essay.feedback_leaks(DRAFT, essay_item(s, c), reply)


@pytest.mark.parametrize("reply", [
    "You could sharpen it, for example by arguing that Madison correctly diagnosed faction in the 1780s but misjudged "
    "parties in the 1790s.",
    "A thesis in this direction: Madison got the 1780s right but the 1790s wrong.",
])
def test_feedback_that_wraps_a_position_is_rejected(reply):
    s, c = build()
    assert essay.feedback_leaks(DRAFT, essay_item(s, c), reply)


MULTI = ("Here is my draft of the Federalist essay.\n\nMadison argued in Federalist No. 10 that a large republic controls "
         "faction better than a small one. The Hamilton-Jefferson conflict in Chapter 8 showed that parties formed anyway, "
         "so his prediction was partly wrong.\n\nShays's Rebellion in Chapter 7 also showed the Articles were weak, and the "
         "debates over the Constitution matter. Please give feedback.")


@pytest.mark.parametrize("q", [MULTI, MULTI.replace("\n\n", " ")])
def test_pasted_multi_paragraph_draft_without_a_colon_gets_feedback(q):
    s, c = build()
    assert essay.draft_request(q)
    g = integrity.match_graded(s, c, q)
    assert g and g["intent"] == "feedback" and "Federalist" in g["title"]


# ------------------------------------------------------------------ 4 flashcards from a graded essay
def test_flashcards_from_a_graded_essay_are_concept_only():
    s, c = build()
    mid = next(m["id"] for m in s.materials(c) if "Federalist Essay" in m["title"] or "essay" in m["title"].lower())
    cards = [{"front": "What does the rubric reward most?", "back": "A clear, arguable thesis and use of evidence.", "cite": 1},
             {"front": "What is my thesis?", "back": "Madison was half right: the 1780s fit his theory but the 1790s did not.", "cite": 1},
             {"front": "Conclusion?", "back": CONCLUSION_TEXT, "cite": 1},
             {"front": "What must the essay address besides evidence?", "back": "One counterargument, such as the Anti-Federalist view.", "cite": 1}]
    p = FakeProvider(json.dumps(cards))
    res = study.make_flashcards(s, c, p, material_id=mid)
    fronts = [x["front"] for x in s.flashcards(c)]
    assert "What does the rubric reward most?" in fronts and "What must the essay address besides evidence?" in fronts
    assert "What is my thesis?" not in fronts and "Conclusion?" not in fronts
    assert res.guarded == 2
    assert "concepts" in p.calls[0]["messages"][0]["content"]


CONCLUSION_TEXT = "In conclusion, the Founders' hope for a republic free of faction was undone by the party conflict of the 1790s."


def test_flashcards_from_a_graded_problem_set_are_concepts_only():  # cycle 6: no longer refused whole
    s, c = build()
    mid = next(m["id"] for m in s.materials(c) if "Problem Set 2" in m["title"] or "problem-set-2" in m["title"].lower())
    assert len(study.make_flashcards(s, c, FakeProvider("[]"), material_id=mid)) == 0


def test_mcp_flashcards_refuse_thesis_cards_but_take_concepts(mcp):
    s, c = mcp
    assert mcp_server.add_flashcards(c, [{"front": "Thesis", "back": CONCLUSION_TEXT}]).startswith("NOT ADDED")
    ok = mcp_server.add_flashcards(c, [{"front": "What does Madison say a small society lacks?",
                                        "back": "Many distinct parties, so a majority forms easily.",
                                        "chunk_id": chunk_of(s, c, "Federalist No. 10")}])
    assert ok.startswith("Added")


# ------------------------------------------------------------------ 5 MCP paraphrase and refusal wording
def test_paraphrased_short_answer_is_refused_and_names_the_right_item(mcp):
    s, c = mcp
    body = ("Because the solute takes up room, the liters you divide by must be the finished solution. Put the solid in a "
            "volumetric flask, dissolve, and fill to the mark.")
    r = mcp_server.save_guide(c, "answer", "Lab notes", body, [])
    assert r.startswith("NOT SAVED") and "Problem 4" in r and "problem is" not in r.lower().replace("graded item", "")
    plan = "Start with a sentence saying the two formulas differ in what they count, then give your example."
    r = mcp_server.save_guide(c, "answer", "Plan", plan, [])
    assert r.startswith("NOT SAVED") and "Problem 3" in r


def test_a_generic_concept_sentence_is_not_refused_for_an_unrelated_item(mcp):
    s, c = mcp
    body = ("To find an empirical formula, assume 100 g of the compound, convert each mass to moles, divide by the "
            "smallest and round to whole numbers.")
    assert mcp_server.save_guide(c, "explain", "Empirical formula method", body, []).startswith("Saved guide")
    r = mcp_server.add_flashcards(c, [{"front": "Empirical formula steps", "back": "Assume 100 g, convert masses to moles, divide by the smallest."}])
    assert r.startswith("Added")


def test_refusal_never_calls_the_students_sentence_the_problem(mcp):
    s, c = mcp
    r = mcp_server.save_guide(c, "answer", "Thesis idea", THESIS_CITED, [chunk_of(s, c, "Chapter 7")])
    assert "is the problem" not in r
    assert "The sentence that matched" in r


# ------------------------------------------------------------------ 6 Explain on the essay's name
def test_explain_on_the_essay_name_pulls_the_prompt_and_rubric_in_concepts_mode():
    s, c = build()
    seen = []

    def f(system, messages):
        seen.append(messages[0]["content"])
        return COACH
    gid = study.explain(s, c, FakeProvider(f), "Essay 2: The Federalist Essay")
    g = s.guide(gid)
    assert g["data"]["mode"] == "concepts"
    joined = "\n".join(seen)
    assert "Grading rubric" in joined and "Open with a clear, arguable thesis" in joined


# ------------------------------------------------------------------ 7 humanities wording
def test_humanities_prompts_do_not_say_numbers_or_formulas():
    s, c = build()
    s2 = Store(":memory:")
    h = s2.add_course("History")
    m = s2.add_material(h, "reading", "Federalist No. 10")
    s2.replace_chunks(m, [{"text": FED10, "loc_kind": "section", "loc_start": 2, "loc_label": "paragraphs 2 to 4"}])
    s2.add_event(h, "exam", "Final", "2099-12-01")
    p = FakeProvider("## Covered\n\nFaction [1].")
    study.practice(s2, h, p, "faction in Federalist 10", count=2)
    study.exam_guide(s2, h, p)
    study.explain(s2, h, p, "faction")
    for call in p.calls:
        text = call["system"] + call["messages"][0]["content"]
        for bad in ("Invent your own numbers", "invent your own numbers", "Key formulas", "make up new numbers", "STEM material",
                    "worked solution"):
            assert bad not in text, bad
    assert study.has_math(s2, h) is False
    assert study.has_math(s, c) is True  # this course has numeric graded problems


def test_math_courses_keep_the_numbers_wording():
    s, c = build()
    p = FakeProvider("## Covered\n\nMolarity [1].")
    s.add_event(c, "exam", "Final", "2099-12-01")
    study.practice(s, c, p, "molarity of solutions", count=2)
    study.exam_guide(s, c, p)
    joined = "\n".join(x["system"] + x["messages"][0]["content"] for x in p.calls)
    assert "Invent your own numbers" in joined and "Key formulas and definitions" in joined
