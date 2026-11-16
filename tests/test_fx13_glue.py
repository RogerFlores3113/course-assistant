"""Cycle 13 glue: an emptied section loses its heading, the held-back note names the item that matched (and fits a maths
concept), the 'stay close' block never repeats the shown answer, and a guide that holds only the held note is never saved.
Every leak test has an honest look-alike. Original fixture text only."""
import pytest

from course_assistant import integrity, study
from course_assistant.grounding import Block
from course_assistant.providers import FakeProvider

from course_assistant import essay

from test_fx5_essay import build, essay_item
from test_fx12_route import eworld, COPY_TEXTS
from test_fx13_essay import TEXT_SRC, TOPIC_Q, FACTS

NAME = "Essay"
CONCEPT_Q = TOPIC_Q  # an Explain topic the essay names: a concept request that still meets the essay's words
LEAK = "Madison's remedy worked on paper but not in practice."
HONEST = FACTS[0]


def open_item():
    s, c = build()
    return s, c, essay_item(s, c)


def scrub(s, c, text, q=CONCEPT_Q):
    with essay.reply_sources(TEXT_SRC):
        return integrity.scrub_parts(s, c, text, q)


# ------------------------------------------------------------------ 1 an emptied section loses its heading
def test_a_section_that_loses_all_its_content_loses_its_heading():
    s, c, it = open_item()
    cleaned, n, note = scrub(s, c, f"## Explain\n\n{HONEST}\n\n## Example\n\n{LEAK}\n", CONCEPT_Q)
    assert n == 1 and "## Example" not in cleaned and "on paper" not in cleaned
    assert "## Explain" in cleaned and HONEST in cleaned


def test_a_section_that_keeps_some_content_keeps_its_heading():
    s, c, it = open_item()
    cleaned, n, note = scrub(s, c, f"## Explain\n\n{HONEST}\n\n## Example\n\n{LEAK}\n\n"
                                                   "Shays' Rebellion frightened men of property [1].\n", CONCEPT_Q)
    assert n == 1 and "## Example" in cleaned and "frightened men" in cleaned and "on paper" not in cleaned


def test_a_nested_heading_chain_that_empties_goes_whole_but_a_sibling_stays():
    s, c, it = open_item()
    text = f"## Explain\n\n{HONEST}\n\n## More\n\n### Example\n\n{LEAK}\n\n## Last\n\nThe Articles left Congress with little power [1].\n"
    cleaned, n, note = scrub(s, c, text)
    assert "## More" not in cleaned and "### Example" not in cleaned
    assert "## Explain" in cleaned and "## Last" in cleaned and "little power" in cleaned


def test_a_heading_that_was_already_empty_is_left_alone():
    s, c, it = open_item()
    cleaned, n, note = scrub(s, c, f"## Explain\n\n## Example\n\n{HONEST}\n", CONCEPT_Q)
    assert n == 0 and "## Explain" in cleaned


# ------------------------------------------------------------------ 2 the note names the item that matched
def test_a_concept_request_that_hit_an_essay_gets_neutral_wording_that_names_it():
    s, c, it = open_item()
    cleaned, n, note = scrub(s, c, f"## Explain\n\n{HONEST}\n\n{LEAK}\n", CONCEPT_Q)
    assert n == 1 and NAME in note
    assert "rubric" not in note and "assignment" not in note and "thesis" not in note.lower()


def test_a_request_about_the_essay_keeps_the_essay_wording_and_names_it():
    s, c, it = open_item()
    cleaned, n, note = scrub(s, c, f"## Draft\n\n{HONEST}\n\n{LEAK}\n", "Write my thesis for the Federalist essay.")
    assert n == 1 and NAME in note and "rubric" in note


def test_a_wholly_held_concept_answer_names_the_item_without_essay_advice():
    s, c, it = open_item()
    cleaned, n, note = scrub(s, c, f"## Example\n\n{LEAK}\n", CONCEPT_Q)
    assert n == 1 and not cleaned.strip() and NAME in note and "held back" in note and "rubric" not in note


def test_a_maths_concept_hit_names_the_problem_item(tmp_path):
    s, c, ids = eworld(tmp_path)
    cleaned, n, note = scrub(s, c, f"## Explain\n\nEigenvalues come from a determinant.\n\n## Example\n\n{COPY_TEXTS[0]}\n",
                                             "eigenvalues")
    assert n == 1 and "Problem 5" in note and "## Example" not in cleaned and "## Explain" in cleaned


def test_the_route_note_never_says_too_close_when_no_item_matched():
    for why in ("reply", "sticky", "soft_essay", "normal_reply"):
        for held in (False, True):
            for mode in ("concepts", "twin"):
                assert "too close" not in integrity.route_note({}, mode, why, held=held), (why, held, mode)


def test_the_route_note_still_names_a_matched_item():
    note = integrity.route_note({"label": "Problem 5", "title": "Set 2: Problem 5"}, "concepts", "reply")
    assert "too close to your graded Problem 5" in note


# ------------------------------------------------------------------ 3 no repeated 'stay close' block
def test_a_kept_sentence_the_shown_answer_already_holds_is_not_shown_again():
    shown = [Block("text", "**Concepts**\n\nA public library lends books to anyone with a card [1]. The Articles left Congress with little power.", [1])]
    assert integrity._not_already_shown(["A public library lends books to anyone with a card [1]."], shown) == []
    assert integrity._not_already_shown(["A public library lends books to anyone with a card."], shown) == []
    assert integrity._not_already_shown(["little power"], shown) == []  # contained in the shown text


def test_a_kept_sentence_the_page_does_not_show_stays():
    shown = [Block("text", "A public library lends books to anyone with a card [1].", [1])]
    kept = ["Paperbacks sold for a few cents [1]."]
    assert integrity._not_already_shown(kept, shown) == kept
    assert integrity._not_already_shown([], shown) == []


# ------------------------------------------------------------------ 4 never save an empty guide
def _guides(s, c):
    return s.guides(c) if hasattr(s, "guides") else []


def test_a_fully_held_explain_saves_nothing_and_says_which_item(tmp_path):
    s, c, ids = eworld(tmp_path)
    before = len(_guides(s, c))
    with pytest.raises(study.HeldBack) as e:
        study.explain(s, c, FakeProvider(COPY_TEXTS[0]), "eigenvalues")
    assert "Problem 5" in str(e.value) and "Nothing was saved" in str(e.value)
    assert len(_guides(s, c)) == before


def test_a_fully_held_practice_set_saves_nothing(tmp_path):
    s, c, ids = eworld(tmp_path)
    before = len(_guides(s, c))
    with pytest.raises(study.HeldBack):
        study.practice(s, c, FakeProvider(COPY_TEXTS[0]), "eigenvalues", count=1)
    assert len(_guides(s, c)) == before


def test_an_explain_with_real_content_still_saves_with_its_note(tmp_path):
    s, c, ids = eworld(tmp_path)
    raw = f"## Main points\n\n- Eigenvalues come from a determinant [1].\n\n## Example\n\n{COPY_TEXTS[0]}\n"
    gid = study.explain(s, c, FakeProvider(raw), "eigenvalues")
    g = s.guide(gid)
    assert "Eigenvalues come from" in g["body_md"] and "## Example" not in g["body_md"]
    assert g["data"]["scrubbed"] == 1 and "Problem 5" in g["body_md"] and not g["data"].get("held_back")
