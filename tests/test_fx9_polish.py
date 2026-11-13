"""Cycle 9 fx9-polish: concept map parents and single cites, fallback heading, one disclaimer, kept list markers,
"Step N" signposts, calculation tables, practice stubs and counts, exam topics, "Don't save", the feedback notice."""
import re

import pytest

from course_assistant import disclaimer, essay, grounding, study, web_helpers
from test_render import ready, sources  # noqa: F401
from test_study import GOOD, Prov, future, graded_hit, stand_ins, world  # noqa: F401


def srcs(n=5):
    return [{"chunk_id": i, "material_id": 1, "loc_label": f"p. {i}", "material_title": f"T{i}", "text": "x",
             "href": "/read/1"} for i in range(1, n + 1)]


def parse(raw, n=5, **kw):
    a = grounding.parse_answer(raw, srcs(n))
    return a, web_helpers.answer_view(a.body_md, grounding.source_rows(srcs(n), a.used), blocks=a.blocks, **kw)


# ---- concept map: the parent concept in chains
MAP = {"nodes": [{"id": "p", "label": "Matrix powers", "cites": [1]}, {"id": "f", "label": "Fibonacci numbers", "cites": [2]},
                 {"id": "m", "label": "Markov steady state", "cites": [3]}, {"id": "d", "label": "Defective matrices", "cites": [1]}],
       "edges": [{"a": "p", "b": "f", "label": "applied to", "cites": [2]}, {"a": "f", "b": "m", "label": "applied to", "cites": [3]},
                 {"a": "m", "b": "d", "label": "fails for", "cites": [4]}]}


def tree_html(ready, data):
    gid = ready.store.add_guide(ready.cid, "concept_map", "Map", "", sources=sources(4), data=data)
    return ready.client.get(f"/guides/{gid}").text.split('<ul class="tree">')[1].split('<div class="mapwrap">')[0]


def test_chain_lines_name_the_parent_concept(ready):
    t = tree_html(ready, MAP)
    assert "<strong data-math>Fibonacci numbers</strong> <span class=\"muted\" data-math>applied to</span>: <strong data-math>Markov steady state" in t
    assert "<strong data-math>Markov steady state</strong> <span class=\"muted\" data-math>fails for</span>: <strong data-math>Defective" in t
    assert "<strong data-math>Matrix powers</strong> <span class=\"muted\" data-math>applied to</span>: <strong data-math>Fibonacci" in t


def test_branch_children_name_their_parent_too(ready):
    data = {"nodes": [{"id": "r", "label": "Root idea"}, {"id": "a", "label": "Left idea"}, {"id": "b", "label": "Right idea"}],
            "edges": [{"a": "r", "b": "a", "label": "splits to"}, {"a": "r", "b": "b", "label": "also to"}]}
    t = tree_html(ready, data)
    assert t.count("<strong data-math>Root idea</strong> <span") == 2


def test_the_root_has_no_parent_text(ready):
    t = tree_html(ready, MAP)
    assert t.startswith("<li><strong data-math>Matrix powers</strong>")


# ---- one link per citation
def test_edge_cite_equal_to_the_node_cite_is_shown_once(ready):
    data = {"nodes": [{"id": "a", "label": "Alpha", "cites": [3]}, {"id": "b", "label": "Beta", "cites": [3]}],
            "edges": [{"a": "a", "b": "b", "label": "feeds", "cites": [3]}]}
    t = tree_html(ready, data)
    assert t.count('aria-label="Source 3') == 2 and "link source" not in t and "(link:" not in t  # Alpha and Beta, no repeat


def test_edge_cite_that_adds_a_source_is_shown_once(ready):
    data = {"nodes": [{"id": "a", "label": "Alpha", "cites": [3]}, {"id": "b", "label": "Beta", "cites": [3]}],
            "edges": [{"a": "a", "b": "b", "label": "feeds", "cites": [3, 2]}]}
    t = tree_html(ready, data)
    assert t.count('aria-label="Source 2') == 1 and "(link source:" in t and t.count('aria-label="Source 3') == 2


def test_uncited_edge_still_says_not_from_materials(ready):
    data = {"nodes": [{"id": "a", "label": "Alpha", "cites": [3]}, {"id": "b", "label": "Beta", "cites": [3]}],
            "edges": [{"a": "a", "b": "b", "label": "feeds"}]}
    assert "Not from your materials" in tree_html(ready, data)


# ---- fallback heading, one disclaimer, one notice
FAILED = essay.FEEDBACK_FAILED + " Here are questions from the assignment's own requirements.\n\n- Which sentence is your claim?"


def ask_page(ready):
    return ready.client.get(f"/courses/{ready.cid}/ask").text


def test_fallback_feedback_page_is_headed_as_feedback(ready):
    ready.store.add_chat_message(ready.cid, "user", "Here is my draft intro for Essay 1: Eigenvectors matter because they make things easy.")
    ready.store.add_chat_message(ready.cid, "assistant", FAILED + "\n\n" + disclaimer.FOOTER, [])
    t = ask_page(ready)
    assert "Feedback on your draft" in t and "Answer to: Here is my draft" not in t
    assert t.count(disclaimer.FOOTER) == 1


def test_an_ordinary_answer_keeps_its_question_heading(ready):
    ready.store.add_chat_message(ready.cid, "user", "What is a pivot?")
    ready.store.add_chat_message(ready.cid, "assistant", "A pivot is the leading entry of a row [1].", sources(1))
    assert "Answer to: What is a pivot?" in ask_page(ready)


def test_disclaimer_shows_once_on_a_page_with_many_answers(ready):
    for i in range(3):
        ready.store.add_chat_message(ready.cid, "user", f"Question {i}?")
        ready.store.add_chat_message(ready.cid, "assistant", f"Answer {i} is a fact [1].\n\n{disclaimer.FOOTER}", sources(1))
    assert ask_page(ready).count(disclaimer.FOOTER) == 1


def test_saved_guide_page_shows_the_disclaimer_once(ready):
    gid = ready.store.add_guide(ready.cid, "answer", "G", f"Fact [1].\n\n{disclaimer.FOOTER}", sources=sources(1))
    assert ready.client.get(f"/guides/{gid}").text.count(disclaimer.FOOTER) == 1


def test_feedback_notice_leaves_the_first_block_and_sits_in_the_chrome_once(ready):
    blocks = [{"kind": "text", "md": essay.NOTE_FEEDBACK, "cites": []},
              {"kind": "text", "md": "Where in your draft does it meet the rubric's evidence line?", "cites": []}]
    gid = ready.store.add_guide(ready.cid, "twin", "F", "x", sources=sources(1),
                                data={"blocks": blocks, "feedback": True, "mode": "concepts", "kind": "essay"})
    t = ready.client.get(f"/guides/{gid}").text
    assert t.count(essay.NOTE_FEEDBACK) == 1 and "data-feedback-note" in t
    v = web_helpers.answer_view("", sources(1), blocks=blocks, data={"feedback": True})
    assert essay.NOTE_FEEDBACK not in "".join(str(b["html"]) for b in v["blocks"]) and v["feedback_note"] == essay.NOTE_FEEDBACK


def test_a_non_feedback_answer_has_no_feedback_notice(ready):
    v = web_helpers.answer_view("A fact [1].", sources(1))
    assert v["feedback_note"] == "" and not v["feedback"]


def test_the_notice_wording_matches_essay_py():
    assert web_helpers.NOTE_FEEDBACK == essay.NOTE_FEEDBACK and web_helpers.NOTE_FEEDBACK_PROOF == essay.NOTE_FEEDBACK_PROOF


# ---- a box never strands a list marker
def test_outside_list_item_keeps_its_number():
    a, v = parse("To find the mass:\n\n1. Convert the volume to litres [1].\n3. <outside>The molar mass of KCl is 74.55 g/mol.</outside>\n4. Finish [2].")
    html = "".join(str(b["html"]) for b in v["blocks"])
    assert "<p>3.</p>" not in html and '<ol start="3"><li>The molar mass of KCl' in html


def test_keep_list_markers_leaves_ordinary_blocks_alone():
    blocks = [grounding.Block("text", "Intro line [1].", [1]), grounding.Block("outside", "Aside.", [])]
    assert [b.md for b in grounding.keep_list_markers(blocks)] == ["Intro line [1].", "Aside."]
    blocks = [grounding.Block("text", "A step count 3.", [1]), grounding.Block("outside", "Aside.", [])]
    assert grounding.keep_list_markers(blocks)[0].md == "A step count 3."


# ---- "Step N" headings are signposts
def test_step_heading_inside_a_paragraph_is_not_boxed():
    a, _ = parse("The method uses elimination [1]. Step 2: Back-substitute from the last row. The last equation fixes one unknown [2].")
    assert all(b.kind == "text" for b in a.blocks)


def test_step_label_followed_by_a_claim_is_still_boxed():
    a, _ = parse("The method uses elimination [1]. Step 2: Gauss was born in Brunswick and trained as a surveyor. The last equation fixes one unknown [2].")
    assert any(b.kind == "outside" for b in a.blocks)
    a, _ = parse("The method uses elimination [1]. Step 3. Eleven is not a prime that divides this.")
    assert any(b.kind == "outside" for b in a.blocks)


# ---- calculation tables
def test_uncited_working_table_next_to_a_cited_sentence_is_not_boxed():
    raw = "Carbon makes up 27.29% of the compound and oxygen 72.71% [1].\n\n| Element | Moles | Ratio |\n|---|---|---|\n| C | 2.27 | 1 |\n| O | 4.54 | 2 |"
    _, v = parse(raw)
    assert not v["outside"] and [b["kind"] for b in v["blocks"]] == ["text", "text"]


def test_uncited_table_of_facts_stays_boxed():
    raw = "Carbon makes up 27.29% of the compound [1].\n\n| Metal | Melting point |\n|---|---|\n| Iron | 1538 |"
    _, v = parse(raw)
    assert v["outside"]
    _, v = parse("| Element | Moles |\n|---|---|\n| C | 2.27 |")
    assert v["outside"]  # no cited sentence beside it


# ---- practice: no stubs, honest counts
Q = "1. Find the rank of a matrix with two pivots in three columns.\n\n<details>\n<summary>Show solution</summary>\n\nRank is two.\n\n</details>\n\n"
Q2 = "2. Use the copied graded question text here.\n\n<details>\n<summary>Show solution</summary>\n\nCopied answer.\n\n</details>\n\n"
Q3 = "3. Describe the nullspace of the zero map on two dimensions.\n\n<details>\n<summary>Show solution</summary>\n\nAll of the plane.\n\n</details>"
NOTE = "A part of this guide was left out because it matched your graded Problem 1. Ask for a similar problem with different numbers instead."


def test_practice_drops_the_question_whose_solution_was_cut():
    before = Q + Q2 + Q3
    after = Q + "2. Use the copied graded question text here.\n\n" + Q3  # only the solution was removed
    out, n, note = study.tidy_practice(before, after, 1, NOTE)
    assert "copied graded question" not in out and n == 1
    assert note.startswith("1 practice question was left out because it matched your graded Problem 1")
    assert re.findall(r"(?m)^(\d+)\. ", out) == ["1", "2"] and "Describe the nullspace" in out


def test_practice_counts_a_dropped_question_and_its_solution_as_one():
    before = Q + Q2 + Q3
    after = Q + Q3  # question and solution both gone: scrub_parts counted 2
    out, n, note = study.tidy_practice(before, after, 2, "2 parts of this guide were left out because they matched your graded Problem 1. Ask for similar problems with different numbers instead.")
    assert n == 1 and note.startswith("1 practice question was left out because")
    assert re.findall(r"(?m)^(\d+)\. ", out) == ["1", "2"]


def test_practice_two_dropped_questions_say_two():
    before = Q + Q2 + Q3
    _, n, note = study.tidy_practice(before, Q, 4, "4 parts of this guide were left out because they matched your graded Problem 1. Ask.")
    assert n == 2 and note.startswith("2 practice questions were left out because")


def test_practice_with_nothing_scrubbed_is_untouched():
    out, n, note = study.tidy_practice(Q + Q3, Q + Q3, 0, "")
    assert out == Q + Q3 and n == 0 and note == ""


# ---- exam guide uses the exam's own topics
def test_exam_guide_uses_the_typed_topics_and_does_not_warn_about_outside_lectures(world):
    s, cid, _ = world
    s.add_event(cid, "lecture", "Gases: Chapter 9", future(-3))
    ex = s.add_event(cid, "exam", "Exam 1", future(4), topics="vector spaces, bases")
    p = Prov(GOOD)
    g = s.guide(study.exam_guide(s, cid, p, event_id=ex))
    prompt = p.calls[0][1][0]["content"]
    assert "vector spaces" in prompt and "bases" in prompt and "Topics this exam covers" in prompt
    assert "Gases" not in prompt and "NO saved materials" not in prompt
    assert "Gases" not in g["body_md"] and g["data"]["missing_lectures"] == []
    assert "No material is linked" not in g["body_md"] and "topics you typed" in g["body_md"]


def test_exam_guide_without_typed_topics_still_warns_about_a_missing_lecture(world):
    s, cid, _ = world
    s.add_event(cid, "lecture", "Gases: Chapter 9", future(-3))
    ex = s.add_event(cid, "exam", "Exam 1", future(4))
    p = Prov(GOOD)
    g = s.guide(study.exam_guide(s, cid, p, event_id=ex))
    assert any("Gases" in m for m in g["data"]["missing_lectures"])


# ---- "Don't save this answer"
def test_dont_save_button_deletes_a_plain_answer(ready):
    gid = ready.store.add_guide(ready.cid, "answer", "Q", "A fact [1].", sources=sources(1), data={"mode": "answer"})
    ready.store.add_chat_message(ready.cid, "user", "Q")
    ready.store.add_chat_message(ready.cid, "assistant", "A fact [1].", sources(1), guide_id=gid)
    t = ask_page(ready)
    assert f'action="/guides/{gid}/unsave"' in t and "Don't save this answer" in t
    r = ready.client.post(f"/guides/{gid}/unsave", follow_redirects=False)
    assert r.status_code == 303 and ready.store.guide(gid) is None
    assert "A fact" in ask_page(ready)  # the answer stays on the page


def test_no_dont_save_for_an_answer_tied_to_graded_work(ready):
    gid = ready.store.add_guide(ready.cid, "answer", "Q", "A fact [1].", sources=sources(1),
                                data={"mode": "answer", "graded_item_id": 7})
    ready.store.add_chat_message(ready.cid, "user", "Q")
    ready.store.add_chat_message(ready.cid, "assistant", "A fact [1].", sources(1), guide_id=gid)
    assert "/unsave" not in ask_page(ready)
    ready.client.post(f"/guides/{gid}/unsave", follow_redirects=False)
    assert ready.store.guide(gid) is not None  # the graded context is kept


def test_no_dont_save_for_a_study_guide(ready):
    gid = ready.store.add_guide(ready.cid, "primer", "P", "x", sources=sources(1))
    ready.client.post(f"/guides/{gid}/unsave", follow_redirects=False)
    assert ready.store.guide(gid) is not None
