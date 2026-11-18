"""fx11-polish: an empty stub removes only itself, Step lines separate, a multi-sentence outside span is labelled per
sentence, typed exam topics are not split on "and", and the practice note counts questions. Original text only."""
import re

from course_assistant import grounding, study, web_md
from test_study import GOOD, Prov, future, world  # noqa: F401

SRC = [{"n": 1, "chunk_id": 1, "title": "Notes", "label": "p. 1", "href": "/read/1#c1"}]


def det(body: str) -> str:
    return f"<details>\n<summary>Show solution</summary>\n\n{body}\n\n</details>"


Q1 = "1. Name one cause of rust."
Q2 = "2. Why does salt speed it up?"
Q3 = "3. What stops iron from rusting in dry air?"
GOOD1 = det("Water and oxygen together cause it [1].")
GOOD3 = det("Without moisture the reaction cannot start [1].")
STUB = "<details>\n<summary>Show solution</summary>\n\n</details>"


def test_stub_after_good_solution_keeps_everything_else():
    text = f"{Q1}\n\n{GOOD1}\n\n{Q2}\n\n{STUB}\n\n{Q3}\n\n{GOOD3}"
    out = study.drop_empty_details(text)
    assert "Water and oxygen together" in out and "Without moisture" in out
    assert Q1 in out and Q2 in out and Q3 in out
    assert out.count("<details>") == 2


def test_stub_after_last_good_solution_does_not_eat_neighbours():
    text = f"{Q1}\n\n{GOOD1}\n\n{Q2}\n\n{GOOD3}\n\n{Q3}\n\n{STUB}"
    out = study.drop_empty_details(text)
    assert "Water and oxygen together" in out and "Without moisture" in out and Q2 in out and Q3 in out
    assert out.count("<details>") == 2


def test_one_line_stub_only_removes_itself():
    text = f"{Q1}\n\n{GOOD1}\n\n{Q2}\n\n<details><summary>Show solution</summary></details>\n\n{Q3}\n\n{GOOD3}"
    out = study.drop_empty_details(text)
    assert out.count("<details>") == 2 and Q2 in out and "Water and oxygen together" in out


def test_truncated_stubs_at_the_end_are_dropped_and_nothing_else():
    for tail in ("<details>", "<details>\n<summary>Show solution</summary>", "<details>\n<summary>Show sol",
                 "<details>\n<summary>Show solution</summary>\n\n"):
        out = study.drop_empty_details(f"{Q1}\n\n{GOOD1}\n\n{Q2}\n\n{tail}")
        assert "Water and oxygen together" in out and Q2 in out, tail
        assert out.count("<details>") == 1 and out.count("Show sol") == 1, tail


def test_filled_block_with_only_a_summary_word_inside_is_kept():
    text = f"{Q1}\n\n{det('The summary of the argument is short.')}"
    assert study.drop_empty_details(text) == text


def test_many_stubs_each_go_alone():
    text = f"{Q1}\n\n{STUB}\n\n{Q2}\n\n{GOOD1}\n\n{Q3}\n\n{STUB}"
    out = study.drop_empty_details(text)
    assert out.count("<details>") == 1 and Q1 in out and Q2 in out and Q3 in out


def test_step_lines_on_single_newlines_are_separate_paragraphs():
    html = str(web_md.render("Step 1. Write the equation.\nStep 2. Move the constant over.\nStep 3. Divide."))
    assert html.count("<p>") == 3 and "Step 2." in html


def test_bold_step_lines_and_colon_form_also_separate():
    html = str(web_md.render("**Step 1.** Start here.\n**Step 2.** Then this.\nStep 3: Finish."))
    assert html.count("<p>") == 3


def test_ordinary_wrapped_prose_still_joins():
    html = str(web_md.render("The first step of the method is to write\nthe equation down and check it.\nSteps vary."))
    assert html.count("<p>") == 1


def test_multi_sentence_outside_span_labels_every_sentence():
    raw = ("Molarity is moles per litre [1]. <outside>A solution holds solute in a solvent. "
           "A saturated one cannot hold more.</outside>")
    a = grounding.parse_answer(raw, SRC)
    inline = " ".join(b.md for b in a.blocks if b.kind == "text")
    outside = " ".join(b.md for b in a.blocks if b.kind == "outside")
    # no marker pair is ever split across blocks
    for b in a.blocks:
        assert b.md.count(grounding.OUTSIDE_OPEN) == b.md.count(grounding.OUTSIDE_CLOSE), b.md
    assert inline.count(grounding.OUTSIDE_OPEN) == 2 and not outside
    assert a.body_md.count("(not from your materials)") + a.body_md.count("Not from your materials") >= 2


def test_outside_span_renders_balanced_html_with_a_label_per_sentence():
    raw = "Molarity is moles per litre [1]. <outside>A solution holds solute. A saturated one holds no more.</outside>"
    a = grounding.parse_answer(raw, SRC)
    html = "".join(str(web_md.render(b.md, SRC)) for b in a.blocks if b.kind == "text")
    assert html.count("<span") == html.count("</span>")
    assert html.count("oi-tag") == 2 and html.count("(not from your materials)") == 2


def test_single_sentence_outside_span_is_unchanged():
    a = grounding.parse_answer("Molarity is moles per litre [1]. <outside>A solution holds solute.</outside>", SRC)
    text = " ".join(b.md for b in a.blocks if b.kind == "text")
    assert text.count(grounding.OUTSIDE_OPEN) == 1 and text.count(grounding.OUTSIDE_CLOSE) == 1


def test_exam_guide_prompt_keeps_a_typed_topic_with_and_whole(world):
    s, cid, _ = world
    s.add_event(cid, "lecture", "Vector spaces", future(-3))
    ex = s.add_event(cid, "exam", "Exam 1", future(4), topics="Bases and dimension, rank")
    p = Prov(GOOD)
    study.exam_guide(s, cid, p, event_id=ex)
    prompt = p.calls[0][1][0]["content"]
    assert "- Bases and dimension" in prompt and "- rank" in prompt
    assert "- Bases\n" not in prompt and "- dimension\n" not in prompt


def test_practice_note_counts_one_unnumbered_question_as_one():
    stem = "**Question 2.** Restate the copied graded question here in words."
    before = f"{Q1}\n\n{GOOD1}\n\n{stem}\n\n{det('Copied answer.')}\n\n{Q3}\n\n{GOOD3}"
    after = f"{Q1}\n\n{GOOD1}\n\n{Q3}\n\n{GOOD3}"
    note = "2 parts of this guide were left out because they matched your graded Problem 2. Ask for similar problems."
    _, n, new = study.tidy_practice(before, after, 2, note)
    assert n == 1 and new.startswith("1 practice question was left out because") and "2 parts" not in new


def test_practice_note_with_two_unnumbered_questions_says_two():
    s2 = "**Question 2.** Restate the copied question."
    s3 = "**Question 3.** Restate the other copied question."
    before = f"{Q1}\n\n{GOOD1}\n\n{s2}\n\n{det('A.')}\n\n{s3}\n\n{det('B.')}"
    note = "4 parts of this guide were left out because they matched your graded Problem 2. Ask."
    _, n, new = study.tidy_practice(before, f"{Q1}\n\n{GOOD1}", 4, note)
    assert n == 2 and new.startswith("2 practice questions were left out")


def test_numbered_questions_still_renumber(monkeypatch):
    before = f"{Q1}\n\n{GOOD1}\n\n{Q2}\n\n{det('x')}\n\n{Q3}\n\n{GOOD3}"
    after = f"{Q1}\n\n{GOOD1}\n\n{Q3}\n\n{GOOD3}"
    out, n, _ = study.tidy_practice(before, after, 2, "2 parts of this guide were left out because a.")
    assert n == 1 and re.findall(r"(?m)^(\d+)\. ", out) == ["1", "2"]
