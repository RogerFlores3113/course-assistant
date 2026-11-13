"""Cycle 3 render fixes (fx3-render): banners, meta and inline outside, mid-paragraph labels, value brackets,
past-offering tags, reader math and links, guide tidy-up, twin routing note, reader media."""
from pathlib import Path

import pytest

from course_assistant import grounding, paths, web_helpers, web_md
from test_render import page, ready, sources  # noqa: F401
from test_fx2_render import srcs


# 1 banner over fully cited answers
def test_f1_cited_answer_with_dont_list_anything_else_has_no_banner(ready):
    a = grounding.parse_answer("Bring goggles and a lab coat [1].\n\nYour materials don't list anything else for Lab 3.", srcs(1))
    assert not a.not_covered and not a.outside
    r = page(ready, grounding.block_dicts(a.blocks), data={"not_covered": True})  # a flag saved by cycle 2
    assert "cover this" not in r.text


def test_f1_banner_still_shows_when_nothing_cited_or_mostly_outside():
    assert grounding.parse_answer("The materials do not cover week 8.", srcs(1)).not_covered
    assert grounding.parse_answer("A general fact about the world, long enough to count as a claim.", []).not_covered
    assert grounding.parse_answer("Mostly general words here and there for a long while now.\n\nTiny [1].", srcs(1)).not_covered


def test_f1_empty_reply_has_no_general_knowledge_banner():
    a = grounding.parse_answer("", srcs(1))
    assert not a.not_covered


# 2 meta sentences, question stems, inline outside
@pytest.mark.parametrize("line", ["Try asking: How are trace and determinant related?",
                                  "Jordan form is not covered in your materials, so ask your instructor.",
                                  "Which of these would you choose, and why?",
                                  "Here are 3 questions to keep in mind"])
def test_f2_meta_and_questions_not_boxed(line):
    blocks, _ = grounding.parse_blocks(f"Eigenvalues are roots [1].\n\n{line}", 2)
    assert not any(b.kind == "outside" for b in blocks)


def test_f2_practice_question_stems_not_boxed_but_solution_is():
    raw = ("1. Calculate the molarity of 2 mol of KCl in 4 L of solution.\n<details>\n<summary>Show solution</summary>\n\n"
           "Divide moles by litres to get 0.5 M.\n\n</details>")
    blocks, _ = grounding.parse_blocks(raw, 2)
    assert blocks[0].kind == "text" and blocks[0].md.startswith("1. Calculate")
    assert blocks[1].kind == "outside" and "<summary>" in blocks[1].md


def test_f2_numbered_questions_without_cites_not_boxed():
    blocks, _ = grounding.parse_blocks("1. What is a pivot?\n2. Find the rank of the matrix below.\n3. Why can a system have no solution?", 2)
    assert [b.kind for b in blocks] == ["text"]


def test_f2_inline_outside_stays_in_its_sentence(ready):
    raw = "Molarity is moles of solute <outside>in 1 liter of solution</outside>, as defined in the notes [1]."
    blocks, used = grounding.parse_blocks(raw, 2)
    assert len(blocks) == 1 and blocks[0].kind == "text" and used == [1]
    assert grounding.OUTSIDE_OPEN in blocks[0].md
    r = page(ready, grounding.block_dicts(blocks), n=2)
    html = r.text.split('<div class="answer">')[1]
    assert '<span class="outside-inline">in 1 liter of solution</span>' in html and 'class="oi-tag"' in html
    assert 'class="outside"' not in html and "{{" not in r.text


def test_f2_block_outside_on_its_own_line_is_still_a_box():
    blocks, _ = grounding.parse_blocks("Claim [1].\n\n<outside>\nEuler used it in 1744.\n</outside>", 2)
    assert blocks[-1].kind == "outside"


def test_f2_inline_outside_body_md_reads_as_words():
    a = grounding.parse_answer("It is x <outside>by definition</outside> true [1].", srcs(1))
    assert "by definition (not from your materials)" in a.body_md and "{{" not in a.body_md


# 3 uncited sentence in the middle
def test_f3_uncited_middle_sentence_is_labelled():
    raw = "The Court claimed review in Marbury [1]. Historians still debate how far judicial review was intended. Hamilton argued for it in Federalist 78 [2]."
    blocks, _ = grounding.parse_blocks(raw, 2)
    kinds = [b.kind for b in blocks]
    assert kinds == ["text", "outside", "text"] and "Historians still debate" in blocks[1].md


# 4 value brackets
def test_f4_descending_list_is_values():
    blocks, used = grounding.parse_blocks("The sorted list is reversed to [3, 2, 1] in the example, as in the slides [1].", 3)
    assert used == [1] and "[3, 2, 1]" in blocks[0].md


def test_f4_range_in_a_vector_sentence_is_values():
    blocks, used = grounding.parse_blocks("Take rows [1-3] of the matrix and stack them as a vector [2].", 3)
    assert used == [2] and "[1-3]" in blocks[0].md
    blocks, used = grounding.parse_blocks("The vector entries [1-3] are listed first [1].", 3)
    assert used == [1]


def test_f4_numbers_beyond_the_source_count_are_values():
    blocks, used = grounding.parse_blocks("The scores were [7, 8, 9] in the experiment [1].", 3)
    assert used == [1] and "[7, 8, 9]" in blocks[0].md and grounding.REMOVED_MARK not in blocks[0].md


def test_f4_real_cites_still_work():
    blocks, used = grounding.parse_blocks("Eigenvalues are roots [1-3]. Also see [1, 3].", 3)
    assert used == [1, 2, 3]
    assert grounding.cited_numbers("As shown in [1, 3] and [2][3]. Range [2-3].") == [1, 3, 2, 3, 2, 3]


# 5 past offering tag
def test_f5_past_offering_tag_in_sources_and_chip_label(ready):
    mid = ready.store.add_material(ready.cid, "past_offering", "Fall 2018 18.06 Lecture Summaries")
    cur = ready.store.add_material(ready.cid, "notes", "My notes")
    src = [{"n": 1, "chunk_id": 1, "material_id": mid, "href": "/read/1?at=1#c1", "title": "Fall 2018 18.06 Lecture Summaries", "label": "Lecture 32"},
           {"n": 2, "chunk_id": 2, "material_id": cur, "href": "/read/2?at=2#c2", "title": "My notes", "label": "p. 2"}]
    gid = ready.store.add_guide(ready.cid, "answer", "G", "b", sources=src,
                                data={"blocks": [{"kind": "text", "md": "Old exam [1]. New notes [2].", "cites": [1, 2]}]})
    html = ready.client.get(f"/guides/{gid}").text
    assert html.count("Past offering: Fall 2018") >= 2  # chip label in the text, visible tag
    assert 'class="past-tag">(Past offering: Fall 2018)' in html
    assert "My notes, p. 2, Past offering" not in html


def test_f5_source_rows_carry_the_label():
    s = srcs(1)
    s[0]["material_kind"] = "past_offering"
    s[0]["material_title"] = "Fall 2018 Notes"
    assert grounding.source_rows(s)[0]["past"] == "Past offering: Fall 2018"
    assert "past" not in grounding.source_rows(srcs(1))[0]


# 6 reader
def test_f6_asterisk_variables_and_flat_equations():
    html = str(web_md.render_reader("Use *C*1*V*1 = *C*2*V*2 to dilute. Then V2= (C1V1)/(C2) gives the volume of *V*1 later."))
    assert "*" not in html
    assert r"$C_{1}V_{1} = C_{2}V_{2}$" in html and r"$V_{2}= \frac{C_{1}V_{1}}{C_{2}}$" in html
    assert str(web_md.render("so *C*1*V*1 is it")).count("<em>") == 2


def test_f6_plain_assignments_stay_text():
    html = str(web_md.render_reader("Set x = 5 and y = 7, then PV = nRT."))
    assert "math" not in html


def test_f6_reader_math_and_pdf_page_link(ready):
    cid = ready.cid
    base = paths.files_dir()
    (base / str(cid)).mkdir(parents=True, exist_ok=True)
    (base / str(cid) / "book.pdf").write_bytes(b"%PDF-1.4")
    mid = ready.store.add_material(cid, "textbook", "Chem", filename=f"{cid}/book.pdf", original_name="book.pdf")
    ready.store.replace_chunks(mid, [
        {"text": "Dilution: *C*1*V*1 = *C*2*V*2", "loc_kind": "page", "loc_start": "3", "loc_label": "p. 3"},
        {"text": "More.", "loc_kind": "page", "loc_start": "4", "loc_label": "p. 4"}])
    ids = [c["id"] for c in ready.store.chunks_of(mid)]
    r = ready.client.get(f"/read/{mid}?at={ids[0]}")
    assert f'href="/files/{mid}#page=3"' in r.text and 'class="math"' in r.text and "*C*" not in r.text
    r2 = ready.client.get(f"/read/{mid}")
    assert f'href="/files/{mid}"' in r2.text and "#page" not in r2.text
    r3 = ready.client.get(f"/read/{mid}?at={ids[1]}")
    assert f'href="/files/{mid}#page=4"' in r3.text


def test_f6_missing_material_says_passage_no_longer_there(ready):
    mid = ready.store.add_material(ready.store.add_course("Gone"), "notes", "Gone soon")
    ready.store.delete_material(mid)
    r = ready.client.get(f"/read/{mid}")
    assert r.status_code == 410 and "This passage is no longer in your materials" in r.text


# 7 guides
def test_f7_flat_concept_lists_hidden_when_tree_shows(ready):
    data = {"nodes": [{"id": "a", "label": "Eigenvalue", "cites": [1]}, {"id": "b", "label": "Trace", "cites": [2]}],
            "edges": [{"a": "a", "b": "b", "label": "sums to", "cites": [2]}],
            "blocks": [{"kind": "text", "md": "Summary of the map [1].", "cites": [1]},
                       {"kind": "text", "md": "## Concepts and links", "cites": []},
                       {"kind": "text", "md": "- **Eigenvalue** [1]\n- **Trace** [2]", "cites": [1, 2]}]}
    gid = ready.store.add_guide(ready.cid, "concept_map", "Map", "b", sources=sources(2), data=data)
    html = ready.client.get(f"/guides/{gid}").text
    assert "Concepts and links" not in html and "Summary of the map" in html
    assert 'class="tree"' in html and html.count("Eigenvalue") >= 1
    assert '[2]</span>' in html.split('<div class="sources">')[1]  # the map's own cites still list their sources


def test_f7_missing_lectures_intro_is_a_list():
    md = "No saved materials for: Lecture 3 (2026-09-01); Lecture 4 (2026-09-03); Lecture 5 (2026-09-08). Add them to cover those topics."
    v = web_helpers.answer_view("", [], blocks=[{"kind": "text", "md": md}])
    html = str(v["blocks"][0]["html"])
    assert html.count("<li>") == 3 and "Lecture 4 (2026-09-03)" in html and ";" not in html


def test_f7_check_line_right_under_the_heading(ready):
    md = f'One {grounding.REMOVED_MARK}. He wrote {grounding.QUOTE_OPEN}"a b c d e f g h"{grounding.QUOTE_CLOSE} [1].'
    r = page(ready, [{"kind": "text", "md": md, "cites": [1]}])
    answer = r.text.split('<div class="answer">')[1]
    assert answer.lstrip().startswith('<p class="check-line"') and "Check: 1 citation removed, 1 quote not found" in answer
    clean = page(ready, [{"kind": "text", "md": "Fine [1].", "cites": [1]}])
    assert "check-line" not in clean.text


# 8 twin pages
def test_f8_twin_shows_routing_note_and_ask_normally_not_banner(ready):
    gid = ready.store.add_guide(ready.cid, "twin", "Similar problem to Problem 1: elimination", "b", sources=sources(1),
                                question="How do I solve Problem 1?",
                                data={"mode": "twin", "not_covered": True, "routing_note": "This looks like your graded Problem 1, so here is a similar problem.",
                                      "blocks": [{"kind": "outside", "md": "A similar system with new numbers.", "cites": []}]})
    html = ready.client.get(f"/guides/{gid}").text
    assert "This looks like your graded Problem 1, so here is a similar problem." in html
    assert "cover this" not in html and "general knowledge" not in html
    assert f'action="/courses/{ready.cid}/ask"' in html and 'name="normal" value="1"' in html
    assert 'name="question" value="How do I solve Problem 1?"' in html and "Ask normally" in html


def test_f8_twin_without_routing_note_falls_back(ready):
    gid = ready.store.add_guide(ready.cid, "twin", "T", "b", question="q", data={"mode": "twin"})
    html = ready.client.get(f"/guides/{gid}").text
    assert "This looks like your graded work" in html and "cover this" not in html


def test_f8_plain_answer_has_no_ask_normally(ready):
    r = page(ready, [{"kind": "text", "md": "Fine [1].", "cites": [1]}])
    assert "Ask normally" not in r.text


# 9 reader media
def test_f9_go_to_label_and_pause_announces(ready):
    cid = ready.cid
    base = paths.files_dir()
    (base / str(cid)).mkdir(parents=True, exist_ok=True)
    (base / str(cid) / "lec.mp3").write_bytes(b"ID3")
    mid = ready.store.add_material(cid, "recording", "Rec", filename=f"{cid}/lec.mp3", original_name="lec.mp3")
    ready.store.replace_chunks(mid, [{"text": "bit", "loc_kind": "time", "loc_start": "49", "loc_label": "0:49"}])
    html = ready.client.get(f"/read/{mid}").text
    assert ">Go to 0:49</button>" in html and "Play from" not in html
    js = (Path(__file__).parent.parent / "src/course_assistant/static/content.js").read_text()
    assert "Paused at " in js and "sr-status" in js
