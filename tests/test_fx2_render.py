"""Cycle 2 render fixes: hidden solutions, meta sentences, lists, value brackets, quotes, concept map, headings."""
import pytest

from course_assistant import grounding, web_helpers, web_md
from test_render import MAP, page, ready, sources  # noqa: F401


def srcs(n, text="text"):
    return [{"chunk_id": i, "material_id": 1, "material_title": "Lec", "material_kind": "slides", "loc_kind": "slide",
             "loc_start": i, "loc_label": f"slide {i}", "heading": "", "text": f"{text} {i}", "href": f"/read/1?at={i}#c{i}",
             "score": 1.0} for i in range(1, n + 1)]


# 1 hidden solutions
PRACTICE = ("**Question 1.** Compute the molarity of 2 mol in 4 L [1].\n\n<details>\n<summary>Show solution</summary>\n\n"
            "Step 1: divide moles by litres.\n\n$M = 2/4 = 0.5$ [1]\n\n</details>")


def test_problem1_details_stay_one_block():
    blocks, used = grounding.parse_blocks(PRACTICE, 2)
    assert len(blocks) == 2 and blocks[1].md.startswith("<details>") and blocks[1].md.endswith("</details>")
    assert blocks[1].kind == "text" and blocks[1].cites == [1]
    assert all(b.kind == "text" for b in blocks)


def test_problem1_uncited_details_is_one_outside_block_not_three():
    raw = "<outside>\n<details>\n<summary>Show solution</summary>\n\nDivide 2 mol by 4 L to get 0.5 M.\n\n</details>\n</outside>"
    blocks, _ = grounding.parse_blocks(raw, 2)
    assert len(blocks) == 1 and "<summary>" in blocks[0].md and "0.5 M" in blocks[0].md


def test_problem1_page_shows_solution_inside_details(ready):
    blocks, used = grounding.parse_blocks(PRACTICE, 2)
    r = page(ready, grounding.block_dicts(blocks), n=2)
    html = r.text.split('<div class="answer">')[1]
    assert "<details><summary>Show solution</summary>" in html and html.index("Step 1") > html.index("<summary>")
    assert html.index("Step 1") < html.index("</details>")


# 2 meta sentences
@pytest.mark.parametrize("line", ["Your materials do not cover this topic.",
                                  "Since your materials do not teach this, it is worth asking your instructor.",
                                  "I can't solve Problem 2 for you, because it looks like graded work."])
def test_problem2_meta_sentences_not_boxed(line):
    blocks, _ = grounding.parse_blocks(f"Eigenvalues are roots [1].\n\n{line}", 2)
    assert not any(b.kind == "outside" for b in blocks)


def test_problem2_real_outside_still_boxed():
    blocks, _ = grounding.parse_blocks("Eigenvalues are roots [1].\n\nEuler introduced the term in 1744 and it spread.", 2)
    assert blocks[-1].kind == "outside"


def test_problem2_banner_wording(ready):
    r = page(ready, [{"kind": "text", "md": "x", "cites": []}], data={"not_covered": True})  # nothing cited
    assert "Your saved materials don&#39;t cover this." in r.text and "general knowledge" not in r.text
    r = page(ready, [{"kind": "text", "md": "x [1]", "cites": [1]}, {"kind": "outside", "md": "Other.", "cites": []}],
             data={"not_covered": True})
    assert "Parts below come from general knowledge." in r.text


# 3 lists and sentences
def test_problem3_uncited_numbered_list_is_one_outside_block():
    raw = "Intro [1].\n\n1. Compute the eigenvalues.\n\n   - Use the trace as a check.\n\n2. Then check the determinant."
    blocks, _ = grounding.parse_blocks(raw, 2)
    outs = [b for b in blocks if b.kind == "outside"]
    assert len(outs) == 1 and "1. Compute" in outs[0].md and "2. Then check" in outs[0].md


def test_problem3_uncited_sentence_in_cited_paragraph_labelled():
    raw = "The formula is $A = X D X^{-1}$ [1]. This sentence has no citation at all and says something new."
    blocks, used = grounding.parse_blocks(raw, 2)
    assert [b.kind for b in blocks] == ["text", "outside"] and blocks[0].cites == [1] and used == [1]


def test_problem3_end_cite_covers_paragraph():
    blocks, _ = grounding.parse_blocks("Eigenvectors keep their direction. Only the length changes. See the example [1].", 2)
    assert len(blocks) == 1 and blocks[0].kind == "text"


def test_problem3_abbreviation_not_sentence_end():
    blocks, _ = grounding.parse_blocks("Use a diagonal matrix, e.g. the identity, here [1].", 2)
    assert len(blocks) == 1


# 4 value brackets
@pytest.mark.parametrize("text", ["x = [1,2] here", "the vector [1, 2, 3] is", "Intervals like [1-3] are closed",
                                  "rows [1,2], [2,3] of A", "A[1,2] is an entry"])
def test_problem4_value_brackets_are_not_cites(text):
    assert grounding.cited_numbers(text) == []
    assert grounding.clean_citations(text, 3)[0] == text


def test_problem4_real_cites_still_work():
    assert grounding.cited_numbers("As shown in [1, 3] and [2][3]. Range [2-3].") == [1, 3, 2, 3, 2, 3]


# 5 quotes
def test_problem5_four_word_quote_checked():
    sources = [{"text": "The judiciary will always be the least dangerous branch."}]
    b = [grounding.Block("text", 'He wrote "the judiciary will always be" [1] and "a made up four words" [1].', [1])]
    grounding.verify_quotes(b, sources)
    assert b[0].md.count(grounding.QUOTE_OPEN) == 1 and "made up four words" in b[0].md.split(grounding.QUOTE_OPEN)[1]


def test_problem5_no_false_positives():
    sources = [{"text": "The judiciary, from the nature of its functions, will always be the least dangerous to the political rights."}]
    for q in ['“the JUDICIARY ... will always be the least dangerous”', '"the judiciary [...] will always be"',
              '"The judiciary [of the Union] will always"', "\"from the nature of its functions\""]:
        b = [grounding.Block("text", f"It says {q} [1].", [1])]
        grounding.verify_quotes(b, sources)
        assert grounding.QUOTE_OPEN not in b[0].md, q


def test_problem5_three_word_quote_ignored():
    b = [grounding.Block("text", 'The "so called thing" is odd [1].', [1])]
    grounding.verify_quotes(b, [{"text": "nothing here"}])
    assert grounding.QUOTE_OPEN not in b[0].md


# 6 concept map
def test_problem6_removed_marker_and_math_and_svg(ready):
    data = {"nodes": [{"id": "a", "label": "Diagonalization $A = X\\Lambda X^{-1}$", "cites": [1], "cites_removed": True},
                      {"id": "b", "label": "Eigen $\\lambda$", "cites": []}],
            "edges": [{"a": "a", "b": "b", "label": "has $\\lambda$", "cites": [2], "cites_removed": True}]}
    gid = ready.store.add_guide(ready.cid, "concept_map", "Map", "", sources=sources(3), data=data)
    t = ready.client.get(f"/guides/{gid}").text
    tree = t.split('<ul class="tree">')[1].split('<div class="mapwrap">')[0]
    assert tree.count("(citation removed)") == 2 and "<strong data-math>" in tree
    svg = t.split("<svg")[1].split("</svg>")[0]
    assert "$" not in svg and "A = X Lambda X^-1" in svg.replace("\n", " ") or "Lambda" in svg
    assert "lambda" in svg


def test_problem6_plain_math():
    assert web_helpers.plain_math("$A = X\\Lambda X^{-1}$") == "A = X Lambda X^-1"
    assert web_helpers.plain_math("$\\frac{a}{b}$") == "(a)/(b)"


# 7 headings
@pytest.mark.parametrize("md", ["# Top\n\n## Sub", "## Top\n\n### Sub"])
def test_problem7_shallowest_heading_maps_to_base(md):
    html = str(web_md.render(md, base_level=2))
    assert "<h2>Top</h2>" in html and "<h3>Sub</h3>" in html
    assert "<h4>Top</h4>" in str(web_md.render(md, base_level=4))


def test_problem7_answer_view_uses_one_shallowest_across_blocks():
    v = web_helpers.answer_view("", sources(2), blocks=[{"kind": "text", "md": "## Big [1]", "cites": [1]},
                                                         {"kind": "text", "md": "### Small [1]", "cites": [1]}], base_level=4)
    html = "".join(str(b["html"]) for b in v["blocks"])
    assert "<h4>Big" in html and "<h5>Small" in html


def test_problem7_guide_h1_then_h2(ready):
    t = page(ready, [{"kind": "text", "md": "## Big ideas [1]", "cites": [1]}]).text
    assert "<h2>Big ideas" in t and "<h3>Big" not in t


# 8 banners and outside group
def test_problem8_outside_group_semantics(ready):
    t = page(ready, [{"kind": "outside", "md": "One.", "cites": []}, {"kind": "outside", "md": "Two.", "cites": []}]).text
    assert t.count('role="group"') == 1 and 'aria-labelledby="out-' in t and 'class="outside-label" id="out-' in t
    assert "One." in t and "Two." in t


def test_problem8_banners_only_from_mode(ready):
    b = [{"kind": "text", "md": "x [1]", "cites": [1]}]
    t = page(ready, b, data={"mode": "concepts"}, kind="twin").text
    assert "Graded work: help with concepts only" in t and "Worked twin problem.</strong>" not in t


# 9 guide subtitle and sources list
def test_problem9_subtitle_hidden_or_labelled(ready):
    def guide(kind):
        gid = ready.store.add_guide(ready.cid, kind, "G", "body [1]", sources=sources(1), question="2026-11-10")
        return ready.client.get(f"/guides/{gid}").text
    assert "Question:" not in guide("explain") and "Lecture:</strong> 2026-11-10" in guide("missed")
    assert "Question:</strong>" in guide("answer")


def test_problem9_sources_merged_numbered_ordered():
    src = [{"n": 1, "title": "Diagonalization", "label": "Diagonalization", "href": "/read/1?at=1"},
           {"n": 2, "title": "Notes", "label": "p. 2", "href": "/read/1?at=2"},
           {"n": 3, "title": "Diagonalization", "label": "Diagonalization", "href": "/read/1?at=3"}]
    v = web_helpers.answer_view("", src, blocks=[{"kind": "text", "md": "a [3] b [1] c [2]", "cites": [1, 2, 3]}])
    rows = v["sources"]
    assert [[x["n"] for x in r["ns"]] for r in rows] == [[1, 3], [2]] and rows[0]["label"] == ""


def test_problem9_sources_rows_show_numbers(ready):
    t = page(ready, [{"kind": "text", "md": "a [2] b [1]", "cites": [2, 1]}]).text
    main = t.split('<div class="sources">')[1]
    assert main.index(">[1]</a>") < main.index(">[2]</a>")


# 10 / 11
def test_problem10_dark_link_contrast_rule():
    css = open(web_helpers.__file__.replace("web_helpers.py", "static/content.css")).read()
    assert ".chunk.hit a" in css and "#c8d6ff" in css
    assert web_helpers.contrast("#c8d6ff", "#5a4b00") >= 4.5


def test_problem11_reader_emphasis_rendered():
    html = str(web_md.render("Use *V*1 and *C*2 with **bold** and \\*literal\\*."))
    assert "<em>V</em>1" in html and "<em>C</em>2" in html and "<strong>bold</strong>" in html
    assert "*literal*" in html and "<em>literal" not in html


def test_chip_labels_and_removed_marker_plain():
    html = str(web_md.render(f"A [1] and B{grounding.REMOVED_MARK}.", sources(2)))
    assert 'aria-label="Source 1: Notes, p. 1"' in html and 'role="img"' not in html and "(citation removed)" in html


def test_ask_prompt_tags_past_offering_sources():
    from course_assistant import grounding
    src = [{"material_title": "18.06 Fall 2018 exams", "material_kind": "past_offering", "loc_label": "section 1",
            "text": "Exam 3 is on Dec 5."},
           {"material_title": "Syllabus", "material_kind": "syllabus", "loc_label": "section 2", "text": "Exam 3 Dec 8."}]
    out = grounding.format_sources(src)
    assert "[1] 18.06 Fall 2018 exams, section 1 (Past offering: Fall 2018)" in out
    assert "[2] Syllabus, section 2\n" in out
    assert "Past offering" in grounding.system_prompt()
