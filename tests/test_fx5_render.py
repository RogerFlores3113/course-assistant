"""Cycle 5 fx5-render: labels only on factual claims, gap sentences inside cited text, splitter, check line, reader math,
past-offering rows, flat concept chains."""
from course_assistant import grounding, web_helpers, web_md
from test_render import material, page, ready, sources, view  # noqa: F401


def parse_view(raw, n=3, **kw):
    a = grounding.parse_answer(raw, sources_for(n))
    v = web_helpers.answer_view(a.body_md, grounding.source_rows(sources_for(n), a.used), blocks=a.blocks, **kw)
    return a, v


def sources_for(n):
    return [{"chunk_id": i, "material_id": 1, "loc_label": f"p. {i}", "material_title": f"T{i}", "text": "x",
             "href": "/read/1"} for i in range(1, n + 1)]


def kinds(v):
    return [b["kind"] for b in v["blocks"]]


# ---- 1 a "materials do not cover X" sentence inside a cited paragraph
def test_inline_outside_gap_sentence_is_plain_and_partly_covered():
    raw = ("Problem 3 is worth 10 points [1]. <outside>The sources I was given do not say whether late work gets "
           "partial credit.</outside>")
    a, v = parse_view(raw)
    assert "{{outside}}" not in a.body_md and not any(b.kind == "outside" for b in a.blocks)
    assert v["partly_covered"] and kinds(v) == ["text"]
    assert "oi-tag" not in str(v["blocks"][0]["html"])


def test_saved_text_with_old_inline_mark_is_unmarked_at_view():
    md = f"Worth 10 points [1]. {grounding.OUTSIDE_OPEN}The sources do not say whether late work counts.{grounding.OUTSIDE_CLOSE}"
    v = web_helpers.answer_view("", sources_for(2), blocks=[{"kind": "text", "md": md, "cites": [1]}])
    assert "oi-tag" not in str(v["blocks"][0]["html"]) and v["partly_covered"]


def test_real_inline_outside_fact_keeps_its_tag():
    a, v = parse_view("Eigenvalues come from det(A - lI) = 0 [1], and <outside>PCA uses them too</outside> in practice.")
    assert "oi-tag" in str(v["blocks"][0]["html"]) and not v["partly_covered"]


# ---- 2 not a factual claim, so no label
def test_non_claims_are_not_boxed():
    cases = [
        "1. Factor A = [[1,2],[3,7]] as LU.\n\n2. Factor B = [[2,1],[4,5]] as LU.\n\n3. Using L from question 2, solve Ax = b.",
        "2026-11-17: practice set.",
        "Here are the steps for A = [[1,2],[3,7]].",
        "Lectures 6, 7 and 8 have no saved materials, so this guide cannot cover them.",
        "Get notes for Lectures 6 to 8 uploaded.",
        "Plan your opening so it states a position that others could dispute.",
        "Textbook evidence (20): Cite two passages. What does Chapter 7 say it showed?",
        "This is last year's exam, not this term's. Your Fall 2026 Exam 3 date is in the syllabus, check that.",
        "Your draft makes a claim but is thin on evidence.",
        "**Worked example: NaCl in water, 0.350 M x 0.2500 L**",
    ]
    for c in cases:
        a, v = parse_view(c + "\n\nMolarity is moles per litre [1].")
        assert "outside" not in kinds(v), c
        assert not any(b.kind == "outside" for b in a.blocks), c


def test_comparison_of_two_cited_facts_is_not_boxed():
    a, v = parse_view("Fall 2018 had four exams [1]. This term has three exams [2]. That is one fewer exam than Fall 2018.")
    assert kinds(v) == ["text"] and not any(b.kind == "outside" for b in a.blocks)


def test_uncited_arithmetic_in_worked_example_is_not_boxed_but_a_fact_is():
    raw = "## Worked example\n\nStep 1: moles of NaCl.\n\nThe same 0.04625 mol is now in 0.0470 L.\n\nMolarity is moles per litre [1]."
    a, v = parse_view(raw)
    assert "outside" not in kinds(v)
    a2, v2 = parse_view("The same 0.04625 mol is now in 0.0470 L.\n\nMolarity is moles per litre [1].")
    assert "outside" in kinds(v2)  # no worked example around it: still a claim


def test_real_claims_still_boxed():
    for c in ("Water boils at 100 C at sea level.", "Jefferson wrote the Declaration in 1776 and later became president."):
        a, v = parse_view(c + "\n\nMolarity is moles per litre [1].")
        assert "outside" in kinds(v), c


def test_feedback_page_third_person_draft_lines_not_boxed():
    blocks = [{"kind": "outside", "md": "The proof skips the base case.", "cites": []}]
    assert kinds(view(blocks, data={"feedback": True}, mode="concepts")) == ["text"]
    assert kinds(view(blocks, mode="concepts")) == ["outside"]


# ---- 3 leading uncited sentence
def test_leading_uncited_claim_before_cited_sentence_is_labelled():
    a, v = parse_view("Jefferson was born in Virginia in 1743 and trained as a lawyer. Madison also thought distrust would block combinations [2].")
    assert [b.kind for b in a.blocks] == ["outside", "text"] and kinds(v) == ["outside", "text"]
    a, _ = parse_view("Here is a summary. Madison also thought distrust would block combinations [2].")
    assert [b.kind for b in a.blocks] == ["text"]


# ---- 4 splitter
def test_abbreviations_do_not_split_sentences():
    for s in ("Essay 2 covers Federalist Nos. 10 and 51 and how far they matched the 1780s conflicts [1].",
              "Gov. Bowdoin sent Gen. Lincoln to Springfield in 1787 [1].",
              "The U.S. Constitution created a large national republic [1].",
              "See pp. 10 to 12 and vol. 2 for details, e.g. Sen. Smith [1]."):
        assert len(grounding._sentences(s)) == 1, s
        a, _ = parse_view(s)
        assert len(a.blocks) == 1 and a.blocks[0].kind == "text", s
    assert len(grounding._sentences("It ended. Then it began [1].")) == 2


# ---- 5 check line and twin note
def test_bottom_count_lines_are_hidden_when_the_check_line_says_it(ready):
    r = page(ready, [{"kind": "outside", "md": f"Exam favourite {grounding.REMOVED_MARK}.", "cites": []}])
    assert "Check: 1 citation removed" in r.text and "<span hidden data-removed-citations>1</span>" in r.text
    assert "pointed to a source that doesn't exist and was removed" not in r.text


def test_twin_check_intro_is_not_said_twice():
    note = "Your message has an answer to your graded Problem 3 in it, so here is a similar problem. The app does not say whether the answer is right."
    intro = ("This was marked graded (Problem 3), so the app cannot say whether your answer to it is right or wrong. "
             "Check your method on this twin problem instead.")
    v = view([{"kind": "text", "md": intro, "cites": []}, {"kind": "text", "md": "Solve it [1].", "cites": [1]}],
             mode="twin", data={"routing_note": note})
    assert "cannot say whether" not in "".join(str(b["html"]) for b in v["blocks"])
    assert v["routing_note"].count("not say whether") == 1 and "Check your method" in v["routing_note"]


# ---- 6 reader
def test_list_letter_is_not_part_of_the_math_span():
    assert web_md.mathify("A. A^TA\\hat{x} = A^Tb") == "A. $A^TA\\hat{x} = A^Tb$"


def test_norm_equation_is_one_span():
    assert web_md.mathify("x^TA^TAx=0=\\|Ax\\|^2") == "$x^TA^TAx=0=\\|Ax\\|^2$"


def test_stoichiometry_fractions_render_as_fractions():
    h = web_md.mathify("% H= (2.5 g H)/(10.0 g compound) × 100%=25%")
    assert "\\frac{2.5\\ \\text{g H}}{10.0\\ \\text{g compound}}" in h and h.startswith("$") and h.endswith("$")
    h2 = web_md.mathify("4.7 g K ((mol K)/(39.10 g K)) = 0.12 mol K")
    assert "\\frac{\\text{mol K}}{39.10\\ \\text{g K}}" in h2 and "0.12\\ \\text{mol K}" in h2
    assert web_md.mathify("V2= (C1V1)/(C2)") == "$V_{2}= \\frac{C_{1}V_{1}}{C_{2}}$"
    assert "$\\frac{1\\ \\text{mol}}{18.02\\ \\text{g}}$ to convert" in web_md.mathify("Multiply by (1 mol)/(18.02 g) to convert grams.")


def test_reader_heading_math_shown_once(ready):
    mid = ready.store.add_material(ready.cid, "notes", "Lec")
    ready.store.add_chunks(mid, [{"text": "body", "loc_kind": "section", "loc_start": "1", "loc_label": "$$c_{ij} = 1$$",
                                  "heading": "$$c_{ij} = 1$$"}]) if hasattr(ready.store, "add_chunks") else None
    t = ready.client.get(f"/read/{mid}").text
    assert t.count("c_{ij} = 1") <= 1


# ---- 7 sources and chains
def test_past_offering_title_not_repeated():
    src = [{"chunk_id": 1, "material_id": 1, "loc_label": "p. 1", "material_title": "CHEM 1210 Fall 2025 Exam 3 (past offering)",
            "text": "x", "material_kind": "past_offering"}]
    assert "(past offering)" not in grounding.format_sources(src).split("\n")[1] or "(past offering)" not in grounding.format_sources(src).replace("Past offering", "")
    m = web_md.source_map([{"n": 1, "title": "CHEM 1210 Fall 2025 Exam 3 (past offering)", "label": "p. 1", "href": "/read/1",
                            "past": "Past offering: Fall 2025"}])
    assert m[1]["title"] == "CHEM 1210 Fall 2025 Exam 3"


def test_concept_chain_is_flat():
    ids = "abcdefgh"
    data = {"nodes": [{"id": c, "label": c.upper(), "cites": [1]} for c in ids],
            "edges": [{"a": a, "b": b, "label": "leads to", "cites": [1]} for a, b in zip(ids, ids[1:])]}
    g = web_helpers.concept_graph(data)
    root = g["forest"][0]
    assert len(root["chain"]) == 7 and root["branches"] == []


def test_concept_chain_page_depth(ready):
    ids = "abcdefgh"
    data = {"nodes": [{"id": c, "label": c.upper(), "cites": [1]} for c in ids],
            "edges": [{"a": a, "b": b, "label": "leads to", "cites": [1]} for a, b in zip(ids, ids[1:])]}
    gid = ready.store.add_guide(ready.cid, "concept_map", "Map", "", sources=sources(3), data=data)
    tree = ready.client.get(f"/guides/{gid}").text.split('<ul class="tree">')[1].split('<div class="mapwrap"')[0]
    assert tree.count("<ul") <= 1 and tree.count('<ol class="chain">') == 1 and tree.count("<li>") >= 8
