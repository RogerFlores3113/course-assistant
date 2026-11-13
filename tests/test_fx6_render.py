"""Cycle 6 fx6-render: quotes across sentences, plain leading claims, non-claims, reader tables and math, one banner,
past-offering rows, heading blocks, long titles, folded source list, summary scrub note, type tokens."""
import re
from pathlib import Path

from course_assistant import grounding, study, web_helpers, web_md


def srcs(n, **extra):
    return [{"chunk_id": i, "material_id": 1, "loc_label": f"p. {i}", "material_title": f"T{i}", "text": "x",
             "href": "/read/1", **extra} for i in range(1, n + 1)]


def parse(raw, n=7):
    a = grounding.parse_answer(raw, srcs(n))
    return a, web_helpers.answer_view(a.body_md, grounding.source_rows(srcs(n), a.used), blocks=a.blocks)


# ---- 1 a quote that spans two sentences is one span with its citation
def test_quote_across_two_sentences_stays_whole():
    raw = 'The lecture says "The zero vector always satisfies A x equals lambda x, for any lambda. So we do not count it." [6]'
    a, v = parse(raw)
    assert [b.kind for b in a.blocks] == ["text"] and a.blocks[0].cites == [6] and a.used == [6]
    assert [b["kind"] for b in v["blocks"]] == ["text"] and not v["outside"]
    assert len(grounding._sentences('He wrote "First part here. Second part there." [2] Then he left.')) == 2


def test_uncited_sentence_after_a_quote_is_still_labelled():
    raw = 'The note says "We drop the lowest score. Nothing else changes." [1] Late work loses ten percent each day of delay.'
    a, _ = parse(raw)
    assert [b.kind for b in a.blocks] == ["text", "outside"]


# ---- 2 a plain leading claim is labelled, a lead-in is not
def test_plain_leading_claims_are_labelled():
    for lead in ("Jefferson founded the University of Virginia.", "Washington warned against parties in his Farewell Address.",
                 "The Civil War began because of slavery.",
                 "Many historians consider Hamilton the architect of the American financial system."):
        a, _ = parse(lead + " Madison also thought distrust would block combinations [2].")
        assert [b.kind for b in a.blocks] == ["outside", "text"], lead
        assert a.blocks[0].md == lead


def test_short_lead_ins_stay_plain():
    for lead in ("Here is a summary.", "Good question.", "Let's look at it.", "In short, yes."):
        a, _ = parse(lead + " Madison also thought distrust would block combinations [2].")
        assert [b.kind for b in a.blocks] == ["text"], lead


# ---- 3 non-claims stay boxed out of the label
def test_pointer_and_draft_feedback_lines_are_not_claims():
    assert grounding.is_non_claim("Concepts worth reviewing: eigenvalues and eigenvectors, diagonalization, the SVD, and elimination")
    assert grounding.is_non_claim("Topics to review: molarity, dilution and percent yield.")
    assert grounding.is_non_claim("Nothing in the draft yet addresses the molar masses.", feedback=True)
    assert grounding.is_non_claim("What the assignment asks for. It has a 'why' part and a 'describe' part.", feedback=True)
    assert not grounding.is_non_claim("Eigenvalues were first studied by Cauchy in 1829.")
    a = grounding.parse_answer("Concepts worth reviewing: molarity, dilution, and percent yield\n\nSee [1].", srcs(1))
    assert not any(b.kind == "outside" for b in a.blocks)


# ---- 4 reader
def test_key_equations_pipe_table_renders_as_a_table_with_math():
    md = "### Key Equations\n\n| M= (mol solute)/(L solution) |\n|---|\n| *C*1*V*1 = *C*2*V*2 |\n"
    h = str(web_md.render_reader(md))
    assert "<table>" in h and "|---|" not in h and "<br>" not in h
    assert "\\frac{\\text{mol solute}}{\\text{L solution}}" in h and "C_{1}V_{1} = C_{2}V_{2}" in h


def test_whole_line_equation_and_nested_fraction():
    assert web_md.mathify("    n=ML") == "    $n=ML$"
    assert web_md.mathify("Some words here").startswith("Some")
    h = web_md.mathify("(molecular mass (amu or (g)/(mol)))/(formula mass (amu or (g)/(mol))) =n units")
    assert h.startswith("$") and h.count("\\frac") == 3 and "(g)/(mol)" not in h


def test_list_number_and_letter_stay_out_of_the_math_span():
    t = ("Make e=b-B\\hat{y} \\perp to every column of B: B^T(b-B\\hat{y})=0 is the idea, so the last column is orthogonal "
         "to B. B^TB\\hat{y} = B^Tc (known as the normal equations) 1. \\hat{y} = (B^TB)^{-1}B^Tc 2. q = B\\hat{y} done")
    h = web_md.mathify(t)
    assert "$B^TB\\hat{y} = B^Tc$" in h and "to B. $B^TB" in h
    assert "1. $\\hat{y} = (B^TB)^{-1}B^Tc$" in h and "2. $q = B\\hat{y}$" in h
    assert "$\\hat{y}$?" in web_md.mathify("How do we find \\hat{y}? Next")


# ---- 5 banners and blocks
def test_not_covered_answer_shows_one_banner():
    a, v = parse("Part is explained here [1]. The sources do not say what the pH is.\n\n<outside>Acetic acid is a weak acid with a pH near 2.9 at that strength in water.</outside>", 1)
    assert not (v["not_covered"] and v["partly_covered"])


def test_blocked_reply_says_its_sentence_once():
    sentence = "This came too close to your graded Problem 1, so I left it out."
    body = sentence + "\n\nYou can ask me about one concept at a time."
    v = web_helpers.answer_view(body, [], blocks=[grounding.Block("text", body, [])], mode="concepts",
                                data={"routing_note": sentence}, course_id=1, question="q")
    html = "".join(str(b["html"]) for b in v["blocks"])
    assert sentence not in html and "one concept at a time" in html and v["routing_note"] == sentence


def test_past_offering_row_says_its_title_once():
    rows = [{"n": 1, "chunk_id": 1, "material_id": 1, "title": "CHEM 1210 Fall 2025 Exam 3 (past offering)",
             "label": "CHEM 1210 Fall 2025 Exam 3 (past offering)", "href": "/read/1", "past": "Past offering: Fall 2025"}]
    v = web_helpers.answer_view("Exam 3 had five questions [1].", rows)
    s = v["sources"][0]
    assert s["title"] == "CHEM 1210 Fall 2025 Exam 3" and s["label"] == ""
    assert grounding.format_sources([{"material_title": "A (past offering)", "loc_label": "A (past offering)", "text": "t",
                                      "material_kind": "past_offering"}]).count("A") <= 2


def test_heading_only_block_joins_the_block_after_it():
    blocks = [grounding.Block("text", "**Twin problem**\n\nSolve it.", []), grounding.Block("text", "**Worked solution of the twin**", []),
              grounding.Block("text", "Step one gives 4 [1].", [1])]
    v = web_helpers.answer_view("", srcs(1), blocks=blocks, mode="twin")
    assert len(v["blocks"]) == 2 and "Worked solution of the twin" in str(v["blocks"][1]["html"]) and "Step one" in str(v["blocks"][1]["html"])


def test_guide_view_can_hide_ask_normally():
    g = {"id": 1, "course_id": 1, "kind": "twin", "question": "How do I do Problem 1?", "body_md": "b", "sources": [],
         "data": {"mode": "concepts", "routing_note": "This looks like your graded Problem 1.", "blocks": [{"kind": "text", "md": "b", "cites": []}]}}
    assert web_helpers.guide_view(g)["ask_normal"] and web_helpers.guide_view(g, ask_normally=False)["ask_normal"] is None


def test_long_draft_question_makes_a_short_heading():
    long_q = "Here is my draft intro for Essay 1: " + "Eigenvectors matter because they make repeated multiplication simple. " * 6
    h = web_helpers.answer_heading(long_q, {})
    assert len(h) <= 80 and h.endswith("...")
    assert web_helpers.answer_heading(long_q, {"feedback": True, "graded_title": "Essay 1"}) == "Feedback on your draft for Essay 1"
    assert web_helpers.answer_heading("What is a basis?", {}) == "What is a basis?"
    v = web_helpers.answer_view("x", [], question=long_q)
    assert len(v["heading"]) <= 80


# ---- 6 also searched
def test_also_searched_is_short_and_relevant():
    rows = srcs(9)
    for i, r in enumerate(rows):
        r["score"] = 10.0 if i == 0 else 6.0 if i < 4 else 2.0
    rows = grounding.source_rows(rows, [1])
    v = web_helpers.answer_view("Cited [1].", rows)
    assert [x["ns"][0]["n"] for x in v["unused_sources"]] == [2, 3, 4]
    many = srcs(12)
    for r in many:
        r["score"] = 5.0
    v = web_helpers.answer_view("Cited [1].", grounding.source_rows(many, [1]))
    assert len(v["unused_sources"]) == 5
    assert len(web_helpers.answer_view("Cited [1].", [{"n": i, "title": f"T{i}", "label": "", "href": ""} for i in range(1, 10)])["unused_sources"]) == 5


# ---- 7 concept map
def test_summary_scrub_says_which_sentence_went():
    text = ("Chapter 7 describes faction [1]. Chapter Eight shows how Hamilton's program led to the first parties [2]. "
            "Both papers use the same terms [3].")

    def judge(t):
        hit = "led to the first parties" in t
        return (t, 1, "A part of this guide was left out because it read like finished writing for your graded work") if hit else (t, 0, "")
    out = study.scrub_sentences(judge, text, grounding._sentences)
    assert "Chapter 7 describes faction [1]." in out and "Both papers use" in out and "led to the first parties" not in out
    assert "sentence 2 of 3" in out and 'which begins "Chapter Eight shows how Hamilton\'s ..."' in out
    assert study.scrub_sentences(lambda t: (t, 0, ""), text, grounding._sentences) == text


def test_summary_scrub_hides_the_start_of_a_sentence_with_numbers():
    text = "Fine opening here [1]. The answer is x = 5 for the system [2]."
    judge = lambda t: (t, 1, "A part was left out") if "x = 5" in t else (t, 0, "")  # noqa: E731
    out = study.scrub_sentences(judge, text, grounding._sentences)
    assert "sentence 2 of 2" in out and "begins" not in out and "x = 5" not in out


def test_map_labels_are_hidden_from_the_text_flow():
    t = Path("src/course_assistant/templates/guide.html").read_text()
    assert t.count('<text') == 1 and 'edge-label' not in t
    assert 'text-anchor="middle" aria-hidden="true">{% for ln in b.lines %}' in t


# ---- 8 type tokens
def test_content_css_uses_type_tokens_for_small_text():
    css = Path("src/course_assistant/static/content.css").read_text()
    assert not re.search(r"font-size:\s*1[0-3]px", css)
