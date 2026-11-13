"""Cycle 4 fx4-render: lists, not-covered lines, claim labels, reader math, quote check, history, labels, snippets."""
import re

from course_assistant import grounding, web_helpers, web_md
from test_render import material, page, ready, sources, view  # noqa: F401


# ---- 1 ordered lists
def test_loose_numbered_list_is_one_list():
    h = str(web_md.render("1. Take any vector.\n\n2. For each one, scale it.\n\n3. Add up."))
    assert h.count("<ol") == 1 and h.count("<li>") == 3


def test_ol_keeps_start_number():
    assert '<ol start="5"><li>five</li><li>six</li></ol>' in str(web_md.render("5. five\n6. six"))


def test_numbering_carries_across_details_and_split_blocks():
    md = "1. Q one\n\n<details><summary>Answer</summary>\n\nsol\n\n</details>\n\n1. Q two\n\n<details><summary>A</summary>\n\nsol\n\n</details>\n\n1. Q three"
    h = str(web_md.render(md))
    assert 'start="2"' in h and 'start="3"' in h
    shared: dict = {}
    a = str(web_md.render("1. a\n2. b", numbering=shared))
    b = str(web_md.render("1. c", numbering=shared))
    assert "start" not in a and 'start="3"' in b
    assert "start" not in str(web_md.render("1. a\n\nA paragraph.\n\n1. c", numbering={}))


def test_twin_steps_split_by_uncited_items_stay_one_numbered_list(ready):
    blocks = [{"kind": "text", "md": "1. Set up [1].", "cites": [1]},
              {"kind": "outside", "md": "2. Row 2 minus 3 times row 1 gives $d = 18$.", "cites": []},
              {"kind": "text", "md": "3. Done [1].", "cites": [1]}]
    v = view(blocks, mode="twin")
    html = "".join(str(b["html"]) for b in v["blocks"])
    assert html.count("<ol") == 1 and html.count("<li>") == 3 and "Not from your materials" not in html


# ---- 2 not covered stays visible
def test_not_covered_sentence_kept_and_flagged_when_cited(ready):
    a = grounding.parse_answer("Exam dates were Dec 3 [1].\n\n<outside>The sources I was given do not include this term's exam dates.</outside>", sources(1))
    assert any("do not include this term" in b.md for b in a.blocks) and all(b.kind == "text" for b in a.blocks)
    v = view([{"kind": b.kind, "md": b.md, "cites": b.cites} for b in a.blocks], 1)
    assert v["partly_covered"] and not v["not_covered"]
    t = page(ready, [{"kind": b.kind, "md": b.md, "cites": b.cites} for b in a.blocks], n=1).text
    assert "Part of your question is not covered by your saved materials." in t
    assert 'class="answer" data-partly-covered' in t and "do not include this term" in t


def test_no_partly_note_when_everything_is_cited(ready):
    t = page(ready, [{"kind": "text", "md": "Fine [1].", "cites": [1]}]).text
    assert "data-partly-covered" not in t


# ---- 3 claim labels
def test_non_claims_are_not_boxed():
    for md in ["Rest.", "Day 7: Rest.", "Read that example first, then try the next one.",
               "- What is your claim?\n- Which source backs it up?", "Where does $x = 2$ come from?",
               "They do not include the code that generates the steps.",
               "The sources I was given only show the total of 60 points.", "Questions to ask yourself:",
               "Here is how your draft measures up.", grounding.EMPTY_REPLY]:
        assert grounding.is_non_claim(md), md
    assert not grounding.is_non_claim("The eigenvalue is 3 and the vector is 7x.")
    assert not grounding.is_non_claim("Water boils at 100 C at sea level.")
    assert grounding.is_non_claim("0.220 mol/L x 0.6000 L = 0.132 mol", calc=True)
    assert not grounding.is_non_claim("0.220 mol/L x 0.6000 L = 0.132 mol")


def test_view_does_not_box_arithmetic_in_twins_or_empty_notice(ready):
    t = page(ready, [{"kind": "text", "md": "Setup [1].", "cites": [1]},
                     {"kind": "outside", "md": "0.220 x 0.6 = 0.132 mol", "cites": []}], data={"mode": "twin"}, kind="twin").text
    assert "0.132 mol" in t and 'class="outside"' not in t
    t = page(ready, [{"kind": "outside", "md": grounding.EMPTY_REPLY, "cites": []}]).text
    assert grounding.EMPTY_REPLY in t and 'class="outside"' not in t
    t = page(ready, [{"kind": "outside", "md": "Water boils at 100 C.", "cites": []}]).text
    assert 'class="outside"' in t


# ---- 4 reader
def test_reader_math_for_raw_tex_and_variables():
    assert web_md.mathify(r"\hat{x} = (A^TA)^{-1}A^Tb") == r"$\hat{x} = (A^TA)^{-1}A^Tb$"
    assert r"$\mathbb{R}^m$" in web_md.mathify(r"vectors in \mathbb{R}^m live here")
    out = web_md.mathify("the solute, **n*1 = *n*2. Thus")
    assert "$n_{1} = n_{2}$" in out and "*" not in out and "cdot" not in out
    line = "% C= (9 mol C × molar mass C)/(molar mass C9H8O4) × 100= (9 × 12.01 g/mol)/(180.159 g/mol) × 100"
    for span in re.findall(r"\$([^$]*)\$", web_md.mathify(line)):
        assert web_md._balanced(span)
    assert "prose words stay" in web_md.mathify(r"prose words stay and \hat{x} here")


def test_reader_hides_empty_parens():
    h = str(web_md.render_reader("Thomas Jefferson () wrote it (see Fig. 2). Bill of Rights ()."))
    assert "()" not in h and "(see Fig. 2)" in h


def test_reader_404_for_never_existed_410_for_deleted(ready):
    r = ready.client.get("/read/9999")
    assert r.status_code == 404 and "No such passage" in r.text
    mid = ready.store.add_material(ready.cid, "notes", "Gone")
    ready.store.delete_material(mid)
    r = ready.client.get(f"/read/{mid}")
    assert r.status_code == 410 and "no longer in your materials" in r.text


def test_reader_time_label_is_the_start_and_window_is_secondary(ready):
    mid, ids = material(ready, [{"text": "Spoken", "loc_kind": "time", "loc_start": "298", "loc_end": "344", "loc_label": "4:58"}])
    t = ready.client.get(f"/read/{mid}").text
    assert '<h2 class="loc">4:58</h2>' in t and "4:58 to 5:44" in t


def test_reader_original_file_link_says_new_tab(ready):
    from course_assistant import paths
    mid = ready.store.add_material(ready.cid, "notes", "F")
    ready.store.replace_chunks(mid, [{"text": "x", "loc_kind": "page", "loc_start": "1", "loc_label": "p. 1"}])
    d = paths.files_dir() / "1"
    d.mkdir(parents=True, exist_ok=True)
    (d / "f.pdf").write_bytes(b"%PDF-1.4")
    ready.store._exec("UPDATE materials SET filename = ? WHERE id = ?", ("1/f.pdf", mid))
    t = ready.client.get(f"/read/{mid}").text
    if "Open the original file" in t:
        assert "(opens in a new tab)" in t


# ---- 5 quote check with math
def test_invented_quote_with_symbol_is_flagged():
    src = [{"text": "An eigenvalue is a scalar lambda with Av = lambda v for some vector.", "material_title": "N", "loc_label": "p. 1"}]
    bad = grounding.Block("text", 'The notes say "the number $\\lambda$ is called the gizmo of the matrix" [1].', [1])
    grounding.verify_quotes([bad], src)
    assert grounding.QUOTE_OPEN in bad.md
    ok = grounding.Block("text", 'The notes say "an eigenvalue is a scalar $\\lambda$ with Av = lambda v" [1].', [1])
    grounding.verify_quotes([ok], src)
    assert grounding.QUOTE_OPEN not in ok.md


# ---- 6 history
def test_history_markers_are_not_sent_to_the_model():
    md = (f'Claim [1]{grounding.REMOVED_MARK}. He said {grounding.QUOTE_OPEN}"a made up line here"{grounding.QUOTE_CLOSE}'
          f' [quote not found in the cited source]\n\n**Not from your materials:** > A thing.\n\n{grounding.disclaimer.FOOTER}')
    _, msgs = grounding.build_messages("q", [], history=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": md}])
    sent = msgs[1]["content"]
    for bad in ("citation removed", "quote not found", "Not from your materials", "{{", "AI can make mistakes"):
        assert bad not in sent
    assert "Claim [1]" in sent and "a made up line here" in sent


# ---- 7 labels
def test_chip_and_sources_list_label_match(ready):
    srcs = [{"n": 1, "chunk_id": 1, "href": "/read/1?at=1#c1", "title": "Diagonalization", "label": "Diagonalization"}]
    t = web_helpers.answer_view("", srcs, blocks=[{"kind": "text", "md": "A [1].", "cites": [1]}])
    assert t["sources"][0]["label"] == ""
    html = str(web_md.render("A [1].", srcs))
    assert 'aria-label="Source 1: Diagonalization"' in html


# ---- 8 twin dedupe and one list
def test_twin_page_says_it_once_and_has_headings(ready):
    blocks = [{"kind": "text", "md": "This was marked graded (Problem 1), so the app solved a twin problem with different numbers instead of your problem. The original is yours to do.", "cites": []},
              {"kind": "text", "md": "**Twin problem**\n\nFind it.", "cites": []},
              {"kind": "text", "md": "**Worked solution of the twin**", "cites": []},
              {"kind": "text", "md": "Step [1].", "cites": [1]}]
    t = page(ready, blocks, data={"mode": "twin", "routing_note": "This looks like your graded Problem 1, so here is a similar problem."}, kind="twin").text
    assert t.count("This looks like your graded Problem 1") == 1 and "This was marked graded" not in t
    assert "The original is yours to do." in t and "Worked twin problem.</strong>" not in t
    assert re.search(r"<h\d>Twin problem</h\d>", t) and re.search(r"<h\d>Worked solution of the twin</h\d>", t)


def test_concept_map_copy_is_one_list():
    blocks = [{"kind": "text", "md": f"- **c{i}** [1]", "cites": [1]} for i in range(7)] + \
             [{"kind": "text", "md": f"- c{i} relates to c{i + 1} [1]", "cites": [1]} for i in range(7)]
    v = view(blocks, 1)
    assert "".join(str(b["html"]) for b in v["blocks"]).count("<ul") == 1


# ---- 9 snippets
def test_snippet_text_is_plain():
    s = web_helpers.snippet_text("...ere is a much better method called [barycentric interpolation](https://people.maths.ox.ac.uk/trefethen/barycen...")
    assert "[" not in s and "http" not in s and "barycentric interpolation" in s and not s.startswith("...ere")
    assert web_helpers.snippet_text("a **bold** and *slanted* word with `code` and $x$") == "a bold and slanted word with code and x"
    assert web_helpers.snippet_text("see [notes](/read/3?at=4) now") == "see notes now"
