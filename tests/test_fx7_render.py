"""Cycle 7 fx7-render: signposts are not boxed, draft comments are not claims, quieter "Also searched", flat-ish concept
map text, named code blocks and copy buttons."""
import subprocess
from pathlib import Path

from course_assistant import grounding, web_helpers, web_md
from test_render import material, page, ready, sources, view  # noqa: F401

STATIC = Path(__file__).resolve().parent.parent / "src" / "course_assistant" / "static"


def srcs(n=5):
    return [{"chunk_id": i, "material_id": 1, "loc_label": f"p. {i}", "material_title": f"T{i}", "text": "x",
             "href": "/read/1"} for i in range(1, n + 1)]


def parse(raw, n=5, **kw):
    a = grounding.parse_answer(raw, srcs(n))
    return a, web_helpers.answer_view(a.body_md, grounding.source_rows(srcs(n), a.used), blocks=a.blocks, **kw)


# ---- 1 signposts
def test_fed10_structure_answer_has_no_boxes():
    raw = ("Madison builds the argument in three steps. First, he defines the problem of faction as a group united by passion [1]. "
           "Second, he compares the two cures for it [2]. Third, he turns to the remedy and its limits [3].")
    a, v = parse(raw)
    assert not any(b.kind == "outside" for b in a.blocks) and not v["outside"]


def test_topic_lead_in_before_cited_sentence_is_plain():
    a, v = parse("On AI, the syllabus says this. Generative tools are not allowed on exams [1].")
    assert [b.kind for b in a.blocks] == ["text"]


def test_look_alike_claims_are_still_boxed():
    for lead in ("Jefferson founded the University of Virginia.", "The Civil War began because of slavery.",
                 "On AI, the syllabus says this."):
        a, _ = parse(lead + " The draft was due in 1787 [2].") if lead.startswith("Jefferson") or lead.startswith("The Civil") \
            else parse(lead + " Washington founded the army in a field.")
        assert any(b.kind == "outside" for b in a.blocks), lead


# ---- 2 comments about the student's own draft
def test_draft_comment_is_not_a_claim():
    md = "The Markov example is a good one to expand."
    assert grounding.is_non_claim(md, feedback=True)
    assert not grounding.is_non_claim(md)  # outside feedback it is a statement about the subject
    assert not grounding.is_non_claim("Markov chains converge to a steady state.", feedback=True)
    assert not grounding.is_non_claim("Madison wrote Federalist 10 in 1787.", feedback=True)


def test_feedback_page_does_not_box_the_comment():
    raw = "Your thesis names a claim [1]. The Markov example is a good one to expand."
    a, _ = parse(raw)
    v = web_helpers.answer_view(a.body_md, grounding.source_rows(srcs(), a.used), blocks=a.blocks, data={"feedback": True})
    assert not v["outside"]


# ---- 3 also searched
def test_also_searched_drops_rows_unrelated_to_the_question():
    rows = [{"n": 1, "chunk_id": 1, "material_id": 1, "label": "p. 1", "title": "Notes", "href": "/read/1",
             "words": " ".join(grounding.topic_stems("LU factorization elimination multipliers lower triangular"))},
            {"n": 2, "chunk_id": 2, "material_id": 1, "label": "p. 2", "title": "Orthonormal basis", "href": "/read/1",
             "words": " ".join(grounding.topic_stems("orthonormal basis Gram Schmidt projection"))},
            {"n": 3, "chunk_id": 3, "material_id": 1, "label": "p. 3", "title": "Old row", "href": "/read/1"},
            {"n": 4, "chunk_id": 4, "material_id": 1, "label": "p. 4", "title": "Cited", "href": "/read/1"}]
    v = web_helpers.answer_view("Elimination steps are listed [4].", rows,
                                question="How do multipliers work in LU factorization elimination?")
    names = [s["title"] for s in v["unused_sources"]]
    assert "Notes" in names and "Orthonormal basis" not in names and "Old row" in names


def test_source_rows_keep_topic_words():
    s = [{"chunk_id": 1, "material_id": 1, "loc_label": "p. 1", "material_title": "T", "text": "Eigenvalues and eigenvectors",
          "href": "/read/1"}]
    assert "eigen" in grounding.source_rows(s)[0]["words"]


# ---- 4 concept map
def deep_map():
    # a: b, c ; b: d, e ; d: f, g ; f: h, i  (branching all the way down)
    edges = [("a", "b", "found with"), ("a", "c", "uses"), ("b", "d", "needs"), ("b", "e", "gives"), ("d", "f", "gives"),
             ("d", "g", "gives"), ("f", "h", "gives"), ("f", "i", "gives")]
    return {"nodes": [{"id": c, "label": c.upper(), "cites": [1]} for c in "abcdefghi"],
            "edges": [{"a": a, "b": b, "label": lab, "cites": [1]} for a, b, lab in edges]}


def test_branching_map_text_nests_at_most_two_lists(ready):
    gid = ready.store.add_guide(ready.cid, "concept_map", "Map", "", sources=sources(3), data=deep_map())
    tree = ready.client.get(f"/guides/{gid}").text.split('<ul class="tree">')[1].split('<div class="mapwrap">')[0]
    depth = best = 0
    import re
    for m in re.finditer(r"<(/?)ul\b", tree):
        depth += -1 if m.group(1) else 1
        best = max(best, depth)
    assert best <= 3  # the tree's own ul, plus at most two branch levels
    for c in "ABCDEFGHI":
        assert f"<strong data-math>{c}</strong>" in tree


def test_map_edge_reads_as_one_phrase(ready):
    gid = ready.store.add_guide(ready.cid, "concept_map", "Map", "", sources=sources(3), data=deep_map())
    t = ready.client.get(f"/guides/{gid}").text
    assert "found with</span>: <strong data-math>B</strong>" in t
    assert "links to, found with" not in t and "links to</span> <span" not in t


def test_map_unlabelled_edge_says_links_to(ready):
    data = {"nodes": [{"id": "a", "label": "A", "cites": [1]}, {"id": "b", "label": "B", "cites": [1]},
                      {"id": "c", "label": "C", "cites": [1]}],
            "edges": [{"a": "a", "b": "b", "label": "", "cites": [1]}, {"a": "a", "b": "c", "label": "", "cites": []}]}
    gid = ready.store.add_guide(ready.cid, "concept_map", "Map", "", sources=sources(3), data=data)
    tree = ready.client.get(f"/guides/{gid}").text.split('<ul class="tree">')[1].split('<div class="mapwrap">')[0]
    assert "links to</span>: <strong data-math>B</strong>" in tree and tree.count("Not from your materials") == 1


# ---- 5 code blocks
def test_pre_has_a_name():
    html = web_md.render("```julia\nx = 1\n```")
    assert 'role="group"' in html and 'aria-label="julia code"' in html
    assert 'aria-label="Code"' in web_md.render("```\nplain\n```")


def test_copy_buttons_get_distinct_names():
    js = (STATIC / "app.js").read_text()
    assert "Copy this code" not in js and '" of " + pres.length' in js
    r = subprocess.run(["node", "--check", str(STATIC / "app.js")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
