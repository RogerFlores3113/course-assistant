import pytest

from course_assistant import disclaimer, grounding
from course_assistant.providers import FakeProvider, ProviderError
from course_assistant.store import Store


def srcs(n):
    return [{"chunk_id": i, "material_id": 1, "material_title": "Lec", "material_kind": "slides", "loc_kind": "slide",
             "loc_start": i, "loc_label": f"slide {i}", "heading": "", "text": f"text {i}", "href": f"/read/1?at={i}#c{i}",
             "score": 1.0} for i in range(1, n + 1)]


def test_build_messages_numbering_and_depth():
    sys, msgs = grounding.build_messages("What is F?", srcs(2), depth="quick")
    assert "[1] Lec, slide 1" in msgs[-1]["content"] and "[2] Lec, slide 2" in msgs[-1]["content"]
    assert "text 2" in msgs[-1]["content"] and "What is F?" in msgs[-1]["content"]
    assert "a few sentences" in sys and "<outside>" in sys and "Never invent page numbers" in sys
    assert "no background" in grounding.build_messages("q", [], depth="from scratch")[0]
    assert "Depth: standard" in grounding.build_messages("q", [], depth="weird")[0]


def test_build_messages_no_sources_and_history():
    _, msgs = grounding.build_messages("q", [], history=[{"role": "user", "content": "a"},
                                                          {"role": "assistant", "content": "b"},
                                                          {"role": "system", "content": "x"}])
    assert "none" in msgs[-1]["content"].lower()
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    _, m2 = grounding.build_messages("q", [], history=[{"role": "user", "content": "dangling"}])
    assert [m["role"] for m in m2] == ["user", "assistant", "user"]


def test_parse_basic_and_used():
    a = grounding.parse_answer("Force is mass times acceleration [1].\n\nMomentum is mv [2][1].", srcs(3))
    assert [b.kind for b in a.blocks] == ["text", "text"]
    assert a.blocks[1].cites == [1, 2] and a.used == [1, 2] and not a.outside and not a.twin
    assert a.body_md.endswith(disclaimer.FOOTER + "_") or a.body_md.endswith(disclaimer.FOOTER)
    assert disclaimer.FOOTER in a.body_md


def test_out_of_range_citations_removed():
    a = grounding.parse_answer("Claim one [9]. Claim two [1][7]. Another [1, 5, 2]. Range [2-3].", srcs(3))
    assert "[9]" not in a.body_md and "[7]" not in a.body_md and "[5]" not in a.body_md
    assert a.used == [1, 2, 3]
    assert "[1][2]" in a.blocks[0].md


def test_paragraph_only_invalid_cite_becomes_outside():
    a = grounding.parse_answer("A long claim about the world that is not supported anywhere at all [8].", srcs(2))
    assert a.blocks[0].kind == "outside" and a.used == [] and a.outside


def test_markdown_link_not_treated_as_citation():
    a = grounding.parse_answer("See [1](http://x.test) for more details about the topic here and now [1].", srcs(1))
    assert "[1](http://x.test)" in a.blocks[0].md


def test_outside_tags():
    raw = ("Cited claim [1].\n\n<outside>Extra background from general knowledge [1].\n\nSecond para.</outside>\n\n"
           "After [2].")
    a = grounding.parse_answer(raw, srcs(2))
    assert [b.kind for b in a.blocks] == ["text", "outside", "outside", "text"]
    assert "[1]" not in a.blocks[1].md and a.outside and a.used == [1, 2]
    assert "Not from your materials" in a.body_md
    assert a.body_md.count("Not from your materials") == 1  # one label for a run


def test_unclosed_outside_tag():
    a = grounding.parse_answer("Fine [1].\n\n<outside>never closed text", srcs(1))
    assert a.blocks[1].kind == "outside" and a.blocks[1].md == "never closed text"


def test_uncited_paragraph_is_outside_but_headings_and_leadins_stay():
    raw = ("## Newton's laws\n\nHere is the idea:\n\nThis paragraph states a lot of facts without any citation at all.\n\n"
           "Short line.\n\n---\n\nCited properly here [1].")
    a = grounding.parse_answer(raw, srcs(1))
    kinds = [(b.kind, b.md[:12]) for b in a.blocks]
    assert kinds[0][0] == "text" and kinds[1][0] == "text"
    assert a.blocks[2].kind == "outside"
    assert a.blocks[3].kind == "text" and a.blocks[4].kind == "text" and a.blocks[5].cites == [1]
    assert a.outside


def test_no_sources_everything_outside():
    a = grounding.parse_answer("Some general answer about physics and forces and motion.\n\nOkay.", [])
    assert all(b.kind == "outside" for b in a.blocks) and a.outside and a.used == []


def test_code_fence_kept_whole():
    a = grounding.parse_answer("Example [1]:\n\n```\nx = 1\n\ny = 2\n```\n\nAfter [1].", srcs(1))
    assert any("x = 1\n\ny = 2" in b.md for b in a.blocks)


def test_empty_reply():
    a = grounding.parse_answer("   ", srcs(1))
    assert a.outside and disclaimer.FOOTER in a.body_md


def make_store():
    s = Store(":memory:")
    c = s.add_course("Physics")
    m = s.add_material(c, "slides", "Lecture 1")
    s.replace_chunks(m, [{"text": "Newton's second law: force equals mass times acceleration.", "loc_kind": "slide",
                          "loc_start": 3, "loc_label": "slide 3"}])
    return s, c, m


def test_ask_saves_guide_and_chat():
    s, c, m = make_store()
    p = FakeProvider(["Force equals mass times acceleration [1]. Nothing [4].\n\n<outside>Also used in engineering.</outside>"])
    a = grounding.ask(s, c, "What is Newton's second law?", p, depth="quick")
    assert a.guide_id and a.used == [1] and a.outside
    g = s.guide(a.guide_id)
    assert g["kind"] == "answer" and g["question"].startswith("What is Newton") and g["body_md"] == a.body_md
    assert g["sources"][0]["n"] == 1 and g["sources"][0]["label"] == "slide 3" and g["sources"][0]["href"].startswith("/read/")
    assert g["data"]["depth"] == "quick"
    sent = p.calls[0]
    assert "[1] Lecture 1, slide 3" in sent["messages"][-1]["content"] and "Depth: quick" in sent["system"]
    chat = s.chat_messages(c)
    assert [x["role"] for x in chat] == ["user", "assistant"] and chat[1]["guide_id"] == a.guide_id


def test_ask_without_sources_tells_provider():
    s, c, m = make_store()
    p = FakeProvider(["<outside>The materials do not cover mitochondria. In general they make energy.</outside>"])
    a = grounding.ask(s, c, "What are mitochondria?", p, record_chat=False)
    assert "SOURCES: none" in p.calls[0]["messages"][-1]["content"]
    assert a.sources == [] and a.outside and a.used == []
    assert s.chat_messages(c) == []
    assert s.guide(a.guide_id)["data"]["no_sources"] is True


def test_ask_excludes_graded_material_from_sources():
    s, c, m = make_store()
    s.update_material(m, graded=1)
    p = FakeProvider(["<outside>nothing</outside>"])
    a = grounding.ask(s, c, "What is Newton's second law of force?", p)
    # the question may match graded text; either way the graded chunk must not be sent as a source
    assert all("Newton's second law" not in x["text"] for x in a.sources)


def test_ask_provider_error_propagates():
    s, c, m = make_store()

    def boom(sys, msgs):
        raise ProviderError("nope")
    with pytest.raises(ProviderError):
        grounding.ask(s, c, "force?", FakeProvider(boom))
    assert s.guides(c) == []


def test_ask_routes_graded_to_twin():
    s, c, m = make_store()
    s.add_graded_item(c, "HW 1 Problem 1", "A 5 kg block is pushed with 20 N. Find the acceleration of the block.")
    twin_reply = ("<twin_problem>A 7 kg cart is pulled with 35 N. Find its acceleration.</twin_problem>"
                  "<solution>Use F = ma [1]. a = 35/7 = 5 m/s^2 [1].</solution>")
    a = grounding.ask(s, c, "A 5 kg block is pushed with 20 N. Find the acceleration of the block.", FakeProvider([twin_reply]))
    assert a.twin and s.guide(a.guide_id)["kind"] == "twin"


# ---- cycle 1 render fixes (panel p3, p4, p1, p2)
def test_matrices_are_not_citations_p3():
    t = "The matrix A = [[4,1],[2,3]] and [[1,1],[0,1]] stay as written [1]."
    cleaned, found = grounding.clean_citations(t, 5)
    assert "[[4,1],[2,3]]" in cleaned and "[[1,1],[0,1]]" in cleaned and found == [1]
    blocks, used = grounding.parse_blocks("Use [[1,1],[0,1]] here, as in the notes [2].", 3)
    assert "[[1,1],[0,1]]" in blocks[0].md and used == [2]


def test_math_and_code_are_never_cited():
    t = "Take $[1, 2]$ and $$x[3]$$ and `a[1]` then\n```\nb = [1]\n```\nend [2]."
    cleaned, found = grounding.clean_citations(t, 2)
    assert cleaned == t and found == [2]
    cleaned, _ = grounding.clean_citations("Vector [2, 1, 1] and [0.5, 3] and [0-1] and [9] gone.", 4)
    assert "[2, 1, 1]" in cleaned and "[0.5, 3]" in cleaned and "[0-1]" in cleaned and "[9]" not in cleaned
    assert grounding.cited_numbers("matrix [[1],[2]] and $[1]$ and real [3]") == [3]


def test_whitespace_in_code_survives_cleaning():
    t = "```python\nx   = 1\n    y = [1]\n```\nTrue [1] ."
    cleaned, _ = grounding.clean_citations(t, 2)
    assert "x   = 1" in cleaned and "    y = [1]" in cleaned and cleaned.endswith("True [1].")


def test_invalid_citation_leaves_visible_marker_p3():
    cleaned, found = grounding.clean_citations("A claim [9].", 3)
    assert grounding.REMOVED_MARK in cleaned and found == []
    a = grounding.parse_answer("Eigenvalues matter [1]. Exam favourite [10].", srcs(2))
    assert any(grounding.REMOVED_MARK in b.md for b in a.blocks)
    assert "(citation removed)" in a.body_md and grounding.REMOVED_MARK not in a.body_md


def test_outside_labelling_skips_structural_lines_p1_p3():
    raw = ("# Plan\n\nStep 2:\n\n**Step 3: Subtract the rows**\n\nAnswer:\n\n$$x = 5$$\n\n7x + 3y + 6z = 8\n\n"
           "```python\nprint(1)\n```\n\nSolve it carefully with the rule from lecture [1].")
    a = grounding.parse_answer(raw, srcs(1))
    assert [b.kind for b in a.blocks] == ["text"] * 8 and a.used == [1]
    b = grounding.parse_answer("Step 4: Subtract row 1 from row 2.", srcs(1)).blocks
    assert b[0].kind == "outside"  # a sentence with a claim in it is still labelled
    assert grounding.parse_answer("The eigenvalue is 3 and the vector is 7x.", srcs(1)).blocks[0].kind == "outside"


def test_not_covered_lines_become_the_note_p3():
    a = grounding.parse_answer("The materials do not cover week 8.\n\nDoors close at noon [1].", srcs(1))
    assert [b.md for b in a.blocks] == ["Doors close at noon [1]."] and a.not_covered
    b = grounding.parse_answer("Real claim [1].", srcs(1))
    assert not b.not_covered
    c = grounding.parse_answer("<outside>The notes don't mention this at all.</outside>", srcs(1))
    assert c.blocks == [] and c.not_covered
    d = grounding.parse_answer("<outside>The materials do not cover it. In general it is true.</outside>", srcs(1))
    assert d.blocks and d.blocks[0].kind == "outside"  # more than the note: kept
    assert grounding.parse_answer("Something general.", []).not_covered
    assert grounding.parse_answer("Mostly general words here and there for a long while now.\n\nTiny [1].", srcs(1)).not_covered


def test_empty_reply_still_gets_a_message():
    assert grounding.parse_answer("", srcs(1)).blocks[0].kind == "outside"


QUOTE_SRC = [{"text": "The  ordinary course of public life\nrequires that merely judg-\nment be exercised by the people always."}]


def test_verify_quotes_flags_a_misquote_p4():
    good = 'Madison writes "the ordinary course of public life requires that merely judgment be exercised" [1].'
    bad = 'He says "the ordinary course of public life requires that neither force nor will exercised" [1].'
    ok = grounding.verify_quotes([grounding.Block("text", good, [1])], QUOTE_SRC)[0]
    assert grounding.QUOTE_OPEN not in ok.md
    flagged = grounding.verify_quotes([grounding.Block("text", bad, [1])], QUOTE_SRC)[0]
    assert flagged.md.count(grounding.QUOTE_OPEN) == 1 and flagged.md.count(grounding.QUOTE_CLOSE) == 1
    assert "quote not found in the cited source" in grounding.render_body([flagged])


def test_verify_quotes_details():
    curly = "He says \u201cthe ordinary course of public life requires that merely judgment be exercised\u201d [1]."
    assert grounding.QUOTE_OPEN not in grounding.verify_quotes([grounding.Block("text", curly, [1])], QUOTE_SRC)[0].md
    ell = 'He says "the ordinary course of public life ... be exercised by the people always" [1].'
    assert grounding.QUOTE_OPEN not in grounding.verify_quotes([grounding.Block("text", ell, [1])], QUOTE_SRC)[0].md
    short = 'The word "not in there" appears [1].'
    assert grounding.QUOTE_OPEN not in grounding.verify_quotes([grounding.Block("text", short, [1])], QUOTE_SRC)[0].md
    other = 'He says "a completely different eight word quote goes here now" [2].'
    assert grounding.QUOTE_OPEN in grounding.verify_quotes([grounding.Block("text", other, [2])], QUOTE_SRC + [{"text": "x"}])[0].md
    nocite = 'He says "a completely different eight word quote goes here now".'
    assert grounding.QUOTE_OPEN not in grounding.verify_quotes([grounding.Block("text", nocite, [])], QUOTE_SRC)[0].md


def test_parse_answer_verifies_quotes():
    s = [{**srcs(1)[0], "text": "Liberty is to faction what air is to fire, an aliment without which it instantly expires."}]
    a = grounding.parse_answer('Madison: "Liberty is to faction what water is to fire, an aliment without which it dies" [1].', s)
    assert grounding.QUOTE_OPEN in a.blocks[0].md


def test_system_prompt_has_format_rules():
    p = grounding.system_prompt()
    assert grounding.FORMAT_RULES in p and "bmatrix" in p and "[[4,1],[2,3]]" in p
