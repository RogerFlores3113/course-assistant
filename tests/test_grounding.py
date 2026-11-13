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
