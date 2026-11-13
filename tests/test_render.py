"""Templates: answer partial, guide page, concept map, reader."""
import re
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from course_assistant import disclaimer, grounding, paths, web, web_helpers
from course_assistant.store import Store

KEY = "k" * 40


@pytest.fixture()
def ready(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_DATA_DIR, str(tmp_path / "data"))
    store = Store(":memory:")
    store.set_setting(disclaimer.ACK_SETTING, "2026-01-01T00:00:00+00:00")
    client = TestClient(web.create_app(store, key=KEY, provider_factory=lambda: None), base_url="http://127.0.0.1:8766")
    client.cookies.set(web.COOKIE, KEY)
    cid = store.add_course("Linear Algebra", code="MATH 2210", term="Fall 2026", color="green")
    return SimpleNamespace(store=store, client=client, cid=cid)


def sources(n=3):
    return [{"n": i, "chunk_id": i, "href": f"/read/1?at={i}#c{i}", "title": "Notes", "label": f"p. {i}"} for i in range(1, n + 1)]


def view(blocks, n=3, **kw):
    return web_helpers.answer_view("", sources(n), blocks=blocks, **kw)


def page(ready, blocks, n=3, data=None, kind="answer"):
    gid = ready.store.add_guide(ready.cid, kind, "G", "body", sources=sources(n),
                                data={"blocks": blocks, **(data or {})})
    return ready.client.get(f"/guides/{gid}")


def test_sources_show_only_cited_with_also_searched_p4(ready):
    r = page(ready, [{"kind": "text", "md": "Claim [2].", "cites": [2]}])
    sec = r.text.split('<div class="sources">')[1]
    main, also = sec.split("Also searched (not cited)")
    assert '[2]</span>' in main and '[1]</span>' not in main and '[3]</span>' not in main
    assert '[1]</span>' in also and '[3]</span>' in also and "<details" in r.text


def test_no_also_searched_when_all_cited(ready):
    r = page(ready, [{"kind": "text", "md": "A [1][2][3].", "cites": [1, 2, 3]}])
    assert "Also searched" not in r.text


def test_not_covered_note_p4(ready):
    r = page(ready, [{"kind": "outside", "md": "General knowledge only.", "cites": []}])
    assert "Your saved materials don&#39;t cover this. Parts below come from general knowledge." in r.text
    r2 = page(ready, [{"kind": "text", "md": "Fine [1].", "cites": [1]}])
    assert "cover this" not in r2.text
    r3 = page(ready, [{"kind": "text", "md": "Fine.", "cites": []}], data={"not_covered": True})
    assert "cover this." in r3.text and "general knowledge" not in r3.text  # no outside content: no extra sentence


def test_removed_citation_footnote_p3(ready):
    r = page(ready, [{"kind": "outside", "md": f"Exam favourite {grounding.REMOVED_MARK}.", "cites": []}])
    assert 'class="cite-removed"' in r.text and "Check: 1 citation removed" in r.text
    assert "pointed to a source that doesn't exist" not in r.text.replace("a citation here pointed", "")  # said once, at the top
    assert "<span hidden data-removed-citations>1</span>" in r.text
    assert "{{" not in r.text


def test_quote_flag_and_footnote_p4(ready):
    md = f'He says {grounding.QUOTE_OPEN}"a b c d e f g h"{grounding.QUOTE_CLOSE} [1].'
    r = page(ready, [{"kind": "text", "md": md, "cites": [1]}])
    assert "quote not found in the cited source" in r.text and "Check: 1 quote not found" in r.text
    assert "could not be found word for word" not in r.text and "<span hidden data-unverified-quotes>1</span>" in r.text


def test_outside_label_read_once_and_no_group_label_p5(ready):
    r = page(ready, [{"kind": "outside", "md": "Background.", "cites": []}])
    assert r.text.count("Not from your materials") == 1 and 'aria-label="Not from your materials"' not in r.text


def test_modes_banners(ready):
    t = page(ready, [{"kind": "text", "md": "x [1]", "cites": [1]}], data={"mode": "twin"}).text
    assert "similar problem" in t and "Worked twin problem.</strong>" not in t  # the routing note says it once
    r = page(ready, [{"kind": "text", "md": "x [1]", "cites": [1]}], data={"mode": "concepts"})
    assert "the ideas behind it" in r.text or "concepts" in r.text
    assert "Worked twin" not in r.text
    assert "Graded work" not in page(ready, [{"kind": "text", "md": "x [1]", "cites": [1]}], data={"mode": "answer"}).text


def test_matrix_in_guide_not_chips_p3(ready):
    r = page(ready, [{"kind": "text", "md": "Take A = [[4,1],[2,3]] and $[[1,1],[0,1]]$ [1].", "cites": [1]}])
    assert r.text.count('class="cite"') == 1 and "[[4,1],[2,3]]" in r.text


def test_answer_view_derives_cites_without_saved_cites():
    v = web_helpers.answer_view("Claim [2].\n\nMatrix [[1],[2]].", sources(3))
    assert [s["n"] for s in v["sources"]] == [2] and [s["n"] for s in v["unused_sources"]] == [1, 3]


# ---- concept map
MAP = {"nodes": [{"id": "a", "label": "A very long concept label about linear transformations and eigenvalues", "cites": [1]},
                 {"id": "b", "label": "B", "cites": []}, {"id": "c", "label": "C"}],
       "edges": [{"a": "a", "b": "b", "label": "implies", "cites": [2]}, {"a": "b", "b": "c", "label": "uncited"},
                 {"a": "a", "b": "c", "label": "cross link", "cites": [3]}, {"a": "c", "b": "a", "label": "back"}]}


def test_concept_graph_text_list_has_each_node_and_edge_once_p4():
    g = web_helpers.concept_graph(MAP)

    def walk(n):
        yield n
        for c in n["children"]:
            yield from walk(c)
    nodes = [n["id"] for r in g["forest"] for n in walk(r)]
    assert sorted(nodes) == ["a", "b", "c"]
    tree_edges = [(n["via"]) for r in g["forest"] for n in walk(r) if n["has_via"]]
    assert len(tree_edges) + len(g["cross"]) == 4
    assert {x["label"] for x in g["cross"]} == {"cross link", "back"}


def test_concept_graph_svg_full_labels_arrows_edge_labels_p4():
    g = web_helpers.concept_graph(MAP)
    full = " ".join(" ".join(b["lines"]) for b in g["boxes"])
    assert "A very long concept label about linear transformations and eigenvalues" in full and "..." not in full
    assert len(g["edges"]) == 4 and any(e["lines"] for e in g["edges"])
    assert all(b["h"] >= 33 for b in g["boxes"]) and max(b["h"] for b in g["boxes"]) > 40


def test_concept_map_page_p4(ready):
    gid = ready.store.add_guide(ready.cid, "concept_map", "Map", "", sources=sources(3), data=MAP)
    t = ready.client.get(f"/guides/{gid}").text
    assert 'role="img"' in t and "aria-hidden" not in t.split("<svg")[1].split(">")[0] and "text list above" in t
    assert 'marker-end="url(#arrow)"' in t and "<marker" in t and "implies" in t
    tree = t.split('<ul class="tree">')[1].split('<div class="mapwrap">')[0]
    assert tree.count("Not from your materials") == 2  # the uncited tree edge and the uncited back link
    assert 'aria-label="Source 2: Notes, p. 2"' in tree and 'aria-label="Source 3: Notes, p. 3"' in tree
    assert "Other links between concepts" in tree and tree.count("cross link") == 1
    assert 'aria-label="Source 1' in tree


def test_empty_concept_map(ready):
    gid = ready.store.add_guide(ready.cid, "concept_map", "Map", "", data={"nodes": [], "edges": []})
    assert "no concepts" in ready.client.get(f"/guides/{gid}").text


# ---- reader
def material(ready, chunks, kind="notes"):
    mid = ready.store.add_material(ready.cid, kind, "Lec")
    ready.store.replace_chunks(mid, chunks)
    return mid, [c["id"] for c in ready.store.chunks_of(mid)]


def test_reader_renders_code_math_and_h2_labels_p3(ready):
    mid, ids = material(ready, [
        {"text": "Intro with $\\lambda$ and C$_6$H$_{12}$O$_6$.\n\n```julia\nfunction f(x)\n    x + 1\nend\n```", "loc_kind": "page",
         "loc_start": "2", "loc_label": "p. 2"},
        {"text": "Spoken words", "loc_kind": "time", "loc_start": "95", "loc_end": "190", "loc_label": "1:35"}])
    t = ready.client.get(f"/read/{mid}?at={ids[1]}").text
    assert '<h2 class="loc">p. 2</h2>' in t and '<h2 class="loc">1:35</h2>' in t and "this part runs 1:35 to 3:10" in t
    assert '<code class="language-julia">function f(x)\n    x + 1\nend</code>' in t and 'class="math"' in t
    assert 'aria-current="location"' in t and "data-at-note" not in t
    assert "```" not in t


def test_reader_bad_at_is_a_note_not_json_p3(ready):
    mid, ids = material(ready, [{"text": "x", "loc_kind": "page", "loc_start": "1", "loc_label": "p. 1"}])
    r = ready.client.get(f"/read/{mid}?at=abc")
    assert r.status_code == 200 and "passage reference is not valid" in r.text and 'aria-current' not in r.text
    r = ready.client.get(f"/read/{mid}?at=999999")
    assert r.status_code == 200 and "no longer in this material" in r.text and 'class="chunk"' in r.text
    other, oids = material(ready, [{"text": "y", "loc_kind": "page", "loc_start": "1", "loc_label": "p. 1"}])
    assert "no longer in this material" in ready.client.get(f"/read/{mid}?at={oids[0]}").text


def test_reader_chunk_headings_sit_below_the_h2(ready):
    mid, _ = material(ready, [{"text": "# Heading in text\n\nBody", "loc_kind": "page", "loc_start": "1", "loc_label": "p. 1"}])
    t = ready.client.get(f"/read/{mid}").text
    assert "<h3>Heading in text</h3>" in t


def test_katex_assets_vendored_and_served(ready):
    for p in ("katex/katex.min.js", "katex/katex.min.css", "katex/auto-render.min.js", "katex/LICENSE",
              "content.js", "content.css"):
        assert ready.client.get(f"/static/{p}").status_code == 200, p
    assert ready.client.get("/static/katex/fonts/KaTeX_Main-Regular.woff2").status_code == 200
    js = ready.client.get("/static/content.js").text
    assert "htmx:afterSwap" in js and "htmlAndMathml" in js and "pre" in js
    assert "0.16.11" in ready.client.get("/static/katex/katex.min.js").text
    assert "style-src 'self' 'unsafe-inline'" in ready.client.get("/").headers["content-security-policy"]


def test_chunk_range_label():
    assert web_helpers.chunk_range_label({"loc_kind": "time", "loc_start": "95", "loc_end": "190", "loc_label": "1:35"}) == "1:35 to 3:10"
    assert web_helpers.chunk_range_label({"loc_kind": "time", "loc_start": "95", "loc_end": "", "loc_label": "1:35"}) == "1:35"
    assert web_helpers.chunk_range_label({"loc_kind": "page", "loc_start": "2", "loc_end": "", "loc_label": "p. 2"}) == "p. 2"
