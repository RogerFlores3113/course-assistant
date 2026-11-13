"""fx10-polish: one link per Sources row, no repeated edge labels, no empty solution stub, the two-try feedback
fallback says so, and a layout opener is a signpost. Fixtures are original text."""
import re

from course_assistant import essay, grounding, study, web_helpers, web_md
from test_render import page, ready, sources  # noqa: F401  (the ready fixture)


def _rows(html: str) -> list[str]:
    main = html.split('<div class="sources">')[1].split("Also searched")[0]
    return re.findall(r"<li>.*?</li>", main, re.S)


def test_sources_rows_have_one_link_each(ready):
    html = page(ready, [{"kind": "text", "md": "Alpha [1]. Beta [2].", "cites": [1, 2]}]).text
    rows = _rows(html)
    assert len(rows) == 2 and all(r.count("<a ") == 1 for r in rows)
    assert "[1]" in rows[0] and "Notes, p. 1" in rows[0]


def test_merged_row_shows_both_numbers_with_one_link(ready):
    src = [{"n": 1, "chunk_id": 1, "href": "/read/1?at=1#c1", "title": "Notes", "label": "p. 1"},
           {"n": 2, "chunk_id": 2, "href": "/read/1?at=1#c1", "title": "Notes", "label": "p. 1"}]
    gid = ready.store.add_guide(ready.cid, "answer", "G", "b", sources=src,
                                data={"blocks": [{"kind": "text", "md": "A [1]. B [2].", "cites": [1, 2]}]})
    rows = _rows(ready.client.get(f"/guides/{gid}").text)
    assert len(rows) == 1 and rows[0].count("<a ") == 1 and "[1]" in rows[0] and "[2]" in rows[0]


def test_citation_chips_in_the_text_still_link(ready):
    html = page(ready, [{"kind": "text", "md": "Alpha [1].", "cites": [1]}]).text
    assert html.split('<div class="sources">')[0].count('class="cite"') == 1


MAP = {"nodes": [{"id": "a", "label": "Seed", "cites": [1]}, {"id": "b", "label": "Sprout", "cites": [2]},
                 {"id": "c", "label": "Tree", "cites": [3]}],
       "edges": [{"a": "a", "b": "b", "label": "germinates into", "cites": [1]},
                 {"a": "b", "b": "c", "label": "matures into", "cites": [2]}]}


def test_concept_map_says_each_edge_label_once(ready):
    gid = ready.store.add_guide(ready.cid, "concept_map", "Map", "", sources=sources(3), data=MAP)
    html = ready.client.get(f"/guides/{gid}").text
    assert html.count("germinates into") == 1 and html.count("matures into") == 1
    svg = html.split("<svg")[1].split("</svg>")[0]
    assert "germinates" not in svg and "<path" in svg and "Sprout" in svg


def test_concept_map_cross_links_keep_their_label(ready):
    data = dict(MAP, edges=MAP["edges"] + [{"a": "a", "b": "c", "label": "also becomes", "cites": [3]}])
    gid = ready.store.add_guide(ready.cid, "concept_map", "Map", "", sources=sources(3), data=data)
    assert ready.client.get(f"/guides/{gid}").text.count("also becomes") == 1


STUB = "1. Name one cause of rust.\n\n<details>\n<summary>Show solution</summary>\n\n</details>\n\n2. Why does salt speed it up?\n\n<details>\n<summary>Show solution</summary>\n\nSalt makes the water carry current better [1].\n\n</details>"


def test_empty_solution_stub_is_not_rendered():
    html = str(web_md.render(STUB))
    assert html.count("<details>") == 1 and "Salt makes the water" in html
    assert html.count("Show solution") == 1


def test_one_line_empty_stub_is_not_rendered():
    html = str(web_md.render("1. Name one cause.\n\n<details><summary>Show solution</summary></details>"))
    assert "Show solution" not in html and "Name one cause" in html


def test_filled_solution_is_kept():
    html = str(web_md.render("1. Q?\n\n<details>\n<summary>Show solution</summary>\n\nA short answer.\n\n</details>"))
    assert "<details>" in html and "A short answer." in html


def test_drop_empty_details_text():
    out = study.drop_empty_details(STUB)
    assert out.count("<details>") == 1 and "Name one cause of rust." in out
    plain = "1. Q?\n\n<details>\n<summary>Show solution</summary>\n\nYes.\n\n</details>"
    assert study.drop_empty_details(plain) == plain
    assert study.drop_empty_details("no blocks here") == "no blocks here"


def test_failed_feedback_page_says_it_tried_twice():
    body = essay.FEEDBACK_FAILED + " Here are questions to ask of your draft.\n\n- Which sentence is your claim?"
    v = web_helpers.answer_view(body, [], blocks=[{"kind": "text", "md": body}], data={"feedback": True})
    html = " ".join(b["html"] for b in v["blocks"])
    assert "I tried twice and could not write safe feedback" in html


def test_feedback_page_that_worked_does_not_claim_two_tries():
    body = "Your claim is clear, and paragraph two needs a source."
    v = web_helpers.answer_view(body, [], blocks=[{"kind": "text", "md": body}], data={"feedback": True})
    assert "tried twice" not in " ".join(b["html"] for b in v["blocks"])


def _kinds(blocks):
    v = web_helpers.answer_view("", sources(3), blocks=blocks)
    return [b["kind"] for b in v["blocks"]]


def test_layout_opener_is_not_boxed():
    for opener in ("Federalist No. 10 moves in a clear order.", "Chapter 3 is organised in a clear sequence."):
        blocks = [{"kind": "outside", "md": opener, "cites": []},
                  {"kind": "text", "md": "First, it names the problem [1].", "cites": [1]}]
        assert "outside" not in _kinds(blocks), opener


def test_signpost_helper():
    assert grounding.signpost_opener("Federalist No. 10 moves in a clear order.")
    assert not grounding.signpost_opener("Federalist No. 10 was written in 1787 by Madison.")
    assert not grounding.signpost_opener("The reading moves in a clear order of 12 steps.")


def test_uncited_fact_next_to_cited_steps_is_still_boxed():
    blocks = [{"kind": "outside", "md": "The author wrote this essay while living in Virginia in 1787.", "cites": []},
              {"kind": "text", "md": "First, it names the problem [1].", "cites": [1]}]
    assert "outside" in _kinds(blocks)
