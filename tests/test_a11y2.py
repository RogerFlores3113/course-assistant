import re
from pathlib import Path

from test_web import KEY, env, mkcourse, ready  # noqa: F401

STATIC = Path(__file__).parent.parent / "src" / "course_assistant" / "static"



def test_n1_flashcards_filter_has_visible_button_and_no_inline_handler(ready):
    cid = mkcourse(ready.store)
    ready.store.add_flashcard(cid, "Q?", "A.")
    t = ready.client.get(f"/flashcards/{cid}").text
    assert "onchange" not in t and "<noscript>" not in t
    if "inline-filter" in t:
        assert re.search(r'<button class="btn small" type="submit">Filter</button>', t)


def test_n7_cram_link_has_no_aria_pressed_and_form_not_in_paragraph(ready):
    cid = mkcourse(ready.store)
    ready.store.add_flashcard(cid, "Q?", "A.")
    t = ready.client.get(f"/flashcards/{cid}").text
    assert "aria-pressed" not in t
    assert not re.search(r"<p[^>]*cram-bar", t)
    assert 'class="cram-bar"' in t


def test_flashcards_cram_finished_offers_another_round(ready):
    cid = mkcourse(ready.store)
    cid_card = ready.store.add_flashcard(cid, "Q?", "A.")
    t = ready.client.get(f"/flashcards/{cid}?cram=1&done={cid_card}").text
    assert "Cram round finished" in t and "Start another round" in t


def test_app_js_never_says_done_from_afterrequest():
    js = (STATIC / "app.js").read_text()
    after_request = js.split('"htmx:afterRequest"')[1].split("});")[0]
    assert "say(" not in after_request
    assert "Done." not in js
    assert "Answer ready." in js and "requestConfig" in js


def test_app_js_tab_switch_uses_recorded_element_and_focus_rule():
    js = (STATIC / "app.js").read_text()
    assert "clicked = elt" in js
    assert "clicked.hasAttribute" in js and "e.detail.elt.hasAttribute" not in js
    # tab switch is tested before the /ask$ path so the tab link itself never matches it
    assert js.index("if (tabSwitch)") < js.index("/\\/ask$/.test(path)")
    assert '" page"' in js


def test_app_js_focuses_flashcard_heading_after_grade():
    js = (STATIC / "app.js").read_text()
    assert 'deck.querySelector("h2")' in js


def test_study_lecture_dropdown_lists_newest_first(ready):
    cid = mkcourse(ready.store)
    for d, n in (("2020-01-06", "Early lecture"), ("2020-01-13", "Middle lecture"), ("2020-01-20", "Latest lecture")):
        ready.store.add_event(cid, "lecture", f"Lecture {d[-2:]}: {n}", d)
    t = ready.client.get(f"/courses/{cid}/study").text
    block = t[t.index("Choose a lecture"):]
    assert block.index("Latest lecture") < block.index("Middle lecture") < block.index("Early lecture")


def test_tab_pages_start_with_h2_under_h1(ready):
    cid = mkcourse(ready.store)
    for tab in ("ask", "study", "calendar"):
        t = ready.client.get(f"/courses/{cid}/{tab}").text
        assert t.count("<h1") == 1
        first = re.search(r"<h([1-6])", t[t.index('id="tab-panel"'):]).group(1)
        assert first == "2"
