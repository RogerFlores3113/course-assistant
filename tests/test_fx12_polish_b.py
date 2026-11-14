"""fx12-polish part 2: rubric retrieval on the first call, unused-source numbers without gaps, trailing empty heading,
uncited exercise stems, continuation sentences, map wrapper, banner and check-line order. Original fixture text."""
import re

from course_assistant import grounding, importers, integrity, web_helpers
from course_assistant.providers import FakeProvider

from test_fx5_essay import build
from test_render import page, ready, sources  # noqa: F401


class Spy(FakeProvider):
    def __init__(self, text):
        super().__init__(lambda system, messages: text)
        self.seen = []

    def complete(self, system, messages, **kw):
        self.seen.append(messages[0]["content"])
        return super().complete(system, messages, **kw)


def _asked(question, reply="The rubric weighs the thesis most [1]."):
    s, c = build()
    p = Spy(reply)
    grounding.ask(s, c, question, p, route=False)
    return p.seen[0]


def test_rubric_question_gets_the_rubric_on_the_first_call():
    seen = _asked("What does the Essay 2 rubric reward?")
    assert "Thesis and argument" in seen and "Citations and format" in seen
    assert "Open with a clear, arguable thesis" in seen


def test_rubric_table_arrives_whole_not_cut_after_its_header():
    seen = _asked("What is the grading rubric for my essay, how many points is each part?")
    assert "| Counterargument | 10 |" in seen and "| Organization and clarity | 10 |" in seen


def test_my_essay_and_requirements_questions_pull_the_assignment():
    assert "Quote or closely paraphrase" in _asked("What are the requirements for my essay?")
    assert "Quote or closely paraphrase" in _asked("When is Essay 2 due and what does it need?")


def test_a_plain_topic_question_does_not_pull_the_assignment():
    seen = _asked("What does Federalist No. 10 say about faction?")
    assert "Grading rubric" not in seen and "Thesis and argument" not in seen


def test_a_problem_set_is_never_pulled_in_by_a_requirements_question():
    s, c = build()
    assert all("Problem" not in (x.get("heading") or "") for x in grounding.assignment_sources(s, c, "What are the requirements for problem set 2?"))


def test_unused_sources_are_numbered_without_gaps(ready):
    src = [{"n": n, "chunk_id": n, "href": f"/read/1?at={n}#c{n}", "title": f"Notes {n}", "label": f"p. {n}"} for n in (1, 3, 4, 7)]
    gid = ready.store.add_guide(ready.cid, "answer", "G", "b", sources=src,
                                data={"blocks": [{"kind": "text", "md": "A [1].", "cites": [1]}]})
    html = ready.client.get(f"/guides/{gid}").text
    also = html.split("Also searched")[1] if "Also searched" in html else ""
    nums = [int(x) for x in re.findall(r"\[(\d+)\]", also.split("</details>")[0])]
    assert nums == list(range(2, 2 + len(nums)))


def test_empty_trailing_heading_does_not_render(ready):
    html = page(ready, [{"kind": "text", "md": "Rank is the number of pivots [1].", "cites": [1]},
                        {"kind": "text", "md": "## Practice", "cites": []}]).text
    assert "Rank is the number" in html and "Practice</h" not in html


def test_a_heading_with_content_under_it_still_renders(ready):
    html = page(ready, [{"kind": "text", "md": "## Practice\n\n1. Count the pivots.", "cites": []}]).text
    assert "Practice</h" in html


def _outside(raw, n=3):
    return [b.md for b in grounding.parse_blocks(raw, n)[0] if b.kind == "outside"]


SOL = "<details>\n<summary>Show solution</summary>\n\n{}\n\n</details>"


def test_uncited_first_practice_question_is_not_boxed_when_the_others_are_not():
    raw = ("## Practice\n\n1. A diagonal matrix has entries 4 and 9. Eigenvalues?\n\n" + SOL.format("Read the diagonal [1].") +
           "\n\n2. Find the trace of that matrix.\n\n" + SOL.format("Add the diagonal [2]."))
    assert _outside(raw) == []


def test_a_claim_with_no_solution_after_it_is_still_boxed():
    assert _outside("Every matrix has exactly three eigenvalues in the real numbers, always, without exception.", 3)


def test_uncited_continuation_sentence_between_boxes_is_labelled():
    raw = ("Back substitution is the final stage of elimination [1]. A rule of thumb is to start from the bottom row. "
           "It also works for larger systems. Another idea is to check the answer in the original equations.")
    out = _outside(raw)
    assert any("works for larger systems" in x for x in out)


def test_a_real_comparison_of_two_cited_facts_stays_plain():
    raw = "Exam 1 covers four chapters [1]. That is one fewer chapter than Exam 2 [2]."
    assert _outside(raw, 2) == []


def test_coaching_that_restates_the_assignment_is_not_boxed():
    raw = "Essay 1 asks you to pick one of three ideas and defend it. The rubric rewards a clear position [1]."
    assert _outside(raw, 1) == []


def test_map_wrapper_is_focusable_and_labelled(ready):
    gid = ready.store.add_guide(ready.cid, "concept_map", "Map", "", sources=sources(2),
                                data={"nodes": [{"id": "a", "label": "A", "cites": [1]}, {"id": "b", "label": "B", "cites": [2]}],
                                      "edges": [{"a": "a", "b": "b", "label": "leads to", "cites": [1]}]})
    html = ready.client.get(f"/guides/{gid}").text
    assert re.search(r'<div class="mapwrap" tabindex="0" role="region" aria-label="[^"]+"', html)


def test_route_note_is_announced_before_the_check_line(ready):
    html = page(ready, [{"kind": "text", "md": f"Rank is the number of pivots [1] {grounding.REMOVED_MARK}.", "cites": [1]}],
                data={"mode": "concepts", "routing_note": "This looks like your graded work, so this covers the ideas behind it."}).text
    assert html.index("data-routing-note") < html.index("data-check-line")


def test_check_line_stays_first_without_a_route_note(ready):
    html = page(ready, [{"kind": "text", "md": f"Rank is the number of pivots [1] {grounding.REMOVED_MARK}.", "cites": [1]}]).text
    assert html.index("data-check-line") < html.index("Rank is the number")


def test_cited_answer_is_never_not_covered(ready):
    html = page(ready, [{"kind": "text", "md": "Rank counts pivots [1].", "cites": [1]},
                        {"kind": "outside", "md": "An unrelated long outside paragraph about something else entirely that is much longer.", "cites": []}],
                data={"not_covered": True}).text
    assert "data-not-covered" not in html


def test_a_rubric_header_cut_from_its_rows_is_joined_to_them():
    from course_assistant.store import Store
    s = Store(":memory:")
    c = s.add_course("Art History")
    m = s.add_material(c, "assignment", "Essay 1: The Gallery Essay")
    s.replace_chunks(m, [
        {"text": "Essay 1: The Gallery Essay\n\nWrite about one painting you saw in person.\n\nGrading rubric (50 points)\n\nCriterion Points",
         "heading": "Essay 1: The Gallery Essay", "loc_kind": "section", "loc_start": 1, "loc_label": "start"},
        {"text": "| Looking closely | 25 |\n| Argument | 25 |", "heading": "Rows", "loc_kind": "section", "loc_start": 2, "loc_label": "rows"}])
    s.add_graded_item(c, "Essay 1: The Gallery Essay", "", m)
    got = grounding.assignment_sources(s, c, "What does the rubric for my essay reward?")
    assert got and "Criterion Points" in got[0]["text"] and "| Looking closely | 25 |" in got[0]["text"]


def test_explain_puts_the_chunk_that_defines_the_topic_before_one_that_mentions_it():
    from course_assistant import study
    mention = {"text": "Markov chains use the eigenvalues of the transition matrix to find the long run.", "heading": "Markov chains"}
    define = {"text": "An eigenvalue is a number that scales a vector the matrix leaves in its own direction.", "heading": "Eigenvalues"}
    out = study._defining_first([mention, define], "eigenvalues")
    assert out[0] is define and out[1] is mention


def test_two_defining_or_two_mentioning_chunks_keep_their_order():
    from course_assistant import study
    a = {"text": "Powers of a matrix use eigenvalues.", "heading": "Powers"}
    b = {"text": "Stability depends on the eigenvalues.", "heading": "Stability"}
    assert study._defining_first([a, b], "eigenvalues") == [a, b]
