"""fx13-polish (decision 10): exam topics named in every branch with whole-word stems, the handout note never beside a
handout row, practice numbering and note grammar, aria labels, defining passages first, cited numbers without gaps,
the zero-graded hint. Original fixture text only; each leak test has an honest look-alike next to it."""
import re

from course_assistant import grounding, integrity, retrieval, study, web_helpers
from course_assistant.providers import FakeProvider
from course_assistant.store import Store

from test_study import GOOD, Prov, future, world  # noqa: F401


# ------------------------------------------------------------------ exam guide: typed topics that matched nothing
def _plain_world():
    s = Store(":memory:")
    c = s.add_course("Science")
    m = s.add_material(c, "textbook", "Chapter 2 Measurement")
    s.replace_chunks(m, [
        {"text": "Every quantity has a unit, and a photon is a packet of light described in the optics section.",
         "heading": "Quantities and units", "loc_kind": "page", "loc_start": 2, "loc_label": "p. 2"},
        {"text": "Molarity and dilution relate moles of solute to the volume of a solution in plain terms.",
         "heading": "Molarity and dilution", "loc_kind": "page", "loc_start": 3, "loc_label": "p. 3"},
        {"text": "An empirical formula gives the simplest ratio of atoms in a compound.",
         "heading": "Empirical formulas", "loc_kind": "page", "loc_start": 4, "loc_label": "p. 4"},
        {"text": "Rates of change are measured over an interval of time in this short section.",
         "heading": "Rates", "loc_kind": "page", "loc_start": 5, "loc_label": "p. 5"}])
    return s, c, m


def _guide(s, c, topics, **kw):
    ex = s.add_event(c, "exam", "Exam 2", future(3), topics=topics)
    p = Prov(GOOD)
    g = s.guide(study.exam_guide(s, c, p, event_id=ex, **kw))
    return g, p.calls[0][1][0]["content"]


def test_quantum_never_matches_quantity_and_photosynthesis_never_matches_photon():
    s, c, _ = _plain_world()
    g, prompt = _guide(s, c, "Molarity and dilution, Quantum numbers, Photosynthesis")
    assert "Nothing in your saved materials matched: Quantum numbers, Photosynthesis." in g["body_md"]
    assert "Quantum numbers" in prompt.split("matched NOTHING")[1]  # the prompt says so too, so the model invents nothing


def test_a_plural_or_singular_of_the_same_word_still_matches():
    s, c, _ = _plain_world()
    g, _ = _guide(s, c, "Molarity and dilution, Empirical formula")
    assert "Nothing in your saved materials matched" not in g["body_md"]
    assert "Molarity and dilution" in g["body_md"]


def test_unmatched_topics_are_named_when_a_dated_material_sits_in_the_window():
    s, c, _ = _plain_world()
    lec = s.add_material(c, "slides", "Lecture 9 slides", lecture_date=future(-2))
    s.replace_chunks(lec, [{"text": "Molarity is moles of solute per liter of solution.", "heading": "Molarity",
                            "loc_kind": "slide", "loc_start": 1, "loc_label": "slide 1"}])
    g, prompt = _guide(s, c, "Molarity, Quantum numbers")
    assert "Lecture 9 slides" in prompt.split("Materials used:")[1].splitlines()[0]
    assert "Nothing in the materials used for this guide matched: Quantum numbers." in g["body_md"]


def test_a_topic_the_window_materials_cover_is_not_named():
    s, c, _ = _plain_world()
    lec = s.add_material(c, "slides", "Lecture 9 slides", lecture_date=future(-2))
    s.replace_chunks(lec, [{"text": "Molarity is moles of solute per liter of solution.", "heading": "Molarity",
                            "loc_kind": "slide", "loc_start": 1, "loc_label": "slide 1"}])
    g, _ = _guide(s, c, "Molarity")
    assert "matched" not in g["body_md"].split("##")[0]


def test_unmatched_topics_are_named_for_chosen_materials_too():
    s, c, m = _plain_world()
    g, _ = _guide(s, c, "Molarity and dilution, Quantum numbers", material_ids=[m])
    assert "Quantum numbers." in g["body_md"] and "Molarity and dilution." not in g["body_md"].split("Nothing")[-1]


def test_whole_word_stems():
    assert study._same_word("eigenvalues", "eigenvalue") and study._same_word("spaces", "space")
    assert study._same_word("empirical", "empirically")
    assert not study._same_word("quantum", "quantity") and not study._same_word("photosynthesis", "photon")
    assert not study._same_word("number", "numerical")


# ------------------------------------------------------------------ handout note never beside a handout row
def _handout_world(tmp_path):
    import fx11_world
    return fx11_world.handout_only_questions(tmp_path)


def test_handout_note_dropped_when_any_row_comes_from_a_graded_handout(tmp_path, monkeypatch):
    s, c, mid = _handout_world(tmp_path)
    q = "How do I use a volumetric flask?"
    row = {"chunk_id": 1, "material_id": mid, "material_title": "Lab handout", "material_kind": "handout", "loc_kind": "page",
           "loc_start": 1, "loc_label": "p. 1", "heading": "Procedure", "href": "/read/1", "score": 1.0,
           "text": "Fill the volumetric flask to the mark with water, then invert it three times to mix."}
    monkeypatch.setattr(integrity, "search_safe", lambda *a, **k: [])
    monkeypatch.setattr(integrity, "handout_sources", lambda *a, **k: ([row], 2))  # the handout's own row, and a note for the rest
    a = grounding.ask(s, c, q, FakeProvider(lambda system, messages: "Fill it to the mark [1]."))
    assert "Part of your graded handout was not searched" not in a.body_md


def test_handout_note_still_shows_when_the_rows_come_from_elsewhere(tmp_path, monkeypatch):
    s, c, mid = _handout_world(tmp_path)
    other = s.add_material(c, "textbook", "Chapter 1")
    row = {"chunk_id": 2, "material_id": other, "material_title": "Chapter 1", "material_kind": "textbook", "loc_kind": "page",
           "loc_start": 1, "loc_label": "p. 1", "heading": "Flasks", "href": "/read/2", "score": 1.0,
           "text": "A volumetric flask holds one exact volume when filled to its etched mark."}
    monkeypatch.setattr(integrity, "search_safe", lambda *a, **k: [])
    monkeypatch.setattr(integrity, "handout_sources", lambda *a, **k: ([row], 2))  # a row from a plain chapter
    a = grounding.ask(s, c, "How do I use a volumetric flask?", FakeProvider(lambda system, messages: "Fill it to the mark [1]."),
                      )
    assert "Part of your graded handout was not searched" in a.body_md


# ------------------------------------------------------------------ practice numbering and the drop note
NOTE_MANY = "2 parts of this guide were left out because they matched your graded Problem 1."
NOTE_ONE = "A part of this guide was left out because it matched your graded Problem 1."


def _practice(nums, keep, note, bold=False):
    def q(i, n):
        label = f"**{n}.**" if bold else f"{n}."
        return f"{label} Question number {i} about solutions?"
    det = "<details>\n<summary>Show solution</summary>\n\nAnswer.\n\n</details>"
    before = "\n\n".join(f"{q(i, n)}\n\n{det}" for i, n in enumerate(nums, 1))
    after = "\n\n".join(f"{q(i, n)}\n\n{det}" for i, n in enumerate(nums, 1) if i in keep)
    return study.tidy_practice(before, after, len(nums) - len(keep), note)


def test_bold_numbers_are_renumbered_after_a_drop():
    text, gone, note = _practice([1, 2, 3, 4], {1, 3, 4}, NOTE_ONE, bold=True)
    assert re.findall(r"^\*\*(\d+)\.\*\*", text, re.M) == ["1", "2", "3"] and gone == 1


def test_plain_numbers_are_renumbered_after_a_drop():
    text, gone, _ = _practice([1, 2, 3], {1, 3}, NOTE_ONE)
    assert re.findall(r"^(\d+)\. Question", text, re.M) == ["1", "2"] and gone == 1


def test_numbers_inside_a_kept_answer_and_decimals_are_left_alone():
    before = "1. First?\n\n<details>\n<summary>S</summary>\n\n2. step\n\n</details>\n\n2. Second?\n\n<details>\n<summary>S</summary>\n\nx\n\n</details>"
    after = "2. Second?\n\n<details>\n<summary>S</summary>\n\nx\n\n</details>"
    text, gone, _ = study.tidy_practice(before, after, 1, NOTE_ONE)
    assert text.startswith("1. Second?") and gone == 1
    para = "10.5 g of salt is weighed.\n\n"
    assert study._NUM_Q.match(para) is None


def test_the_drop_note_grammar_follows_the_count():
    _, _, one = _practice([1, 2, 3], {1, 3}, NOTE_MANY)
    assert one.startswith("1 practice question was left out because it matched")
    _, _, many = _practice([1, 2, 3, 4], {1}, NOTE_ONE)
    assert many.startswith("3 practice questions were left out because they matched")


# ------------------------------------------------------------------ aria labels on "Ask normally"
def test_ask_normally_buttons_name_the_question():
    from jinja2 import Environment, FileSystemLoader
    import course_assistant
    env = Environment(loader=FileSystemLoader(str(__import__("pathlib").Path(course_assistant.__file__).parent / "templates")), autoescape=True)
    tpl = env.get_template("_answer.html").module
    html = []
    for q in ("How do I find the NaCl mass?", "Walk me through the LU problem"):
        v = web_helpers.answer_view("Similar problem [1].", [{"n": 1, "chunk_id": 1, "href": "/r", "title": "Notes", "label": "p. 1"}],
                                    mode="twin", data={"routing_note": "Graded work."}, course_id=1, question=q)
        html.append(str(tpl.answer(v)))
    labels = [re.search(r'aria-label="([^"]+)"', h).group(1) for h in html]
    assert len(set(labels)) == 2 and all(x.startswith("Ask normally") for x in labels)
    assert "NaCl" in labels[0] and "LU problem" in labels[1]


# ------------------------------------------------------------------ cited numbers run without gaps
def _srcs(n):
    return [{"n": i, "chunk_id": i, "href": f"/read/1?at={i}", "title": f"Doc {i}", "label": f"p. {i}"} for i in range(1, n + 1)]


def test_cited_numbers_are_renumbered_without_gaps_and_the_text_follows():
    v = web_helpers.answer_view("First [4].\n\nSecond [9].\n\nBoth [4][9].", _srcs(10))
    assert [x["ns"][0]["n"] for x in v["sources"]] == [1, 2]
    html = "".join(str(b["html"]) for b in v["blocks"])
    assert ">[1]</a>" in html and ">[2]</a>" in html and "[4]" not in html and "[9]" not in html
    assert [x["title"] for x in v["sources"]] == ["Doc 4", "Doc 9"]
    assert [x["ns"][0]["n"] for x in v["unused_sources"]][:2] == [3, 4]  # "Also searched" carries on after the cited list


def test_numbers_already_without_gaps_are_untouched():
    v = web_helpers.answer_view("A [1].\n\nB [2].", _srcs(3))
    assert [x["ns"][0]["n"] for x in v["sources"]] == [1, 2]


def test_a_matrix_in_the_text_is_not_a_citation_when_renumbering():
    v = web_helpers.answer_view("Take A = [[4,1],[2,3]] and use it [5].", _srcs(6))
    html = "".join(str(b["html"]) for b in v["blocks"])
    assert "[[4,1],[2,3]]" in html and ">[1]</a>" in html


# ------------------------------------------------------------------ defining passages first
DEF = {"heading": "Eigenvalues", "text": "A number x is an eigenvalue of a matrix A when Av = xv has a nonzero solution v. "
                                          "That is the whole idea."}
USE = {"heading": "Markov chains", "text": "The steady state of a Markov chain is found by looking at the eigenvalues of its "
                                            "transition matrix and picking the one equal to 1."}


def test_a_passage_whose_first_lines_define_the_term_outscores_one_that_only_uses_it():
    d = retrieval._definition_score(["eigenvalue"], DEF)
    u = retrieval._definition_score(["eigenvalue"], USE)
    assert d >= 1.5 and u < 1.5


def test_a_short_heading_over_a_defining_sentence_defines_and_a_long_one_does_not():
    short = {"heading": "Rank", "text": "The rank of a matrix is the number of independent columns it has."}
    longh = {"heading": "How the rank of a large matrix is used in many applied settings", "text": "Rank often shows up in fits."}
    assert retrieval._definition_score(["rank"], short) >= 1.5
    assert retrieval._definition_score(["rank"], longh) < 1.5


def test_explain_puts_the_defining_passage_first_even_when_search_ranked_it_low(world, monkeypatch):
    s, cid, _ = world
    rows = []
    for i, d in enumerate([USE, USE, USE, USE, DEF], 1):
        rows.append({"chunk_id": i, "material_id": 1, "material_title": "Notes", "material_kind": "notes", "loc_kind": "page",
                     "loc_start": i, "loc_label": f"p. {i}", "heading": d["heading"], "text": d["text"] + f" ({i})",
                     "href": "/r", "score": 1.0})

    class R:
        def __init__(self):
            self.limits = []
        def search(self, store, course_id, query, *, limit=8, **kw):
            self.limits.append(limit)
            return rows[:limit]
        _definition_score = staticmethod(retrieval._definition_score)
        normalize_query = staticmethod(retrieval.normalize_query)
        _GENERIC_TOPIC = retrieval._GENERIC_TOPIC
    r = R()
    monkeypatch.setattr(study, "_mod", lambda name: r if name == "retrieval" else __import__(f"course_assistant.{name}", fromlist=["x"]))
    out = study._defining_first(study._search(s, cid, "eigenvalues", 24), "eigenvalues")[:10]
    assert out[0]["heading"] == "Eigenvalues" and max(r.limits) >= 24


# ------------------------------------------------------------------ zero graded items hint
from test_render import ready  # noqa: E402,F401


def test_zero_graded_items_says_how_they_are_marked(ready):
    m = ready.store.add_material(ready.cid, "assignment", "Problem Set 1")
    ready.store.replace_chunks(m, [{"text": "Problem 1. Find the sum of two numbers of your choice.", "loc_kind": "page",
                                    "loc_start": 1, "loc_label": "p. 1"}])
    t = ready.client.get(f"/courses/{ready.cid}?tab=materials").text
    assert "0 graded items" in t and "Mark as graded" in t.split('id="materials-summary"')[1].split("</p>")[0]


def test_the_hint_is_gone_once_something_is_graded(ready):
    ready.store.add_graded_item(ready.cid, "Problem Set 1: Problem 1", "Find the sum of two numbers of your choice.", None,
                                kind="problem", label="Problem 1")
    t = ready.client.get(f"/courses/{ready.cid}?tab=materials").text
    summary = t.split('id="materials-summary"')[1].split("</p>")[0]
    assert "1 graded item" in summary and "Mark as graded" not in summary
