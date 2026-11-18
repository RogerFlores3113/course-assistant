"""fx12-polish: each typed exam topic keeps its own best materials. Original text only."""
from course_assistant import study
from test_study import GOOD, Prov, future, world  # noqa: F401


def _chapter(s, cid, title, headings):
    m = s.add_material(cid, "textbook", title)
    s.replace_chunks(m, [{"text": f"{h} is explained here in a few plain sentences about {h}.", "heading": h,
                          "loc_kind": "page", "loc_start": i + 2, "loc_label": f"p. {i + 2}"} for i, h in enumerate(headings)])
    return m


def _guide(world, topics):
    s, cid, _ = world
    ch3 = _chapter(s, cid, "Chapter 3 Stoichiometry", ["Molarity and dilution", "Empirical and molecular formulas", "Molarity units"])
    ch4 = _chapter(s, cid, "Chapter 4 Reactions", ["Percent yield", "Limiting reagents"])
    ex = s.add_event(cid, "exam", "Exam 3", future(3), topics=topics)
    p = Prov(GOOD)
    g = s.guide(study.exam_guide(s, cid, p, event_id=ex))
    return g, p.calls[0][1][0]["content"], (ch3, ch4)


def test_each_typed_topic_keeps_its_own_material(world):
    g, prompt, _ = _guide(world, "Molarity and dilution, Empirical and molecular formulas, Percent yield")
    assert "Chapter 4 Reactions" in prompt and "Chapter 3 Stoichiometry" in prompt
    assert "Nothing in your saved materials matched" not in g["body_md"]


def test_a_topic_with_no_material_is_named_plainly(world):
    g, prompt, _ = _guide(world, "Molarity and dilution, Buffer capacity")
    assert "Nothing in your saved materials matched: Buffer capacity." in g["body_md"]
    assert "Chapter 3 Stoichiometry" in prompt and "Chapter 4 Reactions" not in prompt.split("Materials used:")[1].splitlines()[0]


def test_one_typed_topic_does_not_pull_in_unrelated_chapters(world):
    g, prompt, _ = _guide(world, "Percent yield")
    used = prompt.split("Materials used:")[1].splitlines()[0]
    assert "Chapter 4 Reactions" in used and "Chapter 3" not in used
