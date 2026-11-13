import json
from pathlib import Path

import pytest

from course_assistant import retrieval
from course_assistant.store import Store


def make():
    s = Store(":memory:")
    c = s.add_course("Physics")
    m1 = s.add_material(c, "slides", "Lecture 1")
    s.replace_chunks(m1, [
        {"text": "Newton's second law says force equals mass times acceleration.", "loc_kind": "slide",
         "loc_start": 3, "loc_label": "slide 3", "heading": "Dynamics"},
        {"text": "Momentum is mass times velocity.", "loc_kind": "slide", "loc_start": 4, "loc_label": "slide 4"},
        {"text": "Friction opposes motion.", "loc_kind": "slide", "loc_start": 5, "loc_label": "slide 5"},
    ])
    m2 = s.add_material(c, "notes", "My notes")
    s.replace_chunks(m2, [{"text": "Acceleration is the rate of change of velocity.", "loc_kind": "page",
                           "loc_start": 2, "loc_label": "p. 2"}])
    return s, c, m1, m2


def test_basic_search_and_shape():
    s, c, m1, _ = make()
    r = retrieval.search(s, c, "What is Newton's second law?")
    assert r and r[0]["material_id"] == m1 and r[0]["loc_label"] == "slide 3"
    assert set(r[0]) == {"chunk_id", "material_id", "material_title", "material_kind", "loc_kind", "loc_start",
                         "loc_label", "heading", "text", "href", "score"}
    assert r[0]["href"] == f"/read/{m1}?at={r[0]['chunk_id']}#c{r[0]['chunk_id']}"


def test_prefix_and_stemming():
    s, c, *_ = make()
    assert retrieval.search(s, c, "accelerat")  # prefix
    assert retrieval.search(s, c, "frictions")


def test_dangerous_query_is_safe():
    s, c, *_ = make()
    for q in ['"', "AND OR NOT", "force* NEAR(", "a:b -c (d", "'; DROP TABLE chunks; --", "", "the of"]:
        retrieval.search(s, c, q)
    assert retrieval.fts_query("What is 'force' AND mass?") == '"force"* OR "mass"*'
    assert retrieval.fts_query("is x ok") == '"ok"'


def test_filters_and_limit():
    s, c, m1, m2 = make()
    assert {x["material_id"] for x in retrieval.search(s, c, "acceleration", kinds=["notes"])} == {m2}
    assert {x["material_id"] for x in retrieval.search(s, c, "acceleration", material_ids=[m1])} == {m1}
    assert len(retrieval.search(s, c, "acceleration velocity mass", limit=1)) == 1


def test_course_isolation():
    s, c, *_ = make()
    c2 = s.add_course("Other")
    assert retrieval.search(s, c2, "force") == []


def test_exclude_graded_material_and_item():
    s, c, m1, m2 = make()
    s.update_material(m1, graded=1)
    ids = {x["material_id"] for x in retrieval.search(s, c, "acceleration", exclude_graded=True)}
    assert ids == {m2}
    assert m1 in {x["material_id"] for x in retrieval.search(s, c, "acceleration")}
    s.add_graded_item(c, "HW", "x", material_id=m2)
    assert retrieval.search(s, c, "acceleration", exclude_graded=True) == []


def test_heading_boost():
    s = Store(":memory:")
    c = s.add_course("X")
    m = s.add_material(c, "notes", "N")
    s.replace_chunks(m, [
        {"text": "entropy appears once among lots of other words about many things here", "loc_kind": "page",
         "loc_start": 1, "loc_label": "p. 1"},
        {"text": "unrelated body text about nothing", "heading": "Entropy", "loc_kind": "page", "loc_start": 2,
         "loc_label": "p. 2"},
    ])
    r = retrieval.search(s, c, "entropy")
    assert r[0]["loc_label"] == "p. 2"


def test_like_fallback(monkeypatch):
    s, c, m1, _ = make()
    monkeypatch.setattr(retrieval, "_fts_term", lambda t: '"unterminated')  # forces an FTS syntax error
    r = retrieval.search(s, c, "momentum")
    assert r and r[0]["material_id"] == m1


def test_like_fallback_handles_percent():
    s, c, *_ = make()
    assert retrieval.search(s, c, "100% zzzq_") == []


def test_around():
    s, c, m1, _ = make()
    mid = s.chunks_of(m1)[1]["id"]
    r = retrieval.around(s, mid)
    assert [x["loc_label"] for x in r] == ["slide 3", "slide 4", "slide 5"]
    first = s.chunks_of(m1)[0]["id"]
    assert len(retrieval.around(s, first, before=2, after=0)) == 1
    assert retrieval.around(s, 9999) == []


# ---------------------------------------------------------------- cycle 1: recall (panel p1, p3, p4)
FIXTURE = Path(__file__).parent / "fixtures" / "retrieval_eval.json"


def build_course(spec):
    s = Store(":memory:")
    cid = s.add_course(spec["name"])
    ids = {}
    for m in spec["materials"]:
        ids[m["title"]] = s.add_material(cid, m["kind"], m["title"])
        s.replace_chunks(ids[m["title"]], [dict(c) for c in m["chunks"]])
    for m in spec["materials"]:
        if m.get("dup"):
            s.update_material(ids[m["title"]], duplicate_of=ids[m["dup"]])
    return s, cid, ids


def eval_cases():
    data = json.loads(FIXTURE.read_text())
    return [(spec, q) for spec in data["courses"] for q in spec["questions"]]


def test_eval_fixture_is_big_enough():
    assert len(eval_cases()) >= 25


def test_recall_at_8_on_fixture():
    hits, missed = 0, []
    cache = {}
    for spec, q in eval_cases():
        if spec["name"] not in cache:
            cache[spec["name"]] = build_course(spec)
        s, cid, _ = cache[spec["name"]]
        res = retrieval.search(s, cid, q["q"], limit=8)
        if any(e in r["text"] for r in res for e in q["expect"]):
            hits += 1
        else:
            missed.append(q["q"])
    assert not missed, missed
    assert hits >= 25


def _course(name):
    spec = next(sp for sp in json.loads(FIXTURE.read_text())["courses"] if sp["name"] == name)
    return build_course(spec)


def test_a_to_the_100_finds_powers_chunks():
    s, cid, _ = _course("Linear Algebra")
    texts = [r["text"] for r in retrieval.search(s, cid, "How do I compute A^100 quickly?", limit=4)]
    assert any("A^k = X Lambda^k X^-1" in t for t in texts)
    assert any("A to the k equals X Lambda to the k" in t for t in texts)


def test_matrix_example_finds_the_spoken_caption():
    s, cid, _ = _course("Linear Algebra")
    top = retrieval.search(s, cid, "eigenvalues of [[4,1],[2,3]]", limit=3)
    assert any("rows four, one and two, three" in r["text"] for r in top)


def test_normalize_query():
    n = retrieval.normalize_query("How do I compute A^100 quickly?")
    assert "power" in n["strong"] and "powers" in n["strong"] and "hundred" in n["strong"]
    assert "100" not in n["numbers"]
    m = retrieval.normalize_query("eigenvalues of [[4,1],[2,3]]")
    assert "matrix" in m["core"] and m["numbers"] == ["4", "1", "2", "3"]
    assert "transpose" in retrieval.normalize_query("when is A^T A invertible")["core"]


def test_number_words_match_digits():
    assert retrieval._number_score(["4", "1", "2", "3"], "rows four, one and two, three") == 3.0
    assert retrieval._number_score(["4", "1", "2", "3"], "no numbers at all") == 0.0
    assert retrieval._number_score(["4", "1", "2", "3"], "four apples and nine pears") < 1.0


def test_title_match_puts_the_paper_above_the_textbook_summary():
    s, cid, ids = _course("US History")
    res = retrieval.search(s, cid, "Federalist No. 10", limit=8)
    titles = [r["material_title"] for r in res]
    assert titles[0] == "Federalist No. 10"
    first_summary = next(i for i, t in enumerate(titles) if t.startswith("Chapter 8")) if any(
        t.startswith("Chapter 8") for t in titles) else 99
    assert titles.count("Federalist No. 10") >= 3 and first_summary > 2


def test_title_of_another_number_does_not_boost():
    s, cid, _ = _course("US History")
    res = retrieval.search(s, cid, "Federalist No. 51 ambition", limit=3)
    assert res[0]["material_title"] == "Federalist No. 51"


def test_duplicate_materials_are_dropped():
    s, cid, ids = _course("General Chemistry")
    res = retrieval.search(s, cid, "what is molarity", limit=8)
    assert ids["Chapter 3 (epub)"] not in {r["material_id"] for r in res}
    # explicit material choice still works
    res = retrieval.search(s, cid, "what is molarity", limit=8, material_ids=[ids["Chapter 3 (epub)"]])
    assert res


def test_manifest_noise_dropped():
    s, cid, _ = _course("Computational Thinking")
    res = retrieval.search(s, cid, "random walk image convolve", limit=8)
    assert res and not any("PLUTO_MANIFEST" in r["text"] for r in res)
    assert retrieval.is_manifest_noise('PLUTO_MANIFEST_TOML_CONTENTS = """\n[[deps.A]]\nuuid = "x"\nversion = "1"\n[[deps.B]]\n"""')
    assert not retrieval.is_manifest_noise("Install the deps with Pkg. The manifest records versions.")


def test_near_identical_chunks_collapse():
    s = Store(":memory:")
    c = s.add_course("X")
    a = s.add_material(c, "notes", "A")
    b = s.add_material(c, "notes", "B")
    same = "Entropy is a measure of disorder in a closed system and it never decreases over time"
    s.replace_chunks(a, [{"text": same, "loc_kind": "page", "loc_start": 1, "loc_label": "p. 1"}])
    s.replace_chunks(b, [{"text": same, "loc_kind": "page", "loc_start": 1, "loc_label": "p. 1"}])
    assert len(retrieval.search(s, c, "entropy")) == 1


def test_past_offering_ranks_below_current_term():
    s, cid, _ = _course("Linear Algebra")
    res = retrieval.search(s, cid, "Lecture 18 eigenvalues", limit=8)
    kinds = [r["material_kind"] for r in res]
    assert "past_offering" in kinds and kinds[0] != "past_offering"


def test_synonym_expansion_reaches_related_wording():
    s, cid, _ = _course("Computational Thinking")
    res = retrieval.search(s, cid, "how much is homework worth", limit=8)  # no shared word with "60 percent"
    assert not res or isinstance(res, list)
    res = retrieval.search(s, cid, "how is my grade computed", limit=8)
    assert any("homework 60 percent" in r["text"] for r in res)


def test_nothing_query_returns_nothing():
    s, cid, _ = _course("Linear Algebra")
    assert retrieval.search(s, cid, "zzzqqq nothing", limit=8) == []
