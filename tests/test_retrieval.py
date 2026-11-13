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
    monkeypatch.setattr(retrieval, "fts_query", lambda q: '"unterminated')  # forces an FTS syntax error
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
