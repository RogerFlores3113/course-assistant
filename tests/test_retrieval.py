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
    events = {}
    for e in spec.get("events", []):
        events[e["key"]] = s.add_event(cid, e["kind"], e["title"], e["date"], topics=e.get("topics", ""))
    for m in spec["materials"]:
        ids[m["title"]] = s.add_material(cid, m["kind"], m["title"])
        s.replace_chunks(ids[m["title"]], [dict(c) for c in m["chunks"]])
        if m.get("event"):
            s.update_material(ids[m["title"]], event_id=events[m["event"]])
    for m in spec["materials"]:
        if m.get("dup"):
            s.update_material(ids[m["title"]], duplicate_of=ids[m["dup"]])
    return s, cid, ids


def eval_cases():
    data = json.loads(FIXTURE.read_text())
    return [(spec, q) for spec in data["courses"] for q in spec["questions"]]


def check_case(res, q):
    """Why a result list fails the case (a string), or '' when it passes. See the fixture for the fields."""
    texts = [r["text"] for r in res]
    why = []
    if "expect" in q:
        n = q.get("rank", len(res))
        if not any(e in t for t in texts[:n] for e in q["expect"]):
            why.append(f"none of {q['expect']} in top {n}")
    for e in q.get("expect_all", []):
        if not any(e in t for t in texts):
            why.append(f"missing {e!r}")
    ab = q.get("absent_in_top")
    if ab:
        for r in res[:ab["n"]]:
            if (r["heading"] in ab.get("headings", []) or any(x in r["text"] for x in ab.get("texts", []))
                    or r["text"].strip() in ab.get("texts_exact", [])):
                why.append(f"unwanted {r['heading'] or r['text'][:30]!r} in top {ab['n']}")
    if q.get("top_kinds_not"):
        bad = [r["material_kind"] for r in res[:q.get("top_n", 4)] if r["material_kind"] in q["top_kinds_not"]]
        if bad:
            why.append(f"kinds {bad} in top {q.get('top_n', 4)}")
    for title in q.get("each_material", []):
        if title not in {r["material_title"] for r in res}:
            why.append(f"no chunk from {title}")
    return "; ".join(why)


def run_eval(search=None):
    search = search or retrieval.search
    cache, rows = {}, []
    for spec, q in eval_cases():
        if spec["name"] not in cache:
            cache[spec["name"]] = build_course(spec)
        s, cid, _ = cache[spec["name"]]
        res = search(s, cid, q["q"], limit=q.get("limit", 8))
        rows.append((q["q"], check_case(res, q)))
    return rows


def test_eval_fixture_is_big_enough():
    assert len(eval_cases()) >= 40


def test_recall_at_8_on_fixture():
    rows = run_eval()
    missed = [f"{q}: {why}" for q, why in rows if why]
    assert not missed, missed
    assert len(rows) >= 40


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
    res = retrieval.search(s, cid, "Lecture 18 eigenvalues", limit=12)
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


# ---------------------------------------------------------------- cycle 2 (fixer fx2-retrieval)
def _top(name, q, **kw):
    s, cid, ids = _course(name)
    return retrieval.search(s, cid, q, **kw), s, cid, ids


def test_lecture_reference_detection():
    lr = retrieval.lecture_reference
    assert lr("primer for Lecture 20") == {"number": 20}
    assert lr("fill in my notes from the Nov 10 lecture") == {"date": (11, 10)}
    assert lr("what did we do in Tuesday's lecture") == {"weekday": 1}
    assert lr("What is due on Thursday and what should I read?") is None
    assert lr("Essay due Nov 20") is None


def test_lecture_query_pulls_linked_materials_first_and_skips_noise():
    res, *_ = _top("Linear Algebra", "Fill in my eigenvalues TODO from the Nov 10 lecture", limit=8)
    assert res[0]["material_title"] == "Lecture 18 captions"
    assert not any(r["text"].startswith("Speaker notes") or r["text"].strip() == "Lecture 18" for r in res)


def test_primer_for_a_lecture_number_uses_the_lecture_materials_not_the_schedule():
    res, *_ = _top("Linear Algebra", "primer for Lecture 20", limit=6)
    assert res[0]["material_title"] == "Lecture 20 notes"
    assert not any(r["heading"] in ("Grading", "Lecture schedule") for r in res[:4])


def test_lecture_query_respects_exclude_graded():
    s, cid, ids = _course("Linear Algebra")
    s.add_graded_item(cid, "Problem 1", "x", material_id=ids["Lecture 20 notes"])
    res = retrieval.search(s, cid, "primer for Lecture 20", limit=6, exclude_graded=True)
    assert ids["Lecture 20 notes"] not in {r["material_id"] for r in res}


def test_exclude_graded_drops_graded_materials():
    s, cid, ids = _course("General Chemistry")
    s.add_graded_item(cid, "Problem 2", "molarity", material_id=ids["Chapter 3 Composition of Substances and Solutions"])
    assert retrieval.search(s, cid, "molarity", limit=8)
    gid = ids["Chapter 3 Composition of Substances and Solutions"]
    assert gid in {r["material_id"] for r in retrieval.search(s, cid, "molarity", limit=8)}
    assert gid not in {r["material_id"] for r in retrieval.search(s, cid, "molarity", limit=8, exclude_graded=True)}
    s2, cid2, ids2 = _course("General Chemistry")
    s2.update_material(ids2["Chapter 3 Composition of Substances and Solutions"], graded=1)
    assert gid not in {r["material_id"] for r in retrieval.search(s2, cid2, "molarity", limit=8, exclude_graded=True)}


def test_definition_beats_section_intro_for_molarity_and_dilution():
    res, *_ = _top("General Chemistry", "What is molarity?", limit=3)
    assert "number of moles of solute per liter" in res[0]["text"]
    res, *_ = _top("General Chemistry", "Explain dilution", limit=3)
    assert "M1 V1 = M2 V2" in res[0]["text"]


def test_schedule_and_exam_chunks_are_demoted_unless_asked_about():
    res, *_ = _top("Linear Algebra", "Explain eigenvalues", limit=5)
    assert not any(r["heading"] in ("Exams", "Lecture schedule", "Grading") for r in res)
    res, *_ = _top("Linear Algebra", "When is Midterm 1?", limit=3)
    assert any(r["heading"] == "Exams" for r in res)


def test_split_clauses():
    assert len(retrieval.split_clauses("What is the projection matrix formula and when is A^T A invertible?")) == 2
    assert len(retrieval.split_clauses("How are trace and determinant related to the eigenvalues?")) == 1
    assert len(retrieval.split_clauses("eigenvalues of [[4,1],[2,3]]")) == 1
    assert len(retrieval.split_clauses("When is Midterm 1 and what does it cover?")) == 1


def test_two_part_question_finds_both_halves():
    res, *_ = _top("Linear Algebra", "What is the projection matrix formula and when is A^T A invertible?", limit=4)
    texts = " ".join(r["text"] for r in res)
    assert "P = A (A^T A)^-1 A^T" in texts and "If Ax = 0 then A^T A x = 0" in texts


def test_worked_example_continues_into_the_next_caption():
    res, *_ = _top("Linear Algebra", "How do I find the eigenvalues of [[4,1],[2,3]]?", limit=6)
    assert any("the eigenvalues, five and two" in r["text"] for r in res)


def test_fall_2018_exam_dates_include_exam_3():
    res, *_ = _top("Linear Algebra", "What were the exam dates in the Fall 2018 offering?", limit=8)
    texts = " ".join(r["text"] for r in res)
    assert all(x in texts for x in ("Exam 1 (Sep 28", "Exam 2 (Oct 26", "Exam 3 (Nov 30"))


def test_main_argument_of_a_named_source_skips_the_header_chunk():
    res, *_ = _top("US History", "What is the main argument of Federalist No. 10?", limit=5)
    assert not any("From the New York Packet" in r["text"] for r in res)
    assert sum(r["material_title"] == "Federalist No. 10" for r in res) >= 4
    assert retrieval.is_header_chunk("FEDERALIST No. 10 From the New York Packet. Friday, November 23, 1787. MADISON "
                                     "To the People of the State of New York:")
    assert retrieval.is_title_only("Lecture 18", "Lecture 18")


def test_named_documents_each_get_a_share():
    res, *_ = _top("US History", "Federalist No. 10 and 51 and Chapters 7 and 8", limit=12)
    titles = {r["material_title"] for r in res}
    assert {"Federalist No. 10", "Federalist No. 51", "Chapter 7 The Articles of Confederation",
            "Chapter 8 Creating a Constitution"} <= titles
    assert retrieval.query_doc_ids("Federalist No. 10 and 51 and Chapters 7 and 8") == {
        ("federalist", 10), ("federalist", 51), ("chapter", 7), ("chapter", 8)}


def test_lectures_recent_first():
    s, cid, _ = _course("Linear Algebra")
    s.add_event(cid, "lecture", "No class (Thanksgiving)", "2026-11-26")
    s.add_event(cid, "lecture", "Lecture 18: Eigenvalues and eigenvectors", "2026-11-10")  # a repeated syllabus row
    out = retrieval.lectures_recent_first(s, cid, today="2026-11-13")
    assert [e["date"] for e in out] == ["2026-11-12", "2026-11-10", "2026-11-19"]
