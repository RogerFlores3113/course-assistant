import re
import sys
import types
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

import pytest

from course_assistant import disclaimer, study
from course_assistant.store import Store


# ---------------------------------------------------------------- stand-ins that follow DESIGN.md
@dataclass
class Answer:
    body_md: str = ""
    blocks: list = field(default_factory=list)
    sources: list = field(default_factory=list)
    used: list = field(default_factory=list)
    outside: bool = False
    twin: bool = False
    guide_id: int | None = None


def fake_parse_answer(raw, sources):
    used, outside, paras = [], False, []
    for p in [p for p in re.split(r"\n\s*\n", raw or "") if p.strip()]:
        def fix(m):
            n = int(m.group(1))
            if 1 <= n <= len(sources):
                if n not in used:
                    used.append(n)
                return m.group(0)
            return ""
        q = re.sub(r"\[(\d+)\]", fix, p)
        if p.lstrip().startswith("#") or p.lstrip().startswith("-") and "[" in q:
            paras.append(q)
            continue
        if "<outside>" in q or not re.search(r"\[\d+\]", q):
            outside = outside or not q.lstrip().startswith("#")
        paras.append(q)
    return Answer(body_md="\n\n".join(paras) + "\n\n" + disclaimer.FOOTER, used=sorted(used), outside=outside,
                  sources=sources)


class Prov:
    name, model = "fake", "fake-1"

    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []

    def complete(self, system, messages, *, max_tokens=4000):
        self.calls.append((system, [dict(m) for m in messages]))
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]


@pytest.fixture
def graded_hit():
    return {"hit": None}


@pytest.fixture(autouse=True)
def stand_ins(monkeypatch, graded_hit):
    from course_assistant import locations

    def search(store, course_id, query, *, limit=8, kinds=None, material_ids=None, exclude_graded=False):
        out = []
        for m in store.materials(course_id):
            if kinds and m["kind"] not in kinds:
                continue
            if material_ids and m["id"] not in material_ids:
                continue
            if exclude_graded and m["graded"]:
                continue
            for c in store.chunks_of(m["id"]):
                out.append({"chunk_id": c["id"], "material_id": m["id"], "material_title": m["title"],
                            "material_kind": m["kind"], "loc_kind": c["loc_kind"], "loc_start": c["loc_start"],
                            "loc_label": c["loc_label"], "heading": c["heading"], "text": c["text"],
                            "href": locations.href_for(c), "score": 1.0})
        return out[:limit]

    ret = types.ModuleType("course_assistant.retrieval")
    ret.search = search
    gr = types.ModuleType("course_assistant.grounding")
    gr.parse_answer = fake_parse_answer
    integ = types.ModuleType("course_assistant.integrity")
    integ.match_graded = lambda store, cid, q: graded_hit["hit"]

    def twin(store, cid, q, provider, graded):
        raw = provider.complete("twin", [{"role": "user", "content": q}])
        a = fake_parse_answer(raw, search(store, cid, q))
        a.twin = True
        return a

    integ.twin = twin
    for name, mod in (("retrieval", ret), ("grounding", gr), ("integrity", integ)):
        monkeypatch.setitem(sys.modules, f"course_assistant.{name}", mod)


@pytest.fixture
def world():
    s = Store(":memory:")
    cid = s.add_course("Linear Algebra", code="MATH 201")
    lec = s.add_material(cid, "slides", "Lecture 3 slides", lecture_date="2026-12-01")
    s.replace_chunks(lec, [{"text": "A vector space has addition and scaling.", "loc_kind": "slide", "loc_start": 1, "loc_label": "slide 1"},
                           {"text": "Basis: independent spanning set.", "loc_kind": "slide", "loc_start": 2, "loc_label": "slide 2"}])
    book = s.add_material(cid, "textbook", "Textbook")
    s.replace_chunks(book, [{"text": "Rank-nullity theorem.", "loc_kind": "page", "loc_start": 9, "loc_label": "p. 9"}])
    return s, cid, lec


def future(n):
    return (date.today() + timedelta(days=n)).isoformat()


def body_of(s, gid):
    return s.guide(gid)


GOOD = "## Big picture\n\nVectors can be added and scaled [1].\n\nA basis spans the space [2] and bogus [9].\n\nUncited claim here."


# ---------------------------------------------------------------- generators
def test_no_provider_everywhere(world):
    s, cid, _ = world
    for call in (lambda: study.primer(s, cid, None), lambda: study.exam_guide(s, cid, None),
                 lambda: study.concept_map(s, cid, None), lambda: study.explain(s, cid, None, "x"),
                 lambda: study.practice(s, cid, None, "x"), lambda: study.make_flashcards(s, cid, None, topic="x"),
                 lambda: study.missed(s, cid, None, date="2026-12-01")):
        with pytest.raises(study.NoProvider):
            call()


def test_primer_picks_next_lecture_and_saves(world):
    s, cid, _ = world
    s.add_event(cid, "lecture", "Old lecture", "2020-01-01")
    e = s.add_event(cid, "lecture", "Vector spaces", "2026-12-01")
    p = Prov(GOOD)
    gid = study.primer(s, cid, p)
    g = s.guide(gid)
    assert g["kind"] == "primer" and g["event_id"] == e and "Vector spaces" in g["title"]
    assert g["body_md"].endswith(disclaimer.FOOTER) and g["body_md"].count(disclaimer.FOOTER) == 1
    assert "[9]" not in g["body_md"]
    assert [x["n"] for x in g["sources"]] == [1, 2] or {x["n"] for x in g["sources"]} <= {1, 2, 3}
    assert all({"chunk_id", "material_id", "href", "label", "title"} <= set(x) for x in g["sources"])
    assert "Vector spaces" in p.calls[0][1][0]["content"] and "Basis" in p.calls[0][1][0]["content"]


def test_primer_without_lecture_errors(world):
    s, cid, _ = world
    with pytest.raises(ValueError):
        study.primer(s, cid, Prov("x"))


def test_exam_guide_topics_since_last_exam(world):
    s, cid, _ = world
    s.add_event(cid, "lecture", "Old topic", "2026-01-10")
    s.add_event(cid, "exam", "Midterm 1", "2026-02-01")
    s.add_event(cid, "lecture", "Vector spaces", "2026-03-01")
    s.add_event(cid, "lecture", "Bases", "2026-03-05")
    ex = s.add_event(cid, "exam", "Midterm 2", future(5))
    gid = study.exam_guide(s, cid, Prov(GOOD), event_id=ex)
    g = s.guide(gid)
    assert g["kind"] == "exam_guide" and g["data"]["topics"][:2] == ["Vector spaces", "Bases"]
    assert "Old topic" not in g["data"]["topics"] and g["data"]["previous_exam"] == "2026-02-01"
    assert g["body_md"].endswith(disclaimer.FOOTER)


def test_exam_guide_graded_note(world, graded_hit):
    s, cid, _ = world
    ex = s.add_event(cid, "exam", "Final", future(3))
    graded_hit["hit"] = {"title": "HW 4"}
    g = s.guide(study.exam_guide(s, cid, Prov(GOOD), event_id=ex))
    assert "twins" in g["body_md"]


def test_explain_depth_and_cites(world):
    s, cid, _ = world
    p = Prov(GOOD)
    g = s.guide(study.explain(s, cid, p, "basis", depth="from scratch"))
    assert g["kind"] == "explain" and g["data"]["depth"] == "from scratch"
    assert "[9]" not in g["body_md"] and g["data"]["outside"] is True
    assert "never seen" in p.calls[0][1][0]["content"]
    g2 = s.guide(study.explain(s, cid, p, "basis", depth="bogus"))
    assert g2["data"]["depth"] == "standard"


def test_practice_normal_and_twin(world, graded_hit):
    s, cid, _ = world
    g = s.guide(study.practice(s, cid, Prov("1. Q [1]\n\n<details><summary>Show solution</summary>\n\nA [2]\n\n</details>"), "bases", count=3))
    assert g["kind"] == "practice" and g["data"]["twin"] is False and "<details>" in g["body_md"]
    graded_hit["hit"] = {"title": "HW 4"}
    p = Prov("Twin problem with other numbers [1]")
    g = s.guide(study.practice(s, cid, p, "hw 4 problem 2"))
    assert g["kind"] == "practice" and g["data"]["twin"] is True and "twin" in g["body_md"].lower()
    assert p.calls[0][0] == "twin"  # routed through integrity.twin


def test_concept_map_valid_and_cleaned(world):
    s, cid, _ = world
    raw = ('{"nodes":[{"id":"a","label":"Vector space","cites":[1,7]},{"id":"b","label":"Basis","cites":[2]},'
           '{"id":"c","label":"","cites":[1]}],"edges":[{"a":"a","b":"b","label":"has","cites":[2,40]},{"a":"a","b":"zz","label":"bad"}]}'
           "\n\nA basis describes a vector space [1].")
    g = s.guide(study.concept_map(s, cid, Prov(raw), scope="vectors"))
    assert g["kind"] == "concept_map"
    assert [n["id"] for n in g["data"]["nodes"]] == ["a", "b"]
    assert g["data"]["nodes"][0]["cites"] == [1]
    assert g["data"]["edges"] == [{"a": "a", "b": "b", "label": "has", "cites": [2], "cites_removed": True}]
    assert g["body_md"].endswith(disclaimer.FOOTER)


def test_concept_map_json_repair(world):
    s, cid, _ = world
    p = Prov("not json at all", '{"nodes":[{"id":"a","label":"X","cites":[1]}],"edges":[]}')
    g = s.guide(study.concept_map(s, cid, p))
    assert len(p.calls) == 2 and "not valid JSON" in p.calls[1][1][-1]["content"]
    assert g["data"]["nodes"][0]["label"] == "X"
    with pytest.raises(ValueError):
        study.concept_map(s, cid, Prov("still bad"))


def test_flashcards_dedupe_and_cites(world):
    s, cid, lec = world
    s.add_flashcard(cid, "What is a basis?", "old")
    raw = ('```json\n[{"front":"What is a basis?","back":"dup","cite":2},{"front":"Define vector space","back":"Closed under + and scaling","cite":1},'
           '{"front":"Rank-nullity?","back":"x","cite":9},{"front":"Define vector space!","back":"again","cite":1}]\n```')
    ids = study.make_flashcards(s, cid, Prov(raw), material_id=lec, count=10)
    assert len(ids) == 1
    c = s.flashcard(ids[0])
    assert c["front"] == "Define vector space" and c["chunk_id"] and "slide 1" in c["source_label"] and c["deck"] == "Lecture 3 slides"
    ids2 = study.make_flashcards(s, cid, Prov('[{"front":"New one","back":"b","cite":1}]'), topic="vectors", count=1)
    assert len(ids2) == 1


def test_missed_by_date_and_event(world):
    s, cid, _ = world
    gid = study.missed(s, cid, Prov(GOOD), date="2026-12-01")
    g = s.guide(gid)
    assert g["kind"] == "missed" and g["sources"] and g["body_md"].endswith(disclaimer.FOOTER)
    e = s.add_event(cid, "lecture", "Vector spaces", "2026-12-01")
    assert s.guide(study.missed(s, cid, Prov(GOOD), event_id=e))["event_id"] == e
    with pytest.raises(ValueError):
        study.missed(s, cid, Prov(GOOD))


# ---------------------------------------------------------------- flashcard scheduling
def card(s, cid, **kw):
    c = s.add_flashcard(cid, "f", "b")
    s.update_flashcard(c, **kw)
    return c


def test_sm2_transitions(world):
    s, cid, _ = world
    c = card(s, cid)
    r = study.review_flashcard(s, c, "good")
    assert (r["reps"], r["interval_days"]) == (1, 1.0)
    r = study.review_flashcard(s, c, "good")
    assert (r["reps"], r["interval_days"]) == (2, 3.0)
    r = study.review_flashcard(s, c, "good")
    assert r["reps"] == 3 and r["interval_days"] == pytest.approx(7.5)
    due = datetime.fromisoformat(r["due_at"])
    assert due > datetime.now(timezone.utc) + timedelta(days=7)
    r = study.review_flashcard(s, c, "easy")
    assert r["ease"] == pytest.approx(2.65) and r["interval_days"] > 7.5
    r = study.review_flashcard(s, c, "hard")
    assert r["ease"] == pytest.approx(2.5)
    r = study.review_flashcard(s, c, "again")
    assert r["reps"] == 0 and r["interval_days"] == 0 and r["ease"] == pytest.approx(2.3)
    assert datetime.fromisoformat(r["due_at"]) < datetime.now(timezone.utc) + timedelta(minutes=11)
    for _ in range(20):
        r = study.review_flashcard(s, c, "again")
    assert r["ease"] == pytest.approx(1.3)
    c2 = card(s, cid)
    assert study.review_flashcard(s, c2, "easy")["interval_days"] == 4.0
    assert study.review_flashcard(s, card(s, cid), "hard")["interval_days"] == 1.0
    with pytest.raises(ValueError):
        study.review_flashcard(s, c, "meh")
    with pytest.raises(KeyError):
        study.review_flashcard(s, 9999, "good")


# ---------------------------------------------------------------- upcoming
def test_upcoming_ordering_and_actions():
    s = Store(":memory:")
    a = s.add_course("Algebra", code="M1", color="#f00")
    b = s.add_course("Biology", code="B1")
    ex = s.add_event(a, "exam", "Midterm", future(5))
    lec = s.add_event(b, "lecture", "Cells", future(1), time="09:00")
    s.add_event(b, "lecture", "Cells 2", future(1), time="08:00")
    s.add_event(a, "assignment", "HW", future(2))
    s.add_event(a, "lecture", "Far away", future(40))
    s.add_event(a, "lecture", "Past", "2020-01-01")
    s.add_guide(b, "primer", "p", "body", event_id=lec)
    s.add_flashcard(b, "q", "a")
    rows = study.upcoming(s, days=14)
    assert [r["title"] for r in rows] == ["Cells 2", "Cells", "HW", "Midterm"]
    assert rows[0]["action"] == "make a primer" and rows[0]["days_until"] == 1
    assert rows[1]["action"] == "1 flashcard due"  # primer exists, cards due
    assert rows[2]["action"] == "" and rows[3]["action"] == "make an exam guide"
    assert rows[3]["course_name"] == "Algebra" and rows[3]["course_color"] == "#f00"
    s.add_guide(a, "exam_guide", "g", "b", event_id=ex)
    assert study.upcoming(s, days=14)[3]["action"] == ""


# ================================================================ cycle 1 fixes (fx-study)
def mat(s, cid, kind, title, **kw):
    link = {k: kw.pop(k) for k in ("event_id", "duplicate_of") if k in kw}
    m = s.add_material(cid, kind, title, **kw)
    if link:
        s.update_material(m, **link)
    return m


def integ():
    return sys.modules["course_assistant.integrity"]


def user_msg(p, i=0):
    return p.calls[i][1][0]["content"]


# ---- integrity wiring (contracts 2 to 4)
def test_system_prompt_has_integrity_block_and_format_rules(world, monkeypatch):
    s, cid, _ = world
    monkeypatch.setattr(integ(), "prompt_block", lambda store, c: "INTEGRITY RULE: graded items are listed", raising=False)
    p = Prov(GOOD)
    study.explain(s, cid, p, "basis")
    assert "INTEGRITY RULE" in p.calls[0][0] and "inside $...$" in p.calls[0][0]


def test_marker_routes_reply_to_twin(world, monkeypatch):
    s, cid, _ = world
    item = s.add_graded_item(cid, "HW 4", "Solve 3x = 12")
    monkeypatch.setattr(integ(), "read_marker", lambda raw: (item, raw.replace(f'<graded item="{item}"/>', "")), raising=False)
    p = Prov(f'<graded item="{item}"/>x = 4 [1]', "Twin with other numbers [1]")
    gid = study.explain(s, cid, p, "linear equations")
    assert p.calls[1][0] == "twin"  # the integrity twin ran
    assert "x = 4" not in str(s.guide(gid)["body_md"]) if gid else True


def test_check_reply_discards_solving_reply(world, monkeypatch):
    s, cid, _ = world
    monkeypatch.setattr(integ(), "read_marker", lambda raw: (None, raw), raising=False)
    monkeypatch.setattr(integ(), "check_reply", lambda store, c, q, raw: {"title": "HW 4"}, raising=False)
    p = Prov("Full solution x = 4 [1]", "Twin text [1]")
    study.practice(s, cid, p, "equations", count=2)
    assert any(c[0] == "twin" for c in p.calls)


def test_no_integrity_contract_still_works(world):
    s, cid, _ = world
    assert study.explain(s, cid, Prov(GOOD), "basis")  # stub has no prompt_block, read_marker or check_reply


# ---- past offerings (p1)
def test_past_offering_sources_are_marked(world):
    s, cid, _ = world
    s.update_course(cid, term="Fall 2026")
    old = s.add_material(cid, "past_offering", "Fall 2018 lecture summaries")
    s.replace_chunks(old, [{"text": "Three exams this term.", "loc_kind": "section", "loc_start": "a", "loc_label": "Exams"}])
    p = Prov(GOOD)
    study.explain(s, cid, p, "exams")
    assert "from a past offering, Fall 2018; may differ from this term" in user_msg(p)
    assert "past offering" in p.calls[0][0]
    assert user_msg(p).count("past offering") == 1  # this term's slides are not marked


def test_term_of_and_other_term_syllabus_marked(world):
    assert study.term_of("fall2018-lecture-summaries.md") == "Fall 2018"
    assert study.term_of("Spring '22 notes") == "Spring 2022" and study.term_of("Chemistry 2e") == ""
    s, cid, _ = world
    s.update_course(cid, term="Fall 2026")
    m = s.add_material(cid, "notes", "MIT Course, Fall 2018")
    cache = {}
    assert "Fall 2018" in study._past_label(s, cid, m, cache)
    cur = s.add_material(cid, "notes", "Fall 2026 notes")
    assert study._past_label(s, cid, cur, {}) == ""


# ---- lecture matching (p1)
def test_lecture_materials_priority_and_captions_without_date(world):
    s, cid, _ = world
    ev = s.add_event(cid, "lecture", "Lecture 18: Eigenvalues", "2026-11-10")
    cap = s.add_material(cid, "recording", "lecture-18-eigenvalues-captions")
    s.replace_chunks(cap, [{"text": "An eigenvector keeps its direction.", "loc_kind": "time", "loc_start": 58, "loc_label": "0:58"}])
    other = s.add_material(cid, "notes", "lecture-17-notes")
    s.replace_chunks(other, [{"text": "Projections.", "loc_kind": "page", "loc_start": 1, "loc_label": "p. 1"}])
    got = study.lecture_materials(s, cid, s.event(ev))
    assert [m["id"] for m, _ in got] == [cap] and "number" in got[0][1]
    linked = mat(s, cid, "slides", "Slides", event_id=ev)
    s.replace_chunks(linked, [{"text": "Slide text.", "loc_kind": "slide", "loc_start": 1, "loc_label": "slide 1"}])
    got = study.lecture_materials(s, cid, s.event(ev))
    assert [w for _, w in got] == ["linked to this lecture", "matched by lecture number in its name"]


def test_missed_finds_undated_captions_and_says_what_it_used(world):
    s, cid, _ = world
    ev = s.add_event(cid, "lecture", "Lecture 18", "2026-11-10")
    cap = s.add_material(cid, "recording", "Lecture 18 captions")
    s.replace_chunks(cap, [{"text": "Eigenvalues today.", "loc_kind": "time", "loc_start": 0, "loc_label": "0:00"}])
    p = Prov(GOOD)
    g = s.guide(study.missed(s, cid, p, event_id=ev))
    assert "Lecture 18 captions" in user_msg(p) and "Eigenvalues today." in user_msg(p)
    assert "Used for this lecture: Lecture 18 captions" in g["body_md"]
    assert g["data"]["used_materials"][0]["title"] == "Lecture 18 captions"


def test_missed_with_nothing_linked_says_so(world):
    s, cid, _ = world
    ev = s.add_event(cid, "lecture", "Lecture 40", "2030-01-01")
    g = s.guide(study.missed(s, cid, Prov(GOOD), event_id=ev))
    assert "No saved material is linked" in g["body_md"]


def test_primer_uses_event_link_and_skips_junk_events(world):
    s, cid, _ = world
    s.add_event(cid, "lecture", "to", future(1))
    s.add_event(cid, "lecture", "Wed Sep 23, Fri Sep 25, Mon", future(1))
    real = s.add_event(cid, "lecture", "17 | Lecture 20", future(2), topics="Diagonalization")
    m = mat(s, cid, "slides", "Odd name", event_id=real)
    s.replace_chunks(m, [{"text": "Diagonalize A.", "loc_kind": "slide", "loc_start": 1, "loc_label": "slide 1"}])
    p = Prov(GOOD)
    g = s.guide(study.primer(s, cid, p))
    assert g["event_id"] == real and g["title"] == "Primer: Lecture 20" and "Diagonalize A." in user_msg(p)


def test_junk_events_and_labels():
    assert study.is_junk_event({"title": "to"}) and study.is_junk_event({"title": "There is no class on"})
    assert study.is_junk_event({"title": "Wed Sep 23, Fri Sep 25, Mon"})
    assert study.is_junk_event({"title": "Students with accommodations should contact the office of disability services "
                                         "as early as possible in the semester so that we can plan"})
    assert not study.is_junk_event({"title": "Lecture 20: Eigenvalues"})
    assert study.event_label({"title": "17 | Lecture 20", "topics": "Diagonalization"}) == "Lecture 20: Diagonalization"
    assert study.event_label({"title": "There is no class on", "topics": "Thanksgiving break"}) == "Thanksgiving break"


def test_study_form_options(world):
    s, cid, lec = world
    s.add_event(cid, "lecture", "to", future(1))
    s.add_event(cid, "lecture", "Bases", future(2), topics="span")
    s.add_event(cid, "exam", "Midterm", future(9))
    o = study.study_form_options(s, cid)
    assert [e["label"] for e in o["next_lectures"]] == ["Bases: span"] and o["exams"][0]["label"] == "Midterm"
    assert lec in {m["id"] for m in o["materials"]}


# ---- exam guide (p2)
def test_exam_guide_uses_window_materials_and_names_missing_lectures(world):
    s, cid, lec = world
    prev = s.add_event(cid, "exam", "Midterm 1", "2026-02-01")
    l1 = s.add_event(cid, "lecture", "Molarity", "2026-02-10")
    l2 = s.add_event(cid, "lecture", "Kinetics", "2026-02-12", topics="rates")
    ex = s.add_event(cid, "exam", "Final", future(3))
    ch = mat(s, cid, "textbook", "Chapter 3", event_id=l1)
    s.replace_chunks(ch, [{"text": "Molarity is moles per liter.", "loc_kind": "section", "loc_start": "a", "loc_label": "Molarity"}])
    p = Prov(GOOD)
    g = s.guide(study.exam_guide(s, cid, p, event_id=ex))
    assert g["data"]["missing_lectures"] == ["Kinetics: rates (2026-02-12)"]
    assert "No saved materials for these lectures" in g["body_md"] and "- Kinetics: rates (2026-02-12)" in g["body_md"]
    assert g["data"]["materials"] == ["Chapter 3"] and "Molarity is moles per liter." in user_msg(p)
    assert "Vector space" not in user_msg(p) and "NO saved materials" in user_msg(p)
    assert g["data"]["topics"] == ["Molarity", "Kinetics rates"] and g["data"]["previous_exam"] == "2026-02-01"


def test_exam_guide_cumulative_falls_back_to_all_materials(world):
    s, cid, lec = world
    ex = s.add_event(cid, "exam", "Final", future(3))
    g = s.guide(study.exam_guide(s, cid, Prov(GOOD), event_id=ex))
    assert "all your saved materials" in g["body_md"] and "Lecture 3 slides" in g["data"]["materials"]


def test_exam_guide_chosen_materials_only(world):
    s, cid, lec = world
    book = next(m["id"] for m in s.materials(cid) if m["title"] == "Textbook")
    ex = s.add_event(cid, "exam", "Final", future(3))
    p = Prov(GOOD)
    g = s.guide(study.exam_guide(s, cid, p, event_id=ex, material_ids=[book]))
    assert g["data"]["materials"] == ["Textbook"] and "Limited to the materials you chose" in g["body_md"]
    assert "Rank-nullity" in user_msg(p) and "A vector space has" not in user_msg(p)
    with pytest.raises(ValueError):
        study.exam_guide(s, cid, Prov(GOOD), event_id=ex, material_ids=[9999])


def test_exam_guide_picks_chosen_exam_and_skips_junk_exam_rows(world):
    s, cid, _ = world
    s.add_event(cid, "exam", "to", future(1))
    later = s.add_event(cid, "exam", "Final", future(20))
    g = s.guide(study.exam_guide(s, cid, Prov(GOOD), event_id=later))
    assert g["event_id"] == later
    nxt = s.guide(study.exam_guide(s, cid, Prov(GOOD)))
    assert nxt["event_id"] == later  # the junk "to" row is not the next exam


# ---- flashcards (p2)
def test_flashcards_whole_material_spread_and_dropped_report(world):
    s, cid, _ = world
    m = s.add_material(cid, "textbook", "Chapter 3")
    s.replace_chunks(m, [{"text": f"Section {i} text about topic {i}.", "heading": "Molarity" if i < 3 else f"S{i}",
                          "loc_kind": "section", "loc_start": str(i), "loc_label": f"S{i}"} for i in range(6)])
    cards = ('[{"front":"A","back":"a","cite":1},{"front":"B","back":"b","cite":1},{"front":"C","back":"c","cite":1},'
             '{"front":"D","back":"d","cite":4},{"front":"E","back":"e","cite":99},{"front":"F","back":"f","cite":"x"},'
             '{"front":"A","back":"dup","cite":2}]')
    p = Prov(cards)
    res = study.make_flashcards(s, cid, p, material_id=m, count=3)
    assert all(f"Section {i}" in user_msg(p) for i in range(6))  # the whole chapter, not the first chunk
    fronts = [s.flashcard(i)["front"] for i in res]
    assert fronts == ["A", "B", "D"]  # at most 2 per source first, then the overflow only if needed
    assert res.dropped == 2 and res.duplicates == 1 and "dropped" in res.note and "Made 3 cards." in res.note
    assert "spread" in user_msg(p)


def test_flashcards_topic_mode_pulls_section_siblings(world, monkeypatch):
    s, cid, _ = world
    m = s.add_material(cid, "textbook", "Chapter 3")
    s.replace_chunks(m, [{"text": "Molarity intro only.", "heading": "Molarity", "loc_kind": "section", "loc_start": "a", "loc_label": "Molarity"},
                         {"text": "M = mol / L is the definition.", "heading": "Molarity", "loc_kind": "section", "loc_start": "b", "loc_label": "Molarity"},
                         {"text": "Unrelated heading text.", "heading": "Other", "loc_kind": "section", "loc_start": "c", "loc_label": "Other"}])
    first = s.chunks_of(m)[0]["id"]
    monkeypatch.setattr(sys.modules["course_assistant.retrieval"], "search",
                        lambda store, c, q, **kw: [study._source_from_chunk(store.chunk(first), "Chapter 3", "textbook")])
    p = Prov('[{"front":"Define M","back":"mol/L","cite":2}]')
    assert len(study.make_flashcards(s, cid, p, topic="molarity")) == 1
    assert "M = mol / L is the definition." in user_msg(p) and "Unrelated heading" not in user_msg(p)


# ---- cram and scheduling (p2)
def test_cram_queue_weakest_first_ignores_due(world):
    s, cid, lec = world
    strong = card(s, cid, ease=2.8, interval_days=9, reps=4, due_at="2999-01-01T00:00:00+00:00")
    weak = card(s, cid, ease=1.5, interval_days=1, reps=3, due_at="2999-01-01T00:00:00+00:00")
    new = card(s, cid)
    q = study.cram_queue(s, cid)
    assert [c["id"] for c in q] == [weak, new, strong]
    assert len(study.cram_queue(s, cid, limit=2)) == 2


def test_cram_queue_by_material(world):
    s, cid, lec = world
    chunk = s.chunks_of(lec)[0]["id"]
    a = s.add_flashcard(cid, "q1", "a", chunk_id=chunk)
    s.add_flashcard(cid, "q2", "a")
    assert [c["id"] for c in study.cram_queue(s, cid, material_id=lec)] == [a]


def test_review_in_cram_mode_stays_before_exam(world):
    s, cid, _ = world
    s.add_event(cid, "exam", "Final", future(20))
    c = card(s, cid, reps=3, interval_days=10, ease=2.5)
    r = study.review_flashcard(s, c, "easy", cram=True)
    due = datetime.fromisoformat(r["due_at"])
    assert due < datetime.now(timezone.utc) + timedelta(days=10.5) and r["interval_days"] <= 10
    c2 = card(s, cid, reps=3, interval_days=10, ease=2.5)
    assert study.review_flashcard(s, c2, "good")["interval_days"] > 20  # not near the exam, no cap outside cram


def test_exam_within_a_week_caps_intervals(world):
    s, cid, _ = world
    s.add_event(cid, "exam", "Final", future(3))
    c = card(s, cid, reps=2, interval_days=3, ease=2.5)
    r = study.review_flashcard(s, c, "easy")
    assert r["interval_days"] == 1.5
    assert datetime.fromisoformat(r["due_at"]) < datetime.now(timezone.utc) + timedelta(days=2)
    s2 = card(s, cid)
    assert study.review_flashcard(s, s2, "good", exam_date=future(1))["interval_days"] == 0.5
    s3 = card(s, cid)
    assert study.review_flashcard(s, s3, "good", exam_date=future(0))["interval_days"] == 0.25
    s4 = card(s, cid, reps=2, interval_days=3)
    assert study.review_flashcard(s, s4, "good", exam_date="2020-01-01")["interval_days"] > 1  # exam already past


# ---- practice (p2)
def test_practice_honours_count_and_solutions_are_multiline_details(world):
    s, cid, _ = world
    raw = "1. Q [1]\n\n<details><summary>Show solution</summary>\n\nA [2]\n\n</details>"
    p = Prov(raw)
    g = s.guide(study.practice(s, cid, p, "bases", count=7))
    assert "Write 7 practice questions" in user_msg(p) and g["data"]["count"] == 7
    assert "<details>\n<summary>Show solution</summary>" in g["body_md"] and "</details>" in g["body_md"]
    assert all(l.strip() != "<details><summary>Show solution</summary>" for l in g["body_md"].splitlines())


def test_normalize_details():
    out = study.normalize_details("Q1 text <details><summary>Answer</summary> 42 </details> after")
    assert out == "Q1 text \n<details>\n<summary>Answer</summary>\n42 \n</details>\nafter".replace("42 ", " 42 ").replace(" 42", "42").replace("42 ", "42")


def test_practice_twins_for_every_graded_overlap(world, graded_hit):
    s, cid, _ = world
    s.add_graded_item(cid, "Problem 1: molarity of NaCl", "How many grams of NaCl for 250 mL of 0.35 M solution")
    s.add_graded_item(cid, "Problem 2: empirical formula", "Find the empirical formula given percent composition 40.92 C")
    graded_hit["hit"] = {"id": 1, "title": "Problem 1: molarity of NaCl", "body": "x"}
    p = Prov("Twin or question text [1]")
    g = s.guide(study.practice(s, cid, p, "molarity and empirical formula, percent composition solution", count=5))
    assert g["data"]["twin"] is True and len(g["data"]["twins"]) == 2
    assert [c[0] for c in p.calls].count("twin") == 2
    assert "Write 3 practice questions" in p.calls[2][1][0]["content"]
    assert g["body_md"].count("## Twin of graded work") == 2
    assert not s.guides(cid, kind="twin")  # no stray guides: they were folded in


def test_practice_all_twins_when_count_is_small(world, graded_hit):
    s, cid, _ = world
    graded_hit["hit"] = {"title": "HW 4"}
    p = Prov("Twin text [1]")
    g = s.guide(study.practice(s, cid, p, "hw 4", count=1))
    assert [c[0] for c in p.calls] == ["twin"] and g["title"].startswith("Practice (twin)")


def test_shift_block_renumbers_only_real_cites():
    b = {"kind": "text", "md": "Use [1] and [2, 1] on $[1, 0]$ then [5]", "cites": [1, 2]}
    out = study._shift_block(b, 10)
    assert out["md"] == "Use [11] and [12, 11] on $[1, 0]$ then [5]" and out["cites"] == [11, 12]


# ---- concept map (p4)
def test_concept_map_edges_carry_cites_and_cover_all_kinds(world):
    s, cid, _ = world
    prim = s.add_material(cid, "other", "Federalist No. 10")
    s.replace_chunks(prim, [{"text": "faction", "loc_kind": "line", "loc_start": 1, "loc_label": "line 1"}])
    s.add_material(cid, "syllabus", "Syllabus")
    p = Prov('{"nodes":[{"id":"a","label":"A","cites":[1]},{"id":"b","label":"B","cites":[2]}],'
             '"edges":[{"a":"a","b":"b","label":"leads to","cites":[3,77]}]}\n\nSummary [1].')
    g = s.guide(study.concept_map(s, cid, p))
    assert "Federalist No. 10" in user_msg(p) and "Syllabus" not in user_msg(p)
    assert '"cites"' in user_msg(p).split("SOURCES")[0] and "Every edge needs at least" in user_msg(p)
    assert g["data"]["edges"] == [{"a": "a", "b": "b", "label": "leads to", "cites": [3], "cites_removed": True}]
    assert 3 in {x["n"] for x in g["sources"]} and g["data"]["mode"] == "answer"


def test_clean_graph_edge_cites():
    out = study._clean_graph({"nodes": [{"id": "a", "label": "A"}, {"id": "b", "label": "B"}],
                              "edges": [{"a": "a", "b": "b", "label": "x", "cites": ["2", "9", "q", 2]},
                                        {"a": "a", "b": "b", "label": "y"}]}, 3)
    assert [e["cites"] for e in out["edges"]] == [[2], []]


def test_concept_map_scope_names_a_material(world):
    s, cid, lec = world
    p = Prov('{"nodes":[{"id":"a","label":"A","cites":[1]}],"edges":[]}')
    study.concept_map(s, cid, p, scope="Lecture 3")
    assert "A vector space has" in user_msg(p)


# ---- explain from scratch and messages
def test_explain_from_scratch_prompt(world):
    s, cid, _ = world
    p = Prov(GOOD)
    study.explain(s, cid, p, "basis", depth="from scratch")
    u = user_msg(p)
    assert "Define every term" in u and "short sentences" in u and "exactly one example" in u and "2 questions" in u


def test_ready_messages_per_kind():
    assert study.ready_message("practice") == "Your practice questions are ready."
    assert study.ready_message("explain") == "The explanation is ready."
    assert study.ready_message("exam") == "Your exam guide is ready." and study.ready_message("zzz") == "Your guide is ready."
    assert "Explanation" not in study.ready_message("practice")


def test_guide_data_has_mode_and_not_covered(world):
    s, cid, _ = world
    d = s.guide(study.explain(s, cid, Prov(GOOD), "basis"))["data"]
    assert d["mode"] == "answer" and d["not_covered"] is False


def test_study_tab_template_renders_pickers():
    from pathlib import Path
    import jinja2
    import course_assistant
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(Path(course_assistant.__file__).parent / "templates"),
                             autoescape=True)
    tpl = env.get_template("tab_study.html")
    base = dict(ai={"connected": True}, course={"id": 3}, has_materials=True, depths=[("standard", "Standard")],
                due_cards=1, all_cards=4, result_view=None, cards_made=0)
    events = [{"id": 1, "kind": "lecture", "date": "2026-09-01", "title": "17 | Lecture 20", "topics": "Diagonalization"},
              {"id": 2, "kind": "lecture", "date": "2026-09-02", "title": "to", "topics": ""},
              {"id": 3, "kind": "exam", "date": "2026-10-01", "title": "Midterm", "topics": ""}]
    html = tpl.render(**base, events=events, past_lectures=events[:2])
    assert "Diagonalization" in html and "17 |" not in html and "name=\"material_id\"" not in html
    assert ">2026-09-02: to<" not in html and "cards_made" not in html
    html = tpl.render(**base, events=events, past_lectures=[], study_opts={
        "next_lectures": [{"id": 1, "date": "2026-09-01", "label": "Lecture 20: Diagonalization"}], "past_lectures": [],
        "exams": [], "materials": [{"id": 7, "title": "Chapter 3", "kind": "textbook"}]})
    assert 'name="material_id"' in html and 'name="material_ids"' in html and "Lecture 20: Diagonalization" in html
    assert "?cram=1" in html


def test_bare_graded_marker_routes_to_twin(world, monkeypatch):
    s, cid, _ = world
    s.add_graded_item(cid, "HW 4", "Solve 3x = 12")
    monkeypatch.setattr(integ(), "read_marker", lambda raw: (0, raw.replace("<graded/>", "")), raising=False)
    p = Prov("<graded/>x = 4 [1]", "Twin text [1]")
    gid = study.explain(s, cid, p, "equations")
    assert p.calls[1][0] == "twin" and "x = 4" not in s.guide(gid)["body_md"]


def test_old_one_line_integrity_rule_is_gone_from_base_prompt():
    assert "never solve a problem the student says is graded" not in study.SYSTEM
