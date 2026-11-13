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
           '{"id":"c","label":"","cites":[1]}],"edges":[{"a":"a","b":"b","label":"has"},{"a":"a","b":"zz","label":"bad"}]}'
           "\n\nA basis describes a vector space [1].")
    g = s.guide(study.concept_map(s, cid, Prov(raw), scope="vectors"))
    assert g["kind"] == "concept_map"
    assert [n["id"] for n in g["data"]["nodes"]] == ["a", "b"]
    assert g["data"]["nodes"][0]["cites"] == [1]
    assert g["data"]["edges"] == [{"a": "a", "b": "b", "label": "has"}]
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
    assert rows[1]["action"] == "1 flashcards due"  # primer exists, cards due
    assert rows[2]["action"] == "" and rows[3]["action"] == "make an exam guide"
    assert rows[3]["course_name"] == "Algebra" and rows[3]["course_color"] == "#f00"
    s.add_guide(a, "exam_guide", "g", "b", event_id=ex)
    assert study.upcoming(s, days=14)[3]["action"] == ""
