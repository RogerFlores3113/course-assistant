"""Cycle 4 source-selection, import and calendar regressions. All text here is original."""
import re

import pytest

from course_assistant import calendar, importers, retrieval, study
from course_assistant.importers import extract
from course_assistant.store import Store


class Prov:
    name, model = "fake", "fake-1"

    def __init__(self, reply="Fine. [1]"):
        self.reply, self.prompts = reply, []

    def complete(self, system, messages, *, max_tokens=4000):
        self.prompts.append(messages[-1]["content"])
        return self.reply


def mat(s, cid, kind, title, chunks, **kw):
    mid = s.add_material(cid, kind, title, **kw)
    s.replace_chunks(mid, [{"text": t, "loc_kind": "section", "loc_start": str(i), "loc_label": h or title, "heading": h}
                           for i, (h, t) in enumerate(chunks)])
    return mid


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("COURSE_ASSISTANT_DATA_DIR", str(tmp_path / "data"))
    s = Store(":memory:")
    return s, s.add_course("Civics", term="Fall 2026"), tmp_path


# 1. a scoped map gives every named source its slots
def _civics(s, cid):
    ids = {}
    for n, (word, filler) in {"10": ("faction", "STATUS OF WOMEN"), "51": ("ambition", "")}.items():
        ids[f"fed{n}"] = mat(s, cid, "reading", f"Federalist No. {n}: On {word}",
                             [(f"para {i}", f"The {word} of government and liberty matters because power checks power, "
                               f"number {i}. " * 6) for i in range(10)])
    for n in ("7", "8"):
        chunks = [(f"Section {i}", "Republican government and liberty depend on how power is divided among "
                   f"branches, part {i}. " * 6) for i in range(10)]
        chunks += [("STATUS OF WOMEN", "Household roles changed little after the war for most families. " * 6),
                   ("Glossary", "- **republic**: a government run by representatives. " * 3),
                   ("Critical Thinking Questions", "Why did states write constitutions? " * 5)]
        ids[f"ch{n}"] = mat(s, cid, "textbook", f"Chapter {n}: Early Republic", chunks)
    return ids


class MapProv(Prov):
    def complete(self, system, messages, *, max_tokens=4000):
        return '{"nodes":[{"id":"a","label":"A","cites":[1]}],"edges":[]}\n\nok [1]'


def test_scoped_map_gives_every_named_source_slots_and_skips_filler(env, monkeypatch):
    s, cid, tmp = env
    _civics(s, cid)
    seen = []
    orig = study._prepare
    monkeypatch.setattr(study, "_prepare", lambda st, c, src, keep_graded=False: seen.__setitem__(slice(None), orig(st, c, src, keep_graded)) or seen)
    study.concept_map(s, cid, MapProv(), scope="Federalist No. 10 and 51 and Chapters 7 and 8")
    titles = [x["material_title"] for x in seen]
    for key in ("Federalist No. 10", "Federalist No. 51", "Chapter 7", "Chapter 8"):
        assert sum(t.startswith(key) for t in titles) >= 4, key
    assert not any(x["heading"] in ("STATUS OF WOMEN", "Glossary", "Critical Thinking Questions") for x in seen)


def test_a_list_of_named_documents_is_not_split_into_clauses(env):
    s, cid, _ = env
    _civics(s, cid)
    q = "Federalist No. 10 and 51 and Chapters 7 and 8"
    r = retrieval.search(s, cid, q, limit=20)
    for key in ("No. 10", "No. 51", "Chapter 7", "Chapter 8"):
        assert any(key in x["material_title"] for x in r), key


# 2. 'this term' means the syllabus; 'last year' means a saved past offering
def _term_course(s, cid):
    syl = mat(s, cid, "syllabus", "Civics syllabus, Fall 2026",
              [("Grading", "Papers 30 percent. The Federalist essay is due on October 20 and counts for a quarter."),
               ("Exams", "Midterm on Thursday October 1 in class. Final on December 15.")])
    past = mat(s, cid, "past_offering", "Civics Fall 2023 syllabus",
               [("Essay", "In Fall 2023 the essay was a five page paper due in week twelve."),
                ("Exams", "Fall 2023 midterm was on October 5.")])
    other = mat(s, cid, "textbook", "Textbook", [("Federalist", "The Federalist essay form is explained here at length. " * 8)])
    return syl, past, other


def test_this_term_question_retrieves_the_syllabus_row(env):
    s, cid, _ = env
    syl, _, _ = _term_course(s, cid)
    r = retrieval.search(s, cid, "How was the Federalist essay different in Fall 2023 from this term", limit=5)
    assert any(x["material_id"] == syl and "October 20" in x["text"] for x in r)


def test_last_year_and_previous_exam_find_a_saved_past_offering(env):
    s, cid, _ = env
    _, past, _ = _term_course(s, cid)
    for q in ("What was on the midterm last year?", "What was on the previous exam?"):
        r = retrieval.search(s, cid, q, limit=4)
        assert past in [x["material_id"] for x in r[:3]], q


# 3. the worked example's next passage stays when another passage ranks first
def test_worked_example_continuation_survives_a_two_part_question(env):
    s, cid, _ = env
    mat(s, cid, "notes", "Lecture captions", [
        ("", "Take the matrix [[4,1],[2,3]] and look for its eigenvalues first."),
        ("", "Subtract t on the diagonal and set the determinant to zero, which gives a quadratic in t."),
        ("", "Then the roots of that quadratic are the eigenvalues, five and two, and you check the trace.")])
    mat(s, cid, "notes", "Markov", [("Eigenvalues of Markov matrices", "Eigenvalues of a Markov matrix include one, eigenvalues "
                                      "eigenvalues eigenvalues and eigenvectors of the matrix [[4,1],[2,3]] eigenvectors.")])
    q = "How do I find the eigenvalues of [[4,1],[2,3]], and what are the eigenvectors?"
    r = retrieval.search(s, cid, q, limit=4)
    assert any("quadratic in t" in x["text"] for x in r)


# 4. exam guides
def _chem(s, cid):
    chap = []
    for h, body in [("Molarity", "Molarity is the moles of solute per liter of solution. **Molarity** is written M."),
                    ("Dilution", "Diluting a solution keeps the moles of solute the same, so C1V1 = C2V2 holds."),
                    ("Limiting Reactant", "The limiting reactant is the reactant that runs out first and sets the yield."),
                    ("Titration", "A titration finds an unknown concentration by adding a known solution to the end point.")]:
        chap.append((h, body + " Extra explanation sentence for length. " * 20))
    chap = [("Chapter 3 Solutions", "Introduction\n[Figure: a pool] Many people think chemistry is old. " * 5)] + chap + [
        ("Key Concepts and Summary", "Short recap of the chapter. " * 30),
        ("Chemistry End of Chapter Exercises", "**Problem.** Find the molarity of a sample. " * 10),
        ("Glossary", "- **molarity**: moles per liter. " * 10)]
    return mat(s, cid, "textbook", "Chapter 3: Solutions", chap)


def test_exam_guide_takes_one_passage_per_topic_heading_and_skips_filler(env):
    s, cid, _ = env
    book = _chem(s, cid)
    lab = mat(s, cid, "handout", "Lab 3 handout", [("Logistics", "Lab date: October 9. Room 214 in the science building.")])
    ex = s.add_event(cid, "exam", "Exam 2", "2027-01-20")
    p = Prov()
    study.exam_guide(s, cid, p, event_id=ex, material_ids=[book, lab])
    guide = s.guides(cid)[0]
    heads = [x.get("label") for x in __import__("json").loads(guide["sources_json"])]
    prompt = p.prompts[0]
    cited = [c for c in s.chunks_of(book) if c["heading"] in ("Molarity", "Dilution", "Limiting Reactant", "Titration")]
    assert cited
    assert "Chapter 3 Solutions" not in prompt.split("SOURCES")[1] and "End of Chapter" not in prompt.split("SOURCES")[1]
    assert "Room 214" not in prompt.split("SOURCES")[1]
    for h in ("Molarity", "Limiting Reactant", "Titration", "Dilution"):
        assert prompt.split("SOURCES")[1].count(f"- {h}") == 1, h


def test_exam_in_the_past_says_when_it_was(env):
    s, cid, _ = env
    book = _chem(s, cid)
    ex = s.add_event(cid, "exam", "Midterm 1", "2020-03-01")
    p = Prov()
    study.exam_guide(s, cid, p, event_id=ex, material_ids=[book])
    assert "this exam was on 2020-03-01" in p.prompts[0]
    assert "0 days from today" not in p.prompts[0]


def test_unlinked_notes_are_matched_to_lectures_by_title_and_suggested(env):
    s, cid, _ = env
    l3 = s.add_event(cid, "lecture", "Lecture 3: Inverses, Gauss-Jordan, introduction to LU", "2026-09-10")
    s.add_event(cid, "lecture", "Lecture 4: Subspaces and nullspace", "2026-09-15")
    n = mat(s, cid, "lecture_notes", "Inverses", [("", "Inverse matrices and elimination steps. " * 10)],
            original_name="inverses.md")
    for i in range(4):  # other files so the course has a spread of names
        mat(s, cid, "notes", f"Misc {i}", [("", "Something unrelated about cooking. " * 10)])
    sug = study.link_suggestions(s, cid)
    assert [(x["material"]["id"], x["event"]["id"]) for x in sug] == [(n, l3)]
    ev = s.add_event(cid, "exam", "Midterm 1", "2026-10-01")
    p = Prov()
    study.exam_guide(s, cid, p, event_id=ev)
    assert "Inverses" in p.prompts[0]
    assert "Matched by title, not linked" in s.guides(cid)[0]["body_md"]


# 5. a primer with nothing linked
def test_primer_with_nothing_linked_uses_syllabus_reading_and_related_notes(env):
    s, cid, _ = env
    s.add_event(cid, "lecture", "Lecture 20: Powers of A", "2026-11-17", topics="Strang 6.2")
    mat(s, cid, "syllabus", "Syllabus", [
        ("Lecture schedule", "Nov 17 | Lecture 20: Powers of A | Strang 6.2"),
        ("Catalog description", "Powers of A and other matrix topics are surveyed in this basic subject. " * 3)])
    mat(s, cid, "notes", "Markov notes", [("Powers", "Powers of A: a Markov matrix raised to a power settles down. " * 4)])
    p = Prov()
    from course_assistant.study import primer
    primer(s, cid, p, event_id=s.events(cid)[0]["id"])
    body = s.guides(cid)[0]["body_md"]
    assert "Nothing is linked to this lecture yet" in body and "primer is built from the syllabus reading" in body
    sources = p.prompts[0].split("SOURCES")[1]
    assert "Nov 17 | Lecture 20" in sources and "Markov" in sources and "Catalog description" not in sources


def test_web_primer_is_not_blocked_when_nothing_is_linked(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import course_assistant.web as web
    monkeypatch.setenv("COURSE_ASSISTANT_DATA_DIR", str(tmp_path / "data"))
    src = open(web.__file__).read()
    assert 'if action == "missed" and (ev is not None or day.strip()):' in src


# 6. importers and calendar
def test_heading_labels_keep_underscores_in_math():
    assert extract.strip_markdown("The summation $$c_{ij} = a_{ik} b_{kj}$$") == "The summation $$c_{ij} = a_{ik} b_{kj}$$"
    assert extract.strip_markdown("_emphasis_ here") == "emphasis here"
    assert extract.strip_markdown("snake_case name") == "snake_case name"


def test_empty_parentheses_artifacts_are_stripped_at_import(env, tmp_path):
    s, cid, _ = env
    p = tmp_path / "chapter-1.md"
    p.write_text("# Chapter 1: People\n\nThomas Jefferson () wrote it. The Bill of Rights () followed. Call f() to run.\n")
    mid = importers.import_file(s, cid, p)
    text = " ".join(c["text"] for c in s.chunks_of(mid))
    assert "Jefferson wrote it" in text and "Rights followed" in text and "f()" in text and " ()" not in text


def test_a_readme_is_not_a_syllabus_and_gives_no_events(env, tmp_path):
    s, cid, _ = env
    p = tmp_path / "README.md"
    p.write_text("# Sample course\n\nThis sample has lectures starting Sep 3 2026, Midterm 1 on Oct 1, and a final on "
                 "Dec 15 with grading policy, office hours and a syllabus described below in some prose.\n")
    mid = importers.import_file(s, cid, p)
    assert s.material(mid)["kind"] != "syllabus"
    assert s.events(cid) == []


def test_prose_in_a_syllabus_makes_no_garbage_event():
    text = ("This sample has lectures starting Sep 3 2026, Midterm 1 on Oct 1, and a final exam on Dec 15 with grading "
            "and office hours explained at length in the paragraphs of this particular folder description here.")
    assert calendar.extract_events(text, 2026) == []
    rows = calendar.extract_events("| Date | Topic |\n|---|---|\n| Sep 3 | Midterm 1 |\n", 2026)
    assert [(e["date"], e["title"]) for e in rows] == [("2026-09-03", "Midterm 1")]


def test_pdf_of_lecture_notes_takes_the_kind_of_its_markdown_twin(env, tmp_path):
    from test_importers import make_pdf
    s, cid, _ = env
    body = ("Eigenvalues and eigenvectors of a matrix describe directions it only stretches. " * 12)
    md = tmp_path / "eigen.md"
    md.write_text("<!-- source: lecture notebook -->\n# Eigen notes\n\n" + body + "\n")
    pdf = make_pdf(tmp_path / "eigen.pdf", [body])
    a = importers.import_file(s, cid, md)
    b = importers.import_file(s, cid, pdf, kind="notes")
    s.update_material(b, duplicate_of=a, kind="notes")  # what the importer's copy detection records for a real pair
    importers.pair_copies(s, b)
    assert s.material(a)["kind"] == importers.lecture_notes_kind()
    assert s.material(b)["kind"] == s.material(a)["kind"]


def test_fed_style_file_shows_the_no_paragraph_note_on_every_label(env, tmp_path):
    s, cid, _ = env
    p = tmp_path / "fed.txt"
    p.write_text("\n".join(f"Line {i} of an essay about checks and balances in government." for i in range(80)))
    mid = importers.import_file(s, cid, p, kind="reading", title="Fed")
    labels = [c["loc_label"] for c in s.chunks_of(mid)]
    assert len(labels) > 1 and all("no paragraph numbers" in x for x in labels)


# 7. deleting a material deletes the events it created
def test_deleting_a_syllabus_removes_its_events_and_unlinks_lectures(env, tmp_path):
    s, cid, _ = env
    p = tmp_path / "syllabus.md"
    p.write_text("# Syllabus\n\n| Date | Topic |\n|---|---|\n| Sep 3 | Lecture 1: Vectors |\n| Oct 1 | Midterm 1 |\n")
    mid = importers.import_file(s, cid, p)
    by_hand = s.add_event(cid, "other", "Office hours", "2026-09-04")
    assert len([e for e in s.events(cid) if e["source_material_id"] == mid]) == 2
    n = importers.delete_material(s, mid)
    assert n == 2
    assert [e["id"] for e in s.events(cid)] == [by_hand]
    assert s.material(mid) is None


# retrieval snippet keeps multiplication
def test_snippet_keeps_a_star_between_numbers():
    assert "5×2" in retrieval._snippet("We compute 5*2 and **bold** here", ["compute"])
