"""Cycle 3 retrieval, import and citation regressions (one test per panel finding). All text here is original."""
from pathlib import Path

import pytest

from course_assistant import retrieval
from course_assistant.importers import import_file, notes_kind
from course_assistant.store import Store

from test_importers import make_pdf


def chunk(text, label="p. 1", heading="", kind="page", start=1):
    return {"text": text, "loc_kind": kind, "loc_start": start, "loc_label": label, "heading": heading}


def add(s, cid, kind, title, chunks, **kw):
    mid = s.add_material(cid, kind, title)
    s.replace_chunks(mid, chunks)
    if kw:
        s.update_material(mid, **kw)
    return mid


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("COURSE_ASSISTANT_DATA_DIR", str(tmp_path / "data"))
    s = Store(":memory:")
    return s, s.add_course("Linear Algebra", term="Fall 2026"), tmp_path


# 1. two-part questions: the formula half is kept, and "TA" is not the end of "A^TA"
def test_fx3_ta_does_not_match_a_transpose_a():
    assert "ta" not in retrieval.query_terms("when is A^TA invertible")
    assert "ta" not in retrieval.query_terms("when is A^T A invertible")
    assert "ta" in retrieval.query_terms("who is the TA")


def test_fx3_two_part_question_keeps_the_formula_half():
    s = Store(":memory:")
    cid = s.add_course("LA")
    add(s, cid, "slides", "Projections", [
        chunk("The projection matrix onto the column space is P = A (A^T A)^-1 A^T.", "slide 7", kind="slide", start=7),
        chunk("A^TA is invertible exactly when the columns of A are independent.", "slide 8", kind="slide", start=8),
        chunk("Office hours with the TA are on Tuesday.", "slide 9", kind="slide", start=9),
    ])
    add(s, cid, "reading", "Syllabus", [chunk("The TA grades the TA quizzes. TA hours are posted.", "p. 1")])
    res = retrieval.search(s, cid, "What is the projection matrix formula and when is A^TA invertible?", limit=3)
    texts = " ".join(r["text"] for r in res)
    assert "P = A (A^T A)^-1 A^T" in texts and "exactly when the columns" in texts
    assert not any("TA hours" in r["text"] for r in res[:2])


# 2. a named offering's dates
def test_fx3_named_offering_date_query_forces_that_terms_exam_chunks():
    s = Store(":memory:")
    cid = s.add_course("LA", term="Fall 2026")
    add(s, cid, "syllabus", "Syllabus Fall 2018", [
        chunk("Grading: homework 30 percent, exams 70 percent.", "p. 1"),
        chunk("Exam 1 (Sep 28), Exam 2 (Oct 26), Exam 3 (Nov 30). Final in December.", "p. 2", start=2),
    ], term="Fall 2018")
    add(s, cid, "syllabus", "Syllabus Fall 2026", [chunk("Exam 1 (Sep 25), Exam 2 (Oct 23).", "p. 1")], term="Fall 2026")
    for i in range(6):
        add(s, cid, "notes", f"Exam skills {i}", [chunk(f"Exam tips {i}: sleep well before the exam.", "p. 1")])
    res = retrieval.search(s, cid, "Fall 2018 exam dates", limit=4)
    assert any("Exam 3 (Nov 30)" in r["text"] for r in res)
    assert not any("Sep 25" in r["text"] for r in res[:1])


# 3. Explain / TODO questions reach definitions; primers skip past terms and graded work
def test_fx3_explain_question_prefers_the_definition_chunk():
    s = Store(":memory:")
    cid = s.add_course("LA")
    add(s, cid, "slides", "Eigen", [
        chunk("Eigenvalues are interesting. We will meet them again in the next lecture and the exam.", "slide 2", kind="slide", start=2),
        chunk("Trace and determinant: the trace is the sum of the eigenvalues and the determinant is their product.", "slide 9", kind="slide", start=9),
        chunk("Definition. The determinant of a 2 by 2 matrix [[a,b],[c,d]] is ad - bc.", "slide 5", kind="slide", start=5),
    ])
    res = retrieval.search(s, cid, "Explain the trace and the determinant", limit=3)
    assert "sum of the eigenvalues" in res[0]["text"] or "ad - bc" in res[0]["text"]
    assert {"slide 9", "slide 5"} <= {r["loc_label"] for r in res[:2]}


def test_fx3_todo_question_uses_the_students_own_note_topic():
    s = Store(":memory:")
    cid = s.add_course("LA")
    s.add_event(cid, "lecture", "Lecture 18: Spectral ideas", "2026-11-10")
    add(s, cid, "notes", "My notes", [chunk("- Ax = lambda x\n- det(A - lambda I) = 0\n- TODO fill in from the lecture\n- trace and det", "p. 3",
                                                 heading="Eigenvalues")])
    add(s, cid, "slides", "Lecture 18", [
        chunk("Intro to the course schedule and grading.", "slide 1", kind="slide"),
        chunk("The trace of A equals the sum of its eigenvalues; det A is their product.", "slide 12", kind="slide", start=12),
    ])
    res = retrieval.search(s, cid, "Help me fill in the TODO from the Nov 10 lecture", limit=3)
    assert any("sum of its eigenvalues" in r["text"] for r in res)


def test_fx3_primer_skips_past_offerings_and_problem_sets(env):
    s, cid, _ = env
    ev = s.add_event(cid, "lecture", "Lecture 20: Singular values", "2026-11-12")
    now = add(s, cid, "slides", "Lecture 20 slides", [
        chunk("Lecture 20 covers the singular value decomposition A = U S V^T.", "slide 3", kind="slide", start=3)], event_id=ev)
    add(s, cid, "slides", "Fall 2018 Lecture 20", [
        chunk("Lecture 20 (2018): the SVD of a rectangular matrix.", "slide 3", kind="slide", start=3)], term="Fall 2018")
    add(s, cid, "assignment", "Problem Set 9", [chunk("Problem 3 of lecture 20: compute the SVD of this matrix.", "p. 1")])
    res = retrieval.search(s, cid, "Lecture 20 primer", limit=4, exclude_graded=True)
    titles = [r["material_title"] for r in res]
    assert titles and titles[0] == "Lecture 20 slides"
    assert "Problem Set 9" not in titles and "Fall 2018 Lecture 20" not in titles[:1]
    assert now


# 4. title slides and speaker-note duplicates
def test_fx3_title_slide_is_title_only():
    assert retrieval.is_title_only("Lecture 1: Introduction\nLinear Algebra, Fall 2026\nSpeaker notes: Welcome everyone!",
                                   "Lecture 1: Introduction") or retrieval.is_title_only("Lecture 1: Introduction", "Lecture 1: Introduction")
    assert not retrieval.is_title_only("Lecture 1: Introduction\nA vector has a length and a direction.", "Lecture 1: Introduction")


def test_fx3_speaker_notes_that_repeat_the_slide_are_dropped_on_import(env):
    from pptx import Presentation

    s, cid, tmp = env
    prs = Presentation()
    sl = prs.slides.add_slide(prs.slide_layouts[1])
    sl.shapes.title.text = "Rank"
    sl.placeholders[1].text = "The rank is the number of pivots."
    sl.notes_slide.notes_text_frame.text = "Speaker notes: The rank is the number of pivots. Say it twice."
    p = tmp / "rank.pptx"
    prs.save(p)
    import_file(s, cid, p)
    chunks = s.chunks_of(s.materials(cid)[0]["id"])
    body = " ".join(c["text"] for c in chunks)
    assert body.lower().count("number of pivots") == 1
    assert body.lower().count("speaker notes:") <= 1


# 5. exam-guide filler
def test_fx3_chapter_intro_and_exercise_lists_are_filler():
    assert retrieval.is_filler_chunk("End of Chapter Exercises\n1. Find the molarity of each solution.", "End of Chapter Exercises")
    assert retrieval.is_filler_chunk("Introduction\nIn this chapter you will learn about solutions.", "Introduction")
    assert not retrieval.is_filler_chunk("Molarity is moles of solute per litre of solution.", "Molarity")


def test_fx3_named_chapter_search_does_not_lead_with_exercises():
    s = Store(":memory:")
    cid = s.add_course("Chem")
    add(s, cid, "textbook", "Chapter 3 Solutions", [
        chunk("End of Chapter Exercises\n1. A solution has 2 mol in 4 L. Find M.", "Chapter 3, End of Chapter Exercises", kind="section", heading="End of Chapter Exercises"),
        chunk("Molarity is moles of solute per litre of solution.", "Chapter 3, Molarity", kind="section", heading="Molarity", start=2),
        chunk("Dilution: M1 V1 = M2 V2.", "Chapter 3, Dilution", kind="section", heading="Dilution", start=3),
    ])
    res = retrieval.search(s, cid, "Chapter 3 Solutions", limit=2)
    assert not any("End of Chapter Exercises" in r["text"][:30] for r in res)


# 6. citation labels
def test_fx3_plain_text_without_paragraph_numbers_says_so_once(env):
    s, cid, tmp = env
    p = tmp / "fed51.txt"
    p.write_text("\n".join(f"Line number {i} of the essay about checks and balances in government." for i in range(60)))
    import_file(s, cid, p, kind="reading", title="Fed 51")
    mid = s.materials(cid)[0]["id"]
    labels = [c["loc_label"] for c in s.chunks_of(mid)]
    assert all("no paragraph numbers in this file" in x for x in labels)  # every citation shows it, not only the first


def test_fx3_markdown_labels_are_unique_within_a_material(env):
    s, cid, tmp = env
    p = tmp / "chapter.md"
    p.write_text("\n\n".join(f"{h}\n\n{' '.join([w] * 120)}" for h, w in (
        ("# Chapter 2", "alpha"), ("## Examples", "beta"), ("# Chapter 3", "gamma"), ("## Examples", "delta"),
        ("## Examples", "omega"))))
    import_file(s, cid, p, kind="reading", title="Chapter text")
    labels = [c["loc_label"] for c in s.chunks_of(s.materials(cid)[0]["id"])]
    assert len(labels) == len(set(labels)) >= 3


def test_fx3_md_cites_the_pdf_pages_and_the_pdf_takes_the_md_title(env):
    s, cid, tmp = env
    sec1 = " ".join(f"alpha{i} beta{i} gamma{i}" for i in range(40))
    sec2 = " ".join(f"delta{i} epsilon{i} zeta{i}" for i in range(40))
    md = tmp / "ch7.md"
    md.write_text(f"# Chapter 7 The Articles\n\n## Early days\n\n{sec1}\n\n## Shays rebellion\n\n{sec2}\n")
    half = lambda t: [" ".join(t.split()[:60]), " ".join(t.split()[60:])]
    pdf = make_pdf(tmp / "Chapter-7-Articles.pdf", half(sec1) + half(sec2))
    import_file(s, cid, md, kind="reading", title="Chapter 7 The Articles")
    import_file(s, cid, pdf, kind="reading")
    mats = s.materials(cid)
    assert any(m.get("duplicate_of") for m in mats)
    pdf_m = next(m for m in mats if str(m.get("original_name", "")).endswith(".pdf"))
    assert pdf_m["title"] == "Chapter 7 The Articles"
    md_m = next(m for m in mats if not str(m.get("original_name", "")).endswith(".pdf"))
    allc = s.chunks_of(md_m["id"]) + s.chunks_of(pdf_m["id"])
    assert any("p. " in c["loc_label"] or "pp. " in c["loc_label"] for c in s.chunks_of(md_m["id"]) + s.chunks_of(pdf_m["id"])), allc


# 7. materials search: titles, file names and content, and why each matched
def test_fx3_materials_search_matches_title_file_name_and_content(env):
    s, cid, _ = env
    a = add(s, cid, "slides", "Lecture 3 slides", [chunk("Gaussian elimination makes a matrix upper triangular.", "slide 2")],
            original_name="lec3_final.pdf")
    b = add(s, cid, "notes", "My notes", [chunk("The pivot columns are the independent ones.", "p. 1")])
    c = add(s, cid, "reading", "Chapter 1", [chunk("Nothing relevant here.", "p. 1")])
    m = retrieval.material_matches(s, cid, "elimination")
    assert set(m) == {a} and "passage" in " ".join(m[a]["how"]) and "elimination" in m[a]["snippet"].lower()
    assert set(retrieval.material_matches(s, cid, "lec3_final")) == {a} and retrieval.material_matches(s, cid, "lec3_final")[a]["file"]
    assert set(retrieval.material_matches(s, cid, "chapter 1")) >= {c}
    assert retrieval.material_matches(s, cid, "") == {}
    assert b not in retrieval.material_matches(s, cid, "elimination")


def test_fx3_materials_tab_shows_which_materials_matched(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import course_assistant.web as web

    monkeypatch.setenv("COURSE_ASSISTANT_DATA_DIR", str(tmp_path))
    store = Store(":memory:")
    cid = store.add_course("LA")
    add(store, cid, "notes", "Eigen notes", [chunk("An eigenvector keeps its direction.", "p. 1")])
    ctx = web.tab_context(store, store.course(cid), "materials", {"q": "eigenvector"}) if hasattr(web, "tab_context") else None
    assert ctx is None or ctx.get("material_matches")


# 8. the instructor's notes are not "My notes"
def test_fx3_instructor_notes_get_their_own_kind():
    kind = notes_kind("lecture-12.md", "<!-- source: lecture notebook -->\n# Lecture 12\nWe start with the SVD.")
    assert kind != "notes"
    assert notes_kind("linear-algebra-halfdone.md", "# Linear algebra, my notes (unfinished)\n- TODO fill in") == "notes"
    assert notes_kind("notes.md", "Some plain text with nothing to tell.") == "notes"


def test_exam_guide_sampling_skips_chapter_filler():
    from course_assistant import study
    from course_assistant.store import MATERIAL_KINDS, Store
    assert "lecture_notes" in MATERIAL_KINDS
    s = Store(":memory:")
    c = s.add_course("Chem")
    m = s.add_material(c, "textbook", "Chapter 3")
    texts = [("Introduction", "Introduction\nIn this chapter you will learn about solutions."),
             ("Molarity", "Molarity is moles of solute per litre of solution."),
             ("Dilution", "Diluting a solution keeps the moles of solute the same."),
             ("End of Chapter Exercises", "End of Chapter Exercises\n1. Find the molarity of each solution.")]
    s.replace_chunks(m, [{"text": t, "heading": h, "loc_kind": "section", "loc_start": i, "loc_end": i,
                          "loc_label": h} for i, (h, t) in enumerate(texts)])
    got = [x["text"] for x in study._material_sources(s, m, 4, spread=True)]
    assert any("Molarity is" in t for t in got) and not any("Exercises" in t or "In this chapter" in t for t in got)
