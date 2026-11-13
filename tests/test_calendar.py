from course_assistant.calendar import extract_events


def by_title(events, start):
    return next(e for e in events if e["title"].startswith(start))


SYLLABUS = """CS 101 Syllabus
Office hours: Tuesday 3-4pm in Room 204
Grading: 3/4 of your grade comes from exams. Homework is 20%.
Midterm 1: Thursday, Oct 9
Midterm 2 - 11/13/2026 at 7:30 pm
Final Exam: December 12, 2026, 9:00 AM
Week 3 (Sep 15)
Week 4 (Sep 22): Eigenvalues
HW 2 assigned Sep 10, due Sep 17
Quiz 1 on 9/24
Sep 15 | Lecture 3 | Gaussian elimination
Oct 13 - No class (fall break)
"""


def test_formats_and_kinds():
    ev = extract_events(SYLLABUS, 2026)
    m1 = by_title(ev, "Midterm 1")
    assert (m1["date"], m1["kind"]) == ("2026-10-09", "exam")
    m2 = by_title(ev, "Midterm 2")
    assert (m2["date"], m2["time"]) == ("2026-11-13", "19:30")
    final = by_title(ev, "Final Exam")
    assert (final["date"], final["time"], final["kind"]) == ("2026-12-12", "09:00", "exam")
    w3 = by_title(ev, "Week 3")
    assert (w3["date"], w3["kind"]) == ("2026-09-15", "lecture")
    assert by_title(ev, "Week 4")["title"] == "Week 4: Eigenvalues"
    hw = by_title(ev, "HW 2")
    assert (hw["date"], hw["kind"]) == ("2026-09-17", "assignment")  # the due date wins
    q = by_title(ev, "Quiz 1")
    assert (q["date"], q["kind"]) == ("2026-09-24", "quiz")
    row = by_title(ev, "Lecture 3")
    assert row["title"] == "Lecture 3: Gaussian elimination" and row["kind"] == "lecture"
    brk = by_title(ev, "No class")
    assert brk["kind"] == "other" and brk["title"] == "No class (fall break)"


def test_conservative_lines_are_skipped():
    ev = extract_events(SYLLABUS, 2026)
    titles = " ".join(e["title"] for e in ev)
    assert "Office" not in titles and "Grading" not in titles
    assert len(ev) == 9


def test_year_hint_and_new_year_rollover():
    ev = extract_events("Dec 5 | Lecture 20\nJan 12: Final project due\nNov 3 Quiz", "Fall 2026")
    assert [e["date"] for e in ev] == ["2026-12-05", "2027-01-12", "2026-11-03"]


def test_explicit_year_beats_hint_and_iso_dates():
    ev = extract_events("Exam on October 14, 2027\nHW 1 due 2026-09-30", 2026)
    assert ev[0]["date"] == "2027-10-14"
    assert ev[1]["date"] == "2026-09-30"


def test_title_on_previous_line_and_prose():
    ev = extract_events("Midterm review\nOct 12\nThe second midterm will be held on November 13 during class "
                        "time, and covers weeks 7 to 12. Please bring a calculator.", 2026)
    assert ev[0]["title"] == "Midterm review" and ev[0]["kind"] == "other"
    assert ev[1]["kind"] == "exam" and ev[1]["date"] == "2026-11-13"
    assert ev[1]["title"] == "Second midterm"


def test_invalid_dates_and_duplicates_dropped():
    ev = extract_events("Exam Feb 30\nQuiz 13/45\nQuiz 2: Oct 2\nQuiz 2: Oct 2", 2026)
    assert [e["title"] for e in ev] == ["Quiz 2"]


def test_empty():
    assert extract_events("") == []
    assert extract_events("No dates at all here.") == []
