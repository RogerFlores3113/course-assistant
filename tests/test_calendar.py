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


# ---- fx-import: dedupe, cleanup, kinds, ranges, weeks ---------------------------------------------------------
from course_assistant.calendar import similar_event


def test_table_rows_get_clean_titles_and_kinds():
    ev = extract_events((__import__("pathlib").Path(__file__).parent / "fixtures" / "import" / "syllabus-table.md").read_text(), 2026)
    kinds = {e["title"]: e["kind"] for e in ev}
    assert not any("|" in t for t in kinds)
    assert any(t.startswith("Lecture 3") and k == "lecture" for t, k in kinds.items())
    assert any(t.startswith("Midterm 1") and k == "exam" for t, k in kinds.items())
    assert any(t.startswith("Problem Set 1") and k == "assignment" for t, k in kinds.items())
    final = [e for e in ev if e["title"].startswith("Final")][0]
    assert (final["date"], final["time"]) == ("2026-12-15", "09:00")
    assert any(e["date"] == "2026-11-26" and e["title"].startswith("No class") for e in ev)


def test_same_event_in_two_places_is_listed_once():
    ev = extract_events("Midterm 1: Oct 1\nMIDTERM 1 (in class, covers Lectures 1-8): Oct 1\nHW 2 due Sep 17\nHW 2 due Sep 17\n", 2026)
    assert len([e for e in ev if e["date"] == "2026-10-01"]) == 1
    assert len([e for e in ev if e["date"] == "2026-09-17"]) == 1


def test_time_range_and_due_vs_released():
    ev = extract_events("Final Exam: December 15, 2026, 9:00 AM to 11:00 AM\nHW 3 released Sep 10, due Sep 17\n", 2026)
    fin = by_title(ev, "Final")
    assert fin["time"] == "09:00"
    assert by_title(ev, "HW 3")["date"] == "2026-09-17"


def test_labs_are_not_exams():
    ev = extract_events("Lab 2: Atomic structure and flame tests: Sep 17\n", 2026)
    assert by_title(ev, "Lab 2")["kind"] != "exam"


def test_week_schedule_rows():
    ev = extract_events("Week 1 (Sep 2): Intro\nWeek 2 (Sep 9): Vectors\n", 2026)
    assert [e["date"] for e in ev] == ["2026-09-02", "2026-09-09"]


def test_similar_event_matches_reworded_duplicates():
    a = {"date": "2026-10-01", "title": "Midterm 1", "kind": "exam"}
    b = {"date": "2026-10-01", "title": "MIDTERM 1 (in class, covers Lectures 1-8)", "kind": "exam"}
    c = {"date": "2026-10-02", "title": "Midterm 1", "kind": "exam"}
    assert similar_event(a, b) and not similar_event(a, c)
