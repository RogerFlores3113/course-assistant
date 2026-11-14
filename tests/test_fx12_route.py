"""Cycle 12 routing and context (fx12-route). Fixtures are original text. Each leak test sits beside an honest look-alike."""
import json

import pytest

from course_assistant import grounding, importers, integrity, study, web_helpers
from course_assistant.providers import FakeProvider
from course_assistant.store import Store

EIG = "Problem 5 (10 points). Find the eigenvalues of the matrix with rows (2, 3) and (3, -6)."
EIG_KEY = ["lambda1 = 3", "lambda2 = -7"]
WORDP = "Problem 7 (10 points). Explain in two sentences why a determinant of zero means the rows are dependent."
NOTES = """# Lecture 18: Eigenvalues

## Main idea

Eigenvalues are the numbers that tell you how a matrix stretches a direction. The matrix with rows 4 1 and 2 3 has eigenvalues 5 and 2.

## Method

To find the eigenvalues of a matrix, solve det(A - lambda I) = 0 and read off the roots. An eigenvector is only stretched.
"""


def eworld(tmp_path):
    s = Store(":memory:")
    c = s.add_course("Course")
    ids = {"eig": s.add_graded_item(c, "Set 2: Problem 5", EIG, None, kind="problem", label="Problem 5",
                                    answer_key=json.dumps(EIG_KEY))}
    f = tmp_path / "lecture-18.md"
    f.write_text(NOTES)
    importers.import_file(s, c, f)
    return s, c, ids


LECTURE_SUMMARY = ("## Main points\n\n- Eigenvalues are the topic the course builds on [1].\n"
                   "- Find the eigenvalues of a matrix by solving det(A - lambda I) = 0 [1].\n"
                   "- The matrix with rows 4 1 and 2 3 has eigenvalues 5 and 2 [1].\n")


COPY_TEXTS = ["Find the eigenvalues of the matrix with rows (2, 3) and (3, -6).",
              "1. Find the eigenvalues of the matrix with rows (2, 3) and (3, -6) and report both."]
HONEST_TEXTS = [
    "Eigenvalues are the topic the course builds on, and to find the eigenvalues of a matrix you solve a determinant.",
    "Find the eigenvalues of the matrix with rows (6, 1) and (2, 5).",
    "The matrix with rows 4 1 and 2 3 has eigenvalues 5 and 2, found from the characteristic equation.",
    "To find the eigenvalues of a matrix with rows of numbers, subtract lambda from the diagonal and take the determinant.",
]


@pytest.mark.parametrize("t", COPY_TEXTS)
def test_d1_a_restated_item_is_still_a_copy(t):
    it = {"kind": "problem", "body": EIG, "id": 1}
    assert integrity.copies_item(t, it, rows=True)


@pytest.mark.parametrize("t", HONEST_TEXTS)
def test_d1_topic_vocabulary_is_never_a_copy(t):
    it = {"kind": "problem", "body": EIG, "id": 1}
    assert not integrity.copies_item(t, it, rows=True)
    assert not integrity.copies_item(t, it)


def test_d1_explain_on_a_shared_topic_is_kept(tmp_path):
    s, c, ids = eworld(tmp_path)
    gid = study.explain(s, c, FakeProvider(LECTURE_SUMMARY), "eigenvalues")
    g = s.guide(gid)
    assert "Main points" in g["body_md"] or "Eigenvalues are the topic" in g["body_md"]
    assert not g["data"].get("scrubbed")


def test_d1_a_fully_held_back_answer_says_so_once_and_truthfully(tmp_path):
    s, c, ids = eworld(tmp_path)
    copy = "Find the eigenvalues of the matrix with rows (2, 3) and (3, -6)."
    gid = study.explain(s, c, FakeProvider(copy), "eigenvalues")
    g = s.guide(gid)
    assert g["data"].get("held_back") and g["data"].get("scrubbed")
    view = web_helpers.guide_view(g, store=s)
    assert not view["not_covered"]
    text = " ".join(b["html"] for b in view["blocks"])
    assert "did not return an answer" not in text.lower()
    assert text.lower().count("held back") == 1
    assert "Problem 5" in text
