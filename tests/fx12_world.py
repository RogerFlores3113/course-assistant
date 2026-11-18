"""Original fixtures for the cycle 12 parse tests: one small made-up course, no sample course content."""
import json

from course_assistant import importers
from course_assistant.store import Store

LU = ("Problem 3 (20 points). (a) Factor the matrix with rows (5, 2, 1), (10, -4, 3) and (-5, 8, 6) as A = LU. Report the "
      "multipliers and U. (b) Solve Ax = b for the vector b = (7, 8, 9) and report x. (c) For which value of d does the system with the "
      "last pivot replaced by d become singular?")
LU_KEY = ["U = [[2, 1, 1], [0, -8, -2], [0, 0, 1]]", "l21 = 2", "l31 = -1", "l32 = -1", "y = (5, -12, 2)", "x = (1, 1, 2)",
          "x1 = 1", "x2 = 1", "x3 = 2", "d = 11", "c1 = 0"]
EIG = "Problem 5 (10 points). Find the eigenvalues and eigenvectors of the 2 by 2 matrix with rows (4, 1) and (2, 3)."
EIG_KEY = ["lambda1 = 5", "lambda2 = 2", "v1 = (1, 1)", "v2 = (1, -2)"]

LECTURE = """# Lecture 4: Perspectives on products

## Quokka recap

The group tried three settings. Run A used 1, run B used 1 and run C used 2 of the small weights, and every run ended
the same way, so the counts alone say little.
"""
LECTURE2 = """# Lecture 18: Eigenvectors

## Worked case

Take the array with first row (3, 1) and second row (2, 2). It has two eigenvalues, five and two. The eigenvector for 5 is (1, 1) while the
eigenvector for 2 is (1, -2).
"""
LECTURE3 = """# Lecture 6: Elimination

## Worked case

Eliminating below the first pivot gives the upper triangular matrix with rows (2, 1, 1), (0, -8, -2) and (0, 0, 1).
The determinant of that case is 11, so the system is singular only when d is 0.
"""


def world(tmp_path):
    s = Store(":memory:")
    c = s.add_course("Course")
    ids = {}
    for label, body, key in (("Problem 3", LU, LU_KEY), ("Problem 5", EIG, EIG_KEY)):
        ids[label] = s.add_graded_item(c, f"Set 2: {label}", body, None, kind="problem", label=label, answer_key=json.dumps(key))
    mats = {}
    for name, text in (("lecture-4", LECTURE), ("lecture-18", LECTURE2), ("lecture-6", LECTURE3)):
        f = tmp_path / f"{name}.md"
        f.write_text(text)
        mid = importers.import_file(s, c, f)
        mats[name] = [ch["id"] for ch in s.chunks_of(mid)]
    return {"s": s, "c": c, "ids": ids, "chunks": mats}
