"""Original fixtures for the cycle 9 routing tests: one small made-up course, no sample course content."""
import json

from course_assistant import integrity
from course_assistant.store import Store

P1 = ("Problem 1 (10 points). Solve the system 2x + y + z = 5, x - y + 3z = 6, 3x + 2y - z = 3 for the vector "
      "(x, y, z).")
P2 = ("Problem 2 (10 points). Let the system be x + 2y + 7z = 3, 2x + 5y + 4z = 8, 3x + 7y + dz = 9. Find the value of d "
      "that makes the coefficient matrix singular.")
P3 = ("Problem 3 (10 points). An unknown ester is 54.5% carbon, 9.1% hydrogen and 36.4% oxygen by mass. Find its empirical "
      "formula.")
P4 = ("Problem 4 (10 points). Prove that if a linear system Ax = b has two different solutions, then it has infinitely "
      "many solutions.")
P5 = ("Problem 5 (10 points). Explain in your own words why a molecular formula can differ from an empirical formula, "
      "using one example you choose.")
ESS = ("Essay 1: What made the canal era matter\n\nIn about 700 words, argue which effect of the canal boom mattered most "
       "for the towns along the route. Use at least two sources from the reading list.\n\nRubric: thesis 30%, evidence 40%.")
KEYS = {"Problem 1": ["x = (1, 1, 2)", "x1 = 1", "x2 = 1", "x3 = 2"], "Problem 2": ["d = 11"],
        "Problem 3": ["C5H10O3", "n = 0.0454 mol"]}


def world(essay=True):
    s = Store(":memory:")
    c = s.add_course("Course")
    ids = {}
    for i, body in enumerate((P1, P2, P3, P4, P5), 1):
        kind = "proof" if i == 4 else "problem"
        ids[f"Problem {i}"] = s.add_graded_item(c, f"Problem Set 4: Problem {i}", body, None, kind=kind, label=f"Problem {i}",
                                                answer_key=json.dumps(KEYS.get(f"Problem {i}", [])) if f"Problem {i}" in KEYS else "")
    if essay:
        ids["Essay 1"] = s.add_graded_item(c, "Essay 1: What made the canal era matter", ESS, None, kind="essay", label="Essay 1")
    return s, c, ids
