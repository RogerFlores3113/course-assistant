"""Original fixtures for the cycle 13 routing tests: one small made-up course, no sample course content."""
import json

from course_assistant import importers, integrity
from course_assistant.store import Store

SET = """# Problem Set 6

## Problem 1

How many grams of NaCl are needed to make 150.0 mL of a 0.600 M solution? Use the molar mass 58.44 g/mol.

## Problem 2

The matrix with rows (1, 1, 1), (1, 1, 3) and (2, 5, d) needs a row exchange at the second step. Find the value of d that makes the
system singular.

## Problem 3

Factor the matrix with rows (5, 2, 1), (10, -4, 3) and (-5, 8, 6) as A = LU. Report the multipliers and U, then solve Ly = b for
b = (7, 8, 9).

## Problem 4

Prove that a linear system Ax = b can never have exactly two solutions.
"""
KEYS = {"Problem 1": ["mass = 5.26 g", "moles = 0.0900 mol", "volume = 0.1500 L"],
        "Problem 2": ["d = 10"],
        "Problem 3": ["l21 = 2", "l31 = -1", "l32 = -1", "u22 = -8", "u23 = -2", "u33 = 1", "y2 = -12", "y3 = 2"]}
LECTURE = """# Lecture 5: Elimination by hand

## Worked example

Take the matrix with rows (7, 9) and (14, 20). Subtract 2 times row 1 from row 2 and the second row becomes (0, 2). The multiplier is 2,
and the upper triangular factor has rows (7, 9) and (0, 2).

## Where this fits

The LU factorization stores each multiplier below the diagonal. Lecture 6 covers the LU factorization in more depth.
"""
CHEM = """# Chapter 4: Solutions

## Mass of solute

To find the mass of a solute such as NaCl, multiply the volume in liters by the molarity to get moles, then multiply by the molar mass.
For example, 200.0 mL of a 0.250 M solution of KCl needs 0.0500 mol, which is 3.73 g at 74.55 g/mol.
"""


def world(tmp_path, keys=True):
    s = Store(":memory:")
    c = s.add_course("Course")
    f = tmp_path / "problem-set-6.md"
    f.write_text(SET)
    mid = importers.import_file(s, c, f)
    integrity.mark_material_graded(s, mid)
    ids = {}
    for it in s.graded_items(c):
        ids[it["label"]] = it["id"]
        if keys and it["label"] in KEYS:
            s.update_graded_item(it["id"], answer_key=json.dumps(KEYS[it["label"]]))
    for name, text in (("lecture-5", LECTURE), ("chapter-4", CHEM)):
        g = tmp_path / f"{name}.md"
        g.write_text(text)
        importers.import_file(s, c, g)
    return s, c, ids
