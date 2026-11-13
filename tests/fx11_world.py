"""Original fixtures for the cycle 11 numeric tests: one small made-up course, no sample course content."""
import json

from course_assistant import importers, integrity
from course_assistant.store import Store

LU = ("Problem 9 (20 points). (a) Factor the matrix with rows (5, 2, 1), (10, -4, 3) and (-5, 8, 6) as A = LU. Report the "
      "multipliers and U. (b) Solve Ax = b for the vector b = (7, 8, 9) and report x.")
LU_KEY = ["U = [[2, 1, 1], [0, -8, -2], [0, 0, 1]]", "l21 = 2", "l31 = -1", "l32 = -1", "y = (5, -12, 2)", "x = (1, 1, 2)",
          "x1 = 1", "x2 = 1", "x3 = 2", "d = 11"]
EIG = "Problem 10 (10 points). Find the eigenvalues and eigenvectors of the 2 by 2 matrix with rows (4, 1) and (2, 3)."
EIG_KEY = ["lambda1 = 5", "lambda2 = 2", "v1 = (1, 1)", "v2 = (1, -2)"]
EIG2 = "Problem 11 (10 points). Find the eigenvalues and eigenvectors of the matrix with rows (1, 2) and (4, 3)."
EIG2_KEY = ["lambda1 = 5", "lambda2 = -1", "v1 = (1, 2)", "v2 = (1, -1)"]
CHEM = ("Problem 12 (10 points). How many grams of table salt do you need to make 0.0875 mol? Report the mass to three "
        "significant figures.")
CHEM_KEY = ["n = 0.0875 mol", "m = 5.11 g"]
PRELAB = "Pre-lab 1. How many grams of the hydrate are needed for 250.0 mL of a 0.0500 M solution?"
PRELAB_KEY = ["m = 2.497 g", "n = 0.0100 mol", "M = 249.68 g/mol"]

LECTURE = """# Lecture 4: Perspectives on products

## Perspective 3: rows times a matrix

Each row of the left factor combines the rows of the right factor. For example, with the left factor with rows (1, 2) and
(0, 3) and the right factor with rows (1, -1) and (3, 0), the first row of the product is 1 times row one plus 2 times
row two.
"""
LECTURE2 = """# Lecture 18: Eigenvectors

## Worked case

Take the array with first row (3, 1) and second row (2, 2). Its eigenvalues are 4 and 1. The eigenvector for 4 is (1, 1) while the
eigenvector for 1 is (1, -2).
"""
BOOK = """# Chapter 2: Solutions

## Worked example

A 355 mL soft drink holds 40.0 g of sucrose, which is 0.117 mol, so the molarity is 0.117 mol divided by 0.355 L, which is 0.330 M.

## Molar masses

The molar mass of sodium chloride is 58.44 g/mol and the molar mass of copper sulfate pentahydrate is 249.68 g/mol.
"""


def world(tmp_path):
    s = Store(":memory:")
    c = s.add_course("Course")
    ids = {}
    for label, body, key, kind in (("Problem 9", LU, LU_KEY, "problem"), ("Problem 10", EIG, EIG_KEY, "problem"),
                                   ("Problem 11", EIG2, EIG2_KEY, "problem"), ("Problem 12", CHEM, CHEM_KEY, "problem"),
                                   ("Pre-lab 1", PRELAB, PRELAB_KEY, "problem")):
        ids[label] = s.add_graded_item(c, f"Set 7: {label}", body, None, kind=kind, label=label, answer_key=json.dumps(key))
    mats = {}
    for name, text in (("lecture-4", LECTURE), ("lecture-18", LECTURE2), ("chapter-2", BOOK)):
        f = tmp_path / f"{name}.md"
        f.write_text(text)
        mid = importers.import_file(s, c, f)
        mats[name] = [ch["id"] for ch in s.chunks_of(mid)]
    return {"s": s, "c": c, "ids": ids, "chunks": mats}


QONLY = """# Lab 6 handout: Conductivity

## Pre-lab questions

A 0.500 L sample of brine holds 14.6 g of sodium chloride. Compute its molarity and say whether the conductivity should rise or fall when it is diluted to 1.00 L.

## Equipment

You will need a conductivity probe, a 250 mL beaker, a squirt bottle of distilled water and a lab notebook.
"""


def handout_only_questions(tmp_path):
    """A graded handout whose question paragraphs sit in a chunk of their own, so a search can hit the questions alone."""
    s = Store(":memory:")
    c = s.add_course("Course")
    f = tmp_path / "lab-6-handout.md"
    f.write_text(QONLY)
    mid = importers.import_file(s, c, f)
    s.update_material(mid, graded=1)
    integrity.mark_material_graded(s, mid)
    return s, c, mid
