"""Original fixtures for the cycle 10 routing tests: one small made-up course, no sample course content."""
import json

from course_assistant import importers, integrity
from course_assistant.store import Store

# a keyed elimination item whose answer includes the triangular system, the multipliers and the solution
ELIM = ("Problem 6 (10 points). Eliminate downward on 3x + 4y + z = 15, 6x + 9y + 3z = 33, -3x + 2y + 8z = 9. Report the "
        "triangular system, the multipliers and the solution.")
ELIM_KEY = ["U = [[3, 4, 1], [0, 1, 1], [0, 0, 3]]", "x = (3, 1, 2)", "l21 = 2", "l31 = -1", "l32 = 6"]
ELIM2 = ("Problem 7 (10 points). Solve 5a + 2b = 19 and a - 3b = -4 by elimination. Report the multiplier and a and b.")
ELIM2_KEY = ["m = 1/5", "a = 3", "b = 2"]

PRELAB_KEY = ["m = 3.12 g", "n = 0.0125 mol", "M = 249.68 g/mol"]
LAB = """# Lab 8 handout: Copper sulfate standards

## Materials

Copper sulfate pentahydrate (molar mass 249.68 g/mol), a 250.0 mL volumetric flask, a spectrometer and five cuvettes.

## Pre-lab questions

Pre-lab 1. How many grams of copper sulfate pentahydrate do you need for 250.0 mL of a 0.0500 M solution?

## Report questions

Report 1. Plot absorbance against concentration for your five standards and say what the slope tells you about the unknown.
"""
BOOK = """# Chapter 2: Solutions

## Worked example

A 0.250 L broth sample holds 0.0875 mol of urea, so the molarity is 0.0875 mol divided by 0.250 L, which is 0.350 M.

## Molar masses

The molar mass of urea is 60.06 g/mol and the molar mass of glucose is 180.16 g/mol.
"""


def world(tmp_path=None, lab=False, prelab_key=True):
    s = Store(":memory:")
    c = s.add_course("Course")
    ids = {}
    ids["Problem 6"] = s.add_graded_item(c, "Set 9: Problem 6", ELIM, None, kind="problem", label="Problem 6",
                                         answer_key=json.dumps(ELIM_KEY))
    ids["Problem 7"] = s.add_graded_item(c, "Set 9: Problem 7", ELIM2, None, kind="problem", label="Problem 7",
                                         answer_key=json.dumps(ELIM2_KEY))
    out = {"s": s, "c": c, "ids": ids}
    if lab and tmp_path is not None:
        f = tmp_path / "lab-8-handout.md"
        f.write_text(LAB)
        mid = importers.import_file(s, c, f)
        s.update_material(mid, graded=1)
        integrity.mark_material_graded(s, mid)
        g = tmp_path / "chapter-2-solutions.md"
        g.write_text(BOOK)
        bid = importers.import_file(s, c, g)
        for it in s.graded_items(c):
            if it["material_id"] == mid and "Pre-lab" in (it.get("title") or ""):
                if prelab_key:
                    s.update_graded_item(it["id"], answer_key=json.dumps(PRELAB_KEY))
                ids["Pre-lab 1"] = it["id"]
            elif it["material_id"] == mid and "Report" in (it.get("title") or ""):
                ids["Report 1"] = it["id"]
        out.update(handout=mid, book=bid, book_chunks=[ch["id"] for ch in s.chunks_of(bid)],
                   handout_chunks=[ch["id"] for ch in s.chunks_of(mid)])

    return out
