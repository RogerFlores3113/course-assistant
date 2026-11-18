"""Original fixtures for the cycle 13 parse tests: a made-up studio course with paint, tile and a small matrix item.
No sample course content from any real subject."""
import json

from course_assistant import importers
from course_assistant.store import Store

PAINT = ("Problem 1 (10 points). A studio mixes a glaze that needs 0.425 g of pigment for every millilitre of base. The tank holds "
         "312.0 mL. Find the mass of pigment to weigh out and the amount of binder at 0.0850 mol per gram of pigment.")
PAINT_KEY = ["mass = 132.6 g", "binder = 11.27 mol"]
ROOTS = "Problem 2 (10 points). Find the eigenvalues of the 2 by 2 matrix with rows (6, 1) and (2, 5)."
ROOTS_KEY = ["lambda 1 = 7", "lambda 2 = 4"]
STEPS = ("Problem 3 (20 points). Factor the matrix with rows (2, 4, 6), (6, 5, 9) and (4, 22, 35) as A = LU and report the "
         "multipliers and U.")
STEPS_KEY = ["U = [[2, 4, 6], [0, -7, -9], [0, 0, 5]]", "l21 = 3", "l31 = 2", "l32 = -2"]
VECS = ("Problem 5 (10 points). Find the eigenvalues and eigenvectors of the 2 by 2 matrix with rows (6, 1) and (2, 5).")
VECS_KEY = ["lambda1 = 7", "lambda2 = 4", "v1 = (1, 1)", "v2 = (1, -2)"]
TILE = ("Problem 4 (10 points). A tile job multiplies a rate of 0.350 by a coverage of 0.2500 and a mass of 58.44. Report the product.")
TILE_KEY = ["product = 5.11"]

DECOY = """# Studio notes: shipping

## Crates

The crate for the kiln weighs 132.6 pounds when full, and the courier meets the truck at 11.27 past nine.
"""
LECT = """# Lecture 9: Characteristic polynomial

## Worked case

Take the array with first row (9, 2) and second row (2, 6). Its characteristic polynomial is lambda^2 - 15 lambda + 50 = 0,
which factors as (lambda - 10)(lambda - 5) = 0, so the eigenvalues are 10 and 5.
"""
UNIT = """# Studio notes: glaze

## Weights

A trial batch needed 132.6 g of pigment, and the binder amount was 11.27 mol for that batch.
"""


def world(tmp_path):
    s = Store(":memory:")
    c = s.add_course("Studio")
    ids = {}
    for label, body, key in (("Problem 1", PAINT, PAINT_KEY), ("Problem 2", ROOTS, ROOTS_KEY),
                             ("Problem 3", STEPS, STEPS_KEY), ("Problem 4", TILE, TILE_KEY),
                             ("Problem 5", VECS, VECS_KEY)):
        ids[label] = s.add_graded_item(c, f"Set 1: {label}", body, None, kind="problem", label=label, answer_key=json.dumps(key))
    mats = {}
    for name, text in (("decoy", DECOY), ("lecture-9", LECT), ("unit", UNIT)):
        f = tmp_path / f"{name}.md"
        f.write_text(text)
        mid = importers.import_file(s, c, f)
        mats[name] = [ch["id"] for ch in s.chunks_of(mid)]
    return {"s": s, "c": c, "ids": ids, "chunks": mats}
