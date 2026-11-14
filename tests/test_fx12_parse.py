"""Cycle 12 parse fixes (decisions 6, 7 and the zero rule of 8): each panel repro next to its honest look-alike.
Fixtures are original (fx12_world)."""
import pytest

import fx12_world as W
from course_assistant import integrity, mcp_server


@pytest.fixture
def w(tmp_path):
    x = W.world(tmp_path)
    mcp_server.set_store(x["s"])
    yield x
    mcp_server.set_store(None)


def item(x, label):
    return next(i for i in integrity.graded_view(x["s"], x["c"]) if i["label"] == label)


def sticky(x, label):
    return dict(item(x, label), reason="sticky")


def save(x, body, chunk="lecture-4"):
    return mcp_server.save_guide(x["c"], "explain", "Notes", body, [x["chunks"][chunk][0]])


def src(x, name):
    return [{"text": x["s"].chunk(x["chunks"][name][0])["text"], "graded": False}]


# ------------------------------------------------------------ decision 6: numbers named in the same sentence
NAMED = ["For Problem 5, the eigenvector for 5 is (1,1).", "Eigenvector for Problem 5: (1,1).",
         "For Problem 5, the eigenvector for 5 is (1, 1)", "In Problem 5 the eigenvector for 2 is (1, -2)."]


@pytest.mark.parametrize("text", NAMED)
def test_d6_ask_hits_the_eigenvector_named_in_the_same_sentence(w, text):
    s, c = w["s"], w["c"]
    hit = integrity.check_reply(s, c, "how do I start Problem 5?", text, force_item=sticky(w, "Problem 5"))
    assert hit and hit["label"] == "Problem 5"
    # a fresh chat that never named it: the sentence names it, so it is judged as named
    hit = integrity.check_reply(s, c, "what next?", text)
    assert hit and hit["label"] == "Problem 5"


@pytest.mark.parametrize("text", NAMED)
def test_d6_mcp_save_guide_refuses_the_named_eigenvector(w, text):
    assert save(w, text).startswith("NOT SAVED")


def test_d6_look_alikes_pass(w):
    s, c = w["s"], w["c"]
    for t in ["For Problem 5, the characteristic polynomial is a quadratic in lambda.",
              "For Problem 5, an eigenvector is a nonzero vector with A v = lambda v.",
              "In Problem 3, the eigenvector idea does not come up, only elimination.",
              "For Problem 5, the eigenvector for 3 is (2, 7)."]:  # numbers that are not the key's
        assert integrity.check_reply(s, c, "how do I start Problem 5?", t, force_item=sticky(w, "Problem 5")) is None, t
        assert save(w, t).startswith("Saved"), t
    # another item's name in the sentence does not blame Problem 5
    assert integrity.key_leak("For Problem 3, the eigenvector is (1, 1).", item(w, "Problem 5"), {}) == []


def test_d6_card_what_are_the_eigenvalues_is_refused_again(w):
    for f, b in [("What are the eigenvalues of A?", "5 and 2"), ("What are the eigenvalues of the matrix?", "5, 2"),
                 ("Eigenvalues of A", "5 and 2")]:
        r = mcp_server.add_flashcards(w["c"], [{"front": f, "back": b}])
        assert r.startswith("NOT ADDED"), (f, b)


def test_d6_card_look_alikes_pass(w):
    for f, b in [("How many eigenvalues does a 2 by 2 matrix have?", "2"), ("What is the eigenvalue of the identity matrix?", "1"),
                 ("Number of eigenvalues of a 2 by 2 matrix?", "two")]:
        assert mcp_server.add_flashcards(w["c"], [{"front": f, "back": b}]).startswith("Added"), f
    # the cited chunk shows that binding: the card is the lecture's own example
    r = mcp_server.add_flashcards(w["c"], [{"front": "What are the eigenvalues of A?", "back": "5 and 2",
                                            "chunk_id": w["chunks"]["lecture-18"][0]}])
    assert r.startswith("Added")
    # a chunk that only has the bare digits does not bless it
    r = mcp_server.add_flashcards(w["c"], [{"front": "What are the eigenvalues of A?", "back": "5 and 2",
                                            "chunk_id": w["chunks"]["lecture-4"][0]}])
    assert r.startswith("NOT ADDED")


def test_d6_number_words_are_judged_like_the_digit_form(w):
    # the lecture chunk says: two eigenvalues, five and two. A cited paraphrase in words or digits saves.
    for body in ("The instructor says the matrix has eigenvalues five and two [1]", "The matrix has eigenvalues 5 and 2 [1]",
                 "The eigenvalues are five and two [1]", "The eigenvalues of that array are 5 and 2 [1]"):
        assert save(w, body, "lecture-18").startswith("Saved"), body
    # laundering: the same sentence cited to a chunk that does not print that binding is refused, words or digits
    for body in ("The instructor says the matrix has eigenvalues five and two [1]", "The eigenvalues are 5 and 2 [1]"):
        assert save(w, body, "lecture-4").startswith("NOT SAVED"), body
    # a different binding than the chunk shows is refused even when cited to the right chunk
    assert save(w, "The eigenvalues are 4 and 3 [1]", "lecture-18").startswith("Saved")  # neither is the key's: not a leak


def test_d6_ask_cited_paraphrase_in_words(w):
    s, c = w["s"], w["c"]
    r = "The lecture example has eigenvalues five and two [1]."
    assert integrity.check_reply(s, c, "what does the lecture example say?", r, force_item=sticky(w, "Problem 5"),
                                 sources=src(w, "lecture-18")) is None
    hit = integrity.check_reply(s, c, "what are the eigenvalues?", r, force_item=sticky(w, "Problem 5"), sources=src(w, "lecture-4"))
    assert hit and hit["label"] == "Problem 5"


# ------------------------------------------------------------ decision 7: save_guide shapes
SLASH = ["U = 2 1 1 / 0 -8 -2 / 0 0 1. [1]", "U = 2 1 1 / 0 -8 -2 / 0 0 1 [1]", "U = 2 1 1 / 0 -8 -2 / 0 0 1 [1].",
         "So U is 2 1 1 / 0 -8 -2 / 0 0 1. [1]"]


@pytest.mark.parametrize("text", SLASH)
def test_d7_slash_rows_with_the_period_before_the_marker_are_refused(w, text):
    assert save(w, text).startswith("NOT SAVED")


def test_d7_slash_rows_parse_with_a_final_full_stop():
    h = integrity._Hay("U = 2 1 1 / 0 -8 -2 / 0 0 1. Next")
    assert [x for _, x in h.seqs if x == ["0", "0", "1"]]


def test_d7_slash_row_look_alikes_save(w):
    # a chunk that prints the same rows: the lecture's own example is data, with or without the period
    for body in ("The lecture reduces it to 2 1 1 / 0 -8 -2 / 0 0 1. [1]", "The lecture reduces it to 2 1 1 / 0 -8 -2 / 0 0 1 [1]"):
        assert save(w, body, "lecture-6").startswith("Saved"), body
    # other rows are not the key
    assert save(w, "A small example is 3 1 / 0 2. [1]").startswith("Saved")
    assert save(w, "Another small example is 4 1 / 2 3 / 0 1. [1]").startswith("Saved")


def test_d7_headerless_table_row_with_a_key_value_is_a_hit(w):
    for body in ("| 11 | singular | [1]", "| d | effect |\n|---|---|\n| 11 | singular | [1]", "| 11 | singular |"):
        assert save(w, body).startswith("NOT SAVED"), body


def test_d7_table_look_alikes_save(w):
    for body in ("| 12 | singular | [1]", "| 11 | Lecture 11 | [1]", "| pivot | meaning |\n|---|---|\n| nonzero | can divide | [1]",
                 "| 3 | pivots | [1]"):
        assert save(w, body).startswith("Saved"), body


# ------------------------------------------------------------ decision 8: the word zero is never a hit by itself
ZERO = ["A row exchange is needed when a pivot position becomes zero.", "The third pivot being zero means no unique solution.",
        "If the first pivot is zero, swap rows.", "The second entry is zero here and the first pivot is zero too."]


@pytest.mark.parametrize("text", ZERO)
def test_d8_zero_never_matches_a_zero_key_entry(w, text):
    s, c = w["s"], w["c"]
    for q in ("when do I need a row exchange?", "what does that mean for Problem 3?"):
        assert integrity.check_reply(s, c, q, text, force_item=sticky(w, "Problem 3")) is None, (q, text)
    assert save(w, text + " [1]").startswith("Saved")
    assert integrity.key_leak(text, item(w, "Problem 3"), {"named": True, "sticky": True, "short": True, "question_refs": True}) == []


def test_d8_a_bound_zero_is_still_a_hit(w):
    s, c = w["s"], w["c"]
    for t in ("c1 = 0", "c1 = zero", "The first multiplier c1 is 0."):
        hit = integrity.check_reply(s, c, "what is c1?", t, force_item=sticky(w, "Problem 3"))
        assert hit and hit["label"] == "Problem 3", t
