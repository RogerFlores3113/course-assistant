"""Cycle 3 integrity: the answer key, key matching, precise routing, follow-ups, MCP, study guides and essays."""
import json
import sys
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from course_assistant import disclaimer, grounding, importers, integrity, mcp_server, study, web
from course_assistant.providers import FakeProvider
from course_assistant.store import Store

FIX = Path(__file__).parent / "fixtures" / "import"
TWIN_OK = ("<twin_problem>How many grams of KCl (molar mass 74.55 g/mol) make 500.0 mL of 0.200 M KCl?</twin_problem>"
           "<solution>Moles = 0.200 mol/L x 0.5000 L = 0.1000 mol, then mass = 0.1000 x 74.55 = 7.455 g. "
           "Apply these steps to your own problem.</solution>")
KEYS = {  # final values of the fixture problems, computed by hand
    ("problem-set-2", "Problem 1"): ["d = 4", "d = 14/3", "x = (-1/2, 5/2, 3/2)"],
    ("problem-set-2", "Problem 2"): ["x1 = (-2, 1, -1)", "x2 = (3, 0, 1)"],
    ("problem-set-2", "Problem 3"): ["L = [[1,0,0],[3,1,0],[0,2,1]]", "U = [[1,2,1],[0,2,-2],[0,0,5]]", "x = (2, 1, -2)"],
    ("problem-set-3", "Problem 1"): ["5.11 g"],
    ("problem-set-3", "Problem 2"): ["C3H4O3"],
}


def course(keyed=True):
    s = Store(":memory:")
    c = s.add_course("Course")
    mids = {}
    for n in ("problem-set-2", "problem-set-3", "essay-assignment-federalist"):
        mids[n] = importers.import_file(s, c, FIX / f"{n}.md")
        integrity.mark_material_graded(s, mids[n])
    if keyed:
        for it in s.graded_items(c):
            for (n, label), key in KEYS.items():
                if it["material_id"] == mids[n] and it["label"] == label:
                    s.update_graded_item(it["id"], answer_key=json.dumps(key))
    return s, c, mids


def item(s, c, mids, n, label):
    return next(i for i in s.graded_items(c) if i["material_id"] == mids[n] and i["label"] == label)


def model(leaks=None, default="Here is a plain answer."):
    """A 'blind' model: twin calls get a good twin, concepts calls coaching, every other call the matching leak."""
    leaks = leaks or {}

    def f(system, messages):
        if "<twin_problem>" in system and "Reply in exactly this format" in system:
            return TWIN_OK
        if "THE STUDENT'S GRADED TASK" in messages[0]["content"]:
            return "Review how the method works. Ask yourself which step you are unsure about and what the rubric asks."
        text = messages[-1]["content"]
        for k, v in leaks.items():
            if k in text:
                return v
        return default
    return f


# ------------------------------------------------------------------ C: key matching
@pytest.mark.parametrize("key,text,hit", [
    (["d = 10"], "so d is ten", True), (["d = 10"], "I have 10 apples and 10 pens", False),
    (["x = (1, 1, 2)"], "x = [1 1 2]^T", True), (["x = (1, 1, 2)"], "x is one, one, two", True),
    (["x = (1, 1, 2)"], "$x=\\begin{bmatrix}1\\\\1\\\\2\\end{bmatrix}$", True),
    (["x = (1, 1, 2)"], "x1 = 1, x2 = 1, x3 = 2", True), (["x = (1, 1, 2)"], "eigenvectors (1,1) and (1,-2)", False),
    (["5.11 g"], "roughly 5.1 g", False), (["5.11 g"], "5.110", True), (["5.11 g"], "5.2 g", False),
    (["C3H4O3"], "The empirical formula is C3H4O3", True), (["C3H4O3"], "C_3H_4O_3", True),
    (["x = -3/2"], "x = -1.5", True), (["L = [[1,0,0],[2,1,0]]"], "L = [1 0 0; 2 1 0]", True),
    (["x = (2, 1, -2)"], "x is two, one, minus two", True),
])
def test_key_hits_tolerates_formatting(key, text, hit):
    assert bool(integrity.key_hits(text, key)) is hit


def test_bare_small_integer_never_counts_alone():
    assert integrity.key_hits("It took 4 steps and 4 rows", ["d = 4"]) == []
    assert integrity.key_hits("then d = 4 follows", ["d = 4"]) == ["d = 4"]


# ------------------------------------------------------------------ B: making the key
def test_build_answer_key_stores_values_and_failures_leave_it_empty():
    s, c, m = course(keyed=False)
    it = item(s, c, m, "problem-set-3", "Problem 1")
    p = FakeProvider('Sure: ["5.11 g"]')
    assert integrity.build_answer_key(s, it["id"], p) == ["5.11 g"]
    assert integrity.item_key(s.graded_item(it["id"])) == ["5.11 g"]
    it2 = item(s, c, m, "problem-set-3", "Problem 2")
    assert integrity.build_answer_key(s, it2["id"], FakeProvider("no idea")) == []
    assert integrity.item_key(s.graded_item(it2["id"])) == []
    essay = item(s, c, m, "essay-assignment-federalist", "")
    assert integrity.build_answer_key(s, essay["id"], p) == []  # only numeric problems get a key


def test_ensure_keys_fills_missing_once():
    s, c, m = course(keyed=False)
    p = FakeProvider('["x = 1"]')
    n = integrity.ensure_keys(s, c, p)
    calls = len(p.calls)
    assert n == calls and n >= 4
    integrity.ensure_keys(s, c, p)
    assert len(p.calls) == calls  # already tried


def test_mcp_set_answer_key_and_list_graded_status(monkeypatch):
    s, c, m = course(keyed=False)
    mcp_server.set_store(s)
    try:
        it = item(s, c, m, "problem-set-3", "Problem 1")
        assert "answer check: off" in mcp_server.list_graded(c)
        out = mcp_server.set_answer_key(it["id"], ["5.11 g"])
        assert "5.11" not in out and "hidden" in out
        assert integrity.item_key(s.graded_item(it["id"])) == ["5.11 g"]
        assert "set_answer_key" in mcp_server.INSTRUCTIONS
    finally:
        mcp_server.set_store(None)


# ------------------------------------------------------------------ over-firing, with and without keys
EIGEN = ("The characteristic polynomial of A = [[4,1],[2,3]] is (4 - l)(3 - l) - 2 = l^2 - 7l + 10, so l = 5 and l = 2. "
         "The trace is 7 and the determinant is 10. Eigenvectors: (1,1) for 5 and (1,-2) for 2. 3 + 4 = 7, 5 x 2 = 10.")
NORMAL = {
    "How do I find the eigenvalues of [[4,1],[2,3]], and what are the eigenvectors?": EIGEN,
    "How do I compute A^100 quickly?": "Diagonalize A = P D P^-1, then A^100 = P D^100 P^-1 with D^100 = diag(5^100, 2^100) [1].",
    "What were the exam dates in the Fall 2018 offering and why do they differ from this term?":
        "In Fall 2018 the exams were on Oct 4, Nov 8 and Nov 30, 2018. This term the schedule has 3 exams in 2026.",
    "What score do I need to get an A?": "The syllabus says 90 of 100 points is an A, so you need 90.",
    "When is the final exam?": "The final exam is on Dec 12 at 8:00 AM.",
    "What is Charles's law and what's the formula for it?": "Charles's law says V1/T1 = V2/T2 at constant pressure, 3 L at 300 K.",
}


@pytest.mark.parametrize("keyed", [True, False])
@pytest.mark.parametrize("q", list(NORMAL))
def test_ordinary_questions_get_normal_answers_with_graded_work_marked(q, keyed):
    s, c, m = course(keyed)
    a = grounding.ask(s, c, q, FakeProvider(model({q: NORMAL[q]})))
    assert a.mode == "answer", q
    assert NORMAL[q].split(".")[0][:25] in a.body_md


@pytest.mark.parametrize("keyed", [True, False])
def test_catch_up_and_explain_with_small_integer_examples_are_not_replaced(keyed):
    s, c, m = course(keyed)
    mat = s.add_material(c, "notes", "Lecture 18 notes", lecture_date="2026-11-10")
    s.replace_chunks(mat, [{"text": "Eigenvalues of [[4,1],[2,3]] are 5 and 2.", "loc_kind": "section", "loc_start": 1,
                            "loc_label": "Eigenvalues"}])
    p = FakeProvider(f"## Main points\n\n{EIGEN} [1]\n\nWorked example: 3x + 4y = 5 gives x = 3, y = 4, z = 5 [1].")
    gid = study.missed(s, c, p, date="2026-11-10")
    g = s.guide(gid)
    assert g["kind"] == "missed" and "characteristic polynomial" in g["body_md"] and "left out" not in g["body_md"]
    gid = study.explain(s, c, FakeProvider(f"## Eigenvalues\n\n{EIGEN} [1]"), "eigenvalues and eigenvectors")
    g = s.guide(gid)
    assert g["kind"] == "explain" and "characteristic polynomial" in g["body_md"]


def test_ordinary_eigenvalue_cheat_sheet_saves_through_mcp():
    for keyed in (True, False):
        s, c, m = course(keyed)
        cid = s.add_material(c, "notes", "Eigen")
        s.replace_chunks(cid, [{"text": "eigen", "loc_kind": "section", "loc_start": 1, "loc_label": "e"}])
        chunk = s.chunks_of(cid)[0]["id"]
        mcp_server.set_store(s)
        try:
            out = mcp_server.save_guide(c, "explain", "Eigenvalue cheat sheet", EIGEN + " [1]", [chunk])
            assert out.startswith("Saved guide"), out
        finally:
            mcp_server.set_store(None)


def test_graded_chat_does_not_hijack_unrelated_questions():
    s, c, m = course()
    p = FakeProvider(model({"A^100": NORMAL["How do I compute A^100 quickly?"]}))
    first = grounding.ask(s, c, "Give me the answer to Problem 1 of Problem Set 2.", p)
    assert first.mode in ("twin", "concepts")
    for q in ("How do I compute A^100 quickly?", "When is the final exam?",
              "What were the exam dates in the Fall 2018 offering and why do they differ from this term?"):
        a = grounding.ask(s, c, q, p)
        assert a.mode == "answer", q


# ------------------------------------------------------------------ under-firing: follow-ups, check asks
@pytest.mark.parametrize("first,follow,leak,key", [
    ("Give me the answer to Problem 1 of Problem Set 2.", "and what is d?", "d = 4", "d = 4"),
    ("What are the empirical formula steps for Problem 2 of Problem Set 3?", "so formula?", "The formula is C3H4O3.", "C3H4O3"),
    ("problem 1 of problem set 3", "what's the final mass again?", "The final mass is 5.11 g.", "5.11"),
    ("problem 3 of problem set 2", "show me L", "L = [[1,0,0],[3,1,0],[0,2,1]]", "[[1,0,0]"),
])
def test_follow_up_after_a_twin_never_gets_the_originals_values(first, follow, leak, key):
    s, c, m = course()
    p = FakeProvider(model({follow: leak}))
    grounding.ask(s, c, first, p)
    a = grounding.ask(s, c, follow, p)
    assert a.mode in ("twin", "concepts")
    assert key not in a.body_md.replace(" ", "") or key in TWIN_OK


@pytest.mark.parametrize("q", ["I got 5.11 g for the molarity problem, right?",
                               "my L = [[1,0,0],[3,1,0],[0,2,1]] for the LU problem, is that right?"])
def test_check_answer_with_no_problem_number_never_confirms(q):
    s, c, m = course()
    a = grounding.ask(s, c, q, FakeProvider(model(default="Yes, that is correct.")))
    assert a.mode in ("twin", "concepts")
    assert "that is correct" not in a.body_md and "5.11" not in a.body_md and "[[1,0,0]" not in a.body_md
    assert "cannot say whether" in a.body_md


# ------------------------------------------------------------------ MCP save_guide and flashcards
PAD = "\n\nTo do this, eliminate below each pivot, record the multipliers, then back substitute from the last row."


@pytest.mark.parametrize("body", [
    "Answer for Problem 3: x = (2, 1, -2)." + PAD,
    "x is two, one, minus two." + PAD,
    "The empirical formula is C3H4O3" ,
    "Problem 3 result: the multipliers are 3, 0 and 2 and U = [[1,2,1],[0,2,-2],[0,0,5]]." + PAD,
])
def test_save_guide_refuses_padded_and_spelled_out_answers(body):
    s, c, m = course()
    mat = s.add_material(c, "notes", "n")
    s.replace_chunks(mat, [{"text": "t", "loc_kind": "section", "loc_start": 1, "loc_label": "t"}])
    mcp_server.set_store(s)
    try:
        out = mcp_server.save_guide(c, "answer", "LU notes", body, [s.chunks_of(mat)[0]["id"]])
        assert out.startswith("NOT SAVED"), out
    finally:
        mcp_server.set_store(None)


def test_save_guide_refusal_names_sentence_and_right_item_not_the_key():
    s, c, m = course()
    mcp_server.set_store(s)
    try:
        out = mcp_server.save_guide(c, "answer", "Notes", "Answer for Problem 3: x = (2, 1, -2)." + PAD, [])
        assert "Problem 3" in out and "Answer for Problem 3: x = (2, 1, -2)." in out
        assert "[[1,2,1]" not in out and "Problem 1" not in out
    finally:
        mcp_server.set_store(None)


def test_correct_twin_without_a_problem_statement_is_saved_and_header_says_your_ai_app():
    s, c, m = course()
    mcp_server.set_store(s)
    try:
        body = ("Eliminate below the first pivot of the matrix with rows (2, 1, 5), (4, 3, 8), (6, 8, 1). "
                "Then x = (3, -1, 2).")
        out = mcp_server.save_guide(c, "twin", "Twin of an LU problem", body, [])
        assert out.startswith("Saved guide"), out
        gid = int(out.split()[2].rstrip(":"))
        g = s.guide(gid)
        assert g["model"] == "your AI app" and "ai_app" not in g["model"]
    finally:
        mcp_server.set_store(None)


def test_mcp_add_flashcards_refuses_key_values():
    s, c, m = course()
    mcp_server.set_store(s)
    try:
        for back in ("d = 4", "x = (2, 1, -2)", "5.11 g", "C3H4O3"):
            out = mcp_server.add_flashcards(c, [{"front": "Problem 1 of PS2: what d forces the row exchange?", "back": back}])
            assert out.startswith("NOT ADDED"), (back, out)
        assert s.flashcards(c) == []
        assert mcp_server.add_flashcards(c, [{"front": "What is a pivot?", "back": "The first nonzero entry used to eliminate."}]
                                         ).startswith("Added")
    finally:
        mcp_server.set_store(None)


def test_web_flashcard_back_with_key_value_is_dropped():
    s, c, m = course()
    assert integrity.check_card(s, c, "Find L for A", "L = [[1,0,0],[3,1,0],[0,2,1]]")
    assert integrity.check_card(s, c, "Mass of NaCl?", "5.11 g")
    assert integrity.check_card(s, c, "Empirical formula?", "C3H4O3")
    assert integrity.check_card(s, c, "What is a pivot?", "The entry used to eliminate.") is None


# ------------------------------------------------------------------ study guides: replace only the leaking block
def study_model(leak):
    def f(system, messages):
        if "<twin_problem>" in system and "Reply in exactly this format" in system:
            return TWIN_OK
        if "THE STUDENT'S GRADED TASK" in messages[0]["content"]:
            return "Review the method. Ask yourself what each step needs."
        return leak
    return f


LEAKY_PRACTICE = ("## Practice\n\n1. Solve the system 5x + 2y = 9 and x - y = 1.\n\n<details>\n<summary>Show solution</summary>\n\n"
                  "x = 11/7, y = 4/7 [1]\n\n</details>\n\n2. Factor the matrix with rows (1, 2, 1), (3, 8, 1), (0, 4, 1).\n\n"
                  "<details>\n<summary>Show solution</summary>\n\nL = [[1,0,0],[3,1,0],[0,2,1]] and x = (2, 1, -2) [1]\n\n</details>")


def test_blind_practice_guide_drops_only_the_leaking_block():
    s, c, m = course()
    mat = s.add_material(c, "notes", "LU notes")
    s.replace_chunks(mat, [{"text": "LU factorization of a matrix by elimination.", "loc_kind": "section", "loc_start": 1,
                            "loc_label": "LU"}])
    g = s.guide(study.practice(s, c, FakeProvider(study_model(LEAKY_PRACTICE)), "LU factorization elimination", count=3))
    body = g["body_md"]
    assert "[[1,0,0]" not in body and "x = (2, 1, -2)" not in body
    assert "left out" in body and "x = 11/7" in body  # the clean question and its solution stay


def test_practice_twins_do_not_count_and_solution_is_hidden():
    s, c, m = course()
    mat = s.add_material(c, "notes", "LU notes")
    s.replace_chunks(mat, [{"text": "LU factorization of a matrix by elimination.", "loc_kind": "section", "loc_start": 1,
                            "loc_label": "LU"}])
    p = FakeProvider(study_model("1. A question.\n\n2. Another.\n\n3. A third."))
    g = s.guide(study.practice(s, c, p, "problem 3 of problem set 2 LU factorization", count=3))
    assert any("Write 3 practice questions" in call["messages"][0]["content"] for call in p.calls)
    assert "do not count" in g["body_md"]
    twin_part = g["body_md"].split("Twin of graded work")[1]
    assert "<details>" in twin_part and twin_part.index("<details>") < twin_part.index("Worked solution") + 400
    assert "0.1000 x 74.55" in twin_part.split("</details>")[0].split("<details>")[1]


def test_blind_exam_guide_replaces_the_leaking_block():
    s, c, m = course()
    ex = s.add_event(c, "exam", "Midterm", "2099-01-10")
    mat = s.add_material(c, "notes", "Notes", lecture_date="2098-12-01")
    s.replace_chunks(mat, [{"text": "Elimination.", "loc_kind": "section", "loc_start": 1, "loc_label": "E"}])
    leak = "## Practice set\n\nWorked: L = [[1,0,0],[3,1,0],[0,2,1]], x = (2, 1, -2).\n\n## Study plan\n\nReview each day."
    g = s.guide(study.exam_guide(s, c, FakeProvider(study_model(leak)), event_id=ex))
    assert "[[1,0,0]" not in g["body_md"] and "Review each day" in g["body_md"] and "left out" in g["body_md"]


def test_concept_map_drops_nodes_that_state_an_answer():
    s, c, m = course()
    mat = s.add_material(c, "notes", "Notes")
    s.replace_chunks(mat, [{"text": "Elimination.", "loc_kind": "section", "loc_start": 1, "loc_label": "E"}])
    graph = ('{"nodes":[{"id":"a","label":"Pivot","cites":[1]},{"id":"b","label":"x = (2, 1, -2)","cites":[1]}],'
             '"edges":[{"a":"a","b":"b","label":"gives","cites":[1]}]}\n\nSummary [1].')
    g = s.guide(study.concept_map(s, c, FakeProvider(graph)))
    assert [n["label"] for n in g["data"]["nodes"]] == ["Pivot"] and g["data"]["edges"] == []


# ------------------------------------------------------------------ E and G: routing sentence, ask normally, titles
def client_for(s, provider):
    s.set_setting(disclaimer.ACK_SETTING, "2026-01-01T00:00:00+00:00")
    cl = TestClient(web.create_app(s, key="k" * 40, provider_factory=lambda: provider), base_url="http://127.0.0.1:8766",
                    follow_redirects=False)
    cl.cookies.set(web.COOKIE, "k" * 40)
    return cl


def test_twin_page_data_has_routing_note_title_and_ask_normally_flag():
    s, c, m = course()
    a = grounding.ask(s, c, "Give me the answer to Problem 1 of Problem Set 2.", FakeProvider(model()))
    g = s.guide(a.guide_id)
    assert g["title"] == "Similar problem to Problem 1: Elimination and LU Factorization"
    assert g["data"]["routing_note"] == "This looks like your graded Problem 1, so here is a similar problem."
    assert g["data"]["not_covered"] is False and g["data"]["ask_normally"] is True


def test_ask_normally_turns_routing_off_but_keeps_key_checks():
    s, c, m = course()
    cl = client_for(s, FakeProvider(model(default="Problem 1 asks you to eliminate, then exchange rows if a pivot is zero.")))
    q = "Give me the answer to Problem 1 of Problem Set 2."
    cl.post(f"/courses/{c}/ask", data={"question": q, "normal": "1"})
    g = s.guides(c)[0]
    assert g["kind"] == "answer"
    cl2 = client_for(s, FakeProvider(model(default="Then d = 4 and x = (-1/2, 5/2, 3/2).")))
    cl2.post(f"/courses/{c}/ask", data={"question": q, "normal": "1"})
    assert s.guides(c)[0]["kind"] == "twin"  # the key check still caught the leak


# ------------------------------------------------------------------ F: essays
def essay_course():
    s, c, m = course()
    fed = s.add_material(c, "reading", "Federalist No. 78")
    s.replace_chunks(fed, [{"text": "Hamilton argues that judges should hold office during good behavior so they can resist "
                                    "the legislature and protect the Constitution.", "loc_kind": "section", "loc_start": 6,
                            "loc_label": "paragraph 6"}])
    return s, c, m


def test_essay_logistics_answer_does_not_cascade():
    s, c, m = essay_course()
    due = ("The Federalist essay is due Friday and should be 1,500 words. Late essays lose one third of a letter grade "
           "per day, and you should cite the textbook and the primary sources with page or paragraph references.")
    p = FakeProvider(lambda sy, me: due if "essay due" in me[-1]["content"] else
                     "Hamilton argued in Chapter 8 that the national bank was needed, while Jefferson opposed it [1].")
    a1 = grounding.ask(s, c, "When is the Federalist essay due and how long should it be?", p)
    assert a1.mode == "answer" and len(p.calls) == 1
    a2 = grounding.ask(s, c, "Summarize the Hamilton-Jefferson conflict from Chapter 8.", p)
    assert a2.mode == "answer" and len(p.calls) == 2 and "national bank" in a2.body_md


def test_two_part_question_keeps_the_fed_78_description():
    s, c, m = essay_course()
    reply = ("The essay is due Friday.\n\nHamilton argues that judges should hold office during good behavior so they can "
             "resist the legislature and protect the Constitution [1].")
    a = grounding.ask(s, c, "When is the Federalist essay due, and what does Federalist 78 say about why judges should "
                            "serve during good behavior?", FakeProvider(reply))
    assert a.mode == "answer" and "good behavior" in a.body_md


def test_the_essay_alone_is_not_a_writing_request():
    assert not integrity.writing_request("When is the Federalist essay due?")
    assert not integrity.writing_request("What does the rubric for the essay reward?")


DRAFT_Q = ('Here is my thesis: "Madison was wrong about faction because the Hamilton-Jefferson conflict showed that parties '
           'form anyway, so a large republic did not control faction." Is that a good thesis, and fix it up for me?')


def test_check_my_draft_gives_rubric_feedback_that_quotes_the_student():
    s, c, m = essay_course()
    good = ('Your thesis says "Madison was wrong about faction", which is a claim the rubric rewards (thesis 30 points). '
            "It does not yet say which part of Federalist No. 10 you test it against. Which passage would you cite, "
            "and what counterargument would a reader raise?")
    p = FakeProvider(lambda sy, me: good if "OWN draft" in sy else "ok")
    a = grounding.ask(s, c, DRAFT_Q, p)
    g = s.guide(a.guide_id)
    assert g["data"]["feedback"] is True and a.mode == "concepts"
    assert "Madison was wrong about faction" in a.body_md and "feedback, not a rewrite" in a.routing_note
    assert "Similar problem" not in g["title"] and g["title"].startswith("Feedback on your draft")


def test_check_my_draft_rejects_a_rewrite_and_falls_back():
    s, c, m = essay_course()
    rewrite = ("A stronger thesis would be: Madison's theory of faction failed because the first party system showed that "
               "ambition organised itself into parties despite the large republic.")
    p = FakeProvider(lambda sy, me: rewrite if "OWN draft" in sy else "ok")
    a = grounding.ask(s, c, DRAFT_Q, p)
    assert "Madison's theory of faction failed" not in a.body_md and "feedback against the rubric" in a.body_md
    assert len([x for x in p.calls if "OWN draft" in x["system"]]) == 2  # one regeneration, then the fixed note


def test_empty_model_reply_shows_no_general_knowledge_banner():
    s, c, m = course()
    a = grounding.ask(s, c, "What is a pivot?", FakeProvider(""))
    assert "did not return an answer" in a.body_md and a.not_covered is False
    assert s.guide(a.guide_id)["data"]["not_covered"] is False


# ------------------------------------------------------------------ logistics
def test_logistics_questions_use_the_assignment_header_not_a_concepts_page():
    s, c, m = course()
    seen = {}

    def f(system, messages):
        seen["prompt"] = messages[-1]["content"]
        return "Problem Set 2 is due Thursday September 17 and is worth 60 points [1]."

    a = grounding.ask(s, c, "When is Problem Set 2 due and how many points is it worth?", FakeProvider(f))
    assert a.mode == "answer" and "due Thursday" in a.body_md
    assert "Due Thursday September 17" in seen["prompt"] and "Gaussian elimination" not in seen["prompt"]
    assert "(15 points)" in seen["prompt"] or "Problem 4" in seen["prompt"]
    a = grounding.ask(s, c, "How many points is Problem 4 worth?", FakeProvider(f))
    assert a.mode == "answer"


# ------------------------------------------------------------------ labels and the split warning
def test_short_proof_is_labelled_proof_not_essay():
    s, c, m = course()
    p4 = item(s, c, m, "problem-set-2", "Problem 4")
    assert integrity._item_kind(p4) == "proof" and integrity.kind_label(p4) == "Proof"
    chem3 = item(s, c, m, "problem-set-3", "Problem 3")
    assert integrity.kind_label(chem3) == "Short answer"


def test_split_warning_clears_after_adding_the_missing_item_with_the_form():
    s, c, m = course()
    mid = m["problem-set-3"]
    gone = item(s, c, m, "problem-set-3", "Problem 3")
    s.delete_graded_item(gone["id"])
    assert integrity.split_report(s, mid)["short"]
    cl = client_for(s, None)
    assert "Found 3 of 4 problems" in cl.get(f"/courses/{c}").text
    r = cl.post(f"/courses/{c}/graded/new", data={"title": "Problem 3", "label": "Problem 3", "kind": "essay",
                                                   "body": gone["body"], "material_id": str(mid)})
    assert r.status_code in (302, 303)
    assert "Found 3 of 4" not in cl.get(f"/courses/{c}").text
    assert integrity.split_report(s, mid)["short"] is False


def test_hand_added_item_without_a_material_also_clears_the_warning():
    s, c, m = course()
    mid = m["problem-set-3"]
    gone = item(s, c, m, "problem-set-3", "Problem 3")
    s.delete_graded_item(gone["id"])
    s.add_graded_item(c, "Problem 3", gone["body"], kind="essay", label="Problem 3")
    assert integrity.split_report(s, mid)["short"] is False


LAB = """# Lab 4 handout

Due at the start of lab. Bring goggles.

## Procedure

Weigh the sample to the nearest 0.001 g and record it.

## Pre-lab questions

What mass of solute is needed to make 100.0 mL of a 0.250 M solution of a salt with molar mass 58.0 g/mol?

What is the molarity after diluting 10.00 mL of that solution to 100.0 mL?

## Report questions

Why must the final volume, not the solvent volume, be used when you compute molarity?
"""


def test_graded_lab_handout_splits_on_its_questions(tmp_path):
    s = Store(":memory:")
    c = s.add_course("Chem")
    f = tmp_path / "lab-4.md"
    f.write_text(LAB)
    mid = importers.import_file(s, c, f)
    ids = integrity.mark_material_graded(s, mid)
    labels = [s.graded_item(i)["label"] for i in ids]
    assert len(ids) == 3 and labels[0].startswith("Pre-lab questions")
    assert integrity.split_report(s, mid)["short"] is False


def test_unsplit_handout_with_several_questions_shows_the_warning(tmp_path):
    s = Store(":memory:")
    c = s.add_course("Chem")
    f = tmp_path / "quiz.md"
    f.write_text("# Take home quiz\n\nWhat is the molarity of 5.0 g NaCl in 250 mL of water?\n\n"
                 "Why does adding solvent lower molarity of the solution?\n\nHow would you prepare 100 mL of a 0.1 M solution?\n")
    mid = importers.import_file(s, c, f)
    integrity.mark_material_graded(s, mid)
    r = integrity.split_report(s, mid)
    assert r["short"] and r["expected"] == 3 and r["found"] == 1


def test_graded_panel_never_receives_the_key():
    s, c, m = course()
    cl = client_for(s, None)
    html = cl.get(f"/courses/{c}").text
    assert "14/3" not in html and "5.11" not in html and "Answer check: on" in html
