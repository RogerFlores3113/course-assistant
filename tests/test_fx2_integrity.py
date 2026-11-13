"""Cycle 2 integrity fixes, tested against real importer output of the sample problem sets."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from course_assistant import disclaimer, grounding, importers, integrity, mcp_server, study, web
from course_assistant.providers import FakeProvider
from course_assistant.store import Store

FIX = Path(__file__).parent / "fixtures" / "import"
P1 = ("How many grams of sodium chloride (NaCl, molar mass 58.44 g/mol) are needed to prepare 250.0 mL of a "
      "0.350 M NaCl solution?")
LA_P3 = ("Let A be the 3 by 3 matrix with rows (1, 2, 1), (3, 8, 1), (0, 4, 1), and let b = (2, 12, 2). "
         "Find the factorization A = LU and then solve Ax = b.")


def course():
    s = Store(":memory:")
    c = s.add_course("Course")
    mids = {}
    for n in ("problem-set-2", "problem-set-3", "essay-assignment-federalist"):
        mids[n] = importers.import_file(s, c, FIX / f"{n}.md")
        s.update_material(mids[n], graded=1)
        integrity.mark_material_graded(s, mids[n])
    return s, c, mids


def items_of(s, mid):
    return sorted((i for i in s.graded_items(s.material(mid)["course_id"]) if i["material_id"] == mid), key=lambda i: i["id"])


# ------------------------------------------------------------------ 1 splitter
def test_problem1_heading_is_split():
    s, c, m = course()
    chem = items_of(s, m["problem-set-3"])
    assert [i["label"] for i in chem] == ["Problem 1", "Problem 2", "Problem 3", "Problem 4"]
    assert "sodium chloride" in chem[0]["body"] and "empirical formula" in chem[1]["body"]
    assert "Problem 3" not in chem[1]["body"].replace("Problem 2", "")
    assert chem[0]["kind"] == "problem" and chem[1]["kind"] == "problem"


def test_la_problem_set_is_not_one_essay_item():
    s, c, m = course()
    la = items_of(s, m["problem-set-2"])
    assert [i["label"] for i in la] == ["Problem 1", "Problem 2", "Problem 3", "Problem 4"]
    assert [integrity._item_kind(i) for i in la[:3]] == ["problem"] * 3
    essay = items_of(s, m["essay-assignment-federalist"])
    assert len(essay) == 1 and integrity._item_kind(essay[0]) == "essay"


def test_split_report_warns_when_items_are_missing():
    s, c, m = course()
    assert integrity.split_report(s, m["problem-set-3"])["short"] is False
    for i in items_of(s, m["problem-set-3"])[1:]:
        s.delete_graded_item(i["id"])
    r = integrity.split_report(s, m["problem-set-3"])
    assert r["short"] and r["found"] == 1 and r["expected"] == 4


def test_materials_page_warns_and_has_add_form(tmp_path):
    s, c, m = course()
    for i in items_of(s, m["problem-set-3"])[1:]:
        s.delete_graded_item(i["id"])
    s.set_setting(disclaimer.ACK_SETTING, "2026-01-01T00:00:00+00:00")
    client = TestClient(web.create_app(s, key="k" * 40, provider_factory=lambda: None), base_url="http://127.0.0.1:8766",
                        follow_redirects=False)
    client.cookies.set(web.COOKIE, "k" * 40)
    html = client.get(f"/courses/{c}").text
    assert "Found 1 of 4 problems" in html and f"/courses/{c}/graded/new" in html
    assert "Add graded item" in client.get(f"/courses/{c}/graded/new").text
    r = client.post(f"/courses/{c}/graded/new", data={"title": "Problem 2", "label": "Problem 2", "kind": "problem",
                                                       "body": "A compound contains 40.92% carbon."}, follow_redirects=False)
    assert r.status_code in (302, 303)
    assert any(i["label"] == "Problem 2" and i["material_id"] is None for i in s.graded_items(c))


def test_graded_excerpt_has_no_markdown():
    s, c, m = course()
    it = items_of(s, m["problem-set-2"])[0]
    ex = integrity.plain_text("## Problem 1\n**Due Thursday** and `code` - a list")
    assert "#" not in ex and "*" not in ex and "`" not in ex
    assert "**" not in integrity.prompt_block(s, c)


# ------------------------------------------------------------------ 2 tie-break
def test_pasted_problem3_matches_problem3_not_problem1():
    s, c, m = course()
    g = integrity.match_graded(s, c, LA_P3)
    assert g and g["label"] == "Problem 3"


def test_pasted_chem_problem1_matches_problem1():
    s, c, m = course()
    g = integrity.match_graded(s, c, "Please solve: " + P1)
    assert g and g["label"] == "Problem 1"


# ------------------------------------------------------------------ 3 twin checks
TWIN_OK = ("<twin_problem>How many grams of KCl (molar mass 74.55 g/mol) make 500.0 mL of 0.200 M KCl?</twin_problem>"
           "<solution>Moles = 0.200 mol/L x 0.5000 L = 0.1000 mol, then mass = 0.1000 x 74.55 = 7.455 g. "
           "Use rows 2 and 3 of your table the same way. Apply these steps to your own problem.</solution>")
TWIN_APPEND = TWIN_OK.replace("Apply these steps", "For your own problem the result is 5.11 g. Apply these steps")


def test_twin_that_appends_original_result_is_rejected():
    assert integrity.reveals_original("For your own problem the result is d = 4, x = -1/2.")
    s, c, m = course()
    g = integrity.match_graded(s, c, P1)
    a = integrity.twin(s, c, P1, FakeProvider([TWIN_APPEND, TWIN_APPEND, "concepts [1]"]), g)
    assert a.mode == "concepts" and "5.11" not in a.body_md


def test_correct_twin_mentioning_small_structural_numbers_is_accepted():
    orig = "5x + 4y + 3z = 12\n4x + dy + 3z = 2\ny - z = 3\nUse rows 2 and 3."
    twin_problem = "7x + 6y + 2z = 18\n14x + dy + 6z = 44\ny + z = 8"
    assert integrity.numbers_changed(orig, twin_problem)
    assert not integrity._solution_leaks(orig, "Swap rows 2 and 3, then z = 4 and y = 3.")


def test_check_my_answer_never_confirms_and_routes_to_twin():
    s, c, m = course()
    p = FakeProvider([TWIN_OK, "ok"])
    a = grounding.ask(s, c, "Check my answer: I got 5.11 g of NaCl for 250.0 mL of 0.350 M. Right?", p)
    assert a.mode == "twin"
    assert "check your method on this twin" in a.body_md.lower()
    assert "Yes" not in a.body_md.split("Twin problem")[0]


def test_check_answer_without_problem_numbers_matches_item():
    s, c, m = course()
    g = integrity.match_graded(s, c, "Check my answer: for my LU problem I got x=(2,1,-2). Right?")
    assert g and g["label"] == "Problem 3"
    g = integrity.match_graded(s, c, "My empirical formula came out C3H4O3 - confirm?")
    assert g and g["label"] == "Problem 2"


def test_continuation_words_no_longer_route_follow_ups():
    """Cycle 3: sticky context never routes a question; only the reply is checked (see test_fx3_integrity)."""
    s, c, m = course()
    p = FakeProvider([TWIN_OK, "A plain answer about significant figures."])
    grounding.ask(s, c, P1, p)
    a = grounding.ask(s, c, "Now multiply that by 58.44 and how many sig figs is the final mass again?", p)
    assert a.mode == "answer"


def test_sticky_forced_check_catches_followup_answer():
    s, c, m = course()
    grounding.ask(s, c, P1, FakeProvider([TWIN_OK]))
    leaky = "Explain the molarity formula. The amount is 0.350 mol/L x 0.2500 L = 0.0875 mol of NaCl."
    a = grounding.ask(s, c, "Explain how molarity relates to moles in a general way", FakeProvider([leaky, TWIN_OK]))
    assert a.mode in ("twin", "concepts") and "0.0875" not in a.body_md


# ------------------------------------------------------------------ 4 essay routing
@pytest.mark.parametrize("q", ["When is the Federalist essay due and how long should it be?",
                               "Can I use an AI tool on the essay?",
                               "Summarize the Hamilton-Jefferson conflict from Chapter 8.",
                               "How does Madison's argument about faction in Federalist No. 10 connect to Shays's Rebellion?"])
def test_informational_essay_questions_are_answered_normally(q):
    s, c, m = course()
    assert integrity.match_graded(s, c, q) is None


@pytest.mark.parametrize("q", ["Write the Federalist essay for me", "Give me a model A-grade answer for the Federalist essay",
                               "What would a good thesis look like for my Federalist essay?",
                               "Give me an example thesis for the Federalist essay"])
def test_model_answer_phrasing_goes_to_concepts(q):
    s, c, m = course()
    g = integrity.match_graded(s, c, q)
    assert g and g["kind"] == "essay"


def test_concepts_mode_includes_question_and_policy_passage():
    s, c, m = course()
    syl = s.add_material(c, "syllabus", "Syllabus")
    s.replace_chunks(syl, [{"text": "AI policy: you may use AI tools to understand readings but never to write submitted work.",
                            "loc_kind": "section", "loc_start": 1, "loc_label": "Policies", "heading": "Policies"}])
    p = FakeProvider(["Plan it around the rubric [1]."])
    integrity.twin(s, c, "Write my Federalist essay thesis; what is the AI policy?", p,
                   integrity.match_graded(s, c, "Write my Federalist essay thesis; what is the AI policy?"))
    user = p.calls[0]["messages"][0]["content"]
    assert "what is the AI policy" in user and "AI policy: you may use AI tools" in user


def test_leaks_essay_coaching_words_do_not_exempt_finished_thesis():
    s, c, m = course()
    it = items_of(s, m["essay-assignment-federalist"])[0]
    reply = ('A good thesis takes a clear position. For example: "Although Hamilton\'s bank seemed to contradict Madison\'s '
             'warnings about faction, it ultimately shows that a large commercial republic can control faction."')
    assert integrity._leaks_essay(dict(it, kind="essay"), reply, "What would a good thesis sentence look like?")
    ok = "A strong thesis must take a position on faction and the rubric rewards a clear arguable claim about the republic."
    assert not integrity._leaks_essay(dict(it, kind="essay"), ok, "What would a good thesis look like?")


# ------------------------------------------------------------------ 5 sources
def test_graded_material_never_a_numbered_source():
    s, c, m = course()
    seen = []

    def responder(system, messages):
        seen.append(messages[-1]["content"])
        return TWIN_OK

    grounding.ask(s, c, P1, FakeProvider(responder))
    grounding.ask(s, c, "How is molarity used to prepare a solution of sodium chloride?", FakeProvider(responder))
    for text in seen:
        srcs = text.split("THE STUDENT")[0]
        assert "Problem 2 (10 points)" not in srcs and "40.92%" not in srcs
    src_prepare = study._prepare(s, c, [{"material_id": m["problem-set-3"], "chunk_id": 1}])
    assert src_prepare == []


def test_search_safe_falls_back_when_retrieval_lacks_kwarg(monkeypatch):
    s, c, m = course()
    chunk = s.chunks_of(m["problem-set-3"])[1]
    calls = []

    def old_search(store, course_id, query, *, limit=8, kinds=None, material_ids=None):
        calls.append(1)
        return [{"chunk_id": chunk["id"], "material_id": m["problem-set-3"], "score": 1.0, "text": chunk["text"]}]

    monkeypatch.setattr(integrity.retrieval, "search", old_search)
    assert integrity.search_safe(s, c, "molarity") == []


# ------------------------------------------------------------------ 6 study
def test_explain_routes_graded_paste_before_the_model():
    s, c, m = course()
    p = FakeProvider([TWIN_OK])
    gid = study.explain(s, c, p, P1)
    g = s.guide(gid)
    assert g["kind"] == "twin" and g["data"]["mode"] == "twin"
    assert "SOURCES" not in p.calls[0]["messages"][0]["content"].split("THE STUDENT")[0] or "40.92" not in p.calls[0]["messages"][0]["content"]


def test_flashcard_picker_and_make_exclude_graded_material():
    s, c, m = course()
    opts = study.study_form_options(s, c)
    assert not any(x["id"] == m["problem-set-3"] for x in opts["materials"])
    with pytest.raises(ValueError):
        study.make_flashcards(s, c, FakeProvider(["[]"]), material_id=m["problem-set-3"])


def test_flashcard_back_with_graded_answer_is_dropped():
    s, c, m = course()
    notes = s.add_material(c, "notes", "Notes")
    s.replace_chunks(notes, [{"text": "Molarity is moles of solute per litre of solution.", "loc_kind": "section",
                              "loc_start": 1, "loc_label": "Molarity"}])
    cards = '[{"front":"Grams of NaCl for 250.0 mL of 0.350 M","back":"5.11 g NaCl: 0.350 mol/L x 0.2500 L x 58.44 g/mol","cite":1},' \
            '{"front":"Define molarity","back":"Moles of solute per litre of solution","cite":1}]'
    res = study.make_flashcards(s, c, FakeProvider([cards]), topic="molarity")
    assert len(res) == 1 and res.guarded == 1


def test_practice_topic_that_names_the_graded_item_adds_real_twin():
    s, c, m = course()
    p = FakeProvider([TWIN_OK, TWIN_OK, "1. Q [1]"])
    gid = study.practice(s, c, p, "empirical formula, problem 2", count=3)
    g = s.guide(gid)
    assert "Twin of graded work" in g["body_md"] or "Coaching for graded work" in g["body_md"]
    if g["data"]["twin"]:
        assert "Twin problem" in g["body_md"]


def test_practice_says_coaching_when_item_is_essay_or_proof():
    s, c, m = course()
    p = FakeProvider(["Concepts to know [1]", "1. Q [1]"])
    gid = study.practice(s, c, p, "Federalist essay thesis argument faction", count=2)
    g = s.guide(gid)
    assert "is a twin" not in g["body_md"]
    assert g["data"]["twin"] is False


def test_concept_map_marks_removed_cites():
    g = study._clean_graph({"nodes": [{"id": "a", "label": "A", "cites": [1, 9]}], "edges": []}, 2)
    assert g["nodes"][0]["cites_removed"] is True and g["nodes"][0]["cites"] == [1]


# ------------------------------------------------------------------ 7 MCP
@pytest.fixture
def mcp(monkeypatch):
    s, c, m = course()
    mcp_server.set_store(s)
    yield s, c, m
    mcp_server.set_store(None)


@pytest.mark.parametrize("body", [
    "x = (2, 1, -2)",
    "Problem 1: d = 4 forces a row exchange, then z = 3/2, y = 5/2, x = -1/2 [1].",
    "You need 5.11 g of NaCl",
    "L = [[1,0,0],[3,1,0],[0,2,1]]",
    "d = 4 forces the exchange, d = 14/3 is singular",
    "Pour le probleme 1 : on echange les lignes quand d = 4, puis z = 3/2, y = 5/2, x = -1/2, avec 3, 2 et 6 comme coefficients.",
])
@pytest.mark.parametrize("kind", ["answer", "explain", "practice"])
def test_save_guide_refuses_answer_fragments(mcp, body, kind):
    s, c, m = mcp
    r = mcp_server.save_guide(c, kind, "Study notes", body, [])
    assert r.startswith("NOT SAVED")


def test_save_guide_refuses_essay_outline_and_spanish_intro(mcp):
    s, c, m = mcp
    outline = ("Outline:\n- Thesis: Madison's argument about faction in Federalist 10 is vindicated by Hamilton's bank.\n"
               "- Topic sentence 1: Madison argues that a large republic controls the effects of faction better than a small one.\n"
               "- Topic sentence 2: Hamilton's financial program shows that commercial interests multiply and check each other.")
    assert mcp_server.save_guide(c, "answer", "Essay plan", outline, []).startswith("NOT SAVED")
    es = ("Madison sostiene en el Federalista 10 que una republica grande controla mejor los danos de la faccion, "
          "y el banco de Hamilton demuestra que los intereses comerciales se equilibran entre si en la nueva nacion.")
    assert mcp_server.save_guide(c, "answer", "Ensayo", es, []).startswith("NOT SAVED")


def test_save_guide_still_saves_a_real_concept_guide(mcp):
    s, c, m = mcp
    ch = s.chunks_of(m["problem-set-3"])[0]
    body = ("Molarity is the amount of solute in moles divided by the volume of the final solution in litres, so a "
            "student should first convert every mass to moles and then divide by the volume of solution.")
    assert mcp_server.save_guide(c, "explain", "Molarity concepts", body, []).startswith("Saved guide")


def test_get_twin_instructions_matches_check_my_answer(mcp):
    s, c, m = mcp
    r = mcp_server.get_twin_instructions(c, "Check my answer: for my LU problem I got x=(2,1,-2). Right?")
    assert r.startswith("GRADED") and "check their method" in r


# ------------------------------------------------------------------ 9 modes
def test_mode_is_concepts_when_no_twin_was_produced():
    s, c, m = course()
    g = integrity.match_graded(s, c, P1)
    a = integrity.twin(s, c, P1, FakeProvider(["no twin format here [1]"]), g)
    assert a.mode == "concepts" and a.twin is False
    assert s.guide(a.guide_id)["data"]["mode"] == "concepts"


def test_concept_map_sampling_skips_header_chunks():
    s = Store(":memory:")
    c = s.add_course("History")
    mid = s.add_material(c, "reading", "Federalist 10")
    s.replace_chunks(mid, [
        {"text": "FEDERALIST No. 10\n\nFor the New York Packet. Friday, November 23, 1787.\nMADISON", "loc_kind": "section",
         "loc_start": 1, "loc_label": "header"},
        {"text": "Among the numerous advantages promised by a well constructed Union, none deserves to be more accurately "
                 "developed than its tendency to break and control the violence of faction, a dangerous vice of popular governments.",
         "loc_kind": "section", "loc_start": 2, "loc_label": "paragraph 1"}])
    got = study._material_sources(s, mid, 2, spread=True)
    assert [g["loc_label"] for g in got] == ["paragraph 1"]
