"""Real modules wired together: importers -> retrieval -> grounding/integrity/study -> web and MCP.

No sys.modules stand-ins. The data folder is a tmp dir; the AI is providers.FakeProvider; the network is never used.
"""
import json
import re
from datetime import date, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from mcp.server.fastmcp.exceptions import ToolError

from course_assistant import (disclaimer, grounding, importers, integrity, mcp_server, offerings, paths, retrieval,
                              study, web)
from course_assistant.providers import FakeProvider
from course_assistant.store import Store

KEY = "k" * 40


def future(n: int) -> str:
    return (date.today() + timedelta(days=n)).isoformat()


# ------------------------------------------------------------------ fixture builders
def make_pdf(path: Path, pages: list[str]) -> Path:
    objs: list[bytes] = [b"<< /Type /Catalog /Pages 2 0 R >>"]
    n = len(pages)
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(n))
    objs.append(f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode())
    font_id = 3 + 2 * n
    for i, text in enumerate(pages):
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {4 + 2 * i} 0 R "
                    f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>".encode())
        stream = f"BT /F1 12 Tf 14 TL 50 740 Td ({text}) Tj ET".encode()
        objs.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, o in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + o + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n".encode() + b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))
    return path


def make_pptx(path: Path) -> Path:
    from pptx import Presentation

    prs = Presentation()
    s1 = prs.slides.add_slide(prs.slide_layouts[1])
    s1.shapes.title.text = "Markov chains"
    s1.placeholders[1].text = "A stationary distribution satisfies pi P = pi."
    s1.notes_slide.notes_text_frame.text = "Mention the transition matrix is stochastic."
    prs.save(str(path))
    return path


LECTURE_MD = """# Eigenvalues

An eigenvalue of a square matrix A is a number lambda with det(A - lambda I) = 0. The eigenvector keeps its
direction when A acts on it, and lambda tells how much it is stretched. This is the core idea of the second unit.

# Diagonalization

A matrix is diagonalizable when A = P D P inverse, where D holds the eigenvalues and P holds the eigenvectors.
Diagonalization makes powers of A easy to compute.
"""

VTT = """WEBVTT

00:00:01.000 --> 00:00:09.000
Today we cover the singular value decomposition and why orthogonal matrices preserve length.

00:01:30.000 --> 00:01:45.000
The Gram-Schmidt process builds an orthonormal basis one vector at a time.
"""

def syllabus_md() -> str:
    return (f"# Syllabus\n\nLinear Algebra course policies.\n\n"
            f"- {future(3)} Lecture: Eigenvalues and eigenvectors\n"
            f"- {future(10)} Lecture: Diagonalization\n"
            f"- {future(20)} Midterm 1 exam\n"
            f"- {future(12)} Homework 3 due\n")


GRADED_PROBLEM = ("Find the eigenvalues of the matrix with rows 7 3 and 3 7 and then compute the determinant of "
                  "A minus 11 times the identity for the 4 by 4 case.")


@pytest.fixture()
def world(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_DATA_DIR, str(tmp_path / "data"))
    store = Store(":memory:")
    cid = store.add_course("Linear Algebra", code="MATH 2210", university="Example U", term="Fall 2026")
    src = tmp_path / "src"
    src.mkdir()
    (src / "lecture 1 slides.md").write_text(LECTURE_MD)
    ids = {}
    ids["md"] = importers.import_file(store, cid, src / "lecture 1 slides.md", lecture_date=future(3))
    ids["pdf"] = importers.import_file(store, cid, make_pdf(src / "textbook chapter 2.pdf", [
        "Orthogonal matrices have orthonormal columns", "The spectral theorem says symmetric matrices diagonalize",
        "Positive definite matrices have positive eigenvalues"]))
    ids["pptx"] = importers.import_file(store, cid, make_pptx(src / "lecture 4 slides.pptx"))
    (src / "lecture 5 captions.vtt").write_text(VTT)
    ids["vtt"] = importers.import_file(store, cid, src / "lecture 5 captions.vtt")
    (src / "syllabus.md").write_text(syllabus_md())
    ids["syl"] = importers.import_file(store, cid, src / "syllabus.md")
    return store, cid, ids, tmp_path


def smart_provider(twin_numbers=None):
    """A FakeProvider that answers by the kind of prompt it gets."""
    twin_calls = []

    def respond(system, messages):
        user = messages[-1]["content"]
        if "THE STUDENT'S GRADED PROBLEM" in user:
            twin_calls.append(user)
            nums = (twin_numbers or [["7", "3"], ["19", "5"]])[min(len(twin_calls) - 1, 1)]
            return (f"<twin_problem>\nFind the eigenvalues of the matrix with rows {nums[0]} {nums[1]} and {nums[1]} {nums[0]}.\n"
                    "</twin_problem>\n<solution>\nSolve det(A - lambda I) = 0 [1].\n\nThen read off the roots [1].\n</solution>")
        if "JSON list" in user:
            return json.dumps([{"front": "What is an eigenvalue?", "back": "A lambda with det(A - lambda I) = 0.", "cite": 1},
                               {"front": "What is diagonalization?", "back": "Writing A = P D P^-1.", "cite": 1}])
        if "concept map" in user.lower():
            return json.dumps({"nodes": [{"id": "eig", "label": "Eigenvalues", "cites": [1]},
                                         {"id": "diag", "label": "Diagonalization", "cites": [1]}],
                               "edges": [{"a": "eig", "b": "diag", "label": "leads to"}]}) + "\n\nBoth ideas connect [1]."
        return ("## The big picture\n\nEigenvalues solve det(A - lambda I) = 0 [1].\n\n"
                "Euler studied rotations long before this course and that is general history with no source.")

    p = FakeProvider(respond)
    p.twin_calls = twin_calls
    return p


# ------------------------------------------------------------------ 1. import and retrieval
def test_importers_produce_chunks_and_syllabus_events(world):
    store, cid, ids, _ = world
    assert store.material(ids["md"])["kind"] == "slides"
    for key in ids:
        assert store.chunks_of(ids[key]), key
    assert [c["loc_kind"] for c in store.chunks_of(ids["pdf"])] == ["page"] * 3
    notes = " ".join(c["text"] for c in store.chunks_of(ids["pptx"]))
    assert "stationary distribution" in notes and "transition matrix is stochastic" in notes
    vtt = store.chunks_of(ids["vtt"])
    assert vtt[0]["loc_kind"] == "time" and "singular value" in vtt[0]["text"]
    assert store.material(ids["syl"])["kind"] == "syllabus"
    events = store.events(cid)
    kinds = {e["kind"] for e in events}
    assert {"lecture", "exam"} <= kinds
    assert all(e["added_by"] == "syllabus" and e["source_material_id"] == ids["syl"] for e in events)


def test_retrieval_finds_right_chunk_and_grounded_answer_resolves(world):
    store, cid, ids, _ = world
    hits = retrieval.search(store, cid, "Gram-Schmidt orthonormal basis")
    assert hits[0]["material_id"] == ids["vtt"] and hits[0]["loc_kind"] == "time"
    hits = retrieval.search(store, cid, "stationary distribution")
    assert hits[0]["material_id"] == ids["pptx"]
    provider = smart_provider()
    ans = grounding.ask(store, cid, "What is an eigenvalue?", provider)
    assert ans.sources and ans.used == [1]
    assert ans.outside and any(b.kind == "outside" for b in ans.blocks)
    assert "**Not from your materials:**" in ans.body_md and disclaimer.FOOTER in ans.body_md
    g = store.guide(ans.guide_id)
    assert g["kind"] == "answer" and [b["kind"] for b in g["data"]["blocks"]] == ["text", "text", "outside"] or \
        "outside" in [b["kind"] for b in g["data"]["blocks"]]
    assert [m["role"] for m in store.chat_messages(cid)] == ["user", "assistant"]
    # every citation link resolves in the web reader
    client = web_client(store)
    for s in ans.sources:
        r = client.get(s["href"])
        assert r.status_code == 200 and f'id="c{s["chunk_id"]}"' in r.text


# ------------------------------------------------------------------ 2. study generators
def test_all_study_generators_with_real_grounding(world):
    store, cid, ids, _ = world
    p = smart_provider()
    lec = next(e for e in store.events(cid) if e["kind"] == "lecture")
    gid = study.primer(store, cid, p)
    g = store.guide(gid)
    assert g["kind"] == "primer" and g["event_id"] == lec["id"]
    assert disclaimer.FOOTER in g["body_md"] and g["sources"] and g["data"]["blocks"]
    assert "Not from your materials" in g["body_md"]
    ex = store.guide(study.exam_guide(store, cid, p))
    assert ex["kind"] == "exam_guide" and ex["data"]["blocks"]
    cm = store.guide(study.concept_map(store, cid, p))
    assert [n["id"] for n in cm["data"]["nodes"]] == ["eig", "diag"] and cm["data"]["blocks"]
    exp = store.guide(study.explain(store, cid, p, "eigenvalues", depth="from scratch"))
    assert exp["kind"] == "explain" and exp["data"]["blocks"]
    pr = store.guide(study.practice(store, cid, p, "eigenvalues", count=3))
    assert pr["kind"] == "practice" and pr["data"]["twin"] is False
    cards = study.make_flashcards(store, cid, p, topic="eigenvalues")
    assert len(cards) == 2
    card = store.flashcard(cards[0])
    assert card["chunk_id"] and card["source_label"]
    after = study.review_flashcard(store, cards[0], "good")
    assert after["reps"] == 1 and after["interval_days"] >= 1
    ms = store.guide(study.missed(store, cid, p, date=future(3)))
    assert ms["kind"] == "missed"
    rows = study.upcoming(store, days=30)
    assert rows and all(r["course_id"] == cid for r in rows)
    assert any(r["action"] == "1 flashcard due" for r in rows) or any(r["flashcards_due"] == 1 for r in rows)
    for r in rows:
        assert "1 flashcards" not in r["action"]


# ------------------------------------------------------------------ 3. graded path
def test_graded_problem_gets_a_twin_never_the_original(world):
    store, cid, ids, _ = world
    store.add_graded_item(cid, "HW 3 Problem 1", GRADED_PROBLEM)
    changing = [["7", "3"], ["19", "5"]]  # first twin reuses the original numbers, the retry changes them
    p = smart_provider(changing)
    ans = grounding.ask(store, cid, GRADED_PROBLEM, p)
    assert ans.twin and len(p.twin_calls) == 2  # the retry happened
    g = store.guide(ans.guide_id)
    assert g["kind"] == "twin" and g["data"]["mode"] == "twin" and g["data"]["blocks"]
    assert "19" in g["data"]["twin_problem"] and "7" not in g["data"]["twin_problem"].replace("17", "")
    assert GRADED_PROBLEM not in ans.body_md
    assert "similar problem" in ans.routing_note
    assert [m["role"] for m in store.chat_messages(cid)] == ["user", "assistant"]  # one chat record, from ask()
    # a graded material works too
    gm = store.add_material(cid, "assignment", "Problem set 7", graded=1)
    store.replace_chunks(gm, [{"text": "Compute the eigenvalues of the matrix with rows 9 4 and 4 9 exactly.",
                               "loc_kind": "page", "loc_start": 1, "loc_label": "p. 1"}])
    hit = integrity.match_graded(store, cid, "Compute the eigenvalues of the matrix with rows 9 4 and 4 9 exactly.")
    assert hit and hit["material_id"] == gm


# ------------------------------------------------------------------ 4. web
def web_client(store, provider=None):
    app = web.create_app(store, key=KEY, provider_factory=lambda: provider)
    client = TestClient(app, base_url="http://127.0.0.1:8766", follow_redirects=True)
    client.cookies.set(web.COOKIE, KEY)
    store.set_setting(disclaimer.ACK_SETTING, "2026-01-01T00:00:00+00:00")
    return client


def test_web_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_DATA_DIR, str(tmp_path / "data"))
    store = Store(":memory:")
    provider = smart_provider()
    app = web.create_app(store, key=KEY, provider_factory=lambda: provider)
    c = TestClient(app, base_url="http://127.0.0.1:8766", follow_redirects=False)
    code = paths.new_launch_code()
    assert c.get(f"/launch?code={code}").status_code == 303
    assert c.get("/").status_code == 303 or "I understand" in c.get("/welcome").text
    assert c.post("/welcome").status_code == 303
    c.follow_redirects = True
    r = c.post("/courses/new", data={"name": "Linear Algebra", "code": "MATH 2210", "university": "Example U",
                                     "term": "Fall 2026", "color": "blue"})
    assert r.status_code == 200
    cid = store.courses()[0]["id"]
    r = c.post(f"/courses/{cid}/materials/upload",
               files=[("files", ("lecture 1 slides.md", LECTURE_MD.encode(), "text/markdown"))])
    assert r.status_code == 200 and store.materials(cid)
    mid = store.materials(cid)[0]["id"]

    # Ask: chip, outside box, sources, footer
    r = c.post(f"/courses/{cid}/ask", data={"question": "What is an eigenvalue?", "depth": "standard"})
    assert r.status_code == 200
    assert re.search(rf'class="cite" href="/read/{mid}\?at=\d+#c\d+"', r.text)
    assert "Not from your materials" in r.text and 'class="outside"' in r.text
    assert 'class="sources"' in r.text and "<h4>Sources</h4>" in r.text and store.material(mid)["title"] in r.text
    assert disclaimer.FOOTER in r.text
    assert len(store.chat_messages(cid)) == 2
    assert "Not from your materials" in c.get(f"/courses/{cid}/ask").text  # history renders from saved blocks
    href = re.search(r'class="cite" href="(/read/[^"]+)"', r.text).group(1)
    rr = c.get(href)
    assert rr.status_code == 200 and re.search(r'id="c\d+"', rr.text) and "eigenvalue" in rr.text.lower()
    # the saved answer guide page renders the same way
    gid = store.guides(cid, "answer")[0]["id"]
    gp = c.get(f"/guides/{gid}")
    assert "Not from your materials" in gp.text and 'class="cite"' in gp.text

    # Study: primer needs an upcoming lecture
    store.add_event(cid, "lecture", "Eigenvalues", future(2))
    r = c.post(f"/courses/{cid}/study/primer")
    assert r.status_code == 200 and "Primer: Eigenvalues" in r.text and 'class="cite"' in r.text
    r = c.post(f"/courses/{cid}/study/flashcards", data={"topic": "eigenvalues"})
    assert "Made 2 cards" in r.text
    card = store.flashcards(cid)[0]
    r = c.get(f"/flashcards/{cid}")
    assert card["front"] in r.text
    r = c.post(f"/flashcards/{cid}/review", data={"card_id": card["id"], "grade": "good"})
    assert r.status_code == 200 and store.flashcard(card["id"])["reps"] == 1

    # Past offerings: paste a link (fetcher stood in at the module level)
    class Page:
        text = "<html><title>Linear Algebra Fall 2022</title></html>"

    class F:
        def get(self, url):
            return Page()

    monkeypatch.setattr(offerings, "_SHARED", F())
    r = c.post(f"/courses/{cid}/offerings/paste", data={"links": "https://example.edu/la/fall2022/"})
    assert "Added 1 link" in r.text
    cand = store.offering_candidates(cid)[0]
    assert cand["title"] == "Linear Algebra Fall 2022" and cand["term_label"] == "Fall 2022"
    assert "Linear Algebra Fall 2022" in c.get(f"/courses/{cid}/offerings").text

    # graded question through the Ask tab: twin, with one chat record per turn
    store.add_graded_item(cid, "HW 3 Problem 1", GRADED_PROBLEM)
    before = len(store.chat_messages(cid))
    r = c.post(f"/courses/{cid}/ask", data={"question": GRADED_PROBLEM})
    assert "data-routing-note" in r.text and "Worked twin problem.</strong>" not in r.text and len(store.chat_messages(cid)) == before + 2


def test_web_find_course_site_search_uses_offerings_module(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_DATA_DIR, str(tmp_path / "data"))
    store = Store(":memory:")
    cid = store.add_course("Algorithms", website_url="https://example.edu/algo/")
    pages = {"https://example.edu/algo/": '<a href="fall2021/">Fall 2021</a>',
             "https://example.edu/algo/fall2021/": "<title>Algo Fall 2021</title>"}

    class Res:
        def __init__(self, t):
            self.text = t

    class F:
        def get(self, url):
            if url in pages:
                return Res(pages[url])
            raise RuntimeError("404")

    monkeypatch.setattr(offerings, "_SHARED", F())
    c = web_client(store)
    r = c.post(f"/courses/{cid}/offerings/search")
    assert "Found 1 possible page" in r.text
    assert store.offering_candidates(cid)[0]["term_label"] == "Fall 2021"


# ------------------------------------------------------------------ 5. MCP tools as functions
def test_mcp_tools_with_real_modules(world):
    store, cid, ids, tmp = world
    mcp_server.set_store(store)
    try:
        out = mcp_server.search_materials(cid, "eigenvalues")
        chunk = retrieval.search(store, cid, "eigenvalues")[0]
        assert "[1]" in out and store.material(chunk["material_id"])["title"] in out
        assert f"/read/{chunk['material_id']}?at={chunk['chunk_id']}#c{chunk['chunk_id']}" in out
        assert "eigenvalue" in mcp_server.read_passage(chunk["chunk_id"]).lower()
        body = ("# Guide\n\nEigenvalues solve the characteristic equation [1].\n\nA claim with a bad cite [9].\n\n"
                "An uncited claim about Euler.")
        res = mcp_server.save_guide(cid, "primer", "My guide", body, [chunk["chunk_id"]])
        assert "[9]" in res and "/guides/" in res
        g = store.guides(cid, "primer")[0]
        assert "[9]" not in g["body_md"] and "**Not from your materials:** A claim with a bad cite" in g["body_md"]
        assert "**Not from your materials:**" in g["body_md"] and disclaimer.FOOTER in g["body_md"]
        kinds = [b["kind"] for b in g["data"]["blocks"]]
        assert kinds.count("outside") == 2 and "text" in kinds
        c = web_client(store)
        page = c.get(f"/guides/{g['id']}")
        assert page.status_code == 200 and "Not from your materials" in page.text and 'class="cite"' in page.text
        assert disclaimer.AI_APP_FOOTER in page.text
        # graded twin instructions
        store.add_graded_item(cid, "HW 3 Problem 1", GRADED_PROBLEM)
        assert mcp_server.get_twin_instructions(cid, GRADED_PROBLEM).startswith("GRADED")
        assert mcp_server.get_twin_instructions(cid, "What is a basis?").startswith("NOT MATCHED")
        assert "Saved" in mcp_server.add_note(cid, "Study note", "Remember that det(A) is the product of eigenvalues.")
        assert any(m["title"] == "Study note" for m in store.materials(cid))
        assert "Midterm" in mcp_server.list_upcoming(days=60, course_id=cid)
        # the offerings tool and the web call the same function
        store.update_course(cid, website_url="https://example.edu/la/")

        class R:
            text = '<a href="fall2022/">Fall 2022</a>'

        class F:
            def get(self, url):
                if url.endswith("fall2022/"):
                    return type("R2", (), {"text": "<title>LA Fall 2022</title>"})()
                return R()

        offerings._SHARED = F()
        try:
            assert "Fall 2022" in mcp_server.find_past_offerings(cid)
        finally:
            offerings._SHARED = None
        with pytest.raises(ToolError):
            mcp_server.save_guide(cid, "nonsense", "t", "b", [])
    finally:
        mcp_server.set_store(None)
