import asyncio
import json
import sys
import types

import pytest
from mcp.server.fastmcp.exceptions import ToolError

from course_assistant import disclaimer, mcp_server as m
from course_assistant.store import Store


@pytest.fixture
def env(monkeypatch):
    s = Store(":memory:")
    m.set_store(s)
    cid = s.add_course("Linear Algebra", code="18.06", university="MIT", website_url="https://math.example.edu/la/")
    mid = s.add_material(cid, "slides", "Lecture 3 slides")
    s.replace_chunks(mid, [
        {"text": "Eigenvalues satisfy det(A - lambda I) = 0.", "loc_kind": "slide", "loc_start": 4, "loc_label": "slide 4"},
        {"text": "Diagonalization writes A = P D P^-1.", "loc_kind": "slide", "loc_start": 5, "loc_label": "slide 5"},
    ])
    gmid = s.add_material(cid, "assignment", "Problem set 2", graded=1)
    s.replace_chunks(gmid, [{"text": "Find eigenvalues of the matrix [[2,1],[1,2]].", "loc_kind": "page",
                             "loc_start": 1, "loc_label": "p. 1"}])
    chunks = [c["id"] for c in s.chunks_of(mid)] + [c["id"] for c in s.chunks_of(gmid)]

    def search(store, course_id, query, *, limit=8, kinds=None, material_ids=None, exclude_graded=False):
        out = []
        for c in store._rows("SELECT * FROM chunks WHERE course_id = ? ORDER BY id", (course_id,)):
            mat = store.material(c["material_id"])
            if exclude_graded and mat["graded"]:
                continue
            if query.split()[0].lower() in c["text"].lower():
                out.append(dict(chunk_id=c["id"], material_id=c["material_id"], material_title=mat["title"],
                                material_kind=mat["kind"], loc_kind=c["loc_kind"], loc_start=c["loc_start"],
                                loc_label=c["loc_label"], heading="", text=c["text"],
                                href=f"/read/{c['material_id']}?at={c['id']}#c{c['id']}", score=1.0))
        return out[:limit]

    def around(store, chunk_id, before=1, after=1):
        c = store.chunk(chunk_id)
        rows = store.chunks_of(c["material_id"])
        i = [r["id"] for r in rows].index(chunk_id)
        return rows[max(0, i - before): i + after + 1]

    monkeypatch.setitem(sys.modules, "course_assistant.retrieval", types.SimpleNamespace(search=search, around=around))

    def match_graded(store, course_id, question):
        return {"title": "Problem set 2"} if "[[2,1],[1,2]]" in question else None

    monkeypatch.setitem(sys.modules, "course_assistant.integrity", types.SimpleNamespace(match_graded=match_graded))
    yield s, cid, mid, gmid, chunks
    m.set_store(None)


def test_no_delete_or_remove_tools():
    tools = asyncio.run(m.mcp.list_tools())
    names = {t.name for t in tools}
    assert {"search_materials", "read_passage", "save_guide", "get_twin_instructions", "find_past_offerings",
            "save_past_offering_candidate", "list_courses", "course_overview", "list_upcoming", "add_course",
            "add_material_from_file", "add_material_from_url", "add_note", "add_event", "add_flashcards",
            "list_graded", "mark_graded"} <= names
    for n in names:
        assert not any(w in n for w in ("delete", "remove", "drop", "clear", "erase"))
    assert "Not from your materials" in m.INSTRUCTIONS and "[n]" in m.INSTRUCTIONS or "[1]" in m.INSTRUCTIONS
    assert disclaimer.FOOTER in m.INSTRUCTIONS


def test_search_numbers_links_and_flags_graded(env):
    s, cid, mid, gmid, chunks = env
    out = m.search_materials(cid, "eigenvalues")
    assert "[1] Lecture 3 slides" in out and "slide 4" in out
    assert f"chunk_id: {chunks[0]}" in out
    assert f"http://127.0.0.1:" in out and f"/read/{mid}?at={chunks[0]}#c{chunks[0]}" in out
    assert "[2] Problem set 2" in out and "GRADED MATERIAL" in out
    assert "GRADED" not in out.split("[2]")[0]
    assert "Problem set 2" not in m.search_materials(cid, "eigenvalues", exclude_graded=True)
    assert "No results" in m.search_materials(cid, "zebra")
    with pytest.raises(ToolError):
        m.search_materials(999, "x")
    with pytest.raises(ToolError):
        m.search_materials(cid, "   ")
    with pytest.raises(ToolError):
        m.search_materials(cid, "x", limit=0)
    with pytest.raises(ToolError):
        m.search_materials(cid, "x", kinds=["bogus"])


def test_read_passage(env):
    s, cid, mid, gmid, chunks = env
    out = m.read_passage(chunks[0], before=0, after=1)
    assert "Eigenvalues" in out and "Diagonalization" in out and "requested passage" in out
    assert f"/read/{mid}?at={chunks[1]}" in out
    with pytest.raises(ToolError):
        m.read_passage(99999)


def test_save_guide_validates_labels_and_footer(env):
    s, cid, mid, gmid, chunks = env
    body = ("# Eigenvalues\n\nEigenvalues solve the characteristic equation [1].\n\n"
            "Diagonalization factors a matrix [2, 7].\n\n"
            "Fun fact about Euler that the slides never mention [9].\n\n"
            "- Bullet cited [1]\n- Bullet uncited\n\n"
            "<outside>Background from elsewhere.</outside>")
    out = m.save_guide(cid, "primer", "Primer", body, [chunks[0], chunks[1], 424242])
    assert "Open it: http://127.0.0.1:" in out and "/guides/" in out
    assert "[7]" in out and "[9]" in out and "424242" in out
    g = s.guides(cid)[0]
    b = g["body_md"]
    assert "characteristic equation [1]." in b
    assert "[2]" in b and "[7]" not in b and "[9]" not in b
    assert "**Not from your materials:** Fun fact about Euler" in b
    assert [x["kind"] for x in g["data"]["blocks"]].count("outside") == 3
    assert "> **Not from your materials:** Bullet uncited" in b and "- Bullet cited [1]" in b
    assert "> Background from elsewhere." in b  # same run as the block before it, one label
    assert "<outside>" not in b
    assert "# Eigenvalues" in b and not b.count("Not from your materials:** # ")
    assert disclaimer.FOOTER in b
    assert g["created_by"] == "ai_app" and {x["chunk_id"] for x in g["sources"]} == {chunks[0], chunks[1]}
    with pytest.raises(ToolError):
        m.save_guide(cid, "nonsense", "t", "b", [])
    with pytest.raises(ToolError):
        m.save_guide(cid, "answer", "t", "", [])


def test_twin_instructions(env):
    s, cid, mid, gmid, chunks = env
    graded = m.get_twin_instructions(cid, "Find eigenvalues of [[2,1],[1,2]]")
    assert graded.startswith("GRADED") and "different numbers" in graded.lower() or "change every number" in graded
    assert "essay" in graded and "Do not solve" in graded
    other = m.get_twin_instructions(cid, "What is a basis?")
    assert other.startswith("NOT MATCHED") and "mark_graded" in other


def test_writes_are_logged_and_work(env, monkeypatch):
    s, cid, mid, gmid, chunks = env
    imp = types.SimpleNamespace(
        SUPPORTED_EXTENSIONS={".pdf": "PDF"}, ImportError_=type("ImportError_", (Exception,), {}),
        import_text=lambda st, c, t, text, *, kind="notes", added_by="student": st.add_material(c, kind, t, added_by=added_by),
        import_url=lambda st, c, url, *, kind="website", fetcher=None, added_by="student": st.add_material(c, kind, "Page", added_by=added_by),
        import_file=lambda st, c, p, **kw: st.add_material(c, kw["kind"] or "other", kw["title"] or p.name, added_by=kw["added_by"]))
    monkeypatch.setitem(sys.modules, "course_assistant.importers", imp)
    pdf = __import__("pathlib").Path(__file__).parent / "x.pdf"
    pdf.write_bytes(b"%PDF")
    try:
        calls = [
            lambda: m.add_course("Physics", university="MIT"),
            lambda: m.add_note(cid, "My note", "text"),
            lambda: m.add_material_from_url(cid, "https://x.edu/page"),
            lambda: m.add_material_from_file(cid, str(pdf), kind="handout", lecture_date="2026-02-03"),
            lambda: m.add_event(cid, "exam", "Midterm", "2026-12-01", time="14:30", topics="eigenvalues"),
            lambda: m.add_flashcards(cid, [{"front": "Q", "back": "A", "chunk_id": chunks[0]}], deck="d"),
            lambda: m.mark_graded(cid, "HW 3", "body"),
            lambda: m.mark_graded(cid, "", material_id=mid),
            lambda: m.save_guide(cid, "answer", "A", "Text [1]", [chunks[0]]),
        ]
        for c in calls:
            before = len(s.ai_actions())
            assert c()
            assert len(s.ai_actions()) == before + 1
    finally:
        pdf.unlink()
    assert s.material(mid)["graded"] == 1
    assert s.materials(cid, "notes")[0]["added_by"] == "ai_app"
    assert s.events(cid)[0]["added_by"] == "ai_app"
    card = s.flashcards(cid)[0]
    assert card["created_by"] == "ai_app" and "slide 4" in card["source_label"]
    assert "HW 3" in m.list_graded(cid) and "Problem set 2" in m.list_graded(cid)
    assert "Midterm" in m.list_upcoming(days=365, course_id=cid)
    assert "Lecture 3 slides" in m.course_overview(cid) and "Linear Algebra" in m.list_courses()


def test_validation_errors(env):
    s, cid, *_ = env
    for fn in (lambda: m.add_event(cid, "party", "x", "2026-01-01"), lambda: m.add_event(cid, "exam", "x", "01/02/2026"),
               lambda: m.add_event(cid, "exam", "x", "2026-01-01", time="2pm"),
               lambda: m.add_course("  "), lambda: m.add_course("X", website_url="ftp://a"),
               lambda: m.add_material_from_file(cid, "relative.pdf"), lambda: m.add_material_from_file(cid, "/nope/x.pdf"),
               lambda: m.add_material_from_url(cid, "file:///etc/passwd"),
               lambda: m.add_flashcards(cid, []), lambda: m.add_flashcards(cid, [{"front": "q"}]),
               lambda: m.add_flashcards(cid, [{"front": "q", "back": "a", "chunk_id": 99999}]),
               lambda: m.mark_graded(cid, "x", material_id=424242), lambda: m.list_upcoming(days=0)):
        with pytest.raises(ToolError):
            fn()


def test_find_and_save_past_offerings(env, monkeypatch):
    s, cid, *_ = env

    class Res:
        text = '<title>LA Fall 2021</title><a href="fall2020/">Fall 2020</a>'

    class F:
        def get(self, url):
            if url.endswith("fall2020/") or url.endswith("fall2021/"):
                r = Res()
                r.text = "<title>LA %s</title>" % ("Fall 2020" if "2020" in url else "Fall 2021")
                return r
            return Res()

    from course_assistant import offerings
    monkeypatch.setattr(offerings, "_SHARED", F())
    monkeypatch.setitem(sys.modules, "course_assistant.fetcher", types.SimpleNamespace(PoliteFetcher=F))
    out = m.find_past_offerings(cid, "https://math.example.edu/la/")
    assert "candidate" in out and "math.example.edu/la/" in out
    out2 = m.find_past_offerings(cid, ["https://math.example.edu/la/fall2021/"])
    assert "Fall 2021" in out2
    with pytest.raises(ToolError):
        m.find_past_offerings(s.add_course("No site"), "any old exams?")
    cand = s.offering_candidates(cid)[0]
    monkeypatch.setitem(sys.modules, "course_assistant.importers", types.SimpleNamespace(
        import_url=lambda st, c, url, *, kind, fetcher=None, added_by: st.add_material(c, kind, "Old LA", added_by=added_by)))
    before = len(s.ai_actions())
    res = m.save_past_offering_candidate(cid, cand["id"])
    assert "Saved material" in res and len(s.ai_actions()) == before + 1
    assert s.materials(cid, "past_offering")
    with pytest.raises(ToolError):
        m.save_past_offering_candidate(cid, 9999)
