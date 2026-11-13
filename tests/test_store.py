from course_assistant.store import Store


def test_course_material_chunks_and_fts():
    s = Store(":memory:")
    cid = s.add_course("Linear Algebra", code="18.06", university="MIT", term="Spring 2023")
    mid = s.add_material(cid, "slides", "Lecture 1")
    n = s.replace_chunks(mid, [
        {"text": "Gaussian elimination reduces a matrix to row echelon form.", "loc_kind": "page", "loc_start": 1,
         "loc_label": "p. 1"},
        {"text": "Eigenvalues satisfy det(A - lambda I) = 0.", "loc_kind": "page", "loc_start": 2, "loc_label": "p. 2"},
    ])
    assert n == 2
    hits = s._rows("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ?", ("eigenvalue*",))
    assert len(hits) == 1
    s.replace_chunks(mid, [{"text": "Only one now", "loc_kind": "page", "loc_start": 1, "loc_label": "p. 1"}])
    assert s._rows("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ?", ("eigenvalue*",)) == []
    s.delete_course(cid)
    assert s.materials(cid) == []
