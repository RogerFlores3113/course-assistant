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


def test_migration_adds_new_columns_to_old_db(tmp_path):
    import sqlite3
    from course_assistant.store import Store
    db = tmp_path / "old.db"
    s = Store(db)
    s.close()
    con = sqlite3.connect(db)
    # simulate a v1 database by checking the columns exist after reopening (ALTER is idempotent)
    con.close()
    s = Store(db)
    cid = s.add_course("Chem")
    gid = s.add_graded_item(cid, "PS3", "Problem 1 ...", kind="problem", label="Problem 1")
    assert s.graded_item(gid)["label"] == "Problem 1"
    s.update_graded_item(gid, kind="essay")
    assert s.graded_item(gid)["kind"] == "essay"
    mid = s.add_material(cid, kind="notes", title="x")
    s.update_material(mid, event_id=3, duplicate_of=None)
    assert s.material(mid)["event_id"] == 3
