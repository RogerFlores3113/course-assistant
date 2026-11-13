"""The one SQLite database: courses, their materials, searchable chunks, calendar, guides and study items.

Everything lives in one file in the data folder. The student can delete things from the app window; the AI app
(through the MCP server) can add and read but never delete (see mcp_server.py).
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 2

MATERIAL_KINDS = {
    "slides": "Lecture slides",
    "handout": "Handout",
    "recording": "Lecture recording",
    "textbook": "Textbook",
    "reading": "Reading",
    "notes": "My notes",
    "website": "Course website",
    "syllabus": "Syllabus",
    "past_offering": "Past offering",
    "assignment": "Assignment",
    "other": "Other",
}

EVENT_KINDS = {
    "lecture": "Lecture",
    "exam": "Exam",
    "quiz": "Quiz",
    "assignment": "Assignment due",
    "reading": "Reading",
    "other": "Other",
}

GUIDE_KINDS = {
    "primer": "Pre-lecture primer",
    "exam_guide": "Exam guide",
    "concept_map": "Concept map",
    "explain": "Re-explanation",
    "practice": "Practice questions",
    "missed": "What did I miss",
    "answer": "Answer",
    "twin": "Worked twin problem",
}

# Where in a material a chunk sits. The reader opens a citation at this spot.
LOC_KINDS = ("page", "slide", "time", "section", "line")

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE IF NOT EXISTS courses (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    code TEXT NOT NULL DEFAULT '',
    university TEXT NOT NULL DEFAULT '',
    term TEXT NOT NULL DEFAULT '',
    instructor TEXT NOT NULL DEFAULT '',
    website_url TEXT NOT NULL DEFAULT '',
    color TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    archived INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS materials (
    id INTEGER PRIMARY KEY,
    course_id INTEGER NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    kind TEXT NOT NULL DEFAULT 'other',
    title TEXT NOT NULL,
    filename TEXT NOT NULL DEFAULT '',      -- relative to paths.files_dir(); '' for pasted text
    original_name TEXT NOT NULL DEFAULT '',
    mime TEXT NOT NULL DEFAULT '',
    source_url TEXT NOT NULL DEFAULT '',
    sha256 TEXT NOT NULL DEFAULT '',
    lecture_date TEXT NOT NULL DEFAULT '',  -- YYYY-MM-DD when it belongs to one lecture
    week INTEGER,
    status TEXT NOT NULL DEFAULT 'ready',   -- ready | processing | needs_transcription | error
    status_note TEXT NOT NULL DEFAULT '',
    page_count INTEGER,
    duration_seconds REAL,
    graded INTEGER NOT NULL DEFAULT 0,      -- the student marked it as graded work (section 8)
    added_by TEXT NOT NULL DEFAULT 'student',  -- student | ai_app
    added_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS materials_course ON materials(course_id);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    material_id INTEGER NOT NULL REFERENCES materials(id) ON DELETE CASCADE,
    course_id INTEGER NOT NULL,
    ord INTEGER NOT NULL,
    heading TEXT NOT NULL DEFAULT '',
    text TEXT NOT NULL,
    loc_kind TEXT NOT NULL,                 -- page | slide | time | section | line
    loc_start TEXT NOT NULL,                -- page/slide number, seconds, or a section anchor id
    loc_end TEXT NOT NULL DEFAULT '',
    loc_label TEXT NOT NULL                 -- what a person reads: "p. 12", "slide 4", "12:05", "Week 3"
);
CREATE INDEX IF NOT EXISTS chunks_material ON chunks(material_id, ord);
CREATE INDEX IF NOT EXISTS chunks_course ON chunks(course_id);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, heading, content='chunks', content_rowid='id', tokenize='porter unicode61 remove_diacritics 2'
);
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text, heading) VALUES (new.id, new.text, new.heading);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text, heading) VALUES ('delete', old.id, old.text, old.heading);
END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text, heading) VALUES ('delete', old.id, old.text, old.heading);
    INSERT INTO chunks_fts(rowid, text, heading) VALUES (new.id, new.text, new.heading);
END;

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,
    course_id INTEGER NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    kind TEXT NOT NULL DEFAULT 'other',
    title TEXT NOT NULL,
    date TEXT NOT NULL,                     -- YYYY-MM-DD
    time TEXT NOT NULL DEFAULT '',          -- HH:MM, optional
    topics TEXT NOT NULL DEFAULT '',
    source_material_id INTEGER,
    source_chunk_id INTEGER,
    added_by TEXT NOT NULL DEFAULT 'student',  -- student | syllabus | ai_app
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_course_date ON events(course_id, date);

CREATE TABLE IF NOT EXISTS guides (
    id INTEGER PRIMARY KEY,
    course_id INTEGER NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    title TEXT NOT NULL,
    question TEXT NOT NULL DEFAULT '',
    body_md TEXT NOT NULL,
    sources_json TEXT NOT NULL DEFAULT '[]',   -- [{n, chunk_id, material_id, label, title, href}]
    data_json TEXT NOT NULL DEFAULT '{}',      -- kind-specific extras (concept map graph, twin problem, ...)
    event_id INTEGER,
    created_by TEXT NOT NULL DEFAULT 'app',    -- app | ai_app
    model TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS guides_course ON guides(course_id, created_at);

CREATE TABLE IF NOT EXISTS graded_items (
    id INTEGER PRIMARY KEY,
    course_id INTEGER NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    body TEXT NOT NULL DEFAULT '',
    material_id INTEGER,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS flashcards (
    id INTEGER PRIMARY KEY,
    course_id INTEGER NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    deck TEXT NOT NULL DEFAULT '',
    front TEXT NOT NULL,
    back TEXT NOT NULL,
    chunk_id INTEGER,
    source_label TEXT NOT NULL DEFAULT '',
    due_at TEXT NOT NULL,
    interval_days REAL NOT NULL DEFAULT 0,
    ease REAL NOT NULL DEFAULT 2.5,
    reps INTEGER NOT NULL DEFAULT 0,
    created_by TEXT NOT NULL DEFAULT 'app',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS flashcards_due ON flashcards(course_id, due_at);

CREATE TABLE IF NOT EXISTS chat_messages (
    id INTEGER PRIMARY KEY,
    course_id INTEGER NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    role TEXT NOT NULL,                      -- user | assistant
    content TEXT NOT NULL,
    sources_json TEXT NOT NULL DEFAULT '[]',
    guide_id INTEGER,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS offering_candidates (
    id INTEGER PRIMARY KEY,
    course_id INTEGER NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    url TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    term_label TEXT NOT NULL DEFAULT '',
    found_by TEXT NOT NULL DEFAULT 'paste',  -- paste | search | crawl | ai_app
    status TEXT NOT NULL DEFAULT 'found',    -- found | saved | dismissed | blocked
    note TEXT NOT NULL DEFAULT '',
    material_id INTEGER,
    created_at TEXT NOT NULL,
    UNIQUE(course_id, url)
);

CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,                      -- transcribe | import | ...
    material_id INTEGER,
    status TEXT NOT NULL DEFAULT 'queued',   -- queued | running | done | error
    progress REAL NOT NULL DEFAULT 0,
    note TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ai_actions (
    id INTEGER PRIMARY KEY,
    at TEXT NOT NULL,
    kind TEXT NOT NULL,
    detail_json TEXT NOT NULL DEFAULT '{}'
);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class Store:
    """Thin wrapper over one SQLite connection. Thread-safe enough for the local server (one lock)."""

    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        if self.path != ":memory:":
            self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.conn.execute("INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))
        self.conn.execute("UPDATE meta SET value = ? WHERE key = 'schema_version'", (str(SCHEMA_VERSION),))
        self.conn.commit()

    # Columns added after the first release. Each is added when missing, so old databases keep working.
    _ADDED_COLUMNS = {
        "materials": [("event_id", "INTEGER"),            # the lecture event this material belongs to
                      ("duplicate_of", "INTEGER")],       # a near-identical material in the same course
        "graded_items": [("kind", "TEXT NOT NULL DEFAULT 'problem'"),   # problem | proof | essay
                         ("label", "TEXT NOT NULL DEFAULT ''")],        # e.g. "Problem 2"
    }

    def _migrate(self) -> None:
        for table, cols in self._ADDED_COLUMNS.items():
            have = {r[1] for r in self.conn.execute(f"PRAGMA table_info({table})")}
            for name, decl in cols:
                if name not in have:
                    self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")

    # ------------------------------------------------------------ helpers
    def _rows(self, sql: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self.conn.execute(sql, tuple(params)).fetchall()]

    def _row(self, sql: str, params: Iterable[Any] = ()) -> dict[str, Any] | None:
        with self._lock:
            r = self.conn.execute(sql, tuple(params)).fetchone()
            return dict(r) if r else None

    def _exec(self, sql: str, params: Iterable[Any] = ()) -> int:
        with self._lock:
            cur = self.conn.execute(sql, tuple(params))
            self.conn.commit()
            return int(cur.lastrowid or 0)

    def _insert(self, table: str, values: dict[str, Any]) -> int:
        cols = ", ".join(values)
        marks = ", ".join("?" for _ in values)
        return self._exec(f"INSERT INTO {table} ({cols}) VALUES ({marks})", values.values())

    def _update(self, table: str, row_id: int, values: dict[str, Any]) -> None:
        if not values:
            return
        sets = ", ".join(f"{k} = ?" for k in values)
        self._exec(f"UPDATE {table} SET {sets} WHERE id = ?", [*values.values(), row_id])

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    # ------------------------------------------------------------ settings
    def get_setting(self, key: str, default: str = "") -> str:
        r = self._row("SELECT value FROM settings WHERE key = ?", (key,))
        return r["value"] if r and r["value"] is not None else default

    def set_setting(self, key: str, value: str) -> None:
        self._exec("INSERT INTO settings(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                   (key, value))

    # ------------------------------------------------------------ courses
    def add_course(self, name: str, code: str = "", university: str = "", term: str = "", instructor: str = "",
                   website_url: str = "", color: str = "") -> int:
        return self._insert("courses", {"name": name.strip(), "code": code.strip(), "university": university.strip(),
                                        "term": term.strip(), "instructor": instructor.strip(),
                                        "website_url": website_url.strip(), "color": color, "created_at": now_iso()})

    def course(self, course_id: int) -> dict[str, Any] | None:
        return self._row("SELECT * FROM courses WHERE id = ?", (course_id,))

    def courses(self, include_archived: bool = False) -> list[dict[str, Any]]:
        where = "" if include_archived else "WHERE archived = 0"
        return self._rows(f"""SELECT c.*, (SELECT count(*) FROM materials m WHERE m.course_id = c.id) AS material_count
                              FROM courses c {where} ORDER BY c.archived, c.name COLLATE NOCASE""")

    def update_course(self, course_id: int, **values: Any) -> None:
        allowed = {"name", "code", "university", "term", "instructor", "website_url", "color", "archived"}
        self._update("courses", course_id, {k: v for k, v in values.items() if k in allowed})

    def delete_course(self, course_id: int) -> None:
        """Student only (the app window). Files on disk are removed by the caller."""
        self._exec("DELETE FROM courses WHERE id = ?", (course_id,))

    # ------------------------------------------------------------ materials
    def add_material(self, course_id: int, kind: str, title: str, **values: Any) -> int:
        row = {"course_id": course_id, "kind": kind if kind in MATERIAL_KINDS else "other", "title": title.strip() or "Untitled",
               "added_at": now_iso()}
        allowed = {"filename", "original_name", "mime", "source_url", "sha256", "lecture_date", "week", "status",
                   "status_note", "page_count", "duration_seconds", "graded", "added_by"}
        row.update({k: v for k, v in values.items() if k in allowed})
        return self._insert("materials", row)

    def material(self, material_id: int) -> dict[str, Any] | None:
        return self._row("SELECT * FROM materials WHERE id = ?", (material_id,))

    def materials(self, course_id: int, kind: str | None = None) -> list[dict[str, Any]]:
        sql = """SELECT m.*, (SELECT count(*) FROM chunks c WHERE c.material_id = m.id) AS chunk_count
                 FROM materials m WHERE m.course_id = ?"""
        params: list[Any] = [course_id]
        if kind:
            sql += " AND m.kind = ?"
            params.append(kind)
        return self._rows(sql + " ORDER BY COALESCE(NULLIF(m.lecture_date, ''), m.added_at), m.id", params)

    def material_by_sha(self, course_id: int, sha256: str) -> dict[str, Any] | None:
        return self._row("SELECT * FROM materials WHERE course_id = ? AND sha256 = ? AND sha256 != ''", (course_id, sha256))

    def update_material(self, material_id: int, **values: Any) -> None:
        allowed = {"kind", "title", "filename", "original_name", "mime", "source_url", "sha256", "lecture_date", "week",
                   "status", "status_note", "page_count", "duration_seconds", "graded", "event_id", "duplicate_of"}
        self._update("materials", material_id, {k: v for k, v in values.items() if k in allowed})

    def delete_material(self, material_id: int) -> None:
        """Student only. Chunks go with it (cascade); the caller removes the file."""
        self._exec("DELETE FROM materials WHERE id = ?", (material_id,))

    # ------------------------------------------------------------ chunks
    def replace_chunks(self, material_id: int, chunks: list[dict[str, Any]]) -> int:
        """Swap a material's chunks for new ones. Each chunk: text, loc_kind, loc_start, loc_label, [loc_end, heading]."""
        m = self.material(material_id)
        if not m:
            raise KeyError(material_id)
        with self._lock:
            self.conn.execute("DELETE FROM chunks WHERE material_id = ?", (material_id,))
            for i, c in enumerate(chunks):
                text = (c.get("text") or "").strip()
                if not text:
                    continue
                kind = c.get("loc_kind", "section")
                if kind not in LOC_KINDS:
                    raise ValueError(f"bad loc_kind {kind!r}")
                self.conn.execute(
                    """INSERT INTO chunks(material_id, course_id, ord, heading, text, loc_kind, loc_start, loc_end, loc_label)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (material_id, m["course_id"], i, c.get("heading", "") or "", text, kind, str(c.get("loc_start", "")),
                     str(c.get("loc_end", "") or ""), c.get("loc_label") or str(c.get("loc_start", ""))))
            self.conn.commit()
            return self.conn.execute("SELECT count(*) FROM chunks WHERE material_id = ?", (material_id,)).fetchone()[0]

    def chunk(self, chunk_id: int) -> dict[str, Any] | None:
        return self._row("""SELECT c.*, m.title AS material_title, m.kind AS material_kind, m.graded AS material_graded
                            FROM chunks c JOIN materials m ON m.id = c.material_id WHERE c.id = ?""", (chunk_id,))

    def chunks_of(self, material_id: int) -> list[dict[str, Any]]:
        return self._rows("SELECT * FROM chunks WHERE material_id = ? ORDER BY ord", (material_id,))

    # ------------------------------------------------------------ events (course calendar)
    def add_event(self, course_id: int, kind: str, title: str, date: str, **values: Any) -> int:
        row = {"course_id": course_id, "kind": kind if kind in EVENT_KINDS else "other", "title": title.strip(),
               "date": date, "created_at": now_iso()}
        allowed = {"time", "topics", "source_material_id", "source_chunk_id", "added_by"}
        row.update({k: v for k, v in values.items() if k in allowed})
        return self._insert("events", row)

    def event(self, event_id: int) -> dict[str, Any] | None:
        return self._row("SELECT * FROM events WHERE id = ?", (event_id,))

    def events(self, course_id: int | None = None, start: str | None = None, end: str | None = None,
               kinds: Iterable[str] | None = None) -> list[dict[str, Any]]:
        sql = "SELECT e.*, c.name AS course_name, c.code AS course_code, c.color AS course_color FROM events e JOIN courses c ON c.id = e.course_id WHERE c.archived = 0"
        params: list[Any] = []
        if course_id is not None:
            sql += " AND e.course_id = ?"
            params.append(course_id)
        if start:
            sql += " AND e.date >= ?"
            params.append(start)
        if end:
            sql += " AND e.date <= ?"
            params.append(end)
        kinds = list(kinds or [])
        if kinds:
            sql += f" AND e.kind IN ({','.join('?' for _ in kinds)})"
            params.extend(kinds)
        return self._rows(sql + " ORDER BY e.date, e.time, e.id", params)

    def update_event(self, event_id: int, **values: Any) -> None:
        allowed = {"kind", "title", "date", "time", "topics"}
        self._update("events", event_id, {k: v for k, v in values.items() if k in allowed})

    def delete_event(self, event_id: int) -> None:
        self._exec("DELETE FROM events WHERE id = ?", (event_id,))

    def clear_syllabus_events(self, material_id: int) -> None:
        """Re-reading a syllabus replaces the dates it gave, and only those."""
        self._exec("DELETE FROM events WHERE source_material_id = ? AND added_by = 'syllabus'", (material_id,))

    # ------------------------------------------------------------ guides (saved answers, primers, exam guides ...)
    def add_guide(self, course_id: int, kind: str, title: str, body_md: str, sources: list[dict[str, Any]] | None = None,
                  data: dict[str, Any] | None = None, **values: Any) -> int:
        row = {"course_id": course_id, "kind": kind, "title": title.strip() or GUIDE_KINDS.get(kind, "Guide"),
               "body_md": body_md, "sources_json": json.dumps(sources or []), "data_json": json.dumps(data or {}),
               "created_at": now_iso()}
        allowed = {"question", "event_id", "created_by", "model"}
        row.update({k: v for k, v in values.items() if k in allowed})
        return self._insert("guides", row)

    @staticmethod
    def _decode_guide(g: dict[str, Any] | None) -> dict[str, Any] | None:
        if g is None:
            return None
        g["sources"] = json.loads(g.get("sources_json") or "[]")
        g["data"] = json.loads(g.get("data_json") or "{}")
        return g

    def guide(self, guide_id: int) -> dict[str, Any] | None:
        return self._decode_guide(self._row("SELECT * FROM guides WHERE id = ?", (guide_id,)))

    def guides(self, course_id: int, kind: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        sql, params = "SELECT * FROM guides WHERE course_id = ?", [course_id]
        if kind:
            sql += " AND kind = ?"
            params.append(kind)
        rows = self._rows(sql + " ORDER BY created_at DESC, id DESC LIMIT ?", [*params, limit])
        return [self._decode_guide(r) for r in rows]  # type: ignore[misc]

    def delete_guide(self, guide_id: int) -> None:
        self._exec("DELETE FROM guides WHERE id = ?", (guide_id,))

    # ------------------------------------------------------------ graded items (academic integrity, section 8)
    def add_graded_item(self, course_id: int, title: str, body: str = "", material_id: int | None = None, *,
                        kind: str = "problem", label: str = "") -> int:
        return self._insert("graded_items", {"course_id": course_id, "title": title.strip(), "body": body,
                                             "material_id": material_id, "kind": kind, "label": label,
                                             "created_at": now_iso()})

    def graded_item(self, item_id: int) -> dict[str, Any] | None:
        return self._row("SELECT * FROM graded_items WHERE id = ?", (item_id,))

    def update_graded_item(self, item_id: int, **values: Any) -> None:
        allowed = {"title", "body", "kind", "label", "material_id"}
        self._update("graded_items", item_id, {k: v for k, v in values.items() if k in allowed})

    def graded_items(self, course_id: int) -> list[dict[str, Any]]:
        return self._rows("SELECT * FROM graded_items WHERE course_id = ? ORDER BY created_at DESC", (course_id,))

    def delete_graded_item(self, item_id: int) -> None:
        self._exec("DELETE FROM graded_items WHERE id = ?", (item_id,))

    # ------------------------------------------------------------ flashcards
    def add_flashcard(self, course_id: int, front: str, back: str, **values: Any) -> int:
        row = {"course_id": course_id, "front": front.strip(), "back": back.strip(), "due_at": now_iso(),
               "created_at": now_iso()}
        allowed = {"deck", "chunk_id", "source_label", "created_by"}
        row.update({k: v for k, v in values.items() if k in allowed})
        return self._insert("flashcards", row)

    def flashcard(self, card_id: int) -> dict[str, Any] | None:
        return self._row("SELECT * FROM flashcards WHERE id = ?", (card_id,))

    def flashcards(self, course_id: int, due_only: bool = False, deck: str | None = None) -> list[dict[str, Any]]:
        sql, params = "SELECT * FROM flashcards WHERE course_id = ?", [course_id]
        if due_only:
            sql += " AND due_at <= ?"
            params.append(now_iso())
        if deck:
            sql += " AND deck = ?"
            params.append(deck)
        return self._rows(sql + " ORDER BY due_at, id", params)

    def update_flashcard(self, card_id: int, **values: Any) -> None:
        allowed = {"front", "back", "deck", "due_at", "interval_days", "ease", "reps"}
        self._update("flashcards", card_id, {k: v for k, v in values.items() if k in allowed})

    def delete_flashcard(self, card_id: int) -> None:
        self._exec("DELETE FROM flashcards WHERE id = ?", (card_id,))

    # ------------------------------------------------------------ chat
    def add_chat_message(self, course_id: int, role: str, content: str, sources: list[dict[str, Any]] | None = None,
                         guide_id: int | None = None) -> int:
        return self._insert("chat_messages", {"course_id": course_id, "role": role, "content": content,
                                              "sources_json": json.dumps(sources or []), "guide_id": guide_id,
                                              "created_at": now_iso()})

    def chat_messages(self, course_id: int, limit: int = 100) -> list[dict[str, Any]]:
        rows = self._rows("SELECT * FROM chat_messages WHERE course_id = ? ORDER BY id DESC LIMIT ?", (course_id, limit))
        for r in rows:
            r["sources"] = json.loads(r.get("sources_json") or "[]")
        return list(reversed(rows))

    # ------------------------------------------------------------ past offerings
    def add_offering_candidate(self, course_id: int, url: str, title: str = "", term_label: str = "",
                               found_by: str = "paste", note: str = "", status: str = "found") -> int:
        existing = self._row("SELECT id FROM offering_candidates WHERE course_id = ? AND url = ?", (course_id, url))
        if existing:
            return int(existing["id"])
        return self._insert("offering_candidates", {"course_id": course_id, "url": url, "title": title,
                                                    "term_label": term_label, "found_by": found_by, "note": note,
                                                    "status": status, "created_at": now_iso()})

    def offering_candidates(self, course_id: int, status: str | None = None) -> list[dict[str, Any]]:
        sql, params = "SELECT * FROM offering_candidates WHERE course_id = ?", [course_id]
        if status:
            sql += " AND status = ?"
            params.append(status)
        return self._rows(sql + " ORDER BY status, term_label DESC, id", params)

    def offering_candidate(self, cand_id: int) -> dict[str, Any] | None:
        return self._row("SELECT * FROM offering_candidates WHERE id = ?", (cand_id,))

    def update_offering_candidate(self, cand_id: int, **values: Any) -> None:
        allowed = {"title", "term_label", "status", "note", "material_id"}
        self._update("offering_candidates", cand_id, {k: v for k, v in values.items() if k in allowed})

    # ------------------------------------------------------------ background jobs
    def add_job(self, kind: str, material_id: int | None = None, note: str = "") -> int:
        t = now_iso()
        return self._insert("jobs", {"kind": kind, "material_id": material_id, "note": note, "created_at": t,
                                     "updated_at": t})

    def update_job(self, job_id: int, **values: Any) -> None:
        allowed = {"status", "progress", "note"}
        vals = {k: v for k, v in values.items() if k in allowed}
        vals["updated_at"] = now_iso()
        self._update("jobs", job_id, vals)

    def jobs(self, status: str | None = None) -> list[dict[str, Any]]:
        if status:
            return self._rows("SELECT * FROM jobs WHERE status = ? ORDER BY id", (status,))
        return self._rows("SELECT * FROM jobs ORDER BY id DESC LIMIT 100")

    def job(self, job_id: int) -> dict[str, Any] | None:
        return self._row("SELECT * FROM jobs WHERE id = ?", (job_id,))

    # ------------------------------------------------------------ AI app action log
    def log_ai(self, kind: str, **detail: Any) -> None:
        self._insert("ai_actions", {"at": now_iso(), "kind": kind, "detail_json": json.dumps(detail, default=str)})

    def ai_actions(self, limit: int = 200) -> list[dict[str, Any]]:
        rows = self._rows("SELECT * FROM ai_actions ORDER BY id DESC LIMIT ?", (limit,))
        for r in rows:
            r["detail"] = json.loads(r.get("detail_json") or "{}")
        return rows
