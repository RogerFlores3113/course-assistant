"""Search a course's chunks: FTS5 with bm25 ranking, a LIKE fallback, and neighbouring-chunk context."""
from __future__ import annotations

import re
from typing import Any

from . import locations

Source = dict[str, Any]

STOPWORDS = frozenset("""
a about above after again all also am an and any are as at be because been before being below between both but by
can could did do does doing down during each few for from further had has have having he her here hers him his how
i if in into is it its itself just me more most my no nor not of off on once only or other our out over own same she
should so some such than that the their theirs them then there these they this those through to too under until up
very was we were what when where which while who whom why will with would you your yours explain tell please give
show describe define mean means
""".split())

_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")


def query_terms(question: str, max_terms: int = 14) -> list[str]:
    """Words worth searching for: no punctuation, no FTS operators, no stopwords, unique, in order."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in _TOKEN.findall(question or ""):
        for t in re.split(r"[-']", raw):
            t = t.lower()
            if not t or t in seen or t in STOPWORDS or t in {"and", "or", "not", "near"}:
                continue
            if len(t) < 2 and not t.isdigit():
                continue
            seen.add(t)
            out.append(t)
    return out[:max_terms]


def fts_query(question: str) -> str:
    """A safe FTS5 MATCH string: quoted terms ORed together, with a prefix * on terms of 4+ chars."""
    parts = []
    for t in query_terms(question):
        parts.append(f'"{t}"*' if len(t) >= 4 else f'"{t}"')
    return " OR ".join(parts)


def _filters(course_id: int, kinds, material_ids, exclude_graded) -> tuple[str, list[Any]]:
    sql = " AND c.course_id = ?"
    params: list[Any] = [course_id]
    if kinds:
        sql += f" AND m.kind IN ({','.join('?' for _ in kinds)})"
        params.extend(kinds)
    if material_ids:
        sql += f" AND c.material_id IN ({','.join('?' for _ in material_ids)})"
        params.extend(int(i) for i in material_ids)
    if exclude_graded:
        sql += (" AND m.graded = 0 AND c.material_id NOT IN "
                "(SELECT material_id FROM graded_items WHERE material_id IS NOT NULL)")
    return sql, params


def _source(row: dict[str, Any], score: float) -> Source:
    return {
        "chunk_id": row["id"], "material_id": row["material_id"], "material_title": row["material_title"],
        "material_kind": row["material_kind"], "loc_kind": row["loc_kind"], "loc_start": row["loc_start"],
        "loc_label": row["loc_label"], "heading": row["heading"], "text": row["text"],
        "href": locations.href_for({"id": row["id"], "material_id": row["material_id"]}),
        "score": round(float(score), 4),
    }


_SELECT = """SELECT c.id, c.material_id, c.heading, c.text, c.loc_kind, c.loc_start, c.loc_label,
                    m.title AS material_title, m.kind AS material_kind"""


def search(store, course_id: int, query: str, *, limit: int = 8, kinds: list[str] | None = None,
           material_ids: list[int] | None = None, exclude_graded: bool = False) -> list[Source]:
    """Best chunks for a free-text question. Higher score is better. Headings count double."""
    terms = query_terms(query)
    if not terms or limit <= 0:
        return []
    extra, fparams = _filters(course_id, kinds, material_ids, exclude_graded)
    match = fts_query(query)
    rows: list[tuple[dict[str, Any], float]] = []
    try:
        res = store._rows(
            f"""{_SELECT}, bm25(chunks_fts, 1.0, 2.5) AS rank
                FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid JOIN materials m ON m.id = c.material_id
                WHERE chunks_fts MATCH ?{extra} ORDER BY rank LIMIT ?""",
            [match, *fparams, limit])
        rows = [(r, -float(r["rank"])) for r in res]
    except Exception:  # a malformed query should never break the app; fall through to LIKE
        rows = []
    if not rows:
        rows = _like_search(store, terms, extra, fparams, limit)
    return [_source(r, s) for r, s in rows]


def _like_search(store, terms: list[str], extra: str, fparams: list[Any], limit: int):
    def esc(t: str) -> str:
        return t.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    conds = " OR ".join("(lower(c.text) LIKE ? ESCAPE '\\' OR lower(c.heading) LIKE ? ESCAPE '\\')" for _ in terms)
    params: list[Any] = []
    for t in terms:
        params += [f"%{esc(t)}%", f"%{esc(t)}%"]
    res = store._rows(f"""{_SELECT} FROM chunks c JOIN materials m ON m.id = c.material_id
                          WHERE ({conds}){extra} LIMIT 500""", [*params, *fparams])
    scored = []
    for r in res:
        text, head = r["text"].lower(), r["heading"].lower()
        score = sum(1.0 for t in terms if t in text) + sum(2.0 for t in terms if t in head)
        scored.append((r, score))
    scored.sort(key=lambda x: (-x[1], x[0]["id"]))
    return scored[:limit]


def around(store, chunk_id: int, before: int = 1, after: int = 1) -> list[dict]:
    """The chunk and its neighbours in the same material, in reading order."""
    ch = store._row("SELECT material_id, ord FROM chunks WHERE id = ?", (chunk_id,))
    if not ch:
        return []
    rows = store._rows(
        """SELECT c.*, m.title AS material_title, m.kind AS material_kind FROM chunks c
           JOIN materials m ON m.id = c.material_id
           WHERE c.material_id = ? AND c.ord BETWEEN ? AND ? ORDER BY c.ord""",
        (ch["material_id"], ch["ord"] - max(0, before), ch["ord"] + max(0, after)))
    for r in rows:
        r["href"] = locations.href_for(r)
    return rows
