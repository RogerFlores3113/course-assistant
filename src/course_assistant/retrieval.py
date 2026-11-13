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
show describe define mean means nothing anything something
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


# ---------------------------------------------------------------- query understanding
_MATRIX = re.compile(r"\[\s*\[[^\]]*\](?:\s*,?\s*\[[^\]]*\])*\s*\]")
_POWER = re.compile(r"\b[A-Za-z]\s*\^\s*(\{?\s*(?:\d+|[a-z])\s*\}?)")
_NUM = re.compile(r"\d+(?:\.\d+)?")
_NUMBER_WORDS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen "
    "eighteen nineteen twenty".split())}
_NUMBER_WORDS.update({"thirty": "30", "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80",
                      "ninety": "90", "negative": "-"})

# Related terms. Keys and values are word stems; a hit on the key also searches the values (at a lower weight).
SYNONYMS: dict[str, list[str]] = {
    "comput": ["calculate", "find", "formula"],
    "calculat": ["compute", "find", "formula"],
    "quick": ["fast", "efficient", "shortcut"],
    "eigenvalu": ["eigenvalue", "characteristic", "polynomial", "lambda"],
    "eigenvector": ["eigenvector", "null", "space"],
    "determin": ["det", "product"],
    "defect": ["diagonalizable", "independent"],
    "exam": ["midterm", "final", "test", "schedule"],
    "midterm": ["exam", "test"],
    "grade": ["grading", "weight", "percent", "points"],
    "due": ["deadline", "submit", "date"],
    "late": ["penalty", "extension"],
    "code": ["function", "julia", "python", "loop"],
    "dilut": ["molarity", "concentration"],
    "molar": ["concentration", "solution", "moles", "liter"],
    "empir": ["formula", "percent", "composition"],
    "republic": ["representative", "government"],
    "faction": ["interest", "majority", "minority"],
}


def _stem(w: str) -> str:
    w = w.lower()
    for suf in ("izations", "ization", "izing", "ized", "izes", "ize", "izable"):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            return w[: -len(suf)] + "iz"
    for suf in ("ations", "ation", "able", "ings", "ing", "ies", "ied", "es", "ed", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)]
    return w


_SPOKEN = {"100": "hundred", "1000": "thousand", **{str(i): w for w, i in {
    "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}.items()}}
_WEAK = frozenset(_stem(w) for w in "compute calculate quickly quick fast find work use get make long way".split())


def _numbers_in_query(q: str) -> list[str]:
    """Numbers worth matching: matrix entries in order, plus bare numbers (not exponents)."""
    out: list[str] = []
    for m in _MATRIX.finditer(q):
        out += _NUM.findall(m.group(0))
    rest = _POWER.sub(" ", _MATRIX.sub(" ", q))
    out += [n for n in _NUM.findall(rest)]
    return out


def normalize_query(question: str) -> dict[str, Any]:
    """Understand a question: core terms (as typed), related terms, numbers and title tokens.

    A^100 becomes 'power powers diagonalization hundred'; [[4,1],[2,3]] becomes 'matrix' plus the numbers 4 1 2 3 for
    worked-example matching. Generic verbs (compute, find, quickly) are kept but never decide the ranking.
    """
    q = question or ""
    boost: list[str] = []  # terms implied by notation: as important as the words typed
    pm = _POWER.search(q)
    if pm:
        boost += ["power", "powers", "diagonalization"]
        exp = re.sub(r"\D", "", pm.group(1))
        if exp and exp in _SPOKEN:
            boost.append(_SPOKEN[exp])
    if re.search(r"\^\s*T\b", q):
        boost.append("transpose")
    if re.search(r"\^\s*\{?\s*-1", q):
        boost.append("inverse")
    numbers = _numbers_in_query(q)
    text = _POWER.sub(" ", q)
    had_matrix = bool(_MATRIX.search(text))
    text = _MATRIX.sub(" matrix ", text)
    core = query_terms(text, max_terms=12)
    if had_matrix and "matrix" not in core:
        core.append("matrix")
    for t in boost:
        if t not in core:
            core.append(t)
    strong = [t for t in core if _stem(t) not in _WEAK] or list(core)
    related: list[str] = []
    seen = set(core)
    for t in strong:
        st = _stem(t)
        for key, vals in SYNONYMS.items():
            if st.startswith(key) or (key.startswith(st) and len(st) >= 5):
                for v in vals:
                    if v not in seen:
                        seen.add(v)
                        related.append(v)
    return {"core": core, "strong": strong, "related": related, "numbers": numbers,
            "stems": {_stem(t) for t in core}}


def fts_query(question: str, max_terms: int = 14) -> str:
    """A safe FTS5 MATCH string: quoted terms ORed together, with a prefix * on terms of 4+ chars."""
    return " OR ".join(_fts_term(t) for t in query_terms(question, max_terms))


def _fts_term(t: str) -> str:
    return f'"{t}"*' if len(t) >= 4 else f'"{t}"'


# ---------------------------------------------------------------- filters and noise
def _filters(course_id: int, kinds, material_ids, exclude_graded) -> tuple[str, list[Any]]:
    sql = " AND c.course_id = ?"
    params: list[Any] = [course_id]
    if kinds:
        sql += f" AND m.kind IN ({','.join('?' for _ in kinds)})"
        params.extend(kinds)
    if material_ids:
        sql += f" AND c.material_id IN ({','.join('?' for _ in material_ids)})"
        params.extend(int(i) for i in material_ids)
    else:
        sql += " AND (m.duplicate_of IS NULL OR m.duplicate_of = 0)"  # a copy of another saved file
    if exclude_graded:
        sql += (" AND m.graded = 0 AND c.material_id NOT IN "
                "(SELECT material_id FROM graded_items WHERE material_id IS NOT NULL)")
    return sql, params


_MANIFEST_MARKS = re.compile(r"manifest_toml_contents|project_toml_contents|pluto_project|pluto_manifest|"
                             r"\[\[deps\.|julia_version\s*=|manifest_format\s*=", re.I)
_TOML_LINE = re.compile(r"^\s*(?:\[\[?[\w.\-\"]+\]\]?|[\w.\-]+\s*=\s*.+)\s*$")
_UUID = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I)


def is_manifest_noise(text: str) -> bool:
    """Chunks that are mostly a generated package manifest (Pluto/Julia TOML) are not course content."""
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return False
    if len(_UUID.findall(text)) >= 5:
        return True
    if not _MANIFEST_MARKS.search(text):
        return False
    toml = sum(1 for ln in lines if _TOML_LINE.match(ln))
    return len(lines) >= 4 and toml / len(lines) >= 0.5


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _shingles(text: str) -> set[str]:
    w = _words(text)
    if len(w) < 4:
        return {" ".join(w)}
    return {" ".join(w[i:i + 3]) for i in range(len(w) - 2)}


def _near_duplicate(a: set[str], b: set[str]) -> bool:
    if not a or not b:
        return False
    return len(a & b) / len(a | b) >= 0.8


# ---------------------------------------------------------------- scoring features
def _numeric_tokens(text: str) -> list[tuple[int, str]]:
    """(token position, digits) for every number in the text, spoken numbers included ('four, one' -> 4, 1)."""
    out = []
    for i, w in enumerate(re.findall(r"-?\d+(?:\.\d+)?|[a-z]+", (text or "").lower())):
        if w in _NUMBER_WORDS and _NUMBER_WORDS[w] != "-":
            out.append((i, _NUMBER_WORDS[w]))
        elif re.fullmatch(r"-?\d+(?:\.\d+)?", w):
            out.append((i, w.lstrip("-")))
    return out


def _number_score(qnums: list[str], text: str) -> float:
    """Worked-example match: the question's numbers in order and close together score high; a share of them, low."""
    if not qnums:
        return 0.0
    toks = _numeric_tokens(text)
    if not toks:
        return 0.0
    digits = [d for _, d in toks]
    pos = [p for p, _ in toks]
    if len(qnums) >= 3:
        k = len(qnums)
        for i in range(len(toks) - k + 1):
            if digits[i:i + k] == qnums and all(pos[i + j + 1] - pos[i + j] <= 6 for j in range(k - 1)):
                return 3.0
    have = set(digits)
    frac = sum(1 for n in qnums if n in have) / len(qnums)
    return (0.8 if len(qnums) >= 3 else 0.5) * frac


def _title_score(title: str, qstems: set[str], qnums: list[str], n_query_terms: int = 0) -> float:
    """How much of the material's title is in the question.

    1.0 when the whole title is asked for ('Federalist No. 10'); a title that is one word of a long question
    ('Determinants' in 'how do trace and determinants relate') scores 0. Two or more shared words give a small partial score.
    """
    ts = [_stem(w) for w in _words(title) if w not in STOPWORDS or w.isdigit()]
    ts = [t for t in dict.fromkeys(ts) if t]
    if not ts:
        return 0.0
    digit_t = [t for t in ts if t.isdigit()]
    hit = sum(1 for t in ts if t in qstems or t in qnums)
    cov = hit / len(ts)
    word_hit = any(t in qstems for t in ts if not t.isdigit())
    wrong_number = any(t not in qnums and t not in qstems for t in digit_t)
    share = hit / max(1, n_query_terms)
    if not wrong_number and cov >= 0.6 and word_hit and (share >= 0.5 or hit >= 3):
        return cov
    if hit >= 2 and not wrong_number:
        return 0.1 * min(hit, 4)  # e.g. 'Fall 2018' in the question and in the title
    return 0.0


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


def _fts_rows(store, match: str, extra: str, fparams: list[Any], limit: int) -> list[tuple[dict[str, Any], float]]:
    res = store._rows(
        f"""{_SELECT}, bm25(chunks_fts, 1.0, 2.5) AS rank
            FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid JOIN materials m ON m.id = c.material_id
            WHERE chunks_fts MATCH ?{extra} ORDER BY rank LIMIT ?""",
        [match, *fparams, limit])
    return [(r, -float(r["rank"])) for r in res]


def _title_pool(store, qstems: set[str], qnums: list[str], nterms: int, extra: str, fparams: list[Any]) -> dict[int, float]:
    """Materials whose title matches the question, as {material_id: title score}."""
    out: dict[int, float] = {}
    for m in store._rows(
            "SELECT DISTINCT m.id, m.title FROM materials m JOIN chunks c ON c.material_id = m.id WHERE 1=1" + extra,
            fparams):
        sc = _title_score(m["title"], qstems, qnums, nterms)
        if sc:
            out[m["id"]] = sc
    return out


def search(store, course_id: int, query: str, *, limit: int = 8, kinds: list[str] | None = None,
           material_ids: list[int] | None = None, exclude_graded: bool = False) -> list[Source]:
    """Best chunks for a free-text question. Higher score is better.

    Two passes (every important word, then any word or related word) are merged and re-ranked with heading and title
    matches, worked-example number matches, and a small related-terms list. Duplicates and manifest noise are dropped.
    """
    nq = normalize_query(query)
    core, related = nq["strong"], nq["related"]
    if not nq["core"] or limit <= 0:
        return []
    qstems, qnums = nq["stems"], nq["numbers"]
    extra, fparams = _filters(course_id, kinds, material_ids, exclude_graded)
    pool: dict[int, dict[str, Any]] = {}  # chunk id -> row
    base: dict[int, float] = {}

    def add(rows, scale: float = 1.0):
        for r, sc in rows:
            pool.setdefault(r["id"], r)
            base[r["id"]] = max(base.get(r["id"], 0.0), sc * scale)

    try:
        add(_fts_rows(store, " OR ".join(_fts_term(t) for t in [*core, *related]), extra, fparams, 120))
        if len(core) >= 2:
            add(_fts_rows(store, " AND ".join(_fts_term(t) for t in core), extra, fparams, 60))
    except Exception:  # a malformed query should never break the app; fall through to LIKE
        pool.clear()
        base.clear()
    if not pool:
        add(_like_search(store, core, extra, fparams, 120))
    titles = _title_pool(store, qstems, qnums, len(nq["core"]), extra, fparams)
    for mid, tsc in titles.items():
        if tsc < 0.6:
            continue
        rows = store._rows(f"{_SELECT} FROM chunks c JOIN materials m ON m.id = c.material_id "
                           f"WHERE c.material_id = ?{extra} ORDER BY c.ord LIMIT 16",
                           [mid, *fparams])
        add([(r, 0.0) for r in rows])
    if not pool:
        return []
    top = max(base.values()) or 1.0
    scored: list[tuple[float, dict[str, Any]]] = []
    for cid, r in pool.items():
        if is_manifest_noise(r["text"]):
            continue
        body = {_stem(w) for w in _words(r["text"])}
        head = {_stem(w) for w in _words(r["heading"])}
        titl = {_stem(w) for w in _words(r["material_title"])}
        cs = [_stem(t) for t in core]
        present = sum(1 for t in cs if t in body or t in head or t in titl)
        in_head = sum(1 for t in cs if t in head)
        score = 3.0 * (base.get(cid, 0.0) / top)
        score += (1.0 if present == len(cs) else 0.6 * present / len(cs))
        score += 1.5 * in_head / len(cs)
        score += 4.5 * titles.get(r["material_id"], 0.0)
        score += _number_score(qnums, r["text"])
        if r["material_kind"] == "past_offering":
            score -= 0.3
        if present == 0 and not titles.get(r["material_id"]):
            score -= 1.0  # only a related word matched
        scored.append((score, r))
    scored.sort(key=lambda x: (-x[0], x[1]["id"]))
    out: list[Source] = []
    kept: list[set[str]] = []
    for score, r in scored:
        sh = _shingles(r["text"])
        if any(_near_duplicate(sh, k) for k in kept):
            continue
        kept.append(sh)
        out.append(_source(r, score))
        if len(out) >= limit:
            break
    return out


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
