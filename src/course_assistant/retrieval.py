"""Search a course's chunks: FTS5 with bm25 ranking, a LIKE fallback, and neighbouring-chunk context."""
from __future__ import annotations

import math
import re
from datetime import date
from typing import Any

from . import locations

Source = dict[str, Any]
DEBUG: list | None = None
BASE_W, COV_W, HEAD_W = 2.2, 2.4, 2.0  # weights of: the FTS rank, how many of the question's words are in a passage, in its heading

STOPWORDS = frozenset("""
a about above after again all also am an and any are as at be because been before being below between both but by
can could did do does doing down during each few for from further had has have having he her here hers him his how
i if in into is it its itself just me more most my no nor not of off on once only or other our out over own same she
should so some such than that the their theirs them then there these they this those through to too under until up
very was we were what when where which while who whom why will with would you your yours explain tell please give
show describe define mean means nothing anything something
""".split())

_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")
# 'A^TA' is A-transpose times A, not the word 'TA': split it so a search for TA (teaching assistant) cannot match it
_GLUE = re.compile(r"\^\s*\{?\s*T\s*\}?(?=[A-Za-z])")


def query_terms(question: str, max_terms: int = 14) -> list[str]:
    """Words worth searching for: no punctuation, no FTS operators, no stopwords, unique, in order."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in _TOKEN.findall(_GLUE.sub("^T ", question or "")):
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
    q = _GLUE.sub("^T ", question or "")
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
    return re.findall(r"[a-z0-9]+", _GLUE.sub("^T ", text or "").lower())


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



# ---------------------------------------------------------------- query intent
_ADMIN_WORDS = frozenset("""exam exams midterm midterms final finals quiz quizzes grade grades grading graded schedule
scheduled calendar date dates deadline deadlines due when policy policies late penalty drop dropped weight weighted
percent points syllabus attendance office hours textbook required extension cutoff cutoffs""".split())
_DATE_WORDS = frozenset("when date dates deadline deadlines due schedule scheduled calendar day days time".split())
_PRACTICE_WORDS = frozenset("exercise exercises practice homework drill drills".split())
_OVERVIEW_WORDS = frozenset("overview introduction intro summary preface outline objectives".split())
_GENERIC_TOPIC = frozenset("""offering offerings term semester year course class lecture lectures week the
past previous last""".split())
_ADMIN_HEAD = re.compile(
    r"^\W*(exams?|midterms?|finals?(\s+exam)?|grading|grades?|grade breakdown|assessment|lecture schedule|"
    r"course schedule|class schedule|weekly schedule|schedule|calendar|important dates|policies|late policy|"
    r"attendance|office hours|required (texts?|materials?)|textbooks?)\W*$", re.I)
_ADMIN_LEAD = re.compile(r"^\W*(exams?|grading|lecture schedule|course schedule|schedule|calendar)\s*[.:]", re.I)
_INTRO_HEAD = re.compile(r"^\W*(introduction|intro|overview|learning objectives|chapter outline|preface|"
                         r"chapter \d+ introduction)\W*$", re.I)
_INTRO_LEAD = re.compile(r"^\W*(in this (section|chapter|unit|lecture|module)|this (section|chapter|unit) "
                         r"(introduces|covers|will)|introduction\b|learning objectives|by the end of this|"
                         r"chapter \d+ introduction)", re.I)
_EXERCISE_HEAD = re.compile(r"(exercises?|review questions|practice (problems|questions)|homework( problems)?)\W*$", re.I)
_EXERCISE_LEAD = re.compile(r"^\W*(end of chapter )?(exercises?|review questions)\b", re.I)
_DEFINE_Q = re.compile(r"^\s*(?:what(?:'s|\s+is|\s+are|\s+does|\s+do)\b|define\b|definition of\b|meaning of\b|"
                       r"explain\b|describe\b|tell me about\b)|\bdefin(?:e|es|ition)\b|\bwhat\s+(?:is|are)\s+an?\b", re.I)
_MAIN_ARG = re.compile(r"\b(main|central|key|big)\s+(argument|point|idea|claim|thesis)\b|\bthesis\b|\bargues?\b|"
                       r"\bsummar(?:y|ize|ise)\b|\bgist\b|\bwhat\s+is\s+.{0,40}\babout\b", re.I)
_WANTS_HEADER = re.compile(r"\b(who wrote|author|authors|written by|published|publication|when was .* (written|published)|"
                           r"subtitle|byline|newspaper)\b", re.I)


_PAST_RX = re.compile(r"\b(past|previous|previously|earlier|prior|old|last\s+(?:year|term|semester|time)|"
                      r"offerings?|(?:in|from)\s+20\d\d)\b", re.I)
_TERM_RX = re.compile(r"\b(fall|autumn|spring|summer|winter)[\s\-_']*((?:20)?\d{2})\b", re.I)
_SEASONS_WORDS = frozenset("fall autumn spring summer winter".split())


def _term_key(text: str) -> str:
    m = _TERM_RX.search(text or "")
    if not m:
        return ""
    year = m.group(2)
    return f"{m.group(1).lower().replace('autumn', 'fall')} {year if len(year) == 4 else '20' + year}"


def _named_offerings(store, query: str, extra: str, fparams: list[Any]) -> dict[int, str]:
    """{material id: 'fall 2018'} for every material whose title or file name carries a term the question names
    ('the Fall 2018 offering', 'the syllabus for Fall 2026'). {} when the question names no term."""
    terms = {_term_key(m.group(0)) for m in _TERM_RX.finditer(query or "")}
    terms.discard("")
    if not terms:
        return {}
    out: dict[int, str] = {}
    for m in store._rows("SELECT DISTINCT m.id, m.title, m.original_name FROM materials m JOIN chunks c "
                         "ON c.material_id = m.id WHERE 1=1" + extra, fparams):
        t = _term_key(f"{m['title']} {m.get('original_name') or ''}")
        if t in terms:
            out[m["id"]] = t
    return out


def _stems(words) -> set[str]:
    return {_stem(w) for w in words}


def query_intent(query: str, nq: dict[str, Any] | None = None) -> dict[str, Any]:
    """What a question is about, apart from its topic words: dates, definitions, overviews, practice."""
    nq = nq or normalize_query(query)
    toks = set(_words(query))
    tstems = _stems(toks)
    admin = bool(toks & _ADMIN_WORDS) or bool(tstems & _stems(_ADMIN_WORDS))
    date = bool(toks & _DATE_WORDS)
    return {
        "admin": admin, "date": date,
        "practice": bool(toks & _PRACTICE_WORDS),
        "overview": bool(toks & _OVERVIEW_WORDS),
        "define": bool(_DEFINE_Q.search(query or "")),
        "main_arg": bool(_MAIN_ARG.search(query or "")),
        "wants_header": bool(_WANTS_HEADER.search(query or "")),
        "past": bool(_PAST_RX.search(query or "")) or bool(_TERM_RX.search(query or "")),
    }


# ---------------------------------------------------------------- chunk kinds
_SALUTE = re.compile(r"to the people of the state of new york|from the (new york )?(packet|independent journal|"
                     r"daily advertiser)|^\W*the federalist\b", re.I)
_TITLE_LEAD = re.compile(r"^\W*(federalist(\s+papers)?\s*(no\.?|number)?\s*\d+|the federalist)\b", re.I)


def is_title_only(text: str, heading: str = "") -> bool:
    """A chunk that is only a title (a title slide, a bare heading)."""
    t = (text or "").strip()
    w = _words(t)
    if not w:
        return True
    if len(w) <= 6 and (t.strip(" .:").lower() == (heading or "").strip(" .:").lower() or re.fullmatch(
            r"(?:lecture|lec|week|chapter|unit|part|section|class)\s*#?\d+[\s:.\-]*(?:\w+\s*){0,3}", t.strip(), re.I)):
        return True
    return False


def is_header_chunk(text: str, title: str = "") -> bool:
    """A chunk that is a document's heading block: title, byline, dateline, salutation and no argument."""
    t = (text or "").strip()
    w = _words(t)
    if not w or len(w) > 80:
        return False
    if _SALUTE.search(t):
        return True
    if (title and t.lower().startswith(title.strip().lower()[:30]) and len(w) <= 40) or _TITLE_LEAD.match(t):
        sents = [x for x in re.split(r"(?<=[.!?])\s+", t) if len(_words(x)) >= 14 and not x.isupper()]
        return not sents or len(w) <= 40
    return False


def _is_admin_chunk(r: dict[str, Any]) -> bool:  # a row or a source: both have heading and text
    return bool(_ADMIN_HEAD.match(r["heading"] or "") or _ADMIN_LEAD.match(r["text"] or ""))


def _is_intro_chunk(r: dict[str, Any]) -> bool:
    return bool(_INTRO_HEAD.match(r["heading"] or "") or _INTRO_LEAD.match(r["text"] or ""))


def _is_exercise_chunk(r: dict[str, Any]) -> bool:
    return bool(_EXERCISE_HEAD.search(r["heading"] or "") or _EXERCISE_LEAD.match(r["text"] or ""))


def _definition_score(terms: list[str], r: dict[str, Any]) -> float:
    """How much a chunk reads as the definition of one of the terms (0 to 2). The chunk must also hold at least
    half of the question's terms, so 'a defective matrix is...' does not define 'projection matrix'."""
    text = r["text"] or ""
    have = {_stem(w) for w in _words(r["heading"] + " " + text)}
    if sum(1 for t in terms if _stem(t) in have) * 2 < len(terms):
        return 0.0
    best = 0.0
    for t in terms:
        if len(t) < 3:
            continue
        stem = re.escape(_stem(t))
        w = rf"\b{stem}\w*"
        sc = 0.0
        if re.search(rf"{w}(?:\s*\([^)]{{0,14}}\))?\s+(?:is|are|means|refers to|denotes|is called|is defined as)\b", text, re.I):
            sc = 2.0
        elif re.search(rf"\b(?:defined as|we call|is called|known as|termed)\b[^.]{{0,40}}{w}|{w}[^.]{{0,40}}"
                       rf"\b(?:defined as|we call|is called|known as)\b", text, re.I):
            sc = 2.0
        elif re.search(rf"\bby\s+(?:an?\s+)?{w}\s+I\s+(?:understand|mean)\b", text, re.I):
            sc = 2.0
        elif re.search(rf"\*\*[^*]*{stem}[^*]*\*\*", text, re.I):
            sc = 1.5
        elif re.search(r"\bdefinition\b", text + " " + (r["heading"] or ""), re.I) and re.search(w, text, re.I):
            sc = 1.5
        elif re.search(rf"\b{stem}\w*\s+(?:is|are)\b", text, re.I):
            sc = 1.0
        best = max(best, sc)
    return best


_DATE_RX = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|"
    r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?\s+\d{1,2}\b|\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}\b|"
    r"\b(?:mon|tues?|wed(?:nes)?|thu(?:rs)?|fri|sat(?:ur)?|sun)(?:day)?\b", re.I)


def has_date(text: str) -> bool:
    return bool(_DATE_RX.search(text or ""))


# ---------------------------------------------------------------- named documents
_ID_WORDS = {"federalist": "federalist", "federalists": "federalist", "chapter": "chapter", "chapters": "chapter",
             "ch": "chapter", "week": "week", "weeks": "week", "unit": "unit", "units": "unit", "lecture": "lecture",
             "lectures": "lecture", "lec": "lecture", "paper": "paper", "papers": "paper", "section": "section",
             "sections": "section", "part": "part", "parts": "part", "lab": "lab", "labs": "lab",
             "problem": "problem", "set": "set"}
_ID_LIST = re.compile(
    r"\b(federalists?|chapters?|ch|weeks?|units?|lectures?|lec|papers?|sections?|parts?|labs?)\b\.?\s*(?:papers?\s*)?"
    r"(?:no\.?|number|#)?\s*(\d+)((?:\s*(?:,|and|&|or|to|-)\s*(?:(?:no\.?|number|#)\s*)?\d+)*)", re.I)
_TITLE_ID = re.compile(r"^\W*(federalist(?:\s+papers?)?|chapter|week|unit|lecture|lec|paper|section|part|lab)\b\.?\s*"
                       r"(?:no\.?|number|#)?\s*(\d+)\b", re.I)


def query_doc_ids(query: str) -> set[tuple[str, int]]:
    """('chapter', 7), ('federalist', 10) ... for every numbered document the question names.
    'Federalist No. 10 and 51 and Chapters 7 and 8' names federalist 10 and 51, chapter 7 and 8."""
    out: set[tuple[str, int]] = set()
    for m in _ID_LIST.finditer(query or ""):
        word = _ID_WORDS.get(m.group(1).lower(), m.group(1).lower())
        out.add((word, int(m.group(2))))
        for n in re.findall(r"\d+", m.group(3) or ""):
            out.add((word, int(n)))
    return out


def title_doc_id(title: str) -> tuple[str, int] | None:
    m = _TITLE_ID.match(title or "")
    if not m:
        return None
    w = m.group(1).lower().split()[0]
    return _ID_WORDS.get(w, w), int(m.group(2))


# ---------------------------------------------------------------- lecture references
_MONTHS = {m: i for i, names in enumerate(
    ["jan january", "feb february", "mar march", "apr april", "may", "jun june", "jul july", "aug august",
     "sep sept september", "oct october", "nov november", "dec december"], start=1) for m in names.split()}
_WEEKDAYS = {"monday": 0, "mon": 0, "tuesday": 1, "tue": 1, "tues": 1, "wednesday": 2, "wed": 2, "thursday": 3,
             "thu": 3, "thur": 3, "thurs": 3, "friday": 4, "fri": 4}
_LECTURE_WORD = re.compile(r"\b(lectures?|lec|class|classes|session)\b", re.I)
_LEC_NUM = re.compile(r"\b(?:lectures?|lec\.?|class|session)\s*#?\s*0*(\d{1,2})\b", re.I)
_MONTH_DAY = re.compile(r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
                        r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?\s+(\d{1,2})(?:st|nd|rd|th)?\b", re.I)
_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_SLASH = re.compile(r"\b(\d{1,2})/(\d{1,2})\b")
_WEEKDAY_RX = re.compile(r"\b(" + "|".join(sorted(_WEEKDAYS, key=len, reverse=True)) + r")(?:'s|s)?\b", re.I)
_LECTURE_NOISE = frozenset("""lecture lectures lec class classes session primer fill notes note todo cover covered
covers did we do done from the in on of for what happened missed miss catch up summary summarize recap today last
yesterday monday tuesday wednesday thursday friday mon tue tues wed thu thur thurs fri s""".split()) | set(_MONTHS)


def lecture_reference(query: str) -> dict[str, Any] | None:
    """The lecture a question names: {'number': 20}, {'date': (month, day)}, {'iso': 'YYYY-MM-DD'} or {'weekday': 1}.
    Needs a lecture word (lecture, class, session) beside the number, date or weekday, so 'due Thursday' is not one."""
    q = query or ""
    ref: dict[str, Any] = {}
    m = _LEC_NUM.search(q)
    if m:
        ref["number"] = int(m.group(1))
    if _LECTURE_WORD.search(q):
        iso = _ISO.search(q)
        md = _MONTH_DAY.search(q)
        sl = _SLASH.search(q)
        wd = _WEEKDAY_RX.search(q)
        if iso:
            ref["iso"] = f"{iso.group(1)}-{iso.group(2)}-{iso.group(3)}"
        elif md:
            ref["date"] = (_MONTHS[md.group(1).lower().rstrip('.')], int(md.group(2)))
        elif sl and 1 <= int(sl.group(1)) <= 12 and 1 <= int(sl.group(2)) <= 31:
            ref["date"] = (int(sl.group(1)), int(sl.group(2)))
        elif wd and "number" not in ref:
            ref["weekday"] = _WEEKDAYS[wd.group(1).lower()]
    return ref or None


def _lecture_number_of(text: str) -> int | None:
    m = re.search(r"\b(?:lecture|lec\.?|class|session)\s*#?\s*0*(\d{1,2})\b", text or "", re.I)
    return int(m.group(1)) if m else None


def _event_matches(ev: dict[str, Any], ref: dict[str, Any], today: str) -> bool:
    if "number" in ref:
        return _lecture_number_of(f"{ev.get('title', '')}") == ref["number"] or _lecture_number_of(
            ev.get("topics", "")) == ref["number"]
    d = ev.get("date", "")
    if "iso" in ref:
        return d == ref["iso"]
    if "date" in ref:
        try:
            return (int(d[5:7]), int(d[8:10])) == ref["date"]
        except ValueError:
            return False
    if "weekday" in ref:
        try:
            return date.fromisoformat(d).weekday() == ref["weekday"]
        except ValueError:
            return False
    return False


def lecture_events(store, course_id: int, query: str, today: str | None = None) -> list[dict[str, Any]]:
    """Lecture events the question names, [] when it names none. A weekday means the latest such lecture
    that has material linked to it and is not in the future."""
    ref = lecture_reference(query)
    if not ref:
        return []
    today = today or date.today().isoformat()
    evs = [e for e in store.events(course_id) if e.get("kind") == "lecture"]
    hit = [e for e in evs if _event_matches(e, ref, today)]
    if "weekday" in ref and hit:
        linked = [e for e in hit if _lecture_materials(store, course_id, e)]
        past = [e for e in (linked or hit) if e["date"] <= today] or (linked or hit)
        best = max(e["date"] for e in past)
        hit = [e for e in past if e["date"] == best]
    return hit


def _lecture_materials(store, course_id: int, ev: dict[str, Any]) -> list[dict[str, Any]]:
    """Materials linked to a lecture event, dated that day, or named for its number and not tied to another lecture."""
    out, seen = [], set()
    mats = store._rows("SELECT * FROM materials WHERE course_id = ? ORDER BY id", (course_id,))
    for m in mats:
        if m.get("event_id") == ev["id"] or (m.get("lecture_date") and m["lecture_date"] == ev["date"] and not m.get("event_id")):
            out.append(m)
            seen.add(m["id"])
    num = _lecture_number_of(ev.get("title", "")) or _lecture_number_of(ev.get("topics", ""))
    if num is not None:
        rx = re.compile(rf"(?<![a-z0-9])(?:lecture|lec|class|session)[\s_\-#.]*0*{num}(?!\d)", re.I)
        for m in mats:
            if m["id"] in seen or m.get("event_id") or m.get("lecture_date"):
                continue
            if rx.search(" ".join(str(m.get(k, "")) for k in ("title", "original_name"))):
                out.append(m)
                seen.add(m["id"])
    return out


def lectures_recent_first(store, course_id: int, today: str | None = None) -> list[dict[str, Any]]:
    """Lecture events for a dropdown: lectures already held first, newest first, then the coming ones, soonest first.
    A repeated event (two copies of a syllabus) shows once and 'No class' rows are left out."""
    today = today or date.today().isoformat()
    seen: set[tuple[str, str]] = set()
    evs = []
    for e in store.events(course_id):
        if e.get("kind") != "lecture" or re.search(r"\b(no class|holiday|break|cancel+ed)\b", e.get("title", ""), re.I):
            continue
        key = (e["date"], re.sub(r"\W+", " ", e.get("title", "").lower()).strip())
        if key in seen:
            continue
        seen.add(key)
        evs.append(e)
    past = sorted((e for e in evs if e["date"] <= today), key=lambda e: (e["date"], e["id"]), reverse=True)
    coming = sorted((e for e in evs if e["date"] > today), key=lambda e: (e["date"], e["id"]))
    return past + coming


# ---------------------------------------------------------------- clauses
def _mask_matrices(q: str) -> tuple[str, list[str]]:
    held: list[str] = []

    def keep(m):
        held.append(m.group(0))
        return f"\x00{len(held) - 1}\x00"

    return _MATRIX.sub(keep, q), held


def _unmask(q: str, held: list[str]) -> str:
    return re.sub(r"\x00(\d+)\x00", lambda m: held[int(m.group(1))], q)


_CLAUSE_SPLIT = re.compile(r"\s*(?:;|\?\s+(?=\S)|,\s+(?:and\s+|but\s+|then\s+)?|\s+(?:and|but|then|also|plus)\s+)\s*", re.I)


def split_clauses(question: str) -> list[str]:
    """A two-part question as its parts ('formula, and when is A^T A invertible' gives two). A part with fewer than
    two search words joins its neighbour, so 'trace and determinant of A' stays whole. [] or one item means no split."""
    masked, held = _mask_matrices((question or "").strip())
    raw = [p for p in _CLAUSE_SPLIT.split(masked) if p and p.strip(" ?.!")]
    parts = [_unmask(p, held).strip(" ?.!") for p in raw]
    if len(parts) < 2:
        return [question]
    out: list[str] = []
    carry = ""
    for p in parts:
        cur = f"{carry} {p}".strip() if carry else p
        if len(normalize_query(cur)["core"]) < 2:
            carry = cur
            continue
        out.append(cur)
        carry = ""
    if carry:
        if out:
            out[-1] = f"{out[-1]} {carry}"
        else:
            return [question]
    return out if len(out) >= 2 else [question]


# ---------------------------------------------------------------- the search
def _chunk_rows(store, material_ids: list[int], extra: str, fparams: list[Any]) -> list[dict[str, Any]]:
    if not material_ids:
        return []
    ph = ",".join("?" for _ in material_ids)
    return store._rows(f"{_SELECT}, c.ord AS ord FROM chunks c JOIN materials m ON m.id = c.material_id "
                       f"WHERE c.material_id IN ({ph}){extra} ORDER BY c.material_id, c.ord",
                       [*material_ids, *fparams])


def _contained(a: set[str], b: set[str], frac: float = 0.7) -> bool:
    if not a or not b:
        return False
    return len(a & b) / min(len(a), len(b)) >= frac


def _spread(items: list[Any], n: int) -> list[Any]:
    """n items spread evenly over the list, in order."""
    if n >= len(items):
        return list(items)
    if n <= 0:
        return []
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def _lecture_pool(store, course_id: int, query: str, evs: list[dict[str, Any]], cap: int, extra: str,
                  fparams: list[Any], nq: dict[str, Any]) -> list[Source]:
    """Chunks of the materials linked to the lectures a question names: the ones about the question's topic first,
    then the rest in reading order. Title-only slides and speaker-note repeats are skipped."""
    mats: list[dict[str, Any]] = []
    for ev in evs:
        for m in _lecture_materials(store, course_id, ev):
            if m["id"] not in {x["id"] for x in mats}:
                mats.append(m)
    if not mats:
        return []
    # recordings first, then slides and notes
    mats.sort(key=lambda m: (0 if m["kind"] == "recording" else 1, m["id"]))
    rows = _chunk_rows(store, [m["id"] for m in mats], extra, fparams)
    rows = [r for r in rows if not is_manifest_noise(r["text"]) and not is_title_only(r["text"], r["heading"])
            and not is_header_chunk(r["text"], r["material_title"])]
    kept: list[dict[str, Any]] = []
    shs: list[set[str]] = []
    for r in rows:
        text = re.sub(r"^\s*speaker notes:\s*", "", r["text"], flags=re.I)
        sh = _shingles(text)
        if any(_contained(sh, k) for k in shs):
            continue
        kept.append(r)
        shs.append(sh)
    if not kept:
        return []
    topic = [t for t in nq["strong"] if t not in _LECTURE_NOISE and not t.isdigit()]
    tstems = {_stem(t) for t in topic}
    scored = []
    for r in kept:
        body = {_stem(w) for w in _words(r["heading"] + " " + r["text"])}
        hits = len(tstems & body)
        scored.append((hits, r))
    topical = sorted((x for x in scored if x[0] > 0), key=lambda x: (-x[0], x[1]["material_id"], x[1]["ord"]))
    rest = [r for h, r in scored if h == 0]
    if topic:
        picked = [r for _, r in topical]
        cap = max(1, cap)
    else:
        picked = []
    pool = picked[:cap]
    if len(pool) < cap:
        left = [r for r in rest]
        # spread what is left over every material, recordings first
        by_mat: dict[int, list[dict[str, Any]]] = {}
        for r in left:
            by_mat.setdefault(r["material_id"], []).append(r)
        need = cap - len(pool)
        per = max(1, -(-need // max(1, len(by_mat))))
        extras: list[dict[str, Any]] = []
        for mid in [m["id"] for m in mats if m["id"] in by_mat]:
            extras += _spread(by_mat[mid], per)
        pool += extras[:need]
    out = []
    for i, r in enumerate(pool):
        out.append(_source(r, 20.0 - 0.1 * i))
    return out


# words that say what kind of answer is wanted, not what it is about: they never decide which passage is best
_ANSWER_KIND = frozenset(_stem(w) for w in """formula formulas equation equations definition definitions example examples
rule rules steps step method methods way ways idea ideas concept concepts meaning relationship relation difference
differences compare comparison differ""".split())


def _term_weights(store, terms: list[str], extra: str, fparams: list[Any]) -> dict[str, float]:
    """How telling each question word is in this course: a word found in few passages ('projection') counts more than
    one found nearly everywhere ('matrix', 'formula'). 1.0 for the rarest, down to 0.25 for a word in every passage."""
    n = store._row("SELECT count(*) AS n FROM chunks c JOIN materials m ON m.id = c.material_id WHERE 1=1" + extra,
                   fparams)
    total = max(1, int((n or {}).get("n", 1)))
    out: dict[str, float] = {}
    for t in terms:
        try:
            row = store._row(
                "SELECT count(*) AS n FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid "
                "JOIN materials m ON m.id = c.material_id WHERE chunks_fts MATCH ?" + extra, [_fts_term(t), *fparams])
            df = int((row or {}).get("n", 0))
        except Exception:
            df = 0
        out[t] = max(0.25, min(1.0, math.log((total + 1) / (df + 1)) / math.log(total + 1) * 1.6)) if df else 1.0
        if _stem(t) in _ANSWER_KIND or _stem(t) in _WEAK:
            out[t] = min(out[t], 0.3)
    return out


def _has_term(t: str, words: set[str], stems: set[str]) -> bool:
    return _stem(t) in stems or (len(t) >= 4 and any(w.startswith(t) for w in words))


def _search_lane(store, course_id: int, query: str, intent: dict[str, Any], extra: str, fparams: list[Any],
                 limit: int) -> list[Source]:
    """One question, scored: words, titles, definitions, and what kind of chunk it is."""
    nq = normalize_query(query)
    core, related = nq["strong"], nq["related"]
    if not nq["core"]:
        return []
    qstems, qnums = nq["stems"], nq["numbers"]
    pool: dict[int, dict[str, Any]] = {}  # chunk id -> row
    base: dict[int, float] = {}
    boost: dict[int, float] = {}

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
    want_ids = query_doc_ids(query)
    named: set[int] = set()
    if want_ids:
        for m in store._rows("SELECT DISTINCT m.id, m.title FROM materials m JOIN chunks c ON c.material_id = m.id "
                             "WHERE 1=1" + extra, fparams):
            if title_doc_id(m["title"]) in want_ids:
                titles[m["id"]] = max(titles.get(m["id"], 0.0), 1.0)
                named.add(m["id"])
    for mid, tsc in titles.items():
        if tsc >= 0.6:
            named.add(mid)
    for mid in named:
        rows = store._rows(f"{_SELECT} FROM chunks c JOIN materials m ON m.id = c.material_id "
                           f"WHERE c.material_id = ?{extra} ORDER BY c.ord LIMIT 40", [mid, *fparams])
        add([(r, 0.0) for r in rows])
    offerings = _named_offerings(store, query, extra, fparams)  # {material id: 'fall 2018'} for a term the question names
    if intent["date"] or (intent["admin"] and offerings):
        # a named offering's dates: every chunk of it that carries a date (and the topic, if the question has one)
        topic = {_stem(t) for t in core if t not in _DATE_WORDS and t not in _GENERIC_TOPIC and not t.isdigit()
                 and t not in _SEASONS_WORDS}
        for mid in {*offerings, *(m for m, tsc in titles.items() if tsc > 0)}:
            for r in store._rows(f"{_SELECT} FROM chunks c JOIN materials m ON m.id = c.material_id "
                                 f"WHERE c.material_id = ?{extra} ORDER BY c.ord", [mid, *fparams]):
                head = {_stem(w) for w in _words(r["heading"])}
                words = head | {_stem(w) for w in _words(r["text"])}
                if has_date(r["text"] + " " + r["heading"]) and (not topic or topic & words):
                    add([(r, 0.0)])
                    boost[r["id"]] = max(boost.get(r["id"], 0.0), 3.5 if topic & head else 2.0)
    if not pool:
        return []
    top = max(base.values()) or 1.0
    weights = _term_weights(store, core, extra, fparams)
    # a definition is wanted of the question's telling words ('projection'), not of 'matrix' or 'formula'
    maxw = max((weights[t] for t in core if t not in _GENERIC_TOPIC), default=1.0)
    defterms = [t for t in core if t not in _GENERIC_TOPIC and weights.get(t, 1.0) >= 0.6 * maxw] if intent["define"] else []
    scored: list[tuple[float, dict[str, Any]]] = []
    for cid, r in pool.items():
        if is_manifest_noise(r["text"]):
            continue
        bwords = _words(r["text"])
        body = {_stem(w) for w in bwords}
        head = {_stem(w) for w in _words(r["heading"])}
        titl = {_stem(w) for w in _words(r["material_title"])}
        cs = [_stem(t) for t in core]
        present = sum(1 for t in cs if t in body or t in head or t in titl)
        wsum = sum(weights[t] for t in core) or 1.0
        wpresent = sum(weights[t] for t, c in zip(core, cs) if c in body or c in head or c in titl) / wsum
        whead = sum(weights[t] for t, c in zip(core, cs) if c in head) / wsum
        if (present == 0 and not titles.get(r["material_id"]) and cid not in boost and r["material_id"] not in named
                and not any(_has_term(t, set(bwords), body) for t in [*core, *related])):
            continue  # found by a stray token (the 'ta' in A^TA), not by a word of the question
        score = BASE_W * (base.get(cid, 0.0) / top)
        score += (COV_W if present == len(cs) else 0.6 * COV_W * wpresent)
        score += HEAD_W * whead
        score += 4.5 * titles.get(r["material_id"], 0.0)
        score += _number_score(qnums, r["text"])
        score += boost.get(cid, 0.0)
        if defterms:
            score += 1.5 * _definition_score(defterms, r)
        if r["material_kind"] == "past_offering" and r["material_id"] not in offerings:
            if not intent["past"]:
                score -= 4.0  # an earlier term's lecture, exam or policy is not this term's, unless it was asked for
            else:
                score -= 0.3 if titles.get(r["material_id"]) else 1.0
        if present == 0 and not titles.get(r["material_id"]):
            score -= 1.0  # only a related word matched
        if not intent["admin"] and _is_admin_chunk(r):
            score -= 1.5  # schedule, grading and exam chunks answer questions about dates and grades
        if not intent["overview"] and _is_intro_chunk(r):
            score -= 1.5
        if not intent["practice"] and _is_exercise_chunk(r):
            score -= 2.5
        if is_title_only(r["text"], r["heading"]) and not titles.get(r["material_id"]) == 1.0:
            score -= 2.0
        elif is_title_only(r["text"], r["heading"]):
            score -= 1.0
        if not intent["wants_header"] and is_header_chunk(r["text"], r["material_title"]):
            score -= 6.0 if (intent["main_arg"] or r["material_id"] in named) else 1.5
        if DEBUG is not None:
            DEBUG.append((cid, round(score, 2), round(3.0 * (base.get(cid, 0.0) / top), 2), round(wpresent, 2), round(whead, 2)))
        scored.append((score, r))
    scored.sort(key=lambda x: (-x[0], x[1]["id"]))
    ranked = _dedupe(scored, max(limit, 1) * 3)
    named_ids = [m for m in dict.fromkeys(r["material_id"] for _, r in ranked) if m in named and titles.get(m, 0) >= 0.6]
    if len(named_ids) >= 2 and want_ids:
        ranked = _fair_share(ranked, named_ids, limit)
    if len(qnums) >= 3 and ranked and _number_score(qnums, ranked[0][1]["text"]) >= 3.0:
        nxt = store._row("SELECT id FROM chunks WHERE material_id = ? AND ord = (SELECT ord + 1 FROM chunks WHERE id = ?)",
                         (ranked[0][1]["material_id"], ranked[0][1]["id"]))
        if nxt and nxt["id"] not in {r["id"] for _, r in ranked[:2]}:
            row = store._rows(f"{_SELECT} FROM chunks c JOIN materials m ON m.id = c.material_id WHERE c.id = ?"
                              f"{extra}", [nxt["id"], *fparams])
            if row:  # the line after a matching worked example is where it is solved
                ranked = [ranked[0], (ranked[0][0] - 0.01, row[0])] + [x for x in ranked[1:] if x[1]["id"] != nxt["id"]]
    return [_source(r, sc) for sc, r in ranked[:limit]]


def _dedupe(scored: list[tuple[float, dict[str, Any]]], n: int) -> list[tuple[float, dict[str, Any]]]:
    out, kept = [], []
    for score, r in scored:
        sh = _shingles(r["text"])
        if any(_near_duplicate(sh, k) for k in kept):
            continue
        kept.append(sh)
        out.append((score, r))
        if len(out) >= n:
            break
    return out


def _fair_share(ranked: list[tuple[float, dict[str, Any]]], named_ids: list[int], limit: int) -> list[tuple[float, dict[str, Any]]]:
    """Every named document gets its share of the slots, best chunks first; the rest go by score."""
    share = max(1, limit // len(named_ids))
    guaranteed: list[tuple[float, dict[str, Any]]] = []
    for mid in named_ids:
        guaranteed += [x for x in ranked if x[1]["material_id"] == mid][:share]
    ids = {x[1]["id"] for x in guaranteed}
    return sorted(guaranteed, key=lambda x: -x[0]) + [x for x in ranked if x[1]["id"] not in ids]


def _round_robin(lanes: list[list[Source]], limit: int, skip: list[set[str]]) -> list[Source]:
    out: list[Source] = []
    seen_ids: set[int] = set()
    kept = list(skip)
    lanes = [list(x) for x in lanes]
    while any(lanes) and len(out) < limit:
        for lane in lanes:
            while lane:
                s = lane.pop(0)
                sh = _shingles(s["text"])
                if s["chunk_id"] in seen_ids or any(_near_duplicate(sh, k) for k in kept):
                    continue
                seen_ids.add(s["chunk_id"])
                kept.append(sh)
                out.append(s)
                break
            if len(out) >= limit:
                break
    return out


def search(store, course_id: int, query: str, *, limit: int = 8, kinds: list[str] | None = None,
           material_ids: list[int] | None = None, exclude_graded: bool = False) -> list[Source]:
    """Best chunks for a free-text question. Higher score is better.

    A question that names a lecture (its number, date or weekday) gets that lecture's chunks first. Otherwise two
    passes (every important word, then any word or related word) are merged and re-ranked with heading and title
    matches, definition passages, worked-example numbers and a small related-terms list. A two-part question is
    searched part by part and merged round robin. Schedule, grading, intro, exercise and title-only chunks rank low
    unless the question is about them. exclude_graded drops chunks of materials that hold graded items.
    """
    nq = normalize_query(query)
    if not nq["core"] or limit <= 0:
        return []
    extra, fparams = _filters(course_id, kinds, material_ids, exclude_graded)
    intent = query_intent(query, nq)
    lead: list[Source] = []
    try:
        evs = lecture_events(store, course_id, query)
    except Exception:  # a bad calendar row must not break search
        evs = []
    if evs:
        topic = [t for t in nq["strong"] if t not in _LECTURE_NOISE and not t.isdigit()]
        cap = limit if not topic else max(1, -(-limit * 3 // 4))
        lead = _lecture_pool(store, course_id, query, evs, cap, extra, fparams, nq)
    clauses = split_clauses(query)
    if lead:
        topic = [t for t in nq["strong"] if t not in _LECTURE_NOISE and not t.isdigit()]
        if not topic:  # a bare 'primer for Lecture 20': search the lecture's own title and topics
            words = " ".join(f"{re.sub(r'^.*?lecture\s*#?\d+\s*[:.\-]*', '', e.get('title', ''), flags=re.I)} "
                             f"{e.get('topics', '')}" for e in evs).strip()
            if words:
                query, clauses = words, [words]
                nq = normalize_query(words)
                intent = query_intent(words, nq)
    if lead:  # the fill comes from the rest of the course, not from the lecture's own materials again
        ids = sorted({s_["material_id"] for s_ in lead})
        extra += f" AND c.material_id NOT IN ({','.join(str(int(i)) for i in ids)})"
    lanes: list[list[Source]] = []
    if len(clauses) >= 2:
        for c in clauses:
            lanes.append(_search_lane(store, course_id, c, query_intent(c), extra, fparams, limit))
    else:
        lanes.append(_search_lane(store, course_id, query, intent, extra, fparams, limit))
    merged = _round_robin(lanes, limit, [])
    if not lead:
        return merged
    kept = [_shingles(s["text"]) for s in lead]
    out = list(lead)
    seen = {s["chunk_id"] for s in lead}
    for s in merged:
        if len(out) >= limit:
            break
        sh = _shingles(s["text"])
        if s["chunk_id"] in seen or any(_near_duplicate(sh, k) or _contained(sh, k, 0.8) for k in kept):
            continue
        if not intent["admin"] and _is_admin_chunk(s):
            continue  # the lecture's own material comes first; the schedule is not what was asked
        if is_title_only(s["text"], s["heading"]):
            continue
        out.append(s)
        seen.add(s["chunk_id"])
    return out[:limit]


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
