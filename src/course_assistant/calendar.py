"""Pull dates (lectures, exams, assignments, quizzes) out of syllabus text.

Conservative on purpose: a line becomes an event only when it has a clear date. Handles "Oct 14", "10/14",
"October 14, 2026", "14 October", "2026-10-14", "Week 3 (Sep 15)", "Midterm 1: Thursday, Oct 9" and table rows
("Sep 15 | Lecture 3 | Eigenvalues").
"""
from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10,
          "nov": 11, "dec": 12}
_MON = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|"
        r"Sept?(?:ember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)")
_ORD = r"(?:st|nd|rd|th)?"
_MDY = re.compile(rf"\b(?P<mon>{_MON})\b\.?\s+(?P<day>\d{{1,2}}){_ORD}\b(?:\s*,?\s*(?P<year>(?:19|20)\d{{2}})\b)?", re.I)
_DMY = re.compile(rf"\b(?P<day>\d{{1,2}}){_ORD}\s+(?:of\s+)?(?P<mon>{_MON})\b\.?(?:\s*,?\s*(?P<year>(?:19|20)\d{{2}})\b)?", re.I)
_NUM = re.compile(r"(?<![\d/.$-])(?P<m>\d{1,2})/(?P<d>\d{1,2})(?:/(?P<y>\d{4}|\d{2}))?(?![\d/])")
_ISO = re.compile(r"\b(?P<y>(?:19|20)\d{2})-(?P<m>\d{2})-(?P<d>\d{2})\b")
_TIME12 = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*([ap])\.?m\b\.?", re.I)
_TIME24 = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")
_WEEKDAY = re.compile(r"\b(?:Mon|Tue|Tues|Wed|Thu|Thur|Thurs|Fri|Sat|Sun)(?:day|sday|nesday|rsday|urday)?\b\.?,?", re.I)
_WEEK = re.compile(r"^\W*(?:week|wk)\s*\.?\s*(\d{1,2})\b", re.I)

_OTHER = re.compile(r"\b(holiday|no class|no lecture|break|recess|thanksgiving|review|office hours?|drop|withdraw|"
                    r"add/drop|registration|reading week|spring|fall|winter)\b", re.I)
_QUIZ = re.compile(r"\b(quiz|quizzes)\b", re.I)
_EXAM = re.compile(r"\b(midterms?|mid-term|exams?|tests?)\b|\bfinal\b(?!\s+(?:project|paper|report|presentation|draft|"
                   r"assignment|submission|portfolio|essay))", re.I)
_ASSIGN = re.compile(r"\b(hw\s*\d*|homeworks?|problem sets?|psets?|ps\s*\d+|assignments?|projects?|labs?|paper|"
                     r"essays?|reports?|proposal|deliverables?|submission|due|presentation)\b", re.I)
_LECTURE = re.compile(r"\b(lectures?|lec\.?\s*\d*|class)\b", re.I)
_CLEAN_EDGES = " \t-–—:;,|·•*>()[]."


def _year_from(hint: Any) -> int:
    if isinstance(hint, int):
        return hint
    if isinstance(hint, (date, datetime)):
        return hint.year
    if isinstance(hint, str):
        m = re.search(r"(?:19|20)\d{2}", hint)
        if m:
            return int(m.group())
    return date.today().year


def _mon(s: str) -> int:
    return MONTHS[s[:3].lower()]


def _find_dates(line: str) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []

    def taken(a: int, b: int) -> bool:
        return any(a < f["end"] and b > f["start"] for f in found)

    for rx in (_ISO, _MDY, _DMY, _NUM):
        for m in rx.finditer(line):
            if taken(m.start(), m.end()):
                continue
            g = m.groupdict()
            if "mon" in g:
                mon_text = g["mon"]
                if mon_text.lower() == "may" and mon_text != "May" and mon_text != "MAY":
                    continue
                month, day = _mon(mon_text), int(g["day"])
                year = int(g["year"]) if g.get("year") else None
            elif rx is _ISO:
                month, day, year = int(g["m"]), int(g["d"]), int(g["y"])
            else:
                month, day = int(g["m"]), int(g["d"])
                tail = line[m.end():m.end() + 8].lstrip().lower()
                if tail.startswith(("of ", "%", "pts", "points", "cup")) or (g["y"] is None and "." in line[m.start():m.end()]):
                    continue
                y = g["y"]
                year = (int(y) + 2000 if len(y) == 2 else int(y)) if y else None
            if not (1 <= month <= 12 and 1 <= day <= 31):
                continue
            found.append({"start": m.start(), "end": m.end(), "month": month, "day": day, "year": year})
    found.sort(key=lambda f: f["start"])
    return found




_RANGE_AP = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(?:([ap])\.?m\b\.?)?\s*(?:-|–|—|to)\s*(\d{1,2})(?::(\d{2}))?\s*([ap])\.?m\b\.?",
                       re.I)
_RANGE_24 = re.compile(r"\b(\d{1,2}):(\d{2})\s*(?:-|–|—|to)\s*(\d{1,2}):(\d{2})\b")


def _hhmm(h: int, m: int) -> str:
    return f"{h:02d}:{m:02d}"


def parse_time(line: str, exclude: list[tuple[int, int]] | None = None) -> tuple[str, list[tuple[int, int]]]:
    """The START time in a line as "HH:MM" and the character spans the time phrase occupies. A range such as
    "8:00 to 10:00", "9:00-9:50 am", "8-10am" or "2-4 pm" gives its start ("08:00", "09:00", "08:00", "14:00")."""
    exclude = exclude or []

    def ok(m: re.Match) -> bool:
        return not any(m.start() < b and m.end() > a for a, b in exclude)

    for m in _RANGE_AP.finditer(line):
        if not ok(m):
            continue
        h1, m1, ap1, h2, m2, ap2 = m.groups()
        h1, h2 = int(h1), int(h2)
        if not (1 <= h1 <= 12 and 1 <= h2 <= 12):
            continue
        ap = (ap1 or ap2).lower()
        start = h1 % 12 + (12 if ap == "p" else 0)
        if not ap1:  # "11-1 pm" starts at 11 am: the start cannot be after the end
            end = h2 % 12 + (12 if ap == "p" else 0)
            if start > end:
                start -= 12
        return _hhmm(start, int(m1 or 0)), [(m.start(), m.end())]
    for m in _RANGE_24.finditer(line):
        if not ok(m):
            continue
        h1, m1 = int(m.group(1)), int(m.group(2))
        if h1 > 23:
            continue
        if 1 <= h1 <= 6:  # "1:00 to 4:50" with no am/pm is an afternoon class
            h1 += 12
        return _hhmm(h1, m1), [(m.start(), m.end())]
    for m in _TIME12.finditer(line):
        if not ok(m):
            continue
        h, mi, ap = int(m.group(1)), int(m.group(2) or 0), m.group(3).lower()
        if not 1 <= h <= 12:
            continue
        return _hhmm(h % 12 + (12 if ap == "p" else 0), mi), [(m.start(), m.end())]
    for m in _TIME24.finditer(line):
        if ok(m):
            return _hhmm(int(m.group(1)), int(m.group(2))), [(m.start(), m.end())]
    return "", []


_LAB = re.compile(r"\blabs?\b|\blaboratory\b", re.I)
_LAB_DUE = re.compile(r"\b(report|due|write-?up|pre-?lab|worksheet)\b", re.I)
_LECTURE_NUM = re.compile(r"^\W*(?:lecture|lec\.?|class|session|week|wk\.?)\s*#?\s*\d{1,2}\b", re.I)
_NO_CLASS = re.compile(r"\bno\s+(class|lecture|lab|school|meeting|session)\b|\b(holiday|break|recess|thanksgiving|"
                       r"reading week)\b", re.I)
_REVIEW = re.compile(r"\breview\b", re.I)
_DATE_ANY = re.compile(rf"\b{_MON}\b\.?\s+\d|\b\d{{1,2}}/\d{{1,2}}|\b(?:19|20)\d{{2}}-\d{{2}}-\d{{2}}|"
                       rf"\b\d{{1,2}}{_ORD}\s+(?:of\s+)?{_MON}\b", re.I)


def classify(line: str, row_like: bool = False) -> str:
    """Event kind for a title. Lab sessions are "other" (the calendar has no lab kind)."""
    if _LECTURE_NUM.match(line) and not _NO_CLASS.search(line):
        return "lecture"
    has_test = bool(_EXAM.search(line) or _QUIZ.search(line))
    if _NO_CLASS.search(line) and not has_test:
        return "other"
    if _OTHER.search(line) and not has_test:
        return "other"
    if _REVIEW.search(line):
        return "other"
    if _LAB.search(line) and not _LAB_DUE.search(line) and not re.search(r"\b(midterms?|exams?|quiz(?:zes)?)\b", line, re.I):
        return "other"
    if _QUIZ.search(line):
        return "quiz"
    if _EXAM.search(line):
        return "exam"
    if _ASSIGN.search(line):
        return "assignment"
    if _LECTURE.search(line) or row_like:
        return "lecture"
    return "other"


def _tidy(c: str) -> str:
    c = re.sub(r"\s+([:;,])", r"\1", c).strip(_CLEAN_EDGES)
    if c.count("(") > c.count(")"):
        c += ")"
    return c


def _md(line: str) -> str:
    line = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", line)
    line = re.sub(r"(\*\*|__|`|~~)", "", line)
    line = re.sub(r"(?<![\w*])\*(?=\S)|(?<=\S)\*(?![\w*])", "", line)  # emphasis stars and footnote stars
    line = re.sub(r"<[^>]+>", " ", line)
    return re.sub(r"[ \t]+", " ", line).strip()


_CAPS_WORDS = re.compile(r"\b(EXAM|EXAMS|MIDTERM|FINAL|QUIZ|TEST|LECTURE|CLASS|REVIEW|HOMEWORK|ASSIGNMENT|PROJECT|PAPER|"
                         r"ESSAY|REPORT|NO|DUE)\b|[A-Z]{7,}")


def _sentence_case(t: str) -> str:
    """"FINAL EXAM" -> "Final exam" (but "HW8: PCA" stays)."""
    letters = [c for c in t if c.isalpha()]
    if len(letters) >= 4 and all(c.isupper() for c in letters) and _CAPS_WORDS.search(t):
        t = t.lower()
        t = t[:1].upper() + t[1:]
        t = re.sub(r"\b(hw|ps|pset|qa|ta)\b", lambda m: m.group(1).upper(), t)
    return t


def _prose_title(sentence: str) -> str:
    """Short title for an event described in a sentence: the words before the date when they name the event,
    otherwise the event word with the two words before it ("second midterm"). "" when nothing names an event."""
    sentence = sentence.strip()
    first = _DATE_ANY.search(sentence)
    prefix = sentence[:first.start()] if first else sentence
    prefix = _WEEKDAY.sub(" ", prefix)
    prefix = re.sub(r"\([^)]*\)?", " ", prefix)
    prefix = re.sub(r"\s+", " ", prefix).strip(_CLEAN_EDGES)
    for _ in range(5):
        prefix = re.sub(r"\b(?:is|are|will|be|was|due|held|scheduled|on|by|at|the|to be)\s*$", "", prefix,
                        flags=re.I).strip(_CLEAN_EDGES)
    if prefix and 1 <= len(prefix.split()) <= 8 and any(rx.search(prefix) for rx in (_QUIZ, _EXAM, _ASSIGN, _LECTURE, _LAB)):
        t = re.sub(r"^(?:the|a|an|your|our)\s+", "", prefix, flags=re.I)
        return (t[:1].upper() + t[1:]).strip(_CLEAN_EDGES)
    for rx in (_QUIZ, _EXAM, _ASSIGN, _LECTURE):
        m = rx.search(sentence)
        if m:
            before = [w for w in sentence[:m.start()].split() if not re.fullmatch(r"[\d,.;:]+", w)][-2:]  # not the tail of a date
            after = re.match(r"\s*(\d{1,2})\b", sentence[m.end():])
            t = " ".join([*before, m.group(0)] + ([after.group(1)] if after else []))
            t = re.sub(r"^(?:the|a|an|your|our)\s+", "", t, flags=re.I)
            return (t[:1].upper() + t[1:]).strip(_CLEAN_EDGES)
    return ""


def _title_and_notes(line: str, dates: list[dict[str, Any]], time_spans: list[tuple[int, int]]) -> tuple[str, str]:
    spans = sorted([(d["start"], d["end"]) for d in dates] + time_spans)
    out, pos = [], 0
    for a, b in spans:
        out.append(line[pos:a])
        out.append(" ")
        pos = max(pos, b)
    out.append(line[pos:])
    rest = "".join(out)
    rest = _WEEKDAY.sub(" ", rest)
    rest = re.sub(r"\(\s*[,;:\-–—]*\s*\)|\[\s*\]", " ", rest)
    rest = re.sub(r"\b(?:on|by|at|due|until|through|thru)\s*(?=[|:;,\-–—)]|$)", " ", rest, flags=re.I)
    rest = re.sub(r"\(\s*[,;]\s*", "(", rest)
    rest = re.sub(r"[\s,;]+(?=\))", "", rest)
    cells = [c.strip(_CLEAN_EDGES) for c in re.split(r"\s*\|\s*|\t+|\s{3,}", rest)]
    cells = [re.sub(r"\s+", " ", c) for c in cells if c.strip(_CLEAN_EDGES)]
    cells = [_tidy(c) for c in cells]
    cells = [c for c in cells if c]
    if not cells:
        return "", ""
    title = cells[0]
    rest_cells = cells[1:]
    if rest_cells and re.fullmatch(r"(?:week|wk|lec(?:ture)?|class|day)?\.?\s*#?\d{1,2}", title, re.I):
        title = f"{title}: {rest_cells[0]}"
        rest_cells = rest_cells[1:]
    return title, " | ".join(rest_cells)


def _sentence_with(line: str, pos: int) -> str:
    """For prose lines: the sentence that holds position pos."""
    start = max(line.rfind(". ", 0, pos), line.rfind("? ", 0, pos), line.rfind("! ", 0, pos))
    start = start + 2 if start != -1 else 0
    ends = [i for i in (line.find(". ", pos), line.find("? ", pos), line.find("! ", pos)) if i != -1]
    end = min(ends) + 1 if ends else len(line)
    return line[start:end]


_WEAK = re.compile(r"^\W*first\s+(class|lecture|meeting|lab|session|day|week)", re.I)
_JUNK_TITLE = re.compile(r"^(to|and|or|the|a|an|of|in|on|at|by|for|from|with|room|hall|note|notes?)\b[\s.:,]*$", re.I)


def _paren_to_notes(title: str, notes: str) -> tuple[str, str]:
    """Exam, quiz and assignment titles keep only their name: "Midterm 1 (in class)" -> "Midterm 1", note "in class"."""
    m = re.match(r"^(.*?)\s*\(([^()]*)\)\s*$", title)
    if m and m.group(1).strip():
        title = m.group(1).strip()
        inner = m.group(2).strip(" ,;")
        for _ in range(3):
            inner = re.sub(r"(?:^|[\s,;]+)(?:is\s+)?(?:due|at|on|by)\s*$", "", inner, flags=re.I).strip(" ,;")
        if inner:
            notes = f"{inner}; {notes}" if notes else inner
    return title, notes


def _finish_title(title: str, notes: str, kind: str) -> tuple[str, str]:
    title = _md(title).strip(" \t-\u2013\u2014:;,|\u00b7\u2022*>.")
    title = re.sub(r"^\d{1,3}\s*[|:.)-]\s+(?=\D)", "", title)  # a leading row number: "17 | Lecture 20"
    title = re.sub(r"^[|:;,\-–— ]+|[|:;,\-–— ]+$", "", title)
    title = _sentence_case(title)
    if kind in ("exam", "quiz", "assignment"):
        title, notes = _paren_to_notes(title, notes)
        if kind == "assignment":
            title = re.sub(r"[\s,;:]+(?:is\s+)?due$", "", title, flags=re.I)
    elif len(title) > 60:
        title, notes = _paren_to_notes(title, notes)
    title = re.sub(r"\(\s*[,;]?\s*\)", "", title).strip(_CLEAN_EDGES)
    if title.count("(") != title.count(")"):
        title = _tidy(title)
    if len(title) > 90:
        title = title[:87].rstrip() + "..."
    return title, notes


def _good_title(title: str) -> bool:
    letters = sum(c.isalpha() for c in title)
    if re.match(r"^\d{1,2}[ ,]+(?:19|20)\d{2}\b", title):
        return False  # "3 2026, Midterm 1": the tail of a date
    if (letters < 3 and not re.match(r"(?i)^(hw|ps|q|lec|wk)\s*\d", title)) or _JUNK_TITLE.match(title):
        return False
    return not re.fullmatch(r"(?:(?:mon|tue|wed|thu|fri|sat|sun)\w*[ ,.]*|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*"
                            r"[ ,.]*|\d+[ ,.]*|and[ ,.]*|to[ ,.]*)+", title, re.I)


# ------------------------------------------------------------------ tables
def _header_role(h: str) -> str:
    h = h.lower()
    if re.search(r"\bdue\b|deadline", h):
        return "due"
    if re.search(r"releas|assigned|posted|\bout\b|\bgiven\b", h):
        return "release"
    if re.search(r"weight|points|\bpts\b|%|worth|score", h):
        return "weight"
    if re.search(r"reading|notebook|\bfile|chapter|source|materials?|resources?|video|slides?", h):
        return "reading"
    if re.search(r"\bdates?\b|week of|\bwhen\b|\bday\b|^week\b|^wk\b", h):
        return "date"
    if re.search(r"\btime\b|room|location|where|instructor", h):
        return "skip"
    return "title"


def _is_filename(c: str) -> bool:
    return bool(re.search(r"\.(md|ipynb|jl|pdf|pptx?|docx?|txt|html?)\b", c, re.I)) or bool(re.match(r"^\(.*\)$", c.strip()))


def _singular(h: str) -> str:
    h = re.sub(r"\(.*?\)", "", h).strip()
    if re.search(r"(sets|labs|exams|quizzes|assignments|homeworks)$", h, re.I):
        h = h[:-1]
    return h.strip()


class _Ctx:
    """Year bookkeeping: a term that runs over New Year moves Dec -> Jan into the next year."""

    def __init__(self, year: int):
        self.year = year
        self.prev_month = 0
        self.wrap = 0

    def resolve(self, d: dict[str, Any]) -> str | None:
        y = d["year"]
        if y is None:
            if self.prev_month and d["month"] < self.prev_month - 6:
                self.wrap = 1
            elif self.prev_month and d["month"] > self.prev_month + 6 and self.wrap:
                self.wrap = 0
            y = self.year + self.wrap
        self.prev_month = d["month"]
        try:
            return date(y, d["month"], d["day"]).isoformat()
        except ValueError:
            return None


def _split_dates(cell: str) -> list[dict[str, Any]]:
    """Dates in a cell: one entry per date, and "Mon Oct 5 to Mon Oct 12" is one entry with a 'through' date."""
    ds = _find_dates(cell)
    out: list[dict[str, Any]] = []
    i = 0
    while i < len(ds):
        d = dict(ds[i], through=None)
        if i + 1 < len(ds):
            between = cell[ds[i]["end"]:ds[i + 1]["start"]]
            if re.fullmatch(r"\s*(?:to|through|thru|until|-|–|—)\s*(?:(?:mon|tue|wed|thu|fri|sat|sun)\w*\.?,?\s*)?",
                            between, re.I):
                d["through"] = ds[i + 1]
                i += 1
        out.append(d)
        i += 1
    return out


def _fmt_md(d: dict[str, Any]) -> str:
    return f"{[k for k, v in MONTHS.items() if v == d['month']][0].capitalize()} {d['day']}"


def _row_events(cells: list[str], header: list[str] | None, ctx: _Ctx, week_n: list[int]) -> list[dict[str, Any]]:
    aligned = bool(header) and len(header) == len(cells)
    n = len(cells)
    if aligned:
        roles = [_header_role(h) for h in header]  # type: ignore[union-attr]
    else:
        # no usable header: the first cell with a date is the date, the first other non-empty cell the title
        date_idx = next((i for i, c in enumerate(cells) if _find_dates(c)), None)
        roles = ["title"] * n
        if date_idx is not None and n > 1:
            roles[date_idx] = "date"
        seen_title = False
        for i in range(n):
            if roles[i] == "date" or not cells[i].strip():
                continue
            roles[i] = "note" if seen_title else "title"
            seen_title = True
        titles = [i for i, r in enumerate(roles) if r == "title"]
        if titles and re.fullmatch(r"(?:week|wk|lec(?:ture)?|class|session|day)\.?\s*#?\d{1,2}", _md(cells[titles[0]]), re.I):
            nxt_note = next((i for i, r in enumerate(roles) if r == "note"), None)
            if nxt_note is not None:
                roles[nxt_note] = "title"
    has_due = "due" in roles and any(_find_dates(_md(cells[i])) for i, r in enumerate(roles) if r == "due")
    date_cells = [i for i, r in enumerate(roles) if r == ("due" if has_due else "date")]
    title_cells = [i for i, r in enumerate(roles) if r == "title" and cells[i].strip()]
    note_cells = [i for i, r in enumerate(roles) if r in ("reading", "note") and cells[i].strip()]
    release_cells = [i for i, r in enumerate(roles) if r == "release" and cells[i].strip()]
    dates: list[dict[str, Any]] = []
    for i in date_cells:
        dates.extend(_split_dates(_md(cells[i])))
    from_title = False
    if not dates:  # the date sits inside the title cell: "Midterm 1 (Thu Oct 1, in class)"
        for i in title_cells:
            ds = _split_dates(_md(cells[i]))
            if ds:
                dates, from_title = ds[:1], True
                break
    if not dates or not title_cells:
        return []
    time = ""
    if aligned:
        for i, r in enumerate(roles):
            if r == ("due" if has_due else "date"):
                time = parse_time(header[i])[0] or time  # type: ignore[index]
    first = _md(cells[title_cells[0]])
    fdates = _find_dates(first) if from_title else []
    ttime, _ = parse_time(first, [(d["start"], d["end"]) for d in fdates])
    time = ttime or time
    pieces: list[str] = []
    for i in title_cells:
        c = _md(cells[i])
        if i == title_cells[0]:
            ds = _find_dates(c) if from_title else []
            tsp = parse_time(c, [(d["start"], d["end"]) for d in ds])[1]
            if ds or tsp:
                c, _ = _title_and_notes(c, ds, tsp)
        pieces.append(c)
    title = pieces[0]
    if re.fullmatch(r"\d{1,3}", title.strip()) and aligned:  # a bare row number: label it with its column name
        label = _singular(header[title_cells[0]])  # type: ignore[index]
        title = f"{label} {title.strip()}".strip()
    if len(pieces) > 1 and pieces[1].strip(_CLEAN_EDGES):
        title = f"{title}: {pieces[1]}"
    notes_parts = [_md(cells[i]) for i in note_cells if not _is_filename(_md(cells[i]))]
    notes_parts += [f"Released {_md(cells[i])}" for i in release_cells]
    notes = "; ".join(x for x in notes_parts if x)
    week_like = aligned and any(re.search(r"week of|^week\b", h.lower()) for h in header)  # type: ignore[union-attr]
    # "Document analysis paper due; Federalists and Democratic-Republicans": a deadline plus the day's lecture topic
    parts = [p.strip() for p in re.split(r";\s+", title)] if ";" in title else [title]
    kinds = [classify(p, True) for p in parts]
    if len(parts) == 2 and kinds[0] in ("assignment", "exam", "quiz") and kinds[1] == "lecture":
        groups = [(parts[0], kinds[0]), (parts[1], "lecture")]
    else:
        groups = [(title, classify(title, True))]
    split = len(groups) == 2
    results: list[dict[str, Any]] = []
    for d in dates:
        iso = ctx.resolve(d)
        if not iso:
            continue
        for gt, kind in groups:
            t, nt = gt, (notes if not split or kind == "lecture" else "")
            if d.get("through"):
                nt = f"{nt}; through {_fmt_md(d['through'])}" if nt else f"through {_fmt_md(d['through'])}"
            if kind == "lecture" and week_like and not _LECTURE_NUM.match(t):
                t = f"Week {week_n[0]}: {t}"
            t, nt = _finish_title(t, nt, kind)
            if not _good_title(t):
                continue
            ev: dict[str, Any] = {"title": t, "kind": kind, "date": iso}
            if time:
                ev["time"] = time
            if nt:
                ev["notes"] = nt[:200]
            results.append(ev)
    if week_like:
        week_n[0] += 1
    return results


def _split_cells(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [c.strip() for c in line.split("|")]


def _is_sep(line: str) -> bool:
    return bool(re.fullmatch(r"\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?", line.strip()))


# ------------------------------------------------------------------ dedupe
def _similar(a: str, b: str) -> bool:
    def toks(t: str) -> set[str]:
        t = re.sub(r"\b(due|assignment|the|a|of|and|for|in|on|week \d+|lecture \d+)\b", " ", t.lower())
        return {w.rstrip("s") for w in re.findall(r"[a-z0-9]+", t)}

    ta, tb = toks(a), toks(b)
    if not ta or not tb:
        return a.lower().strip() == b.lower().strip()
    if ta <= tb or tb <= ta:
        return True
    return len(ta & tb) / len(ta | tb) >= 0.6


def _same_kind(a: str, b: str) -> bool:
    return a == b or (a in ("lecture", "other") and b in ("lecture", "other"))


def similar_event(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """The same event written twice: same date, a compatible kind and a similar title."""
    return a["date"] == b["date"] and _same_kind(a["kind"], b["kind"]) and _similar(a["title"], b["title"])


def merge_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop repeats. The first stays and takes a time or note that its repeat knew. "First class" remarks are
    dropped when the day already has a real event."""
    out: list[dict[str, Any]] = []
    for ev in events:
        dup = next((o for o in out if similar_event(o, ev)), None)
        if dup is None:
            out.append(dict(ev))
            continue
        if ev.get("time") and not dup.get("time"):
            dup["time"] = ev["time"]
        if ev.get("notes") and not dup.get("notes"):
            dup["notes"] = ev["notes"]
    return [ev for ev in out if not (_WEAK.match(ev["title"]) and any(
        o is not ev and o["date"] == ev["date"] and not _WEAK.match(o["title"]) for o in out))]


# ------------------------------------------------------------------ extraction
def _line_events(line: str, dates: list[dict[str, Any]], ctx: _Ctx, prev_text: str) -> list[dict[str, Any]]:
    chosen = dates[0]
    due = re.search(r"\bdue\b", line, re.I)
    if due:
        after = [d for d in dates if d["start"] >= due.end()]
        if after:
            chosen = after[0]
    work = line
    offset_dates = dates
    prose = len(line.split()) > 14 or bool(re.search(r"[a-z)][.!?]\s+[A-Z]", line))
    if len(line) > 140 or prose:  # prose: look at the sentence around the date
        sent = _sentence_with(line, chosen["start"])
        shift = line.find(sent)
        work = sent
        offset_dates = [dict(d, start=d["start"] - shift, end=d["end"] - shift) for d in dates
                        if d["start"] >= shift and d["end"] <= shift + len(sent)]
        chosen = dict(chosen, start=chosen["start"] - shift, end=chosen["end"] - shift)
        if not offset_dates:
            return []
    iso = ctx.resolve(chosen)
    if not iso:
        return []
    date_spans = [(d["start"], d["end"]) for d in offset_dates]
    time, tspans = parse_time(work, date_spans)
    title, notes = _title_and_notes(work, offset_dates, tspans)
    ctx_text = work
    if not title and prev_text:
        title = prev_text.strip(_CLEAN_EDGES)
        ctx_text = f"{title} {work}"
    sentence_prose = len(work.split()) > 14 or prose
    nc = re.match(r"\W*(?:there\s+(?:is|will\s+be)\s+|we\s+(?:have|will\s+have)\s+)no\s+(class|lecture|lab|session|meeting)\b",
                  work, re.I)
    if nc:  # "There is no class on Thursday Nov 26 (Thanksgiving)."
        paren = re.search(r"\(([^)]*)\)", work)
        t2 = f"No {nc.group(1).lower()}" + (f" ({paren.group(1).strip()})" if paren and paren.group(1).strip() else "")
        return [{"title": t2, "kind": "other", "date": iso}]
    if sentence_prose and len(work.split()) > 28:
        return []  # a long paragraph that mentions dates is not a schedule line
    if sentence_prose:
        t2 = _prose_title(work)
        if not t2:
            return []  # a policy sentence that happens to carry a date
        notes = re.sub(r"\s+", " ", line if len(line) <= 200 else work).strip()
        title = t2
    title = re.sub(r"\s+assigned$", "", title, flags=re.I) if due else title
    if not title:
        wk = _WEEK.match(line)
        title = f"Week {wk.group(1)}" if wk else ""
    if not title:
        return []
    row_like = bool(_WEEK.match(line)) or dates[0]["start"] <= 2
    kind = classify(title if sentence_prose else ctx_text, row_like and not sentence_prose)
    title, notes = _finish_title(title, notes, kind)
    if not _good_title(title):
        return []
    ev: dict[str, Any] = {"title": title, "kind": kind, "date": iso}
    if time:
        ev["time"] = time
    if notes:
        ev["notes"] = notes[:200]
    return [ev]


def extract_events(text: str, year_hint: int | str | None = None) -> list[dict[str, Any]]:
    """Return [{title, kind, date 'YYYY-MM-DD', time?, notes?}] for lines with a clear date.

    Tables are read by their header: a Due column wins over Released, "Week of" rows become "Week N: topic", a cell
    with several dates ("Wed Sep 9, Fri Sep 11") gives one event per date, and a time range gives its start.
    Repeats (the same date, kind and similar title) are dropped, and junk rows are skipped."""
    ctx = _Ctx(_year_from(year_hint))
    events: list[dict[str, Any]] = []
    lines = [(ln.strip() if ln.lstrip().startswith("|") else _md(ln)) for ln in text.replace("\r", "\n").split("\n")]
    nonblank = [i for i, ln in enumerate(lines) if ln.strip()]
    header: list[str] | None = None
    in_table = False
    week_n = [1]
    prev_text = ""
    for pos, i in enumerate(nonblank):
        line = lines[i]
        nxt = lines[nonblank[pos + 1]] if pos + 1 < len(nonblank) else ""
        if _is_sep(line):
            continue
        cells = _split_cells(line) if "|" in line else []
        if len(cells) >= 2:
            first_row = not in_table
            if first_row:
                header, week_n = None, [1]
            in_table = True
            has_date = any(_find_dates(_md(c)) for c in cells)
            if not has_date and (first_row or _is_sep(nxt)):
                header = [_md(c) for c in cells]
                continue
            events.extend(_row_events([_md(c) for c in cells], header, ctx, week_n))
            continue
        in_table = False
        header = None
        line = re.sub(r"^[\s>*•·\-–—]+(?=\S)", "", line)
        if line.startswith("#"):
            continue
        dates = _find_dates(line)
        if not dates:
            if len(line) < 80:
                prev_text = line
            continue
        events.extend(_line_events(line, dates, ctx, prev_text))
        prev_text = ""
    return merge_events(events)
