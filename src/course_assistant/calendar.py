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
_ASSIGN = re.compile(r"\b(hw\s*\d*|homeworks?|problem sets?|psets?|ps\s*\d+|assignments?|projects?|labs?|papers?|"
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


def _times(line: str, exclude: list[tuple[int, int]]) -> str:
    def ok(m: re.Match) -> bool:
        return not any(m.start() < b and m.end() > a for a, b in exclude)

    for m in _TIME12.finditer(line):
        if not ok(m):
            continue
        h, mi, ap = int(m.group(1)), int(m.group(2) or 0), m.group(3).lower()
        if not 1 <= h <= 12:
            continue
        h = h % 12 + (12 if ap == "p" else 0)
        return f"{h:02d}:{mi:02d}"
    for m in _TIME24.finditer(line):
        if ok(m):
            return f"{int(m.group(1)):02d}:{m.group(2)}"
    return ""


def classify(line: str, row_like: bool = False) -> str:
    if _OTHER.search(line) and not (_EXAM.search(line) or _QUIZ.search(line)) or re.search(r"\breview\b", line, re.I):
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


def _prose_title(sentence: str) -> str:
    words = sentence.split()
    for rx in (_QUIZ, _EXAM, _ASSIGN, _LECTURE):
        m = rx.search(sentence)
        if m:
            before = sentence[:m.start()].split()[-2:]
            after = re.match(r"\s*(\d{1,2})\b", sentence[m.end():])
            t = " ".join([*before, m.group(0)] + ([after.group(1)] if after else []))
            t = re.sub(r"^(?:the|a|an|your|our)\s+", "", t, flags=re.I)
            return (t[:1].upper() + t[1:]).strip(_CLEAN_EDGES)
    return " ".join(words[:8]).strip(_CLEAN_EDGES)


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


def extract_events(text: str, year_hint: int | str | None = None) -> list[dict[str, Any]]:
    """Return [{title, kind, date 'YYYY-MM-DD', time?, notes?}] for lines with a clear date."""
    year = _year_from(year_hint)
    events: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    prev_month = 0
    prev_text = ""
    wrap = 0
    for raw in text.replace("\r", "\n").split("\n"):
        line = raw.strip()
        if not line:
            continue
        line = re.sub(r"^[\s>*•·\-–—]+(?=\S)", "", line)
        dates = _find_dates(line)
        if not dates:
            if len(line) < 80:
                prev_text = line
            continue
        chosen = dates[0]
        due = re.search(r"\bdue\b", line, re.I)
        if due:
            after = [d for d in dates if d["start"] >= due.end()]
            if after:
                chosen = after[0]
        work = line
        offset_dates = dates
        if len(line) > 140:  # prose: look at the sentence around the date
            sent = _sentence_with(line, chosen["start"])
            shift = line.find(sent)
            work = sent
            offset_dates = [dict(d, start=d["start"] - shift, end=d["end"] - shift) for d in dates
                            if d["start"] >= shift and d["end"] <= shift + len(sent)]
            chosen = dict(chosen, start=chosen["start"] - shift, end=chosen["end"] - shift)
        y = chosen["year"]
        if y is None:
            # a term that runs over New Year: Dec then Jan moves to the next year, and back if the dates jump back
            if prev_month and chosen["month"] < prev_month - 6:
                wrap = 1
            elif prev_month and chosen["month"] > prev_month + 6 and wrap:
                wrap = 0
            y = year + wrap
        prev_month = chosen["month"]
        try:
            iso = date(y, chosen["month"], chosen["day"]).isoformat()
        except ValueError:
            continue
        date_spans = [(d["start"], d["end"]) for d in offset_dates]
        time = _times(work, date_spans)
        tspans = [(m.start(), m.end()) for rx in (_TIME12, _TIME24) for m in rx.finditer(work)
                  if time and not any(m.start() < b and m.end() > a for a, b in date_spans)] if time else []
        title, notes = _title_and_notes(work, offset_dates, tspans)
        ctx = work
        if not title and prev_text:
            title = prev_text.strip(_CLEAN_EDGES)
            ctx = f"{title} {work}"
        prose = len(work.split()) > 14
        if prose and title:
            notes = re.sub(r"\s+", " ", work).strip()
            title = _prose_title(work)
        title = re.sub(r"\s+assigned$", "", title, flags=re.I) if due else title
        if not title:
            wk = _WEEK.match(line)
            title = f"Week {wk.group(1)}" if wk else ""
        if not title:
            continue
        if len(title) > 90:
            title = title[:87].rstrip() + "..."
        row_like = bool(_WEEK.match(line)) or dates[0]["start"] <= 2 or bool(re.search(r"\|\t", line)) or "|" in line
        kind = classify(ctx, row_like and not prose)
        key = (title.lower(), iso)
        if key in seen:
            continue
        seen.add(key)
        ev: dict[str, Any] = {"title": title, "kind": kind, "date": iso}
        if time:
            ev["time"] = time
        if notes:
            ev["notes"] = notes[:200]
        events.append(ev)
        prev_text = ""
    return events
