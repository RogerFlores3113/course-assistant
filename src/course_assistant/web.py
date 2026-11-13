"""The app window: a local web UI on 127.0.0.1 opened in the student's browser.

Other modules (importers, grounding, study, integrity, offerings, connect, providers) are imported lazily inside
the routes, so this file loads even while they are being written and tests can swap them out.
"""
from __future__ import annotations

import functools
import hmac
import importlib
import json
import logging
import mimetypes
import re
import shutil
import tempfile
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlsplit

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import __version__, disclaimer, locations, paths
from .store import EVENT_KINDS, GUIDE_KINDS, MATERIAL_KINDS, Store, now_iso
from .web_md import render as render_md
from .web_md import render_reader, source_map as _source_map
from .web_helpers import (plain_math, DEPTHS, MODELS, PALETTE, TABS, THEMES, accent_style, answer_view, chunk_range_label, clean_color,
                          clean_event_title, concept_graph, course_color, day_label, enrich_sources, group_guides, guide_view,
                          merge_events, parse_day, snippet_text, time_label, week_label, week_start)

LOCAL_HOSTS = {"127.0.0.1", "localhost", "[::1]", "::1"}
UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
OPEN_PATHS = {"/launch", "/api/health"}
UNGATED_PATHS = {"/welcome", "/privacy", "/launch", "/api/health"}
COOKIE = "ca_key"
KEY_HEADER = "x-course-assistant-key"
KEY_MAX_AGE = 400 * 24 * 3600
FLASH = "ca_flash"
MAX_UPLOAD = 2 * 1024 * 1024 * 1024  # a long lecture recording
CSP = ("default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
       "object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'")
CODE_EXT = {".py", ".java", ".c", ".h", ".cpp", ".js", ".ts", ".ipynb", ".r", ".m", ".go", ".rs", ".rb", ".sh", ".sql", ".cs"}
MEDIA_EXT = {".mp3": "audio", ".m4a": "audio", ".wav": "audio", ".mp4": "video", ".mov": "video", ".webm": "video"}
INLINE_TYPES = ("application/pdf", "audio/", "video/", "image/png", "image/jpeg", "image/gif", "image/webp", "text/plain")
GRADED_KINDS = {"problem": "Problem", "proof": "Proof", "essay": "Essay or paper"}
FLASH_DRAFT_MAX = 800
log = logging.getLogger(__name__)


class _Missing(Exception):
    """A module another builder is writing is not there yet."""


def _mod(name: str):
    try:
        return importlib.import_module(f"course_assistant.{name}")
    except ImportError as e:
        raise _Missing(name) from e


def health_proof(key: str, nonce: str, port: int) -> str:
    """HMAC-SHA256 proof for /api/health, same scheme as Talent Search with this app's labels:
    subkey = HMAC(key, "course-assistant:health"); proof = hex HMAC(subkey, "course-assistant-health:<port>:" + nonce)."""
    sub = hmac.new(key.encode(), b"course-assistant:health", "sha256").digest()
    return hmac.new(sub, f"course-assistant-health:{int(port)}:".encode() + nonce.encode(), "sha256").hexdigest()


def _hostname(host: str) -> str:
    host = host.strip().lower()
    if host.startswith("["):
        return host.split("]")[0] + "]"
    return host.split(":")[0]


def _is_no_provider(e: Exception) -> bool:
    return type(e).__name__ == "NoProvider"


def _safe_mod(name: str):
    try:
        return _mod(name)
    except _Missing:
        return None


def _friendly(e: Exception) -> str:
    if isinstance(e, _Missing):
        return "That part of the app is not installed yet."
    text = str(e).strip() or type(e).__name__
    return text if len(text) <= 400 else text[:397] + "..."


def create_app(store: Store, *, key: str | None = None, provider_factory=None, start_jobs: bool = False) -> FastAPI:
    """start_jobs=False (the default, so tests never spawn threads); `course-assistant serve` passes True and the app
    then runs the background worker that reads photos of notes and transcribes recordings."""
    try:  # remember which materials were deleted, so the reader can tell "deleted" (410) from "never existed" (404)
        store._exec("CREATE TABLE IF NOT EXISTS deleted_materials (id INTEGER PRIMARY KEY)")
        store._exec("CREATE TRIGGER IF NOT EXISTS materials_deleted AFTER DELETE ON materials "
                    "BEGIN INSERT OR IGNORE INTO deleted_materials(id) VALUES (old.id); END")
    except Exception:  # noqa: BLE001 - the reader falls back to a guess from the ids in use
        pass
    key = key or paths.app_key()
    res = paths.resource_dir()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if start_jobs:
            try:
                starter = getattr(_mod("transcribe"), "start_worker", None)
                _app.state.worker = starter(store) if callable(starter) else None
            except Exception:  # a broken worker must never stop the app from opening
                log.exception("Could not start the background worker")
        try:
            yield
        finally:
            w = getattr(_app.state, "worker", None)
            if w is not None and callable(getattr(w, "stop", None)):
                try:
                    w.stop()
                except Exception:
                    log.exception("Could not stop the background worker")
            _app.state.worker = None

    app = FastAPI(title="Course Assistant", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.worker = None
    app.state.start_jobs = start_jobs
    templates = Jinja2Templates(directory=str(res / "templates"))
    env = templates.env
    env.globals.update(version=__version__, disclaimer_full=disclaimer.FULL, tabs=TABS, depths=DEPTHS, models=MODELS,
                       palette=PALETTE, material_kinds=MATERIAL_KINDS, event_kinds=EVENT_KINDS,
                       guide_kinds=GUIDE_KINDS, course_color=course_color, accent_style=accent_style,
                       graded_kinds=GRADED_KINDS, page_title="")
    env.globals["shortcuts_enabled"] = lambda: store.get_setting("card_shortcuts", "1") != "0"
    env.globals["theme"] = lambda: (store.get_setting("theme", "system") if store.get_setting("theme", "system") in THEMES else "system")
    env.filters["day"] = day_label
    env.filters["clean"] = clean_event_title
    env.filters["clock"] = time_label
    env.filters["snippet"] = snippet_text
    env.filters["quote"] = lambda s: quote(str(s), safe="")
    app.mount("/static", StaticFiles(directory=str(res / "static")), name="static")

    # ------------------------------------------------------------------ helpers
    def acknowledged() -> bool:
        return bool(store.get_setting(disclaimer.ACK_SETTING, ""))

    def is_htmx(request: Request) -> bool:
        return bool(request.headers.get("HX-Request")) and not request.headers.get("HX-History-Restore-Request")

    def provider():
        if provider_factory is not None:
            try:
                return provider_factory()
            except TypeError:
                return provider_factory(store)
        try:
            return _mod("providers").get_provider(store)
        except _Missing:
            return None

    def provider_state() -> dict[str, Any]:
        kind = (store.get_setting("ai_provider", "") or "none").lower()
        try:
            p = provider()
        except Exception:  # a broken setting must not break the page
            p = None
        return {"kind": kind, "connected": p is not None, "name": getattr(p, "name", ""), "model": getattr(p, "model", "")}

    async def work(fn, *args, **kw):
        return await run_in_threadpool(functools.partial(fn, *args, **kw))

    def page(request: Request, name: str, status_code: int = 200, **ctx: Any) -> HTMLResponse:
        flash = {}
        raw = request.cookies.get(FLASH)
        if raw:
            try:
                flash = json.loads(unquote(raw))
            except ValueError:
                flash = {}
        ctx.setdefault("notice", flash.get("notice", ""))
        ctx.setdefault("error", flash.get("error", ""))
        ctx.setdefault("draft", flash.get("draft", ""))
        resp = templates.TemplateResponse(request, name, ctx, status_code=status_code)
        if raw:
            resp.delete_cookie(FLASH, path="/")
        return resp

    def redirect(url: str, notice: str = "", error: str = "", draft: str = "") -> RedirectResponse:
        resp = RedirectResponse(url, status_code=303)
        if notice or error:
            data = {"notice": notice[:500], "error": error[:500]}
            if draft:  # so a plain (no JavaScript) form post does not lose what the student typed
                data["draft"] = draft[:FLASH_DRAFT_MAX]
            resp.set_cookie(FLASH, quote(json.dumps(data)), max_age=120, httponly=True, samesite="strict", path="/")
        return resp

    def require_course(course_id: int) -> dict[str, Any]:
        c = store.course(course_id)
        if not c:
            raise _NotFound("That course was not found.")
        return c

    class _NotFound(Exception):
        pass

    def friendly_page(request: Request, status: int, title: str, message: str, headers: dict | None = None):
        resp = templates.TemplateResponse(request, "message.html", {"title": title, "message": message}, status_code=status,
                                          headers=headers)
        return secure(resp)

    @app.exception_handler(_NotFound)
    async def not_found(request: Request, exc: _NotFound):
        return friendly_page(request, 404, "Not found", str(exc))

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        text = {404: ("Page not found", "That page does not exist. It may have been deleted, or the address may be mistyped."),
                405: ("Not allowed", "That address does not accept this kind of request."),
                403: ("Not allowed", "That request was not allowed.")}.get(
            exc.status_code, ("Something went wrong", "The app could not show that page."))
        return friendly_page(request, exc.status_code, *text, headers=getattr(exc, "headers", None))

    @app.exception_handler(RequestValidationError)
    async def bad_request(request: Request, exc: RequestValidationError):
        if any(e.get("loc") and e["loc"][0] == "path" for e in exc.errors()):
            return friendly_page(request, 404, "Page not found", "That page does not exist. Check the address.")
        return friendly_page(request, 422, "That could not be read",
                             "Something in that request was missing or not in the right form. Go back, check the "
                             "form and try again.")

    @app.exception_handler(Exception)
    async def crashed(request: Request, exc: Exception):
        log.exception("Unhandled error on %s", request.url.path)
        return friendly_page(request, 500, "Something went wrong",
                             "The app hit an unexpected problem. Your saved materials are safe. Go back and try again; "
                             "if it keeps happening, restart Course Assistant.")

    def course_url(cid: int, tab: str = "materials") -> str:
        return f"/courses/{cid}" if tab == "materials" else f"/courses/{cid}/{tab}"

    # ------------------------------------------------------------------ security
    @app.middleware("http")
    async def local_only(request: Request, call_next):
        host_header = (request.headers.get("host") or "").strip().lower()
        path = request.url.path
        if _hostname(host_header) not in LOCAL_HOSTS:
            return PlainTextResponse("Forbidden", status_code=403)
        given = request.cookies.get(COOKIE) or request.headers.get(KEY_HEADER) or ""
        request.state.unlocked = hmac.compare_digest(given.encode(), key.encode())
        if not request.state.unlocked and path not in OPEN_PATHS and not path.startswith("/static/"):
            resp: Response = templates.TemplateResponse(request, "locked.html", {}, status_code=401)
            return secure(resp)
        fetch_site = (request.headers.get("sec-fetch-site") or "").lower()
        if request.method in UNSAFE_METHODS:
            origin = request.headers.get("origin")
            if origin is not None:
                if origin.strip().lower() == "null" or urlsplit(origin.strip().lower()).netloc != host_header:
                    return PlainTextResponse("Forbidden", status_code=403)
            elif fetch_site and fetch_site not in ("same-origin", "none"):
                return PlainTextResponse("Forbidden", status_code=403)
        if (path not in UNGATED_PATHS and not path.startswith("/static/") and not acknowledged()):
            return secure(RedirectResponse("/welcome", status_code=303))
        return secure(await call_next(request), static=path.startswith("/static/"))

    def secure(resp: Response, static: bool = False) -> Response:
        resp.headers.setdefault("X-Frame-Options", "DENY")
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("Referrer-Policy", "same-origin")
        resp.headers.setdefault("Content-Security-Policy", CSP)
        if not static:
            resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.get("/launch")
    def launch(request: Request, code: str = ""):
        if not (request.state.unlocked or paths.use_launch_code(code)):
            return secure(templates.TemplateResponse(request, "locked.html", {}, status_code=401))
        resp = RedirectResponse("/", status_code=303)
        resp.set_cookie(COOKIE, key, max_age=KEY_MAX_AGE, httponly=True, samesite="strict", path="/")
        return resp

    @app.get("/api/health")
    def health(request: Request, nonce: str = ""):
        """Open without the key, so a second launch can find the running app. With a nonce it answers `proof`
        (health_proof), bound to the port the request really arrived on, so the CLI can tell this app from
        another program on the port before it sends a launch code."""
        out: dict[str, Any] = {"ok": True, "version": __version__}
        server = request.scope.get("server") or ()
        if nonce and len(nonce) <= 128 and len(server) == 2 and isinstance(server[1], int):
            out["proof"] = health_proof(key, nonce, server[1])
        return out

    # ------------------------------------------------------------------ welcome, privacy
    @app.get("/welcome", response_class=HTMLResponse)
    def welcome(request: Request):
        return page(request, "welcome.html", acknowledged=acknowledged())

    @app.post("/welcome")
    def welcome_ack():
        store.set_setting(disclaimer.ACK_SETTING, now_iso())
        # first run: offer the "Connect an AI" step once, unless an AI is already set up
        if not store.get_setting("setup_done", "") and not provider_state()["connected"]:
            return redirect("/setup?first=1")
        return redirect("/")

    @app.get("/privacy", response_class=HTMLResponse)
    def privacy(request: Request):
        return page(request, "privacy.html")

    # ------------------------------------------------------------------ dashboard
    def upcoming_rows() -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        try:
            raw = _mod("study").upcoming(store, days=14)
            for r in raw or []:
                if isinstance(r, dict) and r.get("date") and r.get("title"):
                    rows.append(dict(r))
        except Exception:
            rows = []
        if not rows:
            today = date.today()
            rows = store.events(None, today.isoformat(), (today + timedelta(days=14)).isoformat())
            for r in rows:
                r["course_id"] = r.get("course_id")
        for r in rows:
            if not r.get("course_id") and r.get("id"):
                ev = store.event(int(r["id"]))
                r["course_id"] = ev["course_id"] if ev else None
            r.setdefault("id", r.get("event_id"))
            c = store.course(int(r["course_id"])) if r.get("course_id") else None
            r["course_name"] = r.get("course_name") or (c or {}).get("name", "")
            r["accent"] = course_color(c) if c else course_color(None)
        return [r for r in rows if r.get("course_id")]

    def next_exam(cid: int) -> dict[str, Any] | None:
        """The next calendar event of kind exam, with the days left. None when no exam is on the calendar."""
        today = date.today()
        try:
            junk = _mod("study").is_junk_event
        except (_Missing, AttributeError):
            junk = lambda e: False  # noqa: E731
        for e in store.events(cid, today.isoformat(), kinds=["exam"]):
            if junk(e):
                continue
            d = parse_day(e["date"])
            if not d:
                continue
            n = (d - today).days
            when = "today" if n == 0 else "tomorrow" if n == 1 else f"in {n} days"
            return {"id": e["id"], "title": clean_event_title(e["title"]), "date": e["date"], "days": n,
                    "label": f"Exam {when}"}
        return None

    def latest_past_lecture(cid: int) -> dict[str, Any] | None:
        today = date.today().isoformat()
        try:
            junk = _mod("study").is_junk_event
        except (_Missing, AttributeError):
            junk = lambda e: False  # noqa: E731
        past = [e for e in merge_events(store.events(cid)) if e["kind"] == "lecture" and e["date"] <= today
                and len(e["title"]) > 3 and not junk(e)]
        return past[-1] if past else None

    def catch_up_options(cid: int, limit: int = 6) -> list[dict[str, Any]]:
        """Past lectures nobody has caught up on yet (no "What did I miss" guide for them), newest first."""
        today = date.today().isoformat()
        try:
            junk = _mod("study").is_junk_event
        except (_Missing, AttributeError):
            junk = lambda e: False  # noqa: E731
        done = {g["event_id"] for g in store.guides(cid) if g.get("kind") == "missed" and g.get("event_id") is not None}
        past = [e for e in merge_events(store.events(cid)) if e["kind"] == "lecture" and e["date"] <= today
                and len(e["title"]) > 3 and not junk(e) and e["id"] not in done]
        return list(reversed(past))[:limit]

    def course_card(c: dict[str, Any]) -> dict[str, Any]:
        cid = c["id"]
        today = date.today().isoformat()
        nxt = store.events(cid, today)[:1]
        return {**c, "accent": course_color(c), "next_event": nxt[0] if nxt else None,
                "next_exam": next_exam(cid), "catch_up_options": catch_up_options(cid),
                "event_count": len(store.events(cid)), "guide_count": len(store.guides(cid)),
                "due_cards": len(store.flashcards(cid, due_only=True))}

    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request):
        courses = [course_card(c) for c in store.courses()]
        upcoming = merge_events(upcoming_rows())
        return page(request, "dashboard.html", courses=courses, upcoming=upcoming, has_ai=provider_state()["connected"])

    # ------------------------------------------------------------------ course create / edit / delete
    def course_values(name, code, university, term, instructor, color, website_url="") -> dict[str, str]:
        return {"name": name.strip(), "code": code.strip(), "university": university.strip(), "term": term.strip(),
                "instructor": instructor.strip(), "color": clean_color(color), "website_url": website_url.strip()}

    @app.get("/courses/new", response_class=HTMLResponse)
    def course_new(request: Request):
        used = {c.get("color") for c in store.courses()}
        free = next((n for n in PALETTE if n not in used), next(iter(PALETTE)))
        return page(request, "course_form.html", course={"color": free}, editing=False)

    @app.post("/courses/new")
    def course_create(request: Request, name: str = Form(""), code: str = Form(""), university: str = Form(""),
                      term: str = Form(""), instructor: str = Form(""), color: str = Form(""), website_url: str = Form("")):
        vals = course_values(name, code, university, term, instructor, color, website_url)
        if not vals["name"]:
            msg = "Give the course a name, like Linear Algebra."
            return page(request, "course_form.html", status_code=400, course=vals, editing=False, error=msg,
                        field_error={"form": "course", "field": "name", "message": msg})
        cid = store.add_course(**vals)
        return redirect(course_url(cid), notice="Course added. Add your syllabus and lecture slides to get started.")

    @app.get("/courses/{course_id}/edit", response_class=HTMLResponse)
    def course_edit(request: Request, course_id: int):
        return page(request, "course_form.html", course=require_course(course_id), editing=True)

    @app.post("/courses/{course_id}/edit")
    def course_update(request: Request, course_id: int, name: str = Form(""), code: str = Form(""),
                      university: str = Form(""), term: str = Form(""), instructor: str = Form(""),
                      color: str = Form(""), website_url: str = Form("")):
        c = require_course(course_id)
        vals = course_values(name, code, university, term, instructor, color, website_url)
        if not vals["name"]:
            msg = "The course needs a name."
            return page(request, "course_form.html", status_code=400, course={**c, **vals}, editing=True, error=msg,
                        field_error={"form": "course", "field": "name", "message": msg})
        store.update_course(course_id, **vals)
        return redirect(course_url(course_id), notice="Course saved.")

    @app.get("/courses/{course_id}/delete", response_class=HTMLResponse)
    def course_delete_confirm(request: Request, course_id: int):
        c = require_course(course_id)
        return page(request, "confirm.html", title=f"Delete {c['name']}?", back=course_url(course_id),
                    action=f"/courses/{course_id}/delete", button="Delete the course",
                    message="This removes the course, every saved material, guide, flashcard and event, and the "
                            "original files, from this computer. It cannot be undone.")

    @app.post("/courses/{course_id}/delete")
    def course_delete(course_id: int):
        require_course(course_id)
        store.delete_course(course_id)
        shutil.rmtree(paths.course_files_dir(course_id), ignore_errors=True)
        return redirect("/", notice="Course deleted.")

    # ------------------------------------------------------------------ course page and tabs
    def worker_alive() -> bool:
        w = getattr(app.state, "worker", None)
        return bool(w is not None and getattr(w, "is_alive", lambda: False)())

    def whisper_available() -> bool:
        try:
            import importlib.util
            return importlib.util.find_spec("faster_whisper") is not None
        except (ImportError, ValueError):
            return False

    def job_info(m: dict[str, Any], j: dict[str, Any] | None, is_media: bool, ai_connected: bool) -> dict[str, Any]:
        """What is happening with a recording or photo, and what the student can do next."""
        waiting = m["status"] == "needs_transcription" or (j is not None and j["status"] in ("queued", "running", "error"))
        if not waiting and not (j and j["status"] == "done"):
            return {"text": "", "next": "", "retry": False, "poll": False}
        noun = "transcription" if is_media else "reading"
        if j is None:
            text = "Waiting to be transcribed" if is_media else "Photo of notes: waiting to be read"
            nxt = "Press Start now to begin. It can take a few minutes for a long recording." if is_media else (
                "Press Start now to read it." if ai_connected else "Connect an AI first, then press Start now.")
            return {"text": text, "next": nxt, "retry": True, "poll": False}
        status = j["status"]
        pct = int(float(j.get("progress") or 0) * 100)
        if status == "queued":
            if worker_alive():
                nxt = "It starts by itself. You can keep using the app; this page does not update on its own, so reload it to check."
            else:
                nxt = "Press Start now to run it right away."
            if not is_media and not ai_connected:
                nxt = "Next step: connect an AI so the app can read this photo, then press Start now."
            elif is_media and not whisper_available():
                nxt = ("Next step: this computer cannot transcribe yet. Add a captions file (.vtt or .srt) for this "
                       "recording, or install the transcribe extra.")
            can_run = worker_alive() and ((ai_connected if not is_media else whisper_available()))
            return {"text": f"Waiting in line for {noun}", "next": nxt, "retry": True, "poll": can_run}
        if status == "running":
            text = f"Transcribing, {pct}% done" if is_media else "Reading the photo"
            return {"text": text, "next": "Keep the app open. This updates by itself.", "retry": False, "poll": True}
        if status == "done":
            return {"text": "Transcription finished" if is_media else "Photo read", "next": "", "retry": False, "poll": False}
        note = (j.get("note") or "").strip()
        return {"text": f"{noun.capitalize()} did not finish" + (f": {note}" if note else ""),
                "next": "Fix what the message says, then press Try again.", "retry": True, "poll": False}

    def material_rows(cid: int) -> list[dict[str, Any]]:
        jobs: dict[int, dict[str, Any]] = {}
        for j in store._rows("SELECT * FROM jobs WHERE material_id IN (SELECT id FROM materials WHERE course_id = ?) ORDER BY id", (cid,)):
            jobs[int(j["material_id"])] = j
        mats = store.materials(cid)
        by_id = {m["id"]: m for m in mats}
        ai_connected = provider_state()["connected"]
        rows = []
        ev_titles = {e["id"]: e for e in store.events(cid)}
        for m in mats:
            is_media = Path(m.get("filename") or "").suffix.lower() in MEDIA_EXT
            info = job_info(m, jobs.get(m["id"]), is_media, ai_connected)
            orig = by_id.get(m.get("duplicate_of")) if m.get("duplicate_of") else None
            rows.append({**m, "kind_label": MATERIAL_KINDS.get(m["kind"], m["kind"]), "job_text": info["text"],
                         "job_next": info["next"], "job_retry": info["retry"], "job_poll": info["poll"], "is_media": is_media,
                         "copy_of": {"id": orig["id"], "title": orig["title"]} if orig else None,
                         "lecture_link": (clean_event_title(ev_titles[m["event_id"]]["title"])
                                          if m.get("event_id") in ev_titles else "")})
        return rows

    @app.get("/materials/{material_id}/status", response_class=HTMLResponse)
    def material_status(request: Request, material_id: int):
        """The little status block of one recording or photo. The Materials tab asks for it every few seconds while
        the job is waiting or running, so the page updates without a reload."""
        m = store.material(material_id)
        if not m:
            raise _NotFound("That material was not found.")
        row = next((r for r in material_rows(m["course_id"]) if r["id"] == material_id), None)
        return templates.TemplateResponse(request, "_job_status.html", {"m": row, "course": store.course(m["course_id"])})

    def graded_panel(cid: int) -> dict[str, Any]:
        """The graded items the app protects, so the student can check what was detected."""
        items = store.graded_items(cid)
        titles = {m["id"]: m["title"] for m in store.materials(cid)}
        integ = _safe_mod("integrity")
        plain = getattr(integ, "plain_text", None) or (lambda t: " ".join((t or "").split()))
        klabel = getattr(integ, "kind_label", None)
        has_key = getattr(integ, "item_key", lambda i: [])
        key_state = getattr(integ, "key_state", lambda i, connected: "on" if has_key(i) else ("failed" if connected else "off"))
        connected = provider() is not None
        rows = []
        for i in sorted(items, key=lambda r: r["id"]):
            row = {k: v for k, v in i.items() if k != "answer_key"}  # the key is never sent to a page
            row.update({"kind_label": klabel(i) if callable(klabel) else GRADED_KINDS.get(i.get("kind") or "problem", "Problem"),
                        "material_title": titles.get(i.get("material_id") or 0, ""),
                        "excerpt": plain(i.get("body") or "")[:160],
                        "answer_check": key_state(i, connected) if (i.get("kind") or "problem") == "problem" else None})
            rows.append(row)
        warnings = []
        report = getattr(integ, "split_report", None)
        if callable(report):
            for m in store.materials(cid):
                if m.get("graded"):
                    r = report(store, m["id"])
                    if r.get("short"):
                        warnings.append(f"Found {r['found']} of {r['expected']} problems in {m['title']}. Check the list "
                                        "and use Add graded item for anything missing, so the app protects all of it.")
        return {"items": rows, "count": len(rows), "warnings": warnings}

    def answer_check_note(cid: int) -> dict[str, str] | None:
        """One line for the Ask page: is the graded-answer check on? Read from integrity.key_state per graded problem."""
        items = [i for i in store.graded_items(cid) if (i.get("kind") or "problem") == "problem"]
        if not items:
            return None
        integ = _safe_mod("integrity")
        has_key = getattr(integ, "item_key", lambda i: [])
        key_state = getattr(integ, "key_state", lambda i, connected: "on" if has_key(i) else ("failed" if connected else "off"))
        connected = provider() is not None
        states = [key_state(i, connected) for i in items]
        on, failed = states.count("on"), states.count("failed")
        n = lambda k: f"{k} graded item{'' if k == 1 else 's'}"
        if on == len(items):
            return {"text": f"Answer check: on for {n(on)}", "tone": "ok-c"}
        if on:
            return {"text": f"Answer check: on for {n(on)}, couldn't be made for {n(len(items) - on)} (Try again on the Materials tab)",
                    "tone": "warn-c"}
        if failed:
            return {"text": f"Answer check: couldn't be made for {n(failed)} (Try again on the Materials tab)", "tone": "warn-c"}
        return {"text": f"Answer check: off for {n(len(items))} (connect an AI to turn it on)", "tone": "warn-c"}

    def link_suggestion_rows(cid: int) -> list[dict[str, Any]]:
        """Notes whose title looks like a lecture but are not linked: one-tap "Link to Lecture N" buttons."""
        fn = getattr(_safe_mod("study"), "link_suggestions", None)
        if not callable(fn):
            return []
        try:
            found = fn(store, cid)
        except Exception:
            return []
        label = getattr(_safe_mod("study"), "event_label", None)
        rows = []
        for x in found:
            e = x["event"]
            rows.append({"material_id": x["material"]["id"], "title": x["material"]["title"], "event_id": e["id"],
                         "lecture": clean_event_title(e["title"]), "why": x.get("why", ""),
                         "label": (label(e) if callable(label) else clean_event_title(e["title"]))})
        return rows

    def tab_context(cid: int, tab: str, extra: dict[str, Any]) -> dict[str, Any]:
        ctx: dict[str, Any] = {}
        today = date.today()
        if tab == "materials":
            ctx["materials"] = material_rows(cid)
            find = getattr(_safe_mod("retrieval"), "material_matches", None)  # titles, file names and passage text
            ctx["material_matches"] = (lambda q: find(store, cid, q)) if callable(find) else (lambda q: {})
            ctx["supported"] = _supported_formats()
            ctx["lectures"] = list(reversed(lecture_choices(cid)))
            ctx["graded"] = graded_panel(cid)
            ctx["link_suggestions"] = link_suggestion_rows(cid)
        elif tab == "calendar":
            events = merge_events(store.events(cid))
            earlier = [e for e in events if e["date"] < today.isoformat()]
            weeks: dict[date, list[dict[str, Any]]] = {}
            for e in events:
                d = parse_day(e["date"])
                if d and e["date"] >= today.isoformat():
                    weeks.setdefault(week_start(d), []).append(e)
            ctx["weeks"] = [{"label": week_label(w), "events": weeks[w]} for w in sorted(weeks)]
            ctx["earlier"] = list(reversed(earlier))
            ctx["today"] = today.isoformat()
        elif tab == "study":
            ctx["events"] = [e for e in merge_events(store.events(cid, today.isoformat())) if e["kind"] in ("lecture", "exam", "quiz")][:30]
            ctx["past_lectures"] = [e for e in merge_events(store.events(cid)) if e["kind"] == "lecture" and e["date"] <= today.isoformat()][-15:]
            ctx["has_materials"] = bool(store.materials(cid))
            ctx["next_exam"] = next_exam(cid)
            ctx["due_cards"] = len(store.flashcards(cid, due_only=True))
            ctx["graded_count"] = len(store.graded_items(cid)) + len([m for m in store.materials(cid) if m.get("graded")])
            ctx["all_cards"] = len(store.flashcards(cid))
            opts = getattr(_safe_mod("study"), "study_form_options", None)
            if callable(opts):
                ctx["study_opts"] = opts(store, cid)
        elif tab == "guides":
            guides = store.guides(cid)
            ctx["guides"] = guides
            ctx["guide_groups"] = group_guides(guides)
        elif tab == "ask":
            history = []
            last_q = ""
            for m in store.chat_messages(cid, 40):
                item: dict[str, Any] = {"role": m["role"], "content": m["content"]}
                if m["role"] == "user":
                    last_q = m["content"]
                if m["role"] == "assistant":
                    g = store.guide(m["guide_id"]) if m.get("guide_id") else None
                    saved = ((g or {}).get("data") or {}).get("blocks")
                    item["view"] = answer_view(m["content"], m.get("sources"),
                                               blocks=saved if isinstance(saved, list) else None,
                                               twin=bool(g and g["kind"] == "twin"),
                                               mode=((g or {}).get("data") or {}).get("mode"),
                                               data=(g or {}).get("data"), store=store, course_id=cid,
                                               question=last_q)
                    item["question"] = last_q
                history.append(item)
            ctx["history"] = history
            ctx["answer_check"] = answer_check_note(cid)
            ctx["graded_count"] = len(store.graded_items(cid)) + len([m for m in store.materials(cid) if m.get("graded")])
        elif tab == "offerings":
            cands = store.offering_candidates(cid)
            ctx["candidates"] = cands
            ctx["pickable"] = [c for c in cands if c["status"] in ("found", "blocked")]
        ctx.update(extra)
        return ctx

    def _supported_formats() -> str:
        try:
            exts = _mod("importers").SUPPORTED_EXTENSIONS
            return ", ".join(sorted(exts))
        except Exception:
            return ".pdf, .pptx, .docx, .epub, .md, .txt, .html, .vtt, .srt, .mp3, .m4a, .wav, .mp4, .mov, .webm, .png, .jpg, .jpeg, .heic"

    TAB_LABELS = dict(TABS)

    def course_page(request: Request, cid: int, tab: str, status_code: int = 200, notice: str = "", error: str = "", **extra):
        course = require_course(cid)
        ctx = {"course": course, "accent": course_color(course), "tab": tab, "ai": provider_state(),
               "tab_label": TAB_LABELS.get(tab, tab.capitalize()),
               "page_title": f"{TAB_LABELS.get(tab, tab.capitalize())} - {course['name']}",
               **tab_context(cid, tab, extra)}
        if notice:
            ctx["notice"] = notice
        if error:
            ctx["error"] = error
        name = "course_body.html" if is_htmx(request) else "course.html"
        resp = page(request, name, status_code=status_code, **ctx)
        if is_htmx(request):
            resp.headers["HX-Push-Url"] = course_url(cid, tab)
        return resp

    def after(request: Request, cid: int, tab: str, notice: str = "", error: str = "", field: tuple[str, str] | None = None,
              **extra):
        """Answer a form: htmx swaps the tab in place; a plain post goes back to the tab. `field` is (form, field name)
        for an error that belongs to one input, so the page can tie the message to it."""
        if is_htmx(request):
            if field and error:
                extra["field_error"] = {"form": field[0], "field": field[1], "message": error}
            return course_page(request, cid, tab, 400 if error and not notice else 200, notice, error, **extra)
        return redirect(course_url(cid, tab), notice=notice, error=error, draft=str(extra.get("draft") or ""))

    @app.get("/courses/{course_id}", response_class=HTMLResponse)
    def course_materials(request: Request, course_id: int):
        return course_page(request, course_id, "materials")

    @app.get("/courses/{course_id}/{tab}", response_class=HTMLResponse)
    def course_tab(request: Request, course_id: int, tab: str):
        if tab not in {t for t, _ in TABS}:
            raise _NotFound("That page was not found.")
        extra: dict[str, Any] = {}
        about = " ".join((request.query_params.get("about") or "").split())[:120]
        if tab == "ask" and about:  # the dashboard's "Ask a question about X": start the question for them
            extra["draft"] = f"About {about}: "
            extra["autofocus_question"] = True
        return course_page(request, course_id, tab, **extra)

    # ------------------------------------------------------------------ materials
    @app.post("/courses/{course_id}/materials/upload")
    async def upload_materials(request: Request, course_id: int, files: list[UploadFile] = File(default=[]),
                               kind: str = Form(""), lecture_date: str = Form(""), event_id: str = Form("")):
        require_course(course_id)
        files = [f for f in files if f.filename]
        if not files:
            return after(request, course_id, "materials", error="Choose at least one file to add.", field=("upload", "files"))
        if lecture_date and not parse_day(lecture_date):
            return after(request, course_id, "materials", error="The lecture date should look like 2026-09-14.",
                         field=("upload", "lecture_date"))
        kind = kind if kind in MATERIAL_KINDS else ""
        lecture = None
        if event_id.strip():
            lecture = store.event(int(event_id)) if event_id.strip().isdigit() else None
            if not lecture or lecture["course_id"] != course_id or lecture["kind"] != "lecture":
                return after(request, course_id, "materials", error="That lecture is not in this course. Pick one from the list.",
                             field=("upload", "event_id"))
            lecture_date = lecture_date.strip() or lecture["date"]

        def run() -> tuple[list[str], list[str], list[str]]:
            importers = _mod("importers")
            bad = getattr(importers, "ImportError_", ())
            done, errors, already = [], [], []
            for up in files:
                name = Path(up.filename or "file").name or "file"
                tmpdir = Path(tempfile.mkdtemp(prefix="ca-upload-"))
                try:
                    tmp = tmpdir / re.sub(r"[^\w.\- ]", "_", name)
                    size = 0
                    with tmp.open("wb") as out:
                        while chunk := up.file.read(1024 * 1024):
                            size += len(chunk)
                            if size > MAX_UPLOAD:
                                raise ValueError("That file is too large (the limit is 2 GB).")
                            out.write(chunk)
                    known = {m["id"] for m in store.materials(course_id)}
                    mid = importers.import_file(store, course_id, tmp, kind=kind or None, original_name=name,
                                                lecture_date=lecture_date)
                    if mid not in known and lecture:
                        store.update_material(mid, event_id=lecture["id"])
                    (already if mid in known else done).append(name)
                except Exception as e:
                    shown = _friendly(e) if (isinstance(e, ValueError) or (bad and isinstance(e, bad))) else f"Could not read it ({_friendly(e)})"
                    errors.append(f"{name}: {shown}")
                finally:
                    shutil.rmtree(tmpdir, ignore_errors=True)
            return done, errors, already

        try:
            done, errors, already = await work(run)
        except _Missing:
            return after(request, course_id, "materials", error="The importer is not available yet.")
        parts = []
        if done:
            parts.append(f"Added {len(done)} file{'s' if len(done) != 1 else ''}: {', '.join(done)}."
                         + (f" Linked to {clean_event_title(lecture['title'])}." if lecture else ""))
        if already:
            parts.append(f"Already saved, nothing added: {', '.join(already)}.")
        return after(request, course_id, "materials", notice=" ".join(parts), error=" ".join(errors))

    @app.post("/courses/{course_id}/materials/url")
    async def add_url(request: Request, course_id: int, url: str = Form(""), kind: str = Form("website")):
        require_course(course_id)
        url = url.strip()
        if not re.match(r"^https?://\S+$", url):
            return after(request, course_id, "materials", error="Paste a web address that starts with http:// or https://.",
                         field=("url", "url"))
        kind = kind if kind in MATERIAL_KINDS else "website"
        try:
            await work(lambda: _mod("importers").import_url(store, course_id, url, kind=kind))
        except Exception as e:
            return after(request, course_id, "materials", error=f"Could not save that page: {_friendly(e)}")
        return after(request, course_id, "materials", notice="Saved that page so you can read it offline.")

    @app.post("/courses/{course_id}/materials/note")
    async def add_note(request: Request, course_id: int, title: str = Form(""), text: str = Form(""),
                       kind: str = Form("notes")):
        require_course(course_id)
        if not text.strip():
            return after(request, course_id, "materials", error="Type or paste some text for the note.", field=("note", "text"))
        kind = kind if kind in MATERIAL_KINDS else "notes"
        try:
            await work(lambda: _mod("importers").import_text(store, course_id, title.strip() or "My note", text, kind=kind))
        except Exception as e:
            return after(request, course_id, "materials", error=f"Could not save the note: {_friendly(e)}")
        return after(request, course_id, "materials", notice="Note saved.")

    def build_keys(item_ids: list[int]) -> None:
        """One model call per numeric item (when an AI is connected) to make its answer key. Failures leave it empty."""
        p = provider()
        build = getattr(_safe_mod("integrity"), "build_answer_key", None)
        if p is None or not callable(build):
            return
        for iid in item_ids:
            try:
                build(store, iid, p)
            except Exception:
                pass

    def mark_graded(m: dict[str, Any]) -> list[dict[str, Any]]:
        """Mark a material as graded work. The integrity module finds the individual problems or essay prompts;
        without it, only the whole-material flag is set. Returns the graded items found for this material."""
        mark = getattr(_safe_mod("integrity"), "mark_material_graded", None)
        if callable(mark):
            mark(store, m["id"])
        store.update_material(m["id"], graded=1)
        return [i for i in store.graded_items(m["course_id"]) if i.get("material_id") == m["id"]]

    def _safe_mod(name: str):
        try:
            return _mod(name)
        except _Missing:
            return None

    @app.post("/materials/{material_id}/graded")
    def toggle_graded(request: Request, material_id: int, graded: str = Form("1")):
        m = store.material(material_id)
        if not m:
            raise _NotFound("That material was not found.")
        if graded == "1":
            items = mark_graded(m)
            build_keys([i["id"] for i in items if not (i.get("answer_key") or "").strip()])
            if items:
                labels = ", ".join(i.get("label") or i.get("title") or "item" for i in items[:8])
                more = f" and {len(items) - 8} more" if len(items) > 8 else ""
                note = (f"Marked {m['title']} as graded work. The app found {len(items)} graded "
                        f"item{'s' if len(items) != 1 else ''} to protect: {labels}{more}. Check the list below the "
                        "materials and fix anything it got wrong. The app will explain and write twin problems, never solve it.")
            else:
                note = (f"Marked {m['title']} as graded work. The app will explain and write twin problems, never solve it.")
        else:
            store.update_material(material_id, graded=0)
            removed = 0
            for i in store.graded_items(m["course_id"]):
                if i.get("material_id") == material_id:
                    store.delete_graded_item(i["id"])
                    removed += 1
            note = f"{m['title']} is no longer marked as graded." + (f" Its {removed} protected item{'s' if removed != 1 else ''} were removed." if removed else "")
        return after(request, m["course_id"], "materials", notice=note)

    def graded_or_404(item_id: int) -> dict[str, Any]:
        it = store.graded_item(item_id)
        if not it:
            raise _NotFound("That graded item was not found.")
        return it

    @app.get("/courses/{course_id}/graded/new", response_class=HTMLResponse)
    def graded_new(request: Request, course_id: int):
        return page(request, "graded_form.html", is_new=True, item={"title": "", "label": "", "kind": "problem", "body": ""},
                    course=require_course(course_id), page_title="Add graded item",
                    graded_materials=[m for m in store.materials(course_id) if m.get("graded")])

    @app.post("/courses/{course_id}/graded/new")
    def graded_create(request: Request, course_id: int, kind: str = Form("problem"), label: str = Form(""),
                      title: str = Form(""), body: str = Form(""), material_id: str = Form("")):
        course = require_course(course_id)
        gm = [m for m in store.materials(course_id) if m.get("graded")]
        mid = int(material_id) if material_id.strip().isdigit() and int(material_id) in {m["id"] for m in gm} else None
        vals = {"kind": kind if kind in GRADED_KINDS else "problem", "label": label.strip(), "title": title.strip(),
                "body": body.strip()}
        if not vals["title"]:
            msg = "Give the item a title, like Problem Set 2."
            return page(request, "graded_form.html", status_code=400, is_new=True, item=vals, course=course, error=msg,
                        field_error={"form": "graded", "field": "title", "message": msg}, page_title="Add graded item",
                        graded_materials=gm)
        iid = store.add_graded_item(course_id, vals["title"], vals["body"], mid, kind=vals["kind"], label=vals["label"])
        build_keys([iid])
        return redirect(course_url(course_id), notice=f"Added {vals['label'] or vals['title']} to the graded list.")

    @app.get("/graded/{item_id}/edit", response_class=HTMLResponse)
    def graded_edit(request: Request, item_id: int):
        it = graded_or_404(item_id)
        integ = _safe_mod("integrity")
        state = getattr(integ, "key_state", lambda i, connected: "off")(it, provider() is not None)
        return page(request, "graded_form.html", item={k: v for k, v in it.items() if k != "answer_key"},
                    course=require_course(it["course_id"]), page_title="Edit graded item", answer_check=state)

    @app.post("/graded/{item_id}/key")
    def graded_key_retry(request: Request, item_id: int):
        """Try again: one more model call to make the item's answer key."""
        it = graded_or_404(item_id)
        p = provider()
        if p is None:
            return after(request, it["course_id"], "materials", error="Connect an AI first, then try again.")
        build = getattr(_safe_mod("integrity"), "build_answer_key", None)
        ok = False
        if callable(build):
            try:
                ok = bool(build(store, item_id, p))
            except Exception:
                ok = False
        if ok:
            return after(request, it["course_id"], "materials", notice="Answer check is on for this item.")
        return after(request, it["course_id"], "materials", error="The answer check still could not be made. Try again in a moment.")

    @app.post("/graded/{item_id}/edit")
    def graded_update(request: Request, item_id: int, kind: str = Form("problem"), label: str = Form(""),
                      title: str = Form(""), body: str = Form("")):
        it = graded_or_404(item_id)
        vals = {"kind": kind if kind in GRADED_KINDS else "problem", "label": label.strip(), "title": title.strip(),
                "body": body.strip()}
        if not vals["title"]:
            msg = "Give the item a title, like Problem Set 2."
            return page(request, "graded_form.html", status_code=400, item={**it, **vals}, course=require_course(it["course_id"]),
                        error=msg, field_error={"form": "graded", "field": "title", "message": msg}, page_title="Edit graded item")
        changed = vals["body"] != (it.get("body") or "") or vals["kind"] != (it.get("kind") or "problem")
        store.update_graded_item(item_id, **vals, **({"answer_key": ""} if changed else {}))
        if changed:
            build_keys([item_id])
        return redirect(course_url(it["course_id"]), notice=f"Saved {vals['label'] or vals['title']}.")

    @app.get("/graded/{item_id}/delete", response_class=HTMLResponse)
    def graded_delete_confirm(request: Request, item_id: int):
        it = graded_or_404(item_id)
        name = it.get("label") or it["title"]
        return page(request, "confirm.html", title=f"Stop protecting {name}?", back=course_url(it["course_id"]),
                    action=f"/graded/{item_id}/delete", button="Remove it",
                    message="The app will no longer treat this as graded work, so it may answer questions about it "
                            "directly. The material itself is not deleted.")

    @app.post("/graded/{item_id}/delete")
    def graded_delete(item_id: int):
        it = graded_or_404(item_id)
        store.delete_graded_item(item_id)
        return redirect(course_url(it["course_id"]), notice=f"Removed {it.get('label') or it['title']} from the graded list.")

    def file_of(m: dict[str, Any]) -> Path | None:
        """The saved original, only if it really sits inside the files folder."""
        name = m.get("filename") or ""
        if not name:
            return None
        base = paths.files_dir().resolve()
        target = (base / name).resolve()
        if target != base and base in target.parents and target.is_file():
            return target
        return None

    # -- edit a material
    def lecture_choices(cid: int) -> list[dict[str, Any]]:
        return [e for e in merge_events(store.events(cid)) if e["kind"] == "lecture"]

    def material_form(request: Request, m: dict[str, Any], *, status_code: int = 200, **ctx: Any):
        orig = store.material(m["duplicate_of"]) if m.get("duplicate_of") else None
        return page(request, "material_form.html", status_code=status_code, m=m, course=require_course(m["course_id"]),
                    lectures=lecture_choices(m["course_id"]), copy_of=orig, page_title=f"Edit {m['title']}", **ctx)

    @app.get("/materials/{material_id}/edit", response_class=HTMLResponse)
    def material_edit(request: Request, material_id: int):
        m = store.material(material_id)
        if not m:
            raise _NotFound("That material was not found.")
        return material_form(request, m)

    @app.post("/materials/{material_id}/edit")
    def material_update(request: Request, material_id: int, title: str = Form(""), kind: str = Form(""),
                        lecture_date: str = Form(""), event_id: str = Form("")):
        m = store.material(material_id)
        if not m:
            raise _NotFound("That material was not found.")
        shown = {**m, "title": title, "kind": kind, "lecture_date": lecture_date,
                 "event_id": int(event_id) if event_id.strip().isdigit() else None}

        def bad(field: str, msg: str):
            return material_form(request, shown, status_code=400, error=msg,
                                 field_error={"form": "material", "field": field, "message": msg})
        if not title.strip():
            return bad("title", "Give the material a title.")
        if kind not in MATERIAL_KINDS:
            return bad("kind", "Choose what this material is.")
        day = lecture_date.strip()
        if day and not parse_day(day):
            return bad("lecture_date", "The lecture date should look like 2026-09-14.")
        ev = store.event(int(event_id)) if event_id.strip().isdigit() else None
        if event_id.strip() and (not ev or ev["course_id"] != m["course_id"]):
            return bad("event_id", "That lecture is not in this course. Pick one from the list.")
        if ev and not day:
            day = ev["date"]  # a lecture was picked: the material belongs to its date
        store.update_material(material_id, title=title.strip(), kind=kind, lecture_date=day,
                              event_id=ev["id"] if ev else None)
        extra = f" It now belongs to {clean_event_title(ev['title'])}." if ev else ""
        return redirect(course_url(m["course_id"]), notice=f"Saved {title.strip()}.{extra}")

    @app.post("/materials/{material_id}/link")
    def material_link(request: Request, material_id: int, event_id: str = Form("")):
        """The one-tap "Link to Lecture N" button: set the material's lecture link (and its date, like the edit form)."""
        m = store.material(material_id)
        if not m:
            raise _NotFound("That material was not found.")
        ev = store.event(int(event_id)) if event_id.strip().isdigit() else None
        if not ev or ev["course_id"] != m["course_id"] or ev["kind"] != "lecture":
            return after(request, m["course_id"], "materials", error="That lecture is not in this course. Pick one from the list.")
        store.update_material(material_id, event_id=ev["id"], lecture_date=m.get("lecture_date") or ev["date"])
        return after(request, m["course_id"], "materials",
                     notice=f"Linked {m['title']} to {clean_event_title(ev['title'])}.")

    @app.post("/materials/{material_id}/not-copy")
    def material_not_copy(request: Request, material_id: int):
        m = store.material(material_id)
        if not m:
            raise _NotFound("That material was not found.")
        store.update_material(material_id, duplicate_of=None)
        return after(request, m["course_id"], "materials",
                     notice=f"{m['title']} is now treated as its own material.")

    @app.post("/materials/{material_id}/keep-copy")
    def material_keep_copy(request: Request, material_id: int):
        """Search this copy (a PDF with page numbers) instead of the file it was matched with."""
        m = store.material(material_id)
        if not m:
            raise _NotFound("That material was not found.")
        from .importers import keep_copy
        orig = keep_copy(store, material_id)
        note = (f"{m['title']} is now the searched copy. The other one is skipped." if orig
                else f"{m['title']} was not marked as a copy.")
        return after(request, m["course_id"], "materials", notice=note)

    @app.post("/materials/{material_id}/process")
    async def material_process(request: Request, material_id: int):
        """Start (or retry) reading a photo or transcribing a recording now."""
        m = store.material(material_id)
        if not m:
            raise _NotFound("That material was not found.")
        is_media = Path(m.get("filename") or "").suffix.lower() in MEDIA_EXT
        kind = "transcribe" if is_media else "ocr"
        job = next((j for j in reversed(store._rows("SELECT * FROM jobs WHERE material_id = ?", (material_id,)))), None)
        if job is None:
            jid = store.add_job(kind, material_id)
        else:
            jid = job["id"]
            store.update_job(jid, status="queued", progress=0, note="")
        store.update_material(material_id, status="needs_transcription", status_note="")
        if worker_alive():
            return after(request, m["course_id"], "materials", notice=f"{m['title']} is queued. It starts by itself.")
        try:
            result = await work(lambda: _mod("transcribe").run_job(store, jid, provider=provider()))
        except Exception as e:
            return after(request, m["course_id"], "materials", error=f"Could not start it: {_friendly(e)}")
        if result.get("status") == "error":
            return after(request, m["course_id"], "materials", error=f"{m['title']}: {result.get('note') or 'it did not finish'}")
        return after(request, m["course_id"], "materials", notice=f"{m['title']} is ready to search.")

    @app.get("/materials/{material_id}/delete", response_class=HTMLResponse)
    def material_delete_confirm(request: Request, material_id: int):
        m = store.material(material_id)
        if not m:
            raise _NotFound("That material was not found.")
        return page(request, "confirm.html", title=f"Delete {m['title']}?", back=course_url(m["course_id"]),
                    action=f"/materials/{material_id}/delete", button="Delete it",
                    message="This removes the material, its searchable text, the calendar dates it added and the saved "
                            "original file from this computer. Guides and answers you already made stay, but their links to this material "
                            "will show \"source deleted\" instead of opening it. It cannot be undone.")

    @app.post("/materials/{material_id}/delete")
    def material_delete(request: Request, material_id: int):
        m = store.material(material_id)
        if not m:
            raise _NotFound("That material was not found.")
        f = file_of(m)
        cascade = getattr(_mod("importers"), "delete_material", None)  # its calendar dates go with it
        if cascade:
            n_events = cascade(store, material_id)
        else:
            n_events = 0
            store.delete_material(material_id)
        if f and not store._row("SELECT id FROM materials WHERE filename = ?", (m["filename"],)):
            f.unlink(missing_ok=True)
        extra = f" Its {n_events} calendar date{'s' if n_events != 1 else ''} {'were' if n_events != 1 else 'was'} removed too." if n_events else ""
        return redirect(course_url(m["course_id"]), notice=f"Deleted {m['title']}.{extra}")

    # ------------------------------------------------------------------ calendar
    def event_fields(kind: str, title: str, day: str, time_: str, topics: str) -> tuple[dict[str, str], str, str]:
        """The cleaned values, an error message and the name of the field it belongs to."""
        vals = {"kind": kind if kind in EVENT_KINDS else "other", "title": title.strip(), "date": day.strip(),
                "time": time_.strip(), "topics": topics.strip()}
        err, field = "", ""
        if not vals["title"]:
            err, field = "Give the event a title.", "title"
        elif not parse_day(vals["date"]):
            err, field = "The date should look like 2026-10-21.", "date"
        elif vals["time"] and not re.match(r"^\d{1,2}:\d{2}$", vals["time"]):
            err, field = "The time should look like 14:30.", "time"
        return vals, err, field

    def event_copies(e: dict[str, Any]) -> list[int]:
        """Other events in the course with the same date and (cleaned) title: the same event listed twice."""
        key = clean_event_title(e["title"]).lower()
        return [o["id"] for o in store.events(e["course_id"], e["date"], e["date"])
                if o["id"] != e["id"] and clean_event_title(o["title"]).lower() == key]

    @app.post("/courses/{course_id}/events")
    def event_add(request: Request, course_id: int, kind: str = Form("lecture"), title: str = Form(""),
                  date_: str = Form("", alias="date"), time_: str = Form("", alias="time"), topics: str = Form("")):
        require_course(course_id)
        vals, err, field = event_fields(kind, title, date_, time_, topics)
        if err:
            return after(request, course_id, "calendar", error=err, field=("event", field))
        store.add_event(course_id, vals.pop("kind"), vals.pop("title"), vals.pop("date"), **vals)
        return after(request, course_id, "calendar", notice="Event added.")

    @app.get("/events/{event_id}/edit", response_class=HTMLResponse)
    def event_edit(request: Request, event_id: int):
        e = store.event(event_id)
        if not e:
            raise _NotFound("That event was not found.")
        return page(request, "event_form.html", event=e, course=require_course(e["course_id"]))

    @app.post("/events/{event_id}/edit")
    def event_update(request: Request, event_id: int, kind: str = Form("other"), title: str = Form(""),
                     date_: str = Form("", alias="date"), time_: str = Form("", alias="time"), topics: str = Form("")):
        e = store.event(event_id)
        if not e:
            raise _NotFound("That event was not found.")
        vals, err, field = event_fields(kind, title, date_, time_, topics)
        if err:
            return page(request, "event_form.html", status_code=400, event={**e, **vals}, course=require_course(e["course_id"]),
                        error=err, field_error={"form": "event", "field": field, "message": err})
        copies = event_copies(e)  # found before the edit changes the date or title
        store.update_event(event_id, **vals)
        for cid_ in copies:  # the same event listed twice becomes one
            store.delete_event(cid_)
        return redirect(course_url(e["course_id"], "calendar"), notice="Event saved." + (" Duplicate copies were merged." if copies else ""))

    @app.get("/events/{event_id}/delete", response_class=HTMLResponse)
    def event_delete_confirm(request: Request, event_id: int):
        e = store.event(event_id)
        if not e:
            raise _NotFound("That event was not found.")
        return page(request, "confirm.html", title=f"Delete {e['title']}?", back=course_url(e["course_id"], "calendar"),
                    action=f"/events/{event_id}/delete", button="Delete the event",
                    message="This removes the event from the course calendar. If the same event is listed more than once, "
                            "every copy is removed.")

    @app.post("/events/{event_id}/delete")
    def event_delete(event_id: int):
        e = store.event(event_id)
        if not e:
            raise _NotFound("That event was not found.")
        for cid_ in event_copies(e):
            store.delete_event(cid_)
        store.delete_event(event_id)
        return redirect(course_url(e["course_id"], "calendar"), notice="Event deleted.")

    # ------------------------------------------------------------------ study tools
    STUDY_LABELS = {"primer": "Pre-lecture primer", "exam": "Exam guide", "map": "Concept map", "explain": "Explanation",
                    "practice": "Practice questions", "flashcards": "Flashcards", "missed": "What did I miss"}

    def nothing_saved(cid: int, event_id: int | None, day: str) -> str:
        """Said in the app, before any model call, when no saved material belongs to the lecture."""
        ev = store.event(event_id) if event_id is not None else None
        if ev and ev["course_id"] != cid:
            return ""
        try:
            study = _mod("study")
            if ev:
                found = study.lecture_materials(store, cid, ev)
            else:
                found = [m for m in store.materials(cid) if m.get("lecture_date") == day and not m.get("graded")
                         and not m.get("duplicate_of")]
        except (_Missing, AttributeError):
            return ""
        if found:
            return ""
        name = clean_event_title(ev["title"]) if ev else day
        return (f"Nothing is saved for {name} yet, so there is nothing to catch up on. Add its notes, slides or captions "
                "on the Materials tab and link them to the lecture (the Which lecture is it for? choice), then try again.")

    @app.post("/courses/{course_id}/study/{action}")
    async def study_run(request: Request, course_id: int, action: str, topic: str = Form(""), depth: str = Form("standard"),
                        count: int = Form(0), event_id: str = Form(""), day: str = Form(""),
                        material_id: str = Form(""), material_ids: list[str] = Form(default=[])):
        require_course(course_id)
        if action not in STUDY_LABELS:
            raise _NotFound("That study tool was not found.")
        p = provider()
        if p is None:
            return after(request, course_id, "study", error="Connect an AI first. See the panel below.")
        ev = int(event_id) if event_id.strip().isdigit() else None
        mat = int(material_id) if material_id.strip().isdigit() else None
        mats = [int(v) for v in material_ids if str(v).strip().isdigit()] or None
        depth = depth if depth in {d for d, _ in DEPTHS} else "standard"
        topic = topic.strip()
        if action in ("explain",) and not topic:
            return after(request, course_id, "study", error="Type the topic you want explained.", field=(action, "topic"))
        if action == "practice" and not topic:
            return after(request, course_id, "study", error="Type the topic you want practice questions on.", field=(action, "topic"))

        # a primer is for a lecture that has not happened: with nothing linked it is built from the syllabus reading
        if action == "missed" and (ev is not None or day.strip()):
            note = nothing_saved(course_id, ev, day.strip())
            if note:
                return after(request, course_id, "study", error=note, field=(action, "event_id"))

        def run():
            study = _mod("study")
            _safe_fill_keys(course_id, p)
            if action == "primer":
                return study.primer(store, course_id, p, event_id=ev)
            if action == "exam":
                return study.exam_guide(store, course_id, p, event_id=ev, material_ids=mats)
            if action == "map":
                return study.concept_map(store, course_id, p, scope=topic)
            if action == "explain":
                return study.explain(store, course_id, p, topic, depth=depth)
            if action == "practice":
                return study.practice(store, course_id, p, topic, count=count or 5)
            if action == "flashcards":
                return study.make_flashcards(store, course_id, p, material_id=mat, topic=topic, count=count or 12)
            return study.missed(store, course_id, p, event_id=ev, date=day.strip())

        try:
            result = await work(run)
        except Exception as e:
            if _is_no_provider(e):
                return after(request, course_id, "study", error="Connect an AI first. See the panel below.")
            return after(request, course_id, "study", error=f"{STUDY_LABELS[action]} did not work: {_friendly(e)}")
        if action == "flashcards":
            n = len(result or [])
            note = (getattr(result, "note", "") or "").strip()
            return after(request, course_id, "study", notice=note or f"Made {n} flashcard{'s' if n != 1 else ''}.",
                         cards_made=n, cards_note=note)
        guide = store.guide(int(result))
        rm = getattr(_mod("study"), "ready_message", None)
        kind = (guide or {}).get("kind") or action
        ready = rm(kind) if callable(rm) else f"{STUDY_LABELS.get(action, 'Your guide')} is ready."
        ready = _mod("essay").ready_message(guide, ready)  # essay and proof pages never say "twin problem"
        if not is_htmx(request):
            return redirect(f"/guides/{int(result)}", notice=ready)
        return after(request, course_id, "study", notice=ready, result_guide=guide,
                     result_view=guide_view(guide, base_level=4, store=store) if guide else None)

    def _safe_fill_keys(cid: int, p) -> None:
        """Lazily make the answer key of graded numeric items that have none (one model call each, once)."""
        try:
            _mod("integrity").ensure_keys(store, cid, p)
        except Exception:
            pass

    # ------------------------------------------------------------------ ask
    @app.post("/courses/{course_id}/ask")
    async def ask(request: Request, course_id: int, question: str = Form(""), depth: str = Form("standard"),
                  normal: str = Form("")):
        require_course(course_id)
        q = question.strip()
        if not q:
            return after(request, course_id, "ask", error="Type a question first.", field=("ask", "question"))
        p = provider()
        if p is None:
            return after(request, course_id, "ask", error="Connect an AI first. See the panel below.", draft=q)
        depth = depth if depth in {d for d, _ in DEPTHS} else "standard"
        history = [{"role": m["role"], "content": m["content"]} for m in store.chat_messages(course_id, 6)]

        try:
            def run():
                _safe_fill_keys(course_id, p)
                return _mod("grounding").ask(store, course_id, q, p, depth=depth, history=history,
                                             route=normal.strip() != "1")

            await work(run)
        except Exception as e:
            if _is_no_provider(e):
                return after(request, course_id, "ask", error="Connect an AI first. See the panel below.", draft=q)
            return after(request, course_id, "ask", error=f"The answer did not work: {_friendly(e)}", draft=q)
        return after(request, course_id, "ask", notice="Answer ready. It is the last message below.")

    # ------------------------------------------------------------------ past offerings
    def _offering_report(off, st, course_id: int, text: str, **kw) -> dict[str, Any]:
        fn = getattr(off, "find_past_offerings_report", None)
        if callable(fn):
            return fn(st, course_id, text, **kw)
        return {"found": off.find_past_offerings(st, course_id, text, **kw), "skipped": [], "unreachable": [], "message": ""}

    @app.post("/courses/{course_id}/offerings/paste")
    async def offerings_paste(request: Request, course_id: int, links: str = Form(""), term_label: str = Form("")):
        require_course(course_id)
        if not re.search(r"https?://\S+", links):
            return after(request, course_id, "offerings", error="Paste one or more web addresses that start with http:// or https://.")
        off = _mod("offerings")
        try:
            rep = await work(lambda: _offering_report(off, store, course_id, links))
        except Exception as e:
            return after(request, course_id, "offerings", error=f"Those links did not work: {_friendly(e)}")
        found = rep["found"]
        label = term_label.strip()
        if label:
            for c in found:
                if not c.get("term_label"):
                    store.update_offering_candidate(c["id"], term_label=label)
        n = len(found)
        skipped = " ".join(off.report_lines(rep)) if hasattr(off, "report_lines") else ""
        if not n and skipped:
            return after(request, course_id, "offerings", error=f"Nothing was added. {skipped}")
        return after(request, course_id, "offerings",
                     notice=f"Added {n} link{'s' if n != 1 else ''} to the list below. Tick the ones to save."
                            + (f" {skipped}" if skipped else ""))

    @app.post("/courses/{course_id}/offerings/search")
    async def offerings_search(request: Request, course_id: int):
        course = require_course(course_id)
        off = _mod("offerings")
        try:
            rep = await work(lambda: _offering_report(off, store, course_id, "", found_by="crawl"))
        except Exception as e:
            return after(request, course_id, "offerings", error=f"The search did not work: {_friendly(e)}")
        found = rep["found"]
        if not found:
            if rep.get("message"):
                return after(request, course_id, "offerings", error=rep["message"])
            return after(request, course_id, "offerings",
                         notice=f"Nothing new turned up for {course['name']}. You can paste links you know of instead.")
        return after(request, course_id, "offerings", notice=f"Found {len(found)} possible page{'s' if len(found) != 1 else ''}. Tick the ones to save.")

    @app.post("/courses/{course_id}/offerings/save")
    async def offerings_save(request: Request, course_id: int):
        require_course(course_id)
        form = await request.form()
        picked = [int(v) for v in form.getlist("pick") if str(v).isdigit()]
        if not picked:
            return after(request, course_id, "offerings", error="Tick at least one page to save.")

        def run() -> tuple[int, list[str]]:
            importers = _mod("importers")
            saved, problems = 0, []
            for cid_ in picked:
                cand = store.offering_candidate(cid_)
                if not cand or cand["course_id"] != course_id or cand["status"] == "saved":
                    continue
                try:
                    mid = importers.import_url(store, course_id, cand["url"], kind="past_offering", added_by="student")
                    store.update_offering_candidate(cid_, status="saved", material_id=mid, note="")
                    saved += 1
                except Exception as e:
                    blocked = type(e).__name__ == "FetchBlocked"
                    msg = _friendly(e)
                    store.update_offering_candidate(cid_, status="blocked" if blocked else "found", note=msg)
                    problems.append(f"{cand['title'] or cand['url']}: {msg}")
            return saved, problems

        try:
            saved, problems = await work(run)
        except _Missing:
            return after(request, course_id, "offerings", error="The importer is not available yet.")
        return after(request, course_id, "offerings",
                     notice=f"Saved {saved} page{'s' if saved != 1 else ''} to your materials." if saved else "",
                     error=" ".join(problems))

    @app.post("/courses/{course_id}/offerings/{cand_id}/dismiss")
    def offerings_dismiss(request: Request, course_id: int, cand_id: int):
        cand = store.offering_candidate(cand_id)
        if cand and cand["course_id"] == course_id:
            store.update_offering_candidate(cand_id, status="dismissed")
        return after(request, course_id, "offerings", notice="Removed from the list.")

    # ------------------------------------------------------------------ reader and files
    def _was_material(material_id: int) -> bool:
        """True when this id belonged to a material that was deleted: it is below the newest id in use (ids are not
        reused below it), or a saved answer, guide or chat message still cites it."""
        try:
            if store._row("SELECT 1 AS x FROM deleted_materials WHERE id = ?", (material_id,)):
                return True
        except Exception:  # noqa: BLE001
            pass
        top = store._row("SELECT MAX(id) AS m FROM materials")
        if material_id <= int((top or {}).get("m") or 0):
            return True
        needle = f'"material_id": {int(material_id)}'
        for table in ("guides", "chat_messages"):
            try:
                if store._row(f"SELECT 1 AS x FROM {table} WHERE sources_json LIKE ? LIMIT 1", (f"%{needle}%",)):
                    return True
            except Exception:  # noqa: BLE001 - a missing table just means no trace
                continue
        return False

    @app.get("/read/{material_id}", response_class=HTMLResponse)
    def reader(request: Request, material_id: int, at: str | None = None):
        m = store.material(material_id)
        if not m and not _was_material(material_id):
            return friendly_page(request, 404, "No such passage", "There is no passage with that address. The link may "
                                 "be mistyped. Go back to your course to open your materials.")
        if not m:
            return friendly_page(request, 410, "This passage is no longer in your materials",
                                 "This passage is no longer in your materials. It came from a material that has been "
                                 "deleted, so it can no longer be opened. Answers and guides that cited it are still "
                                 "saved, but their link to it is gone. Add the material again if you still need it.")
        course = store.course(m["course_id"])
        chunks = store.chunks_of(material_id)
        media = MEDIA_EXT.get(Path(m.get("filename") or "").suffix.lower(), "") if file_of(m) else ""
        at_id: int | None = None
        at_note = ""
        if at not in (None, ""):
            if re.fullmatch(r"\d{1,12}", str(at).strip()):
                at_id = int(str(at).strip())
                if at_id not in {c["id"] for c in chunks}:
                    at_id = None
                    at_note = ("The passage this link points to is no longer in this material. It may have been "
                               "re-imported or replaced. The whole material is shown below.")
            else:
                at_note = "This link's passage reference is not valid, so no passage is highlighted."
        code_like = Path(m.get("filename") or m.get("original_name") or "").suffix.lower() in CODE_EXT
        for c in chunks:
            c["seek"] = int(float(c["loc_start"])) if c["loc_kind"] == "time" and re.match(r"^\d+(\.\d+)?$", str(c["loc_start"])) else None
            c["range_label"] = chunk_range_label(c)
            c["start_label"] = locations.time_label(float(c["loc_start"])) if c["seek"] is not None else ""
            c["html"] = render_reader(c["text"], base_level=3, math=not code_like)
        file_frag = ""
        if at_id and Path(m.get("filename") or m.get("original_name") or "").suffix.lower() == ".pdf":
            hit = next((c for c in chunks if c["id"] == at_id), None)
            if hit and hit.get("loc_kind") == "page" and re.fullmatch(r"\d{1,5}", str(hit.get("loc_start") or "").split(".")[0]):
                file_frag = f"#page={str(hit['loc_start']).split('.')[0]}"
        source = m["source_url"] if re.match(r"^https?://\S+$", m.get("source_url") or "") else ""
        return page(request, "read.html", m=m, course=course, accent=course_color(course), chunks=chunks, at=at_id,
                    at_note=at_note, media=media, has_file=bool(file_of(m)), source_url=source, file_frag=file_frag,
                    kind_label=MATERIAL_KINDS.get(m["kind"], m["kind"]))

    @app.get("/files/{material_id}")
    def original_file(material_id: int):
        m = store.material(material_id)
        f = file_of(m) if m else None
        if not f:
            return PlainTextResponse("File not found", status_code=404)
        ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
        headers = {"X-Content-Type-Options": "nosniff"}
        name = Path(m.get("original_name") or f.name).name
        safe_name = quote(name)
        if ctype in ("text/html", "application/xhtml+xml"):
            # a saved web page: show it, but with scripts and outside requests switched off
            headers["Content-Security-Policy"] = "sandbox; default-src 'none'; img-src data:; style-src 'unsafe-inline'"
            disp = "inline"
        elif ctype.startswith(INLINE_TYPES):
            disp = "inline"
        else:
            disp = "attachment"
        headers["Content-Disposition"] = f"{disp}; filename*=UTF-8''{safe_name}"
        return FileResponse(f, media_type=ctype, headers=headers)

    # ------------------------------------------------------------------ guides
    @app.get("/guides/{guide_id}", response_class=HTMLResponse)
    def guide_page(request: Request, guide_id: int):
        g = store.guide(guide_id)
        if not g:
            raise _NotFound("That guide was not found.")
        course = store.course(g["course_id"])
        graph = concept_graph(g["data"]) if g["kind"] == "concept_map" else None
        smap = _source_map(enrich_sources(store, g.get("sources")))
        tree = bool(graph and not graph.get("empty"))
        return page(request, "guide.html", g=g, course=course, accent=course_color(course),
                    view=guide_view(g, store=store, tree_shown=tree),
                    graph=graph, smap=smap, kind_label=_mod("essay").guide_kind_label(g, GUIDE_KINDS.get(g["kind"], g["kind"])))

    @app.get("/guides/{guide_id}/delete", response_class=HTMLResponse)
    def guide_delete_confirm(request: Request, guide_id: int):
        g = store.guide(guide_id)
        if not g:
            raise _NotFound("That guide was not found.")
        return page(request, "confirm.html", title=f"Delete {g['title']}?", back=f"/guides/{guide_id}",
                    action=f"/guides/{guide_id}/delete", button="Delete the guide",
                    message="This removes the saved guide. Your materials are not touched.")

    @app.post("/guides/{guide_id}/delete")
    def guide_delete(guide_id: int):
        g = store.guide(guide_id)
        if not g:
            raise _NotFound("That guide was not found.")
        store.delete_guide(guide_id)
        return redirect(course_url(g["course_id"], "guides"), notice="Guide deleted.")

    # ------------------------------------------------------------------ flashcards
    def id_list(value: str) -> list[int]:
        return [int(x) for x in str(value).split(",") if x.strip().isdigit()][-200:]

    def cram_args(course_id: int, cram: str, material: str, done: str) -> tuple[bool, int | None, list[int]]:
        mat = int(material) if str(material).strip().isdigit() else None
        if mat is not None:
            m = store.material(mat)
            if not m or m["course_id"] != course_id:
                mat = None
        return str(cram) == "1", mat, id_list(done)

    def flashcard_url(course_id: int, cram: bool, material: int | None, done: list[int] | None = None,
                      missed: list[int] | None = None, only: list[int] | None = None) -> str:
        q = []
        if cram:
            q.append("cram=1")
        if material is not None:
            q.append(f"material={material}")
        if cram and done:
            q.append("done=" + ",".join(str(i) for i in done))
        if cram and missed:
            q.append("missed=" + ",".join(str(i) for i in missed))
        if cram and only:
            q.append("only=" + ",".join(str(i) for i in only))
        return f"/flashcards/{course_id}" + ("?" + "&".join(q) if q else "")

    def card_topic(c: dict[str, Any]) -> str:
        """What to restudy, for the weak-topic list: the source section heading when it has one (with the material in
        brackets), otherwise the material or deck."""
        ch = store.chunk(c["chunk_id"]) if c.get("chunk_id") else None
        if ch:
            head = re.sub(r"[#*_`]+", "", ch.get("heading") or "").strip()
            title = ch["material_title"]
            if head and head.lower() != title.lower():
                return f"{head} ({title})"
            return title
        return (c.get("deck") or "").strip() or "Typed or other cards"

    def card_source_href(c: dict[str, Any]) -> str:
        ch = store.chunk(c["chunk_id"]) if c.get("chunk_id") else None
        return f"/read/{ch['material_id']}?at={ch['id']}#c{ch['id']}" if ch else ""

    def next_exam_date(course_id: int) -> str:
        try:
            study = _mod("study")
            return study._exam_date_for(store, course_id)
        except (_Missing, AttributeError):
            return ""

    @app.get("/flashcards/{course_id}", response_class=HTMLResponse)
    def flashcards(request: Request, course_id: int, cram: str = "", material: str = "", done: str = "",
                   missed: str = "", only: str = ""):
        course = require_course(course_id)
        cram_on, mat, done_ids = cram_args(course_id, cram, material, done)
        missed_ids = [i for i in id_list(missed) if i in done_ids]
        only_ids = id_list(only)
        due = store.flashcards(course_id, due_only=True)
        total = len(store.flashcards(course_id))
        if cram_on:
            try:
                queue = _mod("study").cram_queue(store, course_id, material_id=mat)
            except _Missing:
                queue = []
            if only_ids:
                queue = [c for c in queue if c["id"] in only_ids]
            queue = [c for c in queue if c["id"] not in done_ids]
            shown, count, position = (queue[0] if queue else None), len(queue) + len(done_ids), len(done_ids) + 1
        else:
            if mat is not None:
                chunk_ids = {c["id"] for c in store.chunks_of(mat)}
                title = (store.material(mat) or {}).get("title", "")
                due = [c for c in due if c.get("chunk_id") in chunk_ids or (title and c.get("deck") == title)]
            shown, count, position = (due[0] if due else None), len(due), 1
        mats = [{"id": m["id"], "title": m["title"]} for m in store.materials(course_id)
                if not m.get("duplicate_of") and not m.get("graded")]
        material_total = None
        if mat is not None:
            chunk_ids = {c["id"] for c in store.chunks_of(mat)}
            mtitle = (store.material(mat) or {}).get("title", "")
            material_total = len([c for c in store.flashcards(course_id)
                                  if c.get("chunk_id") in chunk_ids or (mtitle and c.get("deck") == mtitle)])
        summary = None
        if cram_on and shown is None and done_ids:
            cards = {i: store.flashcard(i) for i in done_ids}
            wrong = [cards[i] for i in missed_ids if cards.get(i)]
            weak: dict[str, int] = {}
            for c in wrong:
                weak[card_topic(c)] = weak.get(card_topic(c), 0) + 1
            summary = {"total": len([c for c in cards.values() if c]), "wrong": len(wrong),
                       "right": len([c for c in cards.values() if c]) - len(wrong),
                       "weak": sorted(weak.items(), key=lambda kv: (-kv[1], kv[0])),
                       "redo_url": flashcard_url(course_id, True, mat, only=[c["id"] for c in wrong]) if wrong else "",
                       "again_url": flashcard_url(course_id, True, mat)}
        return page(request, "flashcards.html", course=course, accent=course_color(course), card=shown,
                    summary=summary, material_total=material_total, missed=",".join(str(i) for i in missed_ids), only=",".join(str(i) for i in only_ids),
                    source_href=card_source_href(shown) if shown else "",
                    due=len(due), total=total, cram=cram_on, material_id=mat, done=",".join(str(i) for i in done_ids),
                    count=count, position=position, filter_materials=mats,
                    toggle_url=flashcard_url(course_id, not cram_on, mat),
                    filter_base=flashcard_url(course_id, cram_on, None),
                    page_title=f"Flashcards - {course['name']}")

    @app.post("/flashcards/{course_id}/review")
    def flashcard_review(course_id: int, card_id: int = Form(...), grade: str = Form(...), cram: str = Form(""),
                         material: str = Form(""), done: str = Form(""), missed: str = Form(""), only: str = Form("")):
        require_course(course_id)
        cram_on, mat, done_ids = cram_args(course_id, cram, material, done)
        missed_ids, only_ids = id_list(missed), id_list(only)
        back = flashcard_url(course_id, cram_on, mat, done_ids, missed_ids, only_ids)
        card = store.flashcard(card_id)
        if not card or card["course_id"] != course_id or grade not in ("again", "hard", "good", "easy"):
            return redirect(back, error="That card could not be graded.")
        kw: dict[str, Any] = {}
        if cram_on:
            kw["cram"] = True
        exam = next_exam_date(course_id)
        if exam:
            try:
                left = (date.fromisoformat(exam) - date.today()).days
            except ValueError:
                left = None
            if left is not None and 0 <= left and (cram_on or left <= 7):
                kw["exam_date"] = exam
        try:
            _mod("study").review_flashcard(store, card_id, grade, **kw)
        except _Missing:
            return redirect(back, error="Flashcard review is not available yet.")
        if cram_on:
            done_ids.append(card_id)
            if grade in ("again", "hard"):
                missed_ids.append(card_id)
        return redirect(flashcard_url(course_id, cram_on, mat, done_ids, missed_ids, only_ids))

    # -- edit and delete a card (plain pages, no script needed)
    def card_or_404(card_id: int) -> dict[str, Any]:
        c = store.flashcard(card_id)
        if not c:
            raise _NotFound("That card was not found. It may already be deleted.")
        return c

    def card_form(request: Request, c: dict[str, Any], *, status_code: int = 200, **ctx: Any):
        course = require_course(c["course_id"])
        return page(request, "card_form.html", status_code=status_code, card=c, course=course,
                    accent=course_color(course), page_title="Edit flashcard", **ctx)

    @app.get("/flashcards/card/{card_id}/edit", response_class=HTMLResponse)
    def card_edit(request: Request, card_id: int):
        return card_form(request, card_or_404(card_id))

    @app.post("/flashcards/card/{card_id}/edit")
    def card_update(request: Request, card_id: int, front: str = Form(""), back: str = Form("")):
        c = card_or_404(card_id)
        shown = {**c, "front": front, "back": back}
        for field, value, label in (("front", front, "front"), ("back", back, "back")):
            if not value.strip():
                msg = f"The {label} of the card cannot be empty."
                return card_form(request, shown, status_code=400, error=msg,
                                 field_error={"form": "card", "field": field, "message": msg})
        store.update_flashcard(card_id, front=front.strip(), back=back.strip())
        return redirect(f"/flashcards/{c['course_id']}", notice="Card saved.")

    @app.get("/flashcards/card/{card_id}/delete", response_class=HTMLResponse)
    def card_delete_confirm(request: Request, card_id: int):
        c = card_or_404(card_id)
        front = c["front"][:120]
        if front.count("$") % 2:  # never cut a formula in half: drop the unfinished one
            front = front[:front.rfind("$")].rstrip()
        front = plain_math(front)  # served as plain text: no raw TeX, nothing left for a script to typeset
        return page(request, "confirm.html", title="Delete this flashcard?",
                    message=f"The card \"{front}\" will be deleted for good. Your materials are not touched.",
                    action=f"/flashcards/card/{card_id}/delete", button="Delete card",
                    back=f"/flashcards/{c['course_id']}", page_title="Delete flashcard")

    @app.post("/flashcards/card/{card_id}/delete")
    def card_delete(card_id: int):
        c = card_or_404(card_id)
        store.delete_flashcard(card_id)
        return redirect(f"/flashcards/{c['course_id']}", notice="Card deleted.")

    def safe_next(value: str, default: str = "/settings") -> str:
        """Only addresses inside this app (a path, never another site)."""
        return value if re.match(r"^/[A-Za-z0-9_\-./?=&,]*$", value or "") and not value.startswith("//") else default

    @app.post("/settings/shortcuts")
    def shortcuts_toggle(on: str = Form(""), next: str = Form("/settings")):
        store.set_setting("card_shortcuts", "1" if on == "1" else "0")
        note = ("Flashcard keyboard shortcuts are on: Space shows the answer, 1 to 4 grades it."
                if on == "1" else "Flashcard keyboard shortcuts are off. Use the buttons on the card.")
        return redirect(safe_next(next), notice=note)

    # ------------------------------------------------------------------ settings and setup
    def settings_ctx() -> dict[str, Any]:
        model = store.get_setting("anthropic_model", "") or MODELS[0][0]
        return {"ai_kind": (store.get_setting("ai_provider", "") or "none").lower(), "has_key": bool(store.get_setting("anthropic_api_key", "")),
                "model": model, "local_url": store.get_setting("local_base_url", ""), "local_model": store.get_setting("local_model", ""),
                "current_theme": store.get_setting("theme", "system"),
                "shortcuts_on": store.get_setting("card_shortcuts", "1") != "0"}

    @app.get("/settings", response_class=HTMLResponse)
    def settings(request: Request):
        return page(request, "settings.html", page_title="Settings", **settings_ctx())

    @app.post("/settings")
    def settings_save(request: Request, ai_provider: str = Form("none"), anthropic_api_key: str = Form(""),
                      clear_key: str = Form(""), anthropic_model: str = Form(""), local_base_url: str = Form(""),
                      local_model: str = Form(""), theme: str = Form("system")):
        if ai_provider not in ("none", "anthropic", "local"):
            ai_provider = "none"
        url = local_base_url.strip()
        if url and not re.match(r"^https?://\S+$", url):
            msg = "The local server address should start with http:// or https://, like http://127.0.0.1:1234."
            return page(request, "settings.html", status_code=400, page_title="Settings", **{**settings_ctx(), "local_url": url}, error=msg,
                        field_error={"form": "settings", "field": "local_base_url", "message": msg})
        if anthropic_api_key.strip():
            store.set_setting("anthropic_api_key", anthropic_api_key.strip())
        if clear_key:
            store.set_setting("anthropic_api_key", "")
        if anthropic_model in {m for m, _ in MODELS}:
            store.set_setting("anthropic_model", anthropic_model)
        store.set_setting("local_base_url", url)
        store.set_setting("local_model", local_model.strip())
        store.set_setting("theme", theme if theme in THEMES else "system")
        store.set_setting("ai_provider", ai_provider)
        if ai_provider == "anthropic" and not store.get_setting("anthropic_api_key", ""):
            return redirect("/settings", notice="Saved. Add your Anthropic API key to finish connecting.")
        return redirect("/settings", notice="Settings saved. Press Test connection to check that the AI answers.")

    async def connection_test() -> tuple[bool, str]:
        """Ask the saved AI one tiny question. Returns (worked, a plain sentence about what happened)."""
        kind = (store.get_setting("ai_provider", "") or "none").lower()
        try:
            p = provider()
        except Exception as e:
            return False, f"The AI settings could not be used: {_friendly(e)}"
        if p is None:
            if kind == "anthropic":
                return False, "Anthropic is selected but no API key is saved. Paste your key, save, then test again."
            return False, ("No AI is set up yet. Choose Anthropic API key or Local model server in Settings, save, then test "
                           "again. (Claude Desktop and Codex are connected from the Connect an AI page and need no test here.)")
        try:
            await work(p.complete, "You are a connection test.", [{"role": "user", "content": "Reply with the single word: ok"}], max_tokens=20)
        except Exception as e:
            return False, f"The test did not work. {_friendly(e)}"
        who = getattr(p, "name", "") or "AI"
        if who == "local":
            return True, f"Connected. Your local model at {getattr(p, 'base_url', 'your computer')} ({getattr(p, 'model', '')}) answered."
        return True, f"Connected. {who.capitalize()} ({getattr(p, 'model', '') or 'default model'}) answered."

    @app.post("/settings/test")
    async def settings_test(request: Request):
        ok, message = await connection_test()
        extra = {"notice": message} if ok else {"error": message}
        return page(request, "settings.html", status_code=200, page_title="Settings", **settings_ctx(), **extra)

    @app.get("/setup", response_class=HTMLResponse)
    def setup(request: Request, first: str = ""):
        return page(request, "setup.html", first=bool(first), status=connect_status(), **settings_ctx(),
                    page_title="Connect an AI")

    @app.post("/setup/skip")
    def setup_skip():
        store.set_setting("setup_done", now_iso())
        return redirect("/", notice="No problem. You can connect an AI any time from Connect an AI at the top of the page.")

    @app.post("/setup/anthropic")
    async def setup_anthropic(request: Request, anthropic_api_key: str = Form(""), anthropic_model: str = Form("")):
        key_ = anthropic_api_key.strip()
        if not key_ and not store.get_setting("anthropic_api_key", ""):
            msg = "Paste your API key first. It starts with sk-ant-."
            return page(request, "setup.html", status_code=400, first=False, status=connect_status(), **settings_ctx(),
                        page_title="Connect an AI", error=msg,
                        field_error={"form": "anthropic", "field": "anthropic_api_key", "message": msg})
        if key_:
            store.set_setting("anthropic_api_key", key_)
        if anthropic_model in {m for m, _ in MODELS}:
            store.set_setting("anthropic_model", anthropic_model)
        store.set_setting("ai_provider", "anthropic")
        ok, message = await connection_test()
        if ok:
            store.set_setting("setup_done", now_iso())
            return redirect("/", notice=message)
        return page(request, "setup.html", first=False, status=connect_status(), **settings_ctx(), page_title="Connect an AI",
                    error=f"Your key was saved, but {message[0].lower() + message[1:]}")

    @app.post("/setup/local")
    async def setup_local(request: Request, local_base_url: str = Form(""), local_model: str = Form("")):
        url = local_base_url.strip() or "http://127.0.0.1:1234"
        if not re.match(r"^https?://\S+$", url):
            msg = "The address should start with http:// or https://, like http://127.0.0.1:1234."
            return page(request, "setup.html", status_code=400, first=False, status=connect_status(),
                        **{**settings_ctx(), "local_url": url}, page_title="Connect an AI", error=msg,
                        field_error={"form": "local", "field": "local_base_url", "message": msg})
        store.set_setting("local_base_url", url)
        store.set_setting("local_model", local_model.strip())
        store.set_setting("ai_provider", "local")
        ok, message = await connection_test()
        if ok:
            store.set_setting("setup_done", now_iso())
            return redirect("/", notice=message)
        return page(request, "setup.html", first=False, status=connect_status(), **settings_ctx(), page_title="Connect an AI",
                    error=f"Your settings were saved, but {message[0].lower() + message[1:]}")

    # ------------------------------------------------------------------ connect
    def connect_status() -> dict[str, Any]:
        out: dict[str, Any] = {"claude": False, "codex": False, "snippet": "", "available": True}
        try:
            c = _mod("connect")
            out["claude"] = bool(c.claude_connected())
            out["codex"] = bool(c.codex_connected())
            out["snippet"] = c.codex_snippet()
        except Exception:
            out["available"] = False
        try:  # what pressing each button would write outside the app folder (shown before the button)
            for t in ("claude", "codex"):
                out["plan_" + t] = _mod("connect").write_plan(t)
        except Exception:
            pass
        return out

    @app.get("/connect", response_class=HTMLResponse)
    def connect_page(request: Request):
        return page(request, "connect.html", status=connect_status(), results=[], page_title="Claude Desktop or Codex")

    @app.post("/connect/{target}", response_class=HTMLResponse)
    async def connect_run(request: Request, target: str):
        if target not in ("claude", "codex"):
            raise _NotFound("That app was not found.")
        try:
            c = _mod("connect")
            raw = await work(c.connect, target)
            lines = [line for r in raw for line in c.describe(r)]
            ok = all(r.get("ok", True) for r in raw if r.get("app") != "Skills")
        except Exception as e:
            lines, ok = [f"Could not connect: {_friendly(e)}"], False
        return page(request, "connect.html", status=connect_status(), results=lines, results_ok=ok,
                    page_title="Claude Desktop or Codex")

    return app
