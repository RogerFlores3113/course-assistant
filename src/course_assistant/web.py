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

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import __version__, disclaimer, paths
from .store import EVENT_KINDS, GUIDE_KINDS, MATERIAL_KINDS, Store, now_iso
from .web_md import render as render_md
from .web_helpers import (chunk_range_label, DEPTHS, MODELS, PALETTE, TABS, THEMES, answer_view, clean_color, concept_graph,
                          course_color, day_label, guide_view, parse_day, time_label, week_label, week_start)

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
MEDIA_EXT = {".mp3": "audio", ".m4a": "audio", ".wav": "audio", ".mp4": "video", ".mov": "video", ".webm": "video"}
INLINE_TYPES = ("application/pdf", "audio/", "video/", "image/png", "image/jpeg", "image/gif", "image/webp", "text/plain")
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


def _friendly(e: Exception) -> str:
    if isinstance(e, _Missing):
        return "That part of the app is not installed yet."
    text = str(e).strip() or type(e).__name__
    return text if len(text) <= 400 else text[:397] + "..."


def create_app(store: Store, *, key: str | None = None, provider_factory=None) -> FastAPI:
    key = key or paths.app_key()
    res = paths.resource_dir()
    app = FastAPI(title="Course Assistant", docs_url=None, redoc_url=None, openapi_url=None)
    templates = Jinja2Templates(directory=str(res / "templates"))
    env = templates.env
    env.globals.update(version=__version__, disclaimer_full=disclaimer.FULL, tabs=TABS, depths=DEPTHS, models=MODELS,
                       palette=PALETTE, material_kinds=MATERIAL_KINDS, event_kinds=EVENT_KINDS,
                       guide_kinds=GUIDE_KINDS, course_color=course_color)
    env.globals["theme"] = lambda: (store.get_setting("theme", "system") if store.get_setting("theme", "system") in THEMES else "system")
    env.filters["day"] = day_label
    env.filters["clock"] = time_label
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
        resp = templates.TemplateResponse(request, name, ctx, status_code=status_code)
        if raw:
            resp.delete_cookie(FLASH, path="/")
        return resp

    def redirect(url: str, notice: str = "", error: str = "") -> RedirectResponse:
        resp = RedirectResponse(url, status_code=303)
        if notice or error:
            resp.set_cookie(FLASH, quote(json.dumps({"notice": notice[:500], "error": error[:500]})), max_age=120,
                            httponly=True, samesite="strict", path="/")
        return resp

    def require_course(course_id: int) -> dict[str, Any]:
        c = store.course(course_id)
        if not c:
            raise _NotFound("That course was not found.")
        return c

    class _NotFound(Exception):
        pass

    @app.exception_handler(_NotFound)
    async def not_found(request: Request, exc: _NotFound):
        return templates.TemplateResponse(request, "message.html", {"title": "Not found", "message": str(exc)}, status_code=404)

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

    def course_card(c: dict[str, Any]) -> dict[str, Any]:
        cid = c["id"]
        today = date.today().isoformat()
        nxt = store.events(cid, today)[:1]
        return {**c, "accent": course_color(c), "next_event": nxt[0] if nxt else None,
                "event_count": len(store.events(cid)), "guide_count": len(store.guides(cid)),
                "due_cards": len(store.flashcards(cid, due_only=True))}

    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request):
        courses = [course_card(c) for c in store.courses()]
        return page(request, "dashboard.html", courses=courses, upcoming=upcoming_rows(), has_ai=provider_state()["connected"])

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
            return page(request, "course_form.html", status_code=400, course=vals, editing=False,
                        error="Give the course a name, like Linear Algebra.")
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
            return page(request, "course_form.html", status_code=400, course={**c, **vals}, editing=True,
                        error="The course needs a name.")
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
    def material_rows(cid: int) -> list[dict[str, Any]]:
        jobs: dict[int, dict[str, Any]] = {}
        for j in store._rows("SELECT * FROM jobs WHERE material_id IN (SELECT id FROM materials WHERE course_id = ?) ORDER BY id", (cid,)):
            jobs[int(j["material_id"])] = j
        rows = []
        for m in store.materials(cid):
            j = jobs.get(m["id"])
            job_text = ""
            if j:
                pct = int(float(j.get("progress") or 0) * 100)
                job_text = {"queued": "Waiting in line for transcription", "running": f"Transcribing, {pct}% done",
                            "done": "Transcription finished", "error": "Transcription failed"}.get(j["status"], j["status"])
                if j.get("note") and j["status"] == "error":
                    job_text += f": {j['note']}"
            rows.append({**m, "kind_label": MATERIAL_KINDS.get(m["kind"], m["kind"]), "job_text": job_text,
                         "is_media": Path(m.get("filename") or "").suffix.lower() in MEDIA_EXT})
        return rows

    def tab_context(cid: int, tab: str, extra: dict[str, Any]) -> dict[str, Any]:
        ctx: dict[str, Any] = {}
        today = date.today()
        if tab == "materials":
            ctx["materials"] = material_rows(cid)
            ctx["supported"] = _supported_formats()
        elif tab == "calendar":
            events = store.events(cid)
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
            ctx["events"] = [e for e in store.events(cid, today.isoformat()) if e["kind"] in ("lecture", "exam", "quiz")][:30]
            ctx["past_lectures"] = [e for e in store.events(cid) if e["kind"] == "lecture" and e["date"] <= today.isoformat()][-15:]
            ctx["has_materials"] = bool(store.materials(cid))
            ctx["due_cards"] = len(store.flashcards(cid, due_only=True))
            ctx["all_cards"] = len(store.flashcards(cid))
        elif tab == "guides":
            ctx["guides"] = store.guides(cid)
        elif tab == "ask":
            history = []
            for m in store.chat_messages(cid, 40):
                item: dict[str, Any] = {"role": m["role"], "content": m["content"]}
                if m["role"] == "assistant":
                    g = store.guide(m["guide_id"]) if m.get("guide_id") else None
                    saved = ((g or {}).get("data") or {}).get("blocks")
                    item["view"] = answer_view(m["content"], m.get("sources"),
                                               blocks=saved if isinstance(saved, list) else None,
                                               twin=bool(g and g["kind"] == "twin"),
                                               mode=((g or {}).get("data") or {}).get("mode"),
                                               data=(g or {}).get("data"))
                history.append(item)
            ctx["history"] = history
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

    def course_page(request: Request, cid: int, tab: str, status_code: int = 200, notice: str = "", error: str = "", **extra):
        course = require_course(cid)
        ctx = {"course": course, "accent": course_color(course), "tab": tab, "ai": provider_state(),
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

    def after(request: Request, cid: int, tab: str, notice: str = "", error: str = "", **extra):
        """Answer a form: htmx swaps the tab in place; a plain post goes back to the tab."""
        if is_htmx(request):
            return course_page(request, cid, tab, 400 if error and not notice else 200, notice, error, **extra)
        return redirect(course_url(cid, tab), notice=notice, error=error)

    @app.get("/courses/{course_id}", response_class=HTMLResponse)
    def course_materials(request: Request, course_id: int):
        return course_page(request, course_id, "materials")

    @app.get("/courses/{course_id}/{tab}", response_class=HTMLResponse)
    def course_tab(request: Request, course_id: int, tab: str):
        if tab not in {t for t, _ in TABS}:
            raise _NotFound("That page was not found.")
        return course_page(request, course_id, tab)

    # ------------------------------------------------------------------ materials
    @app.post("/courses/{course_id}/materials/upload")
    async def upload_materials(request: Request, course_id: int, files: list[UploadFile] = File(default=[]),
                               kind: str = Form(""), lecture_date: str = Form("")):
        require_course(course_id)
        files = [f for f in files if f.filename]
        if not files:
            return after(request, course_id, "materials", error="Choose at least one file to add.")
        if lecture_date and not parse_day(lecture_date):
            return after(request, course_id, "materials", error="The lecture date should look like 2026-09-14.")
        kind = kind if kind in MATERIAL_KINDS else ""

        def run() -> tuple[list[str], list[str]]:
            importers = _mod("importers")
            bad = getattr(importers, "ImportError_", ())
            done, errors = [], []
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
                    importers.import_file(store, course_id, tmp, kind=kind or None, original_name=name,
                                          lecture_date=lecture_date)
                    done.append(name)
                except Exception as e:
                    shown = _friendly(e) if (isinstance(e, ValueError) or (bad and isinstance(e, bad))) else f"Could not read it ({_friendly(e)})"
                    errors.append(f"{name}: {shown}")
                finally:
                    shutil.rmtree(tmpdir, ignore_errors=True)
            return done, errors

        try:
            done, errors = await work(run)
        except _Missing:
            return after(request, course_id, "materials", error="The importer is not available yet.")
        notice = f"Added {len(done)} file{'s' if len(done) != 1 else ''}: {', '.join(done)}." if done else ""
        return after(request, course_id, "materials", notice=notice, error=" ".join(errors))

    @app.post("/courses/{course_id}/materials/url")
    async def add_url(request: Request, course_id: int, url: str = Form(""), kind: str = Form("website")):
        require_course(course_id)
        url = url.strip()
        if not re.match(r"^https?://\S+$", url):
            return after(request, course_id, "materials", error="Paste a web address that starts with http:// or https://.")
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
            return after(request, course_id, "materials", error="Type or paste some text for the note.")
        kind = kind if kind in MATERIAL_KINDS else "notes"
        try:
            await work(lambda: _mod("importers").import_text(store, course_id, title.strip() or "My note", text, kind=kind))
        except Exception as e:
            return after(request, course_id, "materials", error=f"Could not save the note: {_friendly(e)}")
        return after(request, course_id, "materials", notice="Note saved.")

    @app.post("/materials/{material_id}/graded")
    def toggle_graded(request: Request, material_id: int, graded: str = Form("1")):
        m = store.material(material_id)
        if not m:
            raise _NotFound("That material was not found.")
        on = graded == "1"
        store.update_material(material_id, graded=1 if on else 0)
        note = (f"Marked {m['title']} as graded work. The app will explain and write twin problems, never solve it."
                if on else f"{m['title']} is no longer marked as graded.")
        return after(request, m["course_id"], "materials", notice=note)

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

    @app.get("/materials/{material_id}/delete", response_class=HTMLResponse)
    def material_delete_confirm(request: Request, material_id: int):
        m = store.material(material_id)
        if not m:
            raise _NotFound("That material was not found.")
        return page(request, "confirm.html", title=f"Delete {m['title']}?", back=course_url(m["course_id"]),
                    action=f"/materials/{material_id}/delete", button="Delete it",
                    message="This removes the material, its searchable text and the saved original file from this "
                            "computer. Guides you already made stay. It cannot be undone.")

    @app.post("/materials/{material_id}/delete")
    def material_delete(request: Request, material_id: int):
        m = store.material(material_id)
        if not m:
            raise _NotFound("That material was not found.")
        f = file_of(m)
        store.delete_material(material_id)
        if f and not store._row("SELECT id FROM materials WHERE filename = ?", (m["filename"],)):
            f.unlink(missing_ok=True)
        return redirect(course_url(m["course_id"]), notice=f"Deleted {m['title']}.")

    # ------------------------------------------------------------------ calendar
    def event_fields(kind: str, title: str, day: str, time_: str, topics: str) -> tuple[dict[str, str], str]:
        vals = {"kind": kind if kind in EVENT_KINDS else "other", "title": title.strip(), "date": day.strip(),
                "time": time_.strip(), "topics": topics.strip()}
        err = ""
        if not vals["title"]:
            err = "Give the event a title."
        elif not parse_day(vals["date"]):
            err = "The date should look like 2026-10-21."
        elif vals["time"] and not re.match(r"^\d{1,2}:\d{2}$", vals["time"]):
            err = "The time should look like 14:30."
        return vals, err

    @app.post("/courses/{course_id}/events")
    def event_add(request: Request, course_id: int, kind: str = Form("lecture"), title: str = Form(""),
                  date_: str = Form("", alias="date"), time_: str = Form("", alias="time"), topics: str = Form("")):
        require_course(course_id)
        vals, err = event_fields(kind, title, date_, time_, topics)
        if err:
            return after(request, course_id, "calendar", error=err)
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
        vals, err = event_fields(kind, title, date_, time_, topics)
        if err:
            return page(request, "event_form.html", status_code=400, event={**e, **vals}, course=require_course(e["course_id"]), error=err)
        store.update_event(event_id, **vals)
        return redirect(course_url(e["course_id"], "calendar"), notice="Event saved.")

    @app.get("/events/{event_id}/delete", response_class=HTMLResponse)
    def event_delete_confirm(request: Request, event_id: int):
        e = store.event(event_id)
        if not e:
            raise _NotFound("That event was not found.")
        return page(request, "confirm.html", title=f"Delete {e['title']}?", back=course_url(e["course_id"], "calendar"),
                    action=f"/events/{event_id}/delete", button="Delete the event",
                    message="This removes the event from the course calendar.")

    @app.post("/events/{event_id}/delete")
    def event_delete(event_id: int):
        e = store.event(event_id)
        if not e:
            raise _NotFound("That event was not found.")
        store.delete_event(event_id)
        return redirect(course_url(e["course_id"], "calendar"), notice="Event deleted.")

    # ------------------------------------------------------------------ study tools
    STUDY_LABELS = {"primer": "Pre-lecture primer", "exam": "Exam guide", "map": "Concept map", "explain": "Explanation",
                    "practice": "Practice questions", "flashcards": "Flashcards", "missed": "What did I miss"}

    @app.post("/courses/{course_id}/study/{action}")
    async def study_run(request: Request, course_id: int, action: str, topic: str = Form(""), depth: str = Form("standard"),
                        count: int = Form(0), event_id: str = Form(""), day: str = Form("")):
        require_course(course_id)
        if action not in STUDY_LABELS:
            raise _NotFound("That study tool was not found.")
        p = provider()
        if p is None:
            return after(request, course_id, "study", error="Connect an AI first. See the panel below.")
        ev = int(event_id) if event_id.strip().isdigit() else None
        depth = depth if depth in {d for d, _ in DEPTHS} else "standard"
        topic = topic.strip()
        if action in ("explain",) and not topic:
            return after(request, course_id, "study", error="Type the topic you want explained.")
        if action == "practice" and not topic:
            return after(request, course_id, "study", error="Type the topic you want practice questions on.")

        def run():
            study = _mod("study")
            if action == "primer":
                return study.primer(store, course_id, p, event_id=ev)
            if action == "exam":
                return study.exam_guide(store, course_id, p, event_id=ev)
            if action == "map":
                return study.concept_map(store, course_id, p, scope=topic)
            if action == "explain":
                return study.explain(store, course_id, p, topic, depth=depth)
            if action == "practice":
                return study.practice(store, course_id, p, topic, count=count or 5)
            if action == "flashcards":
                return study.make_flashcards(store, course_id, p, topic=topic, count=count or 12)
            return study.missed(store, course_id, p, event_id=ev, date=day.strip())

        try:
            result = await work(run)
        except Exception as e:
            if _is_no_provider(e):
                return after(request, course_id, "study", error="Connect an AI first. See the panel below.")
            return after(request, course_id, "study", error=f"{STUDY_LABELS[action]} did not work: {_friendly(e)}")
        if action == "flashcards":
            n = len(result or [])
            return after(request, course_id, "study", notice=f"Made {n} flashcard{'s' if n != 1 else ''}.", cards_made=n)
        guide = store.guide(int(result))
        if not is_htmx(request):
            return redirect(f"/guides/{int(result)}", notice=f"{STUDY_LABELS[action]} is ready.")
        return after(request, course_id, "study", notice=f"{STUDY_LABELS[action]} is ready.", result_guide=guide,
                     result_view=guide_view(guide) if guide else None)

    # ------------------------------------------------------------------ ask
    @app.post("/courses/{course_id}/ask")
    async def ask(request: Request, course_id: int, question: str = Form(""), depth: str = Form("standard")):
        require_course(course_id)
        q = question.strip()
        if not q:
            return after(request, course_id, "ask", error="Type a question first.")
        p = provider()
        if p is None:
            return after(request, course_id, "ask", error="Connect an AI first. See the panel below.", draft=q)
        depth = depth if depth in {d for d, _ in DEPTHS} else "standard"
        history = [{"role": m["role"], "content": m["content"]} for m in store.chat_messages(course_id, 6)]

        try:
            await work(lambda: _mod("grounding").ask(store, course_id, q, p, depth=depth, history=history))
        except Exception as e:
            if _is_no_provider(e):
                return after(request, course_id, "ask", error="Connect an AI first. See the panel below.", draft=q)
            return after(request, course_id, "ask", error=f"The answer did not work: {_friendly(e)}", draft=q)
        return after(request, course_id, "ask")

    # ------------------------------------------------------------------ past offerings
    @app.post("/courses/{course_id}/offerings/paste")
    async def offerings_paste(request: Request, course_id: int, links: str = Form(""), term_label: str = Form("")):
        require_course(course_id)
        if not re.search(r"https?://\S+", links):
            return after(request, course_id, "offerings", error="Paste one or more web addresses that start with http:// or https://.")
        try:
            found = await work(lambda: _mod("offerings").find_past_offerings(store, course_id, links))
        except Exception as e:
            return after(request, course_id, "offerings", error=f"Those links did not work: {_friendly(e)}")
        label = term_label.strip()
        if label:
            for c in found:
                if not c.get("term_label"):
                    store.update_offering_candidate(c["id"], term_label=label)
        n = len(found)
        return after(request, course_id, "offerings",
                     notice=f"Added {n} link{'s' if n != 1 else ''} to the list below. Tick the ones to save.")

    @app.post("/courses/{course_id}/offerings/search")
    async def offerings_search(request: Request, course_id: int):
        course = require_course(course_id)
        try:
            found = await work(lambda: _mod("offerings").find_past_offerings(store, course_id, "", found_by="crawl"))
        except Exception as e:
            return after(request, course_id, "offerings", error=f"The search did not work: {_friendly(e)}")
        if not found:
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
                    msg = ("This site's robots.txt asks programs not to read it, so nothing was saved." if blocked
                           else _friendly(e))
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
    @app.get("/read/{material_id}", response_class=HTMLResponse)
    def reader(request: Request, material_id: int, at: str | None = None):
        m = store.material(material_id)
        if not m:
            raise _NotFound("That material was not found.")
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
        for c in chunks:
            c["seek"] = int(float(c["loc_start"])) if c["loc_kind"] == "time" and re.match(r"^\d+(\.\d+)?$", str(c["loc_start"])) else None
            c["range_label"] = chunk_range_label(c)
            c["html"] = render_md(c["text"], heading_offset=2)
        source = m["source_url"] if re.match(r"^https?://\S+$", m.get("source_url") or "") else ""
        return page(request, "read.html", m=m, course=course, accent=course_color(course), chunks=chunks, at=at_id,
                    at_note=at_note, media=media, has_file=bool(file_of(m)), source_url=source,
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
        smap = {s.get("n", i): s for i, s in enumerate(g.get("sources") or [], start=1)}
        return page(request, "guide.html", g=g, course=course, accent=course_color(course), view=guide_view(g),
                    graph=graph, smap=smap, kind_label=GUIDE_KINDS.get(g["kind"], g["kind"]))

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
    @app.get("/flashcards/{course_id}", response_class=HTMLResponse)
    def flashcards(request: Request, course_id: int):
        course = require_course(course_id)
        due = store.flashcards(course_id, due_only=True)
        return page(request, "flashcards.html", course=course, accent=course_color(course), card=due[0] if due else None,
                    due=len(due), total=len(store.flashcards(course_id)))

    @app.post("/flashcards/{course_id}/review")
    def flashcard_review(course_id: int, card_id: int = Form(...), grade: str = Form(...)):
        require_course(course_id)
        card = store.flashcard(card_id)
        if not card or card["course_id"] != course_id or grade not in ("again", "hard", "good", "easy"):
            return redirect(f"/flashcards/{course_id}", error="That card could not be graded.")
        try:
            _mod("study").review_flashcard(store, card_id, grade)
        except _Missing:
            return redirect(f"/flashcards/{course_id}", error="Flashcard review is not available yet.")
        return redirect(f"/flashcards/{course_id}")

    # ------------------------------------------------------------------ settings
    def settings_ctx() -> dict[str, Any]:
        model = store.get_setting("anthropic_model", "") or MODELS[0][0]
        return {"ai_kind": (store.get_setting("ai_provider", "") or "none").lower(), "has_key": bool(store.get_setting("anthropic_api_key", "")),
                "model": model, "local_url": store.get_setting("local_base_url", ""), "local_model": store.get_setting("local_model", ""),
                "current_theme": store.get_setting("theme", "system")}

    @app.get("/settings", response_class=HTMLResponse)
    def settings(request: Request):
        return page(request, "settings.html", **settings_ctx())

    @app.post("/settings")
    def settings_save(request: Request, ai_provider: str = Form("none"), anthropic_api_key: str = Form(""),
                      clear_key: str = Form(""), anthropic_model: str = Form(""), local_base_url: str = Form(""),
                      local_model: str = Form(""), theme: str = Form("system")):
        if ai_provider not in ("none", "anthropic", "local"):
            ai_provider = "none"
        if anthropic_api_key.strip():
            store.set_setting("anthropic_api_key", anthropic_api_key.strip())
        if clear_key:
            store.set_setting("anthropic_api_key", "")
        if anthropic_model in {m for m, _ in MODELS}:
            store.set_setting("anthropic_model", anthropic_model)
        url = local_base_url.strip()
        if url and not re.match(r"^https?://\S+$", url):
            return page(request, "settings.html", status_code=400, **{**settings_ctx(), "local_url": url},
                        error="The local server address should start with http:// or https://, like http://127.0.0.1:1234.")
        store.set_setting("local_base_url", url)
        store.set_setting("local_model", local_model.strip())
        store.set_setting("theme", theme if theme in THEMES else "system")
        store.set_setting("ai_provider", ai_provider)
        if ai_provider == "anthropic" and not store.get_setting("anthropic_api_key", ""):
            return redirect("/settings", notice="Saved. Add your Anthropic API key to finish connecting.")
        return redirect("/settings", notice="Settings saved.")

    @app.post("/settings/test")
    async def settings_test(request: Request):
        try:
            p = provider()
        except Exception as e:
            p, err = None, _friendly(e)
        else:
            err = ""
        if p is None:
            return page(request, "settings.html", status_code=200, **settings_ctx(),
                        error=err or "No AI is set up yet. Choose a provider, save, then test again.")
        try:
            reply = await work(p.complete, "You are a connection test.", [{"role": "user", "content": "Reply with the single word: ok"}], max_tokens=20)
        except Exception as e:
            return page(request, "settings.html", **settings_ctx(), error=f"The test did not work: {_friendly(e)}")
        return page(request, "settings.html", **settings_ctx(),
                    notice=f"Connected. {getattr(p, 'name', 'The AI')} ({getattr(p, 'model', '') or 'default model'}) answered.")

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
        return out

    @app.get("/connect", response_class=HTMLResponse)
    def connect_page(request: Request):
        return page(request, "connect.html", status=connect_status(), results=[])

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
        return page(request, "connect.html", status=connect_status(), results=lines, results_ok=ok)

    return app
