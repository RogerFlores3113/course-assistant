"""Command line entry point.

Double-clicking the packaged app runs with no arguments: start the local server and open the browser. AI apps
launch it with `mcp` to talk over stdio.

    course-assistant [serve] [--port N] [--no-browser]
    course-assistant open
    course-assistant mcp
    course-assistant connect [claude|codex]
    course-assistant import <course name or id> <files...> [--create]
    course-assistant version

web.py and mcp_server.py are imported inside the commands that need them, so this module imports cheaply and
`version`, `connect` and `import` work without starting a server.
"""
from __future__ import annotations

import argparse
import hmac
import logging
import os
import secrets
import socket
import sys
import threading
import webbrowser
from pathlib import Path

import httpx

from . import __version__, paths
from .store import MATERIAL_KINDS

COMMANDS = ("serve", "open", "mcp", "connect", "import", "version")


# ------------------------------------------------------------------ server helpers
def _private(path: Path) -> Path:
    """Create the file readable by this user only (0600). Not on Windows."""
    if os.name != "nt":
        try:
            os.close(os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600))
            os.chmod(path, 0o600)
        except OSError:
            pass
    return path


def _write_port_file(port: int) -> None:
    try:  # so the AI app's links point at the right port
        _private(paths.port_file()).write_text(str(port))
    except OSError:
        pass


def _recorded_port() -> int | None:
    try:
        return int(paths.port_file().read_text().strip())
    except (OSError, ValueError):
        return None


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def _pick_port(preferred: int) -> int:
    if _port_free(preferred):
        return preferred
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _answers(port: int) -> bool:
    """This install's app answers on the port: /api/health returns a proof (HMAC with the app key, bound to the
    port and a fresh nonce), so another program on the same port is never taken for the app."""
    from .web import health_proof

    nonce = secrets.token_urlsafe(24)
    try:
        # trust_env=False: this is this computer talking to itself, never through HTTP(S)_PROXY
        r = httpx.get(f"http://127.0.0.1:{port}/api/health", params={"nonce": nonce}, timeout=0.7,
                      follow_redirects=False, trust_env=False)
        body = r.json()
    except (httpx.HTTPError, ValueError):
        return False
    proof = body.get("proof") if isinstance(body, dict) else None
    return (r.status_code == 200 and isinstance(proof, str)
            and hmac.compare_digest(proof, health_proof(paths.app_key(), nonce, port)))


def find_running() -> int | None:
    """The port the running app answers on (the recorded one), or None. A dead recorded port is dropped."""
    port = _recorded_port()
    if port is None:
        return None
    if _answers(port):
        return port
    try:
        paths.port_file().unlink()
    except OSError:
        pass
    return None


def launch_url(port: int) -> str:
    """The address to open in the browser: a one-time launch code, never the app key itself."""
    return f"http://127.0.0.1:{port}/launch?code={paths.new_launch_code()}"


def open_window() -> bool:
    """Open (and unlock) the window of the running app. False when it isn't running."""
    port = find_running()
    if port is None:
        return False
    webbrowser.open(launch_url(port))
    return True


def _setup_logging() -> None:
    handlers: list[logging.Handler] = [logging.FileHandler(_private(paths.data_dir() / "app.log"), encoding="utf-8")]
    if sys.stderr is not None:
        handlers.append(logging.StreamHandler(sys.stderr))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", handlers=handlers)


def run_app(port: int = paths.DEFAULT_PORT, open_browser: bool = True) -> None:
    """Start the server on 127.0.0.1, or just open the window if this app is already running."""
    log = logging.getLogger(__name__)
    running = find_running()
    if running is not None:
        if open_browser:
            webbrowser.open(launch_url(running))
        log.info("Course Assistant is already running at http://127.0.0.1:%s/. %s", running,
                 "Opened its window." if open_browser else "Run course-assistant open to open its window.")
        return

    import uvicorn

    from .store import Store
    from .web import create_app

    wanted = port
    port = _pick_port(port)
    if port != wanted:
        log.warning("Port %s is in use by another program, starting on port %s instead.", wanted, port)
    # start_jobs=True: the app itself starts the background worker that reads photos and transcribes recordings
    app = create_app(Store(paths.db_path()), start_jobs=True)
    _write_port_file(port)
    if open_browser:
        threading.Timer(1.2, lambda: webbrowser.open(launch_url(port))).start()
    log.info("Course Assistant running at http://127.0.0.1:%s/ (data: %s)", port, paths.data_dir())
    # log_config=None: the frozen windowed build has no console for uvicorn's default handlers.
    try:
        uvicorn.run(app, host="127.0.0.1", port=port, log_config=None, access_log=False)
    finally:  # stopped: the port file no longer says where the app is
        if _recorded_port() == port:
            try:
                paths.port_file().unlink()
            except OSError:
                pass


# ------------------------------------------------------------------ commands
def cmd_serve(args: argparse.Namespace) -> int:
    _setup_logging()
    run_app(port=args.port, open_browser=not args.no_browser)
    return 0


def cmd_open(args: argparse.Namespace) -> int:
    if open_window():
        print("Opened Course Assistant in your browser.")
        return 0
    print("Course Assistant isn't running. Start it with: course-assistant", file=sys.stderr)
    return 1


def cmd_mcp(args: argparse.Namespace) -> int:
    from . import mcp_server

    mcp_server.main()
    return 0


def cmd_connect(args: argparse.Namespace) -> int:
    from . import connect

    plan = connect.write_plan(args.target, with_skills=not args.no_skills)
    if plan["writes"]:
        print("Connecting will change files outside Course Assistant's own folder:")
        for line in connect.plan_lines(plan):
            print("  " + line)
        if not args.yes:
            try:
                answer = input("Go ahead? [y/N] ").strip().lower()
            except (EOFError, OSError):
                answer = ""
            if answer not in ("y", "yes"):
                print("Nothing was changed. Run again with --yes to skip this question.")
                return 1
    else:
        for line in connect.plan_lines(plan):
            print(line)
    try:
        results = connect.connect(args.target, with_skills=not args.no_skills)
    except (ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    failed = False
    for result in results:
        for line in connect.describe(result):
            print(line)
        if result["app"] != "Skills" and not result.get("ok"):
            failed = True
    return 1 if failed else 0


def resolve_course(store, ref: str, create: bool = False) -> dict | None:
    """A course by id, exact name (any case), or a name that contains the text when only one does."""
    ref = ref.strip()
    courses = store.courses(include_archived=True)
    if ref.isdigit() and (found := store.course(int(ref))):
        return found
    lowered = ref.lower()
    exact = [c for c in courses if c["name"].lower() == lowered or (c.get("code") or "").lower() == lowered]
    if len(exact) == 1:
        return exact[0]
    partial = [c for c in courses if lowered in c["name"].lower()]
    if not exact and len(partial) == 1:
        return partial[0]
    if exact or partial:
        names = ", ".join(c["name"] for c in (exact or partial))
        raise ValueError(f"More than one course matches '{ref}': {names}. Use the course id instead.")
    if create:
        return store.course(store.add_course(ref))
    return None


def cmd_import(args: argparse.Namespace) -> int:
    from .importers import ImportError_, import_file
    from .store import Store

    store = Store(paths.db_path())
    try:
        course = resolve_course(store, args.course, create=args.create)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if course is None:
        print(f"No course matches '{args.course}'. Add --create to make it, or pick one of: "
              + (", ".join(c["name"] for c in store.courses()) or "(no courses yet)"), file=sys.stderr)
        return 1
    bad = 0
    known = {m["id"] for m in store.materials(course["id"])}
    for name in args.files:
        src = Path(name).expanduser()
        if not src.is_file():
            print(f"Skipped {name}: not a file.", file=sys.stderr)
            bad += 1
            continue
        try:
            material_id = import_file(store, course["id"], src, kind=args.kind, lecture_date=args.lecture_date or "")
        except ImportError_ as exc:
            print(f"Could not import {src.name}: {exc}", file=sys.stderr)
            bad += 1
            continue
        if material_id in known:
            print(f"Already saved: {src.name} is already in {course['name']} (material {material_id}). Nothing was added.")
        else:
            print(f"Saved {src.name} to {course['name']} (material {material_id}).")
            known.add(material_id)
    return 1 if bad else 0


def cmd_version(args: argparse.Namespace) -> int:
    print(f"Course Assistant {__version__}")
    return 0


# ------------------------------------------------------------------ parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="course-assistant", description="Local-first study help from your own course materials.")
    sub = parser.add_subparsers(dest="command")

    serve = sub.add_parser("serve", help="start the app and open it in your browser (the default)")
    serve.add_argument("--port", type=int, default=paths.DEFAULT_PORT,
                       help="port to listen on (default: %(default)s; a free one is used if it is taken)")
    serve.add_argument("--no-browser", action="store_true", help="do not open the browser")
    serve.set_defaults(func=cmd_serve)

    sub.add_parser("open", help="open the window of the running app").set_defaults(func=cmd_open)
    sub.add_parser("mcp", help="run the MCP server over stdio (AI apps start this)").set_defaults(func=cmd_mcp)

    con = sub.add_parser("connect", help="add Course Assistant to Claude Desktop and/or Codex, and install the skills")
    con.add_argument("target", nargs="?", choices=["claude", "codex", "all"], default="all")
    con.add_argument("--no-skills", action="store_true", help="only write the MCP settings")
    con.add_argument("--yes", "-y", action="store_true", help="don't ask before writing outside the app folder")
    con.set_defaults(func=cmd_connect)

    imp = sub.add_parser("import", help="save files into a course")
    imp.add_argument("course", help="course name or id")
    imp.add_argument("files", nargs="+")
    imp.add_argument("--create", action="store_true", help="make the course if it doesn't exist")
    imp.add_argument("--kind", default=None, choices=sorted(MATERIAL_KINDS), metavar="KIND",
                     help="what the files are: " + ", ".join(sorted(MATERIAL_KINDS)) + " (guessed when left out)")
    imp.add_argument("--lecture-date", default="", help="YYYY-MM-DD, for lecture slides or recordings")
    imp.set_defaults(func=cmd_import)

    sub.add_parser("version", help="print the version").set_defaults(func=cmd_version)
    return parser


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    # No command (a double-click), or only serve options: serve.
    if not argv or (argv[0] not in COMMANDS and argv[0] not in ("-h", "--help")):
        argv = ["serve", *argv]
    args = build_parser().parse_args(argv)
    code = args.func(args)
    if code:
        sys.exit(code)


if __name__ == "__main__":
    main()
