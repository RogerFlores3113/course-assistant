"""Where the app keeps its data.

Packaged app: the platform's per-user data folder
  (macOS: ~/Library/Application Support/Course Assistant,
   Windows: %LOCALAPPDATA%\\Course Assistant, Linux: ~/.local/share/Course Assistant).
Development: ./data, or COURSE_ASSISTANT_DATA_DIR if set.
"""
from __future__ import annotations

import hmac
import os
import secrets
import sys
import time
from pathlib import Path

APP_NAME = "Course Assistant"
APP_SLUG = "course-assistant"
DEFAULT_PORT = 8766  # Talent Search uses 8765
ENV_DATA_DIR = "COURSE_ASSISTANT_DATA_DIR"


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def _private(path: Path, mode: int) -> None:
    """Owner-only access where the OS has POSIX permissions (not Windows)."""
    if os.name == "nt":
        return
    try:
        os.chmod(path, mode)
    except OSError:
        pass


def private_dir(path: Path) -> Path:
    path.mkdir(parents=True, mode=0o700, exist_ok=True)
    if path.stat().st_mode & 0o077:
        _private(path, 0o700)
    return path


def data_dir() -> Path:
    override = os.environ.get(ENV_DATA_DIR)
    if override:
        base = Path(override)
    elif is_frozen():
        from platformdirs import user_data_dir

        base = Path(user_data_dir(APP_NAME, appauthor=False))
    else:
        base = Path("data")
    return private_dir(base)


def db_path() -> Path:
    """The database file, created owner-only (0600) before SQLite opens it."""
    path = data_dir() / "course-assistant.db"
    if os.name != "nt" and not path.exists():
        try:
            os.close(os.open(path, os.O_CREAT | os.O_WRONLY, 0o600))
        except OSError:
            pass
    return path


def files_dir() -> Path:
    """Original course files (PDFs, slides, recordings, snapshots), one folder per course."""
    return private_dir(data_dir() / "files")


def course_files_dir(course_id: int) -> Path:
    return private_dir(files_dir() / str(course_id))


def resource_dir() -> Path:
    """Folder holding templates/ and static/ (inside the bundle when frozen)."""
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS")) / "course_assistant"
    return Path(__file__).parent


def port_file() -> Path:
    return data_dir() / "app.port"


def app_url(path: str = "/") -> str:
    try:
        port = int(port_file().read_text().strip())
    except (OSError, ValueError):
        port = DEFAULT_PORT
    return f"http://127.0.0.1:{port}{path}"


# ------------------------------------------------------------ app window key
# The window is locked with a per-install key so other programs and accounts on this computer can't read it.
# The key lives in a 0600 file and reaches the browser only as an HttpOnly cookie; the launch URL carries a
# one-time code instead of the key.
LAUNCH_CODE_TTL = 300


def _write_private(path: Path, text: str) -> None:
    fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(text)
    _private(path, 0o600)


def app_key() -> str:
    path = data_dir() / "app.key"
    try:
        key = path.read_text().strip()
        if len(key) >= 32:
            return key
    except OSError:
        pass
    key = secrets.token_urlsafe(32)
    _write_private(path, key)
    return key


def new_launch_code() -> str:
    code = secrets.token_urlsafe(24)
    _write_private(data_dir() / "launch.code", f"{code} {int(time.time())}")
    return code


def use_launch_code(code: str) -> bool:
    """True once for a fresh code; the code is spent either way."""
    path = data_dir() / "launch.code"
    try:
        saved, at = path.read_text().split()
        path.unlink()
    except (OSError, ValueError):
        return False
    fresh = time.time() - int(at) <= LAUNCH_CODE_TTL
    return bool(code) and fresh and hmac.compare_digest(saved.encode(), code.encode())
