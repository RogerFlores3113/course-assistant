"""One-step hookup to AI desktop apps.

`course-assistant connect` writes this app's MCP server entry into Claude Desktop's claude_desktop_config.json and
into Codex's ~/.codex/config.toml, and copies the bundled skills where Claude Code and Codex look for them.

Safety rules, the same as the sister app Talent Search:
  * Only our own `course-assistant` entry is ever added or replaced. Every other server is kept.
  * One backup is saved next to the file (name.bak) before the first change, and never overwritten after that.
  * A settings file we can't parse is left alone and reported.
  * If the app isn't installed (its settings folder doesn't exist) nothing is created.
  * Running it twice changes nothing the second time.
  * A skills folder that we didn't create (no marker file) is never overwritten.
  * Nothing is written for an AI app that isn't installed (no settings folder, no program on the PATH), and
    write_plan() lists every path outside the app's own folder before anything is written.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import stat
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .paths import ENV_DATA_DIR, data_dir, is_frozen

SERVER_KEY = "course-assistant"
SKILL_MARKER = ".installed-by-course-assistant"


@dataclass
class LaunchSpec:
    command: str
    args: list[str]
    env: dict[str, str]

    def as_json_entry(self) -> dict[str, Any]:
        return {"command": self.command, "args": self.args, "env": self.env}


def mcp_launch_spec() -> LaunchSpec:
    """How an AI app should start our MCP server on this machine."""
    env = {ENV_DATA_DIR: str(data_dir().resolve())}
    if is_frozen():
        exe = Path(sys.executable)
        if sys.platform == "win32":
            # Windows ships a separate console build for stdio; the main exe is windowed.
            console = exe.with_name("course-assistant-mcp.exe")
            return LaunchSpec(str(console if console.exists() else exe), ["mcp"], env)
        return LaunchSpec(str(exe), ["mcp"], env)
    # From source: the Python that is running now, so it finds the same installed packages (`uv run` style).
    return LaunchSpec(sys.executable, ["-m", "course_assistant.cli", "mcp"], env)


# ------------------------------------------------------------ config locations
def _home(home: Path | None = None) -> Path:
    return home or Path.home()


def _msix_claude_dirs(env: dict[str, str], home: Path) -> list[Path]:
    """Claude Desktop from the Microsoft Store keeps its settings in a virtualized copy under
    %LOCALAPPDATA%\\Packages\\Claude_*\\LocalCache\\Roaming\\Claude."""
    local = env.get("LOCALAPPDATA")
    base = Path(local) if local else home / "AppData" / "Local"
    try:
        found = [d for d in (base / "Packages").glob("Claude_*") if (d / "LocalCache" / "Roaming" / "Claude").is_dir()]
    except OSError:
        return []
    dirs = [d / "LocalCache" / "Roaming" / "Claude" for d in found]
    dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return dirs


def claude_config_path(platform: str | None = None, env: dict[str, str] | None = None,
                       home: Path | None = None) -> Path:
    platform = platform or sys.platform
    env = os.environ if env is None else env
    h = _home(home)
    if platform == "darwin":
        folder = h / "Library" / "Application Support" / "Claude"
    elif platform == "win32":
        if store := _msix_claude_dirs(env, h):
            folder = store[0]
        else:
            appdata = env.get("APPDATA")
            folder = (Path(appdata) if appdata else h / "AppData" / "Roaming") / "Claude"
    else:
        folder = h / ".config" / "Claude"
    return folder / "claude_desktop_config.json"


def codex_config_path(home: Path | None = None) -> Path:
    override = os.environ.get("CODEX_HOME")
    return (Path(override) if override else _home(home) / ".codex") / "config.toml"


# ------------------------------------------------------------ file helpers
def _backup(path: Path) -> Path | None:
    """One backup next to the file (config.json.bak), made only when there isn't one yet."""
    if not path.exists():
        return None
    dest = path.with_name(f"{path.name}.bak")
    if not dest.exists():
        shutil.copy2(path, dest)
    return dest


def _write(path: Path, text: str) -> None:
    """Replace the file in one step, keeping its permissions (new files are owner-only). No BOM."""
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o600
    tmp = path.with_name(path.name + ".tmp")
    try:
        fd = os.open(tmp, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, mode)
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        if os.name != "nt":
            os.chmod(tmp, mode)
        tmp.replace(path)
    except PermissionError as exc:
        raise OSError(f"Couldn't save {path}: it's locked or read-only. Quit the app and try again.") from exc
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


def _not_installed(path: Path, app_name: str) -> dict[str, Any] | None:
    if path.parent.is_dir():
        return None
    return {"app": app_name, "path": str(path), "installed": False, "ok": False, "backup": None,
            "message": f"{app_name} doesn't seem to be installed on this computer. Install it, open it once, "
                       "then try again."}


def launch_problem(spec: LaunchSpec) -> str | None:
    """Why an AI app couldn't start this server, or None."""
    if not Path(spec.command).exists():
        return f"The program {spec.command} wasn't found, so nothing was changed."
    return None


def _done(app_name: str, path: Path, backup: Path | None, already: bool, entry: dict[str, Any]) -> dict[str, Any]:
    return {"app": app_name, "path": str(path), "installed": True, "ok": True, "backup": str(backup) if backup else None,
            "already_connected": already, "entry": entry, "next_step": NEXT_STEPS[app_name]}


NEXT_STEPS = {
    "Claude Desktop": ("Quit Claude Desktop completely (not just the window) and open it again. Then look for "
                       "course-assistant under Settings > Developer."),
    "Codex": "Restart Codex, then type /mcp to see course-assistant in the list.",
}


# ------------------------------------------------------------ Claude Desktop (JSON)
def _read_json_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8-sig")
    if not text.strip():
        return {}
    try:
        config = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Claude Desktop's settings file isn't valid JSON, so we left it alone "
                         f"({exc.msg}, line {exc.lineno}).") from exc
    if not isinstance(config, dict) or ("mcpServers" in config and not isinstance(config["mcpServers"], dict)):
        raise ValueError("Claude Desktop's settings file has an unexpected shape, so we left it alone.")
    return config


def connect_claude(path: Path | None = None, spec: LaunchSpec | None = None) -> dict[str, Any]:
    path = path or claude_config_path()
    spec = spec or mcp_launch_spec()
    if missing := _not_installed(path, "Claude Desktop"):
        return missing
    if problem := launch_problem(spec):
        return {"app": "Claude Desktop", "path": str(path), "installed": True, "ok": False, "problem": problem,
                "backup": None}
    config = _read_json_config(path)  # validated before anything is touched
    servers = config.setdefault("mcpServers", {})
    want = spec.as_json_entry()
    already = servers.get(SERVER_KEY) == want
    backup = path.with_name(path.name + ".bak") if path.with_name(path.name + ".bak").exists() else None
    if not already:
        backup = _backup(path)
        servers[SERVER_KEY] = want
        _write(path, json.dumps(config, indent=2) + "\n")
    back = _read_json_config(path).get("mcpServers", {}).get(SERVER_KEY)
    if back != want:
        raise ValueError("We wrote Claude Desktop's settings file, but reading it back the entry wasn't what we "
                         "meant to write. Nothing more was changed.")
    return _done("Claude Desktop", path, backup, already, want)


def claude_connected(path: Path | None = None) -> bool:
    try:
        entry = _read_json_config(path or claude_config_path()).get("mcpServers", {}).get(SERVER_KEY)
    except (OSError, ValueError):
        return False
    return isinstance(entry, dict) and Path(str(entry.get("command", ""))).exists()


# ------------------------------------------------------------ Codex (TOML)
def codex_snippet(spec: LaunchSpec | None = None) -> str:
    spec = spec or mcp_launch_spec()
    # JSON string escaping is valid TOML basic-string escaping (handles Windows backslashes).
    args = ", ".join(json.dumps(a) for a in spec.args)
    env = ", ".join(f"{k} = {json.dumps(v)}" for k, v in spec.env.items())
    return (f"[mcp_servers.{SERVER_KEY}]\n"
            f"command = {json.dumps(spec.command)}\n"
            f"args = [{args}]\n"
            f"env = {{ {env} }}\n")


# Our table and any sub-table of it, e.g. [mcp_servers.course-assistant.env].
_SECTION_RE = re.compile(rf"^\[mcp_servers\.(?:{re.escape(SERVER_KEY)}|\"{re.escape(SERVER_KEY)}\")(?:\.[^\]]+)?\]\s*$")


def _strip_our_section(text: str) -> str:
    out, skipping = [], False
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if _SECTION_RE.match(stripped):
            skipping = True
            continue
        if skipping and stripped.startswith("["):
            skipping = False
        if not skipping:
            out.append(line)
    return "".join(out)


def _codex_entry(path: Path) -> dict[str, Any] | None:
    try:
        entry = tomllib.loads(path.read_text(encoding="utf-8-sig")).get("mcp_servers", {}).get(SERVER_KEY)
    except (OSError, tomllib.TOMLDecodeError, AttributeError):
        return None
    return dict(entry) if isinstance(entry, dict) else None


def connect_codex(path: Path | None = None, spec: LaunchSpec | None = None) -> dict[str, Any]:
    path = path or codex_config_path()
    spec = spec or mcp_launch_spec()
    if missing := _not_installed(path, "Codex"):
        return missing
    if problem := launch_problem(spec):
        return {"app": "Codex", "path": str(path), "installed": True, "ok": False, "problem": problem, "backup": None}
    want = spec.as_json_entry()
    original = path.read_text(encoding="utf-8-sig") if path.exists() else ""
    if original.strip():
        try:
            tomllib.loads(original)
        except tomllib.TOMLDecodeError as exc:
            raise ValueError(f"Codex's settings file isn't valid TOML, so we left it alone ({exc}).") from exc
    current = _codex_entry(path)
    already = current is not None and {k: current.get(k) for k in ("command", "args", "env")} == want
    backup = path.with_name(path.name + ".bak") if path.with_name(path.name + ".bak").exists() else None
    if not already:
        backup = _backup(path)
        base = _strip_our_section(original).rstrip()
        text = (base + "\n\n" if base else "") + codex_snippet(spec)
        tomllib.loads(text)  # never write something Codex can't read
        _write(path, text)
    back = _codex_entry(path)
    if not back or {k: back.get(k) for k in ("command", "args", "env")} != want:
        raise ValueError("We wrote Codex's settings file, but reading it back the entry wasn't what we meant to "
                         "write. Nothing more was changed.")
    return _done("Codex", path, backup, already, want)


def codex_connected(path: Path | None = None) -> bool:
    entry = _codex_entry(path or codex_config_path())
    return entry is not None and Path(str(entry.get("command", ""))).exists()


# ------------------------------------------------------------ skills
def skills_source_dir() -> Path:
    """The bundled skills/ folder: inside the app when packaged, the repo's skills/ when run from source."""
    if is_frozen():
        base = Path(getattr(sys, "_MEIPASS", None) or Path(sys.executable).parent)
        return base / "skills"
    return Path(__file__).resolve().parents[2] / "skills"


def skill_targets(home: Path | None = None) -> dict[str, Path]:
    """Where each AI app looks for personal skills."""
    h = _home(home)
    return {"Claude Code": h / ".claude" / "skills", "Codex": h / ".agents" / "skills"}


APP_SKILL_FOLDERS = {"Claude Code": "claude", "Codex": "codex"}


def _skills_in(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_dir() and (p / "SKILL.md").is_file())


def list_skills(src: Path | None = None, app: str | None = None) -> list[Path]:
    """Skill folders under src. The repo keeps one folder per app (skills/claude/<skill>, skills/codex/<skill>), so
    when `app` is given and its folder exists those are used; skills placed directly under src are shared by both."""
    src = src or skills_source_dir()
    shared = _skills_in(src)
    per_app: list[Path] = []
    if app is None:
        for sub in APP_SKILL_FOLDERS.values():
            per_app += _skills_in(src / sub)
    elif app in APP_SKILL_FOLDERS:
        per_app = _skills_in(src / APP_SKILL_FOLDERS[app])
    seen: dict[str, Path] = {}
    for p in [*shared, *per_app]:
        seen.setdefault(p.name, p)
    return sorted(seen.values())


def _copy_skill(skill: Path, dest: Path) -> str:
    """Copy one skill folder. Returns new, updated, unchanged or kept (a folder of the same name that isn't ours)."""
    marker = dest / SKILL_MARKER
    if dest.exists() and (dest.is_symlink() or not marker.is_file()):
        return "kept"
    files = {p.relative_to(skill): p.read_bytes() for p in skill.rglob("*") if p.is_file()}
    status = "new"
    if dest.exists():
        have = {p.relative_to(dest): p.read_bytes() for p in dest.rglob("*") if p.is_file() and p != marker}
        if have == files:
            return "unchanged"
        shutil.rmtree(dest)
        status = "updated"
    dest.mkdir(parents=True)
    for rel, data in files.items():
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
    marker.write_text("Installed by Course Assistant. Safe to delete this folder; it is replaced on update.\n",
                      encoding="utf-8")
    return status


def detect_clients(home: Path | None = None) -> dict[str, bool]:
    """Which AI apps look installed here. Claude Desktop and Codex: their settings folder exists (Codex also: a
    `codex` program on the PATH). Claude Code: ~/.claude exists or a `claude` program is on the PATH."""
    h = _home(home)
    return {
        "Claude Desktop": claude_config_path(home=h).parent.is_dir(),
        "Codex": codex_config_path(h).parent.is_dir() or shutil.which("codex") is not None,
        "Claude Code": (h / ".claude").is_dir() or shutil.which("claude") is not None,
    }


_TARGET_APPS = {"claude": ("Claude Desktop", "Claude Code"), "codex": ("Codex",), "all": ("Claude Desktop", "Claude Code", "Codex")}


def write_plan(target: str = "all", *, with_skills: bool = True, home: Path | None = None) -> dict[str, Any]:
    """What connecting would write outside the app's own folder, for installed apps only.
    {"writes": [{"app", "path", "what"}], "skipped": [apps that aren't installed, so nothing is written for them]}"""
    h = _home(home)
    found = detect_clients(h)
    writes: list[dict[str, str]] = []
    skipped: list[str] = []
    for app in _TARGET_APPS[target]:
        if app == "Claude Code" and not with_skills:
            continue
        if not found[app]:
            skipped.append(app)
            continue
        if app == "Claude Desktop":
            writes.append({"app": app, "path": str(claude_config_path(home=h)), "what": "add Course Assistant to its settings"})
        elif app == "Codex":
            writes.append({"app": app, "path": str(codex_config_path(h)), "what": "add Course Assistant to its settings"})
            if with_skills:
                writes.append({"app": app, "path": str(skill_targets(h)["Codex"]), "what": "add the study skills"})
        else:
            writes.append({"app": app, "path": str(skill_targets(h)["Claude Code"]), "what": "add the study skills"})
    return {"writes": writes, "skipped": skipped}


def plan_lines(plan: dict[str, Any]) -> list[str]:
    lines = [f"{w['app']}: {w['what']} ({w['path']})" for w in plan["writes"]]
    if plan["skipped"]:
        lines.append("Not installed here, so nothing will be written for: " + ", ".join(plan["skipped"]) + ".")
    return lines


def install_skills(home: Path | None = None, src: Path | None = None, only: list[str] | None = None,
                   only_installed: bool = False) -> dict[str, Any]:
    """Copy every bundled skill into each AI app's skills folder. A missing skills/ folder is not an error.
    With only_installed, an app that doesn't look installed gets nothing (it is listed under "skipped")."""
    src = src or skills_source_dir()
    result: dict[str, Any] = {"found": [s.name for s in list_skills(src)], "targets": {}}
    if not result["found"]:
        result["message"] = ("This copy of the app has no skill files to install (looked in "
                             f"{src}). Connecting still works; your AI app just won't get the "
                             "extra study instructions.")
        return result
    found = detect_clients(home) if only_installed else {}
    result["skipped"] = []
    for app, base in skill_targets(home).items():
        if only and app not in only:
            continue
        if only_installed and not found.get(app):
            result["skipped"].append(app)
            continue
        skills = list_skills(src, app)
        if not skills:
            continue
        base.mkdir(parents=True, exist_ok=True)
        result["targets"][app] = {"path": str(base), "skills": {s.name: _copy_skill(s, base / s.name) for s in skills}}
    return result


# ------------------------------------------------------------ everything
def connect(target: str = "all", *, with_skills: bool = True, home: Path | None = None) -> list[dict[str, Any]]:
    """Connect `claude`, `codex` or `all`. Returns one result dict per app (plus skills). Only installed apps are
    touched; an app that isn't installed is reported and nothing is written for it."""
    results: list[dict[str, Any]] = []
    if target in ("claude", "all"):
        results.append(connect_claude(claude_config_path(home=home)))
    if target in ("codex", "all"):
        results.append(connect_codex(codex_config_path(home)))
    if with_skills:
        only = {"claude": ["Claude Code"], "codex": ["Codex"], "all": None}[target]
        results.append({"app": "Skills", **install_skills(home=home, only=only, only_installed=True)})
    return results


def describe(result: dict[str, Any]) -> list[str]:
    """Plain lines for a person, one result dict at a time."""
    app = result["app"]
    if app == "Skills":
        if not result["targets"]:
            if result.get("skipped"):
                return ["No skills were installed: " + " and ".join(result["skipped"]) + " doesn't seem to be installed here."]
            return [result.get("message", "No skills installed.")]
        lines = []
        for name, info in result["targets"].items():
            counts: dict[str, int] = {}
            for status in info["skills"].values():
                counts[status] = counts.get(status, 0) + 1
            lines.append(f"{name} skills in {info['path']}: " + ", ".join(f"{n} {s}" for s, n in sorted(counts.items())))
        return lines
    if not result.get("installed", True):
        return [result["message"]]
    if not result["ok"]:
        return [f"{app}: {result['problem']}"]
    head = f"{app} was already connected." if result["already_connected"] else f"Connected {app} ({result['path']})."
    lines = [head]
    if result.get("backup") and not result["already_connected"]:
        lines.append(f"Your old settings are saved as {result['backup']}.")
    lines.append(result["next_step"])
    return lines
