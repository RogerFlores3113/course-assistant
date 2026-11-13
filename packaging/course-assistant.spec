# PyInstaller spec for Course Assistant: macOS .app, Windows folder app, Linux folder.
#
# Windows gets two executables in one folder:
#   "Course Assistant.exe"      windowed, what the user double-clicks
#   "course-assistant-mcp.exe"  console, what Claude Desktop / Codex launch for MCP over stdio
# A windowed Windows exe has no usable stdin/stdout, so the MCP side needs its own console build.
# macOS and Linux use one binary for both (`... mcp`).
#
# Run from the repo root:  uv run pyinstaller packaging/course-assistant.spec --noconfirm

import os
import sys
from PyInstaller.utils.hooks import collect_submodules

HERE = SPECPATH                       # .../packaging
SRC = os.path.join(HERE, "..", "src")

hidden = (
    collect_submodules("uvicorn")
    + collect_submodules("mcp", filter=lambda name: not name.startswith("mcp.cli"))
    + collect_submodules("course_assistant")
    + ["python_multipart", "pypdf", "pptx", "docx", "anthropic"]
)

def _data(src, dest):
    """Bundle a folder only if it exists (skills/ is optional)."""
    return [(src, dest)] if os.path.isdir(src) else []


a = Analysis(
    [os.path.join(HERE, "entry.py")],
    pathex=[SRC],
    datas=(
        _data(os.path.join(SRC, "course_assistant", "templates"), os.path.join("course_assistant", "templates"))
        + _data(os.path.join(SRC, "course_assistant", "static"), os.path.join("course_assistant", "static"))
        # Skills that `course-assistant connect` installs for Claude Code / Codex (read by connect.skills_source_dir)
        + _data(os.path.join(HERE, "..", "skills"), "skills")
    ),
    hiddenimports=hidden,
    excludes=["tkinter", "pytest", "mcp.cli", "typer"],
    noarchive=False,
)
pyz = PYZ(a.pure)

app_exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="Course Assistant" if sys.platform != "linux" else "course-assistant",
    console=sys.platform == "linux",
    upx=False,          # UPX breaks macOS codesign and trips Windows antivirus
    icon=None,          # TODO: app icon (icns / ico)
)

executables = [app_exe]
if sys.platform == "win32":
    mcp_exe = EXE(
        pyz, a.scripts, [],
        exclude_binaries=True,
        name="course-assistant-mcp",
        console=True,
        upx=False,
    )
    executables.append(mcp_exe)

coll = COLLECT(*executables, a.binaries, a.datas, upx=False, name="Course Assistant")

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="Course Assistant.app",
        bundle_identifier="dev.rogerflores.courseassistant",
        version="0.1.0",
        info_plist={"NSHighResolutionCapable": True, "LSUIElement": False},
    )
