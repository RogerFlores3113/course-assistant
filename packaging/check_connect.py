"""CI check: `connect claude` with a built executable, in a throwaway home folder, run twice (it must be idempotent).

Usage: python packaging/check_connect.py <path-to-executable>
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def main(exe: str) -> None:
    exe = str(Path(exe).resolve())
    home = Path(tempfile.mkdtemp())
    env = {**os.environ, "HOME": str(home), "USERPROFILE": str(home), "COURSE_ASSISTANT_DATA_DIR": str(home / "data"),
           "CODEX_HOME": str(home / ".codex")}
    if sys.platform == "darwin":
        folder = home / "Library" / "Application Support" / "Claude"
    elif sys.platform == "win32":
        env["APPDATA"] = str(home / "Roaming")
        env["LOCALAPPDATA"] = str(home / "Local")
        folder = home / "Roaming" / "Claude"
    else:
        folder = home / ".config" / "Claude"
    folder.mkdir(parents=True)
    config = folder / "claude_desktop_config.json"
    config.write_text(json.dumps({"mcpServers": {"other": {"command": "x"}}}), encoding="utf-8")

    for attempt in (1, 2):
        done = subprocess.run([exe, "connect", "claude"], env=env, capture_output=True, text=True, timeout=180)
        print(f"$ {exe} connect claude (run {attempt})\n{done.stdout}{done.stderr}")
        assert done.returncode == 0, f"connect claude exited with {done.returncode}"
        servers = json.loads(config.read_text(encoding="utf-8"))["mcpServers"]
        assert "other" in servers, "connect removed another server"
        written = servers["course-assistant"]["command"]
        expected = exe
        if sys.platform == "win32":  # Windows starts the console build for stdio
            expected = str(Path(exe).with_name("course-assistant-mcp.exe"))
        assert os.path.samefile(written, expected), f"wrote {written}, expected {expected}"
    print(f"Connect check passed: {exe}")


if __name__ == "__main__":
    main(sys.argv[1])
