import json
import sys
import tomllib
from pathlib import Path

import pytest

from course_assistant import connect, paths


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("HOME", str(h))
    monkeypatch.setenv("USERPROFILE", str(h))
    monkeypatch.setenv("APPDATA", str(h / "Roaming"))
    monkeypatch.setenv("LOCALAPPDATA", str(h / "Local"))
    monkeypatch.setenv("CODEX_HOME", str(h / ".codex"))
    monkeypatch.setenv(paths.ENV_DATA_DIR, str(tmp_path / "data"))
    return h


@pytest.fixture
def spec():
    return connect.LaunchSpec(sys.executable, ["-m", "course_assistant.cli", "mcp"], {paths.ENV_DATA_DIR: "/tmp/x"})


def make_skills(root: Path):
    for name in ("course-assistant-study", "course-assistant-ground"):
        (root / name).mkdir(parents=True)
        (root / name / "SKILL.md").write_text(f"---\nname: {name}\ndescription: d\n---\nbody\n")
    (root / "not-a-skill").mkdir()
    return root


def test_launch_spec_from_source():
    spec = connect.mcp_launch_spec()
    assert spec.command == sys.executable and spec.args == ["-m", "course_assistant.cli", "mcp"]
    assert paths.ENV_DATA_DIR in spec.env


def test_launch_spec_frozen(monkeypatch, tmp_path):
    exe = tmp_path / "course-assistant"
    exe.write_text("")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe))
    spec = connect.mcp_launch_spec()
    assert spec.command == str(exe) and spec.args == ["mcp"]


def test_config_paths(tmp_path):
    h = tmp_path
    assert connect.claude_config_path("darwin", {}, h) == h / "Library/Application Support/Claude/claude_desktop_config.json"
    assert connect.claude_config_path("linux", {}, h) == h / ".config/Claude/claude_desktop_config.json"
    assert connect.claude_config_path("win32", {"APPDATA": str(h / "R")}, h) == h / "R/Claude/claude_desktop_config.json"


def test_claude_not_installed_creates_nothing(home, spec):
    res = connect.connect_claude(home / "nope" / "claude_desktop_config.json", spec)
    assert res["installed"] is False and not (home / "nope").exists()


def test_claude_keeps_other_servers_backs_up_and_is_idempotent(home, spec):
    path = home / "Claude" / "claude_desktop_config.json"
    path.parent.mkdir()
    original = {"theme": "dark", "mcpServers": {"other": {"command": "x"}}}
    path.write_text(json.dumps(original))
    first = connect.connect_claude(path, spec)
    assert first["ok"] and not first["already_connected"]
    data = json.loads(path.read_text())
    assert data["theme"] == "dark" and data["mcpServers"]["other"] == {"command": "x"}
    assert data["mcpServers"]["course-assistant"]["args"] == ["-m", "course_assistant.cli", "mcp"]
    assert json.loads(Path(first["backup"]).read_text()) == original
    before = path.read_text()
    second = connect.connect_claude(path, spec)
    assert second["already_connected"] and path.read_text() == before
    assert json.loads(Path(second["backup"]).read_text()) == original  # backup from before the first change survives


def test_claude_refreshes_stale_entry(home, spec):
    path = home / "Claude" / "claude_desktop_config.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"mcpServers": {"course-assistant": {"command": "/old"}}}))
    res = connect.connect_claude(path, spec)
    assert res["ok"] and json.loads(path.read_text())["mcpServers"]["course-assistant"]["command"] == sys.executable


def test_claude_new_file_and_bad_json(home, spec):
    path = home / "Claude" / "claude_desktop_config.json"
    path.parent.mkdir()
    assert connect.connect_claude(path, spec)["ok"] and connect.claude_connected(path)
    path.write_text("{ not json")
    with pytest.raises(ValueError, match="left it alone"):
        connect.connect_claude(path, spec)
    assert path.read_text() == "{ not json"


def test_claude_missing_program_refused(home):
    path = home / "Claude" / "claude_desktop_config.json"
    path.parent.mkdir()
    res = connect.connect_claude(path, connect.LaunchSpec(str(home / "gone"), ["mcp"], {}))
    assert res["ok"] is False and not path.exists()


def test_codex_preserves_and_is_idempotent(home, spec):
    path = home / ".codex" / "config.toml"
    path.parent.mkdir()
    path.write_text('model = "x"\n\n[mcp_servers.other]\ncommand = "o"\nargs = ["a"]\n')
    res = connect.connect_codex(path, spec)
    assert res["ok"] and not res["already_connected"]
    data = tomllib.loads(path.read_text())
    assert data["model"] == "x" and data["mcp_servers"]["other"] == {"command": "o", "args": ["a"]}
    assert data["mcp_servers"]["course-assistant"]["args"] == ["-m", "course_assistant.cli", "mcp"]
    assert tomllib.loads(Path(res["backup"]).read_text())["model"] == "x"
    before = path.read_text()
    again = connect.connect_codex(path, spec)
    assert again["already_connected"] and path.read_text() == before and connect.codex_connected(path)


def test_codex_replaces_only_our_tables(home, spec):
    path = home / ".codex" / "config.toml"
    path.parent.mkdir()
    path.write_text('[mcp_servers.course-assistant]\ncommand = "/old"\n\n[mcp_servers.course-assistant.env]\nA = "1"\n\n'
                    '[mcp_servers.keep]\ncommand = "k"\n')
    connect.connect_codex(path, spec)
    data = tomllib.loads(path.read_text())
    assert data["mcp_servers"]["keep"] == {"command": "k"}
    assert data["mcp_servers"]["course-assistant"]["command"] == sys.executable
    assert path.read_text().count("[mcp_servers.course-assistant") == 1


def test_codex_bad_toml_left_alone(home, spec):
    path = home / ".codex" / "config.toml"
    path.parent.mkdir()
    path.write_text("[broken")
    with pytest.raises(ValueError):
        connect.connect_codex(path, spec)
    assert path.read_text() == "[broken"


def test_skills_copy_both_apps_idempotent_and_update(home, tmp_path):
    src = make_skills(tmp_path / "skills")
    res = connect.install_skills(home, src)
    assert res["found"] == ["course-assistant-ground", "course-assistant-study"]
    for base in (home / ".claude/skills", home / ".agents/skills"):
        assert (base / "course-assistant-study/SKILL.md").is_file()
        assert not (base / "not-a-skill").exists()
    assert set(res["targets"]["Codex"]["skills"].values()) == {"new"}
    assert set(connect.install_skills(home, src)["targets"]["Claude Code"]["skills"].values()) == {"unchanged"}
    (src / "course-assistant-study/SKILL.md").write_text("changed")
    again = connect.install_skills(home, src)["targets"]["Claude Code"]["skills"]
    assert again["course-assistant-study"] == "updated" and again["course-assistant-ground"] == "unchanged"
    assert (home / ".claude/skills/course-assistant-study/SKILL.md").read_text() == "changed"


def test_skills_never_overwrite_foreign_folder(home, tmp_path):
    src = make_skills(tmp_path / "skills")
    mine = home / ".claude/skills/course-assistant-study"
    mine.mkdir(parents=True)
    (mine / "SKILL.md").write_text("the user's own")
    res = connect.install_skills(home, src, only=["Claude Code"])
    assert res["targets"]["Claude Code"]["skills"]["course-assistant-study"] == "kept"
    assert (mine / "SKILL.md").read_text() == "the user's own"
    assert "Codex" not in res["targets"]


def test_skills_missing_folder_is_not_an_error(home, tmp_path):
    res = connect.install_skills(home, tmp_path / "nothing")
    assert res["found"] == [] and res["targets"] == {} and "message" in res
    assert connect.describe({"app": "Skills", **res})


def test_connect_all_in_fake_home(home, monkeypatch, tmp_path):
    claude = connect.claude_config_path()
    claude.parent.mkdir(parents=True)
    (home / ".codex").mkdir()
    monkeypatch.setattr(connect, "skills_source_dir", lambda: make_skills(tmp_path / "skills"))
    results = connect.connect("all")
    assert [r["app"] for r in results] == ["Claude Desktop", "Codex", "Skills"]
    assert all(r["ok"] for r in results[:2])
    assert (home / ".claude/skills/course-assistant-study/SKILL.md").is_file()
    assert all(line for r in results for line in connect.describe(r))
    assert all(r["already_connected"] for r in connect.connect("all", with_skills=False))


def make_app_skills(root: Path):
    """The repo layout: skills/<app>/<skill>/SKILL.md."""
    for app in ("claude", "codex"):
        d = root / app / "course-assistant"
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(f"---\nname: course-assistant\ndescription: {app}\n---\nbody\n")
    return root


def test_skills_found_in_per_app_folders_regression(home, tmp_path):
    """Panel p3: 'No skills were bundled' was printed although skills/claude and skills/codex exist."""
    src = make_app_skills(tmp_path / "skills")
    assert [p.name for p in connect.list_skills(src)] == ["course-assistant"]
    res = connect.install_skills(home, src)
    assert "message" not in res and res["found"] == ["course-assistant"]
    assert "description: claude" in (home / ".claude/skills/course-assistant/SKILL.md").read_text()
    assert "description: codex" in (home / ".agents/skills/course-assistant/SKILL.md").read_text()


def test_repo_skills_folder_is_found():
    assert connect.list_skills(), "the repo's own skills/ folder must be found"
    assert connect.list_skills(app="Claude Code") and connect.list_skills(app="Codex")


def test_no_skills_message_explains_and_does_not_fail(home, tmp_path):
    res = connect.install_skills(home, tmp_path / "empty")
    lines = connect.describe({"app": "Skills", **res})
    assert lines and "no skill files" in lines[0] and "Connecting still works" in lines[0]
