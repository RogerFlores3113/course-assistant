import sys
import types

import pytest

from course_assistant import __version__, cli, paths
from course_assistant.store import Store


@pytest.fixture(autouse=True)
def data(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_DATA_DIR, str(tmp_path / "data"))
    return tmp_path


def run(*argv):
    try:
        cli.main(list(argv))
    except SystemExit as e:
        return e.code or 0
    return 0


def test_version(capsys):
    assert run("version") == 0
    assert __version__ in capsys.readouterr().out


def test_no_args_and_bare_options_mean_serve(monkeypatch):
    seen = []
    monkeypatch.setattr(cli, "run_app", lambda port, open_browser: seen.append((port, open_browser)))
    monkeypatch.setattr(cli, "_setup_logging", lambda: None)
    run()
    run("--no-browser", "--port", "9000")
    run("serve", "--no-browser")
    assert seen == [(paths.DEFAULT_PORT, True), (9000, False), (paths.DEFAULT_PORT, False)]


def test_mcp_dispatches_to_server_main(monkeypatch):
    called = []
    monkeypatch.setitem(sys.modules, "course_assistant.mcp_server", types.SimpleNamespace(main=lambda: called.append(1)))
    monkeypatch.setattr("course_assistant.mcp_server", sys.modules["course_assistant.mcp_server"], raising=False)
    assert run("mcp") == 0
    assert called == [1]


def test_open_when_not_running(capsys):
    assert run("open") == 1
    assert "isn't running" in capsys.readouterr().err


def test_open_when_running_uses_launch_code(monkeypatch):
    paths.port_file().write_text("8123")
    monkeypatch.setattr(cli, "_answers", lambda port: True)
    opened = []
    monkeypatch.setattr(cli.webbrowser, "open", opened.append)
    assert run("open") == 0
    assert len(opened) == 1 and opened[0].startswith("http://127.0.0.1:8123/launch?code=")
    code = opened[0].split("code=")[1]
    assert paths.use_launch_code(code)


def test_stale_port_file_is_dropped(monkeypatch):
    paths.port_file().write_text("8123")
    monkeypatch.setattr(cli, "_answers", lambda port: False)
    assert cli.find_running() is None
    assert not paths.port_file().exists()


def test_serve_starts_uvicorn_and_writes_port_file(monkeypatch):
    created = {}
    web = types.ModuleType("course_assistant.web")
    web.create_app = lambda store, **kw: created.setdefault("app", object())
    monkeypatch.setitem(sys.modules, "course_assistant.web", web)
    uv = types.ModuleType("uvicorn")
    seen = {}

    def fake_run(app, **kw):
        seen.update(kw, app=app, port_file=paths.port_file().read_text())

    uv.run = fake_run
    monkeypatch.setitem(sys.modules, "uvicorn", uv)
    monkeypatch.setattr(cli, "find_running", lambda: None)
    monkeypatch.setattr(cli, "_pick_port", lambda p: 8999)
    cli.run_app(port=8766, open_browser=False)
    assert seen["host"] == "127.0.0.1" and seen["port"] == 8999 and seen["app"] is created["app"]
    assert seen["port_file"] == "8999"
    assert not paths.port_file().exists()  # removed on exit


def test_serve_when_already_running_only_opens(monkeypatch):
    monkeypatch.setattr(cli, "find_running", lambda: 8766)
    opened = []
    monkeypatch.setattr(cli.webbrowser, "open", opened.append)
    cli.run_app(open_browser=True)
    assert opened and "/launch?code=" in opened[0]
    opened.clear()
    cli.run_app(open_browser=False)
    assert not opened


def test_connect_dispatch(monkeypatch, capsys):
    from course_assistant import connect

    calls = []
    monkeypatch.setattr(connect, "connect", lambda target, with_skills=True: calls.append((target, with_skills)) or
                        [{"app": "Skills", "targets": {}, "message": "none"}])
    assert run("connect") == 0
    assert run("connect", "codex", "--no-skills") == 0
    assert calls == [("all", True), ("codex", False)]


def test_connect_reports_failure(monkeypatch):
    from course_assistant import connect

    monkeypatch.setattr(connect, "connect", lambda *a, **k: [{"app": "Codex", "installed": False, "ok": False, "message": "nope"}])
    assert run("connect", "codex") == 1


def fake_importers(monkeypatch, fail=()):
    mod = types.ModuleType("course_assistant.importers")

    class ImportError_(Exception):
        pass

    imported = []

    def import_file(store, course_id, src, **kw):
        if src.name in fail:
            raise ImportError_("unreadable")
        imported.append((course_id, src.name, kw))
        return 7

    mod.ImportError_, mod.import_file = ImportError_, import_file
    monkeypatch.setitem(sys.modules, "course_assistant.importers", mod)
    return imported


def test_import_requires_course_or_create(data, monkeypatch, capsys):
    imported = fake_importers(monkeypatch)
    f = data / "a.pdf"
    f.write_text("x")
    assert run("import", "Calculus", str(f)) == 1
    assert "--create" in capsys.readouterr().err
    assert run("import", "Calculus", str(f), "--create", "--kind", "slides") == 0
    assert imported[0][1] == "a.pdf" and imported[0][2]["kind"] == "slides"
    store = Store(paths.db_path())
    assert [c["name"] for c in store.courses()] == ["Calculus"]
    # second time finds it by a partial, case-insensitive name and by id
    assert run("import", "calc", str(f)) == 0
    assert run("import", str(store.courses()[0]["id"]), str(f)) == 0
    assert len(store.courses()) == 1


def test_import_ambiguous_and_errors(data, monkeypatch, capsys):
    fake_importers(monkeypatch, fail={"bad.pdf"})
    store = Store(paths.db_path())
    store.add_course("Calculus I")
    store.add_course("Calculus II")
    good, bad = data / "good.pdf", data / "bad.pdf"
    good.write_text("x")
    bad.write_text("x")
    assert run("import", "Calculus", str(good)) == 1
    assert "More than one course" in capsys.readouterr().err
    assert run("import", "Calculus I", str(good), str(bad), str(data / "missing.pdf")) == 1
    err = capsys.readouterr().err
    assert "Could not import bad.pdf: unreadable" in err and "not a file" in err
