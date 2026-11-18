import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from course_assistant import disclaimer, paths, web
from course_assistant.store import Store

KEY = "k" * 40
HOST = "http://127.0.0.1:8766"
STATIC = Path(__file__).parent.parent / "src" / "course_assistant" / "static"


@pytest.fixture()
def ready(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_DATA_DIR, str(tmp_path / "data"))
    store = Store(":memory:")
    store.set_setting(disclaimer.ACK_SETTING, "2026-01-01T00:00:00+00:00")
    app = web.create_app(store, key=KEY, provider_factory=lambda: None)
    c = TestClient(app, base_url=HOST)
    c.cookies.set(web.COOKIE, KEY)
    return SimpleNamespace(store=store, client=c)


def test_css_has_no_stray_px_font_sizes_or_radii():
    css = (STATIC / "style.css").read_text()
    assert not re.search(r"font-size:\s*\d+px", css.replace("--fs-xs: 12px", ""))
    assert not re.search(r"border-radius:\s*\d+px", css)
    assert "border-radius: 999px" not in css


def test_settings_explains_answer_check(ready):
    t = ready.client.get("/settings").text
    assert "What Answer check protects" in t and "does not check your own work" in t


def test_add_exam_on_dashboard_and_calendar(ready):
    cid = ready.store.add_course("Chem")
    assert f"/courses/{cid}/calendar?add=exam#add-event" in ready.client.get("/").text
    t = ready.client.get(f"/courses/{cid}/calendar?add=exam").text
    assert ">Add exam<" in t and re.search(r'<option value="exam"\s+selected', t)


def test_materials_summary_line(ready):
    cid = ready.store.add_course("Chem")
    t = ready.client.get(f"/courses/{cid}/materials").text
    assert 'id="materials-summary"' in t and "0 files" in t and "0 graded items" in t


def test_copy_buttons_in_app_js():
    js = (STATIC / "app.js").read_text()
    assert "addCopyButtons" in js and 'say("Copied")' in js and 'type = "button"' in js
    assert "htmx:afterSwap" in js
