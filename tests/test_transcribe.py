from types import SimpleNamespace

import pytest

from course_assistant import transcribe
from course_assistant.importers import import_file
from course_assistant.store import Store


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("COURSE_ASSISTANT_DATA_DIR", str(tmp_path / "data"))
    s = Store(":memory:")
    cid = s.add_course("Physics")
    return s, cid, tmp_path


def add_audio(s, cid, tmp, name="lecture 2.mp3"):
    f = tmp / name
    f.write_bytes(b"ID3" + name.encode() * 10)
    return import_file(s, cid, f)


def add_image(s, cid, tmp, name="notes.png"):
    f = tmp / name
    f.write_bytes(b"\x89PNG\r\n\x1a\n" + name.encode() * 10)
    return import_file(s, cid, f)


class FakeModel:
    def transcribe(self, path, **kw):
        segs = [SimpleNamespace(start=0.0, end=20.0, text=" Welcome back."),
                SimpleNamespace(start=20.0, end=100.0, text=" Today is conservation of energy."),
                SimpleNamespace(start=100.0, end=130.0, text=" Questions?")]
        return iter(segs), SimpleNamespace(duration=130.0)


def test_no_whisper_installed_marks_job_failed(env):
    s, cid, tmp = env
    mid = add_audio(s, cid, tmp)
    job = s.jobs("queued")[0]
    out = transcribe.run_job(s, job["id"], whisper_factory=lambda store: None)
    assert out["status"] == "error"
    assert "captions" in out["note"] and "transcribe" in out["note"]
    m = s.material(mid)
    assert m["status"] == "needs_transcription" and "captions" in m["status_note"]


def test_default_loader_without_faster_whisper(env):
    # faster-whisper is an optional extra and is not installed in the test environment
    s, cid, tmp = env
    add_audio(s, cid, tmp)
    results = transcribe.run_pending(s)
    assert [r["status"] for r in results] == ["error"]


def test_whisper_success(env):
    s, cid, tmp = env
    mid = add_audio(s, cid, tmp)
    results = transcribe.run_pending(s, whisper_factory=lambda store: FakeModel())
    assert results[0]["status"] == "done" and results[0]["progress"] == 1.0
    m = s.material(mid)
    assert m["status"] == "ready" and m["duration_seconds"] == 130.0
    ch = s.chunks_of(mid)
    assert ch and ch[0]["loc_kind"] == "time" and ch[0]["loc_start"] == "0"
    assert "conservation of energy" in " ".join(c["text"] for c in ch)
    assert s.jobs("queued") == []


def test_whisper_model_load_failure_keeps_material_waiting(env):
    s, cid, tmp = env
    mid = add_audio(s, cid, tmp)

    def boom(store):
        raise OSError("offline")
    out = transcribe.run_pending(s, whisper_factory=boom)[0]
    assert out["status"] == "error" and "internet" in out["note"]
    assert s.material(mid)["status"] == "needs_transcription"


def test_whisper_runtime_error_marks_material_error(env):
    s, cid, tmp = env

    class Bad:
        def transcribe(self, *a, **k):
            raise RuntimeError("decode failed")
    mid = add_audio(s, cid, tmp)
    out = transcribe.run_pending(s, whisper_factory=lambda st: Bad())[0]
    assert out["status"] == "error" and "decode failed" in out["note"]
    assert s.material(mid)["status"] == "error"


def test_run_job_skips_finished_and_unknown(env):
    s, cid, tmp = env
    add_audio(s, cid, tmp)
    jid = s.jobs("queued")[0]["id"]
    transcribe.run_job(s, jid, whisper_factory=lambda st: FakeModel())
    again = transcribe.run_job(s, jid, whisper_factory=lambda st: None)
    assert again["status"] == "done"
    with pytest.raises(KeyError):
        transcribe.run_job(s, 9999)
    other = s.add_job("mystery")
    assert transcribe.run_job(s, other)["status"] == "error"


class FakeVision:
    def __init__(self, text):
        self.text = text
        self.calls = []

    def complete_image(self, system, image_bytes, media_type, prompt):
        self.calls.append((media_type, len(image_bytes)))
        return self.text


def test_ocr_success_creates_section_chunks(env):
    s, cid, tmp = env
    mid = add_image(s, cid, tmp)
    prov = FakeVision("Newton's second law: F = ma.\n\nWork is force times distance.")
    out = transcribe.run_pending(s, provider=prov)[0]
    assert out["status"] == "done" and prov.calls[0][0] == "image/png"
    ch = s.chunks_of(mid)
    assert len(ch) == 1 and ch[0]["loc_kind"] == "section" and ch[0]["heading"] == "Photo of notes"
    assert ch[0]["loc_label"] == "Photo of notes"
    assert s.material(mid)["status"] == "ready"


def test_ocr_long_text_splits(env):
    s, cid, tmp = env
    mid = add_image(s, cid, tmp)
    transcribe.run_pending(s, provider=FakeVision("word " * 800))
    ch = s.chunks_of(mid)
    assert len(ch) == 3 and ch[1]["heading"] == "Photo of notes (part 2)"


def test_ocr_without_provider_fails_friendly(env, monkeypatch):
    s, cid, tmp = env
    mid = add_image(s, cid, tmp)
    out = transcribe.run_pending(s)[0]  # no providers module to get one from (or none configured)
    assert out["status"] == "error" and "Settings" in out["note"]
    assert s.material(mid)["status"] == "needs_transcription"


def test_ocr_provider_without_vision_and_empty_result(env):
    s, cid, tmp = env
    add_image(s, cid, tmp)
    out = transcribe.run_pending(s, provider=object())[0]
    assert out["status"] == "error"
    mid2 = add_image(s, cid, tmp, "b.jpg")
    out = transcribe.run_pending(s, provider=FakeVision("   "))[0]
    assert out["status"] == "error" and "readable" in out["note"]
    assert s.material(mid2)["status"] == "error"
