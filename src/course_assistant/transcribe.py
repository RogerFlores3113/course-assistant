"""Background jobs that turn recordings and photos of notes into text chunks.

Transcription uses faster-whisper when it is installed (the optional `transcribe` extra) and runs on this
computer. Photo OCR uses the AI provider's vision when one is set up. Without either, the job fails with a
message that says what to do instead.
"""
from __future__ import annotations

from typing import Any, Callable

from . import paths
from .importers.common import IMAGES, MIME, chunk, chunk_cues, split_text
from .store import Store

NO_WHISPER = ("Transcription isn't installed. Add a captions file (.vtt or .srt) for this recording, or install "
              "the transcribe extra (pip install \"course-assistant[transcribe]\") and try again.")
NO_PROVIDER = ("Reading photos of notes needs an AI with vision. Connect one in Settings, or type or paste the "
               "notes in as text instead.")
OCR_SYSTEM = ("You transcribe photos of a student's handwritten or printed course notes. Return only the text that "
              "is in the picture, in reading order, as plain text or Markdown. Write equations in plain text or "
              "LaTeX. Do not explain, summarize or add anything that is not on the page. If nothing is readable, "
              "return an empty reply.")
OCR_PROMPT = "Transcribe this photo of notes."


def _fail(store: Store, job: dict[str, Any], message: str, *, material_error: bool = False) -> dict[str, Any]:
    store.update_job(job["id"], status="error", note=message)
    if job.get("material_id"):
        if material_error:
            store.update_material(job["material_id"], status="error", status_note=message)
        else:
            store.update_material(job["material_id"], status_note=message)
    return store.job(job["id"])  # type: ignore[return-value]


def _load_whisper(store: Store) -> Callable[..., Any] | None:
    """A faster-whisper model object, or None when the extra isn't installed. Models download on first use."""
    try:
        from faster_whisper import WhisperModel  # type: ignore[import-not-found]
    except ImportError:
        return None
    size = store.get_setting("whisper_model", "base")
    return WhisperModel(size, device="auto", compute_type="int8")


def _transcribe(store: Store, job: dict[str, Any], material: dict[str, Any], whisper_factory) -> dict[str, Any]:
    try:
        model = (whisper_factory or _load_whisper)(store)
    except Exception:  # noqa: BLE001
        return _fail(store, job, "I couldn't load the speech model. It downloads the first time it is used, so "
                                 "check your internet connection and try again, or add a captions file instead.")
    if model is None:
        return _fail(store, job, NO_WHISPER)
    path = paths.files_dir() / material["filename"]
    if not path.is_file():
        return _fail(store, job, "The recording's file is missing from this computer, so I can't transcribe it.",
                     material_error=True)
    segments, info = model.transcribe(str(path), vad_filter=True)
    total = float(getattr(info, "duration", 0) or 0)
    cues: list[tuple[float, float, str]] = []
    for seg in segments:
        cues.append((float(seg.start), float(seg.end), str(seg.text)))
        if total:
            store.update_job(job["id"], progress=min(0.99, float(seg.end) / total))
    chunks = chunk_cues(cues)
    if not chunks:
        return _fail(store, job, "I couldn't hear any speech in this recording. Add a captions file instead.",
                     material_error=True)
    store.replace_chunks(material["id"], chunks)
    duration = total or max(c[1] for c in cues)
    store.update_material(material["id"], status="ready", status_note="", duration_seconds=duration)
    store.update_job(job["id"], status="done", progress=1.0, note="Transcribed")
    return store.job(job["id"])  # type: ignore[return-value]


def _ocr(store: Store, job: dict[str, Any], material: dict[str, Any], provider) -> dict[str, Any]:
    if provider is None:
        try:
            from . import providers  # written by another module; may be absent
            provider = providers.get_provider(store)
        except ImportError:
            provider = None
    if provider is None or not hasattr(provider, "complete_image"):
        return _fail(store, job, NO_PROVIDER)
    path = paths.files_dir() / material["filename"]
    if not path.is_file():
        return _fail(store, job, "The photo's file is missing from this computer, so I can't read it.",
                     material_error=True)
    ext = path.suffix.lower()
    if ext not in IMAGES:
        return _fail(store, job, "That file is not a photo, so I can't read it with OCR.", material_error=True)
    try:
        text = provider.complete_image(OCR_SYSTEM, path.read_bytes(), MIME.get(ext, "image/jpeg"), OCR_PROMPT)
    except NotImplementedError:
        return _fail(store, job, NO_PROVIDER)
    text = (text or "").strip()
    if not text:
        return _fail(store, job, "I couldn't find readable text in this photo. Try a sharper, well lit photo, or "
                                 "type the notes in.", material_error=True)
    pieces = split_text(text)
    chunks = [chunk(p, "section", f"photo-{i}", heading="Photo of notes" if i == 1 else f"Photo of notes (part {i})")
              for i, p in enumerate(pieces, start=1)]
    store.replace_chunks(material["id"], chunks)
    store.update_material(material["id"], status="ready", status_note="")
    store.update_job(job["id"], status="done", progress=1.0, note="Text read from photo")
    return store.job(job["id"])  # type: ignore[return-value]


def run_job(store: Store, job_id: int, *, whisper_factory=None, provider=None) -> dict[str, Any]:
    """Run one queued job. Returns the job row. Never raises for expected problems: the job is marked error."""
    job = store.job(job_id)
    if not job:
        raise KeyError(job_id)
    if job["status"] not in ("queued", "running"):
        return job
    material = store.material(job["material_id"]) if job.get("material_id") else None
    if job["kind"] not in ("transcribe", "ocr"):
        return _fail(store, job, f"I don't know how to run a '{job['kind']}' job.")
    if not material:
        return _fail(store, job, "The material for this job was deleted.")
    store.update_job(job_id, status="running", progress=0.0)
    try:
        if job["kind"] == "transcribe":
            return _transcribe(store, job, material, whisper_factory)
        return _ocr(store, job, material, provider)
    except Exception as exc:  # noqa: BLE001 - a background job must not crash the app
        return _fail(store, job, f"This job failed: {str(exc)[:200] or type(exc).__name__}. You can try again, or "
                                 "add the text or a captions file yourself.", material_error=True)


def run_pending(store: Store, *, whisper_factory=None, provider=None) -> list[dict[str, Any]]:
    """Run every queued transcribe and OCR job, oldest first."""
    out = []
    for job in store.jobs("queued"):
        if job["kind"] in ("transcribe", "ocr"):
            out.append(run_job(store, job["id"], whisper_factory=whisper_factory, provider=provider))
    return out
