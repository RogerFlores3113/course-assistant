import zipfile
from pathlib import Path

import pytest

from course_assistant import importers
from course_assistant.importers import (ImportError_, SUPPORTED_EXTENSIONS, detect_kind, import_file, import_text,
                                        import_url, reimport)
from course_assistant.store import Store


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("COURSE_ASSISTANT_DATA_DIR", str(tmp_path / "data"))
    s = Store(":memory:")
    cid = s.add_course("Linear Algebra", term="Fall 2026")
    return s, cid, tmp_path


# ------------------------------------------------------------------ fixture builders
def make_pdf(path: Path, pages: list[str]) -> Path:
    """A minimal valid multi-page text PDF written by hand. An empty string makes a page with no text."""
    objs: list[bytes] = []
    n = len(pages)
    objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(n))
    objs.append(f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode())
    font_id = 3 + 2 * n
    for i, text in enumerate(pages):
        content_id = 4 + 2 * i
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content_id} 0 R "
                    f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>".encode())
        lines = text.split("\n") if text else []
        ops = "BT /F1 12 Tf 14 TL 50 740 Td " + " ".join(
            f"({ln.replace(chr(92), chr(92) * 2).replace('(', chr(92) + '(').replace(')', chr(92) + ')')}) Tj T*"
            for ln in lines) + " ET" if lines else ""
        stream = ops.encode()
        objs.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, o in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + o + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n".encode() + b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))
    return path


def make_pptx(path: Path) -> Path:
    from pptx import Presentation

    prs = Presentation()
    s1 = prs.slides.add_slide(prs.slide_layouts[1])
    s1.shapes.title.text = "Eigenvalues"
    s1.placeholders[1].text = "An eigenvector keeps its direction under A."
    s1.notes_slide.notes_text_frame.text = "Mention the characteristic polynomial."
    s2 = prs.slides.add_slide(prs.slide_layouts[1])
    s2.shapes.title.text = "Diagonalization"
    s2.placeholders[1].text = "A = P D P inverse."
    s3 = prs.slides.add_slide(prs.slide_layouts[6])  # blank, no text
    prs.save(str(path))
    return path


def make_docx(path: Path) -> Path:
    import docx

    d = docx.Document()
    d.add_paragraph("Intro line before any heading.")
    d.add_heading("Vector spaces", level=1)
    d.add_paragraph("A vector space is a set closed under addition and scaling. " * 6)
    d.add_heading("Subspaces", level=2)
    d.add_paragraph("A subspace contains zero and is closed under both operations. " * 6)
    t = d.add_table(rows=1, cols=2)
    t.rows[0].cells[0].text = "Span"
    t.rows[0].cells[1].text = "All linear combinations"
    d.save(str(path))
    return path


def make_epub(path: Path) -> Path:
    ch = ('<?xml version="1.0"?><html xmlns="http://www.w3.org/1999/xhtml"><head><title>{t}</title></head><body>'
          "<h1>{t}</h1><p>{p}</p><h2>{t} part two</h2><p>{p}</p></body></html>")
    para = "Words about the topic that are long enough to be a proper section of a book. " * 5
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("mimetype", "application/epub+zip")
        z.writestr("META-INF/container.xml",
                   '<?xml version="1.0"?><container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" '
                   'version="1.0"><rootfiles><rootfile full-path="OEBPS/content.opf" '
                   'media-type="application/oebps-package+xml"/></rootfiles></container>')
        z.writestr("OEBPS/content.opf",
                   '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="3.0">'
                   '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Little Linear Algebra</dc:title>'
                   '</metadata><manifest><item id="c1" href="ch1.xhtml" media-type="application/xhtml+xml"/>'
                   '<item id="c2" href="text/ch2.xhtml" media-type="application/xhtml+xml"/></manifest>'
                   '<spine><itemref idref="c1"/><itemref idref="c2"/></spine></package>')
        z.writestr("OEBPS/ch1.xhtml", ch.format(t="Matrices", p=para))
        z.writestr("OEBPS/text/ch2.xhtml", ch.format(t="Determinants", p=para))
    return path


VTT = """WEBVTT

00:00:01.000 --> 00:00:05.000
Welcome to lecture one.

00:00:05.000 --> 00:00:40.000
Today we cover <v Prof>row reduction.

00:01:35.000 --> 00:01:50.000
Next, pivots and free variables.

00:03:10.500 --> 00:03:20.000
Finally an example.
"""
SRT = """1
00:00:00,500 --> 00:00:04,000
First line

2
00:02:00,000 --> 00:02:04,000
Second line
"""


# ------------------------------------------------------------------ PDF
def test_pdf_chunk_per_page(env):
    s, cid, tmp = env
    pdf = make_pdf(tmp / "lecture 1 slides.pdf", ["Gaussian elimination basics", "Eigenvalues and eigenvectors"])
    mid = import_file(s, cid, pdf)
    m = s.material(mid)
    assert m["status"] == "ready" and m["page_count"] == 2 and m["mime"] == "application/pdf"
    assert m["kind"] == "slides" and m["title"] == "lecture 1 slides"
    ch = s.chunks_of(mid)
    assert [(c["loc_kind"], c["loc_start"], c["loc_label"]) for c in ch] == [("page", "1", "p. 1"), ("page", "2", "p. 2")]
    assert "Eigenvalues" in ch[1]["text"]
    assert (Path(s.material(mid)["filename"]).parts[0]) == str(cid)


def test_pdf_long_page_splits_with_same_loc(env):
    s, cid, tmp = env
    lines = [f"Sentence number {i} says something about matrices and vectors." for i in range(120)]
    mid = import_file(s, cid, make_pdf(tmp / "book.pdf", ["\n".join(lines), "short page text here"]))
    ch = s.chunks_of(mid)
    assert len([c for c in ch if c["loc_start"] == "1"]) >= 2
    assert all(len(c["text"].split()) <= 350 for c in ch)
    assert ch[-1]["loc_start"] == "2"


def test_scanned_pdf_error(env):
    s, cid, tmp = env
    with pytest.raises(ImportError_) as e:
        import_file(s, cid, make_pdf(tmp / "scan.pdf", ["", ""]))
    assert "scanned" in str(e.value) and "photo" in str(e.value).lower()
    assert s.materials(cid) == []


def test_corrupt_pdf_error(env):
    s, cid, tmp = env
    bad = tmp / "bad.pdf"
    bad.write_bytes(b"%PDF-1.4 this is not really a pdf")
    with pytest.raises(ImportError_):
        import_file(s, cid, bad)


def test_dedupe_by_sha(env):
    s, cid, tmp = env
    pdf = make_pdf(tmp / "a.pdf", ["Some text on a page that is long enough"])
    first = import_file(s, cid, pdf)
    copy = tmp / "renamed copy.pdf"
    copy.write_bytes(pdf.read_bytes())
    assert import_file(s, cid, copy) == first
    assert len(s.materials(cid)) == 1
    other = s.add_course("Other")
    assert import_file(s, other, copy) != first  # dedupe is per course


# ------------------------------------------------------------------ PPTX / DOCX / EPUB
def test_pptx_slides_with_notes(env):
    s, cid, tmp = env
    mid = import_file(s, cid, make_pptx(tmp / "lec2.pptx"))
    ch = s.chunks_of(mid)
    assert [c["loc_label"] for c in ch] == ["slide 1", "slide 2"]  # blank slide skipped
    assert "Mention the characteristic polynomial" in ch[0]["text"]
    assert ch[0]["heading"] == "Eigenvalues"
    assert s.material(mid)["kind"] == "slides" and s.material(mid)["page_count"] == 3


def test_docx_by_heading(env):
    s, cid, tmp = env
    mid = import_file(s, cid, make_docx(tmp / "my notes.docx"))
    ch = s.chunks_of(mid)
    heads = [c["heading"] for c in ch]
    assert "Vector spaces" in heads and "Subspaces" in heads
    assert all(c["loc_kind"] == "section" for c in ch)
    sub = next(c for c in ch if c["heading"] == "Subspaces")
    assert "All linear combinations" in sub["text"] and sub["loc_label"] == "Subspaces"
    assert s.material(mid)["kind"] == "notes"


def test_epub_chapters(env):
    s, cid, tmp = env
    mid = import_file(s, cid, make_epub(tmp / "book.epub"))
    m = s.material(mid)
    assert m["kind"] == "textbook" and m["title"] == "Little Linear Algebra"
    ch = s.chunks_of(mid)
    assert [c["heading"] for c in ch] == ["Matrices", "Matrices part two", "Determinants", "Determinants part two"]
    assert ch[0]["loc_start"].startswith("ch1") and ch[2]["loc_start"].startswith("ch2")
    assert len({c["loc_start"] for c in ch}) == 4


def test_bad_epub(env):
    s, cid, tmp = env
    f = tmp / "x.epub"
    f.write_bytes(b"not a zip")
    with pytest.raises(ImportError_):
        import_file(s, cid, f)


# ------------------------------------------------------------------ text, markdown, html
def test_markdown_by_heading(env):
    s, cid, tmp = env
    body = "word " * 40
    md = tmp / "notes.md"
    md.write_text(f"# Rank\n{body}\n\n```\n# not a heading\n```\n\n## Nullity\n{body}\n", encoding="utf-8")
    mid = import_file(s, cid, md)
    ch = s.chunks_of(mid)
    assert [c["heading"] for c in ch] == ["Rank", "Nullity"]
    assert "# not a heading" in ch[0]["text"]
    assert s.material(mid)["title"] == "Rank"


def test_plain_text_line_ranges(env):
    s, cid, tmp = env
    txt = tmp / "raw.txt"
    txt.write_text("\n".join(f"line {i} has a few words in it for testing" for i in range(1, 121)), encoding="utf-8")
    mid = import_file(s, cid, txt)
    ch = s.chunks_of(mid)
    assert len(ch) >= 2
    assert all(c["loc_kind"] == "line" for c in ch)
    assert ch[0]["loc_start"] == "1" and ch[0]["loc_label"] == "line 1"
    assert int(ch[1]["loc_start"]) == int(ch[0]["loc_end"]) + 1


def test_empty_text_error(env):
    s, cid, tmp = env
    f = tmp / "e.txt"
    f.write_text("   \n", encoding="utf-8")
    with pytest.raises(ImportError_):
        import_file(s, cid, f)


def test_html_snapshot_cleaned_and_saved(env):
    s, cid, tmp = env
    page = tmp / "course.html"
    page.write_text("""<html><head><title>CS 101 Home</title><style>p{color:red}</style>
      <script>alert('x')</script></head><body onload="evil()"><nav>menu menu</nav>
      <h1>Welcome</h1><p>This course covers sorting and searching in depth, with many examples. """ + "text " * 40 +
                    """</p><script>steal()</script><h2 id="hw">Homework</h2><p>""" + "more " * 40 +
                    """ <a href="javascript:bad()">x</a> <a href="/rel">ok</a></p><form><input name=q></form>
      </body></html>""", encoding="utf-8")
    mid = import_file(s, cid, page, kind="website")
    m = s.material(mid)
    assert m["title"] == "CS 101 Home" and m["filename"].endswith(".html")
    from course_assistant import paths
    cleaned = (paths.files_dir() / m["filename"]).read_text(encoding="utf-8")
    assert "script" not in cleaned.lower() and "onload" not in cleaned and "<form" not in cleaned
    assert "javascript:" not in cleaned and "<style" not in cleaned
    assert '<h1 id="welcome">' in cleaned and '<h2 id="hw">' in cleaned
    ch = s.chunks_of(mid)
    assert [c["heading"] for c in ch] == ["Welcome", "Homework"]
    assert ch[1]["loc_start"] == "hw"
    assert "steal" not in " ".join(c["text"] for c in ch)


# ------------------------------------------------------------------ captions and media
def test_vtt_time_chunks(env):
    s, cid, tmp = env
    f = tmp / "lecture3.vtt"
    f.write_text(VTT, encoding="utf-8")
    mid = import_file(s, cid, f)
    ch = s.chunks_of(mid)
    assert [(c["loc_kind"], c["loc_start"], c["loc_label"]) for c in ch] == [
        ("time", "1", "0:01"), ("time", "95", "1:35"), ("time", "190.5", "3:10")]
    assert "row reduction" in ch[0]["text"] and "<v" not in ch[0]["text"]
    m = s.material(mid)
    assert m["kind"] == "recording" and m["duration_seconds"] == 200.0


def test_srt_chunks(env):
    s, cid, tmp = env
    f = tmp / "talk.srt"
    f.write_text(SRT, encoding="utf-8")
    ch = s.chunks_of(import_file(s, cid, f))
    assert [c["loc_start"] for c in ch] == ["0.5", "120"]


def test_audio_needs_transcription_and_job(env):
    s, cid, tmp = env
    f = tmp / "Lecture 5.mp3"
    f.write_bytes(b"ID3" + b"\x00" * 200)
    mid = import_file(s, cid, f, lecture_date="2026-09-15")
    m = s.material(mid)
    assert m["status"] == "needs_transcription" and m["kind"] == "recording" and m["lecture_date"] == "2026-09-15"
    jobs = s.jobs("queued")
    assert len(jobs) == 1 and jobs[0]["kind"] == "transcribe" and jobs[0]["material_id"] == mid
    assert s.chunks_of(mid) == []


def test_media_with_captions_uses_them(env):
    s, cid, tmp = env
    f = tmp / "rec.mp4"
    f.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 100)
    cap = tmp / "rec.vtt"
    cap.write_text(VTT, encoding="utf-8")
    mid = import_file(s, cid, f, captions=cap)
    assert s.material(mid)["status"] == "ready"
    assert len(s.chunks_of(mid)) == 3
    assert s.jobs("queued") == []
    # reimport reads the saved captions beside the media file
    s.replace_chunks(mid, [{"text": "stale", "loc_kind": "time", "loc_start": 0, "loc_label": "0:00"}])
    reimport(s, mid)
    assert len(s.chunks_of(mid)) == 3


def test_image_needs_ocr_job(env):
    s, cid, tmp = env
    f = tmp / "whiteboard.jpg"
    f.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 50)
    mid = import_file(s, cid, f)
    m = s.material(mid)
    assert m["status"] == "needs_transcription" and m["kind"] == "notes"
    assert [j["kind"] for j in s.jobs("queued")] == ["ocr"]


# ------------------------------------------------------------------ errors
def test_unsupported_and_missing_and_too_big(env, monkeypatch):
    s, cid, tmp = env
    f = tmp / "thing.xyz"
    f.write_text("hi")
    with pytest.raises(ImportError_) as e:
        import_file(s, cid, f)
    assert "can't read" in str(e.value) and "pdf" in str(e.value)
    with pytest.raises(ImportError_):
        import_file(s, cid, tmp / "nope.pdf")
    big = make_pdf(tmp / "big.pdf", ["text on a page here"])
    monkeypatch.setattr("course_assistant.importers.MAX_FILE_BYTES", 10)
    with pytest.raises(ImportError_) as e:
        import_file(s, cid, big)
    assert "500 MB" in str(e.value)
    assert "—" not in str(e.value)


def test_supported_extensions_cover_design():
    for ext in (".pdf", ".pptx", ".docx", ".epub", ".md", ".txt", ".html", ".htm", ".vtt", ".srt", ".mp3", ".m4a",
                ".wav", ".mp4", ".mov", ".webm", ".png", ".jpg", ".jpeg", ".heic"):
        assert ext in SUPPORTED_EXTENSIONS


# ------------------------------------------------------------------ detect_kind
@pytest.mark.parametrize("name,sample,kind", [
    ("Syllabus.pdf", "", "syllabus"),
    ("CS101_Course_Outline.docx", "", "syllabus"),
    ("hw3.pdf", "", "assignment"),
    ("Problem Set 4.pdf", "", "assignment"),
    ("assignment_2_solutions.pdf", "", "assignment"),
    ("Lecture 04 - Eigenvalues.pdf", "", "slides"),
    ("week3_slides.pptx", "", "slides"),
    ("Chapter 5.pdf", "", "textbook"),
    ("strang.epub", "", "textbook"),
    ("my notes week 2.md", "", "notes"),
    ("lecture7.mp3", "", "recording"),
    ("IMG_2231.jpg", "", "notes"),
    ("handout-matrices.pdf", "", "handout"),
    ("scan001.pdf", "Course info. Office hours Tue 3pm. Grading: exams 60%. Midterm on Oct 9.", "syllabus"),
    ("doc.pdf", "Chapter 3\nLinear maps", "textbook"),
    ("doc.pdf", "Lecture 12: Markov chains", "slides"),
    ("doc.pdf", "Homework 5. Due Friday. Problem set on rank.", "assignment"),
    ("random.pdf", "nothing special here", "other"),
])
def test_detect_kind(name, sample, kind):
    assert detect_kind(name, sample) == kind


# ------------------------------------------------------------------ syllabus events
SYLLABUS_TEXT = """# MATH 201 Syllabus
Office hours: Tuesdays 3-4pm. Grading: homework 30%, exams 70%.

## Schedule
Midterm 1: Thursday, Oct 9
Final Exam: December 12, 2026
Sep 15 | Lecture 3 | Gaussian elimination
"""


def test_syllabus_events_added_and_replaced(env):
    s, cid, tmp = env
    f = tmp / "syllabus.md"
    f.write_text(SYLLABUS_TEXT, encoding="utf-8")
    mid = import_file(s, cid, f)
    assert s.material(mid)["kind"] == "syllabus"
    ev = s.events(cid)
    assert [(e["date"], e["kind"], e["added_by"], e["source_material_id"]) for e in ev] == [
        ("2026-09-15", "lecture", "syllabus", mid), ("2026-10-09", "exam", "syllabus", mid),
        ("2026-12-12", "exam", "syllabus", mid)]  # year comes from the course term
    mine = s.add_event(cid, "other", "My own reminder", "2026-10-01")  # student events survive a re-read
    reimport(s, mid)
    ev = s.events(cid)
    assert len(ev) == 4 and any(e["id"] == mine for e in ev)


def test_import_text_syllabus_and_dedupe(env):
    s, cid, _ = env
    mid = import_text(s, cid, "Pasted syllabus", SYLLABUS_TEXT, kind="syllabus")
    assert len(s.events(cid)) == 3
    assert import_text(s, cid, "again", SYLLABUS_TEXT, kind="syllabus") == mid
    n = import_text(s, cid, "Notes", "# Hello\n" + "word " * 30, added_by="ai_app")
    assert s.material(n)["added_by"] == "ai_app" and s.material(n)["filename"] == ""
    with pytest.raises(ImportError_):
        import_text(s, cid, "x", "   ")


def test_reimport_pdf_refreshes_chunks(env):
    s, cid, tmp = env
    mid = import_file(s, cid, make_pdf(tmp / "x.pdf", ["first page text", "second page text"]))
    s.replace_chunks(mid, [{"text": "junk", "loc_kind": "page", "loc_start": 1, "loc_label": "p. 1"}])
    reimport(s, mid)
    assert len(s.chunks_of(mid)) == 2


# ------------------------------------------------------------------ import_url
class FakeResult:
    def __init__(self, url, ctype, content):
        self.url = self.final_url = url
        self.status = 200
        self.content_type = ctype
        self.content = content
        self.text = content.decode("utf-8", "replace") if "pdf" not in ctype else ""


class FakeFetcher:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def get(self, url):
        self.calls.append(url)
        return self.result


def test_import_url_html_snapshot(env):
    s, cid, _ = env
    html = ("<html><head><title>Past Offering</title></head><body><h1>Syllabus 2023</h1><p>" + "info " * 50 +
            "</p><a href='/hw'>hw</a><script>x()</script></body></html>").encode()
    f = FakeFetcher(FakeResult("https://cs.example.edu/c/2023/", "text/html; charset=utf-8".split(";")[0], html))
    mid = import_url(s, cid, "https://cs.example.edu/c/2023/", kind="past_offering", fetcher=f)
    m = s.material(mid)
    assert m["kind"] == "past_offering" and m["source_url"] == "https://cs.example.edu/c/2023/"
    assert m["title"] == "Past Offering" and m["filename"].endswith(".html")
    from course_assistant import paths
    cleaned = (paths.files_dir() / m["filename"]).read_text()
    assert "https://cs.example.edu/hw" in cleaned and "<script" not in cleaned
    assert import_url(s, cid, "https://cs.example.edu/c/2023/", kind="past_offering", fetcher=f) == mid


def test_import_url_pdf_and_errors(env):
    s, cid, tmp = env
    pdf = make_pdf(tmp / "t.pdf", ["Lecture notes text on page one"]).read_bytes()
    f = FakeFetcher(FakeResult("https://x.edu/files/hw3.pdf", "application/pdf", pdf))
    mid = import_url(s, cid, "https://x.edu/files/hw3.pdf", fetcher=f)
    m = s.material(mid)
    assert m["mime"] == "application/pdf" and m["kind"] == "assignment" and m["source_url"].endswith("hw3.pdf")
    assert len(s.chunks_of(mid)) == 1

    f2 = FakeFetcher(FakeResult("https://x.edu/a.zip", "application/zip", b"PK"))
    with pytest.raises(ImportError_):
        import_url(s, cid, "https://x.edu/a.zip", fetcher=f2)

    from course_assistant.fetcher import FetchBlocked

    class Blocked:
        def get(self, url):
            raise FetchBlocked("robots says no")

    with pytest.raises(ImportError_, match="robots says no"):
        import_url(s, cid, "https://x.edu/", fetcher=Blocked())


def test_api_surface():
    for name in ("import_file", "import_text", "import_url", "reimport", "detect_kind", "SUPPORTED_EXTENSIONS",
                 "ImportError_"):
        assert hasattr(importers, name)
