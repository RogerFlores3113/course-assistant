# Course Assistant: design and module contracts

Course Assistant is a free, open source, local-first desktop app (Mac, Windows, Linux) that saves a college
student's course materials and helps them study from those materials only. It is a sister app to Talent Search
(github.com/RogerFlores3113/talent-tracker) and follows its shape: Python 3.12, uv, FastAPI + Jinja2 + htmx
served on 127.0.0.1 and opened in the browser, one SQLite file in the per-user data folder, an MCP server for
Claude Desktop / Codex, Claude and Codex skills, PyInstaller builds, install.sh / install.ps1, no telemetry.

## Product decisions (from the decisions doc)

1. Audience: STEM first (math, CS, science); other subjects work but STEM is tuned first.
2. Desktop app, all data on the student's machine.
3. Saves: lecture slides and handouts (PDF, PowerPoint), lecture recordings transcribed locally (Whisper) and cited
   by timestamp, textbooks (PDF, EPUB), the student's own notes (Markdown, Word, PDF, photos of handwritten notes),
   course website snapshots readable offline, the syllabus with exam and assignment dates pulled into a course
   calendar, and past offerings of the same course at the same university.
4. Past offerings: the student pastes links and the app saves them; the app can also search, shows what it found,
   and saves only the pages the student picks. The app obeys robots.txt and goes slowly (a few pages a minute
   per site). There is no blanket no-scraping rule in this app. Terms of service: the app can't read terms, so
   robots.txt is the practical check, saved pages stay on the student's machine, and the README tells students
   that following each site's terms is on them. The app never signs in anywhere on the student's behalf.
5. Study help, all seven: pre-lecture primer, pre-exam guide, concept map, re-explain at a chosen depth (quick,
   standard, from scratch), practice questions, flashcards, "what did I miss" for a skipped lecture.
6. Grounding (required): citations go to the exact spot (page, slide, timestamp, section); every answer shows the
   list of sources it used; when the materials don't cover something, the app MAY answer but must label that part
   clearly as "Not from your materials".
7. Disclaimer (disclaimer.py): on first run with an "I understand" button, a small footer on every AI answer, and
   in the README and LICENSE.
8. Academic integrity: the student marks items as graded. For graded work the app never solves the student's
   actual problem. It writes a twin problem with different numbers, solves the twin step by step with citations,
   and leaves the original for the student. For graded work with no numbers to change (essays, proofs), it explains
   the concepts and how to approach it, with citations, and never writes the piece.
9. AI: MCP server for Claude Desktop, Claude and Codex skills, and a bring-your-own-key chat inside the app
   (Anthropic API key, or a local model server such as LM Studio). Voice later.
10. Views: course list (one card per course), course page (materials, calendar, guides, study tools), a reader that
    opens a cited page or timestamp, and a dashboard of upcoming lectures and exams.
11. Principles: free and open source, local data, no telemetry, the AI app never deletes anything, the AI never
    does graded work for the student.

## Layout

```
src/course_assistant/
  __init__.py        version
  paths.py           data folder, files folder, app key, launch codes, port file        (core)
  store.py           SQLite schema + CRUD for every table                               (core, Opus only)
  locations.py       citation link + location labels                                    (core)
  disclaimer.py      the disclaimer text                                                (core)
  importers/         file and web importers -> materials + chunks (+ syllabus events)   (Builder A)
  fetcher.py         polite fetcher: robots.txt, per-host rate limit, size caps          (Builder A)
  calendar.py        syllabus date extraction                                            (Builder A)
  transcribe.py      background transcription jobs (faster-whisper optional)            (Builder A)
  retrieval.py       FTS5 search over chunks -> Source dicts                             (Builder B)
  providers.py       AI providers: Anthropic SDK, local OpenAI-compatible server, Fake  (Builder B)
  grounding.py       prompt building, citation parsing/validation, "not from materials" (Builder B)
  integrity.py       graded-work detection and twin problems                            (Builder B)
  study.py           primer, exam guide, concept map, explain, practice, flashcards, missed (Builder C)
  web.py, templates/, static/   every page and route                                    (Builder D)
  mcp_server.py      MCP tools for Claude Desktop / Codex                                (Builder E)
  offerings.py       past-offering discovery (paste, crawl, search) -> candidates        (Builder E)
  cli.py, connect.py, packaging/, install.sh, install.ps1, .github/   launch, connect, build, CI  (Builder F)
skills/              Claude / Codex skills                                               (Builder E)
tests/               pytest, one file per module
```

Only edit the files your builder owns. If you need something from store.py that it lacks, write a small helper in
your own module using `store._rows(...)`, `store._row(...)`, `store._exec(...)` and note it in your report.

## Shared contracts

### Store (store.py)
Read it. Key tables: courses, materials, chunks (+ chunks_fts FTS5), events, guides, graded_items, flashcards,
chat_messages, offering_candidates, jobs, ai_actions, settings. `Store(":memory:")` works for tests.

### Chunks and locations
A chunk is `{text, loc_kind, loc_start, loc_label, loc_end?, heading?}` with loc_kind one of page | slide | time |
section | line. Use `locations.label(...)` for loc_label. Aim for chunks of roughly 80 to 350 words; split long pages
into several chunks with the same page. `store.replace_chunks(material_id, chunks)`.
Citation link: `locations.href_for(chunk)` = `/read/<material_id>?at=<chunk_id>#c<chunk_id>`.

### Importers (Builder A): `course_assistant.importers`
```python
import_file(store, course_id: int, src: Path, *, kind: str | None = None, title: str | None = None,
            original_name: str | None = None, lecture_date: str = "", added_by: str = "student",
            captions: Path | None = None) -> int          # material_id; copies src into paths.course_files_dir
import_text(store, course_id, title, text, *, kind="notes", added_by="student") -> int
import_url(store, course_id, url, *, kind="website", fetcher=None, added_by="student") -> int   # offline snapshot
reimport(store, material_id) -> None                   # re-extract chunks from the saved file
detect_kind(filename: str, text_sample: str = "") -> str   # guess slides / syllabus / textbook / ...
SUPPORTED_EXTENSIONS: dict[str, str]                    # ".pdf": "PDF", ...
class ImportError_(Exception)                           # message is shown to the student as is
```
Formats: .pdf (pypdf, chunk per page), .pptx (python-pptx, chunk per slide, speaker notes included), .docx
(python-docx, by heading), .epub (zipfile + html parsing, by chapter/section), .md/.txt (by heading),
.html/.htm (saved snapshot, by heading with anchors), .vtt/.srt captions (time chunks), audio/video
(.mp3 .m4a .wav .mp4 .mov .webm): saved, status needs_transcription and a 'transcribe' job, or captions if given;
images (.png .jpg .jpeg .heic): saved with status needs_transcription ("photo of notes") and an OCR job that uses
the AI provider's vision if configured. Syllabus: after chunking, `calendar.extract_events(text, year_hint)` and
add events with added_by='syllabus' and source_material_id.

### Fetcher (Builder A): `course_assistant.fetcher`
```python
class PoliteFetcher:   # one per process; robots.txt cache; >= 15 s between requests to one host by default
    def __init__(self, user_agent: str = DEFAULT_UA, min_interval: float = 15.0, max_bytes: int = 25_000_000,
                 transport=None)    # httpx transport for tests
    def allowed(self, url: str) -> bool
    def get(self, url: str) -> FetchResult    # raises FetchBlocked (robots), FetchError
```
Never sends cookies or credentials, http(s) only, refuses private/loopback addresses.

### Retrieval (Builder B): `course_assistant.retrieval`
```python
search(store, course_id: int, query: str, *, limit: int = 8, kinds: list[str] | None = None,
       material_ids: list[int] | None = None, exclude_graded: bool = False) -> list[Source]
Source = dict(chunk_id, material_id, material_title, material_kind, loc_kind, loc_start, loc_label, heading, text,
              href, score)
around(store, chunk_id, before=1, after=1) -> list[dict]   # neighbouring chunks for context
```

### Providers (Builder B): `course_assistant.providers`
```python
class Provider:  name: str; model: str
    def complete(self, system: str, messages: list[dict], *, max_tokens: int = 4000) -> str
    def complete_image(self, system: str, image_bytes: bytes, media_type: str, prompt: str) -> str  # optional
get_provider(store) -> Provider | None        # from settings: ai_provider (anthropic | local | none), keys, model
FakeProvider(responses: list[str] | callable)  # tests
```
Anthropic: official `anthropic` SDK, default model `claude-opus-5-5` (settings offer `claude-sonnet-5-5` and
`claude-haiku-5-5`), adaptive thinking, `output_config.effort` "medium", server-side refusal fallbacks
(`client.beta.messages.create(..., betas=["server-side-fallback-2026-07-01"], fallbacks="default")`), check
`stop_reason` for "refusal" before reading content, typed error handling. Local: OpenAI-compatible
`/v1/chat/completions` via httpx at a base URL the student sets (LM Studio default http://127.0.0.1:1234).

### Grounding (Builder B): `course_assistant.grounding`
```python
ask(store, course_id, question, provider, *, depth="standard", history=None) -> Answer   # saves a guide kind 'answer'
build_messages(question, sources, ...) -> (system, messages)
parse_answer(raw: str, sources) -> Answer
Answer = dataclass(body_md, blocks: list[Block], sources: list[Source], used: list[int], outside: bool,
                   twin: bool, guide_id: int | None)
Block = dataclass(kind: 'text'|'outside', md: str, cites: list[int])
```
Rules: sources are numbered [1]..[n]; the model cites with [n]; any [n] not in the list is removed; text the model
marks as outside knowledge (<outside>...</outside>) is shown under a "Not from your materials" label; a text
paragraph with no citation is treated as outside. Every answer carries disclaimer.FOOTER.

### Integrity (Builder B): `course_assistant.integrity`
```python
match_graded(store, course_id, question) -> dict | None   # graded_items or graded materials that the question matches
twin(store, course_id, question, provider, graded: dict) -> Answer   # saves guide kind 'twin'; data has twin problem
```

### Study (Builder C): `course_assistant.study`
Each generator searches the course, asks the provider, validates citations through grounding, saves a guide and
returns its id. Without a provider they raise `NoProvider` (the web shows "connect an AI" and the MCP path
instead).
```python
primer(store, course_id, provider, *, event_id=None) -> int           # next lecture if None
exam_guide(store, course_id, provider, *, event_id=None) -> int       # next exam if None; topics since last exam
concept_map(store, course_id, provider, *, scope: str = "") -> int    # data = {nodes:[{id,label,cites}], edges:[{a,b,label}]}
explain(store, course_id, provider, topic: str, *, depth="standard") -> int   # quick | standard | from scratch
practice(store, course_id, provider, topic: str, *, count=5) -> int   # graded topics -> twin rule
make_flashcards(store, course_id, provider, *, material_id=None, topic="", count=12) -> list[int]
review_flashcard(store, card_id, grade: str) -> dict                  # again | hard | good | easy (SM-2 style)
missed(store, course_id, provider, *, event_id=None, date="") -> int  # summary of one lecture's materials
upcoming(store, days=14) -> list[dict]                                # dashboard rows across courses
```

### Web (Builder D): `course_assistant.web.create_app(store, *, key=None, provider_factory=None) -> FastAPI`
Local-only middleware (Host must be 127.0.0.1/localhost; Origin / Sec-Fetch-Site CSRF checks), app key cookie
unlocked by /launch?code=, security headers with a strict CSP, no CDN (htmx vendored). Pages: / (dashboard +
course cards), /courses/new, /courses/{id} (tabs: Materials, Calendar, Study, Guides, Ask, Past offerings),
/read/{material_id}?at= (reader), /guides/{id}, /flashcards/{course_id}, /settings (AI provider, key, model,
theme), /welcome (first run with disclaimer and "I understand"), /connect (Claude Desktop / Codex connect),
/privacy. Every AI answer: blocks with citation chips linking to the reader, a "Sources" list, the outside label,
and the disclaimer footer. Accessible: semantic landmarks, skip link, labels on every control, visible focus,
WCAG AA contrast, keyboard-only use, aria-live for async results, reduced motion.

### MCP (Builder E): `course_assistant.mcp_server`
FastMCP server over stdio. Tools return citation links as full app URLs (paths.app_url(href)). No delete tools.
Every write goes in store.log_ai. Tools: list_courses, course_overview, search_materials, read_passage,
list_upcoming, add_course, add_material_from_file, add_material_from_url, add_note, add_event, save_guide (body
with [n] citations validated against given chunk ids), add_flashcards, list_graded, mark_graded,
get_twin_instructions, find_past_offerings, save_past_offering_candidate. Instructions tell the AI app to ground
every claim in search results, cite [n] with links, label outside knowledge, and follow the twin rule for graded
work.

### CLI (Builder F): `course-assistant` = `serve` (default: start server, open browser), `open`, `mcp`,
`connect [claude|codex]`, `import <course> <files...>`, `version`.
