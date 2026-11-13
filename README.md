# Course Assistant

A free, open source study app for college students. It runs on your own computer, saves your course materials, and
helps you study from those materials only. Every answer points to the exact page, slide, timestamp or section it
came from, and anything that does not come from your materials is labeled as such.

Course Assistant is built for STEM courses first (math, computer science, science). Other subjects work too.

## Disclaimer

Course Assistant is a study aid. It will make mistakes from time to time, so check anything important against your course materials and your instructor. It is here to help you study, not to replace your own notes, organization or judgment. The author takes no responsibility for errors in its answers or for how they are used.

The app shows this on first run with an "I understand" button, and a short reminder sits under every AI answer.

## What it does

**Save your materials.** Lecture slides and handouts (PDF, PowerPoint), textbooks (PDF, EPUB), your own notes
(Markdown, Word, PDF, photos of handwritten notes), lecture recordings (transcribed on your computer, then cited by
timestamp), course website snapshots you can read offline, and your syllabus. Exam and assignment dates in the syllabus
go into a course calendar.

**Past offerings.** Paste links to earlier runs of the same course at your university and the app saves them. It can
also search for them and show you what it found, and it only saves the pages you pick. It follows each site's
robots.txt and goes slowly, a few pages a minute per site. It never signs in anywhere for you. Robots.txt is a
practical check, not a reading of a site's terms: **following each site's terms of use is on you.** Saved pages stay on
your computer.

**Study tools.** A primer before a lecture, a guide before an exam, a concept map, an explanation at the depth you
choose (quick, standard, from scratch), practice questions, flashcards with spaced repetition, and "what did I miss"
for a lecture you skipped.

**Grounding and citations.** Answers cite the exact spot in your materials, every answer lists the sources it used,
and anything the materials do not cover is labeled "Not from your materials".

**Graded work.** You can mark an assignment or exam as graded. For graded work the app never solves your actual
problem. It writes a twin problem with different numbers, solves the twin step by step with citations, and leaves the
original for you. When there are no numbers to change (an essay, a proof), it explains the concepts and how to approach
the task, with citations, and never writes the piece for you.

## Install

macOS (Apple Silicon) and Linux (x86_64):

```sh
curl -fsSL https://raw.githubusercontent.com/RogerFlores3113/course-assistant/main/install.sh | sh
```

Windows 10 or 11, in PowerShell:

```powershell
irm https://raw.githubusercontent.com/RogerFlores3113/course-assistant/main/install.ps1 | iex
```

The script downloads the latest release from GitHub, checks it against `SHA256SUMS.txt`, installs it and opens the app
in your browser. Your data lives in its own folder and is never touched by an update:

- macOS: `~/Library/Application Support/Course Assistant`
- Windows: `%LOCALAPPDATA%\Course Assistant`
- Linux: `~/.local/share/Course Assistant`

An AI assistant can do the install for you by following [INSTALL.md](INSTALL.md).

From source (Python 3.11 or newer and [uv](https://docs.astral.sh/uv/)):

```sh
uv sync --all-groups
uv run course-assistant
```

## Use it with Claude Desktop or Codex

Open the Connect page in the app, or run:

```sh
course-assistant connect            # Claude Desktop and Codex
course-assistant connect claude
course-assistant connect codex
```

This adds a `course-assistant` MCP server to Claude Desktop's `claude_desktop_config.json` and to Codex's
`~/.codex/config.toml`, keeping every other server and saving a backup first, and installs the Course Assistant skills
for Claude Code and Codex. Restart the AI app afterwards. Your AI app can then search your materials, read passages,
add notes and dates, and save study guides with citations. It cannot delete anything, and it follows the graded-work
rule above.

## Chat inside the app (bring your own key)

The app has its own chat that uses a key you provide: an Anthropic API key, or a local model server such as LM Studio
(default `http://127.0.0.1:1234`). Set it under Settings. Your key stays on your computer.

## Other commands

```sh
course-assistant open                                  # open the running app in your browser
course-assistant import "Calculus I" notes.pdf --create # save files into a course
course-assistant version
```

## Privacy

All data stays on your computer in one SQLite file and a folder of your original files. There is no telemetry and no
account. The app only talks to the network when you ask it to: fetching pages you chose for past offerings, or sending
questions and the matching passages from your materials to the AI service you set up. See [PRIVACY.md](PRIVACY.md).

## License

MIT, with the disclaimer above kept as a notice. See [LICENSE](LICENSE).
