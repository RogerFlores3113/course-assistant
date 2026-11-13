# Installing Course Assistant (instructions for an AI assistant)

A person asked you to install Course Assistant from this repository. You are probably
Claude in the Claude desktop app (Code tab or Cowork), Claude Code or Codex, running on
their own computer. Follow these steps exactly. Talk to the person in plain, friendly
words; they may not be technical.

## 0. Check where you are running

Make sure you are on the person's own computer and not in a cloud sandbox. If you are
in a sandbox (for example Claude's Cowork or a cloud container), stop. Installing there
would not put Course Assistant on their computer. Give the person the one-line command for
their system (see step 2) and ask them to run it themselves in a terminal (Mac, Linux)
or PowerShell (Windows).

## 1. Detect the operating system

- macOS on Apple Silicon (`uname -m` prints `arm64`): use the shell script.
- Linux x86_64 (`uname -m` prints `x86_64`): use the shell script.
- Windows 10 or 11, 64-bit: use the PowerShell script.
- Intel Mac or anything else: stop and go to "If it fails".

## 2. Run the installer

Run exactly one script, the one for this operating system, then stop. Do not run both,
and do not run it again unless it failed.

macOS or Linux, from any folder:

```sh
curl -fsSL https://raw.githubusercontent.com/RogerFlores3113/course-assistant/main/install.sh | sh
```

Windows, in PowerShell:

```powershell
irm https://raw.githubusercontent.com/RogerFlores3113/course-assistant/main/install.ps1 | iex
```

If this repository is already cloned on the computer, run the local script instead:
`sh install.sh` (macOS, Linux) or `powershell -ExecutionPolicy Bypass -File install.ps1`
(Windows; this only applies to this one script run).

The script downloads the latest release for this computer, checks it against
`SHA256SUMS.txt`, installs it, and opens Course Assistant. Add `--no-open` (`-NoOpen` on
Windows) only if the person does not want the app to open.

Where it installs:

- macOS: `/Applications/Course Assistant.app`, or `~/Applications` if `/Applications` is not writable
- Windows: `%LOCALAPPDATA%\Programs\Course Assistant`
- Linux: `~/.local/share/course-assistant-app/Course Assistant`, with a `course-assistant` command in `~/.local/bin`

If Course Assistant is already installed, the script closes it and replaces it. The
person's courses and settings are stored in a separate folder and are kept.

## 3. What success looks like

Confirm the app opened: the installer prints that Course Assistant should be open in the
browser, or the person sees a browser tab at `http://127.0.0.1:8766`. If that page says
it is locked, tell the person to open Course Assistant from its app icon (Applications on
Mac, Start menu on Windows, applications menu on Linux) instead.

The script prints "Download verified", then "Installed to ...", then a closing message,
and exits with status 0. Course Assistant opens in the person's web browser at
`http://127.0.0.1:8766`.

Then tell the person, in plain words: "Course Assistant is installed and open in your
browser. Your first screen shows a short disclaimer; press **I understand** to continue.
To use Claude Desktop or Codex with your courses, open the **Connect** page in the app."
The Connect page links the app to Claude Desktop and Codex and adds the Course Assistant
skills for Claude Code and Codex. You do not need to edit any configuration files
yourself. The same thing from a terminal is `course-assistant connect` (the command is
inside the installed app folder on Windows, and on the path on Linux and from source).

Before they connect, warn them: Claude Desktop must be quit completely and opened again
to load the new server, so this chat may close, and they can come back to it afterwards.
Save anything they are typing in Claude first.

If the installer ends with "Already connected. Restart Claude Desktop to use the new
version.", Course Assistant was already linked to Claude Desktop. Tell them to restart
Claude Desktop and skip Connect.

If Claude Desktop is not installed: Connect still works for Codex and Claude Code skills.
If the person wants Claude Desktop, they should install it from claude.ai/download first,
open it once, then connect again.

## About the checksum

The checksum file comes from the same release as the download. It catches a broken or
partial download, but it would not catch a tampered release.

## If it fails

Stop and tell the person what happened, in one or two plain sentences. Do not try to
work around it.

- Intel Mac: only Apple Silicon builds exist. Say so and stop.
- No network, or a download error: say so, and suggest they check their connection and ask again.
- Checksum mismatch: the file was not installed on purpose. Report it and stop. Never
  skip or edit the check, and never install a file that failed it.
- Anything else: show the script's last message and stop.

## Do not

- Do not change Gatekeeper, SmartScreen, antivirus, firewall or any other security setting,
  and do not disable any security feature, even if something asks you to.
- Do not remove macOS quarantine flags or run the installer with elevated permissions
  (no `sudo`, no "Run as administrator").
- Do not download the app from anywhere other than this repository's GitHub releases.
- Do not delete the person's Course Assistant data folder.
