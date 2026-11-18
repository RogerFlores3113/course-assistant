#!/bin/sh
# Course Assistant installer for macOS (Apple Silicon) and Linux (x86_64).
#
#   curl -fsSL https://raw.githubusercontent.com/RogerFlores3113/course-assistant/main/install.sh | sh
#
# Options:  --no-open   install but do not open the app
# Testing:  COURSE_ASSISTANT_INSTALL_FROM=<folder or base URL holding the release files>
#           COURSE_ASSISTANT_INSTALL_DIR=<folder to install into>
set -eu

BASE_URL="https://github.com/RogerFlores3113/course-assistant/releases/latest/download"
OPEN_APP=1
UPGRADE=0
for arg in "$@"; do
  case "$arg" in
    --no-open) OPEN_APP=0 ;;
    -h|--help)
      echo "Usage: install.sh [--no-open]"
      exit 0
      ;;
    *)
      echo "Unknown option: $arg" >&2
      exit 2
      ;;
  esac
done

say() { printf '%s\n' "$*"; }
fail() {
  printf '\nCourse Assistant was not installed.\n%s\n' "$*" >&2
  exit 1
}

SOURCE="${COURSE_ASSISTANT_INSTALL_FROM:-$BASE_URL}"
OVERRIDE_DIR="${COURSE_ASSISTANT_INSTALL_DIR:-}"

OS="$(uname -s)"
ARCH="$(uname -m)"
case "$OS" in
  Darwin)
    case "$ARCH" in
      arm64) ;;
      *) fail "This Mac has an Intel processor. Course Assistant is only built for Apple Silicon Macs (M1 or later) for now." ;;
    esac
    ASSET="CourseAssistant-mac.zip"
    ;;
  Linux)
    case "$ARCH" in
      x86_64|amd64) ;;
      *) fail "This computer is $ARCH. Course Assistant is only built for Linux x86_64 for now." ;;
    esac
    ASSET="CourseAssistant-linux.tar.gz"
    ;;
  *)
    fail "This script supports macOS and Linux. On Windows, use install.ps1 (see INSTALL.md)."
    ;;
esac

WORK="$(mktemp -d "${TMPDIR:-/tmp}/course-assistant-install.XXXXXX")"
cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT INT TERM

fetch() { # fetch <name> <dest>
  case "$SOURCE" in
    http://*|https://*)
      curl -fsSL "${SOURCE%/}/$1" -o "$2" || return 1
      ;;
    *)
      [ -f "$SOURCE/$1" ] || return 1
      cp "$SOURCE/$1" "$2" || return 1
      ;;
  esac
}

say "Installing Course Assistant..."
say "Downloading $ASSET"
fetch "$ASSET" "$WORK/$ASSET" || fail "Could not download $ASSET. Check your internet connection and try again."
fetch "SHA256SUMS.txt" "$WORK/SHA256SUMS.txt" || fail "Could not download SHA256SUMS.txt, so the download cannot be checked. Check your internet connection and try again."

say "Checking the download"
if command -v shasum >/dev/null 2>&1 && [ "$OS" = "Darwin" ]; then
  ACTUAL="$(shasum -a 256 "$WORK/$ASSET" | awk '{print $1}')"
elif command -v sha256sum >/dev/null 2>&1; then
  ACTUAL="$(sha256sum "$WORK/$ASSET" | awk '{print $1}')"
elif command -v shasum >/dev/null 2>&1; then
  ACTUAL="$(shasum -a 256 "$WORK/$ASSET" | awk '{print $1}')"
else
  fail "No tool found to check the download (sha256sum or shasum)."
fi
EXPECTED="$(awk -v n="$ASSET" '{f=$2; sub(/^\*/, "", f); if (f == n) {print $1; exit}}' "$WORK/SHA256SUMS.txt" | tr 'A-F' 'a-f')"
[ -n "$EXPECTED" ] || fail "SHA256SUMS.txt does not list $ASSET, so the download cannot be checked."
if [ "$ACTUAL" != "$EXPECTED" ]; then
  fail "The download does not match its checksum, so it was not installed. Expected $EXPECTED but got $ACTUAL. Try again later, and tell the Course Assistant maintainer if it keeps happening."
fi
say "Download verified"

mkdir -p "$WORK/unpacked"
if [ "$OS" = "Darwin" ]; then
  ditto -x -k "$WORK/$ASSET" "$WORK/unpacked" || fail "Could not unpack $ASSET."
  [ -d "$WORK/unpacked/Course Assistant.app" ] || fail "The download did not contain Course Assistant.app."
else
  tar -xzf "$WORK/$ASSET" -C "$WORK/unpacked" || fail "Could not unpack $ASSET."
  [ -d "$WORK/unpacked/Course Assistant" ] || fail "The download did not contain the Course Assistant folder."
fi

if [ "$OS" = "Darwin" ]; then
  if [ -n "$OVERRIDE_DIR" ]; then
    TARGET_DIR="$OVERRIDE_DIR"
  elif [ -w /Applications ]; then
    TARGET_DIR="/Applications"
  else
    TARGET_DIR="$HOME/Applications"
  fi
  mkdir -p "$TARGET_DIR" || fail "Could not create $TARGET_DIR."
  APP="$TARGET_DIR/Course Assistant.app"

  if [ -z "$OVERRIDE_DIR" ] && pgrep -f "Course Assistant.app/Contents/MacOS" >/dev/null 2>&1; then
    say "Closing Course Assistant so it can be updated. Your courses and settings are stored elsewhere and are kept."
    osascript -e 'tell application "Course Assistant" to quit' >/dev/null 2>&1 || true
    i=0
    while pgrep -f "Course Assistant.app/Contents/MacOS" >/dev/null 2>&1 && [ "$i" -lt 20 ]; do
      sleep 1
      i=$((i + 1))
    done
    if pgrep -f "Course Assistant.app/Contents/MacOS" >/dev/null 2>&1; then
      fail "Course Assistant is still open. Please quit it (Course Assistant menu, then Quit) and run this again. Your data is safe."
    fi
  fi

  if [ -e "$APP" ]; then
    UPGRADE=1
    say "Replacing the existing copy. Your courses and settings are kept."
    rm -rf "$APP" || fail "Could not remove the old copy at $APP."
  fi
  ditto "$WORK/unpacked/Course Assistant.app" "$APP" || fail "Could not copy the app into $TARGET_DIR."
  chmod +x "$APP/Contents/MacOS/"* 2>/dev/null || true
  say "Installed to $APP"

  if [ "$OPEN_APP" = "1" ]; then
    say "Opening Course Assistant"
    open "$APP" || say "Could not open it automatically. Open Course Assistant from $TARGET_DIR."
  fi
else
  if [ -n "$OVERRIDE_DIR" ]; then
    TARGET_DIR="$OVERRIDE_DIR"
  else
    TARGET_DIR="$HOME/.local/share/course-assistant-app"
  fi
  mkdir -p "$TARGET_DIR" || fail "Could not create $TARGET_DIR."
  APP_DIR="$TARGET_DIR/Course Assistant"
  EXE="$APP_DIR/course-assistant"

  if [ -z "$OVERRIDE_DIR" ] && pgrep -f "$EXE" >/dev/null 2>&1; then
    say "Closing Course Assistant so it can be updated. Your courses and settings are stored elsewhere and are kept."
    pkill -TERM -f "$EXE" >/dev/null 2>&1 || true
    sleep 2
    if pgrep -f "$EXE" >/dev/null 2>&1; then
      fail "Course Assistant is still open. Please close it and run this again. Your data is safe."
    fi
  fi

  if [ -e "$APP_DIR" ]; then
    UPGRADE=1
    say "Replacing the existing copy. Your courses and settings are kept."
    rm -rf "$APP_DIR" || fail "Could not remove the old copy at $APP_DIR."
  fi
  cp -R "$WORK/unpacked/Course Assistant" "$APP_DIR" || fail "Could not copy the app into $TARGET_DIR."
  chmod +x "$EXE" 2>/dev/null || true
  say "Installed to $APP_DIR"

  if [ -z "$OVERRIDE_DIR" ]; then
    mkdir -p "$HOME/.local/bin" "$HOME/.local/share/applications"
    if [ ! -e "$HOME/.local/bin/course-assistant" ] || [ -L "$HOME/.local/bin/course-assistant" ]; then
      ln -sf "$EXE" "$HOME/.local/bin/course-assistant"
    else
      say "Note: $HOME/.local/bin/course-assistant already exists and is not a link, so I left it alone."
    fi
    cat > "$HOME/.local/share/applications/course-assistant.desktop" <<DESKTOP
[Desktop Entry]
Type=Application
Name=Course Assistant
Comment=Study help from your own course materials
Exec="$EXE"
Terminal=false
Categories=Office;
DESKTOP
    say "Added a Course Assistant entry to your applications menu, and the command course-assistant"
  fi

  if [ "$OPEN_APP" = "1" ]; then
    say "Opening Course Assistant"
    nohup "$EXE" >/dev/null 2>&1 &
  fi
fi

say ""
say "Done. Your courses and settings live in their own folder and are never touched by installs or updates."
if [ "$OS" = "Darwin" ]; then
  say "Course Assistant is in your Applications folder (or ~/Applications)."
  CLAUDE_CFG="$HOME/Library/Application Support/Claude/claude_desktop_config.json"
else
  say "Course Assistant is in your applications menu, or run course-assistant."
  CLAUDE_CFG="$HOME/.config/Claude/claude_desktop_config.json"
fi
if [ "$OPEN_APP" = "1" ]; then
  say "Course Assistant should now be open in your web browser."
fi
if [ "$UPGRADE" = "1" ] && [ -f "$CLAUDE_CFG" ] && grep -q "course-assistant" "$CLAUDE_CFG" 2>/dev/null; then
  say "Already connected. Restart Claude Desktop to use the new version."
elif [ "$OPEN_APP" = "1" ]; then
  say "Next: to use Claude Desktop or Codex with your courses, open the Connect page in the app."
else
  say "Next: to use Claude Desktop or Codex with your courses, open Course Assistant and go to its Connect page."
fi
if [ "$OS" = "Darwin" ]; then
  say "Connecting Claude Desktop needs it to be restarted, so save anything you are typing there first."
fi
