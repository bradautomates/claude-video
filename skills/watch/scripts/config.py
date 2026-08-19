#!/usr/bin/env python3
"""Shared /watch configuration helpers."""
from __future__ import annotations

import os
import platform
import sys
from pathlib import Path


CONFIG_DIR = Path.home() / ".config" / "watch"
CONFIG_FILE = CONFIG_DIR / ".env"

DEFAULT_DETAIL = "balanced"

DETAILS = {"transcript", "efficient", "balanced", "token-burner"}

# Package name to install for each required binary, per platform. ffprobe ships
# inside the ffmpeg package everywhere, so it maps to the same hint.
INSTALL_HINTS: dict[str, dict[str, str]] = {
    "Darwin": {
        "ffmpeg": "brew install ffmpeg",
        "yt-dlp": "brew install yt-dlp",
    },
    "Linux": {
        "ffmpeg": "sudo apt install ffmpeg  (or: sudo dnf install ffmpeg)",
        "yt-dlp": "pipx install yt-dlp  (or: pip install --user yt-dlp)",
    },
    "Windows": {
        "ffmpeg": "winget install Gyan.FFmpeg",
        "yt-dlp": "winget install yt-dlp.yt-dlp  (or: pip install --user yt-dlp)",
    },
}


def force_utf8_stdio() -> None:
    """Make stdout/stderr UTF-8 no matter what the console codepage is.

    On Windows the interpreter picks its stdio encoding from the system
    codepage, and the report text is not ASCII: it carries em dashes (U+2014,
    absent from cp949) and arrows (U+2192, absent from cp1252). Printing one
    raises UnicodeEncodeError and kills the run *after* the download, frame
    extraction, and transcription have already been paid for. The report is
    consumed by an agent reading a pipe, so UTF-8 is what the receiver wants
    regardless of the console. ``errors="replace"`` keeps a hostile
    PYTHONIOENCODING from being fatal too.

    Call once at the top of every entry point, before the first print.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:  # a replaced/wrapped stream, e.g. under pytest
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass


def python_command() -> str:
    """Interpreter name to print in a copy-pasteable hint.

    `python3` does not resolve on Windows — it is the Microsoft Store stub,
    which opens the Store instead of running the script. SKILL.md already tells
    the agent to substitute `python` there; these hints must agree.
    """
    return "python" if os.name == "nt" else "python3"


def install_hint(binary: str, system: str | None = None) -> str:
    """Platform-correct install command for one required binary."""
    package = "ffmpeg" if binary in ("ffmpeg", "ffprobe") else binary
    hints = INSTALL_HINTS.get(system or platform.system(), {})
    return hints.get(package, f"install {package} and put it on your PATH")


def missing_binary_message(binary: str) -> str:
    """The SystemExit text used everywhere a required binary is absent."""
    return f"{binary} is not installed. Install with: {install_hint(binary)}"


def read_env_file(path: Path | None = None) -> dict[str, str]:
    if path is None:
        path = CONFIG_FILE
    values: dict[str, str] = {}
    if not path.exists():
        return values
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    for line in lines:
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, _, value = raw.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] in ('"', "'") and value[-1] == value[0]:
            value = value[1:-1]
        else:
            # Strip an inline comment (a '#' preceded by whitespace) from an
            # unquoted value. Without this, `WATCH_DETAIL=balanced  # note`
            # parses as "balanced  # note", fails validation, and silently
            # falls back to the default. Keeps '#' inside quotes / API keys.
            for i, ch in enumerate(value):
                if ch == "#" and i > 0 and value[i - 1] in " \t":
                    value = value[:i].rstrip()
                    break
        values[key.strip()] = value
    return values


def get_config() -> dict[str, object]:
    file_values = read_env_file()

    detail = (
        os.environ.get("WATCH_DETAIL")
        or file_values.get("WATCH_DETAIL")
        or DEFAULT_DETAIL
    )
    if detail not in DETAILS:
        detail = DEFAULT_DETAIL

    return {
        "detail": detail,
        "config_file": str(CONFIG_FILE),
    }


def frame_cap(detail: str) -> int | None:
    if detail == "efficient":
        return 50
    if detail == "balanced":
        return 100
    if detail == "token-burner":
        return None
    if detail == "transcript":
        return None
    return 100
