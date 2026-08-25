#!/usr/bin/env python3
"""Shared /watch configuration helpers."""
from __future__ import annotations

import os
from pathlib import Path


CONFIG_DIR = Path.home() / ".config" / "watch"
CONFIG_FILE = CONFIG_DIR / ".env"

DEFAULT_DETAIL = "balanced"

DETAILS = {"transcript", "efficient", "balanced", "token-burner"}

# Which speech-to-text backend to use when a video has no caption track.
# "auto" resolves at runtime: local if faster-whisper is importable, else Groq,
# else OpenAI. Naming one pins it and turns a missing prerequisite into an error
# instead of a silent fallback.
DEFAULT_WHISPER = "auto"

WHISPER_BACKENDS = {"auto", "local", "groq", "openai"}


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


def _setting(file_values: dict[str, str], name: str, default: str = "") -> str:
    """Env var wins, then the config file, then the default.

    `or` rather than a None-check on purpose: a blank-but-present value (common
    when a key is scaffolded and never filled in) should fall through to the next
    source, not be honoured as an empty setting.
    """
    return (os.environ.get(name) or file_values.get(name) or default).strip()


def get_config() -> dict[str, object]:
    file_values = read_env_file()

    detail = _setting(file_values, "WATCH_DETAIL", DEFAULT_DETAIL)
    if detail not in DETAILS:
        detail = DEFAULT_DETAIL

    whisper = _setting(file_values, "WATCH_WHISPER", DEFAULT_WHISPER).lower()
    if whisper not in WHISPER_BACKENDS:
        whisper = DEFAULT_WHISPER

    return {
        "detail": detail,
        "whisper": whisper,
        # Empty means "let the local backend pick its own default" — validating
        # these here would mean hardcoding a model list that goes stale.
        "whisper_model": _setting(file_values, "WATCH_WHISPER_MODEL"),
        "whisper_device": _setting(file_values, "WATCH_WHISPER_DEVICE"),
        "whisper_compute": _setting(file_values, "WATCH_WHISPER_COMPUTE"),
        "whisper_language": _setting(file_values, "WATCH_WHISPER_LANGUAGE"),
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
