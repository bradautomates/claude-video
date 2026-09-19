#!/usr/bin/env python3
"""Shared /watch configuration helpers."""
from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path


CONFIG_DIR = Path.home() / ".config" / "watch"
CONFIG_FILE = CONFIG_DIR / ".env"

DEFAULT_DETAIL = "balanced"

DETAILS = {"transcript", "efficient", "balanced", "token-burner"}


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


def env_search_paths() -> list[Path]:
    """Files a setting may live in, highest precedence first.

    Resolved per call rather than at import time so a changed HOME is honoured.
    """
    return [Path.home() / ".config" / "watch" / ".env", Path.cwd() / ".env"]


def read_env_value(
    name: str,
    paths: list[Path] | None = None,
    on_file: Callable[[Path], None] | None = None,
) -> str | None:
    """Resolve one setting: real environment first, then each .env in order.

    The single parser for every consumer. whisper.py and setup.py each used to
    carry their own copy, and both predated the inline-comment handling in
    read_env_file() -- so `GROQ_API_KEY=sk-x  # note` kept the comment as part
    of the key while config.py read the same line correctly.

    ``on_file`` is invoked with each existing path before it is read, so a
    caller can hook in side effects such as a permission warning.
    """
    value = os.environ.get(name)
    if value and value.strip():
        return value.strip()
    for path in (env_search_paths() if paths is None else paths):
        if not path.exists():
            continue
        if on_file is not None:
            on_file(path)
        value = read_env_file(path).get(name)
        if value:
            return value
    return None


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
