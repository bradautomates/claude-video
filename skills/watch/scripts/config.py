#!/usr/bin/env python3
"""Shared /watch configuration helpers."""
from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path


CONFIG_DIR = Path.home() / ".config" / "watch"
CONFIG_FILE = CONFIG_DIR / ".env"

DEFAULT_DETAIL = "balanced"

DETAILS = {"transcript", "efficient", "balanced", "token-burner"}



def force_utf8_output() -> None:
    """Make stdout/stderr able to carry the report's non-ASCII characters.

    The markdown report uses em dashes and arrows (``—``, ``→``). On Windows
    the console encoding defaults to cp1252, which cannot encode them, so
    printing the report raises UnicodeEncodeError partway through and the run
    dies after the video has already been downloaded and transcribed.

    Reconfiguring both streams once at entry fixes every print site at once.
    ``errors="replace"`` keeps a genuinely undecodable terminal from crashing
    the run.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:  # not a TextIOWrapper (e.g. captured in tests)
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass


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
        values[key.strip()] = _parse_value(value.strip())
    return values


def _parse_value(value: str) -> str:
    """Parse one .env value: unwrap quotes, strip an inline comment.

    A quoted value may still carry a trailing comment after the closing quote
    (`KEY="balanced"  # note`); resolve the quotes first, then accept the rest
    only if it is empty or a comment. An unquoted value strips a '#' preceded
    by whitespace, keeping '#' that is part of the value (e.g. inside a key).
    """
    if len(value) >= 2 and value[0] in ('"', "'"):
        quote = value[0]
        end = value.find(quote, 1)
        if end != -1:
            remainder = value[end + 1:].strip()
            if not remainder or remainder.startswith("#"):
                return value[1:end]
    for i, ch in enumerate(value):
        if ch == "#" and i > 0 and value[i - 1] in " \t":
            return value[:i].rstrip()
    return value


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
