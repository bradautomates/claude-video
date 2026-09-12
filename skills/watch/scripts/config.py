#!/usr/bin/env python3
"""Shared /watch configuration helpers."""
from __future__ import annotations

import os
from pathlib import Path


CONFIG_DIR = Path.home() / ".config" / "watch"
CONFIG_FILE = CONFIG_DIR / ".env"

DEFAULT_DETAIL = "balanced"

DETAILS = {"transcript", "efficient", "balanced", "token-burner"}


def _parse_value(value: str) -> str:
    """Parse a single .env value: unwrap quotes, strip inline comments.

    A quoted value (`KEY="balanced"  # note`) may still have a trailing
    comment after the closing quote — that comment must be stripped without
    losing the quote-unwrapping, so quotes are resolved first and the
    remainder after the closing quote is checked for a comment separately.
    An unquoted value strips a '#' preceded by whitespace (keeps '#' that's
    part of the value, e.g. inside an API key).
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
