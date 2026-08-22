#!/usr/bin/env python3
"""Shared /watch configuration helpers."""
from __future__ import annotations

import os
from pathlib import Path


CONFIG_DIR = Path.home() / ".config" / "watch"
CONFIG_FILE = CONFIG_DIR / ".env"

DEFAULT_DETAIL = "balanced"

DETAILS = {"transcript", "efficient", "balanced", "token-burner"}

# yt-dlp --sub-langs selector: the video's own original-language track, plus
# English. ".*-orig" is a regex yt-dlp matches against the available tracks, so
# it resolves to "pt-orig", "de-orig", … whatever the video was published in.
#
# Requesting "en.*" alone breaks every non-English video: YouTube has no native
# English track, so it tries to machine-translate one on demand, rate-limits
# (HTTP 429), and the run ends with no captions at all — indistinguishable from
# a video that genuinely has none.
#
# Must stay bounded. "all" pulls YouTube's hundreds of auto-translated tracks
# and stalls the run for minutes; see tests/test_download.py.
DEFAULT_SUBLANGS = ".*-orig,en.*"


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


def sub_langs(file_values: dict[str, str] | None = None) -> str:
    """Resolve the yt-dlp --sub-langs selector.

    Override with WATCH_SUBLANGS (env var or ~/.config/watch/.env) when you
    want a specific set, e.g. "es.*,en.*". Falls back to DEFAULT_SUBLANGS.
    """
    if file_values is None:
        file_values = read_env_file()
    value = (
        os.environ.get("WATCH_SUBLANGS")
        or file_values.get("WATCH_SUBLANGS")
        or DEFAULT_SUBLANGS
    )
    return value.strip() or DEFAULT_SUBLANGS


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
