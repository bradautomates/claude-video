#!/usr/bin/env python3
"""Shared /watch configuration helpers."""
from __future__ import annotations

import codecs
import os
import sys
from pathlib import Path


CONFIG_DIR = Path.home() / ".config" / "watch"
CONFIG_FILE = CONFIG_DIR / ".env"

# Byte-order marks that identify a .env the platform did not write as UTF-8.
# Longest prefix first so UTF-16LE ('\xff\xfe') cannot shadow a longer match.
_BOM_ENCODINGS = (
    (codecs.BOM_UTF8, "utf-8-sig"),
    (codecs.BOM_UTF16_LE, "utf-16"),
    (codecs.BOM_UTF16_BE, "utf-16"),
)

DEFAULT_DETAIL = "balanced"

DETAILS = {"transcript", "efficient", "balanced", "token-burner"}


def decode_env_bytes(data: bytes) -> str:
    """Decode .env bytes written by any of the encodings Windows produces.

    Reading the file as strict UTF-8 fails four different ways on Windows, and
    only one of them was loud (verified on PowerShell 5.1.26100, Win 11):

      - `... | Out-File .env`      -> UTF-16LE + BOM -> UnicodeDecodeError.
      - `Set-Content .env "...ó"`  -> ANSI code page -> UnicodeDecodeError.
      - UTF-16LE with no BOM       -> decodes to NUL-riddled junk, so every key
                                      silently vanishes.
      - Notepad's "UTF-8 with BOM" -> first key becomes '﻿WATCH_DETAIL'
                                      and is silently ignored.

    UnicodeDecodeError subclasses ValueError, NOT OSError, so the two loud
    cases escaped `except OSError` and took the whole /watch run down instead
    of falling back to defaults. Decoding here is total: the last resort uses
    errors="replace", which cannot raise.
    """
    for bom, encoding in _BOM_ENCODINGS:
        if data.startswith(bom):
            candidate = encoding
            break
    else:
        # No BOM. Interleaved NULs in an otherwise ASCII file mean UTF-16LE;
        # a real .env never contains a NUL byte.
        candidate = "utf-16-le" if b"\x00" in data[:128] else "utf-8"
    try:
        return data.decode(candidate)
    except UnicodeDecodeError:
        # Legacy single-byte code page, or a truncated UTF-16 file. ASCII keys
        # and API keys survive; only the offending bytes become U+FFFD.
        print(
            "watch: %s is not valid UTF-8; some characters were replaced. "
            "Re-save it as UTF-8 if a setting looks wrong." % CONFIG_FILE,
            file=sys.stderr,
        )
        return data.decode("utf-8", errors="replace")


def read_env_file(path: Path | None = None) -> dict[str, str]:
    if path is None:
        path = CONFIG_FILE
    values: dict[str, str] = {}
    if not path.exists():
        return values
    try:
        lines = decode_env_bytes(path.read_bytes()).splitlines()
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
