#!/usr/bin/env python3
"""Shared /watch configuration helpers."""
from __future__ import annotations

import codecs
import os
import shlex
import shutil
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


# Byte-order marks that identify a .env the platform did not write as UTF-8.
# Longest prefix first so UTF-16LE ('\xff\xfe') cannot shadow a longer match.
_BOM_ENCODINGS = (
    (codecs.BOM_UTF8, "utf-8-sig"),
    (codecs.BOM_UTF16_LE, "utf-16"),
    (codecs.BOM_UTF16_BE, "utf-16"),
)

DEFAULT_DETAIL = "balanced"

DETAILS = {"transcript", "efficient", "balanced", "token-burner"}


def decode_env_bytes(data: bytes, path: Path | None = None) -> str:
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

    Shared by every .env reader in the package (config, setup, whisper) so a
    fix here cannot go stale in one of them; setup's preflight in particular
    runs before this module is even consulted.
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
            "Re-save it as UTF-8 if a setting looks wrong."
            % (path if path is not None else CONFIG_FILE),
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
        lines = decode_env_bytes(path.read_bytes(), path).splitlines()
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


def ytdlp_cmd() -> list[str]:
    """The command that runs yt-dlp, as an argv prefix.

    ``WATCH_YTDLP`` (environment or ~/.config/watch/.env) overrides it: either
    a path to a specific binary — useful when Homebrew's curl_cffi-less copy
    shadows the pipx one — or a command such as ``python -m yt_dlp``. Without
    it, ``yt-dlp`` is resolved on PATH once and run by that path, so the
    binary probed is the binary executed (Windows' CreateProcess otherwise
    only appends .exe and may pick a different copy than shutil.which()).
    """
    override = read_env_value("WATCH_YTDLP")
    if override:
        raw = override.replace("\\", "\\\\") if os.name == "nt" else override
        tokens = shlex.split(raw)
        if tokens:
            return tokens
    return [shutil.which("yt-dlp") or "yt-dlp"]
