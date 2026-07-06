#!/usr/bin/env python3
"""Setup / preflight for /watch.

Modes:
  setup.py --check      Silent preflight. Exit 0 if ready, 2 on missing binaries.
  setup.py --json       Machine-readable status for Claude to parse.
  setup.py              Installer. Auto-installs deps, scaffolds .env, marks SETUP_COMPLETE.

Design:
- Transcription runs FULLY LOCAL by default (faster-whisper via `uv`), so an API
  key is never required. `uv` on PATH is all that's needed. Cloud Whisper (Groq /
  OpenAI) stays available as an optional, explicit opt-in.
- Silent on success: --check exits 0 with no output when everything's ready so
  that /watch doesn't spam "setup is complete" on every turn.
- Idempotent: re-running the installer is safe — it never clobbers existing
  keys and only appends missing ones.
- SETUP_COMPLETE=true in ~/.config/watch/.env tells us the user has been
  through a successful installer run at least once.
- Never sudo. On macOS, auto-install via brew. Elsewhere, print exact commands.
- Never write an API key to disk automatically — only scaffold placeholders.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from config import get_config  # noqa: E402


# `uv` drives the fully-local faster-whisper transcription backend (the default),
# so it's required alongside the media tools. ffmpeg/ffprobe extract frames +
# audio; yt-dlp downloads URLs.
REQUIRED_BINARIES = ["ffmpeg", "ffprobe", "yt-dlp", "uv"]
CONFIG_DIR = Path.home() / ".config" / "watch"
CONFIG_FILE = CONFIG_DIR / ".env"
ENV_TEMPLATE = """# /watch configuration
#
# Transcription runs FULLY LOCAL by default (faster-whisper via uv) — no API
# key required, no audio ever leaves the machine. The keys below are OPTIONAL:
# set one only if you'd rather force a cloud Whisper backend with --whisper.
#
#   Groq (whisper-large-v3):  https://console.groq.com/keys
#   OpenAI (whisper-1):       https://platform.openai.com/api-keys
#
# Pick the local model with WATCH_WHISPER_MODEL (tiny|base|small|medium|large-v3;
# default base). Bigger = more accurate + slower. Downloaded once, then offline.

WATCH_WHISPER_MODEL=base
GROQ_API_KEY=
OPENAI_API_KEY=

# Default watch behavior (the /watch first-run wizard sets this for you).
# Allowed values: transcript | efficient | balanced | token-burner
# Keep the value on its own line with no trailing comment.
# WATCH_DETAIL=balanced
"""


def _which(name: str) -> str | None:
    return shutil.which(name)


def _check_binaries() -> list[str]:
    return [b for b in REQUIRED_BINARIES if not _which(b)]


_PERM_WARNED: set[str] = set()


def _check_file_permissions(path: Path) -> None:
    """Warn to stderr (once per path per process) if a secrets file is
    world/group readable."""
    key = str(path)
    if key in _PERM_WARNED:
        return
    try:
        mode = path.stat().st_mode
        if mode & 0o044:
            _PERM_WARNED.add(key)
            sys.stderr.write(
                f"[watch] WARNING: {path} is readable by other users. "
                f"Run: chmod 600 {path}\n"
            )
            sys.stderr.flush()
    except OSError:
        pass


def _read_env_key(name: str) -> str | None:
    value = os.environ.get(name)
    if value and value.strip():
        return value.strip()
    if not CONFIG_FILE.exists():
        return None
    _check_file_permissions(CONFIG_FILE)
    try:
        for line in CONFIG_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, raw = line.partition("=")
            if key.strip() != name:
                continue
            raw = raw.strip()
            if len(raw) >= 2 and raw[0] in ('"', "'") and raw[-1] == raw[0]:
                raw = raw[1:-1]
            return raw or None
    except OSError:
        return None
    return None


def _have_api_key() -> tuple[bool, str | None]:
    if _read_env_key("GROQ_API_KEY"):
        return True, "groq"
    if _read_env_key("OPENAI_API_KEY"):
        return True, "openai"
    return False, None


def _have_local_asr() -> bool:
    """True when the fully-local faster-whisper backend can run (`uv` present).

    Overridable via WATCH_LOCAL_ASR (1/0) so tests can pin the capability
    without mutating PATH.
    """
    override = os.environ.get("WATCH_LOCAL_ASR")
    if override is not None:
        return override.strip().lower() not in ("", "0", "false", "no")
    return _which("uv") is not None


def is_first_run() -> bool:
    """True if the installer hasn't completed successfully yet."""
    return _read_env_key("SETUP_COMPLETE") != "true"


def _scaffold_env() -> bool:
    """Create ~/.config/watch/.env with placeholders if missing."""
    if CONFIG_FILE.exists():
        return False
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(ENV_TEMPLATE, encoding="utf-8")
    try:
        CONFIG_FILE.chmod(0o600)
    except OSError:
        pass
    return True


def _write_setup_complete() -> None:
    """Idempotently append SETUP_COMPLETE=true to .env.

    Used after a successful install (binaries present — no key needed, since
    transcription is local). Future sessions detect this marker to stay silent.
    """
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    existing = ""
    if CONFIG_FILE.exists():
        existing = CONFIG_FILE.read_text(encoding="utf-8")
        for line in existing.splitlines():
            if line.strip().startswith("SETUP_COMPLETE="):
                return
        if existing and not existing.endswith("\n"):
            existing += "\n"
        CONFIG_FILE.write_text(existing + "SETUP_COMPLETE=true\n", encoding="utf-8")
    else:
        CONFIG_FILE.write_text(ENV_TEMPLATE + "\nSETUP_COMPLETE=true\n", encoding="utf-8")
    try:
        CONFIG_FILE.chmod(0o600)
    except OSError:
        pass


def _brew_pkg(missing: list[str]) -> list[str]:
    pkgs: list[str] = []
    for bin_name in missing:
        if bin_name in ("ffmpeg", "ffprobe"):
            if "ffmpeg" not in pkgs:
                pkgs.append("ffmpeg")
        elif bin_name == "yt-dlp":
            if "yt-dlp" not in pkgs:
                pkgs.append("yt-dlp")
        else:  # uv, and any future single-name formula
            pkgs.append(bin_name)
    return pkgs


def _install_macos(missing: list[str]) -> tuple[bool, str]:
    if _which("brew") is None:
        return False, (
            "Homebrew is not installed. Install it from https://brew.sh, then re-run setup. "
            "Or install manually: `brew install " + " ".join(_brew_pkg(missing)) + "`"
        )
    pkgs = _brew_pkg(missing)
    if not pkgs:
        return True, "nothing to install"
    cmd = ["brew", "install", *pkgs]
    print(f"[setup] running: {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        return False, f"brew install failed with exit code {result.returncode}"
    return True, f"installed via brew: {', '.join(pkgs)}"


def _install_hint_linux(missing: list[str]) -> str:
    pkgs = _brew_pkg(missing)
    hints = []
    if "ffmpeg" in pkgs:
        hints.append("apt: `sudo apt install ffmpeg` or dnf: `sudo dnf install ffmpeg`")
    if "yt-dlp" in pkgs:
        hints.append("`pipx install yt-dlp` (recommended) or `pip install --user yt-dlp`")
    if "uv" in pkgs:
        hints.append("`curl -LsSf https://astral.sh/uv/install.sh | sh` (see https://astral.sh/uv)")
    return "\n  ".join(hints) if hints else "nothing to install"


def _install_hint_windows(missing: list[str]) -> str:
    pkgs = _brew_pkg(missing)
    hints = []
    if "ffmpeg" in pkgs:
        hints.append("winget: `winget install Gyan.FFmpeg`")
    if "yt-dlp" in pkgs:
        hints.append("winget: `winget install yt-dlp.yt-dlp` or pip: `pip install --user yt-dlp`")
    if "uv" in pkgs:
        hints.append("winget: `winget install astral-sh.uv` or `pip install uv`")
    return "\n  ".join(hints) if hints else "nothing to install"


def _status() -> dict:
    """Structured preflight snapshot.

    Transcription defaults to the fully-local faster-whisper backend, so an API
    key is never required — the media binaries plus `uv` are enough. `status` is
    `ready` once all required binaries are present; a cloud key is purely
    optional (used only when the caller forces --whisper groq|openai).
    """
    missing = _check_binaries()
    has_key, key_backend = _have_api_key()
    local_asr = _have_local_asr()
    setup_complete = not is_first_run()

    status = "ready" if not missing else "needs_install"
    can_proceed = not missing

    # The backend /watch will use by default: local when available, else the
    # configured cloud key (if any).
    default_backend = "local" if local_asr else (key_backend if has_key else None)

    cfg = get_config()
    return {
        "status": status,
        "can_proceed": can_proceed,
        "first_run": not setup_complete,
        "setup_complete": setup_complete,
        "missing_binaries": missing,
        "local_asr": local_asr,
        "whisper_backend": default_backend,
        "has_api_key": has_key,
        "config_file": str(CONFIG_FILE),
        "watch_detail": cfg["detail"],
        "platform": platform.system(),
    }


def cmd_check() -> int:
    """Silent-on-success preflight.

    Exit 0 with no output when /watch can run — transcription is local, so no
    API key is required; only the binaries matter.

    On a state that blocks /watch, print one actionable line to stderr:
      2 → binaries missing (ffmpeg / ffprobe / yt-dlp / uv)
    """
    s = _status()
    if s["can_proceed"]:
        return 0

    installer = Path(__file__).resolve()
    sys.stderr.write(
        f"[watch] setup incomplete (missing binaries: {', '.join(s['missing_binaries'])}). "
        f"Run: python3 {installer}\n"
    )
    sys.stderr.flush()
    return 2


def cmd_json() -> int:
    json.dump(_status(), sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def cmd_install() -> int:
    missing = _check_binaries()
    installed_deps = False
    if missing:
        system = platform.system()
        if system == "Darwin":
            ok, msg = _install_macos(missing)
            print(f"[setup] {msg}", file=sys.stderr)
            if not ok:
                return 2
            still_missing = _check_binaries()
            if still_missing:
                print(f"[setup] still missing after install: {', '.join(still_missing)}", file=sys.stderr)
                return 2
            installed_deps = True
        elif system == "Linux":
            print("[setup] dependencies missing on Linux — please install:", file=sys.stderr)
            print("  " + _install_hint_linux(missing), file=sys.stderr)
            return 2
        elif system == "Windows":
            print("[setup] dependencies missing on Windows — please install:", file=sys.stderr)
            print("  " + _install_hint_windows(missing), file=sys.stderr)
            return 2
        else:
            print(f"[setup] unsupported platform ({system}) for auto-install. Install manually:", file=sys.stderr)
            print(f"  missing: {', '.join(missing)}", file=sys.stderr)
            return 2

    created = _scaffold_env()
    if created:
        print(f"[setup] created config: {CONFIG_FILE}")
    else:
        print(f"[setup] config exists: {CONFIG_FILE}")

    # Binaries are in place → local faster-whisper transcription works with no
    # API key. Setup is complete; a cloud key is optional.
    _write_setup_complete()
    has_key, backend = _have_api_key()
    if installed_deps:
        print("[setup] installed dependencies.")
    print("[setup] ready. transcription: fully local (faster-whisper via uv, no API key).")
    print("[setup] the local model downloads once on first /watch, then runs offline.")
    if has_key:
        print(f"[setup] optional cloud backend also available: {backend} (use --whisper {backend}).")
    return 0


def main() -> int:
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "--check":
            return cmd_check()
        if arg == "--json":
            return cmd_json()
    return cmd_install()


if __name__ == "__main__":
    raise SystemExit(main())
