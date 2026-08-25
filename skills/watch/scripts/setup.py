#!/usr/bin/env python3
"""Setup / preflight for /watch.

Modes:
  setup.py --check      Silent preflight. Exit 0 if ready, 2/3/4 on failure.
  setup.py --json       Machine-readable status for Claude to parse.
  setup.py              Installer. Auto-installs deps, scaffolds .env, marks SETUP_COMPLETE.

Design:
- Silent on success: --check exits 0 with no output when everything's ready so
  that /watch doesn't spam "setup is complete" on every turn.
- Idempotent: re-running the installer is safe — it never clobbers existing
  keys and only appends missing ones.
- SETUP_COMPLETE=true in ~/.config/watch/.env tells us the user has been
  through a successful installer run at least once.
- Never sudo. On macOS, auto-install via brew. Elsewhere, print exact commands.
- Never write an API key to disk automatically — only scaffold placeholders.
- Transcription is satisfied by EITHER an API key or the local faster-whisper
  backend, so a keyless machine with the package installed is fully ready.
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


REQUIRED_BINARIES = ["ffmpeg", "ffprobe", "yt-dlp"]
CONFIG_DIR = Path.home() / ".config" / "watch"
CONFIG_FILE = CONFIG_DIR / ".env"
ENV_TEMPLATE = """# /watch configuration
#
# Whisper transcription fallback — used only when yt-dlp cannot get captions
# (or when you point /watch at a local file with no subtitles).
#
# There are three backends. You need exactly one of them:
#
#   local  — runs on this machine, no key and no upload. Install with
#            `pip install "faster-whisper>=1.0"`. First use downloads the model
#            (2.9 GB for large-v3, 464 MB for small) to the Hugging Face cache;
#            later runs re-check it there unless you set HF_HUB_OFFLINE=1.
#   groq   — whisper-large-v3 at a fraction of OpenAI's price, and faster in
#            practice. Get a key: https://console.groq.com/keys
#   openai — the compatible fallback.
#            Get a key: https://platform.openai.com/api-keys
#
# With none of them, /watch still works — videos without native captions just
# come back frames-only.

GROQ_API_KEY=
OPENAI_API_KEY=

# Which backend to use. auto = an API key if one is set, else local.
# Allowed values: auto | local | groq | openai
# WATCH_WHISPER=auto

# Local backend tuning. All optional; blank means "let the backend decide".
#   MODEL    a faster-whisper size (tiny|base|small|medium|large-v3|
#            distil-large-v3), a Hugging Face repo id, or a local model path.
#   DEVICE   auto | cpu | cuda      COMPUTE  auto | int8 | int8_float16 | float16
#   LANGUAGE an ISO code (en, de, …). Blank auto-detects.
# WATCH_WHISPER_MODEL=large-v3
# WATCH_WHISPER_DEVICE=auto
# WATCH_WHISPER_COMPUTE=auto
# WATCH_WHISPER_LANGUAGE=

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


def _have_local_whisper() -> bool:
    """True if the on-device backend is installed.

    Note this reports installed, not working: whether CUDA is usable, or whether
    the model weights have been downloaded yet, is not knowable without loading
    a model — too slow for a preflight that runs on every /watch call.
    """
    try:
        import local_whisper
    except Exception:
        return False
    return local_whisper.is_available()


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

    Used only after a fully successful install (deps + key). Future sessions
    detect this marker to skip wizard-style UI and stay silent.
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
        else:
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
    return "\n  ".join(hints) if hints else "nothing to install"


def _install_hint_windows(missing: list[str]) -> str:
    pkgs = _brew_pkg(missing)
    hints = []
    if "ffmpeg" in pkgs:
        hints.append("winget: `winget install Gyan.FFmpeg`")
    if "yt-dlp" in pkgs:
        hints.append("winget: `winget install yt-dlp.yt-dlp` or pip: `pip install --user yt-dlp`")
    return "\n  ".join(hints) if hints else "nothing to install"


def _status() -> dict:
    """Structured preflight snapshot.

    `status` describes the *ideal* state, so a machine with no transcription
    backend at all still reports `needs_key` on the very first run — that's the
    agent's cue to suggest one. A key is only one of two ways to satisfy it:
    faster-whisper counts, and needs no account.

    `can_proceed` is the operational gate: /watch can run as long as the
    binaries are present AND transcription is available OR the user already
    finished setup (consciously opting out of Whisper). Someone who completed
    setup without either backend is NOT nagged on every call.
    """
    missing = _check_binaries()
    has_key, backend = _have_api_key()
    has_local = _have_local_whisper()
    has_transcription = has_key or has_local
    setup_complete = not is_first_run()

    if not backend and has_local:
        backend = "local"

    if not missing and has_transcription:
        status = "ready"
    elif missing and not has_transcription:
        status = "needs_install_and_key"
    elif missing:
        status = "needs_install"
    else:
        status = "needs_key"

    can_proceed = (not missing) and (has_transcription or setup_complete)

    cfg = get_config()
    return {
        "status": status,
        "can_proceed": can_proceed,
        "first_run": not setup_complete,
        "setup_complete": setup_complete,
        "missing_binaries": missing,
        "whisper_backend": backend,
        "has_api_key": has_key,
        "has_local_whisper": has_local,
        "has_transcription": has_transcription,
        "config_file": str(CONFIG_FILE),
        "watch_detail": cfg["detail"],
        "whisper_setting": cfg["whisper"],
        "platform": platform.system(),
    }


def cmd_check() -> int:
    """Silent-on-success preflight.

    Exit 0 with no output when /watch can run. A user who already finished setup
    (SETUP_COMPLETE=true) counts as ready even with no transcription backend —
    Whisper is encouraged, not required — so they are never nagged on follow-up
    calls.

    On a state that blocks /watch, print one actionable line to stderr:
      2 → binaries missing
      3 → genuine first run with no transcription backend
      4 → both missing
    """
    s = _status()
    if s["can_proceed"]:
        return 0

    parts = []
    if s["missing_binaries"]:
        parts.append(f"missing binaries: {', '.join(s['missing_binaries'])}")
    if not s["has_transcription"] and not s["setup_complete"]:
        parts.append(
            "no Whisper backend (pip install \"faster-whisper>=1.0\", "
            "or set GROQ_API_KEY / OPENAI_API_KEY)"
        )
    installer = Path(__file__).resolve()
    sys.stderr.write(
        f"[watch] setup incomplete ({'; '.join(parts)}). "
        f"Run: python3 {installer}\n"
    )
    sys.stderr.flush()

    if s["missing_binaries"] and not s["has_transcription"]:
        return 4
    if s["missing_binaries"]:
        return 2
    return 3


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

    has_key, backend = _have_api_key()
    has_local = _have_local_whisper()
    if has_key or has_local:
        _write_setup_complete()
        print(f"[setup] ready. whisper backend: {backend or 'local'}")
        if has_local and not has_key:
            print("[setup] transcription runs on this machine — no API key needed.")
            print("[setup] first use downloads the model to the Hugging Face cache;")
            print("[setup] later runs re-check it there unless HF_HUB_OFFLINE=1 is set.")
        if installed_deps:
            print("[setup] installed dependencies; /watch is fully set up.")
        return 0

    print("")
    print("[setup] one step left: pick a Whisper backend. Either works — you need one.")
    print("")
    print("  On this machine, no account:")
    print("    pip install \"faster-whisper>=1.0\"")
    print("    (downloads the model on first use: 2.9 GB for large-v3, 464 MB for small)")
    print("")
    print(f"  Or via an API — edit {CONFIG_FILE} and set one of:")
    print("    GROQ_API_KEY=...    (cheaper and faster than OpenAI; console.groq.com/keys)")
    print("    OPENAI_API_KEY=...  (fallback; platform.openai.com/api-keys)")
    print("")
    print("  With neither, /watch still works but videos without captions come back frames-only.")
    return 3


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
