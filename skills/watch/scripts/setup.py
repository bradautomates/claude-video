#!/usr/bin/env python3
"""Setup / preflight for /watch.

Modes:
  setup.py --check          Silent preflight. Exit 0 if ready, 2/3/4 on failure.
  setup.py --json           Machine-readable status for Claude to parse.
  setup.py --set-key NAME   Interactively store a Whisper key (TTY ONLY).
  setup.py --set-detail M   Store the default detail mode (non-secret).
  setup.py --complete       Mark setup complete.
  setup.py                  Installer. Auto-installs deps, scaffolds .env.

Design:
- Silent on success: --check exits 0 with no output when everything's ready so
  that /watch doesn't spam "setup is complete" on every turn.
- Idempotent: re-running the installer is safe — it never clobbers existing
  keys and only appends missing ones.
- SETUP_COMPLETE=true in ~/.config/watch/.env tells us the user has been
  through a successful installer run at least once.
- Never sudo. On macOS, auto-install via brew. Elsewhere, print exact commands.
- Never write an API key to disk automatically — only scaffold placeholders.
- Credentials are entered by the HUMAN, never by an agent. `--set-key` reads the
  key from an interactive terminal with getpass and refuses to run when stdin
  is not a TTY, so an agent driving this script through a non-interactive shell
  cannot supply, capture, or echo the secret. The key is never accepted as a
  command-line argument (argv is readable via the process table) and is never
  printed back.
"""
from __future__ import annotations

import getpass
import json
import os
import platform
import re
import shutil
import threading
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
ENV_TEMPLATE = """# /watch API configuration
#
# Whisper transcription fallback — used only when yt-dlp cannot get captions
# (or when you point /watch at a local file with no subtitles).
#
# Groq is preferred: it runs whisper-large-v3 at a fraction of OpenAI's price
# and is faster in practice. OpenAI is the compatible fallback.
#
# Get a Groq key:  https://console.groq.com/keys
# Get an OpenAI key:  https://platform.openai.com/api-keys
#
# Leave both blank to disable Whisper — /watch will still work, but videos
# without native captions will come back frames-only.
#
# Add a key YOURSELF, either by editing the line below in your own editor or by
# running this in your own terminal (input is hidden, never echoed):
#   python3 <skill>/scripts/setup.py --set-key groq
# Never paste an API key into a chat with an agent.

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
    world/group readable.

    POSIX only: on Windows `st_mode` always reports the group/other read bits
    regardless of the actual NTFS ACL, so this check fires on every single call
    and drowns the silent-on-success contract in false warnings. Access control
    there comes from the profile ACL, not mode bits.
    """
    if os.name != "posix":
        return
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

    `status` describes the *ideal* state (a Whisper key is encouraged), so a
    keyless install still reports `needs_key` on the very first run — that's
    the agent's cue to encourage adding one.

    `can_proceed` is the operational gate: /watch can run as long as the
    binaries are present AND the user has either set a key or already finished
    setup (consciously opting out of Whisper). A keyless user who completed
    setup is NOT nagged on every call.
    """
    missing = _check_binaries()
    has_key, backend = _have_api_key()
    setup_complete = not is_first_run()

    if not missing and has_key:
        status = "ready"
    elif missing and not has_key:
        status = "needs_install_and_key"
    elif missing:
        status = "needs_install"
    else:
        status = "needs_key"

    can_proceed = (not missing) and (has_key or setup_complete)

    cfg = get_config()
    return {
        "status": status,
        "can_proceed": can_proceed,
        "first_run": not setup_complete,
        "setup_complete": setup_complete,
        "missing_binaries": missing,
        "whisper_backend": backend,
        "has_api_key": has_key,
        "config_file": str(CONFIG_FILE),
        "watch_detail": cfg["detail"],
        "platform": platform.system(),
    }


def cmd_check() -> int:
    """Silent-on-success preflight.

    Exit 0 with no output when /watch can run. A keyless user who already
    finished setup (SETUP_COMPLETE=true) counts as ready — Whisper is
    encouraged, not required — so they are never nagged on follow-up calls.

    On a state that blocks /watch, print one actionable line to stderr:
      2 → binaries missing
      3 → genuine first run with no API key (encourage one)
      4 → both missing
    """
    s = _status()
    if s["can_proceed"]:
        return 0

    parts = []
    if s["missing_binaries"]:
        parts.append(f"missing binaries: {', '.join(s['missing_binaries'])}")
    if not s["has_api_key"] and not s["setup_complete"]:
        parts.append("no Whisper API key (GROQ_API_KEY or OPENAI_API_KEY)")
    installer = Path(__file__).resolve()
    sys.stderr.write(
        f"[watch] setup incomplete ({'; '.join(parts)}). "
        f"Run: python3 {installer}\n"
    )
    sys.stderr.flush()

    if s["missing_binaries"] and not s["has_api_key"]:
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
    if has_key:
        _write_setup_complete()
        print(f"[setup] ready. whisper backend: {backend}")
        if installed_deps:
            print("[setup] installed dependencies; /watch is fully set up.")
        return 0

    print("")
    print("[setup] optional step left: add a Whisper API key (you, not the agent).")
    print("")
    print("  In YOUR OWN terminal, run one of these (input is hidden, never echoed):")
    print(f"    python3 {Path(__file__).resolve()} --set-key groq     (preferred: console.groq.com/keys)")
    print(f"    python3 {Path(__file__).resolve()} --set-key openai   (fallback: platform.openai.com/api-keys)")
    print("")
    print(f"  Or edit {CONFIG_FILE} yourself and fill in the blank key line.")
    print("  Never paste an API key into a chat with an agent.")
    print("")
    print("  Without a key, /watch still works but videos without captions come back frames-only.")
    print(f"  To proceed keyless and stop being asked: python3 {Path(__file__).resolve()} --complete")
    return 3


VALID_KEY_NAMES = {"GROQ_API_KEY", "OPENAI_API_KEY"}
_KEY_CHARS_RE = re.compile(r"^[A-Za-z0-9_\-.]{16,256}$")


def _upsert_env(name: str, value: str) -> None:
    """Set (or append) `name=value` in the config file, preserving everything
    else. Creates the file from the template when absent, re-asserts 0600."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if CONFIG_FILE.exists():
        text = CONFIG_FILE.read_text(encoding="utf-8")
    else:
        text = ENV_TEMPLATE
    lines = text.splitlines()
    replaced = False
    for i, line in enumerate(lines):
        if line.strip().startswith(f"{name}="):
            lines[i] = f"{name}={value}"
            replaced = True
            break
    if not replaced:
        lines.append(f"{name}={value}")
    CONFIG_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        CONFIG_FILE.chmod(0o600)
    except OSError:
        pass


# A TTY check alone is not a reliable human check: an agent harness can hand the
# script a pty. So the read is also bounded -- if nobody types a key within
# SECRET_READ_TIMEOUT seconds we refuse rather than block the caller forever.
# Net effect: an agent invoking this either gets an immediate refusal or a
# timeout refusal, and never a channel through which it could pass a secret.
SECRET_READ_TIMEOUT = 60.0


def _read_secret_interactive(name: str, which: str) -> str | None:
    """Read a secret from a real human at the terminal, or return None.

    The value is never echoed and never leaves this function except into
    `_upsert_env`, which writes it to a 0600 file.
    """
    sys.stderr.write(
        "[setup] Interactive key entry for {0}.\n"
        "        Type it at the prompt within {1:.0f}s. Input is hidden.\n"
        "        If you are an automated agent: abort now -- you must not handle\n"
        "        this credential. Tell the user to run this command themselves.\n".format(
            name, SECRET_READ_TIMEOUT
        )
    )
    sys.stderr.flush()

    box: dict[str, str] = {}

    def _read() -> None:
        try:
            box["value"] = getpass.getpass(f"Paste your {name} (hidden): ").strip()
        except (EOFError, KeyboardInterrupt, OSError):
            box["value"] = ""

    thread = threading.Thread(target=_read, daemon=True)
    thread.start()
    thread.join(SECRET_READ_TIMEOUT)
    if thread.is_alive():
        sys.stderr.write(
            "\n[setup] timed out waiting for interactive input; nothing written.\n"
            f"        Run this yourself in your own terminal:\n"
            f"          python3 {Path(__file__).resolve()} --set-key {which}\n"
            f"        or edit {CONFIG_FILE} and set {name}=... by hand.\n"
        )
        sys.stderr.flush()
        # The reader thread is parked on a blocking read; abandon the process
        # rather than leave the caller hanging.
        os._exit(2)
    return box.get("value", "")


def cmd_set_key(argv: list[str]) -> int:
    """Store a Whisper API key typed by the human at an interactive terminal.

    Hard requirements, by design:
      * the key is NEVER read from argv (the process table is readable);
      * stdin must be a TTY, so a non-interactive agent shell can neither pipe
        a secret in nor scrape the prompt;
      * the value is never echoed, logged, or printed back.
    """
    which = (argv[0] if argv else "").strip().lower()
    if which in ("groq", "groq_api_key"):
        name = "GROQ_API_KEY"
    elif which in ("openai", "openai_api_key"):
        name = "OPENAI_API_KEY"
    else:
        sys.stderr.write("usage: setup.py --set-key groq|openai\n")
        return 2
    if len(argv) > 1:
        sys.stderr.write(
            "[setup] refusing: the key must not be passed as an argument. "
            "Run `--set-key groq` with no value and type it at the prompt.\n"
        )
        return 2
    if not sys.stdin.isatty():
        sys.stderr.write(
            "[setup] refusing: no interactive terminal.\n"
            "        API keys are entered by you, not by an agent. Open your own\n"
            f"        terminal and run:  python3 {Path(__file__).resolve()} --set-key {which}\n"
            f"        or edit {CONFIG_FILE} directly and set {name}=...\n"
        )
        return 2
    secret = _read_secret_interactive(name, which)
    if secret is None:
        return 2
    if not secret:
        sys.stderr.write("[setup] no key entered; nothing written.\n")
        return 2
    if not _KEY_CHARS_RE.match(secret):
        sys.stderr.write(
            "[setup] that does not look like an API key "
            "(expected 16-256 chars of A-Z a-z 0-9 _ - .); nothing written.\n"
        )
        return 2
    _upsert_env(name, secret)
    del secret
    print(f"[setup] stored {name} in {CONFIG_FILE} (mode 0600). Value not echoed.")
    _write_setup_complete()
    return 0


def cmd_set_detail(argv: list[str]) -> int:
    """Store the default detail mode. Non-secret, safe for an agent to call."""
    from config import DETAILS

    mode = (argv[0] if argv else "").strip()
    if mode not in DETAILS:
        sys.stderr.write("usage: setup.py --set-detail " + "|".join(sorted(DETAILS)) + "\n")
        return 2
    _upsert_env("WATCH_DETAIL", mode)
    print(f"[setup] WATCH_DETAIL={mode}")
    return 0


def cmd_complete() -> int:
    """Mark setup complete so keyless users are never nagged again."""
    _scaffold_env()
    _write_setup_complete()
    print(f"[setup] setup marked complete in {CONFIG_FILE}")
    return 0


def main() -> int:
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        rest = sys.argv[2:]
        if arg == "--check":
            return cmd_check()
        if arg == "--json":
            return cmd_json()
        if arg == "--set-key":
            return cmd_set_key(rest)
        if arg == "--set-detail":
            return cmd_set_detail(rest)
        if arg == "--complete":
            return cmd_complete()
        if arg in ("-h", "--help"):
            print(__doc__)
            return 0
        sys.stderr.write(f"[setup] unknown option: {arg}\n")
        return 2
    return cmd_install()


if __name__ == "__main__":
    raise SystemExit(main())
