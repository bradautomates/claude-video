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
- yt-dlp staleness check is local-only: yt-dlp's version numbers ARE release
  dates (`2026.08.19`), so "is it stale" is a pure date comparison against
  the installed binary's own --version output. No network call, ever --
  querying GitHub/PyPI for the latest release would add a failure mode and
  a privacy surface to a check that runs on every invocation. A stale
  binary still exits 0 (it's a warning, not a blocker) and stays silent
  when the version string doesn't parse as a date.
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from config import force_utf8_output, get_config, js_runtime, read_env_value, ytdlp_cmd  # noqa: E402


REQUIRED_BINARIES = ["ffmpeg", "yt-dlp"]
# ffprobe normally ships with ffmpeg, but Windows Application Control can block
# it while allowing ffmpeg.exe (#128). The scripts fall back to `ffmpeg -i` for
# metadata, so a missing ffprobe is worth a note, not a hard failure.
OPTIONAL_BINARIES = ["ffprobe"]
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

GROQ_API_KEY=
OPENAI_API_KEY=

# Local / self-hosted transcription. Point this at any server exposing OpenAI's
# /v1/audio/transcriptions route (whisper.cpp `server`, faster-whisper-server,
# speaches, LM Studio) and audio never leaves the machine. When set, it takes
# precedence over Groq and OpenAI; override per-run with --whisper groq|openai.
# A bare origin is fine — the route is appended automatically.
# WATCH_WHISPER_BASE_URL=http://localhost:8080
# WATCH_WHISPER_MODEL=whisper-1
# WATCH_WHISPER_API_KEY=          # usually unnecessary for a local server

# Default watch behavior (the /watch first-run wizard sets this for you).
# Allowed values: transcript | efficient | balanced | token-burner
# Keep the value on its own line with no trailing comment.
# WATCH_DETAIL=balanced
"""

# yt-dlp releases roughly every few weeks as YouTube rotates its
# client/signature scheme; a binary that has gone quiet for a lot longer than
# that is the most common cause of a silent "HTTP 403: Forbidden". Chosen
# from real data, not a guess: a healthy gap between two consecutive releases
# was 46 days, and the binary that actually started 403ing in the
# field was ~76 days old. 60 sits comfortably above the observed healthy
# gap -- so it won't nag right after an ordinary release cycle and train the
# user to ignore it -- while still giving roughly two weeks' warning before
# the age that has already been proven to break downloads.
YT_DLP_STALE_DAYS = 60

_YT_DLP_VERSION_RE = re.compile(r"^(\d{4})\.(\d{1,2})\.(\d{1,2})")


def _which(name: str) -> str | None:
    return shutil.which(name)


def _check_binaries() -> list[str]:
    def present(name: str) -> bool:
        # WATCH_YTDLP may name a specific binary or a `python -m yt_dlp` command.
        return _which(ytdlp_cmd()[0] if name == "yt-dlp" else name) is not None
    return [b for b in REQUIRED_BINARIES if not present(b)]


def _yt_dlp_version() -> str | None:
    """Raw `yt-dlp --version` output, or None if it can't be read.

    Shells out rather than `import yt_dlp` + reading `__version__`: yt-dlp's
    own recommended Linux install path is pipx (see `_install_hint_linux`
    below), which puts it in an isolated venv this interpreter cannot import
    from at all -- verified on the reference machine, where `yt-dlp
    --version` succeeds while `import yt_dlp` raises ModuleNotFoundError from
    both the system Python and this project's own venv. Only the binary's
    PATH entry is guaranteed to resolve regardless of install method (pipx,
    brew, apt, pip --user, the standalone release binary), so that's the only
    source that works uniformly. No network call -- this just execs the
    already-installed binary.
    """
    try:
        proc = subprocess.run(
            [*ytdlp_cmd(), "--version"], capture_output=True, text=True, timeout=5
        )
        return proc.stdout.strip() or None
    except Exception:
        return None


def _yt_dlp_stale_days_from_version(version: str) -> int | None:
    """Age in days of a yt-dlp version string, if it's old enough to flag.

    Returns None (stay quiet) when the version doesn't parse as a leading
    YYYY.MM.DD release date -- forks and distro builds use non-date version
    schemes, and a false "your yt-dlp is stale" on a perfectly good binary is
    worse than saying nothing. Also None when it parses but isn't past the
    threshold yet. Pure function of the version string and today's date -- no
    subprocess, no network.
    """
    match = _YT_DLP_VERSION_RE.match(version.strip())
    if not match:
        return None
    try:
        released = date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        return None
    age_days = (date.today() - released).days
    return age_days if age_days > YT_DLP_STALE_DAYS else None


def _yt_dlp_staleness(missing_binaries: list[str]) -> int | None:
    """Flagged staleness (in days), or None if fresh/unknown/not installed.

    Skips the version lookup entirely when yt-dlp is missing -- that's
    already exit 2's job, and there's nothing to date-check.
    """
    if "yt-dlp" in missing_binaries:
        return None
    version = _yt_dlp_version()
    if not version:
        return None
    return _yt_dlp_stale_days_from_version(version)


def _stale_note(days: int) -> str:
    return (
        f"yt-dlp is {days} days old — YouTube rotates its download scheme "
        f"every few weeks, so a stale yt-dlp is the most common cause of "
        f'"HTTP 403: Forbidden" errors. Update: yt-dlp -U '
        f"(or: pipx upgrade yt-dlp / brew upgrade yt-dlp)"
    )


YTDLP_FULL_INSTALL = "pipx install --force 'yt-dlp[default,curl-cffi]'  (or: pip install -U 'yt-dlp[default,curl-cffi]')"


def _yt_dlp_impersonation(missing_binaries: list[str]) -> bool | None:
    """True if yt-dlp has at least one usable browser-impersonation target.

    YouTube refuses the *media* stream (403) to clients it cannot fingerprint
    while still serving titles and captions, which is why a missing curl_cffi
    looks like a video-specific bug. Homebrew's yt-dlp formula omits curl_cffi
    (#93). None when unknown (yt-dlp missing, old build without the flag).
    """
    if "yt-dlp" in missing_binaries:
        return None
    try:
        proc = subprocess.run(
            [*ytdlp_cmd(), "--list-impersonate-targets"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10,
        )
    except Exception:
        return None
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0 and "impersonate" not in out.lower():
        return None  # flag unknown to this build
    rows = [
        line for line in out.splitlines()
        if line.strip() and not line.startswith(("[", "Client", "---"))
    ]
    if not rows:
        return None  # no target table at all: can't tell, don't warn
    usable = [r for r in rows if "unavailable" not in r.lower()]
    return bool(usable)


def _js_runtime() -> str | None:
    """Name of a JavaScript runtime yt-dlp can use for YouTube's challenge solver."""
    found = js_runtime()
    return found[0] if found else None


def _ytdlp_capability_notes(missing_binaries: list[str]) -> list[str]:
    """Warnings for a yt-dlp that exists but is likely to 403 on YouTube."""
    notes: list[str] = []
    if "yt-dlp" in missing_binaries:
        return notes
    if _yt_dlp_impersonation(missing_binaries) is False:
        notes.append(
            "yt-dlp has no browser-impersonation targets (built without curl_cffi — "
            "Homebrew's formula omits it). YouTube will likely return 403 for the video "
            f"stream while captions still work. Fix: {YTDLP_FULL_INSTALL}"
        )
    found = js_runtime()
    if found is None:
        notes.append(
            "no JavaScript runtime found (deno preferred; node/quickjs/bun also work). "
            "Recent yt-dlp uses one to solve YouTube's player challenge; without it some "
            "formats go missing and downloads degrade to lower quality. "
            "Install deno: brew install deno / winget install DenoLand.Deno"
        )
    # A non-deno runtime is handled automatically (download.py passes
    # --js-runtimes), so it is not a warning: --check stays silent on success.
    return notes


_PERM_WARNED: set[str] = set()

# POSIX mode bits do not govern access on Windows and cannot be set
# from Python — os.chmod there only toggles the read-only attribute, verified by
# creating a file with and without chmod(0o600) and diffing icacls: identical,
# both inheriting SYSTEM / Administrators / user. So the mode is neither
# meaningful to read nor settable to write, and both halves are skipped below.
_IS_WINDOWS = os.name == "nt"


def _check_file_permissions(path: Path) -> None:
    """Warn to stderr (once per path per process) if a secrets file is
    world/group readable.

    No-op on Windows: st_mode there is synthesized from the read-only attribute,
    so the group/other bits are always set and this would warn on every run with
    a `chmod 600` fix that cannot change anything.
    """
    if _IS_WINDOWS:
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
    """Read one setting, warning first if the secrets file is too permissive.

    Parsing lives in config.read_env_value so this agrees with every other
    consumer; only the permission warning is local to setup.
    """
    # Same search order as whisper.py (config first, then ./.env) so the
    # preflight and the transcription step agree on whether a key exists (#136).
    return read_env_value(name, on_file=_check_file_permissions)


def _have_api_key() -> tuple[bool, str | None]:
    if _read_env_key("GROQ_API_KEY"):
        return True, "groq"
    if _read_env_key("OPENAI_API_KEY"):
        return True, "openai"
    return False, None


def is_first_run() -> bool:
    """True if the installer hasn't completed successfully yet."""
    return _read_env_key("SETUP_COMPLETE") != "true"


def _restrict_config_file() -> None:
    """Restrict .env to the owner where the platform supports it.

    POSIX: mode 0600. Windows: nothing — the file keeps the ACL it inherits from
    the user profile, which grants the user, SYSTEM and Administrators. That is
    the normal posture for user secrets on Windows (an administrator can read
    any file regardless), and no other ordinary user is granted access. Locking
    it further would mean icacls /inheritance:r, which mostly breaks backup and
    AV agents for no gain against an attacker who is already an admin.
    """
    if _IS_WINDOWS:
        return
    try:
        CONFIG_FILE.chmod(0o600)
    except OSError:
        pass


def _scaffold_env() -> bool:
    """Create ~/.config/watch/.env with placeholders if missing."""
    if CONFIG_FILE.exists():
        return False
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(ENV_TEMPLATE, encoding="utf-8")
    _restrict_config_file()
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
    _restrict_config_file()


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
    yt_dlp_stale_days = _yt_dlp_staleness(missing)
    yt_dlp_notes = _ytdlp_capability_notes(missing)

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
        "yt_dlp_stale_days": yt_dlp_stale_days,
        "yt_dlp_notes": yt_dlp_notes,
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
    A stale yt-dlp never changes the exit code -- it's a warning, printed
    even when otherwise ready (still exit 0) and folded into the failure
    message otherwise. A stale binary that still works must not block a run.
    """
    s = _status()
    stale_days = s["yt_dlp_stale_days"]

    if s["can_proceed"]:
        if stale_days is not None:
            sys.stderr.write(f"[watch] {_stale_note(stale_days)}\n")
        for note in s.get("yt_dlp_notes", []):
            sys.stderr.write(f"[watch] WARNING: {note}\n")
        missing_optional = [b for b in OPTIONAL_BINARIES if not _which(b)]
        if missing_optional:
            sys.stderr.write(
                f"[watch] note: {', '.join(missing_optional)} not found — metadata will be "
                "read via ffmpeg instead (slower, but works).\n"
            )
        sys.stderr.flush()
        return 0

    parts = []
    if s["missing_binaries"]:
        parts.append(f"missing binaries: {', '.join(s['missing_binaries'])}")
    if not s["has_api_key"] and not s["setup_complete"]:
        parts.append("no Whisper API key (GROQ_API_KEY or OPENAI_API_KEY)")
    if stale_days is not None:
        parts.append(_stale_note(stale_days))
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
    print("[setup] one step left: add a Whisper API key.")
    print("")
    print(f"  Edit {CONFIG_FILE} and set either:")
    print("    GROQ_API_KEY=...    (preferred — cheaper, faster; get one at console.groq.com/keys)")
    print("    OPENAI_API_KEY=...  (fallback; get one at platform.openai.com/api-keys)")
    print("")
    print("  Without a key, /watch still works but videos without captions come back frames-only.")
    return 3


def main() -> int:
    force_utf8_output()
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "--check":
            return cmd_check()
        if arg == "--json":
            return cmd_json()
    return cmd_install()


if __name__ == "__main__":
    raise SystemExit(main())
