"""yt-dlp staleness preflight in setup.py.

Real incident (2026-09-17): the installed yt-dlp was 2026.07.04 (~76 days
old), every download 403'd, and `--check` reported everything fine because
it only ever checked *presence* (`shutil.which`), never *freshness*. This
covers the local, no-network staleness check that closes that gap.

Two layers:
  - Pure function tests against `setup._yt_dlp_stale_days_from_version`,
    with every version INJECTED via a synthetic string built relative to
    `date.today()` -- never against whatever yt-dlp happens to be installed
    on the machine running this suite, so the suite doesn't rot as the
    calendar (or the real binary) moves.
  - Integration tests against `setup.cmd_check()` / `setup._status()`,
    monkeypatching `setup._yt_dlp_version` (the one subprocess boundary) so
    every case is deterministic and network/subprocess-free.

Import path: `tests/conftest.py` already inserts
`skills/watch/scripts` onto `sys.path`, so `import setup` here resolves to
the real module under test -- mirroring `tests/test_config.py`.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

import setup

SETUP_PATH = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts" / "setup.py"

THRESHOLD = setup.YT_DLP_STALE_DAYS


def _version(days_old: int) -> str:
    """Build a yt-dlp-style `YYYY.MM.DD` version string `days_old` days old."""
    d = date.today() - timedelta(days=days_old)
    return f"{d.year:04d}.{d.month:02d}.{d.day:02d}"


# ---------------------------------------------------------------------------
# Pure function: version string -> flagged age or None
# ---------------------------------------------------------------------------


def test_stale_version_flags_with_age():
    """~76-days-old shape (the real incident) is flagged with its exact age."""
    v = _version(THRESHOLD + 16)
    days = setup._yt_dlp_stale_days_from_version(v)
    assert days == THRESHOLD + 16


def test_fresh_version_is_silent():
    v = _version(5)
    assert setup._yt_dlp_stale_days_from_version(v) is None


@pytest.mark.parametrize(
    "bogus", ["unknown", "custom-build", "nightly", "", "v2026", "2026-08-19"]
)
def test_unparseable_version_is_silent(bogus):
    """Forks/distro builds carry non-date version strings -- must stay quiet,
    never warn on a binary this can't actually judge."""
    assert setup._yt_dlp_stale_days_from_version(bogus) is None


def test_invalid_calendar_date_is_silent():
    """Right shape (YYYY.MM.DD), not a real date (month 13, day 40) -- must
    not raise, and must not warn."""
    assert setup._yt_dlp_stale_days_from_version("2026.13.40") is None


def test_boundary_exactly_at_threshold_is_not_flagged():
    v = _version(THRESHOLD)
    assert setup._yt_dlp_stale_days_from_version(v) is None


def test_boundary_one_day_past_threshold_is_flagged():
    v = _version(THRESHOLD + 1)
    assert setup._yt_dlp_stale_days_from_version(v) == THRESHOLD + 1


def test_trailing_suffix_after_date_still_parses():
    """Nightly/patch builds like 2026.08.19.123 -- leading date still counts."""
    d = date.today() - timedelta(days=THRESHOLD + 10)
    v = f"{d.year:04d}.{d.month:02d}.{d.day:02d}.7"
    assert setup._yt_dlp_stale_days_from_version(v) == THRESHOLD + 10


# ---------------------------------------------------------------------------
# _yt_dlp_staleness(): the missing-binary skip
# ---------------------------------------------------------------------------


def test_missing_binary_skips_the_check_entirely(monkeypatch):
    """yt-dlp missing is already exit 2's job -- must not even attempt a
    version read."""

    def _boom():
        raise AssertionError("_yt_dlp_version() must not be called when yt-dlp is missing")

    monkeypatch.setattr(setup, "_yt_dlp_version", _boom)
    assert setup._yt_dlp_staleness(["yt-dlp"]) is None


def test_unreadable_version_is_silent(monkeypatch):
    monkeypatch.setattr(setup, "_yt_dlp_version", lambda: None)
    assert setup._yt_dlp_staleness([]) is None


# ---------------------------------------------------------------------------
# Integration: cmd_check() / _status() -- exit code untouched, silent-on-
# success preserved, --json consistent with human mode.
# ---------------------------------------------------------------------------


def _patch_ready(monkeypatch, version):
    """Make the world look otherwise-ready (no missing binaries, a real
    key) with yt-dlp reporting `version` (or None if unreadable)."""
    monkeypatch.setattr(setup, "_check_binaries", lambda: [])
    monkeypatch.setattr(setup, "_have_api_key", lambda: (True, "groq"))
    monkeypatch.setattr(setup, "is_first_run", lambda: False)
    monkeypatch.setattr(setup, "_yt_dlp_version", lambda: version)
    # Impersonation / JS-runtime notes are a separate concern with their own
    # tests; on a CI runner whose pip yt-dlp lacks curl_cffi they would fire.
    monkeypatch.setattr(setup, "_ytdlp_capability_notes", lambda missing: [])


def test_stale_warns_and_still_exits_zero(monkeypatch, capsys):
    _patch_ready(monkeypatch, _version(THRESHOLD + 16))
    code = setup.cmd_check()
    captured = capsys.readouterr()
    assert code == 0
    assert captured.out == ""
    assert "yt-dlp" in captured.err
    assert str(THRESHOLD + 16) in captured.err


def test_fresh_is_byte_for_byte_silent(monkeypatch, capsys):
    """Silent-on-success must be preserved exactly: a fresh yt-dlp prints
    nothing at all, not even a blank line."""
    _patch_ready(monkeypatch, _version(5))
    code = setup.cmd_check()
    captured = capsys.readouterr()
    assert code == 0
    assert captured.out == ""
    assert captured.err == ""


def test_unparseable_is_byte_for_byte_silent(monkeypatch, capsys):
    _patch_ready(monkeypatch, "unknown-build")
    code = setup.cmd_check()
    captured = capsys.readouterr()
    assert code == 0
    assert captured.out == ""
    assert captured.err == ""


def test_exit_code_unchanged_across_all_staleness_outcomes(monkeypatch):
    """Exit code must be exactly 0 in the ready case regardless of which of
    the three staleness outcomes (stale / fresh / unparseable) applies --
    staleness is a warning, never a gate."""
    for candidate in (_version(THRESHOLD + 16), _version(5), "unknown-build"):
        _patch_ready(monkeypatch, candidate)
        assert setup.cmd_check() == 0


def test_json_reports_staleness_consistently(monkeypatch):
    _patch_ready(monkeypatch, _version(THRESHOLD + 16))
    s = setup._status()
    assert "yt_dlp_stale_days" in s
    assert s["yt_dlp_stale_days"] == THRESHOLD + 16

    _patch_ready(monkeypatch, _version(5))
    s2 = setup._status()
    assert s2["yt_dlp_stale_days"] is None


def test_check_skipped_when_yt_dlp_missing(monkeypatch, capsys):
    """yt-dlp missing already exits 2 via the existing binary check -- the
    staleness note must not appear, and the exit code must be exactly 2
    (not shifted by the new field)."""
    monkeypatch.setattr(setup, "_check_binaries", lambda: ["yt-dlp"])
    monkeypatch.setattr(setup, "_have_api_key", lambda: (True, "groq"))
    monkeypatch.setattr(setup, "is_first_run", lambda: False)

    def _boom():
        raise AssertionError("_yt_dlp_version() must not be called when yt-dlp is missing")

    monkeypatch.setattr(setup, "_yt_dlp_version", _boom)
    code = setup.cmd_check()
    captured = capsys.readouterr()
    assert code == 2
    assert "yt-dlp is" not in captured.err  # no staleness note, only the missing-binary line


def test_stale_folds_into_failure_message_without_changing_exit_code(monkeypatch, capsys):
    """A stale yt-dlp alongside a genuinely missing key must not change the
    exit code (still 3) -- it's appended to the same message, never a
    separate gate."""
    monkeypatch.setattr(setup, "_check_binaries", lambda: [])
    monkeypatch.setattr(setup, "_have_api_key", lambda: (False, None))
    monkeypatch.setattr(setup, "is_first_run", lambda: True)
    monkeypatch.setattr(setup, "_yt_dlp_version", lambda: _version(THRESHOLD + 16))
    code = setup.cmd_check()
    captured = capsys.readouterr()
    assert code == 3
    assert "yt-dlp is" in captured.err


# ---------------------------------------------------------------------------
# Real subprocess entry point: prove the whole CLI path is wired up, not
# just the monkeypatched unit.
# ---------------------------------------------------------------------------


def _run_cli(args, *, home, extra_env=None):
    env = dict(os.environ)
    env.pop("WATCH_DETAIL", None)
    env.pop("GROQ_API_KEY", None)
    env.pop("OPENAI_API_KEY", None)
    env.pop("SETUP_COMPLETE", None)
    env["HOME"] = str(home)
    env["USERPROFILE"] = str(home)  # Windows
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, str(SETUP_PATH), *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace", env=env,
    )


def test_cli_json_includes_stale_field_key(tmp_path):
    """Whatever yt-dlp is actually installed on the machine running the
    suite, the --json contract must always include the key (value may be
    an int or None depending on the real binary's freshness)."""
    cfg = tmp_path / ".config" / "watch"
    cfg.mkdir(parents=True)
    (cfg / ".env").write_text("GROQ_API_KEY=sk-test\n", encoding="utf-8")
    proc = _run_cli(["--json"], home=tmp_path)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert "yt_dlp_stale_days" in data


# --- impersonation / JS-runtime capability notes (#93, #67) -----------------

import setup as _setup  # noqa: E402


def test_impersonation_false_when_all_targets_unavailable(monkeypatch):
    class R:
        returncode = 0
        stdout = ("[info] Available impersonate targets\nClient OS Source\n----\n"
                  "Chrome-133 Macos-15 curl_cffi (unavailable)\n")
        stderr = ""
    monkeypatch.setattr(_setup.subprocess, "run", lambda *a, **k: R())
    assert _setup._yt_dlp_impersonation([]) is False


def test_impersonation_true_with_a_usable_target(monkeypatch):
    class R:
        returncode = 0
        stdout = "[info] Available impersonate targets\nClient OS Source\n----\nChrome-133 Macos-15 curl_cffi\n"
        stderr = ""
    monkeypatch.setattr(_setup.subprocess, "run", lambda *a, **k: R())
    assert _setup._yt_dlp_impersonation([]) is True


def test_capability_notes_skip_when_ytdlp_missing():
    assert _setup._ytdlp_capability_notes(["yt-dlp"]) == []


def test_capability_notes_mention_js_runtime(monkeypatch):
    import config as _cfg
    monkeypatch.setattr(_setup, "_yt_dlp_impersonation", lambda missing: True)
    monkeypatch.setattr(_cfg.shutil, "which", lambda n: None)
    notes = _setup._ytdlp_capability_notes([])
    assert len(notes) == 1 and "JavaScript runtime" in notes[0]


# --- JS runtime: yt-dlp only enables deno by default (#237 review) ----------

import config as _config  # noqa: E402


def _which_only(*names):
    return lambda n: f"/usr/bin/{n}" if n in names else None


def test_deno_present_needs_no_flag(monkeypatch):
    monkeypatch.setattr(_config.shutil, "which", _which_only("deno", "node"))
    assert _config.js_runtime_args() == []
    assert _setup._ytdlp_capability_notes([]) == [] or all("deno not found" not in n for n in _setup._ytdlp_capability_notes([]))


def test_node_without_deno_is_passed_to_ytdlp(monkeypatch):
    monkeypatch.setattr(_config.shutil, "which", _which_only("node", "yt-dlp", "ffmpeg"))
    assert _config.js_runtime_args() == ["--js-runtimes", "node"]
    monkeypatch.setattr(_setup, "_yt_dlp_impersonation", lambda missing: True)
    # Handled automatically via --js-runtimes, so --check must stay silent.
    assert _setup._ytdlp_capability_notes([]) == []


def test_qjs_binary_maps_to_quickjs_runtime_name(monkeypatch):
    monkeypatch.setattr(_config.shutil, "which", _which_only("qjs"))
    assert _config.js_runtime_args() == ["--js-runtimes", "quickjs"]


def test_no_runtime_warns_about_degradation_not_failure(monkeypatch):
    monkeypatch.setattr(_config.shutil, "which", _which_only("yt-dlp", "ffmpeg"))
    monkeypatch.setattr(_setup, "_yt_dlp_impersonation", lambda missing: True)
    notes = _setup._ytdlp_capability_notes([])
    assert len(notes) == 1 and "formats go missing" in notes[0] and "fail" not in notes[0].lower()
