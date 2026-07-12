"""setup.py --json surfaces the resolved watch detail."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import setup as watch_setup  # importable via conftest.py's sys.path.insert

SETUP = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts" / "setup.py"


def _run(args, *, home=None, extra_env=None):
    env = dict(os.environ)
    env.pop("WATCH_DETAIL", None)
    # Don't let a real key in the developer's shell env leak into the test.
    env.pop("GROQ_API_KEY", None)
    env.pop("OPENAI_API_KEY", None)
    env.pop("SETUP_COMPLETE", None)
    if home is not None:
        env["HOME"] = str(home)
        env["USERPROFILE"] = str(home)  # Windows
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, str(SETUP), *args],
        capture_output=True, text=True, env=env,
    )


def _write_env(home: Path, body: str) -> None:
    cfg = home / ".config" / "watch"
    cfg.mkdir(parents=True, exist_ok=True)
    f = cfg / ".env"
    f.write_text(body, encoding="utf-8")
    f.chmod(0o600)


def test_json_reports_watch_detail():
    proc = _run(["--json"])
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["watch_detail"] == "balanced"


def test_keyless_completed_setup_proceeds_silently(tmp_path):
    """A user who finished setup without a key must NOT be nagged forever."""
    _write_env(tmp_path, "GROQ_API_KEY=\nOPENAI_API_KEY=\nSETUP_COMPLETE=true\n")
    chk = _run(["--check"], home=tmp_path)
    assert chk.returncode == 0, f"keyless-complete should pass --check; got {chk.returncode}: {chk.stderr}"
    assert chk.stdout == "" and chk.stderr == ""

    js = json.loads(_run(["--json"], home=tmp_path).stdout)
    assert js["can_proceed"] is True
    assert js["first_run"] is False
    assert js["setup_complete"] is True
    # status still encourages a key even though we can proceed
    assert js["status"] == "needs_key"


def test_keyless_first_run_is_encouraged(tmp_path):
    """Genuine first run with no key: --check reports exit 3 (encourage a key)."""
    _write_env(tmp_path, "GROQ_API_KEY=\nOPENAI_API_KEY=\n")
    chk = _run(["--check"], home=tmp_path)
    assert chk.returncode == 3, chk.stderr

    js = json.loads(_run(["--json"], home=tmp_path).stdout)
    assert js["can_proceed"] is False
    assert js["first_run"] is True


def test_key_present_is_ready(tmp_path):
    _write_env(tmp_path, "GROQ_API_KEY=sk-test-abc\n")
    chk = _run(["--check"], home=tmp_path)
    assert chk.returncode == 0, chk.stderr

    js = json.loads(_run(["--json"], home=tmp_path).stdout)
    assert js["status"] == "ready"
    assert js["can_proceed"] is True
    assert js["whisper_backend"] == "groq"


def test_missing_youtube_deps_is_reported_but_never_blocks(tmp_path):
    """missing_youtube_deps is informational only — it must never change
    status/can_proceed, regardless of whether this machine happens to have
    deno installed (issue #67)."""
    _write_env(tmp_path, "GROQ_API_KEY=sk-test-abc\n")
    js = json.loads(_run(["--json"], home=tmp_path).stdout)

    assert isinstance(js["missing_youtube_deps"], list)
    assert set(js["missing_youtube_deps"]) <= set(watch_setup.YOUTUBE_OPTIONAL_BINARIES)
    # Same assertions as test_key_present_is_ready — must hold no matter what
    # missing_youtube_deps says, since youtube deps are unrelated to readiness.
    assert js["status"] == "ready"
    assert js["can_proceed"] is True


def test_check_youtube_deps_detects_missing_deno(monkeypatch):
    monkeypatch.setattr(watch_setup, "_which", lambda name: None)
    assert watch_setup._check_youtube_deps() == ["deno"]


def test_check_youtube_deps_empty_when_present(monkeypatch):
    monkeypatch.setattr(watch_setup, "_which", lambda name: f"/usr/bin/{name}")
    assert watch_setup._check_youtube_deps() == []


def test_youtube_dep_hint_mentions_curl_cffi_on_every_platform():
    """curl_cffi can't be detected (see YOUTUBE_OPTIONAL_BINARIES), so the
    hint text is the only place it's ever surfaced to the user — it must
    always be there."""
    for system in ("Darwin", "Windows", "Linux", "SomeOtherOS"):
        assert "curl_cffi" in watch_setup._youtube_dep_hint(system)
