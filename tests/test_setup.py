"""setup.py preflight — local-first (no API key required) contract."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SETUP = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts" / "setup.py"

# Most assertions here need all required binaries present (incl. uv, which drives
# the local faster-whisper backend). Skip the binary-dependent cases when uv is
# absent so the suite stays green on machines that haven't run setup.
HAS_UV = shutil.which("uv") is not None
needs_uv = pytest.mark.skipif(not HAS_UV, reason="requires uv (local ASR backend) on PATH")


def _run(args, *, home=None, extra_env=None):
    env = dict(os.environ)
    env.pop("WATCH_DETAIL", None)
    # Don't let a real key in the developer's shell env leak into the test.
    env.pop("GROQ_API_KEY", None)
    env.pop("OPENAI_API_KEY", None)
    env.pop("SETUP_COMPLETE", None)
    env.pop("WATCH_LOCAL_ASR", None)
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


@needs_uv
def test_keyless_first_run_is_ready_via_local(tmp_path):
    """No key, no prior setup: local transcription makes this READY, not blocked.

    This is the local-first contract — a Whisper API key is never required.
    """
    _write_env(tmp_path, "GROQ_API_KEY=\nOPENAI_API_KEY=\n")
    chk = _run(["--check"], home=tmp_path)
    assert chk.returncode == 0, f"keyless first run should pass --check; got {chk.returncode}: {chk.stderr}"
    assert chk.stdout == "" and chk.stderr == ""

    js = json.loads(_run(["--json"], home=tmp_path).stdout)
    assert js["status"] == "ready"
    assert js["can_proceed"] is True
    assert js["local_asr"] is True
    assert js["whisper_backend"] == "local"
    assert js["has_api_key"] is False


@needs_uv
def test_keyless_completed_setup_proceeds_silently(tmp_path):
    """A user who finished setup without a key runs silently and ready."""
    _write_env(tmp_path, "GROQ_API_KEY=\nOPENAI_API_KEY=\nSETUP_COMPLETE=true\n")
    chk = _run(["--check"], home=tmp_path)
    assert chk.returncode == 0, f"keyless-complete should pass --check; got {chk.returncode}: {chk.stderr}"
    assert chk.stdout == "" and chk.stderr == ""

    js = json.loads(_run(["--json"], home=tmp_path).stdout)
    assert js["can_proceed"] is True
    assert js["first_run"] is False
    assert js["setup_complete"] is True
    assert js["status"] == "ready"


@needs_uv
def test_key_present_still_defaults_to_local(tmp_path):
    """An optional cloud key is recorded, but local stays the default backend."""
    _write_env(tmp_path, "GROQ_API_KEY=sk-test-abc\n")
    chk = _run(["--check"], home=tmp_path)
    assert chk.returncode == 0, chk.stderr

    js = json.loads(_run(["--json"], home=tmp_path).stdout)
    assert js["status"] == "ready"
    assert js["can_proceed"] is True
    assert js["has_api_key"] is True
    assert js["whisper_backend"] == "local"  # local-first: key is opt-in, not default


def test_local_asr_override_disables_local():
    """WATCH_LOCAL_ASR=0 forces the capability off (used to simulate no-uv)."""
    js = json.loads(_run(["--json"], extra_env={"WATCH_LOCAL_ASR": "0"}).stdout)
    assert js["local_asr"] is False
