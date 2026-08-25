"""setup.py preflight: dependency status and Whisper backend detection."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

SETUP = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts" / "setup.py"


def _run(args, *, home=None, extra_env=None, local_whisper=None):
    """Invoke setup.py in a subprocess with a controlled environment.

    `local_whisper` forces whether faster-whisper appears installed. Leaving it
    None inherits the host's real state, which makes a test's result depend on
    the developer's machine — so every test that asserts on transcription
    availability sets it explicitly.
    """
    env = dict(os.environ)
    env.pop("WATCH_DETAIL", None)
    # Don't let a real key in the developer's shell env leak into the test.
    env.pop("GROQ_API_KEY", None)
    env.pop("OPENAI_API_KEY", None)
    env.pop("SETUP_COMPLETE", None)
    env.pop("WATCH_WHISPER", None)
    if home is not None:
        env["HOME"] = str(home)
        env["USERPROFILE"] = str(home)  # Windows
    if local_whisper is not None:
        shim = _shim_dir(local_whisper)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(shim)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
        )
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, str(SETUP), *args],
        capture_output=True, text=True, env=env,
    )


_SHIM_CACHE: dict[bool, Path] = {}


def _shim_dir(present: bool) -> Path:
    """A sys.path entry whose faster_whisper module is present or explodes.

    Shadowing the real package is the only way to test both branches on one
    machine: the module is imported in a subprocess, so monkeypatch can't reach
    it, and relying on whether the host happens to have faster-whisper installed
    makes the suite pass or fail for reasons unrelated to the code.
    """
    if present in _SHIM_CACHE:
        return _SHIM_CACHE[present]
    import tempfile

    d = Path(tempfile.mkdtemp(prefix=f"fw-shim-{present}-"))
    body = (
        "class WhisperModel:\n    pass\n"
        if present
        else "raise ImportError('faster_whisper not installed (test shim)')\n"
    )
    (d / "faster_whisper.py").write_text(body, encoding="utf-8")
    _SHIM_CACHE[present] = d
    return d


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
    chk = _run(["--check"], home=tmp_path, local_whisper=False)
    assert chk.returncode == 0, f"keyless-complete should pass --check; got {chk.returncode}: {chk.stderr}"
    assert chk.stdout == "" and chk.stderr == ""

    js = json.loads(_run(["--json"], home=tmp_path, local_whisper=False).stdout)
    assert js["can_proceed"] is True
    assert js["first_run"] is False
    assert js["setup_complete"] is True
    # status still encourages a backend even though we can proceed
    assert js["status"] == "needs_key"


def test_keyless_first_run_is_encouraged(tmp_path):
    """First run with no key and no local backend: exit 3 (suggest one)."""
    _write_env(tmp_path, "GROQ_API_KEY=\nOPENAI_API_KEY=\n")
    chk = _run(["--check"], home=tmp_path, local_whisper=False)
    assert chk.returncode == 3, chk.stderr
    assert "faster-whisper" in chk.stderr

    js = json.loads(_run(["--json"], home=tmp_path, local_whisper=False).stdout)
    assert js["can_proceed"] is False
    assert js["first_run"] is True
    assert js["has_transcription"] is False


def test_key_present_is_ready(tmp_path):
    _write_env(tmp_path, "GROQ_API_KEY=sk-test-abc\n")
    chk = _run(["--check"], home=tmp_path, local_whisper=False)
    assert chk.returncode == 0, chk.stderr

    js = json.loads(_run(["--json"], home=tmp_path, local_whisper=False).stdout)
    assert js["status"] == "ready"
    assert js["can_proceed"] is True
    assert js["whisper_backend"] == "groq"


def test_local_backend_alone_is_ready(tmp_path):
    """faster-whisper with no key is a complete install, not a half-configured one."""
    _write_env(tmp_path, "GROQ_API_KEY=\nOPENAI_API_KEY=\n")
    chk = _run(["--check"], home=tmp_path, local_whisper=True)
    assert chk.returncode == 0, chk.stderr
    assert chk.stdout == "" and chk.stderr == ""

    js = json.loads(_run(["--json"], home=tmp_path, local_whisper=True).stdout)
    assert js["status"] == "ready"
    assert js["can_proceed"] is True
    assert js["has_api_key"] is False
    assert js["has_local_whisper"] is True
    assert js["whisper_backend"] == "local"


def test_api_key_named_over_local_in_status(tmp_path):
    """With both available, the reported backend matches resolve_backend()."""
    _write_env(tmp_path, "GROQ_API_KEY=sk-test-abc\n")
    js = json.loads(_run(["--json"], home=tmp_path, local_whisper=True).stdout)
    assert js["whisper_backend"] == "groq"
    assert js["has_local_whisper"] is True


def test_json_reports_whisper_setting(tmp_path):
    _write_env(tmp_path, "WATCH_WHISPER=local\n")
    js = json.loads(_run(["--json"], home=tmp_path, local_whisper=True).stdout)
    assert js["whisper_setting"] == "local"
