"""setup.py --json surfaces the resolved watch detail."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

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


def _load_setup_module():
    spec = importlib.util.spec_from_file_location("watch_setup_under_test", SETUP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeStat:
    def __init__(self, mode):
        self.st_mode = mode


class _FakePath:
    """Stands in for the .env path — _check_file_permissions only needs stat + str."""
    def __init__(self, mode):
        self._mode = mode

    def stat(self):
        return _FakeStat(self._mode)

    def __str__(self):
        return "/home/u/.config/watch/.env"


def test_permission_warning_fires_on_posix(monkeypatch, capsys):
    """The world-readable warning must still work where st_mode is meaningful."""
    mod = _load_setup_module()
    monkeypatch.setattr(mod.os, "name", "posix")
    mod._check_file_permissions(_FakePath(0o644))
    assert "readable by other users" in capsys.readouterr().err


def test_permission_warning_silent_on_posix_when_locked_down(monkeypatch, capsys):
    mod = _load_setup_module()
    monkeypatch.setattr(mod.os, "name", "posix")
    mod._check_file_permissions(_FakePath(0o600))
    assert capsys.readouterr().err == ""


def test_permission_warning_skipped_on_windows(monkeypatch, capsys):
    """st_mode on Windows is synthesized from the read-only bit, not the ACL, so
    0o644 there is not evidence of anything and chmod 600 cannot clear it."""
    mod = _load_setup_module()
    monkeypatch.setattr(mod.os, "name", "nt")
    mod._check_file_permissions(_FakePath(0o644))
    assert capsys.readouterr().err == ""


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
