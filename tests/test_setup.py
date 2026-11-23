"""setup.py --json surfaces the resolved watch detail."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SETUP = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts" / "setup.py"


@pytest.fixture(autouse=True)
def _stub_host_binaries(fake_bin_dir, monkeypatch):
    """Pretend every required binary is installed.

    These tests assert on the *key* state machine; whether the developer's
    machine has ffmpeg/yt-dlp is an unrelated variable that otherwise short
    -circuits `--check` with exit 2 before the key logic ever runs. `_run`
    copies `os.environ`, so prepending here is enough to reach the subprocess.
    """
    monkeypatch.setenv("PATH", f"{fake_bin_dir}{os.pathsep}{os.environ['PATH']}")


def _run(args, *, home=None, extra_env=None):
    env = dict(os.environ)
    env.pop("WATCH_DETAIL", None)
    # Don't let a real key in the developer's shell env leak into the test.
    env.pop("GROQ_API_KEY", None)
    env.pop("OPENAI_API_KEY", None)
    env.pop("SETUP_COMPLETE", None)
    for var in ("WATCH_WHISPER_BACKEND", "WATCH_WHISPER_BASE_URL",
                "WATCH_PARAKEET_CMD", "WATCH_TRANSCRIBE_CMD"):
        env.pop(var, None)
    if home is not None:
        env["HOME"] = str(home)
        env["USERPROFILE"] = str(home)  # Windows
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, str(SETUP), *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace", env=env,
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


def _stub_exe(bin_dir: Path, name: str) -> None:
    bin_dir.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        (bin_dir / f"{name}.bat").write_text("@echo off\r\nexit /b 0\r\n", encoding="utf-8")
        return
    stub = bin_dir / name
    stub.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    stub.chmod(0o755)


def _path_with(bin_dir: Path) -> dict:
    return {"PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}


def test_parakeet_backend_is_ready_without_a_key(tmp_path):
    """A configured on-device backend is a transcriber: no key, no nag.

    The key-only check reported `needs_key` / exit 3 here on a first run.
    WATCH_PARAKEET_CMD points at a stub so the host's own parakeet-mlx (or
    its absence) cannot decide the result.
    """
    _stub_exe(tmp_path / "bin", "stub-parakeet")
    _write_env(tmp_path, "GROQ_API_KEY=\nOPENAI_API_KEY=\nWATCH_WHISPER_BACKEND=parakeet\n"
                         "WATCH_PARAKEET_CMD=stub-parakeet {audio} {out_dir}\n")
    chk = _run(["--check"], home=tmp_path, extra_env=_path_with(tmp_path / "bin"))
    assert chk.returncode == 0, chk.stderr
    assert chk.stdout == "" and chk.stderr == ""

    js = json.loads(_run(["--json"], home=tmp_path, extra_env=_path_with(tmp_path / "bin")).stdout)
    assert js["status"] == "ready"
    assert js["can_proceed"] is True
    assert js["whisper_backend"] == "parakeet"
    assert js["has_transcriber"] is True
    assert js["has_api_key"] is False  # still the literal key check


def test_missing_parakeet_names_the_install_not_a_key(tmp_path):
    _write_env(tmp_path, "WATCH_WHISPER_BACKEND=parakeet\n"
                         "WATCH_PARAKEET_CMD=no-such-parakeet {audio} {out_dir}\n")
    chk = _run(["--check"], home=tmp_path)
    assert chk.returncode == 3
    assert "no-such-parakeet is not on PATH" in chk.stderr
    assert "GROQ_API_KEY" not in chk.stderr

    js = json.loads(_run(["--json"], home=tmp_path).stdout)
    assert js["status"] == "needs_key"
    assert js["has_transcriber"] is False
    assert js["whisper_backend"] == "parakeet"


def test_local_server_is_ready_without_a_key(tmp_path):
    _write_env(tmp_path, "WATCH_WHISPER_BASE_URL=http://localhost:8080\n")
    chk = _run(["--check"], home=tmp_path)
    assert chk.returncode == 0, chk.stderr

    js = json.loads(_run(["--json"], home=tmp_path).stdout)
    assert js["status"] == "ready"
    assert js["whisper_backend"] == "local"


def test_local_backend_without_base_url_is_not_ready(tmp_path):
    _write_env(tmp_path, "GROQ_API_KEY=sk-test-abc\nWATCH_WHISPER_BACKEND=local\n")
    chk = _run(["--check"], home=tmp_path)
    assert chk.returncode == 3
    assert "WATCH_WHISPER_BASE_URL" in chk.stderr


def test_installer_completes_setup_for_parakeet(tmp_path):
    _stub_exe(tmp_path / "bin", "stub-parakeet")
    _write_env(tmp_path, "WATCH_WHISPER_BACKEND=parakeet\n"
                         "WATCH_PARAKEET_CMD=stub-parakeet {audio} {out_dir}\n")
    inst = _run([], home=tmp_path, extra_env=_path_with(tmp_path / "bin"))
    assert inst.returncode == 0, inst.stdout + inst.stderr
    assert "whisper backend: parakeet" in inst.stdout
    env_text = (tmp_path / ".config" / "watch" / ".env").read_text(encoding="utf-8")
    assert "SETUP_COMPLETE=true" in env_text


def test_config_write_is_atomic_and_private(tmp_path, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location("watch_setup_atomic", SETUP)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SETUP.parent))
    spec.loader.exec_module(mod)
    cfg = tmp_path / ".config" / "watch"
    monkeypatch.setattr(mod, "CONFIG_DIR", cfg)
    monkeypatch.setattr(mod, "CONFIG_FILE", cfg / ".env")
    (cfg).mkdir(parents=True)
    (cfg / ".env").write_text("GROQ_API_KEY=keep\n", encoding="utf-8")

    real_replace = mod.os.replace
    def boom(*a, **k):
        raise OSError("disk gone")
    monkeypatch.setattr(mod.os, "replace", boom)
    try:
        mod._write_setup_complete()
    except OSError:
        pass
    # An interrupted write leaves the old file intact and no temp file behind.
    assert (cfg / ".env").read_text(encoding="utf-8") == "GROQ_API_KEY=keep\n"
    assert [p.name for p in cfg.iterdir()] == [".env"]

    monkeypatch.setattr(mod.os, "replace", real_replace)
    mod._write_setup_complete()
    assert (cfg / ".env").read_text(encoding="utf-8").endswith("SETUP_COMPLETE=true\n")
    if os.name != "nt":
        assert (cfg / ".env").stat().st_mode & 0o777 == 0o600
