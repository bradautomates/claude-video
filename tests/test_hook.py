"""SessionStart hook command survives a plugin root containing a space.

`${CLAUDE_PLUGIN_ROOT}` expands inside a shell command string, so an unquoted
root such as `~/Library/Application Support/...` or a username with a space splits
into two words and bash tries to execute the first half. From upstream 93822ce.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HOOK = ROOT / "hooks" / "scripts" / "check-setup.sh"


def _hook_command() -> str:
    data = json.loads((ROOT / "hooks" / "hooks.json").read_text(encoding="utf-8"))
    return data["hooks"]["SessionStart"][0]["hooks"][0]["command"]


def _spaced_plugin(tmp_path: Path) -> Path:
    plugin = tmp_path / "plugin with spaces"
    (plugin / "hooks" / "scripts").mkdir(parents=True)
    shutil.copy(HOOK, plugin / "hooks" / "scripts" / "check-setup.sh")
    return plugin


def _run(command: str, plugin: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [shutil.which("bash") or "bash", "-c", command],
        env={**os.environ, "CLAUDE_PLUGIN_ROOT": str(plugin)},
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )


pytestmark = pytest.mark.skipif(
    os.name == "nt" or shutil.which("bash") is None, reason="POSIX shell fixture"
)


def test_hook_command_runs_from_a_spaced_plugin_root(tmp_path: Path):
    result = _run(_hook_command(), _spaced_plugin(tmp_path))
    assert result.returncode == 0, result.stderr
    assert "No such file" not in result.stderr


def test_unquoted_command_would_have_failed(tmp_path: Path):
    """Positive control: prove the fixture actually exercises the bug, so the
    test above cannot pass merely because the path happened to have no space."""
    unquoted = "bash ${CLAUDE_PLUGIN_ROOT}/hooks/scripts/check-setup.sh"
    result = _run(unquoted, _spaced_plugin(tmp_path))
    assert result.returncode != 0
    assert "No such file" in result.stderr


def test_hook_command_quotes_the_plugin_root():
    assert '"${CLAUDE_PLUGIN_ROOT}/hooks/scripts/check-setup.sh"' in _hook_command()


# ---------------------------------------------------------------------------
# A keyless transcription backend counts as configured, as in setup.py: a
# configured parakeet was told to add a Groq/OpenAI key on every session start.
# ---------------------------------------------------------------------------


def _run_hook(home: Path, fake_bin_dir: Path, env_body: str) -> subprocess.CompletedProcess:
    cfg = home / ".config" / "watch"
    cfg.mkdir(parents=True)
    (cfg / ".env").write_text(env_body, encoding="utf-8")
    (cfg / ".env").chmod(0o600)
    env = {k: v for k, v in os.environ.items()
           if k not in ("GROQ_API_KEY", "OPENAI_API_KEY", "SETUP_COMPLETE",
                        "WATCH_WHISPER_BACKEND", "WATCH_WHISPER_BASE_URL")}
    env.update(HOME=str(home), PATH=f"{fake_bin_dir}{os.pathsep}{os.environ['PATH']}")
    return subprocess.run(
        [shutil.which("bash") or "bash", str(HOOK)],
        env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )


@pytest.mark.parametrize("body", [
    "WATCH_WHISPER_BACKEND=parakeet\n",
    "WATCH_WHISPER_BACKEND=cli\n",
    "WATCH_WHISPER_BASE_URL=http://localhost:8080\n",
])
def test_keyless_backend_is_not_told_to_add_a_key(tmp_path: Path, fake_bin_dir: Path, body: str):
    result = _run_hook(tmp_path, fake_bin_dir, "GROQ_API_KEY=\nOPENAI_API_KEY=\n" + body)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "/watch: ready."


def test_no_backend_still_gets_the_key_hint(tmp_path: Path, fake_bin_dir: Path):
    """Positive control: the hint itself is unchanged when nothing is configured."""
    result = _run_hook(tmp_path, fake_bin_dir, "GROQ_API_KEY=\nOPENAI_API_KEY=\n")
    assert result.returncode == 0, result.stderr
    assert "Add GROQ_API_KEY" in result.stdout
