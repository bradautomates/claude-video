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
