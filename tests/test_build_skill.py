"""Release bundle guard regressions."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "skills" / "watch" / "scripts" / "build-skill.sh"


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )


def _build_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    scripts = root / "skills" / "watch" / "scripts"
    references = root / "skills" / "watch" / "references"
    scripts.mkdir(parents=True)
    references.mkdir()
    (root / "skills/watch/SKILL.md").write_text("---\nname: watch\n---\n")
    (scripts / "visual_harness.py").write_text("print('watch')\n")
    (scripts / "_visual_runtime.py").write_text("RUNTIME = True\n")
    (references / "trusted-review.md").write_text("# Trusted review\n")
    (scripts / "build-skill.sh").write_bytes(BUILD.read_bytes())
    (scripts / "build-skill.sh").chmod(0o755)
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(
        root,
        "-c",
        "user.name=Watch",
        "-c",
        "user.email=watch@example.invalid",
        "commit",
        "-qm",
        "fixture",
    )
    return root


def test_build_rejects_untracked_files(tmp_path: Path) -> None:
    root = _build_repo(tmp_path)
    (root / "skills/watch/scripts/omitted.py").write_text("OMITTED = True\n")

    result = subprocess.run(
        ["bash", "skills/watch/scripts/build-skill.sh"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 1
    assert "working tree is dirty" in result.stderr
    assert not (root / "dist/watch.skill").exists()


def test_clean_build_contains_required_runtime_files(tmp_path: Path) -> None:
    root = _build_repo(tmp_path)

    result = subprocess.run(
        ["bash", "skills/watch/scripts/build-skill.sh"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    listing = subprocess.run(
        ["unzip", "-Z1", str(root / "dist/watch.skill")],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    ).stdout.splitlines()
    assert "watch/SKILL.md" in listing
    assert "watch/scripts/visual_harness.py" in listing
    assert "watch/scripts/_visual_runtime.py" in listing
    assert "watch/references/trusted-review.md" in listing
    assert "watch/scripts/build-skill.sh" not in listing
