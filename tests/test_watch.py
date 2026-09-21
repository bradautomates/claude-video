"""End-to-end routing of --detail through watch.py on a local clip."""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

WATCH = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts" / "watch.py"


def _run(clip: Path, *args: str, env_extra: dict | None = None) -> str:
    env = dict(os.environ)
    env.pop("WATCH_DETAIL", None)
    if env_extra:
        env.update(env_extra)
    proc = subprocess.run(
        [sys.executable, str(WATCH), str(clip), "--no-whisper", *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace", env=env,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_efficient_uses_keyframe_engine(cut_clip: Path):
    out = _run(cut_clip, "--detail", "efficient")
    assert "(keyframe" in out
    assert "**Detail:** efficient" in out


def test_balanced_uses_scene_engine(cut_clip: Path):
    out = _run(cut_clip, "--detail", "balanced")
    assert "(scene" in out
    assert "**Detail:** balanced" in out


def test_token_burner_uses_scene_engine(cut_clip: Path):
    out = _run(cut_clip, "--detail", "token-burner")
    assert "(scene" in out


def test_transcript_skips_frames(cut_clip: Path):
    out = _run(cut_clip, "--detail", "transcript")
    assert "skipped" in out
    assert "frame_0000.jpg" not in out


def test_flag_overrides_env(cut_clip: Path):
    out = _run(cut_clip, "--detail", "efficient", env_extra={"WATCH_DETAIL": "balanced"})
    assert "(keyframe" in out


def test_default_is_balanced(cut_clip: Path):
    out = _run(cut_clip)  # no flag, WATCH_DETAIL cleared
    assert "**Detail:** balanced" in out
    assert "(scene" in out


def test_timestamps_add_cue_frames_to_detail(cut_clip: Path):
    out = _run(cut_clip, "--detail", "balanced", "--timestamps", "1,3")
    assert "reason=transcript-cue" in out
    assert "reason=scene-change" in out  # detail frames still present (additive)


def test_timestamps_with_transcript_detail_is_cue_only(cut_clip: Path):
    out = _run(cut_clip, "--detail", "transcript", "--timestamps", "1,3")
    assert "reason=transcript-cue" in out
    assert "reason=scene-change" not in out
    assert "reason=keyframe" not in out


def _frame_lines(out: str) -> int:
    # The report prints native paths, so frame lines carry backslashes on
    # Windows and this counted zero of them. Normalize before matching.
    lines = out.replace("\\", "/").splitlines()
    return sum(1 for line in lines if "/frames/frame_" in line and "(t=" in line)


def test_dedup_collapses_static_by_default(static_clip: Path):
    out = _run(static_clip)  # solid blue → identical frames collapse to one
    assert "near-duplicate" in out
    assert _frame_lines(out) == 1


def test_no_dedup_preserves_static_frames(static_clip: Path):
    out = _run(static_clip, "--no-dedup")
    assert "near-duplicate" not in out
    assert _frame_lines(out) > 1


def _frames_line(out: str) -> str:
    line = next(ln for ln in out.splitlines() if ln.startswith("- **Frames:**"))
    return line


def test_fallback_line_names_the_detector_that_fell_short(static_clip: Path):
    """The engine already reads "uniform" on a fallback, so the note names the
    detector that came up short and how many candidates it found."""
    scene = _frames_line(_run(static_clip, "--detail", "balanced"))
    assert re.search(r"after only \d+ scene candidates?", scene), scene
    assert "uniform fallback" not in scene

    keyframe = _frames_line(_run(static_clip, "--detail", "efficient"))
    assert re.search(r"after only \d+ keyframe candidates?", keyframe), keyframe
    assert "uniform fallback" not in keyframe


def test_fallback_line_never_selects_more_than_it_had(static_clip: Path):
    for detail in ("balanced", "efficient"):
        line = _frames_line(_run(static_clip, "--detail", detail))
        m = re.search(r"\*\*Frames:\*\* (\d+) selected from (\d+) candidates", line)
        assert m, line
        selected, candidates = int(m.group(1)), int(m.group(2))
        assert candidates >= selected, line
        # No dedup note at all means nothing was dropped.
        dedup_note = re.search(r"(\d+) near-duplicate", line)
        dropped = int(dedup_note.group(1)) if dedup_note else 0
        assert candidates - dropped == selected, line
