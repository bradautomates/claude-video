"""End-to-end routing of --detail through watch.py on a local clip."""
from __future__ import annotations

import json
import os
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
        capture_output=True, text=True, env=env,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def _manifest(out_dir: Path) -> dict:
    return json.loads((out_dir / "frame-index.json").read_text())


def test_efficient_uses_keyframe_engine(cut_clip: Path, tmp_path: Path):
    out = _run(cut_clip, "--detail", "efficient", "--out-dir", str(tmp_path))
    assert "(keyframe" in out
    assert "**Detail:** efficient" in out
    assert _manifest(tmp_path)["frame_count"] > 0


def test_balanced_uses_scene_engine(cut_clip: Path, tmp_path: Path):
    out = _run(cut_clip, "--detail", "balanced", "--out-dir", str(tmp_path))
    assert "(scene" in out
    assert "**Detail:** balanced" in out


def test_token_burner_uses_scene_engine(cut_clip: Path, tmp_path: Path):
    out = _run(cut_clip, "--detail", "token-burner", "--out-dir", str(tmp_path))
    assert "(scene" in out


def test_transcript_skips_frames(cut_clip: Path, tmp_path: Path):
    out = _run(cut_clip, "--detail", "transcript", "--out-dir", str(tmp_path))
    assert "skipped" in out
    assert "frame_0000.jpg" not in out
    assert not (tmp_path / "frame-index.json").exists()


def test_flag_overrides_env(cut_clip: Path, tmp_path: Path):
    out = _run(
        cut_clip, "--detail", "efficient", "--out-dir", str(tmp_path),
        env_extra={"WATCH_DETAIL": "balanced"},
    )
    assert "(keyframe" in out


def test_default_is_balanced(cut_clip: Path, tmp_path: Path):
    out = _run(cut_clip, "--out-dir", str(tmp_path))
    assert "**Detail:** balanced" in out
    assert "(scene" in out


def test_timestamps_add_cue_frames_to_detail(cut_clip: Path, tmp_path: Path):
    _run(
        cut_clip, "--detail", "balanced", "--timestamps", "1,3",
        "--out-dir", str(tmp_path),
    )
    reasons = {frame["reason"] for frame in _manifest(tmp_path)["frames"]}
    assert "transcript-cue" in reasons
    assert "scene-change" in reasons


def test_timestamps_with_transcript_detail_is_cue_only(cut_clip: Path, tmp_path: Path):
    _run(
        cut_clip, "--detail", "transcript", "--timestamps", "1,3",
        "--out-dir", str(tmp_path),
    )
    reasons = {frame["reason"] for frame in _manifest(tmp_path)["frames"]}
    assert reasons == {"transcript-cue"}


def test_focused_transcript_cues_preserve_semantics(cut_clip: Path, tmp_path: Path):
    _run(
        cut_clip, "--detail", "transcript", "--timestamps", "0.5,2,4",
        "--start", "1", "--end", "3", "--out-dir", str(tmp_path),
    )
    frames = _manifest(tmp_path)["frames"]
    assert [(frame["timestamp_seconds"], frame["reason"]) for frame in frames] == [
        (2.0, "transcript-cue")
    ]


def test_stdout_lists_no_individual_frame_paths(cut_clip: Path, tmp_path: Path):
    out = _run(cut_clip, "--detail", "balanced", "--out-dir", str(tmp_path))
    assert "/frames/frame_" not in out
    assert "/frames/cue_" not in out
    assert "frame-index.json" in out
    assert "overview_0001.jpg" in out


def test_dedup_collapses_static_by_default(static_clip: Path, tmp_path: Path):
    out = _run(static_clip, "--out-dir", str(tmp_path))
    assert "near-duplicate" in out
    assert _manifest(tmp_path)["frame_count"] == 1


def test_no_dedup_preserves_static_frames(static_clip: Path, tmp_path: Path):
    out = _run(static_clip, "--no-dedup", "--out-dir", str(tmp_path))
    assert "near-duplicate" not in out
    assert _manifest(tmp_path)["frame_count"] > 1
