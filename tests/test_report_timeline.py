"""End-to-end: the Timeline section, post-fade pins, and detail-mode gating.

These assert what the *model reading the report* actually gets, which is the
only surface that matters — a boundary measured but not reported is a boundary
the answer cannot use.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

WATCH = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts" / "watch.py"


def _run(clip: Path, *args: str) -> str:
    env = dict(os.environ)
    env.pop("WATCH_DETAIL", None)
    proc = subprocess.run(
        [sys.executable, str(WATCH), str(clip), "--no-whisper", *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace", env=env,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_balanced_emits_a_timeline(clustered_clip: Path, tmp_path: Path):
    out = _run(clustered_clip, "--detail", "balanced", "--out-dir", str(tmp_path / "w"))
    assert "## Timeline" in out
    assert "Fades / cuts to black" in out
    assert "Held / frozen picture" in out


def test_timeline_times_are_subsecond(clustered_clip: Path, tmp_path: Path):
    """A whole-second timeline is no more precise than the frames it supplements."""
    out = _run(clustered_clip, "--detail", "balanced", "--out-dir", str(tmp_path / "w"))
    timeline = out.split("## Timeline", 1)[1].split("## Frames", 1)[0]
    rows = [line for line in timeline.splitlines() if line.strip().startswith("- 0")]
    assert rows, "no interval rows rendered"
    assert any("." in row.split("→")[0] for row in rows), f"times look whole-second: {rows[:3]}"


def test_post_fade_frames_are_pinned_and_labelled(clustered_clip: Path, tmp_path: Path):
    out = _run(clustered_clip, "--detail", "balanced", "--out-dir", str(tmp_path / "w"))
    assert "**Post-fade frames:**" in out
    assert "reason=post-fade" in out, "the frame list must name why the frame is there"


def test_efficient_skips_the_boundaries_pass(clustered_clip: Path, tmp_path: Path):
    """`efficient` sells speed; an extra full decode would contradict that."""
    out = _run(clustered_clip, "--detail", "efficient", "--out-dir", str(tmp_path / "w"))
    assert "## Timeline" not in out


def test_boundaries_can_be_forced_on_for_efficient(clustered_clip: Path, tmp_path: Path):
    out = _run(clustered_clip, "--detail", "efficient", "--boundaries", "--out-dir", str(tmp_path / "w"))
    assert "## Timeline" in out


def test_no_boundaries_flag_suppresses_the_section(clustered_clip: Path, tmp_path: Path):
    out = _run(clustered_clip, "--detail", "balanced", "--no-boundaries", "--out-dir", str(tmp_path / "w"))
    assert "## Timeline" not in out
    assert "**Post-fade frames:**" not in out


def test_transcript_detail_has_no_timeline(clustered_clip: Path, tmp_path: Path):
    """No video is decoded at transcript detail, so there is nothing to measure."""
    out = _run(clustered_clip, "--detail", "transcript", "--out-dir", str(tmp_path / "w"))
    assert "## Timeline" not in out


def test_silent_source_says_no_audio_rather_than_no_silence(
    clustered_clip_silent: Path, tmp_path: Path
):
    out = _run(clustered_clip_silent, "--detail", "balanced", "--out-dir", str(tmp_path / "w"))
    assert "no audio stream" in out
    assert "none detected" not in out


def test_hybrid_is_reported_as_such_not_as_a_fallback(clustered_clip: Path, tmp_path: Path):
    out = _run(clustered_clip, "--detail", "balanced", "--out-dir", str(tmp_path / "w"))
    assert "scene+uniform" in out
    assert "scene cuts pinned" in out
    assert "uniform fill" in out


def test_sparse_fallback_prints_the_cuts_it_discarded(static_clip: Path, tmp_path: Path):
    """"No frame there" is not "nothing happened there"."""
    out = _run(static_clip, "--detail", "balanced", "--out-dir", str(tmp_path / "w"))
    assert "Detected scene cuts (not sampled)" in out
    assert "boundary evidence, not as moments you have seen" in out


def test_cut_list_is_absent_when_frames_were_kept(clustered_clip: Path, tmp_path: Path):
    """The hybrid keeps its cuts as frames, so there is nothing unsampled to warn about."""
    out = _run(clustered_clip, "--detail", "balanced", "--out-dir", str(tmp_path / "w"))
    assert "Detected scene cuts (not sampled)" not in out


def test_focused_range_keeps_timeline_on_source_time(clustered_clip: Path, tmp_path: Path):
    """--start shifts ffmpeg's clock; a timeline on offset time would misdate
    every boundary against the transcript and the frame captions."""
    out = _run(
        clustered_clip, "--detail", "balanced", "--start", "8", "--end", "20",
        "--out-dir", str(tmp_path / "w"),
    )
    if "## Timeline" not in out:
        return  # nothing structural inside the window on this build
    timeline = out.split("## Timeline", 1)[1].split("## Frames", 1)[0]
    rows = [line for line in timeline.splitlines() if line.strip().startswith("- 0")]
    for row in rows:
        mmss = row.strip().lstrip("- ").split(" ")[0]
        minutes, seconds = mmss.split(":")
        assert int(minutes) * 60 + float(seconds) >= 7.0, f"{row} predates the window"
