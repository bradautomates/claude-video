"""Keyframe engine + preserved scene/uniform fallbacks."""
from __future__ import annotations

import subprocess
from pathlib import Path

import frames


def test_keyframe_engine_on_cut_clip(cut_clip: Path, tmp_path: Path):
    out, meta = frames.extract_keyframes(str(cut_clip), tmp_path / "f", max_frames=50)
    assert meta["engine"] == "keyframe"
    assert meta["fallback"] is False
    assert len(out) >= frames.KEYFRAME_MIN
    assert all(fr["reason"] == "keyframe" for fr in out)
    assert len(out) == len(list((tmp_path / "f").glob("frame_*.jpg")))


def test_keyframe_even_sampling_caps_and_spans(cut_clip: Path, tmp_path: Path):
    out, meta = frames.extract_keyframes(str(cut_clip), tmp_path / "f", max_frames=5)
    assert meta["engine"] == "keyframe"
    assert len(out) == 5
    assert meta["selected_count"] == 5
    assert meta["candidate_count"] > 5
    ts = [fr["timestamp_seconds"] for fr in out]
    assert ts == sorted(ts)
    assert ts[0] < ts[-1]  # spans first → last keyframe
    assert [fr["index"] for fr in out] == [0, 1, 2, 3, 4]


def test_keyframe_fallback_on_static_clip(static_clip: Path, tmp_path: Path):
    out, meta = frames.extract_keyframes(str(static_clip), tmp_path / "f", max_frames=50)
    assert meta["engine"] == "uniform"
    assert meta["fallback"] is True
    assert len(out) > 0
    assert all(fr["reason"] == "uniform" for fr in out)


def test_scene_engine_on_cut_clip(cut_clip: Path, tmp_path: Path):
    out, meta = frames.extract_scene_or_uniform(
        str(cut_clip), tmp_path / "f", fps=2.0, target_frames=50, max_frames=100,
    )
    assert meta["engine"] == "scene"
    assert meta["fallback"] is False
    assert len(out) >= frames.SCENE_MIN_FRAMES


def test_scene_even_sampling_caps_and_spans(cut_clip: Path, tmp_path: Path):
    """Over-cap scene detection must even-sample across the whole clip, not keep
    the first N cuts and drop the tail (the long-video coverage bug)."""
    out, meta = frames.extract_scene_or_uniform(
        str(cut_clip), tmp_path / "f", fps=2.0, target_frames=50, max_frames=5,
    )
    assert meta["engine"] == "scene"
    assert meta["fallback"] is False
    assert len(out) == 5
    assert meta["selected_count"] == 5
    assert meta["candidate_count"] > 5  # all cuts detected, then sampled down
    ts = [fr["timestamp_seconds"] for fr in out]
    assert ts == sorted(ts)
    assert ts[-1] > 4.0  # spans the full ~5.6s clip, not just the first ~1.6s
    assert len(out) == len(list((tmp_path / "f").glob("frame_*.jpg")))
    assert [fr["index"] for fr in out] == [0, 1, 2, 3, 4]


def test_scene_fallback_on_static_clip(static_clip: Path, tmp_path: Path):
    out, meta = frames.extract_scene_or_uniform(
        str(static_clip), tmp_path / "f", fps=2.0, target_frames=12, max_frames=100,
    )
    assert meta["engine"] == "uniform"
    assert meta["fallback"] is True


def test_frame_sync_flag_is_accepted_by_this_ffmpeg():
    """The frame-sync flag must be one the installed ffmpeg actually accepts.

    `-vsync` was removed in ffmpeg 8, so hardcoding it aborted every scene and
    keyframe extraction with "Unrecognized option 'vsync'". Hardcoding
    `-fps_mode` instead would break ffmpeg 4.x, which predates it. Guard the
    resolved flag against the real binary so neither regression can return.
    """
    args = frames._frame_sync_args()
    assert args in (["-fps_mode", "vfr"], ["-vsync", "vfr"])

    probe = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "nullsrc=s=16x16:d=0.04",
            *args, "-frames:v", "1", "-f", "null", "-",
        ],
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0, f"ffmpeg rejected {args}: {probe.stderr.strip()}"


def test_frame_sync_probe_failure_is_not_cached(monkeypatch):
    """A failed probe must not pin the fallback for the rest of the process.

    ffmpeg can be absent when the first call happens and present later — a test
    harness adjusting PATH, or a caller that installs it on demand. Caching the
    fallback on OSError would keep returning `-vsync` on ffmpeg 8+, which is the
    exact breakage this probe exists to avoid.
    """
    monkeypatch.setattr(frames, "_FRAME_SYNC_ARGS", None)

    def _boom(*_args, **_kwargs):
        raise OSError("ffmpeg not found")

    monkeypatch.setattr(frames.subprocess, "run", _boom)
    assert frames._frame_sync_args() == ["-vsync", "vfr"]
    assert frames._FRAME_SYNC_ARGS is None, "failed probe must not be cached"

    monkeypatch.setattr(
        frames.subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(a[0] if a else [], 0),
    )
    assert frames._frame_sync_args() == ["-fps_mode", "vfr"]
    assert frames._FRAME_SYNC_ARGS == ["-fps_mode", "vfr"], "success must be cached"

