"""Keyframe engine + preserved scene/uniform fallbacks."""
from __future__ import annotations

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


def test_fps_mode_flag_matches_local_ffmpeg():
    """The probe agrees with the ffmpeg actually on PATH (it just ran the fixtures)."""
    assert frames._fps_mode_flag() in {"-fps_mode", "-vsync"}


def test_fps_mode_flag_picks_spelling_by_major_version(monkeypatch):
    """ffmpeg <5 only knows -vsync; 5+ knows -fps_mode and 8+ dropped -vsync."""
    cases = [
        ("ffmpeg version 4.4.2-0ubuntu0.22.04.1 Copyright (c) 2000-2021", "-vsync"),
        ("ffmpeg version n4.3.1 Copyright (c) 2000-2020", "-vsync"),
        ("ffmpeg version 5.1.4 Copyright (c) 2000-2023", "-fps_mode"),
        ("ffmpeg version n6.0 Copyright (c) 2000-2023", "-fps_mode"),
        ("ffmpeg version 9.0.1 Copyright (c) 2000-2026", "-fps_mode"),
        # Git/dated snapshot builds carry no parseable major; assume modern.
        ("ffmpeg version N-113402-g1a2b3c4d Copyright (c) 2000-2026", "-fps_mode"),
    ]
    for banner, expected in cases:
        frames._fps_mode_flag.cache_clear()
        monkeypatch.setattr(
            frames.subprocess,
            "run",
            lambda *a, **k: type("R", (), {"stdout": banner})(),
        )
        assert frames._fps_mode_flag() == expected, banner
    frames._fps_mode_flag.cache_clear()
