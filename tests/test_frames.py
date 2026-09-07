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


def test_even_time_indices_spreads_over_clustered_candidates():
    """36 of 40 candidates packed into the first 6% of a 100s clip: index
    spacing keeps ~90% of the budget inside that 6%, time spacing must not."""
    times = [i * 0.17 for i in range(36)] + [40.0, 60.0, 80.0, 100.0]

    picked = [times[i] for i in frames._even_time_indices(times, 5)]
    assert picked == sorted(picked)
    assert picked[0] == times[0] and picked[-1] == times[-1]
    assert sum(1 for t in picked if t <= 6.0) <= 2  # index spacing gives 4

    # Contract kept from _even_indices.
    assert frames._even_time_indices(times, 99) == list(range(len(times)))
    assert frames._even_time_indices(times, 1) == [0]
    assert frames._even_time_indices([5.0] * 10, 3) == [0, 4, 9]  # flat → index


def test_extract_lowers_fps_so_cap_spans_the_range(cut_clip: Path, tmp_path: Path):
    """4 frames at 2fps covers only the opening 2s of a 5.6s clip; the cap has
    to stretch over the whole range instead of clumping at the head."""
    out = frames.extract(str(cut_clip), tmp_path / "f", fps=2.0, max_frames=4)
    assert len(out) == 4
    assert out[-1]["timestamp_seconds"] > 3.0
