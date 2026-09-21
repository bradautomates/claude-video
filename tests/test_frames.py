"""Keyframe engine + preserved scene/uniform fallbacks."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import frames


@pytest.fixture
def uncached_vfr_flag():
    """The flag probe is cached process-wide; isolate tests that fake it."""
    frames._vfr_flag.cache_clear()
    yield
    frames._vfr_flag.cache_clear()


def test_vfr_flag_prefers_fps_mode(monkeypatch, uncached_vfr_flag):
    monkeypatch.setattr(
        frames.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(stdout="  -fps_mode[:stream_specifier]  E..V.....\n", stderr=""),
    )
    assert frames._vfr_flag() == ("-fps_mode", "vfr")


def test_vfr_flag_falls_back_to_vsync(monkeypatch, uncached_vfr_flag):
    monkeypatch.setattr(
        frames.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(stdout="  -vsync <int>  E..V.....\n", stderr=""),
    )
    assert frames._vfr_flag() == ("-vsync", "vfr")


def test_vfr_flag_matches_installed_ffmpeg():
    """Whatever this ffmpeg is, the flag it gets must be one it accepts."""
    flag = frames._vfr_flag()
    assert flag in {("-fps_mode", "vfr"), ("-vsync", "vfr")}
    probe = frames.subprocess.run(
        ["ffmpeg", "-hide_banner", "-f", "lavfi", "-i", "testsrc=duration=0.2:rate=10",
         *flag, "-f", "null", "-"],
        capture_output=True,
        text=True, encoding="utf-8", errors="replace",
    )
    assert probe.returncode == 0, probe.stderr.strip()


def _assert_candidates_add_up(meta: dict, out: list) -> None:
    """A uniform fallback counts the uniform frames it actually extracted.

    The fallback caps in ``extract`` and never even-samples, so every candidate
    that survives dedup is selected and the three counts reconcile exactly.
    """
    assert meta["selected_count"] == len(out)
    assert meta["candidate_count"] >= meta["selected_count"]
    assert meta["candidate_count"] - meta["deduped_count"] == meta["selected_count"]


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
    assert meta["fallback_from"] == "keyframe"
    assert len(out) > 0
    assert all(fr["reason"] == "uniform" for fr in out)
    _assert_candidates_add_up(meta, out)


def test_keyframe_fallback_when_range_has_no_keyframes(static_clip: Path, tmp_path: Path):
    """A --start/--end window with no keyframes must fall back to uniform, not abort.

    ffmpeg fails at encoder init ("No filtered frames for output stream") rather
    than exiting 0 with no output, so a bare returncode check killed the whole
    run instead of reaching the too-few-keyframes fallback below it.
    """
    out, meta = frames.extract_keyframes(
        str(static_clip),
        tmp_path / "f",
        max_frames=50,
        start_seconds=1.0,
        end_seconds=3.0,
    )
    assert meta["engine"] == "uniform"
    assert meta["fallback"] is True
    assert len(out) > 0
    assert all(fr["timestamp_seconds"] >= 1.0 for fr in out)


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
    assert meta["fallback_from"] == "scene"
    _assert_candidates_add_up(meta, out)


def test_scene_fallback_candidates_hold_without_dedup(static_clip: Path, tmp_path: Path):
    """With dedup off nothing is dropped, so every uniform frame is selected."""
    out, meta = frames.extract_scene_or_uniform(
        str(static_clip), tmp_path / "f", fps=2.0, target_frames=12, max_frames=100,
        dedup=False,
    )
    assert meta["fallback"] is True
    assert meta["deduped_count"] == 0
    assert meta["candidate_count"] == meta["selected_count"] == len(out)


def test_keyframe_fallback_candidates_hold_without_dedup(static_clip: Path, tmp_path: Path):
    out, meta = frames.extract_keyframes(
        str(static_clip), tmp_path / "f", max_frames=50, dedup=False,
    )
    assert meta["fallback"] is True
    assert meta["deduped_count"] == 0
    assert meta["candidate_count"] == meta["selected_count"] == len(out)



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


def test_covers_range_rejects_clustered_candidates():
    """The real shape that motivated this: a 600s range where scene detection found
    23 candidates, 15 of them inside seven seconds, leaving a 3.4-minute hole."""
    clustered = [39.0, 39.5, 41.0, 41.5, 42.0, 43.0, 43.5, 44.0, 44.5, 45.0,
                 45.5, 46.0, 46.5, 63.0, 91.0, 100.0, 103.0, 112.0, 131.0, 336.0]
    assert frames._covers_range(clustered, 0.0, 600.0) is False

    spread = [i * 25.0 for i in range(1, 24)]
    assert frames._covers_range(spread, 0.0, 600.0) is True

    # Clears the count floor but still skips most of the range.
    assert frames._covers_range([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 0.0, 600.0) is False

    # Degenerate inputs never force a fallback.
    assert frames._covers_range([], 0.0, 600.0) is True
    assert frames._covers_range([1.0], 5.0, 5.0) is True


def test_worst_gap_counts_the_edges():
    assert frames._worst_gap([50.0], 0.0, 100.0) == 50.0
    assert frames._worst_gap([10.0, 20.0], 0.0, 100.0) == 80.0
    assert frames._worst_gap([], 0.0, 42.0) == 42.0


def test_metadata_via_ffmpeg_matches_ffprobe(cut_clip: Path):
    """When ffprobe is blocked (Windows App Control, #128) the ffmpeg banner
    must yield the same duration/size/codec/audio answers."""
    via_probe = frames.get_metadata(str(cut_clip))
    via_ffmpeg = frames._metadata_via_ffmpeg(str(cut_clip))
    assert abs(via_probe["duration_seconds"] - via_ffmpeg["duration_seconds"]) < 0.1
    assert (via_probe["width"], via_probe["height"]) == (via_ffmpeg["width"], via_ffmpeg["height"])
    assert via_probe["has_audio"] == via_ffmpeg["has_audio"]
    assert via_probe["codec"] == via_ffmpeg["codec"]


def test_get_metadata_falls_back_when_ffprobe_missing(cut_clip: Path, monkeypatch):
    real_which = frames.shutil.which
    monkeypatch.setattr(frames.shutil, "which", lambda n: None if n == "ffprobe" else real_which(n))
    meta = frames.get_metadata(str(cut_clip))
    assert meta["duration_seconds"] > 0 and meta["width"]
