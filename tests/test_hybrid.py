"""Hybrid scene+uniform selection, and the cut list a fallback leaves behind.

The regression these guard is one shape: a montage whose hard cuts all sit in
its card sections while its long body only *fades*. The coverage check correctly
says the cuts do not span the range — but the old response was to delete them
and sample uniformly, throwing away the only frames that landed exactly on a
boundary and spending the whole budget re-showing what the cuts already showed.
A rejected detector still measured something real.
"""
from __future__ import annotations

from pathlib import Path

import frames


def _times(selected: list[dict]) -> list[float]:
    return [f["timestamp_seconds"] for f in selected]


def _reasons(selected: list[dict]) -> set[str]:
    return {f["reason"] for f in selected}


def test_clustered_cuts_trigger_the_hybrid_not_the_fallback(clustered_clip: Path, tmp_path: Path):
    meta_only = frames.extract_scene_candidates(str(clustered_clip), tmp_path / "probe", max_frames=None)
    cut_times = [f["timestamp_seconds"] for f in meta_only]
    # Preconditions: enough cuts to clear the floor, but bunched.
    assert len(cut_times) >= frames.SCENE_MIN_FRAMES
    assert not frames._covers_range(cut_times, 0.0, 34.0)

    fps, target = frames.auto_fps(34.0, max_frames=100)
    _, meta = frames.extract_scene_or_uniform(
        str(clustered_clip), tmp_path / "out", fps=fps, target_frames=target, max_frames=100
    )
    assert meta["engine"] == "scene+uniform"
    assert meta["hybrid"] is True
    assert meta["fallback"] is False, "keeping the cuts is not a fallback"
    assert meta["hybrid_reason"] == "coverage"


def test_hybrid_keeps_the_cuts_the_old_fallback_deleted(clustered_clip: Path, tmp_path: Path):
    """The whole point: scene-change frames survive into the selection."""
    fps, target = frames.auto_fps(34.0, max_frames=100)
    selected, meta = frames.extract_scene_or_uniform(
        str(clustered_clip), tmp_path / "out", fps=fps, target_frames=target, max_frames=100
    )
    assert "scene-change" in _reasons(selected)
    assert "uniform" in _reasons(selected), "and uniform still fills the rest"
    assert meta["pinned_count"] >= frames.SCENE_MIN_FRAMES
    assert meta["fill_count"] > 0
    for frame in selected:
        assert Path(frame["path"]).exists(), "a selected frame's JPEG must survive"


def test_hybrid_fill_reaches_past_the_cut_cluster(clustered_clip: Path, tmp_path: Path):
    """Uniform fill must cover the body, which is why it is there at all."""
    fps, target = frames.auto_fps(34.0, max_frames=100)
    selected, _ = frames.extract_scene_or_uniform(
        str(clustered_clip), tmp_path / "out", fps=fps, target_frames=target, max_frames=100
    )
    times = _times(selected)
    cuts = [f["timestamp_seconds"] for f in selected if f["reason"] in ("scene-change", "first-frame")]
    assert max(times) > max(cuts) + 5.0, (
        f"selection stops at {max(times)}s, barely past the cuts ending at {max(cuts)}s"
    )


def test_hybrid_respects_the_frame_cap(clustered_clip: Path, tmp_path: Path):
    fps, target = frames.auto_fps(34.0, max_frames=20)
    selected, meta = frames.extract_scene_or_uniform(
        str(clustered_clip), tmp_path / "out", fps=fps, target_frames=target, max_frames=20
    )
    assert len(selected) <= 20
    assert meta["selected_count"] == len(selected)


def test_hybrid_counts_reconcile(clustered_clip: Path, tmp_path: Path):
    """candidates - dropped == selected, the arithmetic PR #224 fixed."""
    fps, target = frames.auto_fps(34.0, max_frames=100)
    _, meta = frames.extract_scene_or_uniform(
        str(clustered_clip), tmp_path / "out", fps=fps, target_frames=target, max_frames=100
    )
    assert meta["candidate_count"] - meta["deduped_count"] == meta["selected_count"]


def test_hybrid_output_is_chronological_and_reindexed(clustered_clip: Path, tmp_path: Path):
    """Two engines write two filename prefixes; the merged list must still read
    in time order with a clean 0..n-1 index, or frame N's caption lies."""
    fps, target = frames.auto_fps(34.0, max_frames=100)
    selected, _ = frames.extract_scene_or_uniform(
        str(clustered_clip), tmp_path / "out", fps=fps, target_frames=target, max_frames=100
    )
    times = _times(selected)
    assert times == sorted(times)
    assert [f["index"] for f in selected] == list(range(len(selected)))


def test_hybrid_has_no_duplicate_instants(clustered_clip: Path, tmp_path: Path):
    """The uniform grid spans the whole range, so it lands on moments the cuts
    already pinned (t=0.0 always). Those cost tokens and show nothing new."""
    fps, target = frames.auto_fps(34.0, max_frames=100)
    selected, _ = frames.extract_scene_or_uniform(
        str(clustered_clip), tmp_path / "out", fps=fps, target_frames=target, max_frames=100
    )
    times = _times(selected)
    assert len(times) == len(set(times))


def test_uniform_fill_does_not_delete_the_pinned_cuts(clustered_clip: Path, tmp_path: Path):
    """extract() wipes its own prefix on entry. If both engines wrote
    frame_*.jpg, the fill pass would erase every cut — the original bug."""
    out = tmp_path / "out"
    fps, target = frames.auto_fps(34.0, max_frames=100)
    selected, _ = frames.extract_scene_or_uniform(
        str(clustered_clip), out, fps=fps, target_frames=target, max_frames=100
    )
    kept = {Path(f["path"]).name for f in selected if f["reason"] in ("scene-change", "first-frame")}
    on_disk = {p.name for p in out.glob("*.jpg")}
    assert kept <= on_disk, f"pinned frames missing from disk: {kept - on_disk}"


def test_extract_prefix_isolates_two_engines(clustered_clip: Path, tmp_path: Path):
    """Directly: writing prefix B must not disturb files under prefix A."""
    out = tmp_path / "iso"
    first = frames.extract(str(clustered_clip), out, fps=1.0, max_frames=5, prefix="frame")
    second = frames.extract(str(clustered_clip), out, fps=1.0, max_frames=5, prefix="fill")
    assert first and second
    assert all(Path(f["path"]).exists() for f in first), "prefix 'fill' deleted prefix 'frame'"
    assert {Path(f["path"]).name.split("_")[0] for f in second} == {"fill"}


def test_dedup_protect_keeps_a_pinned_duplicate():
    """A pinned frame was chosen for WHEN it is; an identical neighbour is not
    grounds to delete it."""
    a, b = b"\x00" * 768, b"\x00" * 768  # identical thumbnails
    cands = [
        {"index": 0, "timestamp_seconds": 0.0, "path": "/nonexistent/a.jpg", "reason": "uniform"},
        {"index": 1, "timestamp_seconds": 1.0, "path": "/nonexistent/b.jpg", "reason": "scene-change"},
    ]
    kept, dropped = frames._dedupe_by_deltas(cands, [a, b], protect={"scene-change"})
    assert dropped == 0 and len(kept) == 2

    kept, dropped = frames._dedupe_by_deltas(
        [dict(c) for c in cands], [a, b], protect=None
    )
    assert dropped == 1, "without protection the duplicate is still dropped"


def test_drop_near_pinned_radius_is_half_the_fill_interval():
    fill = [{"index": i, "timestamp_seconds": float(i * 10), "path": f"/nonexistent/{i}.jpg",
             "reason": "uniform"} for i in range(5)]  # 0,10,20,30,40 → radius 5.0
    kept, dropped = frames._drop_near_pinned(fill, [20.0])
    assert dropped == 1
    assert 20.0 not in [f["timestamp_seconds"] for f in kept]
    assert [f["index"] for f in kept] == list(range(len(kept)))


def test_drop_near_pinned_is_a_noop_without_pins():
    fill = [{"index": 0, "timestamp_seconds": 1.0, "path": "/nonexistent/x.jpg", "reason": "uniform"}]
    kept, dropped = frames._drop_near_pinned(fill, [])
    assert dropped == 0 and kept == fill


def test_sparse_fallback_still_reports_where_the_cuts_were(static_clip: Path, tmp_path: Path):
    """Too few cuts to keep, but where they were is still evidence."""
    fps, target = frames.auto_fps(3.0, max_frames=20)
    _, meta = frames.extract_scene_or_uniform(
        str(static_clip), tmp_path / "out", fps=fps, target_frames=target, max_frames=20
    )
    assert meta["fallback"] is True
    assert meta["fallback_reason"] == "sparse"
    assert "scene_times" in meta, "the rejected detector's timeline must survive"


def test_covered_scene_path_also_reports_its_times(cut_clip: Path, tmp_path: Path):
    """scene_times is present on every scene-engine result, not just failures."""
    fps, target = frames.auto_fps(5.6, max_frames=50)
    _, meta = frames.extract_scene_or_uniform(
        str(cut_clip), tmp_path / "out", fps=fps, target_frames=target, max_frames=50
    )
    assert meta["scene_times"]
    assert meta["scene_times"] == sorted(meta["scene_times"])
