"""Uniform extract() must SPREAD frames across the requested range when fps
and max_frames disagree, not truncate at the head.

Bug: extract() built ``-vf fps={fps}`` (samples evenly across the *whole*
input) plus ``-frames:v {max_frames}`` (ffmpeg stops after N output frames).
When ``fps * duration`` exceeds ``max_frames``, ffmpeg emits the FIRST N
samples — the head of the clip — while the result looks like a full-range
sample. The auto path (auto_fps / auto_fps_focus) never hits this because it
derives its target FROM max_frames, so fps*duration never exceeds the cap;
it's reachable only when the two disagree, most obviously an explicit fps
override.

frames.py already names this exact failure shape in
``extract_scene_or_uniform``'s docstring, for the scene engine: "capping
detection with -frames:v instead would keep only the first max_frames cuts
and drop the tail of long videos". This fix applies the same principle to
the uniform engine, which had never gotten it.
"""
from __future__ import annotations

from pathlib import Path

import frames

# Matches conftest.long_cut_clip: n=14 segments * seg=3.0s each.
LONG_CLIP_DURATION = 42.0


def test_conflicting_fps_and_max_frames_spread_full_video(long_cut_clip: Path, tmp_path: Path):
    """fps=2, max_frames=10 on a ~42s clip wants ~84 samples but is capped at
    10. The 10 produced frames must SPREAD across the full clip, not cluster
    at the head.

    Measured against the unfixed extract() (2026-09-17): the last frame lands
    at 4.5s — 10 frames at the fps=2 filter rate (0.5s apart), with
    -frames:v stopping the run long before it reaches the back of the clip.
    A correct implementation must land the last frame near the END of the
    clip instead, so the bound below (>75% of duration) is one the broken
    code cannot satisfy while a fix comfortably clears it.
    """
    out = frames.extract(str(long_cut_clip), tmp_path / "f", fps=2, max_frames=10)
    assert len(out) > 0, "no frames produced"
    assert len(out) <= 10, f"produced {len(out)} frames, cap was 10"

    last_ts = out[-1]["timestamp_seconds"]
    assert last_ts > LONG_CLIP_DURATION * 0.75, (
        f"last frame at {last_ts}s is not near the end of a "
        f"{LONG_CLIP_DURATION}s clip — frames are clustered at the head "
        f"instead of spread across the range"
    )

    ts = [f["timestamp_seconds"] for f in out]
    assert ts == sorted(ts), "timestamps not increasing"
    assert out[0]["timestamp_seconds"] < 5.0, "first frame should be near t=0"


def test_conflicting_fps_respects_explicit_range(long_cut_clip: Path, tmp_path: Path):
    """Same fps/max_frames conflict, scoped to an explicit --start/--end
    range: the spread must cover THAT range, not the full clip."""
    start, end = 6.0, 36.0  # 30s window
    out = frames.extract(
        str(long_cut_clip), tmp_path / "f", fps=2, max_frames=8,
        start_seconds=start, end_seconds=end,
    )
    assert len(out) > 0, "no frames produced"

    last_ts = out[-1]["timestamp_seconds"]
    assert last_ts > start + (end - start) * 0.75, (
        f"last frame at {last_ts}s is not near the end of the requested "
        f"{start}-{end}s range"
    )
    assert last_ts <= end + 1.0, f"last frame {last_ts}s overshoots requested end {end}s"
    assert out[0]["timestamp_seconds"] >= start - 0.5, (
        f"first frame {out[0]['timestamp_seconds']}s is before requested start {start}s"
    )


def test_non_conflicting_fps_is_unaffected(long_cut_clip: Path, tmp_path: Path):
    """When fps*duration is already within max_frames (the normal / auto-path
    shape), behavior must be untouched — every requested sample is taken at
    the requested rate, not silently slowed down."""
    out = frames.extract(str(long_cut_clip), tmp_path / "f", fps=0.2, max_frames=100)

    # 0.2 fps over ~42s wants ~9 samples (0, 5, 10, ..., 40), comfortably
    # under the cap.
    assert 7 <= len(out) <= 9, f"expected ~8-9 frames at fps=0.2 over 42s, got {len(out)}"

    ts = [f["timestamp_seconds"] for f in out]
    assert ts == sorted(ts)
    # Spacing between consecutive frames must match the REQUESTED fps
    # (1/0.2 = 5s apart) — proving the non-conflicting path is untouched.
    for a, b in zip(ts, ts[1:]):
        assert abs((b - a) - 5.0) < 0.15, f"frame spacing {b - a}s != requested 1/fps=5s"


# Snapshot of auto_fps / auto_fps_focus captured from the pre-fix
# implementation (2026-09-17). Neither function is touched by this fix, and
# the fix inside extract() must be a no-op on the auto path — it always
# derives target FROM max_frames, so fps*duration never exceeds the cap —
# so the fixed code must reproduce these exact values. This is the
# regression guard for "short videos stay dense, long videos stay capped".
AUTO_FPS_SNAPSHOT = {
    (5, 100): (2.0, 10),
    (10, 100): (1.2, 12),
    (20, 100): (1.0, 20),
    (45, 100): (0.8888888888888888, 40),
    (90, 100): (0.6666666666666666, 60),
    (200, 100): (0.4, 80),
    (400, 100): (0.2, 80),
    (900, 100): (0.1111111111111111, 100),
    (400, 40): (0.1, 40),
}

AUTO_FPS_FOCUS_SNAPSHOT = {
    (3, 100): (2.0, 6),
    (10, 100): (2.0, 20),
    (20, 100): (2.0, 40),
    (45, 100): (1.7777777777777777, 80),
    (90, 100): (1.1111111111111112, 100),
    (200, 100): (0.5, 100),
}


def test_auto_path_budgets_unchanged():
    """auto_fps / auto_fps_focus must return the exact same (fps, target)
    pairs before and after the fix. Any change here is a regression: short
    videos must stay dense and long videos must stay capped exactly as
    before."""
    for (duration, max_frames), expected in AUTO_FPS_SNAPSHOT.items():
        got = frames.auto_fps(duration, max_frames=max_frames)
        assert got == expected, (
            f"auto_fps({duration}, max_frames={max_frames}) = {got}, "
            f"expected {expected} (regression in the auto path)"
        )

    for (duration, max_frames), expected in AUTO_FPS_FOCUS_SNAPSHOT.items():
        got = frames.auto_fps_focus(duration, max_frames=max_frames)
        assert got == expected, (
            f"auto_fps_focus({duration}, max_frames={max_frames}) = {got}, "
            f"expected {expected} (regression in the auto path)"
        )


def test_scene_and_keyframe_engines_untouched(cut_clip: Path, tmp_path: Path):
    """This fix touches only the uniform extract() engine — the scene and
    keyframe engines (which already spread correctly, see
    extract_scene_or_uniform's own docstring) must behave exactly as
    before."""
    scene_out, scene_meta = frames.extract_scene_or_uniform(
        str(cut_clip), tmp_path / "scene", fps=2.0, target_frames=50, max_frames=5,
    )
    assert scene_meta["engine"] == "scene"
    assert scene_meta["fallback"] is False
    assert len(scene_out) == 5
    ts = [f["timestamp_seconds"] for f in scene_out]
    assert ts == sorted(ts)
    assert ts[-1] > 4.0  # spans the full ~5.6s clip, not just the head

    key_out, key_meta = frames.extract_keyframes(str(cut_clip), tmp_path / "key", max_frames=50)
    assert key_meta["engine"] == "keyframe"
    assert key_meta["fallback"] is False
    assert len(key_out) >= frames.KEYFRAME_MIN
