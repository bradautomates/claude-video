"""Structural boundaries pass: fades, freezes, silence, and the post-fade pins.

The premise under test is that `blackdetect`/`freezedetect` see edits that
`select=gt(scene,…)` structurally cannot — so several of these tests assert the
scene detector's *failure* alongside the boundary detector's success. If a future
change makes scene detection catch fades too, those assertions should be
revisited rather than deleted: the point is the comparison, not the miss.
"""
from __future__ import annotations

from pathlib import Path

import boundaries
import frames


def test_detects_the_fade_the_scene_engine_misses(clustered_clip: Path, tmp_path: Path):
    """The mid-clip fade produces a black interval; no scene cut lands near it."""
    result = boundaries.detect_boundaries(str(clustered_clip))
    assert result.get("error") is None
    assert result["black"], "the fade through black was not detected"

    fade = result["black"][0]
    assert fade["end"] > fade["start"]
    assert fade["duration"] >= boundaries.BLACK_MIN_DURATION

    cuts = [
        f["timestamp_seconds"]
        for f in frames.extract_scene_candidates(str(clustered_clip), tmp_path / "s", max_frames=None)
    ]
    assert cuts, "sanity: the detector should still find the head cuts"
    # Every scene cut sits in the card section; none within a second of the fade.
    assert all(abs(c - fade["start"]) > 1.0 for c in cuts), (
        f"expected no scene cut near the fade at {fade['start']}s, got {cuts}"
    )


def test_freeze_detects_held_cards(clustered_clip: Path):
    """A held card changes nothing, so only an interval detector can see it."""
    result = boundaries.detect_boundaries(str(clustered_clip))
    assert len(result["freeze"]) >= 5
    assert all(seg["start"] is not None for seg in result["freeze"])


def test_silence_skipped_without_audio_stream(clustered_clip_silent: Path):
    """`-af` on a video with no audio makes ffmpeg exit 1; it must not be sent."""
    result = boundaries.detect_boundaries(str(clustered_clip_silent), has_audio=False)
    assert result.get("error") is None
    assert result["silence"] == []
    assert result["has_audio"] is False
    assert result["black"], "video detection must still work with audio off"


def test_times_are_absolute_under_a_focus_window(clustered_clip: Path):
    """`-ss` shifts ffmpeg's clock; reported times must be back on source time."""
    full = boundaries.detect_boundaries(str(clustered_clip))
    assert full["black"]
    fade_start = full["black"][0]["start"]

    window = boundaries.detect_boundaries(
        str(clustered_clip), start_seconds=fade_start - 2.0, end_seconds=fade_start + 3.0
    )
    assert window.get("error") is None
    assert window["black"], "the fade should still be inside the window"
    assert abs(window["black"][0]["start"] - fade_start) < 0.5, (
        f"windowed start {window['black'][0]['start']} is not absolute "
        f"(full-clip start was {fade_start})"
    )


def test_post_fade_pin_lands_after_the_fade(clustered_clip: Path):
    result = boundaries.detect_boundaries(str(clustered_clip))
    points = boundaries.post_fade_timestamps(result)
    assert points
    fade = result["black"][0]
    assert points[0] == round(fade["end"] + boundaries.POST_FADE_OFFSET, 2)
    assert points[0] > fade["end"]


def test_post_fade_skips_a_fade_that_never_ends():
    """Fading out to end of video has no incoming shot — nothing to pin."""
    unterminated = {"black": [{"start": 10.0, "end": None, "duration": None}]}
    assert boundaries.post_fade_timestamps(unterminated) == []


def test_post_fade_skips_points_past_the_duration():
    data = {"black": [{"start": 5.0, "end": 9.9, "duration": 4.9}]}
    assert boundaries.post_fade_timestamps(data, duration=10.0) == []
    assert boundaries.post_fade_timestamps(data, duration=30.0) == [10.4]


def test_post_fade_limit_keeps_the_longest_fades_in_time_order():
    """A tight budget should buy the strongest section breaks, not the first N."""
    data = {"black": [
        {"start": 1.0, "end": 1.2, "duration": 0.2},
        {"start": 50.0, "end": 53.0, "duration": 3.0},
        {"start": 20.0, "end": 22.0, "duration": 2.0},
    ]}
    # end + POST_FADE_OFFSET, for the two longest fades, in chronological order.
    assert boundaries.post_fade_timestamps(data, limit=2) == [22.5, 53.5]


def test_pair_intervals_tolerates_an_open_segment():
    out = boundaries._pair_intervals([1.0, 5.0], [2.0])
    assert out[0] == {"start": 1.0, "end": 2.0, "duration": 1.0}
    assert out[1] == {"start": 5.0, "end": None, "duration": None}


def test_missing_ffmpeg_fails_open(monkeypatch):
    """A boundaries failure must degrade the report, never kill a paid-for run."""
    monkeypatch.setattr(boundaries.shutil, "which", lambda name: None)
    result = boundaries.detect_boundaries("whatever.mp4")
    assert result["black"] == [] and result["freeze"] == []
    assert "error" in result


def test_ffmpeg_failure_fails_open(monkeypatch, clustered_clip: Path):
    class Boom:
        returncode = 1
        stderr = "something broke"
        stdout = ""

    monkeypatch.setattr(boundaries.subprocess, "run", lambda *a, **k: Boom())
    result = boundaries.detect_boundaries(str(clustered_clip))
    assert result["black"] == []
    assert "something broke" in result["error"]


def test_timeline_renders_subsecond_times():
    """Whole-second rendering would erase the precision this section exists for."""
    data = {
        "black": [{"start": 93.33, "end": 94.5, "duration": 1.17}],
        "freeze": [],
        "silence": [],
        "has_audio": True,
    }
    lines = "\n".join(boundaries.format_timeline(data))
    assert "01:33.33" in lines
    assert "01:34.50" in lines


def test_timeline_renders_hours():
    data = {"black": [{"start": 3661.5, "end": 3662.0, "duration": 0.5}],
            "freeze": [], "silence": [], "has_audio": True}
    assert "1:01:01.50" in "\n".join(boundaries.format_timeline(data))


def test_timeline_says_when_there_is_no_audio():
    data = {"black": [], "freeze": [], "silence": [], "has_audio": False}
    assert "no audio stream" in "\n".join(boundaries.format_timeline(data))


def test_timeline_distinguishes_no_audio_from_no_silence():
    """"No silence found" and "there was nothing to listen to" are different
    claims; collapsing them would let the report imply continuous sound over a
    silent file."""
    silent_file = {"black": [], "freeze": [], "silence": [], "has_audio": False}
    continuous = {"black": [], "freeze": [], "silence": [], "has_audio": True}
    assert "none detected" not in "\n".join(boundaries.format_timeline(silent_file))
    assert "none detected" in "\n".join(boundaries.format_timeline(continuous))


def test_timeline_truncates_long_lists():
    data = {
        "black": [{"start": float(i), "end": i + 0.5, "duration": 0.5} for i in range(60)],
        "freeze": [], "silence": [], "has_audio": True,
    }
    lines = boundaries.format_timeline(data, max_rows=10)
    assert any("and 50 more" in line for line in lines)
