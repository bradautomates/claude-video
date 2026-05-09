"""Tests for scripts/frames.py - pure-Python time parsing and fps budget logic.

These tests require no external binaries and run with just pytest.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from frames import (  # noqa: E402
    MAX_FPS,
    _clamp_fps,
    auto_fps,
    auto_fps_focus,
    format_time,
    parse_time,
)


# ---------------------------------------------------------------------------
# parse_time
# ---------------------------------------------------------------------------

class TestParseTime:
    def test_none_returns_none(self):
        assert parse_time(None) is None

    def test_empty_string_returns_none(self):
        assert parse_time("") is None

    def test_integer_passthrough(self):
        assert parse_time(30) == pytest.approx(30.0)

    def test_float_passthrough(self):
        assert parse_time(1.5) == pytest.approx(1.5)

    def test_seconds_only(self):
        assert parse_time("45") == pytest.approx(45.0)

    def test_seconds_with_decimal(self):
        assert parse_time("45.5") == pytest.approx(45.5)

    def test_mm_ss(self):
        assert parse_time("1:30") == pytest.approx(90.0)

    def test_mm_ss_with_decimal(self):
        assert parse_time("1:30.5") == pytest.approx(90.5)

    def test_hh_mm_ss(self):
        assert parse_time("1:02:03") == pytest.approx(3723.0)

    def test_hh_mm_ss_with_decimal(self):
        assert parse_time("1:02:03.5") == pytest.approx(3723.5)

    def test_large_hours(self):
        assert parse_time("2:00:00") == pytest.approx(7200.0)

    def test_invalid_raises_system_exit(self):
        with pytest.raises(SystemExit):
            parse_time("not-a-time")


# ---------------------------------------------------------------------------
# format_time
# ---------------------------------------------------------------------------

class TestFormatTime:
    def test_zero(self):
        assert format_time(0.0) == "00:00"

    def test_under_a_minute(self):
        assert format_time(45.0) == "00:45"

    def test_exactly_one_minute(self):
        assert format_time(60.0) == "01:00"

    def test_minutes_and_seconds(self):
        assert format_time(90.0) == "01:30"

    def test_one_hour(self):
        assert format_time(3600.0) == "1:00:00"

    def test_one_hour_thirty(self):
        assert format_time(5400.0) == "1:30:00"

    def test_rounding(self):
        # 59.6 rounds to 60, which is 01:00
        assert format_time(59.6) == "01:00"

    def test_roundtrip_with_parse_time(self):
        for t in [0, 30, 90, 3600, 3723]:
            assert parse_time(format_time(float(t))) == pytest.approx(float(t))


# ---------------------------------------------------------------------------
# _clamp_fps
# ---------------------------------------------------------------------------

class TestClampFps:
    def test_respects_max_fps(self):
        fps, _ = _clamp_fps(10.0, 60.0, 100)
        assert fps <= MAX_FPS

    def test_target_does_not_exceed_max_frames(self):
        _, target = _clamp_fps(2.0, 120.0, 50)
        assert target <= 50

    def test_target_is_at_least_one(self):
        _, target = _clamp_fps(0.001, 1.0, 100)
        assert target >= 1


# ---------------------------------------------------------------------------
# auto_fps (full-video budgets)
# ---------------------------------------------------------------------------

class TestAutoFps:
    def _check(self, duration, expected_max_target, max_frames=100):
        fps, target = auto_fps(duration, max_frames=max_frames)
        assert fps <= MAX_FPS, f"fps {fps} > MAX_FPS for duration {duration}"
        assert target <= max_frames
        assert target <= expected_max_target

    def test_very_short_video(self):
        fps, target = auto_fps(10.0)
        assert fps <= MAX_FPS
        assert target >= 10  # at least 1 frame per second for 10s clip

    def test_30_second_boundary(self):
        self._check(30.0, 30)

    def test_one_minute(self):
        self._check(60.0, 40)

    def test_three_minutes(self):
        self._check(180.0, 60)

    def test_ten_minutes(self):
        self._check(600.0, 80)

    def test_long_video_caps_at_max_frames(self):
        fps, target = auto_fps(3600.0, max_frames=100)
        assert target == 100

    def test_zero_duration(self):
        fps, target = auto_fps(0.0)
        assert target >= 1

    def test_custom_max_frames_respected(self):
        _, target = auto_fps(60.0, max_frames=20)
        assert target <= 20


# ---------------------------------------------------------------------------
# auto_fps_focus (focused / zoomed-in budgets)
# ---------------------------------------------------------------------------

class TestAutoFpsFocus:
    def test_always_at_most_max_fps(self):
        for duration in [1, 5, 15, 30, 60, 120, 300]:
            fps, _ = auto_fps_focus(float(duration))
            assert fps <= MAX_FPS, f"fps {fps} > MAX_FPS for duration {duration}"

    def test_denser_than_auto_fps_for_short_clips(self):
        """Focus mode should produce more frames per second than full-video mode."""
        for duration in [5.0, 15.0, 30.0]:
            fps_full, _ = auto_fps(duration)
            fps_focus, _ = auto_fps_focus(duration)
            assert fps_focus >= fps_full, (
                f"focus fps {fps_focus} < full fps {fps_full} for duration {duration}s"
            )

    def test_zero_duration(self):
        fps, target = auto_fps_focus(0.0)
        assert target >= 1
        assert fps <= MAX_FPS

    def test_custom_max_frames_respected(self):
        _, target = auto_fps_focus(30.0, max_frames=10)
        assert target <= 10

    def test_very_short_clip_gets_dense_coverage(self):
        fps, target = auto_fps_focus(3.0)
        assert target >= 10  # at least 10 frames for a 3-second clip
