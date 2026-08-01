import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from media_analysis import (  # noqa: E402
    _detect_band_onsets,
    _parse_frame_metadata,
    _parse_silences,
    analyze_media,
    classify_visual_changes,
    format_precise,
    plan_strip_windows,
    plan_zoom_windows,
    should_analyze,
)


class AnalysisPlanningTests(unittest.TestCase):
    def test_auto_mode_targets_short_clips_and_focused_ranges(self) -> None:
        self.assertTrue(should_analyze("auto", 5.0))
        self.assertTrue(should_analyze("auto", 60.0))
        self.assertFalse(should_analyze("auto", 60.01))
        self.assertTrue(should_analyze("cinematic", 600.0))
        self.assertFalse(should_analyze("standard", 5.0))

    def test_five_second_clip_keeps_one_second_motion_strips(self) -> None:
        windows = plan_strip_windows(0.0, 5.0)

        self.assertEqual(5, len(windows))
        self.assertEqual((0.0, 1.0), windows[0])
        self.assertEqual((4.0, 5.0), windows[-1])

    def test_longer_range_stays_under_strip_cap(self) -> None:
        windows = plan_strip_windows(20.0, 80.0)

        self.assertLessEqual(len(windows), 15)
        self.assertEqual(20.0, windows[0][0])
        self.assertEqual(80.0, windows[-1][1])

    def test_precise_time_preserves_subsecond_events(self) -> None:
        self.assertEqual("00:05.37", format_precise(5.37))
        self.assertEqual("1:01:01.05", format_precise(3661.05))

    def test_zoom_windows_are_clamped_limited_and_non_overlapping(self) -> None:
        events = [
            {"time": time, "score": 30 - index, "changed_percent": 70, "kind": "cut_or_flash"}
            for index, time in enumerate((0.1, 0.4, 1.4, 2.5, 3.6, 4.8))
        ]

        windows = plan_zoom_windows(events, 0.0, 5.0, max_zooms=4)

        self.assertLessEqual(len(windows), 4)
        self.assertGreaterEqual(windows[0]["start"], 0.0)
        self.assertLessEqual(windows[-1]["end"], 5.0)
        for previous, following in zip(windows, windows[1:]):
            self.assertLessEqual(previous["end"], following["start"])


class AnalysisParsingTests(unittest.TestCase):
    def test_ffmpeg_metadata_is_grouped_by_timestamp(self) -> None:
        output = """\
frame:0 pts:0 pts_time:0
lavfi.signalstats.YAVG=1.25
frame:1 pts:1 pts_time:0.1
lavfi.signalstats.YAVG=9.5
"""

        points = _parse_frame_metadata(
            output,
            {"lavfi.signalstats.YAVG": "motion"},
            offset=3.0,
        )

        self.assertEqual(
            [
                {"time": 3.0, "motion": 1.25},
                {"time": 3.1, "motion": 9.5},
            ],
            points,
        )

    def test_silence_ranges_receive_source_timeline_offset(self) -> None:
        output = """\
[silencedetect] silence_start: 0.25
[silencedetect] silence_end: 0.75 | silence_duration: 0.5
"""

        ranges = _parse_silences(output, offset=10.0, end=12.0)

        self.assertEqual(
            [{"start": 10.25, "end": 10.75, "duration": 0.5}],
            ranges,
        )


class EvidenceClassificationTests(unittest.TestCase):
    def test_isolated_high_coverage_peak_is_cut_or_flash_like(self) -> None:
        points = [
            {"time": 0.0, "motion": 2.0, "changed_percent": 3.0},
            {"time": 0.1, "motion": 3.0, "changed_percent": 5.0},
            {"time": 0.2, "motion": 25.0, "changed_percent": 75.0},
            {"time": 0.3, "motion": 3.0, "changed_percent": 4.0},
            {"time": 0.4, "motion": 2.0, "changed_percent": 3.0},
        ]

        events = classify_visual_changes(points, median_motion=3.0, p90_motion=20.0)

        self.assertEqual(1, len(events))
        self.assertEqual("cut_or_flash", events[0]["kind"])
        self.assertEqual(75.0, events[0]["changed_percent"])

    def test_persistent_high_change_is_sustained_motion_or_transition(self) -> None:
        points = [
            {"time": 0.0, "motion": 2.0, "changed_percent": 3.0},
            {"time": 0.1, "motion": 12.0, "changed_percent": 50.0},
            {"time": 0.2, "motion": 18.0, "changed_percent": 60.0},
            {"time": 0.3, "motion": 25.0, "changed_percent": 75.0},
            {"time": 0.4, "motion": 20.0, "changed_percent": 65.0},
            {"time": 0.5, "motion": 15.0, "changed_percent": 55.0},
            {"time": 0.6, "motion": 2.0, "changed_percent": 3.0},
        ]

        events = classify_visual_changes(points, median_motion=3.0, p90_motion=20.0)

        self.assertEqual(1, len(events))
        self.assertEqual("sustained_motion_or_transition", events[0]["kind"])
        self.assertEqual(5, events[0]["active_frames"])

    def test_one_continuous_gesture_does_not_emit_repeated_local_peaks(self) -> None:
        points = [
            {"time": 0.0, "motion": 2.0, "changed_percent": 3.0},
            {"time": 0.1, "motion": 12.0, "changed_percent": 16.0},
            {"time": 0.2, "motion": 20.0, "changed_percent": 24.0},
            {"time": 0.3, "motion": 16.0, "changed_percent": 20.0},
            {"time": 0.4, "motion": 22.0, "changed_percent": 28.0},
            {"time": 0.5, "motion": 15.0, "changed_percent": 18.0},
            {"time": 0.6, "motion": 2.0, "changed_percent": 3.0},
        ]

        events = classify_visual_changes(points, median_motion=3.0, p90_motion=18.0)

        self.assertEqual(1, len(events))
        self.assertEqual("localized_sustained_motion", events[0]["kind"])
        self.assertEqual(0.4, events[0]["time"])

    def test_frequency_band_peaks_merge_into_one_broadband_onset(self) -> None:
        def band(peak_time: float, peak_rms: float) -> list[dict]:
            return [
                {"time": 0.0, "rms_db": -40.0},
                {"time": 0.1, "rms_db": -40.0},
                {"time": 0.2, "rms_db": -40.0},
                {"time": peak_time, "rms_db": peak_rms},
                {"time": 0.7, "rms_db": -40.0},
            ]

        events = _detect_band_onsets({
            "low": band(0.50, -22.0),
            "mid": band(0.52, -24.0),
            "high": band(0.49, -25.0),
        })

        self.assertEqual(1, len(events))
        self.assertEqual("broadband onset candidate", events[0]["label"])
        self.assertEqual(["low", "mid", "high"], events[0]["bands"])


def _supports_cinematic_filters() -> bool:
    if shutil.which("ffmpeg") is None:
        return False
    result = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True)
    filters = result.stdout + result.stderr
    return all(
        name in filters
        for name in ("aspectralstats", "showspectrumpic", "tblend", "tile", "blackframe")
    )


@unittest.skipUnless(_supports_cinematic_filters(), "ffmpeg cinematic filters unavailable")
class CinematicAnalysisIntegrationTests(unittest.TestCase):
    def test_real_short_clip_produces_motion_and_sound_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "fixture.mp4"
            subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "testsrc2=size=160x90:rate=20:duration=2",
                    "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000:duration=2",
                    "-c:v", "mpeg4", "-c:a", "aac", "-shortest", str(source),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            result = analyze_media(
                str(source),
                root / "analysis",
                has_video=True,
                has_audio=True,
                start=0.0,
                end=2.0,
                resolution=320,
            )

            self.assertFalse(result["errors"])
            self.assertTrue(result["video"]["strips"])
            self.assertTrue(Path(result["video"]["strips"][0]["path"]).exists())
            self.assertTrue(Path(result["audio"]["waveform_path"]).exists())
            self.assertTrue(Path(result["audio"]["spectrogram_path"]).exists())
            self.assertTrue(result["audio"]["transient_peaks"])
            self.assertIn("zoom_strips", result["video"])
            self.assertIn("band_onsets", result["audio"])

            cached = analyze_media(
                str(source),
                root / "analysis",
                has_video=True,
                has_audio=True,
                start=0.0,
                end=2.0,
                resolution=320,
            )
            self.assertTrue(cached["cache_hit"])


if __name__ == "__main__":
    unittest.main()
