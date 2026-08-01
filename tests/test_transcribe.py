import sys
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from transcribe import _dedupe, _looks_like_rolling_captions  # noqa: E402


class RollingCaptionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.segments = [
            {
                "start": 0.0,
                "end": 1.0,
                "text": "The quick brown fox jumps over",
            },
            {
                "start": 1.0,
                "end": 2.0,
                "text": "brown fox jumps over the lazy dog",
            },
            {
                "start": 2.0,
                "end": 3.0,
                "text": "the lazy dog then runs away",
            },
        ]

    def test_rolling_overlap_is_removed(self) -> None:
        cleaned = _dedupe(self.segments, rolling=True)

        self.assertEqual(
            [
                "The quick brown fox jumps over",
                "the lazy dog then runs away",
            ],
            [segment["text"] for segment in cleaned],
        )

    def test_manual_caption_overlap_is_preserved(self) -> None:
        cleaned = _dedupe(self.segments, rolling=False)

        self.assertEqual(
            [segment["text"] for segment in self.segments],
            [segment["text"] for segment in cleaned],
        )

    def test_rolling_track_is_detected(self) -> None:
        self.assertTrue(_looks_like_rolling_captions(self.segments))


if __name__ == "__main__":
    unittest.main()
