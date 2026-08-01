import sys
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from frames import plan_timestamps  # noqa: E402


class PlanTimestampsTests(unittest.TestCase):
    def test_busy_intro_cannot_starve_later_timeline(self) -> None:
        scenes = [(float(i) / 2, 1.0 - i / 1000) for i in range(1, 40)]

        timestamps = plan_timestamps(scenes, start=0.0, end=60.0, target=20)

        self.assertEqual(20, len(timestamps))
        self.assertEqual(0.0, timestamps[0])
        self.assertTrue(any(timestamp >= 48.0 for timestamp in timestamps))
        gaps = [
            right - left
            for left, right in zip(timestamps, timestamps[1:] + [60.0])
        ]
        self.assertLessEqual(max(gaps), 12.0)

    def test_timestamps_remain_sorted_and_unique(self) -> None:
        scenes = [(2.0, 0.9), (2.1, 0.8), (8.0, 0.7)]

        timestamps = plan_timestamps(scenes, start=0.0, end=10.0, target=8)

        self.assertEqual(timestamps, sorted(timestamps))
        self.assertEqual(len(timestamps), len(set(timestamps)))
        self.assertTrue(all(0.0 <= timestamp < 10.0 for timestamp in timestamps))


if __name__ == "__main__":
    unittest.main()
