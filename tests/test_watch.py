import sys
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from watch import DEFAULT_MAX_FRAMES  # noqa: E402


class WatchDefaultsTests(unittest.TestCase):
    def test_default_matches_documented_long_video_budget(self) -> None:
        self.assertEqual(100, DEFAULT_MAX_FRAMES)


if __name__ == "__main__":
    unittest.main()
