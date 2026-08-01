import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from download import _load_info  # noqa: E402


class DownloadInfoTests(unittest.TestCase):
    def test_source_chapters_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out_dir = Path(directory)
            (out_dir / "video.info.json").write_text(json.dumps({
                "title": "Lesson",
                "duration": 120,
                "chapters": [
                    {"start_time": 0, "end_time": 60, "title": "Intro"},
                    {"start_time": 60, "end_time": 120, "title": "Demo"},
                ],
            }))

            info = _load_info(out_dir, "https://example.com/video")

        self.assertEqual(2, len(info["chapters"]))
        self.assertEqual("Intro", info["chapters"][0]["title"])
        self.assertEqual(60, info["chapters"][1]["start_time"])


if __name__ == "__main__":
    unittest.main()
