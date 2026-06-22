#!/usr/bin/env python3
"""Unit tests for the TwelveLabs provider helpers (pure stdlib, no pytest).

Run: python3 tests/test_provider.py
These cover the pieces with real logic — chunk planning, segment-list parsing,
prompt construction, and the shape-tolerant analyze-result extractor. The HTTP
calls themselves are exercised by the live smoke test, not mocked here.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import chunk as chunkmod  # noqa: E402
import twelvelabs as tl  # noqa: E402
import watch  # noqa: E402

MB = 1024 * 1024


class ChunkPlanning(unittest.TestCase):
    def test_no_chunk_when_short_and_small(self):
        # 10 min, 50 MB, 30-min threshold → single asset.
        self.assertFalse(chunkmod.needs_chunking(600, 50 * MB, 1800))

    def test_chunk_when_over_duration(self):
        # 45 min, small file → chunk by duration.
        self.assertTrue(chunkmod.needs_chunking(2700, 50 * MB, 1800))

    def test_chunk_when_over_size(self):
        # 20 min but 300 MB → chunk to fit the 200 MB upload cap.
        self.assertTrue(chunkmod.needs_chunking(1200, 300 * MB, 1800))

    def test_plan_uses_time_limit_when_bitrate_low(self):
        # Low bitrate: size never binds, so chunk length == requested 30 min.
        secs = chunkmod.plan_chunk_seconds(7200, 100 * MB, 1800)
        self.assertAlmostEqual(secs, 1800, delta=1)

    def test_plan_shrinks_chunks_for_high_bitrate(self):
        # 60 min @ ~1.2 GB → avg ~340 KB/s → ~190 MB fits ~556 s, well under 30 min.
        secs = chunkmod.plan_chunk_seconds(3600, 1200 * MB, 1800)
        self.assertLess(secs, 1800)
        self.assertGreater(secs, 60)
        # Each planned chunk must stay under the 190 MB cap by average bitrate.
        bytes_per_sec = (1200 * MB) / 3600
        self.assertLessEqual(secs * bytes_per_sec, chunkmod.MAX_DIRECT_UPLOAD_BYTES)

    def test_plan_never_below_min_clip(self):
        secs = chunkmod.plan_chunk_seconds(3600, 100_000 * MB, 1800)
        self.assertGreaterEqual(secs, chunkmod.MIN_CLIP_SECONDS)


class SegmentList(unittest.TestCase):
    def test_reads_absolute_offsets(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            for name in ("chunk_000.mp4", "chunk_001.mp4"):
                (out / name).write_bytes(b"x")
            (out / "segments.csv").write_text(
                "chunk_000.mp4,0.000000,1800.000000\n"
                "chunk_001.mp4,1800.000000,3600.000000\n",
                encoding="utf-8",
            )
            chunks = chunkmod._read_segment_list(out / "segments.csv", out)
            self.assertEqual(len(chunks), 2)
            self.assertEqual(chunks[0]["start_seconds"], 0.0)
            self.assertEqual(chunks[1]["start_seconds"], 1800.0)
            self.assertAlmostEqual(chunks[1]["duration"], 1800.0, delta=0.01)


class ExtractText(unittest.TestCase):
    def test_plain_string(self):
        self.assertEqual(tl._extract_text("hello"), "hello")

    def test_data_field(self):
        self.assertEqual(tl._extract_text({"data": "the analysis"}), "the analysis")

    def test_text_field_fallback(self):
        self.assertEqual(tl._extract_text({"text": "txt"}), "txt")

    def test_segments_list(self):
        result = {"segments": [{"text": "a"}, {"text": "b"}]}
        self.assertEqual(tl._extract_text(result), "a\nb")

    def test_empty(self):
        self.assertEqual(tl._extract_text(None), "")

    def test_unknown_dict_serializes(self):
        # Unrecognized shape should not crash — falls back to JSON so nothing is lost.
        out = tl._extract_text({"weird": 1})
        self.assertIn("weird", out)


class Prompt(unittest.TestCase):
    def test_default_has_sections(self):
        p = watch._tl_prompt(None)
        for section in ("## Transcript", "## Visual walkthrough", "## Summary"):
            self.assertIn(section, p)

    def test_question_is_embedded(self):
        p = watch._tl_prompt("what hook did they open with?")
        self.assertIn("what hook did they open with?", p)
        self.assertIn("## Transcript", p)


class ClampMaxTokens(unittest.TestCase):
    def test_pegasus15_in_range_unchanged(self):
        self.assertEqual(tl.clamp_max_tokens("pegasus1.5", 16384), 16384)

    def test_pegasus15_below_floor_clamped_up(self):
        self.assertEqual(tl.clamp_max_tokens("pegasus1.5", 500), 2048)

    def test_pegasus12_default_clamped_to_ceiling(self):
        # The 16384 default exceeds pegasus1.2's 4096 ceiling — must clamp, not 400.
        self.assertEqual(tl.clamp_max_tokens("pegasus1.2", 16384), 4096)

    def test_unknown_model_falls_back(self):
        self.assertEqual(tl.clamp_max_tokens("pegasusX", 8000), 8000)


class SafeFilename(unittest.TestCase):
    def test_strips_quotes_and_newlines(self):
        out = tl._safe_filename('evil";name="x.mp4')
        self.assertNotIn('"', out)
        out2 = tl._safe_filename("a\r\nb.mp4")
        self.assertNotIn("\n", out2)
        self.assertNotIn("\r", out2)

    def test_empty_falls_back(self):
        self.assertEqual(tl._safe_filename(""), "video")

    def test_normal_name_preserved(self):
        self.assertEqual(tl._safe_filename("lecture.mp4"), "lecture.mp4")


if __name__ == "__main__":
    unittest.main(verbosity=2)
