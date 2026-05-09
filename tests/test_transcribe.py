"""Tests for scripts/transcribe.py - pure-Python VTT parsing logic.

These tests require no external binaries (ffmpeg, yt-dlp) and run in any
standard Python 3.10+ environment with just pytest installed.
"""
import sys
import textwrap
from pathlib import Path

import pytest

# Make scripts/ importable without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from transcribe import (  # noqa: E402
    _dedupe,
    filter_range,
    format_transcript,
    parse_vtt,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _vtt(body):
    """Wrap body text in a minimal WebVTT header."""
    return "WEBVTT\n\n" + textwrap.dedent(body).lstrip()


# ---------------------------------------------------------------------------
# parse_vtt
# ---------------------------------------------------------------------------

class TestParseVtt:
    def test_single_cue(self, tmp_path):
        vtt = _vtt("""
            00:00:01.000 --> 00:00:03.000
            Hello world
        """)
        p = tmp_path / "sub.vtt"
        p.write_text(vtt, encoding="utf-8")
        segs = parse_vtt(str(p))
        assert len(segs) == 1
        assert segs[0]["text"] == "Hello world"
        assert segs[0]["start"] == pytest.approx(1.0)
        assert segs[0]["end"] == pytest.approx(3.0)

    def test_strips_html_tags(self, tmp_path):
        vtt = _vtt("""
            00:00:00.000 --> 00:00:02.000
            <c.colorE5E5E5>some text</c>
        """)
        p = tmp_path / "sub.vtt"
        p.write_text(vtt, encoding="utf-8")
        segs = parse_vtt(str(p))
        assert segs[0]["text"] == "some text"

    def test_multiple_cues(self, tmp_path):
        vtt = _vtt("""
            00:00:01.000 --> 00:00:02.000
            First

            00:00:03.000 --> 00:00:04.000
            Second
        """)
        p = tmp_path / "sub.vtt"
        p.write_text(vtt, encoding="utf-8")
        segs = parse_vtt(str(p))
        assert [s["text"] for s in segs] == ["First", "Second"]

    def test_deduplication_identical(self, tmp_path):
        """YouTube auto-subs repeat the same cue several times."""
        vtt = _vtt("""
            00:00:01.000 --> 00:00:03.000
            Repeated line

            00:00:02.000 --> 00:00:04.000
            Repeated line
        """)
        p = tmp_path / "sub.vtt"
        p.write_text(vtt, encoding="utf-8")
        segs = parse_vtt(str(p))
        assert len(segs) == 1
        assert segs[0]["text"] == "Repeated line"
        assert segs[0]["end"] == pytest.approx(4.0)

    def test_deduplication_rolling(self, tmp_path):
        """Rolling-style subs where new words are appended."""
        vtt = _vtt("""
            00:00:01.000 --> 00:00:02.000
            Hello

            00:00:01.500 --> 00:00:03.000
            Hello world
        """)
        p = tmp_path / "sub.vtt"
        p.write_text(vtt, encoding="utf-8")
        segs = parse_vtt(str(p))
        assert len(segs) == 1
        assert segs[0]["text"] == "Hello world"

    def test_comma_decimal_separator(self, tmp_path):
        """VTT files sometimes use commas instead of dots in timestamps."""
        vtt = "WEBVTT\n\n00:00:05,000 --> 00:00:07,500\nComma sep\n"
        p = tmp_path / "sub.vtt"
        p.write_text(vtt, encoding="utf-8")
        segs = parse_vtt(str(p))
        assert segs[0]["start"] == pytest.approx(5.0)
        assert segs[0]["end"] == pytest.approx(7.5)

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.vtt"
        p.write_text("WEBVTT\n\n", encoding="utf-8")
        assert parse_vtt(str(p)) == []


# ---------------------------------------------------------------------------
# _dedupe
# ---------------------------------------------------------------------------

class TestDedupe:
    def test_no_duplicates(self):
        segs = [
            {"start": 0.0, "end": 1.0, "text": "A"},
            {"start": 1.0, "end": 2.0, "text": "B"},
        ]
        result = _dedupe(segs)
        assert len(result) == 2

    def test_exact_duplicate_merges_end(self):
        segs = [
            {"start": 0.0, "end": 1.0, "text": "Same"},
            {"start": 0.5, "end": 2.0, "text": "Same"},
        ]
        result = _dedupe(segs)
        assert len(result) == 1
        assert result[0]["end"] == pytest.approx(2.0)

    def test_rolling_duplicate_extends_text(self):
        segs = [
            {"start": 0.0, "end": 1.0, "text": "Hello"},
            {"start": 0.5, "end": 2.0, "text": "Hello world"},
        ]
        result = _dedupe(segs)
        assert len(result) == 1
        assert result[0]["text"] == "Hello world"
        assert result[0]["end"] == pytest.approx(2.0)

    def test_empty(self):
        assert _dedupe([]) == []


# ---------------------------------------------------------------------------
# filter_range
# ---------------------------------------------------------------------------

class TestFilterRange:
    SEGS = [
        {"start": 0.0, "end": 2.0, "text": "A"},
        {"start": 5.0, "end": 7.0, "text": "B"},
        {"start": 10.0, "end": 12.0, "text": "C"},
    ]

    def test_no_filter(self):
        assert filter_range(self.SEGS, None, None) == self.SEGS

    def test_start_only(self):
        result = filter_range(self.SEGS, 4.0, None)
        assert [s["text"] for s in result] == ["B", "C"]

    def test_end_only(self):
        result = filter_range(self.SEGS, None, 6.0)
        assert [s["text"] for s in result] == ["A", "B"]

    def test_start_and_end(self):
        result = filter_range(self.SEGS, 4.0, 8.0)
        assert [s["text"] for s in result] == ["B"]

    def test_boundary_overlap(self):
        result = filter_range(self.SEGS, 2.0, 5.0)
        texts = [s["text"] for s in result]
        assert "A" in texts
        assert "B" in texts

    def test_no_match(self):
        assert filter_range(self.SEGS, 20.0, 30.0) == []


# ---------------------------------------------------------------------------
# format_transcript
# ---------------------------------------------------------------------------

class TestFormatTranscript:
    def test_basic(self):
        segs = [{"start": 0.0, "end": 2.0, "text": "Hello"}]
        assert format_transcript(segs) == "[00:00] Hello"

    def test_minute_boundary(self):
        segs = [{"start": 65.0, "end": 67.0, "text": "One minute in"}]
        assert format_transcript(segs) == "[01:05] One minute in"

    def test_multiple_segments(self):
        segs = [
            {"start": 0.0, "end": 2.0, "text": "First"},
            {"start": 10.0, "end": 12.0, "text": "Second"},
        ]
        lines = format_transcript(segs).splitlines()
        assert lines[0] == "[00:00] First"
        assert lines[1] == "[00:10] Second"

    def test_empty(self):
        assert format_transcript([]) == ""
