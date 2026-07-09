"""WebVTT timestamp parsing."""
from __future__ import annotations

from pathlib import Path

import transcribe


def _write_vtt(path: Path, body: str) -> Path:
    path.write_text("WEBVTT\n\n" + body, encoding="utf-8")
    return path


def test_parse_vtt_accepts_minute_second_timestamps(tmp_path: Path):
    vtt = _write_vtt(
        tmp_path / "captions.vtt",
        "00:01.000 --> 00:02.500\nhello\n\n",
    )

    assert transcribe.parse_vtt(str(vtt)) == [
        {"start": 1.0, "end": 2.5, "text": "hello"},
    ]


def test_parse_vtt_accepts_hour_minute_second_timestamps(tmp_path: Path):
    vtt = _write_vtt(
        tmp_path / "captions.vtt",
        "01:02:03.000 --> 01:02:04.250\n<v Speaker>hello</v>\n\n",
    )

    assert transcribe.parse_vtt(str(vtt)) == [
        {"start": 3723.0, "end": 3724.25, "text": "hello"},
    ]


def test_parse_vtt_unescapes_html_entities(tmp_path: Path):
    vtt = _write_vtt(
        tmp_path / "captions.vtt",
        "00:01.000 --> 00:02.000\nTom &amp; Jerry says &quot;hi&quot;\n\n",
    )

    assert transcribe.parse_vtt(str(vtt)) == [
        {"start": 1.0, "end": 2.0, "text": 'Tom & Jerry says "hi"'},
    ]


def test_format_transcript_uses_hour_stamp_after_one_hour():
    assert transcribe.format_transcript([
        {"start": 3661.0, "end": 3662.0, "text": "after the hour"},
    ]) == "[1:01:01] after the hour"


def test_format_transcript_keeps_minute_second_stamp_under_one_hour():
    assert transcribe.format_transcript([
        {"start": 61.0, "end": 62.0, "text": "after a minute"},
    ]) == "[01:01] after a minute"
