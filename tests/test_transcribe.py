"""VTT parsing and roll-up caption dedup: exact, prefix and partial overlap."""
from __future__ import annotations

from pathlib import Path

import transcribe


def _segments(*texts: str) -> list[dict]:
    return [
        {"start": float(i), "end": float(i + 1), "text": text}
        for i, text in enumerate(texts)
    ]


# --- _strip_leading_overlap: suffix of previous == prefix of current ---------

def test_strip_overlap_trims_repeated_tail():
    previous = "modeling this glass bottle. I'm Kevin Kennedy and this is day two of"
    current = "I'm Kevin Kennedy and this is day two of learn Fusion in 30 days."
    assert transcribe._strip_leading_overlap(previous, current) == "learn Fusion in 30 days."


def test_strip_overlap_returns_empty_when_nothing_new():
    assert transcribe._strip_leading_overlap("hello world", "hello world") == ""


def test_strip_overlap_keeps_text_without_overlap():
    previous = "the shell traces the entire"
    current = "outer contour. If the contour"
    assert transcribe._strip_leading_overlap(previous, current) == current


def test_strip_overlap_ignores_single_word_repetition():
    # Below MIN_OVERLAP_WORDS, so genuine repetition is preserved.
    assert transcribe._strip_leading_overlap("no", "no, no") == "no, no"


def test_strip_overlap_prefers_longest_overlap():
    previous = "one two three two three"
    current = "two three four"
    assert transcribe._strip_leading_overlap(previous, current) == "four"


# --- _dedupe: the three shapes of duplication --------------------------------

def test_dedupe_collapses_identical_cues():
    out = transcribe._dedupe(_segments("same line", "same line"))
    assert [seg["text"] for seg in out] == ["same line"]
    assert out[0]["end"] == 2.0  # time range merged


def test_dedupe_absorbs_prefix_extension():
    out = transcribe._dedupe(_segments("hello", "hello there friend"))
    assert [seg["text"] for seg in out] == ["hello there friend"]


def test_dedupe_trims_partial_overlap():
    out = transcribe._dedupe(_segments(
        "activate the line tool. Click the origin",
        "Click the origin point to start the first line.",
    ))
    assert [seg["text"] for seg in out] == [
        "activate the line tool. Click the origin",
        "point to start the first line.",
    ]


def test_dedupe_keeps_distinct_cues():
    texts = ("first cue here", "completely different text")
    assert [seg["text"] for seg in transcribe._dedupe(_segments(*texts))] == list(texts)


def test_dedupe_preserves_start_of_kept_segment():
    out = transcribe._dedupe(_segments("alpha beta gamma", "beta gamma delta"))
    assert out[1]["start"] == 1.0
    assert out[1]["text"] == "delta"


# --- parse_vtt: end to end on a real roll-up caption file ---------------------

ROLLING_VTT = """WEBVTT

00:00:01.000 --> 00:00:04.000
Your second challenge starts right now,

00:00:04.000 --> 00:00:07.000
Your second challenge starts right now, modeling this glass bottle.

00:00:07.000 --> 00:00:09.000
modeling this glass bottle. Today we're
reinforcing concepts from day one

00:00:09.000 --> 00:00:11.000
reinforcing concepts from day one while you learn to use reference images.
"""


def test_parse_vtt_collapses_rolling_captions(tmp_path: Path):
    path = tmp_path / "rolling.vtt"
    path.write_text(ROLLING_VTT, encoding="utf-8")

    texts = [seg["text"] for seg in transcribe.parse_vtt(str(path))]

    assert texts == [
        "Your second challenge starts right now, modeling this glass bottle.",
        "Today we're reinforcing concepts from day one",
        "while you learn to use reference images.",
    ]


def test_parse_vtt_strips_inline_tags(tmp_path: Path):
    path = tmp_path / "tagged.vtt"
    path.write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\n"
        "<c.colorE5E5E5>styled</c> text\n",
        encoding="utf-8",
    )
    assert [seg["text"] for seg in transcribe.parse_vtt(str(path))] == ["styled text"]
