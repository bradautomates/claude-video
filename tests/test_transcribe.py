"""Caption parsing: WebVTT cues, rolling-duplicate collapse, range filter."""
from __future__ import annotations

from pathlib import Path

import transcribe


def _vtt(tmp_path: Path, body: str) -> str:
    path = tmp_path / "subs.vtt"
    path.write_text("WEBVTT\n\n" + body, encoding="utf-8")
    return str(path)


def _texts(segments: list[dict]) -> list[str]:
    return [seg["text"] for seg in segments]


# --- strip_rolling_overlap: drop the repeated tail of the previous cue --------

def test_strip_overlap_removes_repeated_tail():
    previous = "What would you say by the end of today's episode that the audience is going to"
    current = "episode that the audience is going to take away and what will they have"
    assert transcribe.strip_rolling_overlap(previous, current) == "take away and what will they have"


def test_strip_overlap_ignores_case_and_punctuation():
    """Cues re-emit the same words with different casing and commas."""
    assert transcribe.strip_rolling_overlap(
        "so we vectorize it, chunk by chunk",
        "Chunk by chunk. Then we store it",
    ) == "Then we store it"


def test_strip_overlap_keeps_short_genuine_repetition():
    """Below MIN_OVERLAP_WORDS the repeat is probably real speech, not scrolling."""
    assert transcribe.strip_rolling_overlap("and I said yeah", "yeah that is right") == "yeah that is right"


def test_strip_overlap_leaves_unrelated_text_alone():
    assert transcribe.strip_rolling_overlap("a totally different sentence", "no overlap at all here") == (
        "no overlap at all here"
    )


def test_strip_overlap_of_a_full_repeat_is_empty():
    assert transcribe.strip_rolling_overlap("the same five words exactly", "the same five words exactly") == ""


# --- parse_vtt + _dedupe: the three rolling shapes ---------------------------

def test_parse_vtt_collapses_scrolling_cues(tmp_path: Path):
    """Each cue repeats the tail of the one before it, the en-orig shape."""
    path = _vtt(tmp_path, """00:00:01.000 --> 00:00:03.000
What would you say by the end of today's

00:00:03.000 --> 00:00:05.000
What would you say by the end of today's episode that the audience

00:00:05.000 --> 00:00:07.000
episode that the audience is going to take away
""")
    assert _texts(transcribe.parse_vtt(path)) == [
        "What would you say by the end of today's episode that the audience",
        "is going to take away",
    ]


def test_parse_vtt_collapses_exact_repeats(tmp_path: Path):
    path = _vtt(tmp_path, """00:00:00.000 --> 00:00:02.000
hello there

00:00:02.000 --> 00:00:04.000
hello there
""")
    segments = transcribe.parse_vtt(path)
    assert _texts(segments) == ["hello there"]
    assert segments[0]["end"] == 4.0  # time range merged


def test_parse_vtt_merges_time_range_of_a_fully_repeated_cue(tmp_path: Path):
    """A cue that is nothing but overlap extends the previous cue instead of vanishing."""
    path = _vtt(tmp_path, """00:00:00.000 --> 00:00:02.000
the same five words exactly

00:00:02.000 --> 00:00:06.000
The same five words exactly!
""")
    segments = transcribe.parse_vtt(path)
    assert _texts(segments) == ["the same five words exactly"]
    assert segments[0]["end"] == 6.0


def test_parse_vtt_leaves_manual_captions_untouched(tmp_path: Path):
    """Non-scrolling (human-authored) cues must survive verbatim."""
    path = _vtt(tmp_path, """00:00:00.000 --> 00:00:02.000
Welcome back to the channel.

00:00:02.000 --> 00:00:04.000
Today we are building a sandbox.

00:00:04.000 --> 00:00:06.000
Then run it. Then run it again if it fails.
""")
    assert _texts(transcribe.parse_vtt(path)) == [
        "Welcome back to the channel.",
        "Today we are building a sandbox.",
        "Then run it. Then run it again if it fails.",
    ]


def test_parse_vtt_is_idempotent(tmp_path: Path):
    """Deduping already-deduped segments changes nothing."""
    path = _vtt(tmp_path, """00:00:01.000 --> 00:00:03.000
the way people make money with Claude is

00:00:03.000 --> 00:00:05.000
make money with Claude is about to change for the last

00:00:05.000 --> 00:00:07.000
about to change for the last couple years the move was simple
""")
    once = transcribe.parse_vtt(path)
    twice = transcribe._dedupe([dict(seg) for seg in once])
    assert _texts(twice) == _texts(once)


def test_parse_vtt_drops_no_spoken_words(tmp_path: Path):
    """Output is an in-order subsequence of the raw cue stream: nothing invented."""
    path = _vtt(tmp_path, """00:00:01.000 --> 00:00:03.000
agentic AI is on its way to replace

00:00:03.000 --> 00:00:05.000
on its way to replace around half of all jobs and if you

00:00:05.000 --> 00:00:07.000
around half of all jobs and if you don't know how to use it
""")
    merged = " ".join(_texts(transcribe.parse_vtt(path))).split()
    assert merged == (
        "agentic AI is on its way to replace around half of all jobs and if you "
        "don't know how to use it"
    ).split()


def test_parse_vtt_strips_inline_tags(tmp_path: Path):
    path = _vtt(tmp_path, """00:00:00.000 --> 00:00:02.000
<c.colorE5E5E5>hello</c> <00:00:01.000>there
""")
    assert _texts(transcribe.parse_vtt(path)) == ["hello there"]


# --- filter_range -------------------------------------------------------------

def test_filter_range_keeps_overlapping_segments():
    segments = [
        {"start": 0.0, "end": 2.0, "text": "a"},
        {"start": 2.0, "end": 4.0, "text": "b"},
        {"start": 4.0, "end": 6.0, "text": "c"},
    ]
    assert _texts(transcribe.filter_range(segments, 3.0, 5.0)) == ["b", "c"]
    assert _texts(transcribe.filter_range(segments, None, None)) == ["a", "b", "c"]


def test_format_transcript_stamps_minutes_and_seconds():
    segments = [{"start": 75.4, "end": 78.0, "text": "past a minute"}]
    assert transcribe.format_transcript(segments) == "[01:15] past a minute"
