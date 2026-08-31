"""VTT parsing: rolling auto-sub dedup, cue-body padding, and range filtering."""
from __future__ import annotations

from pathlib import Path

import transcribe


# A faithful slice of a YouTube auto-sub track. Note the two traps:
#   - a whitespace-only padding line as the first line of a cue body
#   - each cue repeating the previous line before adding new words
ROLLING_VTT = """WEBVTT
Kind: captions
Language: en

00:00:15.030 --> 00:00:15.040 align:start position:0%
 
 

00:00:15.040 --> 00:00:18.230 align:start position:0%
 
&gt;&gt; Hi<00:00:15.160><c> everyone.</c><00:00:16.280><c> My</c><00:00:16.440><c> goal</c>

00:00:18.230 --> 00:00:18.240 align:start position:0%
&gt;&gt; Hi everyone. My goal
 

00:00:18.240 --> 00:00:20.030 align:start position:0%
&gt;&gt; Hi everyone. My goal
is<00:00:18.360><c> that</c><00:00:18.520><c> you'll</c><00:00:19.920><c> be</c>

00:00:20.030 --> 00:00:20.040 align:start position:0%
is that you'll be
 

00:00:20.040 --> 00:00:22.150 align:start position:0%
is that you'll be
able<00:00:20.280><c> to</c><00:00:20.400><c> assemble</c><00:00:21.920><c> agent</c><00:00:22.040><c> teams</c>
"""

# No inline word tags: a hand-authored track must pass through structurally intact.
PLAIN_VTT = """WEBVTT

00:00:01.000 --> 00:00:03.000
The quick brown fox

00:00:03.000 --> 00:00:05.000
jumps over the lazy dog
"""


def _write(tmp_path: Path, name: str, body: str) -> str:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return str(path)


# --- rolling auto-subs -------------------------------------------------------

def test_rolling_cues_are_not_duplicated(tmp_path: Path):
    text = transcribe.format_transcript(
        transcribe.parse_vtt(_write(tmp_path, "roll.vtt", ROLLING_VTT))
    )
    words = text.split()
    assert words.count("goal") == 1
    assert words.count("assemble") == 1
    assert text.count("is that you'll be") == 1


def test_padding_line_does_not_drop_cue_content(tmp_path: Path):
    """A cue whose body starts with a whitespace-only line must still be read."""
    text = transcribe.format_transcript(
        transcribe.parse_vtt(_write(tmp_path, "roll.vtt", ROLLING_VTT))
    )
    # This line lives only in a padded cue body; naive parsing loses it.
    assert "Hi everyone. My goal" in text


def test_html_entities_are_unescaped(tmp_path: Path):
    text = transcribe.format_transcript(
        transcribe.parse_vtt(_write(tmp_path, "roll.vtt", ROLLING_VTT))
    )
    assert "&gt;" not in text
    assert ">>" in text


def test_full_spoken_text_survives_in_order(tmp_path: Path):
    segments = transcribe.parse_vtt(_write(tmp_path, "roll.vtt", ROLLING_VTT))
    spoken = " ".join(seg["text"] for seg in segments)
    assert spoken == ">> Hi everyone. My goal is that you'll be able to assemble agent teams"


def test_timestamps_stay_aligned_to_new_words(tmp_path: Path):
    segments = transcribe.parse_vtt(_write(tmp_path, "roll.vtt", ROLLING_VTT))
    starts = {seg["text"]: seg["start"] for seg in segments}
    assert starts["is that you'll be"] == 18.24
    assert starts["able to assemble agent teams"] == 20.04


# --- plain (non-rolling) subtitles -------------------------------------------

def test_plain_vtt_is_left_intact(tmp_path: Path):
    segments = transcribe.parse_vtt(_write(tmp_path, "plain.vtt", PLAIN_VTT))
    assert [seg["text"] for seg in segments] == [
        "The quick brown fox",
        "jumps over the lazy dog",
    ]


# --- _overlap / _dedupe ------------------------------------------------------

def test_overlap_finds_the_longest_shared_run():
    assert transcribe._overlap("a b c d".split(), "c d e f".split()) == 2
    assert transcribe._overlap("a b c".split(), "a b c d".split()) == 3
    assert transcribe._overlap("a b c".split(), "x y z".split()) == 0


def test_overlap_ignores_case_and_punctuation():
    assert transcribe._overlap("into Goose.".split(), "goose is now".split()) == 1


def test_short_incidental_overlap_is_kept():
    """One shared word is likely real speech, not a rolling repeat: keep it."""
    segments = [
        {"start": 0.0, "end": 1.0, "text": "and that's cool"},
        {"start": 1.0, "end": 2.0, "text": "cool thing about these guys"},
    ]
    out = transcribe._dedupe(segments)
    assert out[1]["text"] == "cool thing about these guys"


def test_exact_duplicate_extends_the_previous_segment():
    segments = [
        {"start": 0.0, "end": 1.0, "text": "same words here"},
        {"start": 1.0, "end": 2.5, "text": "same words here"},
    ]
    out = transcribe._dedupe(segments)
    assert len(out) == 1
    assert out[0]["end"] == 2.5


def test_fully_contained_cue_is_dropped_without_losing_time():
    segments = [
        {"start": 0.0, "end": 1.0, "text": "one two three four"},
        {"start": 1.0, "end": 3.0, "text": "three four"},
    ]
    out = transcribe._dedupe(segments)
    assert len(out) == 1
    assert out[0]["end"] == 3.0


# --- filter_range ------------------------------------------------------------

def test_filter_range_keeps_overlapping_segments():
    segments = [
        {"start": 0.0, "end": 5.0, "text": "a"},
        {"start": 5.0, "end": 10.0, "text": "b"},
        {"start": 10.0, "end": 15.0, "text": "c"},
    ]
    kept = [seg["text"] for seg in transcribe.filter_range(segments, 6.0, 9.0)]
    assert kept == ["b"]
