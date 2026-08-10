"""VTT dedup: rolling auto-caption windows, exact duplicates, segment grouping."""
from __future__ import annotations

from pathlib import Path

import transcribe


def _parse(tmp_path: Path, vtt_text: str) -> list[dict]:
    path = tmp_path / "captions.vtt"
    path.write_text(vtt_text, encoding="utf-8")
    return transcribe.parse_vtt(str(path))


def _all_text(segments: list[dict]) -> str:
    return " ".join(seg["text"] for seg in segments)


# --- real rolling-window sample ----------------------------------------------
# Actual excerpt captured from a YouTube German auto-caption VTT. Reproduces
# the exact rolling-window 3-cue pattern: growing content cue -> ~10ms settle
# transition -> next growing content cue whose first line repeats the settled
# text.
ROLLING_SAMPLE_VTT = """WEBVTT
Kind: captions
Language: de

00:00:07.069 --> 00:00:09.470 align:start position:0%
[Musik]
ich<00:00:08.069><c> begrüße</c><00:00:08.429><c> ich</c><00:00:08.460><c> herzlich</c><00:00:09.000><c> zu</c><00:00:09.090><c> einer</c><00:00:09.210><c> neuen</c>

00:00:09.470 --> 00:00:09.480 align:start position:0%
ich begrüße ich herzlich zu einer neuen


00:00:09.480 --> 00:00:11.810 align:start position:0%
ich begrüße ich herzlich zu einer neuen
folge<00:00:09.840><c> von</c><00:00:10.050><c> alpha</c><00:00:10.710><c> baby</c><00:00:11.099><c> mein</c><00:00:11.400><c> name</c><00:00:11.670><c> ist</c>

00:00:11.810 --> 00:00:11.820 align:start position:0%
folge von alpha baby mein name ist


00:00:11.820 --> 00:00:13.910 align:start position:0%
folge von alpha baby mein name ist
antrag<00:00:12.210><c> fabry</c><00:00:12.570><c> und</c><00:00:12.809><c> als</c><00:00:13.349><c> astrologin</c>
"""


def test_rolling_window_line_appears_once(tmp_path):
    segments = _parse(tmp_path, ROLLING_SAMPLE_VTT)
    combined = _all_text(segments)
    for phrase in [
        "ich begrüße ich herzlich zu einer neuen",
        "folge von alpha baby mein name ist",
    ]:
        assert combined.count(phrase) == 1, f"expected {phrase!r} once, found in: {combined!r}"


def test_rolling_window_reconstructs_full_sentence(tmp_path):
    segments = _parse(tmp_path, ROLLING_SAMPLE_VTT)
    assert _all_text(segments) == (
        "[Musik] ich begrüße ich herzlich zu einer neuen "
        "folge von alpha baby mein name ist antrag fabry und als astrologin"
    )


def test_rolling_window_timestamps_stay_monotonic(tmp_path):
    segments = _parse(tmp_path, ROLLING_SAMPLE_VTT)
    assert segments
    prev_end = -1.0
    for seg in segments:
        assert seg["start"] >= prev_end
        assert seg["end"] >= seg["start"]
        prev_end = seg["end"]
    assert segments[0]["start"] >= 7.0
    assert segments[-1]["end"] <= 13.91


# --- exact-duplicate case (already handled pre-fix, must keep working) ------

EXACT_DUPLICATE_VTT = """WEBVTT

00:00:01.000 --> 00:00:02.000
Hallo zusammen

00:00:02.000 --> 00:00:03.000
Hallo zusammen

00:00:03.000 --> 00:00:05.000
und willkommen
"""


def test_identical_consecutive_lines_collapse_once(tmp_path):
    combined = _all_text(_parse(tmp_path, EXACT_DUPLICATE_VTT))
    assert combined.count("Hallo zusammen") == 1
    assert "und willkommen" in combined


# --- safety case: a genuine spoken repetition must survive -------------------
# Only a fully identical line is ever collapsed - a partial/substring overlap
# (e.g. an idiom repeated across two distinct, non-identical lines) is not
# enough, so it's never mistaken for a rolling-caption artifact.

GENUINE_REPETITION_VTT = """WEBVTT

00:00:01.000 --> 00:00:03.000
das war das erste Mal dass ich sowas erlebt habe

00:00:03.000 --> 00:00:05.000
das erste Mal war ich total überrascht
"""


def test_repeated_idiom_across_distinct_lines_not_dropped(tmp_path):
    combined = _all_text(_parse(tmp_path, GENUINE_REPETITION_VTT))
    assert combined.count("das erste Mal") == 2
    assert "das war das erste Mal dass ich sowas erlebt habe" in combined
    assert "das erste Mal war ich total überrascht" in combined


# --- segment grouping ---------------------------------------------------------

def test_long_silence_starts_a_new_segment(tmp_path):
    vtt = """WEBVTT

00:00:01.000 --> 00:00:02.000
erste Zeile

00:00:10.000 --> 00:00:11.000
zweite Zeile nach einer Pause
"""
    assert len(_parse(tmp_path, vtt)) == 2


def test_long_continuous_speech_is_chunked(tmp_path):
    lines = []
    t = 0.0
    for i in range(40):
        start, end = t, t + 0.5
        lines.append(f"00:00:{start:06.3f} --> 00:00:{end:06.3f}\nwort{i}\n")
        t = end
    vtt = "WEBVTT\n\n" + "\n".join(lines)
    segments = _parse(tmp_path, vtt)
    assert len(segments) > 1, "40 words with no repeats should split into multiple segments"
    tokens = _all_text(segments).split()
    for i in range(40):
        assert tokens.count(f"wort{i}") == 1
