"""Caption parsing: rolling auto-sub dedupe, and cue-boundary detection."""
from __future__ import annotations

from pathlib import Path

from conftest import build_rolling_vtt

import transcribe

SENTENCES = [
    "welcome back to the channel today",
    "we are going to talk about frame budgets",
    "because token cost scales with frame count",
    "and that is the constraint that actually matters",
    "so the script targets a budget by duration",
    "instead of a fixed sampling rate",
    "which keeps short clips dense",
    "and stops long ones from blowing the budget",
    "that is the whole idea behind auto fps",
    "thanks for watching and see you next time",
]


def _vtt(tmp_path: Path) -> str:
    path = tmp_path / "rolling.vtt"
    build_rolling_vtt(path, SENTENCES)
    return str(path)


def test_rolling_cues_are_deduped(tmp_path: Path) -> None:
    """Each sentence is emitted in two cues; it must survive exactly once."""
    text = transcribe.format_transcript(transcribe.parse_vtt(_vtt(tmp_path)))
    for sentence in SENTENCES:
        assert text.count(sentence) == 1, (
            f"{sentence!r} appears {text.count(sentence)}x -- rolling dedupe did not collapse it"
        )


def test_dedupe_preserves_segment_granularity(tmp_path: Path) -> None:
    """Dedupe must strip repeated words, not merge cues into one blob.

    Splicing overlapping cues into their predecessor also deduplicates, but
    collapses the whole track into one or two segments -- which leaves a
    27-minute transcript carrying two [MM:SS] stamps and nothing to align
    frames against.
    """
    segments = transcribe.parse_vtt(_vtt(tmp_path))
    assert len(segments) >= len(SENTENCES) - 2, (
        f"expected roughly one segment per cue, got {len(segments)} -- cues were merged, not deduped"
    )


def test_filter_range_returns_a_slice_not_everything(tmp_path: Path) -> None:
    """A narrow window must exclude content from outside it.

    Asserted semantically rather than as a size ratio: filter_range keeps any
    segment merely OVERLAPPING the window, so the character count of a short
    window on a short track is legitimately a large fraction of the whole and
    proves nothing either way. What must never happen is the tail of the video
    surfacing in a window near the start -- which is exactly what happens once
    segments are coarse enough to span the whole track.
    """
    segments = transcribe.parse_vtt(_vtt(tmp_path))
    windowed = transcribe.format_transcript(transcribe.filter_range(segments, 2.0, 6.0))
    assert SENTENCES[1] in windowed, "in-window content missing from the filtered transcript"
    assert SENTENCES[-1] not in windowed, (
        "content from the end of the track leaked into an early window -- "
        "segments are too coarse for filter_range to narrow:\n" + windowed
    )


def test_non_rolling_captions_pass_through_unchanged(tmp_path: Path) -> None:
    """Hand-authored one-cue-per-utterance subtitles must not be mangled."""
    path = tmp_path / "plain.vtt"
    path.write_text(
        "WEBVTT\n\n"
        "00:00:00.000 --> 00:00:02.000\nFirst independent line.\n\n"
        "00:00:02.000 --> 00:00:04.000\nSecond unrelated line.\n\n"
        "00:00:04.000 --> 00:00:06.000\nThird distinct line.\n",
        encoding="utf-8",
    )
    segments = transcribe.parse_vtt(str(path))
    assert [s["text"] for s in segments] == [
        "First independent line.",
        "Second unrelated line.",
        "Third distinct line.",
    ]


def test_cue_starting_with_a_lone_space_is_not_dropped(tmp_path: Path) -> None:
    """A whitespace-only line is not a VTT cue terminator."""
    path = tmp_path / "space.vtt"
    path.write_text(
        "WEBVTT\n\n"
        "00:00:00.000 --> 00:00:02.000\n \nthe opening words of the video\n\n"
        "00:00:02.000 --> 00:00:04.000\na second unrelated cue\n",
        encoding="utf-8",
    )
    text = transcribe.format_transcript(transcribe.parse_vtt(str(path)))
    assert "the opening words of the video" in text, (
        "cue whose first line is a lone space was silently dropped:\n" + text
    )
