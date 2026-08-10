#!/usr/bin/env python3
"""Parse a WebVTT subtitle file into a clean, timestamped transcript.

YouTube auto-subs emit rolling-duplicate cues (each line appears 2-3 times as it
scrolls). We dedupe consecutive identical cues and merge their time ranges.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


TS_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[.,](\d{3})\s+-->\s+(\d{2}):(\d{2}):(\d{2})[.,](\d{3})"
)
TAG_RE = re.compile(r"<[^>]+>")


def _to_seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_vtt(path: str) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    # One entry per physical cue LINE (not per cue) so _dedupe can compare
    # lines individually — YouTube's rolling captions repeat whole lines
    # verbatim across adjacent cues, never partial lines.
    cue_lines: list[dict] = []
    i = 0
    while i < len(lines):
        match = TS_RE.match(lines[i])
        if not match:
            i += 1
            continue

        start = round(_to_seconds(*match.groups()[:4]), 2)
        end = round(_to_seconds(*match.groups()[4:]), 2)
        i += 1

        while i < len(lines) and lines[i].strip():
            cleaned = TAG_RE.sub("", lines[i]).strip()
            if cleaned:
                cue_lines.append({"text": cleaned, "start": start, "end": end})
            i += 1
        i += 1

    return _dedupe(cue_lines)


MAX_SEGMENT_WORDS = 30
MAX_SEGMENT_GAP = 1.5  # seconds of silence that force a new segment


def _dedupe(cue_lines: list[dict]) -> list[dict]:
    """Collapse rolling duplicates common in YouTube auto-subs and group lines
    into readable segments.

    YouTube's rolling auto-captions re-render each spoken line 2-3 times as
    the on-screen window advances: once while it's the growing second line,
    once settled as a solo transition cue, once again as the next cue's
    first line. Each re-render is a verbatim repeat of the whole line, so we
    drop a line only when it exactly matches the immediately preceding kept
    line — never on a partial/substring match, which would risk deleting a
    genuine spoken repetition (e.g. an idiom repeated across two distinct
    lines) instead of a rendering artifact.
    """
    out: list[dict] = []
    open_seg: dict | None = None

    for line in cue_lines:
        if open_seg is not None and line["text"] == open_seg["last_line"]:
            open_seg["end"] = line["end"]
            continue

        start_new = (
            open_seg is None
            or line["start"] - open_seg["end"] > MAX_SEGMENT_GAP
            or open_seg["word_count"] >= MAX_SEGMENT_WORDS
        )
        if start_new:
            if open_seg is not None:
                out.append({"start": open_seg["start"], "end": open_seg["end"], "text": open_seg["text"]})
            open_seg = {
                "start": line["start"],
                "end": line["end"],
                "text": line["text"],
                "last_line": line["text"],
                "word_count": len(line["text"].split()),
            }
        else:
            open_seg["text"] += " " + line["text"]
            open_seg["last_line"] = line["text"]
            open_seg["word_count"] += len(line["text"].split())
            open_seg["end"] = line["end"]

    if open_seg is not None:
        out.append({"start": open_seg["start"], "end": open_seg["end"], "text": open_seg["text"]})
    return out


def filter_range(
    segments: list[dict],
    start_seconds: float | None,
    end_seconds: float | None,
) -> list[dict]:
    """Return segments whose time range overlaps [start, end]."""
    if start_seconds is None and end_seconds is None:
        return segments
    lo = start_seconds if start_seconds is not None else float("-inf")
    hi = end_seconds if end_seconds is not None else float("inf")
    return [seg for seg in segments if seg["end"] >= lo and seg["start"] <= hi]


def format_transcript(segments: list[dict]) -> str:
    lines = []
    for seg in segments:
        start = int(seg["start"])
        stamp = f"[{start // 60:02d}:{start % 60:02d}]"
        lines.append(f"{stamp} {seg['text']}")
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: transcribe.py <vtt-path>", file=sys.stderr)
        raise SystemExit(2)
    print(format_transcript(parse_vtt(sys.argv[1])))
