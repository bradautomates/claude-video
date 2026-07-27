#!/usr/bin/env python3
"""Parse a WebVTT subtitle file into a clean, timestamped transcript.

YouTube auto-subs emit rolling-duplicate cues (each line appears 2-3 times as it
scrolls). We dedupe consecutive identical cues and merge their time ranges.

Three rolling shapes are collapsed, see _dedupe: an exact repeat, a cue that
grows by appending to the previous one, and a scrolling cue whose leading words
repeat the tail of the previous cue. That last shape is the common one on
`en-orig` tracks and roughly doubles transcript size if left in.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


TS_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[.,](\d{3})\s+-->\s+(\d{2}):(\d{2}):(\d{2})[.,](\d{3})"
)
TAG_RE = re.compile(r"<[^>]+>")
PUNCT_RE = re.compile(r"[^\w']+")

# A scrolling cue repeats the tail of the one before it. Requiring at least this
# many matching words keeps a genuine short repetition ("yeah, yeah") intact.
MIN_OVERLAP_WORDS = 3


def _to_seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_vtt(path: str) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    segments: list[dict] = []
    i = 0
    while i < len(lines):
        match = TS_RE.match(lines[i])
        if not match:
            i += 1
            continue

        start = _to_seconds(*match.groups()[:4])
        end = _to_seconds(*match.groups()[4:])
        i += 1

        cue_lines: list[str] = []
        while i < len(lines) and lines[i].strip():
            cleaned = TAG_RE.sub("", lines[i]).strip()
            if cleaned:
                cue_lines.append(cleaned)
            i += 1

        cue_text = " ".join(cue_lines).strip()
        if cue_text:
            segments.append({"start": round(start, 2), "end": round(end, 2), "text": cue_text})
        i += 1

    return _dedupe(segments)


def _key(words: list[str]) -> list[str]:
    """Compare words ignoring case and punctuation, which cues vary on."""
    return [PUNCT_RE.sub("", w).lower() for w in words]


def strip_rolling_overlap(previous: str, current: str) -> str:
    """Drop the leading words of `current` that just repeat the tail of `previous`."""
    prev_words = previous.split()
    cur_words = current.split()
    limit = min(len(prev_words), len(cur_words))
    for size in range(limit, MIN_OVERLAP_WORDS - 1, -1):
        if _key(prev_words[-size:]) == _key(cur_words[:size]):
            return " ".join(cur_words[size:])
    return current


def _dedupe(segments: list[dict]) -> list[dict]:
    """Collapse rolling duplicates common in YouTube auto-subs."""
    out: list[dict] = []
    for seg in segments:
        if out and seg["text"] == out[-1]["text"]:
            out[-1]["end"] = seg["end"]
            continue
        if out and seg["text"].startswith(out[-1]["text"] + " "):
            out[-1]["text"] = seg["text"]
            out[-1]["end"] = seg["end"]
            continue
        if out:
            trimmed = strip_rolling_overlap(out[-1]["text"], seg["text"])
            if not trimmed:
                out[-1]["end"] = seg["end"]
                continue
            seg = {**seg, "text": trimmed}
        out.append(seg)
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
