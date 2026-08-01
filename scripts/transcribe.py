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


def parse_vtt(path: str, rolling: bool | None = None) -> list[dict]:
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

    if rolling is None:
        rolling = _looks_like_rolling_captions(segments)
    return _dedupe(segments, rolling=rolling)


def _normalized_words(text: str) -> list[str]:
    return [re.sub(r"^\W+|\W+$", "", word).casefold() for word in text.split()]


def _rolling_overlap(left: str, right: str, min_words: int = 3) -> int:
    """Return the longest word overlap between left's suffix and right's prefix."""
    left_words = _normalized_words(left)
    right_words = _normalized_words(right)
    for size in range(min(len(left_words), len(right_words)), min_words - 1, -1):
        if left_words[-size:] == right_words[:size]:
            return size
    return 0


def _looks_like_rolling_captions(segments: list[dict]) -> bool:
    """Detect YouTube-style rolling captions without assuming a track type.

    Manual captions occasionally repeat a phrase, but auto captions overlap
    adjacent cues consistently. Requiring multiple overlaps and a minimum
    ratio keeps normal subtitle tracks untouched.
    """
    sample = segments[:200]
    if len(sample) < 3:
        return False
    hits = sum(
        1
        for previous, current in zip(sample, sample[1:])
        if _rolling_overlap(previous["text"], current["text"])
    )
    return hits >= 2 and hits / (len(sample) - 1) >= 0.08


def _dedupe(segments: list[dict], rolling: bool = False) -> list[dict]:
    """Collapse exact duplicates and optional rolling-caption word overlap."""
    out: list[dict] = []
    for seg in segments:
        if out and seg["text"] == out[-1]["text"]:
            out[-1]["end"] = seg["end"]
            continue
        if out and seg["text"].startswith(out[-1]["text"] + " "):
            out[-1]["text"] = seg["text"]
            out[-1]["end"] = seg["end"]
            continue
        cleaned = dict(seg)
        if rolling and out:
            overlap = _rolling_overlap(out[-1]["text"], cleaned["text"])
            if overlap:
                remaining = cleaned["text"].split()[overlap:]
                if not remaining:
                    out[-1]["end"] = cleaned["end"]
                    continue
                cleaned["text"] = " ".join(remaining)
        out.append(cleaned)
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
        if start >= 3600:
            stamp = f"[{start // 3600}:{start % 3600 // 60:02d}:{start % 60:02d}]"
        else:
            stamp = f"[{start // 60:02d}:{start % 60:02d}]"
        lines.append(f"{stamp} {seg['text']}")
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: transcribe.py <vtt-path>", file=sys.stderr)
        raise SystemExit(2)
    print(format_transcript(parse_vtt(sys.argv[1])))
