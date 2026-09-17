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
        # A VTT cue ends at a truly empty line. `.strip()` also treats a line
        # holding only whitespace as the terminator, which silently drops any
        # cue whose first line is a lone space -- a placeholder YouTube emits.
        while i < len(lines) and lines[i] != "":
            cleaned = TAG_RE.sub("", lines[i]).strip()
            if cleaned:
                cue_lines.append(cleaned)
            i += 1

        cue_text = " ".join(cue_lines).strip()
        if cue_text:
            segments.append({"start": round(start, 2), "end": round(end, 2), "text": cue_text})
        i += 1

    return _dedupe(segments)


_MIN_OVERLAP_WORDS = 3  # below this, a shared word is coincidence, not a caption scroll
_MAX_OVERLAP_GAP_SECONDS = 1.0  # rolling cues are back-to-back; a real gap means unrelated content


def _tail_head_overlap(prev_words: list[str], new_words: list[str]) -> int:
    """Longest run where the tail of prev_words equals the head of new_words."""
    for k in range(min(len(prev_words), len(new_words)), 0, -1):
        if prev_words[-k:] == new_words[:k]:
            return k
    return 0


def _dedupe(segments: list[dict]) -> list[dict]:
    """Collapse rolling duplicates common in YouTube auto-subs.

    YouTube auto-subs scroll two lines per cue: line 2 of cue N becomes line 1
    of cue N+1. That is neither byte-identical nor a strict prefix extension of
    the previous cue, so neither existing branch fires and every line ships
    twice. Find the longest run where the tail of the previous text equals the
    head of the incoming cue, and strip those words off the FRONT of the
    incoming cue.

    This check must run BEFORE the strict-prefix branch below. Stripping the
    overlap leaves a short segment that the NEXT cue legitimately starts with,
    so a prefix check running first re-merges the pair and undoes the
    granularity this preserves. The gap guard keeps the two branches apart:
    only back-to-back (rolling) cues reach the overlap path, so genuinely
    growing cues still fall through to the prefix branch.

    The cue stays its own segment with its own start/end. It is deliberately
    not spliced into the predecessor: rolling cues chain almost continuously,
    so splicing collapses most of a video into one or two giant segments --
    which destroys `[MM:SS]` granularity and makes filter_range() return the
    whole transcript for every window.
    """
    out: list[dict] = []
    for seg in segments:
        if out and seg["text"] == out[-1]["text"]:
            out[-1]["end"] = seg["end"]
            continue
        if out and seg["start"] - out[-1]["end"] <= _MAX_OVERLAP_GAP_SECONDS:
            prev_words = out[-1]["text"].split()
            new_words = seg["text"].split()
            overlap = _tail_head_overlap(prev_words, new_words)
            if overlap >= _MIN_OVERLAP_WORDS:
                remaining = new_words[overlap:]
                if not remaining:
                    continue  # cue fully consumed by the overlap
                seg = {"start": seg["start"], "end": seg["end"], "text": " ".join(remaining)}
        if out and seg["text"].startswith(out[-1]["text"] + " "):
            out[-1]["text"] = seg["text"]
            out[-1]["end"] = seg["end"]
            continue
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
