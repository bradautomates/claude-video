#!/usr/bin/env python3
"""Parse a WebVTT subtitle file into a clean, timestamped transcript.

YouTube auto-subs use a *rolling* cue format that needs two things handled:

1. A cue body may start with a whitespace-only padding line. Terminating the
   body there (rather than on the real, empty separator line) silently drops the
   cue's content.
2. Each cue repeats the tail of the previous cue as a plain carry-over line
   before adding new words, and interleaves ~10ms "settle" cues that re-state the
   line just completed. Joining every line blindly roughly doubles the word count.

In that format only the lines carrying inline word-timing tags
(``<00:00:15.160><c> word</c>``) hold new content, so those identify the real
text exactly. Files without such tags (manual subtitles, Whisper output) are left
structurally alone and only pass through the overlap guard in `_dedupe`.
"""
from __future__ import annotations

import html
import re
import sys
from pathlib import Path


TS_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[.,](\d{3})\s+-->\s+(\d{2}):(\d{2}):(\d{2})[.,](\d{3})"
)
TAG_RE = re.compile(r"<[^>]+>")
# Inline per-word timing tag. Its presence marks the rolling auto-sub format,
# and marks which line inside a cue is new rather than carried over.
CUE_TAG_RE = re.compile(r"<\d{2}:\d{2}:\d{2}[.,]\d{3}>")

# A repeated run this long is a rolling duplicate, not a phrase the speaker
# happened to repeat across a cue boundary. Below it we keep the words: a stray
# duplicate is cheap, dropped speech is not.
MIN_OVERLAP_WORDS = 3

# Cues are short; never scan more than this many words looking for an overlap.
MAX_OVERLAP_WINDOW = 40


def _to_seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def _clean(line: str) -> str:
    return html.unescape(TAG_RE.sub("", line)).strip()


def parse_vtt(path: str) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    rolling = bool(CUE_TAG_RE.search(text))
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

        # The cue body ends at the blank separator line. A whitespace-only line
        # is padding *inside* the body, so skip it instead of stopping.
        body: list[str] = []
        while i < len(lines) and lines[i] != "" and not TS_RE.match(lines[i]):
            if lines[i].strip():
                body.append(lines[i])
            i += 1

        if rolling:
            # Only tagged lines carry new words. A cue with none is a settle
            # repeat; keep its lines and let _dedupe drop what was already said,
            # so a trailing line with no follow-up cue still survives.
            body = [line for line in body if CUE_TAG_RE.search(line)] or body

        cue_text = " ".join(filter(None, (_clean(line) for line in body))).strip()
        if cue_text:
            segments.append({"start": round(start, 2), "end": round(end, 2), "text": cue_text})
        i += 1

    return _dedupe(segments)


def _norm(word: str) -> str:
    """Comparison key for a word: case- and punctuation-insensitive."""
    return word.casefold().strip(".,!?;:\"'()[]")


def _overlap(prev_words: list[str], new_words: list[str]) -> int:
    """Length of the longest tail of prev_words that repeats the head of new_words."""
    limit = min(len(prev_words), len(new_words), MAX_OVERLAP_WINDOW)
    if not limit:
        return 0
    prev_tail = [_norm(w) for w in prev_words[-limit:]]
    new_head = [_norm(w) for w in new_words[:limit]]
    for k in range(limit, 0, -1):
        if prev_tail[-k:] == new_head[:k]:
            return k
    return 0


def _dedupe(segments: list[dict]) -> list[dict]:
    """Drop text a segment repeats from the one before it.

    Each segment keeps only the words it actually adds. A segment that adds
    nothing extends the previous one's end time, so no time coverage is lost.
    """
    out: list[dict] = []
    for seg in segments:
        if not out:
            out.append(dict(seg))
            continue

        prev = out[-1]
        if seg["text"] == prev["text"]:
            prev["end"] = max(prev["end"], seg["end"])
            continue

        prev_words = prev["text"].split()
        new_words = seg["text"].split()
        k = _overlap(prev_words, new_words)

        # Trust the overlap when it is long enough to be a real carry-over, or
        # when it swallows a whole multi-word cue (a settle repeat).
        if not (k >= MIN_OVERLAP_WORDS or (k == len(new_words) and k >= 2)):
            out.append(dict(seg))
            continue

        remainder = new_words[k:]
        if not remainder:
            prev["end"] = max(prev["end"], seg["end"])
            continue
        out.append({"start": seg["start"], "end": seg["end"], "text": " ".join(remainder)})

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
