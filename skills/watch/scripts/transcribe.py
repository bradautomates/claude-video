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


_PUNCT = ".,!?;:\"'()[]—–-"


def _norm(word: str) -> str:
    return word.strip(_PUNCT).lower()


def _overlap(prev_words: list[str], cur_words: list[str]) -> int:
    """Longest k where prev's last k words equal cur's first k.

    Compared punctuation- and case-insensitively: YouTube re-emits the same
    words with different trailing punctuation as a cue scrolls ("blogai" then
    "blogai,"), which an exact match would miss.

    k must be >= 2. A single shared word is as likely to be a coincidence
    ("ir", "ne") as a real overlap, and dropping a legitimate word is a worse
    outcome than leaving one duplicated.
    """
    for k in range(min(len(prev_words), len(cur_words)), 1, -1):
        if [_norm(w) for w in prev_words[-k:]] == [_norm(w) for w in cur_words[:k]]:
            return k
    return 0


def _dedupe(segments: list[dict]) -> list[dict]:
    """Collapse the rolling duplication in YouTube auto-subs.

    A cue cycles through three phases as it scrolls:

        1. "100 metų, pasirodo buvo tiesa. A"
        2. "100 metų, pasirodo buvo tiesa. A pasirodo visos imperijos blogai"
        3. "pasirodo visos imperijos blogai, net"

    Phase 2 contains all of phase 1 and extends it -- the same line still being
    revealed, so it replaces its predecessor. Phase 3 has scrolled: it begins
    with the *tail* of phase 2, not the whole of it, so a `startswith(prev)`
    test never fires and the repetition survives. Handling only phase 1->2
    halves the cue count and leaves ~85% of the remaining segments repeating
    their predecessor, roughly doubling the transcript handed to the model.

    Phase 3 is the general case; phase 1->2 is the special case where the
    overlap covers all of the previous cue.
    """
    out: list[dict] = []
    for seg in segments:
        if not out:
            out.append(dict(seg))
            continue

        prev = out[-1]
        if seg["text"] == prev["text"]:
            prev["end"] = seg["end"]
            continue

        prev_words, cur_words = prev["text"].split(), seg["text"].split()
        k = _overlap(prev_words, cur_words)
        if k == 0:
            out.append(dict(seg))
            continue

        if k == len(prev_words):
            # Still the same line being revealed -- let it replace its predecessor.
            prev["text"] = seg["text"]
            prev["end"] = seg["end"]
            continue

        rest = cur_words[k:]
        if not rest:
            prev["end"] = seg["end"]
            continue
        out.append({"start": seg["start"], "end": seg["end"], "text": " ".join(rest)})
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
