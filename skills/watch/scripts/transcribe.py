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


def _strip_overlap(prev_text: str, curr_text: str) -> str:
    """If curr starts with a suffix of prev (rolling-caption overlap), drop the overlap."""
    prev_words = prev_text.split()
    curr_words = curr_text.split()
    max_overlap = min(len(prev_words), len(curr_words))
    for k in range(max_overlap, 0, -1):
        if prev_words[-k:] == curr_words[:k]:
            return " ".join(curr_words[k:])
    return curr_text


def _dedupe(segments: list[dict]) -> list[dict]:
    """Collapse rolling duplicates common in YouTube auto-subs.

    Handles three patterns:
      1. Identical consecutive cues (extend the previous end time).
      2. Extension cues where next text starts with prev + " ..." (replace text).
      3. Overlap cues where next text repeats the tail of prev then adds new
         content (YouTube's default rolling display) — strip the repeated tail.
    """
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
            stripped = _strip_overlap(out[-1]["text"], seg["text"])
            if stripped != seg["text"] and stripped:
                out.append({"start": seg["start"], "end": seg["end"], "text": stripped})
                continue
            if not stripped:
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


def write_transcript_file(
    segments: list[dict],
    out_path: Path | str,
    *,
    source_label: str,
    title: str | None = None,
    focus_range: tuple[float, float] | None = None,
    bucket_seconds: int = 30,
) -> Path:
    """Write a clean, paragraph-grouped transcript markdown to disk.

    Groups segments into ~bucket_seconds paragraphs so the file reads as prose
    rather than a wall of two-second cues. Assumes segments have already been
    passed through parse_vtt / _dedupe (or Whisper output).
    """
    out_path = Path(out_path)
    header = ["# Transcript"]
    if title:
        header[0] = f"# Transcript — {title}"
    header.append("")
    header.append(f"Source: {source_label}.")
    if focus_range:
        s, e = focus_range
        header.append(
            f"Focus range: {int(s)//60:02d}:{int(s)%60:02d} → "
            f"{int(e)//60:02d}:{int(e)%60:02d}."
        )
    header.append(
        f"Timestamps are absolute (MM:SS). Grouped into ~{bucket_seconds}-second paragraphs."
    )
    header.append("")

    if not segments:
        header.append("_No transcript segments available._")
        out_path.write_text("\n".join(header), encoding="utf-8")
        return out_path

    buckets: dict[int, list[dict]] = {}
    for seg in segments:
        key = int(seg["start"]) // bucket_seconds
        buckets.setdefault(key, []).append(seg)

    body = []
    for key in sorted(buckets):
        start = int(buckets[key][0]["start"])
        stamp = f"**[{start // 60:02d}:{start % 60:02d}]**"
        text = " ".join(seg["text"] for seg in buckets[key])
        body.append(f"{stamp} {text}")
        body.append("")

    out_path.write_text("\n".join(header + body), encoding="utf-8")
    return out_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: transcribe.py <vtt-path>", file=sys.stderr)
        raise SystemExit(2)
    print(format_transcript(parse_vtt(sys.argv[1])))
