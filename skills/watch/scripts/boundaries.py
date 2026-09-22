#!/usr/bin/env python3
"""Structural boundaries: fades, freezes and silence, in one cheap ffmpeg pass.

Scene detection scores *rate* of pixel change per frame. That is the wrong
instrument for the two edits that matter most in a montage:

* a **fade** spreads its change over 20-30 frames, so no single frame crosses the
  scene threshold — while a spinning slot reel or a panning camera clears it on
  every frame. The detector reliably misses the boundary and reliably fires on
  the middle of a shot.
* a **held card** changes nothing at all, so it is invisible to a change detector
  even though "this frame stayed on screen for 3 seconds" is exactly the kind of
  structure a viewer reads as a section break.

``blackdetect``, ``freezedetect`` and ``silencedetect`` are *state* detectors —
they report intervals, not per-frame deltas — so they see precisely what the
scene score cannot. One pass over the video yields a structural timeline to 0.1s,
which is better than any frame-sampling interval can offer, and the frame right
after a fade-up is where title and loader cards live.

Cost: one decode at ``PROBE_WIDTH`` with ``-f null`` (no encode, no files
written). Roughly a tenth of the scene pass on the same clip.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path


# Downscale before detection: these filters look at frame statistics, which
# survive a shrink, and decoding to 160px is where the cheapness comes from.
PROBE_WIDTH = 160
# blackdetect: d = minimum duration to report. 0.1s is two or three frames at
# 30fps — long enough to skip a single black frame inside normal footage, short
# enough to catch a fast dip-to-black between segments.
BLACK_MIN_DURATION = 0.10
BLACK_PIXEL_THRESHOLD = 0.10
# freezedetect: n = noise tolerance, d = minimum duration. 0.5s of a genuinely
# static picture. -60dB tolerates encoder noise on an otherwise identical frame.
FREEZE_NOISE = "-60dB"
FREEZE_MIN_DURATION = 0.5
# silencedetect: -30dB for 1s. Tuned to separate "music/speech" from "nothing",
# not to find pauses between words.
SILENCE_NOISE = "-30dB"
SILENCE_MIN_DURATION = 1.0
# Where to pin a frame relative to a fade-up. Far enough past black_end that the
# incoming picture has actually arrived, close enough that a 3s card is still on
# screen. This is the frame the scene detector structurally cannot find.
POST_FADE_OFFSET = 0.5

_BLACK_RE = re.compile(
    r"black_start:([0-9.]+)\s+black_end:([0-9.]+)\s+black_duration:([0-9.]+)"
)
# freezedetect and silencedetect report start and end on separate lines.
_FREEZE_START_RE = re.compile(r"freeze_start:\s*([0-9.]+)")
_FREEZE_END_RE = re.compile(r"freeze_end:\s*([0-9.]+)")
_SILENCE_START_RE = re.compile(r"silence_start:\s*(-?[0-9.]+)")
_SILENCE_END_RE = re.compile(r"silence_end:\s*(-?[0-9.]+)")


def _pair_intervals(starts: list[float], ends: list[float]) -> list[dict]:
    """Zip start/end marks into intervals, tolerating an unterminated last one.

    A segment still open when the stream ends prints its start and never its
    end; that is real structure (a video fading out on black), so it is kept
    with ``end=None`` rather than dropped.
    """
    out: list[dict] = []
    for i, start in enumerate(starts):
        end = ends[i] if i < len(ends) else None
        out.append({
            "start": round(start, 2),
            "end": round(end, 2) if end is not None else None,
            "duration": round(end - start, 2) if end is not None else None,
        })
    return out


def detect_boundaries(
    video_path: str,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
    has_audio: bool = True,
) -> dict:
    """Run one ffmpeg pass and return black / freeze / silence intervals.

    Times are absolute source seconds (the ``-ss`` offset is added back).
    Fail-open: any ffmpeg problem returns empty lists with an ``error`` key, so a
    boundaries failure degrades the report rather than killing a run that has
    already paid for a download.
    """
    empty: dict = {"black": [], "freeze": [], "silence": [], "has_audio": has_audio}
    if shutil.which("ffmpeg") is None:
        return {**empty, "error": "ffmpeg not installed"}

    cmd: list[str] = ["ffmpeg", "-hide_banner", "-nostats", "-loglevel", "info", "-y"]
    if start_seconds is not None:
        cmd += ["-ss", f"{start_seconds:.3f}"]
    if end_seconds is not None:
        cmd += ["-to", f"{end_seconds:.3f}"]
    cmd += ["-i", str(Path(video_path).resolve())]
    cmd += [
        "-vf",
        f"scale={PROBE_WIDTH}:-2,"
        f"blackdetect=d={BLACK_MIN_DURATION}:pix_th={BLACK_PIXEL_THRESHOLD},"
        f"freezedetect=n={FREEZE_NOISE}:d={FREEZE_MIN_DURATION}",
    ]
    if has_audio:
        # -af on a file with no audio stream makes ffmpeg exit 1 with
        # "Stream specifier ':a' ... matches no streams", so it is conditional.
        cmd += ["-af", f"silencedetect=n={SILENCE_NOISE}:d={SILENCE_MIN_DURATION}"]
    cmd += ["-f", "null", "-"]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
    except OSError as exc:
        return {**empty, "error": str(exc)}
    if result.returncode != 0:
        return {**empty, "error": (result.stderr or "").strip()[-300:]}

    log = result.stderr or ""
    offset = start_seconds or 0.0

    black = [
        {
            "start": round(offset + float(m.group(1)), 2),
            "end": round(offset + float(m.group(2)), 2),
            "duration": round(float(m.group(3)), 2),
        }
        for m in _BLACK_RE.finditer(log)
    ]

    freeze_starts = [offset + float(m.group(1)) for m in _FREEZE_START_RE.finditer(log)]
    freeze_ends = [offset + float(m.group(1)) for m in _FREEZE_END_RE.finditer(log)]
    freeze = _pair_intervals(freeze_starts, freeze_ends)

    silence: list[dict] = []
    if has_audio:
        # silencedetect clamps a silence starting at t=0 to a small negative
        # number on some builds; max(0, ...) keeps times on the real timeline.
        sil_starts = [offset + max(0.0, float(m.group(1))) for m in _SILENCE_START_RE.finditer(log)]
        sil_ends = [offset + max(0.0, float(m.group(1))) for m in _SILENCE_END_RE.finditer(log)]
        silence = _pair_intervals(sil_starts, sil_ends)

    return {"black": black, "freeze": freeze, "silence": silence, "has_audio": has_audio}


def post_fade_timestamps(
    boundaries: dict,
    limit: int | None = None,
    duration: float | None = None,
    offset: float = POST_FADE_OFFSET,
) -> list[float]:
    """Times to pin a frame at: just after each fade-up out of black.

    ``black_end`` is the instant the picture returns, so ``black_end + offset``
    is the incoming shot with its titles settled — the frame a scene detector
    structurally cannot find, and where loader and claim cards live.

    ``limit`` (when the frame budget is tight) keeps the *longest* fades, which
    are the strongest section breaks, then returns them in chronological order.
    """
    points: list[tuple[float, float]] = []
    for seg in boundaries.get("black", []):
        end = seg.get("end")
        if end is None:
            continue  # video ends on black; there is no incoming shot to pin
        t = round(end + offset, 2)
        if duration is not None and t >= duration:
            continue
        points.append((seg.get("duration") or 0.0, t))

    if limit is not None and len(points) > limit:
        points = sorted(points, key=lambda p: (-p[0], p[1]))[:limit]
    return sorted({t for _, t in points})


def _fmt(seconds: float | None) -> str:
    """``MM:SS.dd`` (``H:MM:SS.dd`` past an hour).

    Deliberately not :func:`frames.format_time`, which rounds to whole seconds:
    the precision IS the point of this section — rendering a 0.70s card as
    "00:01 → 00:01" would throw away exactly what makes it worth measuring.
    """
    if seconds is None:
        return "…"
    hours, rest = divmod(float(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{int(hours)}:{int(minutes):02d}:{secs:05.2f}"
    return f"{int(minutes):02d}:{secs:05.2f}"


def format_timeline(boundaries: dict, max_rows: int = 40) -> list[str]:
    """Render the boundaries as markdown lines for the report's Timeline section.

    Returns [] when nothing was detected, so the caller can omit the section.
    """
    lines: list[str] = []
    black = boundaries.get("black") or []
    freeze = boundaries.get("freeze") or []
    silence = boundaries.get("silence") or []

    if black:
        lines.append(
            f"- **Fades / cuts to black ({len(black)}).** A frame is pinned "
            f"{POST_FADE_OFFSET:g}s after each one (`reason=post-fade`); these are "
            "section boundaries a scene detector cannot see."
        )
        for seg in black[:max_rows]:
            lines.append(
                f"  - {_fmt(seg['start'])} → {_fmt(seg['end'])} "
                f"({seg['duration']:.2f}s black)" if seg.get("duration") is not None
                else f"  - {_fmt(seg['start'])} → end (black to end of range)"
            )
        if len(black) > max_rows:
            lines.append(f"  - …and {len(black) - max_rows} more")

    if freeze:
        lines.append(
            f"- **Held / frozen picture ({len(freeze)}).** Static for "
            f"{FREEZE_MIN_DURATION:g}s or more — title cards, paused playback, a slide left up."
        )
        for seg in freeze[:max_rows]:
            dur = f"{seg['duration']:.2f}s" if seg.get("duration") is not None else "to end of range"
            lines.append(f"  - {_fmt(seg['start'])} → {_fmt(seg['end'])} ({dur})")
        if len(freeze) > max_rows:
            lines.append(f"  - …and {len(freeze) - max_rows} more")

    if not boundaries.get("has_audio"):
        lines.append("- **Audio:** none — the source has no audio stream.")
    elif silence:
        total = sum(s["duration"] for s in silence if s.get("duration") is not None)
        lines.append(
            f"- **Silence ({len(silence)} stretch(es), {total:.1f}s total).** "
            "Below the speech floor; a transcript covering these is suspect."
        )
        for seg in silence[:max_rows]:
            dur = f"{seg['duration']:.2f}s" if seg.get("duration") is not None else "to end of range"
            lines.append(f"  - {_fmt(seg['start'])} → {_fmt(seg['end'])} ({dur})")
        if len(silence) > max_rows:
            lines.append(f"  - …and {len(silence) - max_rows} more")
    else:
        lines.append("- **Silence:** none detected — audio is continuous across the range.")

    return lines
