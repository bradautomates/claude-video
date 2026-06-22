#!/usr/bin/env python3
"""Split a long video into chunks for the TwelveLabs provider.

Two reasons to chunk before handing a video to Pegasus on-the-fly analysis:

  1. Duration — the user asked for ~30-minute chunks so each analysis stays
     focused and temporally dense (Pegasus itself tops out at 2 hours).
  2. Upload size — direct asset upload is capped at 200 MB. We size chunks by
     the file's average bitrate so each piece lands under that cap without
     re-encoding (ffmpeg `-c copy`, keyframe-snapped, fast). Because stream-copy
     can only cut on keyframes (so a segment runs to the next keyframe past the
     requested time) and bitrate is non-uniform, we verify every produced chunk
     and re-split at a smaller interval if any piece still came out too large.

Chunk offsets come from ffmpeg's segment list (absolute seconds in the source
timeline), so the merged report can label each chunk's real start time.
"""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path


# Keep chunks under the TwelveLabs direct-upload ceiling (200 MB). 190 MiB is a
# deliberate pre-buffer; the 0.85 planning factor below leaves further room for
# VBR spikes and keyframe snap-forward, and split() re-splits anything that
# still overshoots.
MAX_DIRECT_UPLOAD_BYTES = 190 * 1024 * 1024
PLAN_SAFETY_FACTOR = 0.85
# Pegasus rejects clips shorter than 4s; never emit a chunk below this.
MIN_CLIP_SECONDS = 4.0
# How many times to halve the interval when a produced chunk is still oversized.
MAX_RESPLIT_PASSES = 3


def needs_chunking(duration_seconds: float, file_bytes: int, chunk_seconds: float,
                   max_bytes: int = MAX_DIRECT_UPLOAD_BYTES) -> bool:
    return duration_seconds > chunk_seconds or file_bytes > max_bytes


def plan_chunk_seconds(duration_seconds: float, file_bytes: int, chunk_seconds: float,
                       max_bytes: int = MAX_DIRECT_UPLOAD_BYTES) -> float:
    """Effective chunk length: the smaller of the requested duration and the
    longest span whose average-bitrate size fits under the upload cap (with a
    safety factor for VBR spikes and keyframe overshoot)."""
    limit = float(chunk_seconds)
    if file_bytes > 0 and duration_seconds > 0:
        bytes_per_sec = file_bytes / duration_seconds
        size_limit = (max_bytes * PLAN_SAFETY_FACTOR) / bytes_per_sec
        limit = min(limit, size_limit)
    return max(MIN_CLIP_SECONDS, limit)


def _ffprobe_duration(path: Path) -> float | None:
    """Probe a file's duration in seconds; None if it can't be determined."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True,
        )
        text = (out.stdout or "").strip()
        return float(text) if text else None
    except (ValueError, OSError):
        return None


def _run_segment(video_path: str, out_dir: Path, chunk_seconds: float) -> list[dict]:
    """One stream-copy segmentation pass. Returns chunk dicts (offsets from the
    segment list, durations from it or from ffprobe; duration is None if unknown)."""
    for existing in out_dir.glob("chunk_*.mp4"):
        existing.unlink()
    segment_list = out_dir / "segments.csv"
    pattern = str(out_dir / "chunk_%03d.mp4")
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(Path(video_path).resolve()),
        "-c", "copy", "-map", "0",
        "-f", "segment",
        "-segment_time", f"{chunk_seconds:.3f}",
        "-reset_timestamps", "1",
        "-segment_list", str(segment_list),
        "-segment_list_type", "csv",
        pattern,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"ffmpeg chunking failed: {result.stderr.strip()}")

    chunks = _read_segment_list(segment_list, out_dir)
    if not chunks:
        # Fall back to nominal offsets if the segment list was unusable.
        for i, p in enumerate(sorted(out_dir.glob("chunk_*.mp4"))):
            chunks.append({"path": str(p), "start_seconds": round(i * chunk_seconds, 2),
                           "duration": _ffprobe_duration(p)})
    return chunks


def split(video_path: str, out_dir: Path, chunk_seconds: float) -> list[dict]:
    """Segment the video into ~chunk_seconds pieces with stream copy.

    Returns a list of {path, start_seconds, duration}, ordered, with absolute
    start offsets read from ffmpeg's segment list. If any produced chunk still
    exceeds the upload cap (VBR spike / long GOP), the interval is halved and the
    whole video re-split, up to MAX_RESPLIT_PASSES times. Pieces shorter than
    MIN_CLIP_SECONDS are dropped (Pegasus would reject them); pieces of unknown
    duration are kept.
    """
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is not installed. Install with: brew install ffmpeg")
    out_dir.mkdir(parents=True, exist_ok=True)

    secs = chunk_seconds
    chunks: list[dict] = []
    for attempt in range(MAX_RESPLIT_PASSES):
        chunks = _run_segment(video_path, out_dir, secs)
        oversized = [c for c in chunks
                     if Path(c["path"]).stat().st_size > MAX_DIRECT_UPLOAD_BYTES]
        if not oversized:
            break
        if attempt < MAX_RESPLIT_PASSES - 1:
            secs = max(MIN_CLIP_SECONDS, secs / 2)
            print(
                f"[watch] {len(oversized)} chunk(s) over the upload cap — "
                f"re-splitting at ~{secs / 60:.1f} min",
                file=sys.stderr,
            )

    # Drop pieces known to be under Pegasus's 4s floor; keep unknown-duration ones.
    kept = [c for c in chunks if c["duration"] is None or c["duration"] >= MIN_CLIP_SECONDS]
    dropped = len(chunks) - len(kept)
    if dropped:
        print(f"[watch] dropped {dropped} sub-{MIN_CLIP_SECONDS:.0f}s trailing chunk(s)", file=sys.stderr)
    return kept


def trim(video_path: str, out_path: Path, start_seconds: float,
         duration_seconds: float | None) -> Path:
    """Stream-copy a [start, start+duration] window to out_path (fast seek)."""
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is not installed. Install with: brew install ffmpeg")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    if start_seconds and start_seconds > 0:
        cmd += ["-ss", f"{start_seconds:.3f}"]
    cmd += ["-i", str(Path(video_path).resolve()), "-c", "copy", "-map", "0"]
    if duration_seconds is not None and duration_seconds > 0:
        cmd += ["-t", f"{duration_seconds:.3f}"]
    cmd += [str(out_path.resolve())]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not out_path.exists() or out_path.stat().st_size == 0:
        raise SystemExit(f"ffmpeg trim failed: {result.stderr.strip()}")
    return out_path


def _read_segment_list(segment_list: Path, out_dir: Path) -> list[dict]:
    """Parse ffmpeg's CSV segment list: filename,start_time,end_time (seconds)."""
    if not segment_list.exists():
        return []
    chunks: list[dict] = []
    try:
        with segment_list.open(newline="", encoding="utf-8") as fh:
            for row in csv.reader(fh):
                if len(row) < 3:
                    continue
                name, start, end = row[0], row[1], row[2]
                try:
                    start_f, end_f = float(start), float(end)
                except ValueError:
                    continue
                path = out_dir / Path(name).name
                if not path.exists():
                    continue
                chunks.append({
                    "path": str(path),
                    "start_seconds": round(start_f, 2),
                    "duration": round(max(0.0, end_f - start_f), 2),
                })
    except OSError:
        return []
    return chunks


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: chunk.py <video-path> <out-dir> [chunk-seconds]", file=sys.stderr)
        raise SystemExit(2)
    secs = float(sys.argv[3]) if len(sys.argv) > 3 else 1800.0
    result = split(sys.argv[1], Path(sys.argv[2]), secs)
    print(json.dumps(result, indent=2))
