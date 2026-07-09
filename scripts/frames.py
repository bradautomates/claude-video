#!/usr/bin/env python3
"""Probe video metadata and extract frames on a fixed budget.

Two sampling strategies share the same duration-based frame budget:

- scene (default): a detection pass finds visual change points, the budget is
  spent on the strongest changes first, and remaining slots split the largest
  temporal gaps so coverage never collapses. Every frame carries new
  information — slide flips and cuts land exactly on a frame instead of
  between two uniform samples, and talking-head stretches stop eating the
  budget with near-identical frames.
- uniform: constant-fps sampling (the pre-scene behavior). Used as automatic
  fallback when detection finds nothing, and forced when the caller pins an
  explicit --fps.

Token cost scales with frame count, so budget-by-duration keeps short videos
dense and long videos capped. When a user-specified range is passed,
focused-mode budgets denser (they are zooming in for detail).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


MAX_FPS = 2.0

# Scene detection: a low threshold collects *candidates* (slide changes and UI
# actions score far below hard cuts); plan_timestamps then ranks by score, so
# noise candidates only win slots after every stronger change already has one.
SCENE_THRESHOLD = 0.04
# Detection decodes the whole range once; scoring at 160px is ~identical to
# full-res for change detection and several times faster.
SCENE_DETECT_WIDTH = 160
# Two selected frames closer than this are near-duplicates — skip the weaker.
MIN_FRAME_GAP = 0.5
EXTRACT_WORKERS = 8


def _clamp_fps(fps: float, duration_seconds: float, max_frames: int) -> tuple[float, int]:
    fps = min(fps, MAX_FPS)
    target = min(max_frames, max(1, int(round(fps * duration_seconds))))
    return fps, target


def parse_time(value: str | float | int | None) -> float | None:
    """Parse SS, MM:SS, or HH:MM:SS (with optional .ms) into seconds."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s:
        return None
    parts = s.split(":")
    try:
        if len(parts) == 1:
            return float(parts[0])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except ValueError:
        pass
    raise SystemExit(f"Cannot parse time value: {value!r} (expected SS, MM:SS, or HH:MM:SS)")


def format_time(seconds: float) -> str:
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, sec = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def get_metadata(video_path: str) -> dict:
    if shutil.which("ffprobe") is None:
        raise SystemExit("ffprobe is not installed. Install with: brew install ffmpeg")

    result = subprocess.run(
        [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            video_path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(f"ffprobe failed: {result.stderr.strip()}")

    data = json.loads(result.stdout or "{}")
    streams = data.get("streams", [])
    fmt = data.get("format", {})
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

    duration = float(fmt.get("duration") or video_stream.get("duration") or 0)
    return {
        "duration_seconds": duration,
        "width": video_stream.get("width"),
        "height": video_stream.get("height"),
        "codec": video_stream.get("codec_name"),
        "size_bytes": int(fmt.get("size") or 0),
        "has_audio": audio_stream is not None,
    }


def auto_fps(duration_seconds: float, max_frames: int = 100) -> tuple[float, int]:
    """Pick fps that targets a sensible frame budget for full-video scans."""
    if duration_seconds <= 0:
        return 1.0, 1

    if duration_seconds <= 30:
        target = min(max_frames, max(12, int(round(duration_seconds))))
    elif duration_seconds <= 60:
        target = min(max_frames, 40)
    elif duration_seconds <= 180:  # 3 min
        target = min(max_frames, 60)
    elif duration_seconds <= 600:  # 10 min
        target = min(max_frames, 80)
    else:
        target = max_frames

    return _clamp_fps(target / duration_seconds, duration_seconds, max_frames)


def auto_fps_focus(duration_seconds: float, max_frames: int = 100) -> tuple[float, int]:
    """Denser budget for user-specified ranges — they are zooming in for detail."""
    if duration_seconds <= 0:
        return min(MAX_FPS, 2.0), 2

    if duration_seconds <= 5:
        target = min(max_frames, max(10, int(round(duration_seconds * 6))))
    elif duration_seconds <= 15:
        target = min(max_frames, max(30, int(round(duration_seconds * 4))))
    elif duration_seconds <= 30:
        target = min(max_frames, 60)
    elif duration_seconds <= 60:
        target = min(max_frames, 80)
    elif duration_seconds <= 180:
        target = max_frames
    else:
        target = max_frames

    return _clamp_fps(target / duration_seconds, duration_seconds, max_frames)


def detect_scenes(
    video_path: str,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
    threshold: float = SCENE_THRESHOLD,
) -> list[tuple[float, float]]:
    """Return (timestamp, score) scene-change candidates on the absolute timeline.

    Runs one decode pass over the requested range with ffmpeg's scene filter.
    An empty result (no changes, or ffmpeg failure) tells the caller to fall
    back to uniform sampling.
    """
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is not installed. Install with: brew install ffmpeg")

    seek: list[str] = []
    offset = 0.0
    if start_seconds is not None and start_seconds > 0:
        seek += ["-ss", f"{start_seconds:.3f}"]
        offset = start_seconds
    if end_seconds is not None:
        seek += ["-to", f"{end_seconds:.3f}"]

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-nostats",
        *seek,
        "-i", video_path,
        "-vf",
        f"scale={SCENE_DETECT_WIDTH}:-2,select='gt(scene,{threshold})',metadata=print:file=-",
        "-f", "null", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[watch] scene detection failed: {result.stderr.strip()[:200]}", file=sys.stderr)
        return []

    # metadata=print emits pairs of lines per selected frame:
    #   frame:12  pts:307200  pts_time:4.096
    #   lavfi.scene_score=0.086520
    scenes: list[tuple[float, float]] = []
    pts_time: float | None = None
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("frame:") and "pts_time:" in line:
            try:
                pts_time = float(line.rsplit("pts_time:", 1)[1].split()[0])
            except (ValueError, IndexError):
                pts_time = None
        elif line.startswith("lavfi.scene_score=") and pts_time is not None:
            try:
                score = float(line.split("=", 1)[1])
            except ValueError:
                score = threshold
            scenes.append((round(offset + pts_time, 3), score))
            pts_time = None
    return scenes


def plan_timestamps(
    scenes: list[tuple[float, float]],
    start: float,
    end: float,
    target: int,
    min_gap: float = MIN_FRAME_GAP,
) -> list[float]:
    """Pick up to `target` timestamps in [start, end): strongest scene changes
    first, then split the largest remaining temporal gaps so quiet stretches
    still get floor coverage."""
    duration = end - start
    if duration <= 0 or target < 1:
        return []

    selected: list[float] = [start]

    def far_enough(t: float) -> bool:
        return all(abs(t - s) >= min_gap for s in selected)

    for t, _score in sorted(scenes, key=lambda x: -x[1]):
        if len(selected) >= target:
            break
        if start <= t < end and far_enough(t):
            selected.append(t)

    while len(selected) < target:
        points = sorted(selected) + [end]
        gap, lo = max((points[i + 1] - points[i], points[i]) for i in range(len(points) - 1))
        if gap < 2 * min_gap:
            break
        selected.append(lo + gap / 2)

    return sorted(selected)


def extract_at_timestamps(
    video_path: str,
    out_dir: Path,
    timestamps: list[float],
    resolution: int = 512,
) -> list[dict]:
    """Grab one frame per timestamp with an accurate seek. Timestamps in the
    report are the requested seek points, exact by construction — no i/fps
    drift. A seek that lands past the last frame is dropped, not fatal."""
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is not installed. Install with: brew install ffmpeg")

    out_dir.mkdir(parents=True, exist_ok=True)
    for existing in out_dir.glob("frame_*.jpg"):
        existing.unlink()

    def grab(item: tuple[int, float]) -> dict | None:
        i, t = item
        out = out_dir / f"frame_{i:04d}.jpg"
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-ss", f"{t:.3f}",
            "-i", video_path,
            "-frames:v", "1",
            "-vf", f"scale={resolution}:-2",
            "-q:v", "4",
            str(out),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 or not out.exists() or out.stat().st_size == 0:
            return None
        return {"index": i, "timestamp_seconds": round(t, 2), "path": str(out)}

    with ThreadPoolExecutor(max_workers=EXTRACT_WORKERS) as pool:
        results = list(pool.map(grab, enumerate(timestamps)))
    return [f for f in results if f is not None]


def extract_smart(
    video_path: str,
    out_dir: Path,
    target: int,
    resolution: int = 512,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
    duration: float | None = None,
) -> tuple[list[dict], str]:
    """Scene-aware extraction with uniform fallback. Returns (frames, mode)
    where mode describes how the budget was spent, e.g. "scene (23 cuts + 57 fill)"."""
    eff_start = start_seconds if start_seconds is not None else 0.0
    if end_seconds is not None:
        eff_end = end_seconds
    elif duration is not None:
        eff_end = duration
    else:
        eff_end = get_metadata(video_path)["duration_seconds"]

    scenes = detect_scenes(video_path, start_seconds, end_seconds)
    if scenes:
        timestamps = plan_timestamps(scenes, eff_start, eff_end, target)
        frames = extract_at_timestamps(video_path, out_dir, timestamps, resolution)
        if frames:
            scene_ts = {t for t, _ in scenes}
            n_cuts = sum(1 for f in frames if any(abs(f["timestamp_seconds"] - t) < 0.01 for t in scene_ts))
            return frames, f"scene ({n_cuts} changes + {len(frames) - n_cuts} fill)"

    eff_duration = max(0.0, eff_end - eff_start)
    fps = min(MAX_FPS, target / eff_duration) if eff_duration > 0 else 1.0
    frames = extract(
        video_path, out_dir,
        fps=fps,
        resolution=resolution,
        max_frames=target,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    )
    return frames, "uniform (no scene changes detected)"


def extract(
    video_path: str,
    out_dir: Path,
    fps: float,
    resolution: int = 512,
    max_frames: int = 100,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
) -> list[dict]:
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is not installed. Install with: brew install ffmpeg")

    out_dir.mkdir(parents=True, exist_ok=True)
    for existing in out_dir.glob("frame_*.jpg"):
        existing.unlink()

    output_pattern = str(out_dir / "frame_%04d.jpg")
    cmd: list[str] = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
    ]

    # -ss before -i = fast seek (keyframe-snap, good enough for preview frames).
    if start_seconds is not None:
        cmd += ["-ss", f"{start_seconds:.3f}"]
    if end_seconds is not None:
        cmd += ["-to", f"{end_seconds:.3f}"]

    cmd += [
        "-i", video_path,
        "-vf", f"fps={fps},scale={resolution}:-2",
        "-frames:v", str(max_frames),
        "-q:v", "4",
        output_pattern,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"ffmpeg frame extraction failed: {result.stderr.strip()}")

    offset = start_seconds or 0.0
    frames = sorted(out_dir.glob("frame_*.jpg"))
    return [
        {
            "index": i,
            "timestamp_seconds": round(offset + (i / fps if fps > 0 else 0.0), 2),
            "path": str(p),
        }
        for i, p in enumerate(frames)
    ]


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "usage: frames.py <video-path> <out-dir> [--fps F] [--resolution W] "
            "[--max-frames N] [--start T] [--end T]",
            file=sys.stderr,
        )
        raise SystemExit(2)

    video = sys.argv[1]
    out = Path(sys.argv[2])
    args = sys.argv[3:]

    fps_override = None
    resolution = 512
    max_frames = 100
    start_arg = None
    end_arg = None
    i = 0
    while i < len(args):
        if args[i] == "--fps":
            fps_override = float(args[i + 1]); i += 2
        elif args[i] == "--resolution":
            resolution = int(args[i + 1]); i += 2
        elif args[i] == "--max-frames":
            max_frames = int(args[i + 1]); i += 2
        elif args[i] == "--start":
            start_arg = args[i + 1]; i += 2
        elif args[i] == "--end":
            end_arg = args[i + 1]; i += 2
        else:
            i += 1

    meta = get_metadata(video)
    start_sec = parse_time(start_arg)
    end_sec = parse_time(end_arg)
    full_duration = meta["duration_seconds"]

    effective_start = start_sec if start_sec is not None else 0.0
    effective_end = end_sec if end_sec is not None else full_duration
    effective_duration = max(0.0, effective_end - effective_start)

    focused = start_sec is not None or end_sec is not None
    if focused:
        fps, target = auto_fps_focus(effective_duration, max_frames=max_frames)
    else:
        fps, target = auto_fps(effective_duration, max_frames=max_frames)
    if fps_override is not None:
        fps = fps_override
        target = max(1, int(round(fps * effective_duration)))

    frames = extract(
        video, out,
        fps=fps,
        resolution=resolution,
        max_frames=max_frames,
        start_seconds=start_sec,
        end_seconds=end_sec,
    )
    print(json.dumps(
        {"meta": meta, "fps": fps, "target": target, "focused": focused, "frames": frames},
        indent=2,
    ))
