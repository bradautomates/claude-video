#!/usr/bin/env python3
"""/watch entry point: download video, extract frames, parse transcript.

Prints a markdown report to stdout listing frame paths + transcript. Claude
then Reads each frame path to see the video.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from download import download, is_url, url_cache_dir  # noqa: E402
from frames import (  # noqa: E402
    MAX_FPS,
    auto_fps,
    auto_fps_focus,
    extract,
    extract_smart,
    format_time,
    get_metadata,
    parse_time,
)
from transcribe import filter_range, format_transcript, parse_vtt  # noqa: E402
from whisper import load_api_key, transcribe_video  # noqa: E402


# Persistent cache root, shared with whisper.py's chunk cache (~/.cache/watch/
# chunks). Downloads land in videos/<url-hash>/, extracted frames in
# frames/<params-hash>/. This is what makes multi-pass deep analysis cheap:
# pass 2's focused re-runs skip the download and often the extraction too.
CACHE_ROOT = Path.home() / ".cache" / "watch"
FRAME_CACHE_VERSION = 1


def _frame_cache_key(
    video_path: str,
    sampling: str,
    fps: float | None,
    target: int,
    resolution: int,
    start: float | None,
    end: float | None,
) -> str:
    """Hash of (video file identity, extraction params). Size + mtime in the
    identity means replacing the source file invalidates entries automatically."""
    p = Path(video_path).resolve()
    try:
        st = p.stat()
    except OSError:
        return ""
    ident = (
        f"v{FRAME_CACHE_VERSION}|{p}|{st.st_size}|{st.st_mtime_ns}|{sampling}|"
        f"{fps if fps is not None else 'auto'}|{target}|{resolution}|"
        f"{start if start is not None else '-'}|{end if end is not None else '-'}"
    )
    return hashlib.sha256(ident.encode()).hexdigest()[:32]


def _frame_cache_load(frames_dir: Path) -> tuple[list[dict], str] | None:
    index_path = frames_dir / "index.json"
    if not index_path.exists():
        return None
    try:
        data = json.loads(index_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if data.get("version") != FRAME_CACHE_VERSION:
        return None
    frames = data.get("frames")
    if not isinstance(frames, list) or not frames:
        return None
    if not all(Path(f["path"]).exists() for f in frames):
        return None
    return frames, str(data.get("mode") or "cached")


def _frame_cache_store(frames_dir: Path, frames: list[dict], mode: str) -> None:
    """Best-effort — a broken cache write must never break the run."""
    try:
        (frames_dir / "index.json").write_text(
            json.dumps({"version": FRAME_CACHE_VERSION, "frames": frames, "mode": mode}, indent=2)
        )
    except OSError:
        pass


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="watch",
        description="Download a video, extract auto-scaled frames, and surface the transcript.",
    )
    ap.add_argument("source", help="Video URL or local file path")
    ap.add_argument("--max-frames", type=int, default=80, help="Cap on frame count (default 80, hard max 100)")
    ap.add_argument("--resolution", type=int, default=512, help="Frame width in pixels (default 512)")
    ap.add_argument("--fps", type=float, default=None, help="Override auto-fps")
    ap.add_argument("--start", type=str, default=None, help="Range start (SS, MM:SS, or HH:MM:SS)")
    ap.add_argument("--end", type=str, default=None, help="Range end (SS, MM:SS, or HH:MM:SS)")
    ap.add_argument("--out-dir", type=str, default=None, help="Working directory (default: tmp)")
    ap.add_argument(
        "--no-whisper",
        action="store_true",
        help="Disable Whisper fallback. Report frames-only if no captions available.",
    )
    ap.add_argument(
        "--whisper",
        choices=["groq", "openai"],
        default=None,
        help="Force a specific Whisper backend. Default: prefer Groq, fall back to OpenAI.",
    )
    ap.add_argument(
        "--sampling",
        choices=["scene", "uniform"],
        default="scene",
        help="Frame sampling strategy (default scene: frames land on visual changes, "
        "uniform fill covers quiet stretches). --fps forces uniform.",
    )
    ap.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass the persistent cache (~/.cache/watch) — download and extract into the temp work dir.",
    )
    args = ap.parse_args()

    max_frames = min(args.max_frames, 100)
    use_cache = not args.no_cache

    if args.out_dir:
        work = Path(args.out_dir).expanduser().resolve()
    else:
        work = Path(tempfile.mkdtemp(prefix="watch-"))
    work.mkdir(parents=True, exist_ok=True)
    print(f"[watch] working dir: {work}", file=sys.stderr)

    if is_url(args.source):
        dl_dir = url_cache_dir(args.source, CACHE_ROOT) if use_cache else work / "download"
        print("[watch] downloading via yt-dlp…", file=sys.stderr)
    else:
        dl_dir = work / "download"
        print("[watch] using local file…", file=sys.stderr)
    dl = download(args.source, dl_dir)
    video_path = dl["video_path"]

    meta = get_metadata(video_path)
    full_duration = meta["duration_seconds"]

    start_sec = parse_time(args.start)
    end_sec = parse_time(args.end)

    if start_sec is not None and start_sec < 0:
        raise SystemExit("--start must be non-negative")
    if end_sec is not None and start_sec is not None and end_sec <= start_sec:
        raise SystemExit("--end must be greater than --start")
    if full_duration > 0 and start_sec is not None and start_sec >= full_duration:
        raise SystemExit(f"--start {start_sec:.1f}s is past end of video ({full_duration:.1f}s)")

    effective_start = start_sec if start_sec is not None else 0.0
    effective_end = end_sec if end_sec is not None else full_duration
    effective_duration = max(0.0, effective_end - effective_start)
    focused = start_sec is not None or end_sec is not None

    if focused:
        fps, target = auto_fps_focus(effective_duration, max_frames=max_frames)
    else:
        fps, target = auto_fps(effective_duration, max_frames=max_frames)
    fps_override: float | None = None
    sampling = args.sampling
    if args.fps is not None:
        fps_override = min(args.fps, MAX_FPS)
        fps = fps_override
        target = max(1, int(round(fps * effective_duration)))
        sampling = "uniform"  # an explicit rate is a uniform-sampling request

    scope = (
        f"{format_time(effective_start)}-{format_time(effective_end)} ({effective_duration:.1f}s)"
        if focused else f"full {effective_duration:.1f}s"
    )

    cache_key = _frame_cache_key(
        video_path, sampling, fps_override, target, args.resolution, start_sec, end_sec,
    ) if use_cache else ""
    frames_dir = CACHE_ROOT / "frames" / cache_key if cache_key else work / "frames"

    cached_frames = _frame_cache_load(frames_dir) if cache_key else None
    if cached_frames is not None:
        frames, sampling_mode = cached_frames
        print(f"[watch] frame cache hit: {len(frames)} frames in {frames_dir}", file=sys.stderr)
    else:
        print(f"[watch] extracting ~{target} frames ({sampling} sampling) over {scope}…", file=sys.stderr)
        frames_dir.mkdir(parents=True, exist_ok=True)
        if sampling == "scene":
            frames, sampling_mode = extract_smart(
                video_path,
                frames_dir,
                target=target,
                resolution=args.resolution,
                start_seconds=start_sec,
                end_seconds=end_sec,
                duration=full_duration,
            )
        else:
            frames = extract(
                video_path,
                frames_dir,
                fps=fps,
                resolution=args.resolution,
                max_frames=max_frames,
                start_seconds=start_sec,
                end_seconds=end_sec,
            )
            sampling_mode = f"uniform @ {fps:.3f} fps"
        if cache_key and frames:
            _frame_cache_store(frames_dir, frames, sampling_mode)

    transcript_segments: list[dict] = []
    transcript_text: str | None = None
    transcript_source: str | None = None
    transcript_failure: str | None = None
    if dl.get("subtitle_path"):
        try:
            all_segments = parse_vtt(dl["subtitle_path"])
            transcript_segments = filter_range(all_segments, start_sec, end_sec) if focused else all_segments
            transcript_text = format_transcript(transcript_segments)
            transcript_source = "captions"
        except Exception as exc:
            print(f"[watch] subtitle parse failed: {exc}", file=sys.stderr)
            transcript_failure = f"Subtitle parse failed: {exc}"

    if not transcript_segments:
        if args.no_whisper:
            transcript_failure = transcript_failure or "Whisper disabled via --no-whisper"
        else:
            backend, api_key = load_api_key(args.whisper)
            if backend and api_key:
                try:
                    all_segments, used_backend, chunk_failures = transcribe_video(
                        video_path,
                        work / "audio.mp3",
                        backend=backend,
                        api_key=api_key,
                        start_seconds=start_sec,
                        end_seconds=end_sec,
                    )
                    transcript_segments = filter_range(all_segments, start_sec, end_sec) if focused else all_segments
                    transcript_text = format_transcript(transcript_segments)
                    transcript_source = f"whisper ({used_backend})"
                    if chunk_failures:
                        ranges = ", ".join(f"{format_time(s)}-{format_time(e)}" for s, e, _ in chunk_failures)
                        transcript_failure = f"{len(chunk_failures)} chunk(s) failed: {ranges}"
                    else:
                        transcript_failure = None
                except SystemExit as exc:
                    print(f"[watch] whisper fallback failed: {exc}", file=sys.stderr)
                    transcript_failure = f"Whisper ({backend}) failed: {exc}"
            else:
                hint = (
                    f"--whisper {args.whisper} was set but the matching API key is missing"
                    if args.whisper else
                    "no subtitles and no Whisper API key found"
                )
                setup_py = SCRIPT_DIR / "setup.py"
                print(
                    f"[watch] {hint} — run `python3 {setup_py}` to enable the Whisper fallback",
                    file=sys.stderr,
                )
                transcript_failure = hint

    info = dl.get("info") or {}

    print()
    print("# watch: video report")
    print()
    print(f"- **Source:** {args.source}")
    if info.get("title"):
        print(f"- **Title:** {info['title']}")
    if info.get("uploader"):
        print(f"- **Uploader:** {info['uploader']}")
    print(f"- **Duration:** {format_time(full_duration)} ({full_duration:.1f}s)")
    if focused:
        print(
            f"- **Focus range:** {format_time(effective_start)} → {format_time(effective_end)} "
            f"({effective_duration:.1f}s)"
        )
    if meta.get("width") and meta.get("height"):
        print(f"- **Resolution:** {meta['width']}x{meta['height']} ({meta.get('codec') or 'unknown codec'})")
    mode = "focused" if focused else "full"
    print(f"- **Frames:** {len(frames)}, {sampling_mode}, {mode} mode (budget {target}, max {max_frames})")
    print(f"- **Frame size:** {args.resolution}px wide")
    if transcript_segments:
        in_range = " in range" if focused else ""
        print(
            f"- **Transcript:** {len(transcript_segments)} segments{in_range} "
            f"(via {transcript_source or 'captions'})"
        )
    else:
        print("- **Transcript:** none available")

    if not focused and full_duration > 600:
        mins = int(full_duration // 60)
        print()
        print(
            f"> **Warning:** This is a {mins}-minute video. Frame coverage is sparse at this length — "
            "accuracy degrades noticeably on anything over 10 minutes. For better results, "
            "re-run with `--start HH:MM:SS --end HH:MM:SS` to zoom into a specific section."
        )

    print()
    print("## Frames")
    print()
    print(f"Frames live at: `{frames_dir}`")
    print()
    print(
        "**Read each frame path below with the Read tool to view the image.** "
        "Frames are in chronological order; `t=MM:SS` is the absolute timestamp in the source video."
    )
    print()
    for frame in frames:
        print(f"- `{frame['path']}` (t={format_time(frame['timestamp_seconds'])})")

    print()
    print("## Transcript")
    print()
    if transcript_text:
        label = transcript_source or "captions"
        if focused:
            print(f"_Source: {label}. Filtered to {format_time(effective_start)} → {format_time(effective_end)}:_")
        else:
            print(f"_Source: {label}._")
        if transcript_failure:
            print()
            print(f"> **Partial transcript:** {transcript_failure}. Those windows are missing from the text below.")
        print()
        print("```")
        print(transcript_text)
        print("```")
    elif focused and dl.get("subtitle_path"):
        print(f"_No transcript lines fell inside {format_time(effective_start)} → {format_time(effective_end)}._")
    else:
        setup_py = SCRIPT_DIR / "setup.py"
        reason = transcript_failure or "no captions available and Whisper not attempted"
        print(
            f"_No transcript available — proceed with frames only. {reason}. "
            f"To configure Whisper, run `python3 {setup_py}`._"
        )

    print()
    print("---")
    if use_cache:
        print(
            f"_Work dir: `{work}` — delete when done. Video and frames are cached under "
            f"`{CACHE_ROOT}` (do NOT delete) — re-runs and focused zooms on this video are fast._"
        )
    else:
        print(f"_Work dir: `{work}` — delete when done._")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
