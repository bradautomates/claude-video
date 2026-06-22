#!/usr/bin/env python3
"""/watch entry point: download video, extract frames, parse transcript.

Prints a markdown report to stdout listing frame paths + transcript. Claude
then Reads each frame path to see the video.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

import chunk as chunkmod  # noqa: E402
import twelvelabs as tl  # noqa: E402
from download import download, is_url  # noqa: E402
from frames import MAX_FPS, auto_fps, auto_fps_focus, extract, format_time, get_metadata, parse_time  # noqa: E402
from transcribe import filter_range, format_transcript, parse_vtt  # noqa: E402
from whisper import load_api_key, transcribe_video  # noqa: E402


def _tl_prompt(question: str | None) -> str:
    """Build the Pegasus prompt. With a question, answer it + transcript;
    without, return a full structured report."""
    if question:
        return (
            "Watch this video and answer the following question, citing [MM:SS] timestamps "
            f'as evidence:\n\n"{question}"\n\n'
            'After your answer, add a section titled "## Transcript" containing a verbatim, '
            "timestamped transcript of the spoken dialogue (one [MM:SS] line per turn, speaker "
            "labels if distinguishable, original language)."
        )
    return (
        "Watch this video and produce a detailed markdown report with these sections:\n\n"
        "## Transcript\n"
        "A verbatim, timestamped transcript of all spoken dialogue. Prefix each line or speaker "
        "turn with an [MM:SS] timestamp. Include speaker labels if distinguishable. Preserve the "
        "original language.\n\n"
        "## On-screen text\n"
        "Any text shown on screen (titles, captions, slides, UI, signs), each with an [MM:SS] "
        "timestamp.\n\n"
        "## Visual walkthrough\n"
        "A chronological, scene-by-scene description of what is shown — setting, people, actions, "
        "notable visuals, and transitions — each with an [MM:SS] timestamp.\n\n"
        "## Summary\n"
        "A concise summary of what the video covers.\n\n"
        "All timestamps must be relative to the start of this clip."
    )


def run_twelvelabs(args, work, source, video_path, info, meta, full_duration,
                   effective_start, effective_duration, focused) -> int:
    """Analyze the video with TwelveLabs Pegasus and print a text report."""
    key = tl.load_api_key()
    if not key:
        setup_py = SCRIPT_DIR / "setup.py"
        raise SystemExit(
            "TWELVELABS_API_KEY not found. Set it in the environment or in "
            f"~/.config/watch/.env, then re-run. (Run `python3 {setup_py}` to scaffold the file.)"
        )

    model = args.tl_model
    chunk_seconds = max(60.0, args.chunk_minutes * 60.0)
    prompt = _tl_prompt(args.tl_prompt)

    # Pegasus rejects clips below its 4s floor — fail fast with a clear message
    # instead of uploading and getting an opaque API 400.
    target_duration = effective_duration if focused else full_duration
    if 0 < target_duration < tl.MIN_CLIP_SECONDS:
        raise SystemExit(
            f"TwelveLabs Pegasus needs clips at least {tl.MIN_CLIP_SECONDS:.0f}s long; "
            f"the {'requested range' if focused else 'video'} is {target_duration:.1f}s. "
            "Widen --start/--end, or use the default frames provider."
        )

    # Resolve the clip to analyze: trim first if the user asked for a range.
    if focused:
        analysis_path = str(work / "tl_clip.mp4")
        chunkmod.trim(video_path, Path(analysis_path), effective_start, effective_duration)
        base_offset = effective_start
        analysis_duration = effective_duration
    else:
        analysis_path = video_path
        base_offset = 0.0
        analysis_duration = full_duration

    size = Path(analysis_path).stat().st_size
    segments: list[dict] = []

    if chunkmod.needs_chunking(analysis_duration, size, chunk_seconds):
        eff = chunkmod.plan_chunk_seconds(analysis_duration, size, chunk_seconds)
        print(f"[watch] splitting into ~{eff / 60:.1f}-min chunks for Pegasus…", file=sys.stderr)
        chunks = chunkmod.split(analysis_path, work / "chunks", eff)
        if not chunks:
            raise SystemExit("chunking produced no analyzable segments")
        for i, c in enumerate(chunks):
            abs_start = base_offset + c["start_seconds"]
            print(
                f"[watch] analyzing chunk {i + 1}/{len(chunks)} (t={format_time(abs_start)})…",
                file=sys.stderr,
            )
            res = tl.analyze_file(c["path"], prompt, key, model=model,
                                  temperature=args.tl_temperature, max_tokens=args.tl_max_tokens)
            segments.append({"start": abs_start, "duration": c["duration"], "result": res})
    else:
        res = tl.analyze_file(analysis_path, prompt, key, model=model,
                              temperature=args.tl_temperature, max_tokens=args.tl_max_tokens)
        segments.append({"start": base_offset, "duration": analysis_duration, "result": res})

    _print_tl_report(source, info, meta, model, full_duration, focused,
                     effective_start, effective_duration, segments, work)
    return 0


def _print_tl_report(source, info, meta, model, full_duration, focused,
                     effective_start, effective_duration, segments, work) -> None:
    print()
    print("# watch: video report (TwelveLabs Pegasus)")
    print()
    print(f"- **Source:** {source}")
    if info.get("title"):
        print(f"- **Title:** {info['title']}")
    if info.get("uploader"):
        print(f"- **Uploader:** {info['uploader']}")
    print(f"- **Duration:** {format_time(full_duration)} ({full_duration:.1f}s)")
    if focused:
        print(
            f"- **Focus range:** {format_time(effective_start)} → "
            f"{format_time(effective_start + effective_duration)} ({effective_duration:.1f}s)"
        )
    if meta.get("width") and meta.get("height"):
        print(f"- **Resolution:** {meta['width']}x{meta['height']} ({meta.get('codec') or 'unknown codec'})")
    print(f"- **Parser:** TwelveLabs Pegasus ({model}) — on-the-fly analysis, no frames, no image tokens")
    print(f"- **Segments analyzed:** {len(segments)}")
    task_ids = ", ".join(str(s["result"].get("task_id") or "?") for s in segments)
    print(f"- **Task IDs:** {task_ids}")

    print()
    print(
        "> This report is **text** that Pegasus generated from the video's pixels and audio (ASR). "
        "There are no frame images to Read — answer the user directly from the analysis below."
    )

    if len(segments) == 1:
        print()
        if focused:
            print(
                f"_Timestamps below are relative to the clip start "
                f"({format_time(effective_start)} in the source video)._"
            )
        print()
        print(segments[0]["result"]["text"])
    else:
        print()
        print(
            f"_The video was split into {len(segments)} segments. Within each segment, timestamps are "
            "relative to that segment's start time (shown in the heading); add the heading's start to "
            "get the absolute time in the source video._"
        )
        for i, s in enumerate(segments):
            a = s["start"]
            b = s["start"] + (s["duration"] or 0.0)
            print()
            print(f"## Segment {i + 1} — {format_time(a)} → {format_time(b)}")
            print()
            print(s["result"]["text"])

    print()
    print("---")
    print(f"_Work dir: `{work}` — delete when done._")


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
        "--provider",
        choices=["frames", "twelvelabs"],
        default="frames",
        help="Parser backend. 'frames' (default): extract frames + transcript for Claude to Read. "
             "'twelvelabs': Pegasus on-the-fly analysis — returns a timestamped transcript + visual "
             "walkthrough as text (no image tokens, no context ceiling, no Whisper key needed).",
    )
    ap.add_argument(
        "--tl-model",
        choices=["pegasus1.5", "pegasus1.2"],
        default="pegasus1.5",
        help="TwelveLabs model for --provider twelvelabs (default pegasus1.5).",
    )
    ap.add_argument(
        "--tl-prompt",
        type=str,
        default=None,
        help="Custom prompt/question for Pegasus. Default: full timestamped transcript + visual walkthrough.",
    )
    ap.add_argument(
        "--tl-max-tokens",
        type=int,
        default=16384,
        help="Max output tokens per Pegasus analysis (default 16384; clamped to the model's range).",
    )
    ap.add_argument(
        "--tl-temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for Pegasus analysis, 0-1 (default 0.2).",
    )
    ap.add_argument(
        "--chunk-minutes",
        type=float,
        default=30.0,
        help="For --provider twelvelabs: split videos longer than this into chunks (default 30).",
    )
    args = ap.parse_args()

    max_frames = min(args.max_frames, 100)

    if args.out_dir:
        work = Path(args.out_dir).expanduser().resolve()
    else:
        work = Path(tempfile.mkdtemp(prefix="watch-"))
    work.mkdir(parents=True, exist_ok=True)
    print(f"[watch] working dir: {work}", file=sys.stderr)

    print(
        "[watch] downloading via yt-dlp…" if is_url(args.source) else "[watch] using local file…",
        file=sys.stderr,
    )
    dl = download(args.source, work / "download")
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

    if args.provider == "twelvelabs":
        return run_twelvelabs(
            args, work,
            source=args.source,
            video_path=video_path,
            info=dl.get("info") or {},
            meta=meta,
            full_duration=full_duration,
            effective_start=effective_start,
            effective_duration=effective_duration,
            focused=focused,
        )

    if focused:
        fps, target = auto_fps_focus(effective_duration, max_frames=max_frames)
    else:
        fps, target = auto_fps(effective_duration, max_frames=max_frames)
    if args.fps is not None:
        fps = min(args.fps, MAX_FPS)
        target = max(1, int(round(fps * effective_duration)))

    scope = (
        f"{format_time(effective_start)}-{format_time(effective_end)} ({effective_duration:.1f}s)"
        if focused else f"full {effective_duration:.1f}s"
    )
    print(f"[watch] extracting ~{target} frames at {fps:.3f} fps over {scope}…", file=sys.stderr)

    frames = extract(
        video_path,
        work / "frames",
        fps=fps,
        resolution=args.resolution,
        max_frames=max_frames,
        start_seconds=start_sec,
        end_seconds=end_sec,
    )

    transcript_segments: list[dict] = []
    transcript_text: str | None = None
    transcript_source: str | None = None
    if dl.get("subtitle_path"):
        try:
            all_segments = parse_vtt(dl["subtitle_path"])
            transcript_segments = filter_range(all_segments, start_sec, end_sec) if focused else all_segments
            transcript_text = format_transcript(transcript_segments)
            transcript_source = "captions"
        except Exception as exc:
            print(f"[watch] subtitle parse failed: {exc}", file=sys.stderr)

    if not transcript_segments and not args.no_whisper:
        backend, api_key = load_api_key(args.whisper)
        if backend and api_key:
            try:
                all_segments, used_backend = transcribe_video(
                    video_path,
                    work / "audio.mp3",
                    backend=backend,
                    api_key=api_key,
                )
                transcript_segments = filter_range(all_segments, start_sec, end_sec) if focused else all_segments
                transcript_text = format_transcript(transcript_segments)
                transcript_source = f"whisper ({used_backend})"
            except SystemExit as exc:
                print(f"[watch] whisper fallback failed: {exc}", file=sys.stderr)
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
    print(f"- **Frames:** {len(frames)} @ {fps:.3f} fps, {mode} mode (budget {target}, max {max_frames})")
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
    print(f"Frames live at: `{work / 'frames'}`")
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
        print()
        print("```")
        print(transcript_text)
        print("```")
    elif focused and dl.get("subtitle_path"):
        print(f"_No transcript lines fell inside {format_time(effective_start)} → {format_time(effective_end)}._")
    else:
        setup_py = SCRIPT_DIR / "setup.py"
        print(
            "_No transcript available — proceed with frames only. "
            "Captions were missing and the Whisper fallback was unavailable "
            "(no API key set, or `--no-whisper` was used). "
            f"Run `python3 {setup_py}` to enable Whisper, then re-run._"
        )

    print()
    print("---")
    print(f"_Work dir: `{work}` — delete when done._")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
