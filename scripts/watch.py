#!/usr/bin/env python3
"""/watch entry point: download video, extract frames, parse transcript.

Prints a markdown report to stdout listing frame paths + transcript. Claude
then Reads each frame path to see the video.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from download import download, is_url  # noqa: E402
from frames import MAX_FPS, auto_fps, auto_fps_focus, extract, format_time, get_metadata, parse_time  # noqa: E402
from safe_name import DEFAULT_UCID_PREFIX, build_save_dir_name, next_ucid  # noqa: E402
from transcribe import filter_range, format_transcript, parse_vtt  # noqa: E402
from whisper import load_api_key, transcribe_video  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="watch",
        description="Download a video, extract auto-scaled frames, and surface the transcript.",
    )
    ap.add_argument("source", help="Video URL or local file path")
    ap.add_argument("--max-frames", type=int, default=80, help="Cap on frame count (default 80, hard max 100)")
    ap.add_argument("--resolution", type=int, default=512, help="Frame width in pixels (default 512)")
    ap.add_argument(
        "--hires-resolution", type=int, default=1600,
        help="Hi-res (publication-grade) frame width in pixels (default 1600, never upscales beyond source)",
    )
    ap.add_argument("--fps", type=float, default=None, help="Override auto-fps")
    ap.add_argument("--start", type=str, default=None, help="Range start (SS, MM:SS, or HH:MM:SS)")
    ap.add_argument("--end", type=str, default=None, help="Range end (SS, MM:SS, or HH:MM:SS)")
    ap.add_argument("--out-dir", type=str, default=None, help="Working directory (default: tmp)")
    ap.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help=(
            "Base directory under which a UCID-named subfolder "
            "'<PREFIX>-NNNN YYYY-MM-DD ShortTitle' (sanitised) is created. "
            "The UCID auto-increments by scanning existing folders under "
            "--save-dir. Mutually exclusive with --out-dir."
        ),
    )
    ap.add_argument(
        "--ucid-prefix",
        type=str,
        default=DEFAULT_UCID_PREFIX,
        help=(
            f"Prefix for the auto-allocated UCID (default: {DEFAULT_UCID_PREFIX}). "
            "Only used with --save-dir."
        ),
    )
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
        "--macp-register",
        action="store_true",
        default=False,
        help=(
            "Request MACP cloud registration after saved output and article assets are complete. "
            "Has no effect without --save-dir. Registration is performed by the Skill after "
            "Step 4.5 completes — not by this script directly."
        ),
    )
    ap.add_argument(
        "--macp-dry-run",
        action="store_true",
        default=False,
        help="Validate/build MACP payload only; no network calls, uploads, or commits.",
    )
    ap.add_argument("--macp-base-url", default=None, help="MACP base URL override")
    ap.add_argument("--macp-brand-id", default=None, help="MACP brand ID override")
    ap.add_argument("--macp-created-by", default=None, help="MACP created-by user ID override")
    ap.add_argument("--macp-editorial-brief-id", default=None, help="MACP editorial brief ID override")
    args = ap.parse_args()

    max_frames = min(args.max_frames, 100)

    if args.save_dir and args.out_dir:
        raise SystemExit("--save-dir and --out-dir are mutually exclusive")

    # When --save-dir is set, we still download into a staging tmp dir
    # first because the final folder name is built from metadata we only
    # have after yt-dlp runs. The whole staging dir is then moved into
    # place (atomic on the same filesystem) before any further processing.
    save_dir = Path(args.save_dir).expanduser().resolve() if args.save_dir else None

    if args.out_dir:
        work = Path(args.out_dir).expanduser().resolve()
    elif save_dir:
        work = Path(tempfile.mkdtemp(prefix="watch-staging-"))
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

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        ucid = next_ucid(save_dir, prefix=args.ucid_prefix)
        folder = build_save_dir_name(dl.get("info"), ucid=ucid)
        target = save_dir / folder
        if target.exists():
            raise SystemExit(
                f"--save-dir target already exists: {target}\n"
                "Remove it first or pick a different --save-dir."
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        old_work = work
        shutil.move(str(work), str(target))
        work = target
        # Rewire any path that pointed inside the old staging dir.
        def _relocate(p: str | None) -> str | None:
            if not p:
                return p
            old = Path(p)
            try:
                rel = old.relative_to(old_work)
            except ValueError:
                return p
            return str(work / rel)
        dl["video_path"] = _relocate(dl["video_path"])
        dl["subtitle_path"] = _relocate(dl.get("subtitle_path"))
        video_path = dl["video_path"]
        print(f"[watch] save-dir: {work}", file=sys.stderr)
        if args.macp_register:
            dry_label = " (dry-run)" if args.macp_dry_run else ""
            print(f"[watch] macp-register intent recorded{dry_label}", file=sys.stderr)

        # Sentinel for Step 4.5 (humanized business article + docx + pdf render).
        # The script only runs Steps 0-4; Step 4.5 lives in the /watch Skill and
        # is silently skipped when the script is invoked directly via Bash.
        # The Skill (or any post-processor) must delete this file once
        # business-article.{md,docx,pdf} have all been produced.
        sentinel = work / "business-article.REQUIRED"
        sentinel.write_text(
            "Step 4.5 of the /watch pipeline (humanized business article + docx + pdf)\n"
            "has not been completed for this UCID folder.\n"
            "\n"
            "Authoring this article is part of the /watch Skill (SKILL.md, Step 4.5),\n"
            "not the watch.py script. If you reached this folder via a batch that\n"
            "called watch.py directly through Bash, you bypassed Step 4.5 — re-run\n"
            "the Skill against the source URL, or author the article manually per\n"
            "MOR-950 and render via markdown-to-apple-deliverable.\n"
            "\n"
            "Delete this file once business-article.{md,docx,pdf} are all present.\n",
            encoding="utf-8",
        )
        print(f"[watch] wrote sentinel: {sentinel.name}", file=sys.stderr)

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
        hires_resolution=args.hires_resolution,
        max_frames=max_frames,
        start_seconds=start_sec,
        end_seconds=end_sec,
        hires_dir=work / "hires",
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

    transcript_md_path: Path | None = None
    if transcript_text:
        transcript_md_path = work / "transcript.md"
        md_lines = ["# Transcript", ""]
        if info.get("title"):
            md_lines.append(f"**Title:** {info['title']}")
        if info.get("uploader"):
            md_lines.append(f"**Uploader:** {info['uploader']}")
        md_lines.append(f"**Duration:** {format_time(full_duration)}")
        md_lines.append(f"**Source:** {transcript_source or 'captions'}")
        if focused:
            md_lines.append(f"**Range:** {format_time(effective_start)} → {format_time(effective_end)}")
        md_lines.extend(["", "---", ""])
        transcript_md_path.write_text("\n".join(md_lines) + "\n" + transcript_text + "\n", encoding="utf-8")

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
    print(
        f"- **Frame size:** {args.resolution}px (Claude reads) · "
        f"up to {args.hires_resolution}px (hi-res, for publication; capped at source width)"
    )
    if transcript_segments:
        in_range = " in range" if focused else ""
        print(
            f"- **Transcript:** {len(transcript_segments)} segments{in_range} "
            f"(via {transcript_source or 'captions'})"
        )
        if transcript_md_path:
            print(f"- **Transcript file:** `{transcript_md_path}`")
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
    print(f"Lo-res frames live at: `{work / 'frames'}`  ·  Hi-res copies at: `{work / 'hires'}`")
    print()
    print(
        "**Read each lo-res frame path below with the Read tool to view the image.** "
        "Frames are in chronological order; `t=MM:SS` is the absolute timestamp in the source video. "
        "The `hi-res` path is the same moment at publication resolution — cite it instead of "
        "the lo-res path whenever the user wants frames for the newsletter, blog, email, or any "
        "embedded use outside this chat. Hi-res files are inside the working directory and will "
        "be deleted at cleanup; copy them to a persistent location before that step."
    )
    print()
    for frame in frames:
        print(
            f"- `{frame['path']}` · hi-res: `{frame['hires_path']}` "
            f"(t={format_time(frame['timestamp_seconds'])})"
        )

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
    if args.macp_register and save_dir:
        dry_label = " --dry-run" if args.macp_dry_run else ""
        extra = ""
        if args.macp_base_url:
            extra += f" --base-url {args.macp_base_url}"
        if args.macp_brand_id:
            extra += f" --brand-id {args.macp_brand_id}"
        if args.macp_created_by:
            extra += f" --created-by {args.macp_created_by}"
        if args.macp_editorial_brief_id:
            extra += f" --editorial-brief-id {args.macp_editorial_brief_id}"
        print(
            f"\n_MACP registration requested. After Step 4.5 completes, the Skill must run:_\n"
            f"`python3 scripts/macp_adapter.py register --folder \"{work}\"{dry_label}{extra}`"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
