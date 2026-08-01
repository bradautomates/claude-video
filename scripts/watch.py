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
from media_analysis import (  # noqa: E402
    ANALYSIS_VERSION,
    analyze_media,
    format_precise,
    should_analyze,
)
from sound_semantics import analyze_sound_semantics  # noqa: E402
from transcribe import filter_range, format_transcript, parse_vtt  # noqa: E402
from whisper import load_api_key, transcribe_video  # noqa: E402


# Persistent cache root, shared with whisper.py's chunk cache (~/.cache/watch/
# chunks). Downloads land in videos/<url-hash>/, extracted frames in
# frames/<params-hash>/. This is what makes multi-pass deep analysis cheap:
# pass 2's focused re-runs skip the download and often the extraction too.
CACHE_ROOT = Path.home() / ".cache" / "watch"
FRAME_CACHE_VERSION = 2
DEFAULT_MAX_FRAMES = 100


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


def _media_analysis_cache_key(
    video_path: str,
    resolution: int,
    start: float,
    end: float,
) -> str:
    """Hash the media identity and range used by the cinematic evidence pass."""
    p = Path(video_path).resolve()
    try:
        st = p.stat()
    except OSError:
        return ""
    ident = (
        f"v{ANALYSIS_VERSION}|{p}|{st.st_size}|{st.st_mtime_ns}|"
        f"{resolution}|{start:.3f}|{end:.3f}"
    )
    return hashlib.sha256(ident.encode()).hexdigest()[:32]


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="watch",
        description=(
            "Analyze video or audio using scene-aware frames, source chapters, "
            "captions, and Whisper fallback."
        ),
    )
    ap.add_argument("source", help="Video/audio URL or local file path")
    ap.add_argument(
        "--max-frames",
        type=int,
        default=DEFAULT_MAX_FRAMES,
        help=f"Cap on frame count (default and hard max {DEFAULT_MAX_FRAMES})",
    )
    ap.add_argument("--resolution", type=int, default=512, help="Frame width in pixels (default 512)")
    ap.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Force uniform sampling at this rate (hard max 2 fps)",
    )
    ap.add_argument("--start", type=str, default=None, help="Range start (SS, MM:SS, or HH:MM:SS)")
    ap.add_argument("--end", type=str, default=None, help="Range end (SS, MM:SS, or HH:MM:SS)")
    ap.add_argument("--out-dir", type=str, default=None, help="Working directory (default: tmp)")
    ap.add_argument(
        "--no-whisper",
        action="store_true",
        help=(
            "Disable Whisper fallback. Caption-less sources have no speech text; "
            "short-form visual and non-speech sound evidence still works."
        ),
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
        "--analysis",
        choices=["auto", "standard", "cinematic"],
        default="auto",
        help=(
            "Temporal and non-speech audio analysis. auto enables cinematic evidence "
            "for clips/focused ranges up to 60s; standard disables it; cinematic forces it."
        ),
    )
    ap.add_argument(
        "--sound-semantics",
        choices=["auto", "api", "off"],
        default="auto",
        help=(
            "Learned sound-event and story-function analysis for ranges up to 60s. "
            "auto runs only when SOUND_SEMANTICS_PROVIDER is configured; api explicitly "
            "enables an audio upload; off disables it."
        ),
    )
    ap.add_argument(
        "--sound-provider",
        choices=["auto", "openai", "gemini"],
        default="auto",
        help="Semantic audio provider (default: configured provider, or available key in api mode).",
    )
    ap.add_argument(
        "--sound-vocab",
        type=str,
        default=None,
        help=(
            "Comma-separated candidate sound concepts or domain terms. These are hints, "
            "not forced labels."
        ),
    )
    ap.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass the persistent cache (~/.cache/watch) — download and extract into the temp work dir.",
    )
    ap.add_argument(
        "--vocab",
        type=str,
        default=None,
        help="Comma-separated proper nouns/terms expected in the audio (product names, "
        "people, tools). Passed as the Whisper prompt to bias spelling.",
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
    has_video = bool(meta.get("width"))
    if not has_video and not meta.get("has_audio"):
        raise SystemExit(f"{args.source} has neither a video nor an audio stream")

    start_sec = parse_time(args.start)
    end_sec = parse_time(args.end)

    if start_sec is not None and start_sec < 0:
        raise SystemExit("--start must be non-negative")
    if end_sec is not None and end_sec <= 0:
        raise SystemExit("--end must be positive")
    if end_sec is not None and start_sec is not None and end_sec <= start_sec:
        raise SystemExit("--end must be greater than --start")
    if full_duration > 0 and start_sec is not None and start_sec >= full_duration:
        raise SystemExit(f"--start {start_sec:.1f}s is past end of video ({full_duration:.1f}s)")

    effective_start = start_sec if start_sec is not None else 0.0
    effective_end = (
        min(end_sec, full_duration)
        if end_sec is not None and full_duration > 0
        else end_sec if end_sec is not None else full_duration
    )
    effective_duration = max(0.0, effective_end - effective_start)
    focused = start_sec is not None or end_sec is not None

    frames: list[dict] = []
    sampling_mode = "none (audio-only source)"
    frames_dir: Path | None = None
    target = 0
    if has_video:
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
    else:
        print(
            "[watch] audio-only source — skipping frames; analyzing sound + transcript",
            file=sys.stderr,
        )

    media_analysis: dict | None = None
    cinematic = should_analyze(args.analysis, effective_duration)
    if cinematic:
        analysis_key = _media_analysis_cache_key(
            video_path,
            args.resolution,
            effective_start,
            effective_end,
        )
        analysis_dir = (
            CACHE_ROOT / "analysis" / analysis_key
            if use_cache and analysis_key
            else work / "analysis"
        )
        print(
            "[watch] creating cinematic motion + sound evidence…",
            file=sys.stderr,
        )
        media_analysis = analyze_media(
            video_path,
            analysis_dir,
            has_video=has_video,
            has_audio=bool(meta.get("has_audio")),
            start=effective_start,
            end=effective_end,
            resolution=args.resolution,
        )
        if media_analysis.get("cache_hit"):
            print(f"[watch] cinematic analysis cache hit: {analysis_dir}", file=sys.stderr)

    transcript_segments: list[dict] = []
    transcript_text: str | None = None
    transcript_source: str | None = None
    transcript_failure: str | None = None
    if dl.get("subtitle_path"):
        try:
            subtitle_kind = dl.get("subtitle_kind")
            rolling = True if subtitle_kind == "auto" else False if subtitle_kind == "manual" else None
            all_segments = parse_vtt(dl["subtitle_path"], rolling=rolling)
            transcript_segments = filter_range(all_segments, start_sec, end_sec) if focused else all_segments
            transcript_text = format_transcript(transcript_segments)
            sub_detail = ", ".join(
                x for x in (dl.get("subtitle_kind"), dl.get("subtitle_lang")) if x
            )
            transcript_source = f"captions ({sub_detail})" if sub_detail else "captions"
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
                        vocab=args.vocab,
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

    semantic_analysis: dict | None = None
    if bool(meta.get("has_audio")) and args.sound_semantics != "off":
        semantic_analysis = analyze_sound_semantics(
            video_path,
            work / "semantic-audio",
            CACHE_ROOT / "semantics",
            effective_start,
            effective_end,
            mode=args.sound_semantics,
            provider=args.sound_provider,
            audio_analysis=media_analysis.get("audio") if media_analysis else None,
            video_analysis=media_analysis.get("video") if media_analysis else None,
            transcript=transcript_text,
            vocab=args.sound_vocab,
            use_cache=use_cache,
        )
        if semantic_analysis.get("status") == "ok":
            cache_note = " (cache hit)" if semantic_analysis.get("cache_hit") else ""
            print(
                f"[watch] semantic sound timeline ready via "
                f"{semantic_analysis.get('provider')}{cache_note}",
                file=sys.stderr,
            )
        elif args.sound_semantics == "api" or args.sound_provider != "auto":
            print(
                f"[watch] semantic sound analysis unavailable: "
                f"{semantic_analysis.get('reason', 'unknown error')}",
                file=sys.stderr,
            )
    elif args.sound_semantics == "api":
        semantic_analysis = {
            "status": "error",
            "reason": "source has no audio stream",
            "events": [],
        }
        print("[watch] semantic sound analysis unavailable: source has no audio stream", file=sys.stderr)

    info = dl.get("info") or {}
    chapters = info.get("chapters") or []

    print()
    print("# watch: media report")
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
    if has_video:
        print(f"- **Frames:** {len(frames)}, {sampling_mode}, {mode} mode (budget {target}, max {max_frames})")
        print(f"- **Frame size:** {args.resolution}px wide")
    else:
        report_kind = "sound evidence + transcript" if media_analysis else "transcript"
        print(f"- **Frames:** none — audio-only source ({report_kind} report)")
    if transcript_segments:
        in_range = " in range" if focused else ""
        print(
            f"- **Transcript:** {len(transcript_segments)} segments{in_range} "
            f"(via {transcript_source or 'captions'})"
        )
    else:
        print("- **Transcript:** none available")
    if chapters:
        print(f"- **Source chapters:** {len(chapters)}")
    if media_analysis:
        evidence = [
            name
            for name, section in (
                ("motion", media_analysis.get("video")),
                ("non-speech sound", media_analysis.get("audio")),
            )
            if section
        ]
        cache_label = "cached" if media_analysis.get("cache_hit") else "generated"
        print(f"- **Cinematic evidence:** {', '.join(evidence)} ({cache_label})")
    if semantic_analysis and semantic_analysis.get("status") == "ok":
        cache_label = "cached" if semantic_analysis.get("cache_hit") else "generated"
        print(
            f"- **Semantic sound:** {len(semantic_analysis.get('events') or [])} events "
            f"via {semantic_analysis.get('provider')} / {semantic_analysis.get('model')} "
            f"({cache_label})"
        )

    if has_video and not focused and full_duration > 600:
        mins = int(full_duration // 60)
        print()
        print(
            f"> **Warning:** This is a {mins}-minute video. Frame coverage is sparse at this length — "
            "accuracy degrades noticeably on anything over 10 minutes. For better results, "
            "re-run with `--start HH:MM:SS --end HH:MM:SS` to zoom into a specific section."
        )

    if chapters:
        print()
        print("## Source chapters")
        print()
        for chapter in chapters:
            chapter_start = float(chapter.get("start_time") or 0.0)
            raw_end = chapter.get("end_time")
            chapter_end = float(raw_end) if raw_end is not None else full_duration
            if focused and (chapter_end < effective_start or chapter_start > effective_end):
                continue
            print(
                f"- {format_time(chapter_start)}-{format_time(chapter_end)}: "
                f"{chapter.get('title') or 'Untitled chapter'}"
            )

    video_analysis = media_analysis.get("video") if media_analysis else None
    if video_analysis:
        print()
        print("## Cinematic timing and motion")
        print()
        print(
            "**Read every motion-strip path below.** Each strip is eight ordered frames, "
            "left to right. Overview strips preserve the whole sequence; adaptive detail "
            "strips zoom into 0.9-second high-change moments."
        )
        print()
        median_motion = video_analysis.get("median_motion")
        p90_motion = video_analysis.get("p90_motion")
        if median_motion is not None and p90_motion is not None:
            print(
                f"- **Pixel-change baseline:** median {median_motion:.2f}/255; "
                f"active-moment p90 {p90_motion:.2f}/255 at "
                f"{video_analysis.get('sample_fps', 10):g} samples/s"
            )
        change_peaks = video_analysis.get("change_peaks") or []
        if change_peaks:
            print("- **Strongest visual-event candidates:**")
            for point in change_peaks:
                kind = (point.get("kind") or "visual_change").replace("_", " ")
                coverage = point.get("changed_percent")
                confidence = point.get("confidence")
                details = [kind, f"score {point['score']:.2f}"]
                if coverage is not None:
                    details.append(f"~{coverage:.1f}% changed pixels")
                if confidence is not None:
                    details.append(f"confidence {confidence:.2f}")
                print(f"  - {format_precise(point['time'])}: {', '.join(details)}")
            print(
                "  - Classification uses change coverage plus temporal persistence; "
                "inspect the corresponding strip because a full-frame flash can still "
                "resemble a cut."
            )
        print("- **Overview motion strips:**")
        for strip in video_analysis.get("strips") or []:
            print(
                f"  - `{strip['path']}` "
                f"(t={format_precise(strip['start'])}-{format_precise(strip['end'])}, "
                f"{strip['samples']} frames left-to-right)"
            )
        zoom_strips = video_analysis.get("zoom_strips") or []
        if zoom_strips:
            print("- **Adaptive 0.9-second detail strips:**")
            for strip in zoom_strips:
                event_kind = (strip.get("event_kind") or "visual_change").replace("_", " ")
                confidence = strip.get("confidence")
                confidence_text = (
                    f", confidence {confidence:.2f}" if confidence is not None else ""
                )
                print(
                    f"  - `{strip['path']}` "
                    f"(t={format_precise(strip['start'])}-{format_precise(strip['end'])}, "
                    f"focus {format_precise(strip['focus_time'])}: {event_kind}"
                    f"{confidence_text}; {strip['samples']} frames left-to-right)"
                )
        for error in video_analysis.get("errors") or []:
            print(f"> **Partial motion analysis:** {error}")

    audio_analysis = media_analysis.get("audio") if media_analysis else None
    if audio_analysis:
        print()
        print("## Sound analysis (beyond speech)")
        print()
        print(
            "**Read the waveform and spectrogram images below.** Time runs left-to-right. "
            "The waveform shows amplitude and rhythm; the spectrogram shows frequency "
            "content from low to high. Use these with the timestamps and transcript."
        )
        print()
        if audio_analysis.get("waveform_path"):
            print(f"- **Waveform:** `{audio_analysis['waveform_path']}`")
        if audio_analysis.get("spectrogram_path"):
            print(f"- **Spectrogram:** `{audio_analysis['spectrogram_path']}`")
        median_rms = audio_analysis.get("median_rms_db")
        dynamic_range = audio_analysis.get("dynamic_range_db")
        if median_rms is not None:
            dynamics = (
                f"; central dynamic range {dynamic_range:.2f} dB"
                if dynamic_range is not None else ""
            )
            print(f"- **Sound level:** median {median_rms:.2f} dBFS{dynamics}")
        energy_peaks = audio_analysis.get("energy_peaks") or []
        if energy_peaks:
            print(
                "- **Loudness peaks:** "
                + ", ".join(
                    f"{format_precise(point['time'])} ({point['rms_db']:.2f} dBFS)"
                    for point in energy_peaks
                )
            )
        transients = audio_analysis.get("transient_peaks") or []
        if transients:
            print("- **Transient/change candidates:**")
            for point in transients:
                details = [point["label"]]
                if point.get("rms_db") is not None:
                    details.append(f"{point['rms_db']:.2f} dBFS")
                if point.get("centroid_hz") is not None:
                    details.append(f"centroid ~{point['centroid_hz']} Hz")
                print(f"  - {format_precise(point['time'])}: {', '.join(details)}")
        band_onsets = audio_analysis.get("band_onsets") or []
        if band_onsets:
            print("- **Multi-band onset candidates:**")
            for point in band_onsets:
                bands = "/".join(point.get("bands") or [])
                print(
                    f"  - {format_precise(point['time'])}: {point['label']}, "
                    f"+{point['rise_db']:.2f} dB novelty in {bands}, "
                    f"confidence {point['confidence']:.2f}"
                )
        silences = audio_analysis.get("silences") or []
        if silences:
            print(
                "- **Silence/near-silence:** "
                + ", ".join(
                    f"{format_precise(item['start'])}-{format_precise(item['end'])} "
                    f"({item['duration']:.2f}s)"
                    for item in silences
                )
            )
        else:
            print(
                f"- **Silence/near-silence:** none lasting {0.12:.2f}s or more "
                "below -42 dB"
            )
        print(
            "- **Interpretation limit:** event labels are signal-based candidates, not "
            "semantic recognition. Do not name an exact sound source unless the visuals, "
            "transcript, or direct audio inspection support it."
        )
        for error in audio_analysis.get("errors") or []:
            print(f"> **Partial sound analysis:** {error}")

    if semantic_analysis and semantic_analysis.get("status") == "ok":
        print()
        print("## Semantic sound timeline")
        print()
        print(
            "This learned layer listened to the selected audio and proposed sound-event "
            "labels and filmmaking functions. Times below are absolute source timestamps."
        )
        print()
        if semantic_analysis.get("summary"):
            print(f"- **Overall design:** {semantic_analysis['summary']}")
        if semantic_analysis.get("soundscape"):
            print(f"- **Soundscape:** {semantic_analysis['soundscape']}")
        music = semantic_analysis.get("music") or {}
        if music.get("present"):
            music_parts = [
                value for value in (
                    music.get("description"),
                    f"tempo: {music['tempo']}" if music.get("tempo") else "",
                    f"mood: {music['mood']}" if music.get("mood") else "",
                ) if value
            ]
            print(f"- **Music:** {'; '.join(music_parts) or 'present'}")
        print("- **Events:**")
        events = semantic_analysis.get("events") or []
        if not events:
            print("  - No distinct semantic events returned.")
        for event in events:
            confidence = float(event.get("confidence") or 0.0)
            print(
                f"  - {format_precise(event['start'])}-{format_precise(event['end'])}: "
                f"**{event['label']}** [{event.get('category', 'unknown')}, "
                f"confidence {confidence:.2f}, diegetic {event.get('diegetic', 'unclear')}]"
            )
            if event.get("description"):
                print(f"    - Audible evidence: {event['description']}")
            if event.get("story_function"):
                print(f"    - Story/edit function: {event['story_function']}")
            grounding = event.get("grounding") or {}
            grounding_parts = []
            if grounding.get("audio_change"):
                grounding_parts.append(
                    f"local audio-change candidate at "
                    f"{format_precise(grounding['audio_change']['time'])}"
                )
            if grounding.get("visual_change"):
                grounding_parts.append(
                    f"visual-change candidate at "
                    f"{format_precise(grounding['visual_change']['time'])}"
                )
            if grounding_parts:
                print(f"    - Timing support: {'; '.join(grounding_parts)}")
        caveats = semantic_analysis.get("caveats") or []
        if caveats:
            print(f"- **Model caveats:** {'; '.join(caveats)}")
        print(
            "- **Interpretation limit:** labels and story functions are model inferences, "
            "and confidence values are not calibrated probabilities. Verify ambiguous "
            "sources against frames, motion strips, waveform/spectrogram, and transcript."
        )
    elif semantic_analysis and semantic_analysis.get("status") == "error":
        print()
        print(
            f"> **Semantic sound analysis unavailable:** "
            f"{semantic_analysis.get('reason', 'unknown error')}"
        )

    if has_video:
        print()
        print("## Frames")
        print()
        print(f"Frames live at: `{frames_dir}`")
        print()
        print(
            "**Read each frame path below with the Read tool to view the image.** "
            "Frames are in chronological order; each `t=` value is the precise absolute timestamp in the source video."
        )
        print()
        for frame in frames:
            print(f"- `{frame['path']}` (t={format_precise(frame['timestamp_seconds'])})")

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
        if has_video:
            fallback = (
                "proceed with visual and non-speech sound evidence"
                if audio_analysis else
                "proceed with visual evidence"
            )
            print(
                f"_No transcript available — {fallback}. {reason}. "
                f"To configure Whisper, run `python3 {setup_py}`._"
            )
        elif audio_analysis:
            print(
                f"_No transcript available — proceed with the non-speech sound evidence. "
                f"{reason}. To configure Whisper, run `python3 {setup_py}`._"
            )
        else:
            print(
                f"_No transcript available, and this audio-only source has no visual frames. {reason}. "
                f"To configure Whisper, run `python3 {setup_py}`._"
            )

    print()
    print("---")
    if use_cache:
        print(
            f"_Work dir: `{work}` — delete when done. Video, frames, and cinematic evidence are cached under "
            f"`{CACHE_ROOT}` (do NOT delete) — re-runs and focused zooms on this video are fast._"
        )
    else:
        print(f"_Work dir: `{work}` — delete when done._")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
