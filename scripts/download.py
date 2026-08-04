#!/usr/bin/env python3
"""Download a video via yt-dlp, or resolve a local file path.

Also fetches subtitles (manual first, then auto-generated) in VTT format so
transcribe.py can parse them without needing Whisper.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse


VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi", ".flv", ".wmv"}
CAPTION_FALLBACKS = ("en", "en-US", "en-GB", "en-orig", "es", "es-ES", "es-orig")


def is_url(source: str) -> bool:
    if source.startswith("-"):
        return False
    parsed = urlparse(source)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def resolve_local(path: str) -> dict:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise SystemExit(f"File not found: {p}")
    if p.suffix.lower() not in VIDEO_EXTS:
        print(
            f"[watch] warning: {p.suffix} is not a known video extension, proceeding anyway",
            file=sys.stderr,
        )
    return {
        "video_path": str(p),
        "subtitle_path": None,
        "info": {"title": p.name, "url": str(p)},
        "downloaded": False,
    }


def _pick_subtitle(out_dir: Path, preferred_languages: list[str] | None = None) -> Path | None:
    candidates = sorted(out_dir.glob("video*.vtt"))
    if not candidates:
        return None
    for language in preferred_languages or []:
        exact = out_dir / f"video.{language}.vtt"
        if exact in candidates:
            return exact
    return candidates[0]


def _pick_video(out_dir: Path) -> Path | None:
    for ext in (".mp4", ".mkv", ".webm", ".mov"):
        for candidate in out_dir.glob(f"video*{ext}"):
            return candidate
    for candidate in out_dir.glob("video.*"):
        if candidate.suffix.lower() in VIDEO_EXTS:
            return candidate
    return None


def _caption_languages(url: str) -> tuple[list[str], list[str]]:
    """Discover the source language before requesting captions.

    YouTube exposes automatic captions for many languages, but requesting a
    fixed language (previously English only) can miss the video's original
    transcript entirely. Keep the request narrow when metadata is available;
    use ``all`` only when discovery itself fails.
    """
    cmd = ["yt-dlp", "--dump-single-json", "--skip-download", "--no-playlist", "--no-warnings", "--", url]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(
            "[watch] caption-language discovery failed; falling back to all caption languages",
            file=sys.stderr,
        )
        return ["all"], []

    try:
        metadata = json.loads(result.stdout)
    except json.JSONDecodeError:
        print(
            "[watch] caption-language metadata was not valid JSON; falling back to all caption languages",
            file=sys.stderr,
        )
        return ["all"], []

    available = set((metadata.get("subtitles") or {}).keys())
    available.update((metadata.get("automatic_captions") or {}).keys())
    source_language = metadata.get("language") or ""

    preferred: list[str] = []

    def add(language: str) -> None:
        if language and language not in preferred:
            preferred.append(language)

    add(source_language)
    if source_language:
        base_language = source_language.split("-", 1)[0]
        add(f"{base_language}-orig")
        add(base_language)
    for language in CAPTION_FALLBACKS:
        add(language)

    selected = [language for language in preferred if language in available]
    if not selected:
        selected = ["all"]
        if available:
            print(
                "[watch] advertised caption languages did not match the source language; requesting all",
                file=sys.stderr,
            )
        else:
            print(
                "[watch] no caption tracks advertised by the source; allowing Whisper fallback",
                file=sys.stderr,
            )
    return selected, preferred


def download_url(url: str, out_dir: Path) -> dict:
    if shutil.which("yt-dlp") is None:
        raise SystemExit("yt-dlp is not installed. Install with: brew install yt-dlp")

    out_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(out_dir / "video.%(ext)s")
    subtitle_languages, preferred_languages = _caption_languages(url)

    cmd = [
        "yt-dlp",
        "-N", "8",
        "-f", "bv*[height<=720]+ba/b[height<=720]/bv+ba/b",
        "--merge-output-format", "mp4",
        "--write-info-json",
        "--write-subs",
        "--write-auto-subs",
        "--sub-langs", ",".join(subtitle_languages),
        "--sub-format", "vtt",
        "--convert-subs", "vtt",
        "--no-playlist",
        "--ignore-errors",
        "--retries", "10",
        "--fragment-retries", "10",
        "--retry-sleep", "http:exp=1:20",
        "--sleep-requests", "5",
        "--sleep-subtitles", "10",
        "-o", output_template,
        "--",
        url,
    ]

    # yt-dlp may exit non-zero if a subtitle variant fails (e.g. 429) even when
    # the video itself downloaded fine. Treat "video file present" as success.
    result = subprocess.run(cmd, stdout=sys.stderr, stderr=sys.stderr)
    video = _pick_video(out_dir)
    if video is None:
        raise SystemExit(
            f"yt-dlp did not produce a video file in {out_dir} (exit {result.returncode})"
        )

    subtitle = _pick_subtitle(out_dir, preferred_languages)
    info_path = out_dir / "video.info.json"
    info: dict = {}
    if info_path.exists():
        try:
            raw = json.loads(info_path.read_text(encoding="utf-8"))
            info = {
                "title": raw.get("title"),
                "uploader": raw.get("uploader") or raw.get("channel"),
                "duration": raw.get("duration"),
                "language": raw.get("language"),
                "url": raw.get("webpage_url") or url,
                "caption_languages": subtitle_languages,
            }
        except Exception as exc:
            print(f"[watch] info.json parse failed: {exc}", file=sys.stderr)
            info = {"url": url}

    return {
        "video_path": str(video),
        "subtitle_path": str(subtitle) if subtitle else None,
        "info": info or {"url": url},
        "downloaded": True,
    }


def download(source: str, out_dir: Path) -> dict:
    if is_url(source):
        return download_url(source, out_dir)
    return resolve_local(source)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: download.py <url-or-path> <out-dir>", file=sys.stderr)
        raise SystemExit(2)
    result = download(sys.argv[1], Path(sys.argv[2]))
    print(json.dumps(result, indent=2))
