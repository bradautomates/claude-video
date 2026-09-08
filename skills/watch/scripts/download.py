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


def _pick_subtitle(out_dir: Path) -> Path | None:
    candidates = sorted(out_dir.glob("video*.vtt"))
    if not candidates:
        return None
    preferred = [
        c for c in candidates
        if any(marker in c.name for marker in (".en.", ".en-US.", ".en-GB.", ".en-orig."))
    ]
    return preferred[0] if preferred else candidates[0]


def _native_lang(out_dir: Path) -> str | None:
    """The video's own language, from the info JSON yt-dlp already wrote.

    Returns a bare code like "de", or None when the field is absent or already
    English. yt-dlp reports either "language" or "original_language" depending
    on the extractor, so both are read.
    """
    info_path = out_dir / "video.info.json"
    if not info_path.exists():
        return None
    try:
        raw = json.loads(info_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    lang = raw.get("language") or raw.get("original_language")
    if not isinstance(lang, str) or not lang:
        return None
    lang = lang.split("-")[0].lower()
    return None if lang == "en" else lang


def _existing_sub(out_dir: Path, lang: str) -> Path | None:
    for candidate in sorted(out_dir.glob(f"video.{lang}*.vtt")):
        return candidate
    return None


def _fetch_native_subs(url: str, out_dir: Path, lang: str) -> None:
    """Second, cheap pass for captions in the video's own language.

    Runs whenever the video is not in English, not only when the English pass
    failed. Measured on 2026-09-07 on a German video: YouTube served an English
    track, and it was a machine translation of the German auto-captions -- both
    rendered "AGI" as "AGE", the same mis-hearing carried onward. Taking English
    there means two lossy steps instead of one.
    """
    cmd = [
        "yt-dlp",
        "--skip-download",
        "--write-subs",
        "--write-auto-subs",
        "--sub-langs", f"{lang}.*",
        "--sub-format", "vtt",
        "--convert-subs", "vtt",
        "--no-playlist",
        "--ignore-errors",
        "-o", str(out_dir / "video.%(ext)s"),
        "--",
        url,
    ]
    subprocess.run(cmd, stdout=sys.stderr, stderr=sys.stderr)


def _subtitle_with_fallback(url: str, out_dir: Path) -> Path | None:
    """The video's own language wins; English is the fallback, not the default.

    For an English video this changes nothing and costs nothing. For any other
    language it adds one cheap subtitle fetch and avoids a translated track.
    At most one extra attempt is ever made.
    """
    lang = _native_lang(out_dir)
    if lang is not None:
        native = _existing_sub(out_dir, lang)
        if native is None:
            print(f"[watch] video language is {lang}; fetching its own captions "
                  f"instead of a translated track", file=sys.stderr)
            _fetch_native_subs(url, out_dir, lang)
            native = _existing_sub(out_dir, lang)
        if native is not None:
            return native
        print(f"[watch] no {lang} captions available; falling back to English",
              file=sys.stderr)
    return _pick_subtitle(out_dir)


def _pick_video(out_dir: Path) -> Path | None:
    for ext in (".mp4", ".mkv", ".webm", ".mov", ".m4a", ".mp3", ".opus"):
        for candidate in out_dir.glob(f"video*{ext}"):
            return candidate
    for candidate in out_dir.glob("video.*"):
        if candidate.suffix.lower() in VIDEO_EXTS:
            return candidate
    return None


def fetch_captions(url: str, out_dir: Path) -> dict:
    """Fetch metadata and best available VTT captions without downloading video."""
    if shutil.which("yt-dlp") is None:
        raise SystemExit("yt-dlp is not installed. Install with: brew install yt-dlp")

    out_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(out_dir / "video.%(ext)s")
    cmd = [
        "yt-dlp",
        "--skip-download",
        "--write-info-json",
        "--write-subs",
        "--write-auto-subs",
        "--sub-langs", "en.*",
        "--sub-format", "vtt",
        "--convert-subs", "vtt",
        "--no-playlist",
        "--ignore-errors",
        "-o", output_template,
        "--",
        url,
    ]
    subprocess.run(cmd, stdout=sys.stderr, stderr=sys.stderr)
    subtitle = _subtitle_with_fallback(url, out_dir)
    info = _read_info(out_dir / "video.info.json", url)
    return {
        "video_path": None,
        "subtitle_path": str(subtitle) if subtitle else None,
        "info": info or {"url": url},
        "downloaded": False,
    }


def _read_info(info_path: Path, url: str) -> dict:
    info: dict = {}
    if info_path.exists():
        try:
            raw = json.loads(info_path.read_text(encoding="utf-8"))
            info = {
                "title": raw.get("title"),
                "uploader": raw.get("uploader") or raw.get("channel"),
                "duration": raw.get("duration"),
                "url": raw.get("webpage_url") or url,
            }
        except Exception as exc:
            print(f"[watch] info.json parse failed: {exc}", file=sys.stderr)
            info = {"url": url}
    return info


def download_url(
    url: str,
    out_dir: Path,
    audio_only: bool = False,
) -> dict:
    if shutil.which("yt-dlp") is None:
        raise SystemExit("yt-dlp is not installed. Install with: brew install yt-dlp")

    out_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(out_dir / "video.%(ext)s")

    fmt = "ba/bestaudio" if audio_only else "bv*[height<=720]+ba/b[height<=720]/bv+ba/b"
    cmd = [
        "yt-dlp",
        "-N", "8",
        "-f", fmt,
        "--merge-output-format", "mp4",
        "--write-info-json",
        "--write-subs",
        "--write-auto-subs",
        "--sub-langs", "en.*",
        "--sub-format", "vtt",
        "--convert-subs", "vtt",
        "--no-playlist",
        "--ignore-errors",
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

    subtitle = _subtitle_with_fallback(url, out_dir)
    info = _read_info(out_dir / "video.info.json", url)

    return {
        "video_path": str(video),
        "subtitle_path": str(subtitle) if subtitle else None,
        "info": info or {"url": url},
        "downloaded": True,
    }


def download(
    source: str,
    out_dir: Path,
    audio_only: bool = False,
) -> dict:
    if is_url(source):
        return download_url(source, out_dir, audio_only=audio_only)
    return resolve_local(source)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: download.py <url-or-path> <out-dir>", file=sys.stderr)
        raise SystemExit(2)
    result = download(sys.argv[1], Path(sys.argv[2]))
    print(json.dumps(result, indent=2))
