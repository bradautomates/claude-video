#!/usr/bin/env python3
"""Download a video via yt-dlp, or resolve a local file path.

Also fetches subtitles (manual first, then auto-generated) in VTT format so
transcribe.py can parse them without needing Whisper.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

from runtime import configure_utf8_output


configure_utf8_output()


VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi", ".flv", ".wmv"}
FALLBACK_SUB_LANGS = (".*-orig", "en-orig", "en", "en-US", "en-GB")


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


def _subtitle_language(path: Path) -> str:
    name = path.name
    return name[len("video."):-len(".vtt")] if name.startswith("video.") else ""


def _pick_subtitle(
    out_dir: Path,
    preferred_languages: list[str] | None = None,
) -> Path | None:
    candidates = sorted(out_dir.glob("video*.vtt"))
    if not candidates:
        return None
    by_language = {_subtitle_language(candidate): candidate for candidate in candidates}
    for language in preferred_languages or []:
        if language in by_language:
            return by_language[language]
    originals = [
        candidate for candidate in candidates
        if _subtitle_language(candidate).endswith("-orig")
    ]
    if originals:
        return originals[0]
    for language in ("en-orig", "en", "en-US", "en-GB"):
        if language in by_language:
            return by_language[language]
    return candidates[0]


def _choose_caption_languages(info: dict) -> list[str]:
    """Choose one best caption track from yt-dlp metadata.

    Source-language manual captions win, followed by source auto-captions. A
    ``*-orig`` track is preferred when yt-dlp cannot declare the language. Only
    then do we fall back to English or another manual track.
    """
    manual = {str(lang) for lang in (info.get("subtitles") or {})}
    automatic = {str(lang) for lang in (info.get("automatic_captions") or {})}
    declared = info.get("language") or info.get("original_language")
    declared = declared.strip() if isinstance(declared, str) else ""
    bases = []
    for language in (declared, declared.split("-", 1)[0] if declared else ""):
        if language and language not in bases:
            bases.append(language)

    for language in bases:
        exact = sorted(
            item for item in manual
            if item == language or item.startswith(f"{language}-")
        )
        if exact:
            return [language if language in exact else exact[0]]

    for language in bases:
        for candidate in (f"{language}-orig", language):
            if candidate in automatic:
                return [candidate]

    originals = sorted(lang for lang in automatic if lang.endswith("-orig"))
    if originals:
        return [originals[0]]

    for language in ("en-orig", "en", "en-US", "en-GB"):
        if language in manual:
            return [language]
    if manual:
        return [sorted(manual)[0]]
    for language in ("en-orig", "en", "en-US", "en-GB"):
        if language in automatic:
            return [language]
    return []


def _probe_caption_languages(url: str) -> list[str]:
    """Read caption metadata without downloading media or subtitle tracks."""
    try:
        result = subprocess.run(
            [
                "yt-dlp", "--skip-download", "--dump-single-json",
                "--no-warnings", "--no-playlist", "--", url,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0 or not result.stdout.strip():
        return []
    try:
        return _choose_caption_languages(json.loads(result.stdout))
    except (json.JSONDecodeError, TypeError):
        return []


def _subtitle_selector(languages: list[str]) -> str:
    if not languages:
        return ",".join(FALLBACK_SUB_LANGS)
    return ",".join(f"^{re.escape(language)}$" for language in languages)


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
    caption_languages = _probe_caption_languages(url)
    cmd = [
        "yt-dlp",
        "--skip-download",
        "--write-info-json",
        "--write-subs",
        "--write-auto-subs",
        "--sub-langs", _subtitle_selector(caption_languages),
        "--sub-format", "vtt",
        "--convert-subs", "vtt",
        "--no-playlist",
        "--ignore-errors",
        "-o", output_template,
        "--",
        url,
    ]
    subprocess.run(cmd, stdout=sys.stderr, stderr=sys.stderr)
    subtitle = _pick_subtitle(out_dir, caption_languages)
    info = _read_info(out_dir / "video.info.json", url)
    return {
        "video_path": None,
        "subtitle_path": str(subtitle) if subtitle else None,
        "info": info or {"url": url},
        "caption_languages": caption_languages,
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
                "language": raw.get("language") or raw.get("original_language"),
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
    caption_languages: list[str] | None = None,
) -> dict:
    if shutil.which("yt-dlp") is None:
        raise SystemExit("yt-dlp is not installed. Install with: brew install yt-dlp")

    out_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(out_dir / "video.%(ext)s")
    if caption_languages is None:
        caption_languages = _probe_caption_languages(url)

    fmt = "ba/bestaudio" if audio_only else "bv*[height<=720]+ba/b[height<=720]/bv+ba/b"
    cmd = [
        "yt-dlp",
        "-N", "8",
        "-f", fmt,
        "--merge-output-format", "mp4",
        "--write-info-json",
        "--write-subs",
        "--write-auto-subs",
        "--sub-langs", _subtitle_selector(caption_languages),
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

    subtitle = _pick_subtitle(out_dir, caption_languages)
    info = _read_info(out_dir / "video.info.json", url)

    return {
        "video_path": str(video),
        "subtitle_path": str(subtitle) if subtitle else None,
        "info": info or {"url": url},
        "caption_languages": caption_languages,
        "downloaded": True,
    }


def download(
    source: str,
    out_dir: Path,
    audio_only: bool = False,
    caption_languages: list[str] | None = None,
) -> dict:
    if is_url(source):
        return download_url(
            source,
            out_dir,
            audio_only=audio_only,
            caption_languages=caption_languages,
        )
    return resolve_local(source)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: download.py <url-or-path> <out-dir>", file=sys.stderr)
        raise SystemExit(2)
    result = download(sys.argv[1], Path(sys.argv[2]))
    print(json.dumps(result, indent=2))
