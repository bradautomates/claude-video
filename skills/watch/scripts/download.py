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


VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi", ".flv", ".wmv"}
DEFAULT_SUB_LANGS = "zh.*,en.*"
LANG_TOKEN_RE = re.compile(r"^[A-Za-z0-9*._-]+$")


def normalize_sub_langs(value: str | None = None) -> str:
    """Return a bounded yt-dlp language selector without accepting flags."""
    raw = DEFAULT_SUB_LANGS if value is None else value
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    if not tokens or any(
        token.startswith("-") or not LANG_TOKEN_RE.fullmatch(token)
        for token in tokens
    ):
        raise SystemExit(
            "--sub-langs must be a comma-separated list such as 'zh.*,en.*'"
        )
    return ",".join(tokens)


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


def _pick_subtitle(out_dir: Path, sub_langs: str | None = None) -> Path | None:
    candidates = sorted(out_dir.glob("video*.vtt"))
    if not candidates:
        return None
    for token in normalize_sub_langs(sub_langs).split(","):
        prefix = token.rstrip("*").rstrip(".").lower()
        preferred = [c for c in candidates if f".{prefix}" in c.name.lower()]
        if preferred:
            return preferred[0]
    return candidates[0]


def _pick_video(out_dir: Path) -> Path | None:
    for ext in (".mp4", ".mkv", ".webm", ".mov", ".m4a", ".mp3", ".opus"):
        for candidate in out_dir.glob(f"video*{ext}"):
            return candidate
    for candidate in out_dir.glob("video.*"):
        if candidate.suffix.lower() in VIDEO_EXTS:
            return candidate
    return None


def _yt_dlp_runtime_args() -> list[str]:
    """Enable yt-dlp's Node runtime when one is available.

    Recent YouTube extraction can require yt-dlp's external JavaScript
    components. Passing the explicit runtime lets a current yt-dlp use Node
    without requiring callers to know its absolute path.
    """
    node = shutil.which("node")
    return ["--js-runtimes", f"node:{node}"] if node else []


def _youtube_impersonation_args(url: str) -> list[str]:
    """Use a Chrome client for YouTube only when yt-dlp reports one is available."""
    host = (urlparse(url).hostname or "").lower()
    if not (host == "youtu.be" or host.endswith(".youtube.com")):
        return []
    try:
        result = subprocess.run(
            ["yt-dlp", "--list-impersonate-targets"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return []
    if result.returncode == 0 and re.search(r"^Chrome[\s-]", result.stdout, re.MULTILINE):
        return ["--impersonate", "chrome"]
    return []


def _is_html_response(path: Path) -> bool:
    """Return whether a purported media file is actually an HTML error page."""
    try:
        with path.open("rb") as stream:
            prefix = stream.read(512).lstrip().lower()
    except OSError:
        return False
    return prefix.startswith((b"<!doctype html", b"<html"))


def fetch_captions(url: str, out_dir: Path, sub_langs: str | None = None) -> dict:
    """Fetch metadata and best available VTT captions without downloading video."""
    if shutil.which("yt-dlp") is None:
        raise SystemExit("yt-dlp is not installed. Install with: brew install yt-dlp")

    out_dir.mkdir(parents=True, exist_ok=True)
    sub_langs = normalize_sub_langs(sub_langs)
    output_template = str(out_dir / "video.%(ext)s")
    cmd = [
        "yt-dlp",
        *_yt_dlp_runtime_args(),
        *_youtube_impersonation_args(url),
        "--skip-download",
        "--write-info-json",
        "--write-subs",
        "--write-auto-subs",
        "--sub-langs", sub_langs,
        "--sub-format", "vtt",
        "--convert-subs", "vtt",
        "--no-playlist",
        "--ignore-errors",
        "-o", output_template,
        "--",
        url,
    ]
    subprocess.run(cmd, stdout=sys.stderr, stderr=sys.stderr)
    subtitle = _pick_subtitle(out_dir, sub_langs)
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
    sub_langs: str | None = None,
) -> dict:
    if shutil.which("yt-dlp") is None:
        raise SystemExit("yt-dlp is not installed. Install with: brew install yt-dlp")

    out_dir.mkdir(parents=True, exist_ok=True)
    sub_langs = normalize_sub_langs(sub_langs)
    output_template = str(out_dir / "video.%(ext)s")

    fmt = "ba/bestaudio" if audio_only else "bv*[height<=720]+ba/b[height<=720]/bv+ba/b"
    cmd = [
        "yt-dlp",
        *_yt_dlp_runtime_args(),
        *_youtube_impersonation_args(url),
        "-N", "8",
        "-f", fmt,
        "--merge-output-format", "mp4",
        "--write-info-json",
        "--write-subs",
        "--write-auto-subs",
        "--sub-langs", sub_langs,
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
    if _is_html_response(video):
        raise SystemExit(
            "yt-dlp saved an HTML error page instead of media; visual extraction "
            "cannot continue. Update yt-dlp with its YouTube EJS dependencies and "
            "a Node runtime, then retry. If it persists, the source or network is "
            "blocking public media delivery."
        )

    subtitle = _pick_subtitle(out_dir, sub_langs)
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
    sub_langs: str | None = None,
) -> dict:
    if is_url(source):
        return download_url(source, out_dir, audio_only=audio_only, sub_langs=sub_langs)
    return resolve_local(source)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: download.py <url-or-path> <out-dir>", file=sys.stderr)
        raise SystemExit(2)
    result = download(sys.argv[1], Path(sys.argv[2]))
    print(json.dumps(result, indent=2))
