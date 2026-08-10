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

# Files a failed attempt can leave behind that would fool the next attempt into thinking the media
# is already downloaded. Deliberately excludes .info.json and .vtt — those are worth keeping.
MEDIA_LEFTOVERS = VIDEO_EXTS | {".m4a", ".mp3", ".opus", ".part", ".ytdl"}

# YouTube forces SABR streaming on yt-dlp's default / web / tv / ios player clients: formats lose
# their URL, media 403s, and captions fail with "The page needs to be reloaded". The `android`
# player client bypasses it. Measured 2026-08-09 on this machine: android was the ONLY client that
# returned media (tv_embedded, ios, mweb, web_safari and the default all failed).
#
# Always try the default client FIRST — it gives better formats whenever SABR is not being forced —
# and fall back to android only when the attempt produced nothing. Never merge "default,android"
# into one run: the format picker re-selects the broken default formats and 403s again.
YOUTUBE_HOSTS = ("youtube.com", "youtu.be", "youtube-nocookie.com")
ANDROID_CLIENT = ["--extractor-args", "youtube:player_client=android"]


def is_url(source: str) -> bool:
    if source.startswith("-"):
        return False
    parsed = urlparse(source)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def is_youtube(url: str) -> bool:
    host = urlparse(url).netloc.lower().split(":")[0]
    if host.startswith("www."):
        host = host[4:]
    return any(host == h or host.endswith("." + h) for h in YOUTUBE_HOSTS)


def _clear_media(out_dir: Path) -> None:
    """Remove media + partial files so a retry re-downloads instead of resuming a broken attempt."""
    for p in out_dir.glob("video.*"):
        if p.suffix.lower() in MEDIA_LEFTOVERS:
            try:
                p.unlink()
            except OSError:
                pass


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


def _usable(p: Path) -> bool:
    """A 403'd or aborted attempt can leave a 0-byte stub; that is not a download."""
    try:
        return p.stat().st_size > 1024
    except OSError:
        return False


def _pick_video(out_dir: Path) -> Path | None:
    for ext in (".mp4", ".mkv", ".webm", ".mov", ".m4a", ".mp3", ".opus"):
        for candidate in out_dir.glob(f"video*{ext}"):
            if _usable(candidate):
                return candidate
    for candidate in out_dir.glob("video.*"):
        if candidate.suffix.lower() in VIDEO_EXTS and _usable(candidate):
            return candidate
    return None


def fetch_captions(url: str, out_dir: Path) -> dict:
    """Fetch metadata and best available VTT captions without downloading video."""
    if shutil.which("yt-dlp") is None:
        raise SystemExit("yt-dlp is not installed. Install with: brew install yt-dlp")

    out_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(out_dir / "video.%(ext)s")

    def run(extra: list[str]) -> None:
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
            *extra,
            "-o", output_template,
            "--",
            url,
        ]
        subprocess.run(cmd, stdout=sys.stderr, stderr=sys.stderr)

    run([])
    subtitle = _pick_subtitle(out_dir)
    if subtitle is None and is_youtube(url):
        print(
            "[watch] no captions from the default client — retrying with the android player client "
            "(YouTube SABR workaround)",
            file=sys.stderr,
        )
        run(ANDROID_CLIENT)
        subtitle = _pick_subtitle(out_dir)
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

    def run(extra: list[str]) -> subprocess.CompletedProcess:
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
            *extra,
            "-o", output_template,
            "--",
            url,
        ]
        # yt-dlp may exit non-zero if a subtitle variant fails (e.g. 429) even when
        # the video itself downloaded fine. Treat "video file present" as success.
        return subprocess.run(cmd, stdout=sys.stderr, stderr=sys.stderr)

    result = run([])
    video = _pick_video(out_dir)
    if video is None and is_youtube(url):
        print(
            "[watch] no media from the default client — retrying with the android player client "
            "(YouTube SABR workaround)",
            file=sys.stderr,
        )
        _clear_media(out_dir)
        result = run(ANDROID_CLIENT)
        video = _pick_video(out_dir)
    if video is None:
        raise SystemExit(
            f"yt-dlp did not produce a video file in {out_dir} (exit {result.returncode})"
        )

    subtitle = _pick_subtitle(out_dir)
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
