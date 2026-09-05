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

# Alternate YouTube player clients, tried in order when the default `web`
# client is refused. YouTube gates `web` behind "Sign in to confirm you're not
# a bot" on datacenter IPs (Claude's cloud container, CI runners, VPSes), and
# also rate-limits them with HTTP 429. These clients hit different API surfaces
# that are not gated identically, so they frequently still return metadata and
# captions when `web` is refused. Ordered cheapest-and-most-permissive first.
_YT_CLIENT_FALLBACKS = ("mweb", "web_embedded", "tv", "android", "ios")


def _is_youtube(url: str) -> bool:
    host = (urlparse(url).netloc or "").lower()
    return any(h in host for h in ("youtube.com", "youtu.be", "youtube-nocookie.com"))


def _run_ytdlp(base_cmd: list[str], url: str, succeeded, label: str):
    """Run yt-dlp, retrying with alternate YouTube player clients on refusal.

    `base_cmd` must NOT include the trailing `-- <url>`; this adds it.
    `succeeded` is a zero-arg predicate checked after each attempt.
    Returns (CompletedProcess, client_used_or_None).
    """
    attempts: list[str | None] = [None]
    if _is_youtube(url):
        attempts += list(_YT_CLIENT_FALLBACKS)

    result = None
    for client in attempts:
        cmd = list(base_cmd)
        if client is not None:
            cmd += ["--extractor-args", f"youtube:player_client={client}"]
            print(
                f"[watch] {label}: default client refused, retrying with "
                f"player_client={client}\u2026",
                file=sys.stderr,
            )
        cmd += ["--", url]
        result = subprocess.run(cmd, stdout=sys.stderr, stderr=sys.stderr)
        if succeeded():
            if client is not None:
                print(f"[watch] {label}: succeeded via player_client={client}", file=sys.stderr)
            return result, client
    return result, None


def fetch_storyboard(url: str, out_dir: Path) -> Path | None:
    """Download YouTube's storyboard mosaic as a last-resort visual source.

    When media formats are bot-gated but the page still resolves, the
    storyboard (a grid of thumbnails spanning the whole video) is usually
    still served. It is low-resolution but gives real frame coverage instead
    of nothing. Returns the .mhtml path, or None.
    """
    if not _is_youtube(url):
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "storyboard.mhtml"
    for fmt in ("sb0", "sb1", "sb2"):
        base = [
            "yt-dlp",
            "--compat-options", "no-certifi",
            "--js-runtimes", "node",
            "-f", fmt,
            "--ignore-no-formats-error",
            "--write-info-json",
            "--no-playlist",
            "-o", str(out_dir / "storyboard.%(ext)s"),
        ]
        _run_ytdlp(base, url, lambda: target.exists(), f"storyboard[{fmt}]")
        if target.exists():
            print(f"[watch] storyboard captured via format {fmt}", file=sys.stderr)
            return target
    return None



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
    ]

    # Retry across player clients: a bot-gated `web` client yields neither
    # info.json nor subtitles, which previously failed silently and left the
    # caller with an empty result and no explanation.
    def _got_something() -> bool:
        return (out_dir / "video.info.json").exists() or _pick_subtitle(out_dir) is not None

    _run_ytdlp(cmd, url, _got_something, "captions")
    subtitle = _pick_subtitle(out_dir)
    info = _read_info(out_dir / "video.info.json", url)
    if not info or not info.get("title"):
        print(
            "[watch] warning: could not resolve metadata for this URL "
            "(bot-gate or rate limit on every player client)",
            file=sys.stderr,
        )
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
    ]

    # yt-dlp may exit non-zero if a subtitle variant fails (e.g. 429) even when
    # the video itself downloaded fine. Treat "video file present" as success.
    result, _client = _run_ytdlp(
        cmd, url, lambda: _pick_video(out_dir) is not None, "media"
    )
    video = _pick_video(out_dir)
    if video is None:
        rc = result.returncode if result is not None else -1
        raise SystemExit(
            f"yt-dlp did not produce a video file in {out_dir} (exit {rc}).\n"
            "Every player client was refused \u2014 this is usually YouTube's "
            "datacenter-IP bot gate, not a bad URL.\n"
            "Frames can still be recovered from the storyboard; captions and "
            "metadata may also still be available."
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
