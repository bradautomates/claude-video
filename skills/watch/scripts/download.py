#!/usr/bin/env python3
"""Download a video via yt-dlp, or resolve a local file path.

Also fetches subtitles (manual first, then auto-generated) in VTT format so
transcribe.py can parse them without needing Whisper.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse


VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi", ".flv", ".wmv"}

# Query params that identify a position or a referrer, not a video. Dropping them
# keeps ?v=X and ?v=X&t=3s pointing at the same cache entry.
VOLATILE_PARAMS = {"t", "start", "end", "time_continue", "feature", "si", "pp",
                   "list", "index", "ab_channel", "app", "themeRefresh"}

# ffprobe duration vs. the source's own metadata. Real files land within a second
# or two; a wrong file is off by minutes.
DURATION_TOLERANCE_SECONDS = 8.0
DURATION_TOLERANCE_RATIO = 0.03


def is_url(source: str) -> bool:
    if source.startswith("-"):
        return False
    parsed = urlparse(source)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def _youtube_id(parsed) -> str | None:
    host = parsed.netloc.lower().removeprefix("www.")
    if host in ("youtu.be",):
        vid = parsed.path.strip("/").split("/")[0]
        return vid or None
    if host not in ("youtube.com", "m.youtube.com", "music.youtube.com"):
        return None
    if parsed.path == "/watch":
        vals = parse_qs(parsed.query).get("v")
        return vals[0] if vals else None
    m = re.match(r"^/(?:shorts|embed|live|v)/([^/]+)", parsed.path)
    return m.group(1) if m else None


def canonical_source(url: str) -> str:
    """Normalise a URL so the same video maps to one cache entry.

    Only used for cache identity — never for fetching. Unknown hosts keep every
    meaningful query param, so a false merge is not possible; the worst case is a
    redundant download.
    """
    parsed = urlparse(url)
    vid = _youtube_id(parsed)
    if vid:
        return f"youtube:{vid}"
    host = parsed.netloc.lower().removeprefix("www.")
    params = sorted(
        (k, v)
        for k, vals in parse_qs(parsed.query).items()
        if k not in VOLATILE_PARAMS
        for v in vals
    )
    query = "&".join(f"{k}={v}" for k, v in params)
    return f"{host}{parsed.path.rstrip('/')}" + (f"?{query}" if query else "")


def source_key(url: str) -> str:
    """Short stable directory name for a source."""
    return hashlib.sha256(canonical_source(url).encode("utf-8")).hexdigest()[:12]


def _probe_duration(path: Path) -> float | None:
    """Actual duration of the media file, or None if it cannot be determined."""
    if shutil.which("ffprobe") is None:
        return None
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(path)],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            return None
        value = json.loads(result.stdout or "{}").get("format", {}).get("duration")
        return float(value) if value else None
    except Exception:
        return None


def _duration_mismatch(out_dir: Path, video: Path) -> str | None:
    """Compare the media file against the duration the source reported for it.

    This is the check that catches a leftover video from a *different* URL:
    yt-dlp rewrites info.json and the subtitles even when it skips the download
    ("has already been downloaded"), so metadata and transcript look perfectly
    correct while the media — and therefore every extracted frame — is the old
    video. Duration is the one field where the lie shows.
    """
    info_path = out_dir / "video.info.json"
    if not info_path.exists():
        return None
    try:
        expected = json.loads(info_path.read_text(encoding="utf-8")).get("duration")
    except Exception:
        return None
    if not expected:
        return None
    actual = _probe_duration(video)
    if actual is None:
        return None
    tolerance = max(DURATION_TOLERANCE_SECONDS, float(expected) * DURATION_TOLERANCE_RATIO)
    if abs(actual - float(expected)) <= tolerance:
        return None
    return (
        f"{video.name} is {actual:.0f}s long but the source reports {float(expected):.0f}s"
    )


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


def _cache_dir(out_dir: Path, url: str, fresh: bool = False) -> Path:
    """One cache directory per source, so two URLs can never share a video.mp4.

    Re-running the same URL still lands in the same directory — that is what
    makes --out-dir a cache. A *different* URL gets a different directory, which
    is what stops it inheriting the previous run's media.
    """
    cache = out_dir / source_key(url)
    if fresh and cache.exists():
        print(f"[watch] --fresh: discarding cached download in {cache}", file=sys.stderr)
        shutil.rmtree(cache, ignore_errors=True)
    return cache


def fetch_captions(url: str, out_dir: Path, fresh: bool = False) -> dict:
    """Fetch metadata and best available VTT captions without downloading video."""
    if shutil.which("yt-dlp") is None:
        raise SystemExit("yt-dlp is not installed. Install with: brew install yt-dlp")

    out_dir = _cache_dir(out_dir, url, fresh=fresh)
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
    fresh: bool = False,
) -> dict:
    if shutil.which("yt-dlp") is None:
        raise SystemExit("yt-dlp is not installed. Install with: brew install yt-dlp")

    out_dir = _cache_dir(out_dir, url, fresh=fresh)
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

    if video is not None:
        problem = _duration_mismatch(out_dir, video)
        if problem:
            # Whatever is cached is not what this URL points at. Wipe and retry
            # once from scratch rather than handing back frames from another video.
            print(
                f"[watch] cached media does not match this source ({problem}) — "
                "discarding and re-downloading",
                file=sys.stderr,
            )
            shutil.rmtree(out_dir, ignore_errors=True)
            out_dir.mkdir(parents=True, exist_ok=True)
            result = subprocess.run(cmd, stdout=sys.stderr, stderr=sys.stderr)
            video = _pick_video(out_dir)
            if video is not None and _duration_mismatch(out_dir, video):
                print(
                    "[watch] WARNING: duration still disagrees with the source metadata "
                    "after a clean re-download. The frames may not match the transcript — "
                    "verify one frame against the transcript before trusting them.",
                    file=sys.stderr,
                )

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
    fresh: bool = False,
) -> dict:
    if is_url(source):
        return download_url(source, out_dir, audio_only=audio_only, fresh=fresh)
    return resolve_local(source)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: download.py <url-or-path> <out-dir>", file=sys.stderr)
        raise SystemExit(2)
    result = download(sys.argv[1], Path(sys.argv[2]))
    print(json.dumps(result, indent=2))
