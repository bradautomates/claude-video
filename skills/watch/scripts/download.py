#!/usr/bin/env python3
"""Download a video via yt-dlp, or resolve a local file path.

Also fetches subtitles (manual first, then auto-generated) in VTT format so
transcribe.py can parse them without needing Whisper.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse


VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi", ".flv", ".wmv"}



# Downloading is by far the slowest step — tens of seconds and tens of MB for a
# clip whose frames then take ~1s to extract. SKILL.md worked around that by
# asking the model to re-point a second run at the file left in the work dir,
# but the same file says to delete that dir when done, and nothing survives
# across sessions. Keying the download by URL makes the reuse automatic.
CACHE_LIMIT_BYTES = 2 * 1024 * 1024 * 1024


def cache_root() -> Path:
    base = os.environ.get("XDG_CACHE_HOME")
    return (Path(base) if base else Path.home() / ".cache") / "watch" / "downloads"


def cache_dir_for(url: str, audio_only: bool, root: Path | None = None) -> Path:
    """Directory holding this URL's download.

    audio_only is part of the key: a `transcript` run fetches audio alone, and
    reusing that for a later frame-extracting run would hand ffmpeg a file with
    no video stream.
    """
    payload = f"{url}\x00{'audio' if audio_only else 'video'}".encode()
    digest = hashlib.sha256(payload).hexdigest()[:16]
    return (root or cache_root()) / digest


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def prune_cache(limit: int = CACHE_LIMIT_BYTES, keep: Path | None = None,
                root: Path | None = None) -> int:
    """Evict least-recently-used entries until the cache fits in ``limit``.

    ``keep`` is never evicted — it is the entry the current run is about to
    use, which would otherwise be the newest and safest thing to delete only in
    the degenerate case where it alone exceeds the limit.
    """
    base = root or cache_root()
    if not base.exists():
        return 0

    entries = []
    for child in base.iterdir():
        if not child.is_dir():
            continue
        try:
            entries.append((child.stat().st_mtime, _dir_size(child), child))
        except OSError:
            continue

    total = sum(size for _, size, _ in entries)
    evicted = 0
    for _, size, child in sorted(entries):
        if total <= limit:
            break
        if keep is not None and child.resolve() == keep.resolve():
            continue
        shutil.rmtree(child, ignore_errors=True)
        total -= size
        evicted += 1
    return evicted


def _cached_download(out_dir: Path) -> dict | None:
    """Return a download result if this directory already holds a usable file."""
    video = _pick_video(out_dir)
    # A run that died mid-download can leave the directory behind. yt-dlp keeps
    # incomplete files under a .part suffix, which _pick_video already ignores,
    # but an empty file would still match — treat it as a miss and re-fetch.
    if video is None or video.stat().st_size == 0:
        return None
    info_path = out_dir / "video.info.json"
    try:  # refresh LRU position
        os.utime(out_dir, None)
    except OSError:
        pass
    return {
        "video_path": str(video),
        "subtitle_path": str(sub) if (sub := _pick_subtitle(out_dir)) else None,
        "info": _read_info(info_path, "") or {},
        "downloaded": False,
        "cached": True,
    }


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
    use_cache: bool = False,
) -> dict:
    if shutil.which("yt-dlp") is None:
        raise SystemExit("yt-dlp is not installed. Install with: brew install yt-dlp")

    if use_cache:
        hit = _cached_download(out_dir)
        if hit is not None:
            print(f"[watch] reusing cached download: {out_dir}", file=sys.stderr)
            hit["info"] = hit["info"] or {"url": url}
            return hit

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

    subtitle = _pick_subtitle(out_dir)
    info = _read_info(out_dir / "video.info.json", url)

    if use_cache:
        evicted = prune_cache(keep=out_dir)
        if evicted:
            print(f"[watch] cache over limit — evicted {evicted} old download(s)", file=sys.stderr)

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
    use_cache: bool = False,
) -> dict:
    if is_url(source):
        return download_url(source, out_dir, audio_only=audio_only, use_cache=use_cache)
    return resolve_local(source)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: download.py <url-or-path> <out-dir>", file=sys.stderr)
        raise SystemExit(2)
    result = download(sys.argv[1], Path(sys.argv[2]))
    print(json.dumps(result, indent=2))
