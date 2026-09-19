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


DEFAULT_SUB_LANGS = "en.*"


def _caption_tracks(info_path: Path) -> tuple[list[str], list[str]]:
    """Return (manual_tags, automatic_tags) the extractor reported.

    Both come from yt-dlp's documented info-dict contract: `subtitles` holds
    tracks the uploader supplied, `automatic_captions` holds generated ones.
    """
    if not info_path.exists():
        return [], []
    try:
        raw = json.loads(info_path.read_text(encoding="utf-8"))
    except Exception:
        return [], []
    manual = list((raw.get("subtitles") or {}).keys())
    automatic = list((raw.get("automatic_captions") or {}).keys())
    return manual, automatic


def _native_sub_langs(info_path: Path) -> str | None:
    """Pick one language tag to retry with when English returned nothing.

    Never widens to `all`: on YouTube `automatic_captions` lists every machine
    translation target (hundreds of tracks), and requesting them stalls for
    minutes. An uploader-supplied track is preferred; failing that, only the
    `-orig` automatic track is taken, since that is the language actually
    spoken rather than a translation of it.
    """
    manual, automatic = _caption_tracks(info_path)

    for tag in manual:
        if tag and tag != "live_chat":
            return tag

    for tag in automatic:
        if tag.endswith("-orig"):
            return tag

    return None


def _sub_lang_args(langs: str) -> list[str]:
    return [
        "--write-subs",
        "--write-auto-subs",
        "--sub-langs", langs,
        "--sub-format", "vtt",
        "--convert-subs", "vtt",
    ]


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
    """Fetch metadata and best available VTT captions without downloading video.

    Tries English first, which is a single round trip for the common case. When
    that yields nothing, the info.json just written names the caption tracks
    that do exist, so retry once against the video's own language rather than
    falling through to a Whisper upload that costs money and minutes.
    """
    if shutil.which("yt-dlp") is None:
        raise SystemExit("yt-dlp is not installed. Install with: brew install yt-dlp")

    out_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(out_dir / "video.%(ext)s")

    def _run(langs: str) -> None:
        cmd = [
            "yt-dlp",
            "--skip-download",
            "--write-info-json",
            *_sub_lang_args(langs),
            "--no-playlist",
            "--ignore-errors",
            "-o", output_template,
            "--",
            url,
        ]
        subprocess.run(cmd, stdout=sys.stderr, stderr=sys.stderr)

    _run(DEFAULT_SUB_LANGS)
    subtitle = _pick_subtitle(out_dir)

    info_path = out_dir / "video.info.json"
    if subtitle is None:
        native = _native_sub_langs(info_path)
        if native:
            print(
                f"[watch] no English captions — retrying in the video's own "
                f"language ({native})",
                file=sys.stderr,
            )
            _run(native)
            subtitle = _pick_subtitle(out_dir)

    info = _read_info(info_path, url)
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
    # fetch_captions runs first and writes info.json into this same directory,
    # so the video's own caption language is already known — no extra probe.
    langs = _native_sub_langs(out_dir / "video.info.json") or DEFAULT_SUB_LANGS
    cmd = [
        "yt-dlp",
        "-N", "8",
        "-f", fmt,
        "--merge-output-format", "mp4",
        "--write-info-json",
        *_sub_lang_args(langs),
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
