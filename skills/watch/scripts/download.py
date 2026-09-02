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


def _lang_list(lang: str) -> list[str]:
    """Split a comma-separated priority list into base codes ('pt-BR' -> 'pt')."""
    out: list[str] = []
    for part in str(lang).split(","):
        base = part.strip().split("-")[0].lower()
        if base and base not in out:
            out.append(base)
    return out or ["en"]


def _sub_langs_arg(lang: str) -> str:
    """Build yt-dlp's --sub-langs value from the priority list.

    Exact codes, deliberately not a `code.*` glob. yt-dlp matches --sub-langs
    case-insensitively, so `pt.*` also selects YouTube's auto-translated pairs
    (`pt-en`, `pt-de`, `pt-PT-en`, ...). On a video that offers several source
    languages that turns one caption fetch into seven, and the extra requests
    draw HTTP 429s that can cost the transcript entirely. `-orig` is yt-dlp's
    own name for the untranslated track, so it is the one variant worth asking
    for by name.
    """
    return ",".join(f"{code},{code}-orig" for code in _lang_list(lang))


def _pick_subtitle(out_dir: Path, lang: str = "en,pt") -> Path | None:
    candidates = sorted(out_dir.glob("video*.vtt"))
    if not candidates:
        return None

    def _match(base: str, suffix: str | None) -> Path | None:
        want = f".{base}-orig." if suffix == "orig" else None
        for c in candidates:
            name = c.name.lower()
            if want is not None:
                if want in name:
                    return c
            elif f".{base}." in name or f".{base}-" in name:
                return c
        return None

    bases = _lang_list(lang)
    # Pass 1 — `-orig` is yt-dlp's name for the track in the language actually
    # spoken in the video, so it outranks the caller's ordering entirely. Without
    # this, a Portuguese video that also publishes an English auto-translation
    # returns the translation under the default `en,pt`: `en` wins the priority
    # loop, and the user silently gets machine-translated English instead of the
    # original speech.
    for base in bases:
        hit = _match(base, "orig")
        if hit is not None:
            return hit
    # Pass 2 — no original-language track among the requested languages, so fall
    # back to the caller's priority order.
    for base in bases:
        hit = _match(base, None)
        if hit is not None:
            return hit
    return candidates[0]


def _pick_video(out_dir: Path) -> Path | None:
    for ext in (".mp4", ".mkv", ".webm", ".mov", ".m4a", ".mp3", ".opus"):
        for candidate in out_dir.glob(f"video*{ext}"):
            return candidate
    for candidate in out_dir.glob("video.*"):
        if candidate.suffix.lower() in VIDEO_EXTS:
            return candidate
    return None


def fetch_captions(url: str, out_dir: Path, lang: str = "en,pt") -> dict:
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
        "--sub-langs", _sub_langs_arg(lang),
        "--sub-format", "vtt",
        "--convert-subs", "vtt",
        "--no-playlist",
        "--ignore-errors",
        "-o", output_template,
        "--",
        url,
    ]
    subprocess.run(cmd, stdout=sys.stderr, stderr=sys.stderr)
    subtitle = _pick_subtitle(out_dir, lang)
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
    lang: str = "en,pt",
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
        "--sub-langs", _sub_langs_arg(lang),
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

    subtitle = _pick_subtitle(out_dir, lang)
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
    lang: str = "en,pt",
) -> dict:
    if is_url(source):
        return download_url(source, out_dir, audio_only=audio_only, lang=lang)
    return resolve_local(source)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: download.py <url-or-path> <out-dir>", file=sys.stderr)
        raise SystemExit(2)
    result = download(sys.argv[1], Path(sys.argv[2]))
    print(json.dumps(result, indent=2))
