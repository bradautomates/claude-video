#!/usr/bin/env python3
"""Download a video via yt-dlp, or resolve a local file path.

Also fetches subtitles (manual first, then auto-generated) in VTT format so
transcribe.py can parse them without needing Whisper.

Downloads are cache-aware: when the output directory already holds a completed
download (marked by COMPLETE_MARKER, written only after yt-dlp succeeds), the
video, subtitles, and info json are reused without touching the network. This
is what makes multi-pass analysis cheap — the video downloads once, every
focused re-run after that is local.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse


VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi", ".flv", ".wmv"}

# Audio-only sources are first-class: watch.py skips frame extraction and
# produces a transcript-only report (podcasts, voice notes, meeting recordings).
AUDIO_EXTS = {".m4a", ".mp3", ".wav", ".aac", ".flac", ".ogg", ".opus", ".wma"}

# Written after a fully successful download; a directory without it (e.g. an
# interrupted yt-dlp run) is never treated as a cache hit.
COMPLETE_MARKER = ".watch-download-complete"

# Written after a subtitle-fallback attempt found nothing, so cache hits on a
# genuinely caption-less video don't re-query the network every run.
NO_SUBS_MARKER = ".watch-no-subtitles"


def url_cache_dir(url: str, cache_root: Path) -> Path:
    """Stable per-URL download directory under the persistent cache."""
    key = hashlib.sha256(url.encode()).hexdigest()[:16]
    return cache_root / "videos" / key


def is_url(source: str) -> bool:
    parsed = urlparse(source)
    return parsed.scheme in ("http", "https")


def resolve_local(path: str) -> dict:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise SystemExit(f"File not found: {p}")
    if p.suffix.lower() not in VIDEO_EXTS | AUDIO_EXTS:
        print(
            f"[watch] warning: {p.suffix} is not a known video/audio extension, proceeding anyway",
            file=sys.stderr,
        )
    return {
        "video_path": str(p),
        "subtitle_path": None,
        "subtitle_lang": None,
        "subtitle_kind": None,
        "info": {"title": p.name, "url": str(p)},
        "downloaded": False,
    }


def _pick_subtitle(out_dir: Path) -> Path | None:
    candidates = sorted(out_dir.glob("video*.vtt"))
    if not candidates:
        return None
    preferred = [c for c in candidates if ".en" in c.name]
    return preferred[0] if preferred else candidates[0]


def _pick_video(out_dir: Path) -> Path | None:
    for ext in (".mp4", ".mkv", ".webm", ".mov"):
        for candidate in out_dir.glob(f"video*{ext}"):
            return candidate
    for candidate in out_dir.glob("video.*"):
        if candidate.suffix.lower() in VIDEO_EXTS:
            return candidate
    return None


def _read_raw_info(out_dir: Path) -> dict:
    info_path = out_dir / "video.info.json"
    if not info_path.exists():
        return {}
    try:
        return json.loads(info_path.read_text())
    except Exception:
        return {}


def _load_info(out_dir: Path, url: str) -> dict:
    raw = _read_raw_info(out_dir)
    if raw:
        return {
            "title": raw.get("title"),
            "uploader": raw.get("uploader") or raw.get("channel"),
            "duration": raw.get("duration"),
            "url": raw.get("webpage_url") or url,
        }
    return {"url": url}


def _subtitle_meta(out_dir: Path, subtitle: Path | None) -> tuple[str | None, str | None]:
    """Return (lang, kind) for the chosen subtitle file, kind in {manual, auto}.

    Auto-generated captions are markedly worse than whisper-large-v3; surfacing
    which one we got lets the caller judge transcript quality instead of
    treating all "captions" as equal.
    """
    if subtitle is None:
        return None, None
    stem = subtitle.name[: -len(".vtt")] if subtitle.name.endswith(".vtt") else subtitle.stem
    lang = stem.split(".", 1)[1] if "." in stem else None
    kind = None
    if lang:
        raw = _read_raw_info(out_dir)
        if lang in (raw.get("subtitles") or {}):
            kind = "manual"
        elif lang in (raw.get("automatic_captions") or {}):
            kind = "auto"
    return lang, kind


def _fetch_fallback_subtitle(url: str, out_dir: Path) -> Path | None:
    """No English subs came back — check info.json for what the source actually
    has and fetch the best alternative: manual captions in any language beat
    auto-generated ones, and the original language beats translations. Keeps
    free, human-made transcripts from silently falling through to paid Whisper
    just because they aren't in English."""
    if (out_dir / NO_SUBS_MARKER).exists():
        return None
    raw = _read_raw_info(out_dir)
    manual = [k for k in (raw.get("subtitles") or {}) if not k.startswith("live")]
    auto = raw.get("automatic_captions") or {}
    orig = raw.get("language")

    lang: str | None = None
    flag: str | None = None
    if manual:
        lang = orig if orig in manual else manual[0]
        flag = "--write-subs"
    elif orig and orig in auto:
        lang = orig
        flag = "--write-auto-subs"

    if not lang or not flag or shutil.which("yt-dlp") is None:
        try:
            (out_dir / NO_SUBS_MARKER).touch()
        except OSError:
            pass
        return None

    kind = "manual" if flag == "--write-subs" else "auto-generated"
    print(f"[watch] no English captions — fetching {lang} ({kind})…", file=sys.stderr)
    subprocess.run(
        [
            "yt-dlp",
            "--skip-download",
            flag,
            "--sub-langs", lang,
            "--sub-format", "vtt",
            "--convert-subs", "vtt",
            "--no-playlist",
            "--ignore-errors",
            "-o", str(out_dir / "video.%(ext)s"),
            url,
        ],
        stdout=sys.stderr,
        stderr=sys.stderr,
    )
    subtitle = _pick_subtitle(out_dir)
    if subtitle is None:
        try:
            (out_dir / NO_SUBS_MARKER).touch()
        except OSError:
            pass
    return subtitle


def _cached_download(url: str, out_dir: Path) -> dict | None:
    if not (out_dir / COMPLETE_MARKER).exists():
        return None
    video = _pick_video(out_dir)
    if video is None:
        return None
    print(f"[watch] download cache hit: {video.name} in {out_dir}", file=sys.stderr)
    # Also heals caches from before the language fallback existed.
    subtitle = _pick_subtitle(out_dir) or _fetch_fallback_subtitle(url, out_dir)
    lang, kind = _subtitle_meta(out_dir, subtitle)
    return {
        "video_path": str(video),
        "subtitle_path": str(subtitle) if subtitle else None,
        "subtitle_lang": lang,
        "subtitle_kind": kind,
        "info": _load_info(out_dir, url),
        "downloaded": False,
    }


def download_url(url: str, out_dir: Path) -> dict:
    cached = _cached_download(url, out_dir)
    if cached is not None:
        return cached

    if shutil.which("yt-dlp") is None:
        raise SystemExit("yt-dlp is not installed. Install with: brew install yt-dlp")

    out_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(out_dir / "video.%(ext)s")

    cmd = [
        "yt-dlp",
        "-N", "8",
        "-f", "bv*[height<=720]+ba/b[height<=720]/bv+ba/b",
        "--merge-output-format", "mp4",
        "--write-info-json",
        "--write-subs",
        "--write-auto-subs",
        "--sub-langs", "en,en-US,en-GB,en-orig",
        "--sub-format", "vtt",
        "--convert-subs", "vtt",
        "--no-playlist",
        "--ignore-errors",
        "-o", output_template,
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

    subtitle = _pick_subtitle(out_dir) or _fetch_fallback_subtitle(url, out_dir)
    lang, kind = _subtitle_meta(out_dir, subtitle)
    try:
        (out_dir / COMPLETE_MARKER).touch()
    except OSError:
        pass

    return {
        "video_path": str(video),
        "subtitle_path": str(subtitle) if subtitle else None,
        "subtitle_lang": lang,
        "subtitle_kind": kind,
        "info": _load_info(out_dir, url),
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
