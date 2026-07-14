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

# Fallback when the per-video probe below can't tell us anything (non-YouTube
# site, network hiccup): English (any variant) plus the single "<lang>-orig"
# track yt-dlp/YouTube tags onto whichever language the video was actually
# spoken in. This is NOT the same as "--sub-langs all" — "-orig" matches at
# most one track, so it can't trigger the multi-minute stall that requesting
# every translated caption causes.
SUB_LANGS = "en.*,.*-orig"


def _probe_langs(url: str) -> dict | None:
    """Cheap, fast, metadata-only lookup of which caption languages exist.

    Downloads nothing: yt-dlp's plain info dump (`-j`) lists every manual and
    automatic-caption language as part of parsing the page/player response —
    that listing is free. Only the *download* step (driven by --sub-langs in
    fetch_captions/download_url) can be slow if asked for too many languages
    at once; this call asks for none, so it stays fast regardless of how many
    translations YouTube offers.

    Returns None on any failure (non-YouTube site, network hiccup, timeout)
    so the caller can fall back to the bounded SUB_LANGS pattern above.
    """
    try:
        result = subprocess.run(
            ["yt-dlp", "--skip-download", "-j", "--no-warnings", "--no-playlist", "--", url],
            capture_output=True, text=True, timeout=30,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        info = json.loads(result.stdout.splitlines()[0])
    except (json.JSONDecodeError, IndexError):
        return None
    return {
        "language": (info.get("language") or "").split("-")[0] or None,
        "manual_langs": set((info.get("subtitles") or {}).keys()),
        "auto_langs": set((info.get("automatic_captions") or {}).keys()),
    }


def _choose_target_lang(probe: dict | None) -> str | None:
    """Priority order: manual (human-written) subtitle in the video's own
    language > any manual subtitle > automatic transcript in the video's own
    language > nothing.

    Manual subtitles never carry the "-orig" tag — YouTube only tags
    automatic captions that way — so checking `manual_langs` directly is the
    only way to prefer a hand-written non-English subtitle over an
    auto-generated one.

    A manual subtitle in the declared language isn't always keyed by the
    plain code: some uploaders' captions (e.g. via a third-party
    localization vendor) land as "en-<vendor-id>" instead of a plain "en".
    Match by prefix (declared + "-"), same as automatic captions' "en-US"/
    "en-GB" variants, not just an exact key — otherwise this falls through
    to "any manual subtitle" and picks one alphabetically, which can land on
    an unrelated language (seen live: "ar" beat a real "en-<vendor-id>"
    track purely because "ar" sorts first).
    """
    if not probe:
        return None
    declared = probe["language"]
    manual = probe["manual_langs"]
    auto = probe["auto_langs"]
    if declared:
        declared_manual = sorted(
            m for m in manual if m == declared or m.startswith(f"{declared}-")
        )
        if declared_manual:
            return declared if declared in declared_manual else declared_manual[0]
    if manual:
        return sorted(manual)[0]
    if declared and f"{declared}-orig" in auto:
        return f"{declared}-orig"
    orig = sorted(k for k in auto if k.endswith("-orig"))
    return orig[0] if orig else None


def _sub_langs_for(url: str) -> str:
    """--sub-langs value to request for this specific video.

    Falls back to the broad SUB_LANGS pattern when the probe fails or the
    site isn't YouTube (the "language"/"-orig" fields are YouTube-specific —
    other yt-dlp-supported sites may not expose them, and the broad pattern
    still behaves exactly as before there).
    """
    target = _choose_target_lang(_probe_langs(url))
    if target is None:
        return SUB_LANGS
    return f"^{re.escape(target)}$"


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
    """Prefer the video's original spoken-language track over a translation.

    yt-dlp tags the true source-language auto-caption as "<lang>-orig"
    (e.g. "it-orig" for an Italian video, "en-orig" for an English one) —
    see SUB_LANGS above. That beats any plain English file, which for a
    non-English video is a machine translation, not the original text.
    English is kept as the second choice for videos with manual (human)
    English subtitles but no "-orig" auto-caption.
    """
    candidates = sorted(out_dir.glob("video*.vtt"))
    if not candidates:
        return None
    orig = [c for c in candidates if "-orig." in c.name]
    if orig:
        return orig[0]
    preferred = [
        c for c in candidates
        if any(marker in c.name for marker in (".en.", ".en-US.", ".en-GB."))
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
        "--sub-langs", _sub_langs_for(url),
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
        "--sub-langs", _sub_langs_for(url),
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
