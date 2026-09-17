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


def _yt_dlp_version() -> str | None:
    """Best-effort `yt-dlp --version` output, or None if it can't be read.

    No network call -- this just execs the already-installed binary. Used only
    to enrich a 403 failure message, so any failure here (missing binary,
    timeout, odd build) degrades to omitting the version rather than raising.
    """
    try:
        proc = subprocess.run(
            ["yt-dlp", "--version"], capture_output=True, text=True, timeout=5
        )
        return proc.stdout.strip() or None
    except Exception:
        return None


def _update_hint() -> str:
    """Upgrade command matching how yt-dlp appears to be installed."""
    path = shutil.which("yt-dlp") or ""
    if "pipx" in path:
        return "pipx upgrade yt-dlp"
    if "Cellar" in path or "homebrew" in path.lower():
        return "brew upgrade yt-dlp"
    return "yt-dlp -U  (or: pip install -U yt-dlp)"


def _download_failure_message(
    output: str, returncode: int, out_dir: Path, subtitle: Path | None
) -> str:
    """Build the error surfaced when yt-dlp produced no video file.

    A bare exit code tells the user nothing actionable. Real incident
    (2026-09-17): a 403 on the media stream was actually a yt-dlp build 2.5
    months stale that had lost YouTube's current client/signature rotation --
    upgrading fixed it with no code change. So a 403/Forbidden in the captured
    output gets the actionable explanation; every other failure keeps the
    original bare message unchanged, since we have no comparable evidence
    about what those mean.
    """
    base = f"yt-dlp did not produce a video file in {out_dir} (exit {returncode})"
    if "403" not in output and "Forbidden" not in output:
        return base
    version = _yt_dlp_version()
    version_note = f" (yt-dlp {version})" if version else ""
    lines = [
        base,
        f"HTTP 403 on the media stream{version_note} -- almost always a yt-dlp that has "
        "fallen behind YouTube's latest signature/client rotation, not a video that's "
        "actually blocked or region-locked.",
        f"Update and retry: {_update_hint()}",
    ]
    if subtitle:
        lines.append(
            f"Captions downloaded fine ({subtitle.name}) -- the transcript is usable even "
            "before you retry the video."
        )
    return "\n".join(lines)


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
        "--",
        url,
    ]

    # yt-dlp may exit non-zero if a subtitle variant fails (e.g. 429) even when
    # the video itself downloaded fine. Treat "video file present" as success.
    #
    # Output is captured (rather than piped straight through to the inherited
    # stderr fd) so a failure can be diagnosed -- e.g. a stale yt-dlp getting
    # 403'd by YouTube's latest signature/client rotation -- instead of only
    # surfacing a bare exit code. It's echoed to stderr after the fact so
    # nothing that was visible before is lost, just no longer live-streamed.
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.stdout:
        sys.stderr.write(result.stdout)

    video = _pick_video(out_dir)
    subtitle = _pick_subtitle(out_dir)
    info = _read_info(out_dir / "video.info.json", url)

    if video is None:
        failure_message = _download_failure_message(
            result.stdout or "", result.returncode, out_dir, subtitle
        )
        if subtitle is None:
            raise SystemExit(failure_message)
        # The media stream is unavailable (e.g. 403'd) but captions DID come
        # down -- hand back a degraded-but-usable result instead of discarding
        # a complete transcript. watch.py finishes the run in transcript-only
        # mode rather than dying, per the 2026-09-17 incident: subtitles were
        # 272 KB and carried essentially the whole talk while only the media
        # stream failed.
        print(failure_message, file=sys.stderr)
        return {
            "video_path": None,
            "subtitle_path": str(subtitle),
            "info": info or {"url": url},
            "downloaded": False,
            "degraded": True,
            "failure_message": failure_message,
        }

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
