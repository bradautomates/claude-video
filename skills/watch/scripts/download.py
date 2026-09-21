#!/usr/bin/env python3
"""Download a video via yt-dlp, or resolve a local file path.

Also fetches subtitles (manual first, then auto-generated) in VTT format so
transcribe.py can parse them without needing Whisper.

Caption language: English is requested first, which keeps English videos at a
single round trip. The info.json yt-dlp writes alongside is then consulted: if
the video is in another language (``language`` / ``original_language``), or
``--lang`` named one, or English returned nothing, one bounded second fetch is
made for that language so the transcript is the speech itself rather than
YouTube's machine translation of it. ``all`` is never requested — on YouTube
that means hundreds of auto-translated tracks and minutes of stalling.
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


DEFAULT_SUB_LANGS = "en.*"


def _read_raw_info(info_path: Path) -> dict:
    if not info_path.exists():
        return {}
    try:
        raw = json.loads(info_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def _base_lang(tag: str) -> str:
    """``de-DE`` / ``de-orig`` / ``DE`` -> ``de``."""
    return tag.split("-")[0].lower()


def _native_lang(raw: dict) -> str | None:
    """The language the video is spoken in, or None if unknown or English.

    yt-dlp reports ``language`` or ``original_language`` depending on the
    extractor. Falls back to the ``-orig`` automatic-caption tag, which YouTube
    only emits for the language actually spoken.
    """
    lang = raw.get("language") or raw.get("original_language")
    if not isinstance(lang, str) or not lang:
        for tag in (raw.get("automatic_captions") or {}):
            if tag.endswith("-orig"):
                lang = tag
                break
    if not isinstance(lang, str) or not lang:
        return None
    base = _base_lang(lang)
    return None if base in ("", "en") else base


def _manual_langs(raw: dict) -> set[str]:
    """Tags of uploader-supplied (human) caption tracks."""
    return {t for t in (raw.get("subtitles") or {}) if t != "live_chat"}


def _sub_langs_for(base: str) -> str:
    """Bounded --sub-langs pattern for one language.

    Exact code, its ``-orig`` variant and region variants (``de-AT``). Not
    ``de.*`` — that also matches unrelated multi-part tags, and every extra
    track is another request that can draw a 429 and cost the transcript.
    """
    return f"{base},{base}-orig,{base}-[A-Za-z][A-Za-z]"


def _sub_lang_args(langs: str) -> list[str]:
    return [
        "--write-subs",
        "--write-auto-subs",
        "--sub-langs", langs,
        "--sub-format", "vtt",
        "--convert-subs", "vtt",
    ]


def _lang_of(path: Path) -> str:
    # video.en-US.vtt -> en-US
    return path.name[len("video."):-len(".vtt")]


def _pick_subtitle(
    out_dir: Path,
    raw_info: dict | None = None,
    prefer: str | None = None,
) -> Path | None:
    """Pick the best VTT in *out_dir*.

    Ranking: a human-authored track always wins (punctuated, speaker-labelled,
    about a third the size of the rolling auto track); then the ``-orig``
    track of the preferred language, then any track of it; then English.
    """
    candidates = sorted(out_dir.glob("video*.vtt"))
    if not candidates:
        return None
    manual = _manual_langs(raw_info or {})
    prefer_base = _base_lang(prefer) if prefer else None

    def rank(path: Path) -> tuple[int, int, int]:
        tag = _lang_of(path)
        base = _base_lang(tag)
        is_manual = 0 if tag in manual else 1
        if prefer_base and base == prefer_base:
            pref = 0 if tag.endswith("-orig") else 1
        else:
            pref = 2
        en_order = {"en": 1, "en-US": 2, "en-GB": 3, "en-orig": 4}.get(tag, 5)
        return (is_manual, pref, en_order)

    return min(candidates, key=rank)


def _has_sub_for(out_dir: Path, base: str) -> bool:
    return any(_base_lang(_lang_of(p)) == base for p in out_dir.glob("video*.vtt"))


def _resolve_subtitle(url: str, out_dir: Path, lang: str | None) -> Path | None:
    """Decide whether the English pass was enough and, if not, fetch once more.

    *lang* is an explicit override (``--lang``); otherwise the video's own
    language from info.json. Makes at most one extra yt-dlp call.
    """
    raw = _read_raw_info(out_dir / "video.info.json")
    target = _base_lang(lang) if lang else _native_lang(raw)

    if target and target != "en" and not _has_sub_for(out_dir, target):
        print(
            f"[watch] video language is {target}; fetching its own captions "
            f"instead of a translated track",
            file=sys.stderr,
        )
        _fetch_subs_only(url, out_dir, _sub_langs_for(target))
        if not _has_sub_for(out_dir, target):
            print(f"[watch] no {target} captions available; falling back to English", file=sys.stderr)
    elif target is None and not any(out_dir.glob("video*.vtt")):
        # Unknown language and English returned nothing: take an uploader track
        # in any language, else the spoken-language -orig automatic track.
        fallback = next(iter(sorted(_manual_langs(raw))), None) or next(
            (t for t in (raw.get("automatic_captions") or {}) if t.endswith("-orig")), None
        )
        if fallback:
            print(f"[watch] no English captions — retrying with {fallback}", file=sys.stderr)
            _fetch_subs_only(url, out_dir, fallback)
            target = _base_lang(fallback)

    return _pick_subtitle(out_dir, raw, prefer=target)


def _fetch_subs_only(url: str, out_dir: Path, langs: str) -> None:
    cmd = [
        "yt-dlp",
        "--skip-download",
        *_sub_lang_args(langs),
        "--no-playlist",
        "--ignore-errors",
        "-o", str(out_dir / "video.%(ext)s"),
        "--",
        url,
    ]
    subprocess.run(cmd, stdout=sys.stderr, stderr=sys.stderr)


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
            ["yt-dlp", "--version"], capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=5
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


def fetch_captions(url: str, out_dir: Path, lang: str | None = None) -> dict:
    """Fetch metadata and best available VTT captions without downloading video.

    *lang* forces a caption language (base code such as ``de``); by default the
    video's own language is used, with English as the fallback.
    """
    if shutil.which("yt-dlp") is None:
        raise SystemExit("yt-dlp is not installed. Install with: brew install yt-dlp")

    out_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(out_dir / "video.%(ext)s")
    cmd = [
        "yt-dlp",
        "--skip-download",
        "--write-info-json",
        *_sub_lang_args(DEFAULT_SUB_LANGS),
        "--no-playlist",
        "--ignore-errors",
        "-o", output_template,
        "--",
        url,
    ]
    subprocess.run(cmd, stdout=sys.stderr, stderr=sys.stderr)
    subtitle = _resolve_subtitle(url, out_dir, lang)
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
    lang: str | None = None,
) -> dict:
    if shutil.which("yt-dlp") is None:
        raise SystemExit("yt-dlp is not installed. Install with: brew install yt-dlp")

    out_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(out_dir / "video.%(ext)s")

    # fetch_captions usually ran first into this same directory, so the video's
    # language is already known and the right track can be asked for up front.
    known = _base_lang(lang) if lang else _native_lang(_read_raw_info(out_dir / "video.info.json"))
    langs = _sub_langs_for(known) if known and known != "en" else DEFAULT_SUB_LANGS

    fmt = "ba/bestaudio" if audio_only else "bv*[height<=720]+ba/b[height<=720]/bv+ba/b"
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
    #
    # Output is captured (rather than piped straight through to the inherited
    # stderr fd) so a failure can be diagnosed -- e.g. a stale yt-dlp getting
    # 403'd by YouTube's latest signature/client rotation -- instead of only
    # surfacing a bare exit code. It's echoed to stderr after the fact so
    # nothing that was visible before is lost, just no longer live-streamed.
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        encoding="utf-8", errors="replace",
    )
    if result.stdout:
        sys.stderr.write(result.stdout)

    video = _pick_video(out_dir)
    subtitle = _resolve_subtitle(url, out_dir, lang)
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
    lang: str | None = None,
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
