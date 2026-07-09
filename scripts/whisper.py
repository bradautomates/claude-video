#!/usr/bin/env python3
"""Transcribe a video via Groq or OpenAI Whisper API.

Strategy: extract audio (mono 16kHz mp3, tiny payload), upload to whichever
API has a key. Returns segments in the same shape as transcribe.parse_vtt so
the rest of the pipeline (filter_range, format_transcript) doesn't care where
the transcript came from.

Pure stdlib — no `pip install groq` or `pip install openai` needed.
"""
from __future__ import annotations

import hashlib
import io
import json
import mimetypes
import os
import shutil
import ssl
import subprocess
import sys
import time
import urllib.error
import uuid
from pathlib import Path
from urllib.request import Request, urlopen


GROQ_ENDPOINT = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_MODEL = "whisper-large-v3"

OPENAI_ENDPOINT = "https://api.openai.com/v1/audio/transcriptions"
OPENAI_MODEL = "whisper-1"

# Audio over this duration is split into chunks before upload. 10 min keeps each
# chunk well under Groq's 25 MB per-file cap (at 64 kbps mono ≈ 4.7 MB/chunk)
# and bounds quota burn on retry — a failing chunk costs 600s, not the full
# video. Also lets one bad chunk be skipped while the rest of the transcript
# still gets through.
CHUNK_DURATION_SECONDS = 600

# Consecutive chunks overlap so a sentence straddling the cut is heard whole by
# at least one chunk. At merge time the overlap region is split at its midpoint:
# the earlier chunk keeps segments starting before it, the later chunk keeps the
# rest — no duplicated text, no mid-word mangling at the seam.
CHUNK_OVERLAP_SECONDS = 8.0

# Whisper's prompt biases recognition. Two uses here: caller-supplied vocab
# (proper nouns that would otherwise be garbled) and continuity — each chunk
# gets the tail of the previous chunk's text so recognition doesn't restart
# cold at every boundary (keeps names spelled consistently across chunks).
# The APIs only read the final ~224 tokens of the prompt, so the tail is capped.
PROMPT_TAIL_CHARS = 400

# Successful transcripts are cached to disk and looked up by
# (source file identity, window, backend, prompt). On cache hit we skip both
# audio extraction and the API call. Bump CACHE_VERSION whenever extract_audio's
# encoding parameters, the chunk window scheme, or the segment schema change.
CACHE_VERSION = 2
CACHE_DIR = Path.home() / ".cache" / "watch" / "chunks"


def load_api_key(preferred: str | None = None) -> tuple[str, str] | tuple[None, None]:
    """Return (backend, api_key). Prefers Groq, falls back to OpenAI.

    If `preferred` is "groq" or "openai", only that backend's key is considered.
    """
    def _from_env(name: str) -> str | None:
        value = os.environ.get(name)
        return value.strip() if value else None

    def _from_dotenv(path: Path, name: str) -> str | None:
        if not path.exists():
            return None
        try:
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                if key.strip() != name:
                    continue
                value = value.strip()
                if len(value) >= 2 and value[0] in ('"', "'") and value[-1] == value[0]:
                    value = value[1:-1]
                return value or None
        except OSError:
            return None
        return None

    dotenv_paths = [
        Path.home() / ".config" / "watch" / ".env",
        Path.cwd() / ".env",
    ]

    candidates = (("GROQ_API_KEY", "groq"), ("OPENAI_API_KEY", "openai"))
    if preferred is not None:
        candidates = tuple(c for c in candidates if c[1] == preferred)

    for key_name, backend in candidates:
        value = _from_env(key_name)
        if not value:
            for candidate in dotenv_paths:
                value = _from_dotenv(candidate, key_name)
                if value:
                    break
        if value:
            return backend, value

    return None, None


def extract_audio(
    video_path: str,
    out_path: Path,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
) -> Path:
    """Extract mono 16kHz 64kbps mp3 — ~480 kB/min, fits any Whisper limit.

    When start_seconds/end_seconds are set, only that window is extracted —
    the rest never reaches Whisper. This keeps focused-mode runs cheap
    (smaller upload, less quota burn) and matches the frame extraction scope.
    """
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is not installed. Install with: brew install ffmpeg")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # -ss before -i is fast-seek (less precise) but plenty for transcription;
    # -to is absolute end position in the source timeline.
    seek = []
    if start_seconds and start_seconds > 0:
        seek += ["-ss", f"{start_seconds:.3f}"]
    if end_seconds is not None:
        seek += ["-to", f"{end_seconds:.3f}"]

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        *seek,
        "-i", video_path,
        "-vn",
        "-acodec", "libmp3lame",
        "-ar", "16000",
        "-ac", "1",
        "-b:a", "64k",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"ffmpeg audio extraction failed: {result.stderr.strip()}")
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise SystemExit("ffmpeg produced no audio — video may have no audio track")
    return out_path


def _build_multipart(fields: dict[str, str], file_path: Path) -> tuple[bytes, str]:
    """Assemble a multipart/form-data body the Whisper APIs accept.

    Whisper's multipart upload is small and predictable — doing it by hand
    keeps us on pure stdlib instead of pulling requests/groq/openai SDKs.
    """
    boundary = f"----WatchBoundary{uuid.uuid4().hex}"
    eol = b"\r\n"
    buf = io.BytesIO()

    for name, value in fields.items():
        buf.write(f"--{boundary}".encode()); buf.write(eol)
        buf.write(f'Content-Disposition: form-data; name="{name}"'.encode()); buf.write(eol)
        buf.write(eol)
        buf.write(str(value).encode()); buf.write(eol)

    mimetype = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    buf.write(f"--{boundary}".encode()); buf.write(eol)
    buf.write(
        f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"'.encode()
    )
    buf.write(eol)
    buf.write(f"Content-Type: {mimetype}".encode()); buf.write(eol)
    buf.write(eol)
    buf.write(file_path.read_bytes())
    buf.write(eol)
    buf.write(f"--{boundary}--".encode()); buf.write(eol)

    return buf.getvalue(), boundary


MAX_ATTEMPTS = 4       # initial + 3 retries (network errors only)
MAX_429_RETRIES = 2
MAX_5XX_RETRIES = 2    # bail after 2 server-error hits — each retry re-uploads
                       # the full audio and counts against per-hour quota (Groq's
                       # ASPH limit). 4× of a 40-min file = ~3 hours of "audio"
                       # billed and exceeds the free-tier hourly cap.
RETRY_BASE_DELAY = 2.0


def _post_whisper(
    endpoint: str, api_key: str, model: str, audio_path: Path, prompt: str | None = None
) -> dict:
    fields = {
        "model": model,
        "response_format": "verbose_json",
        "temperature": "0",
    }
    if prompt:
        fields["prompt"] = prompt
    body, boundary = _build_multipart(fields, audio_path)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        # Groq sits behind Cloudflare — the default `Python-urllib/3.x` UA
        # trips WAF rule 1010 (403) before auth even runs. Any non-default
        # UA clears it; we identify honestly.
        "User-Agent": "watch-skill/1.0 (+claude-code; python-urllib)",
    }

    context = ssl.create_default_context()
    rate_limit_hits = 0
    server_error_hits = 0
    last_exc: Exception | None = None
    last_detail = ""

    for attempt in range(MAX_ATTEMPTS):
        request = Request(endpoint, data=body, headers=headers, method="POST")
        try:
            with urlopen(request, timeout=300, context=context) as response:
                payload = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = _read_error_body(exc)
            last_exc, last_detail = exc, detail

            # 4xx other than 429 are client errors — no retry will fix them.
            if 400 <= exc.code < 500 and exc.code != 429:
                raise SystemExit(f"Whisper request failed: {exc}{detail}")

            if exc.code == 429:
                rate_limit_hits += 1
                if rate_limit_hits >= MAX_429_RETRIES:
                    raise SystemExit(f"Whisper request failed: {exc}{detail}")
                delay = _retry_after(exc) or RETRY_BASE_DELAY * (2 ** attempt) + 1
            elif 500 <= exc.code < 600:
                server_error_hits += 1
                if server_error_hits >= MAX_5XX_RETRIES:
                    raise SystemExit(f"Whisper request failed: {exc}{detail}")
                delay = RETRY_BASE_DELAY * (2 ** attempt)
            else:
                delay = RETRY_BASE_DELAY * (2 ** attempt)

            if attempt < MAX_ATTEMPTS - 1:
                print(
                    f"[watch] whisper HTTP {exc.code} — retrying in {delay:.1f}s "
                    f"(attempt {attempt + 2}/{MAX_ATTEMPTS})",
                    file=sys.stderr,
                )
                time.sleep(delay)
            continue
        except (urllib.error.URLError, TimeoutError, ConnectionResetError, OSError) as exc:
            last_exc, last_detail = exc, ""
            if attempt < MAX_ATTEMPTS - 1:
                delay = RETRY_BASE_DELAY * (attempt + 1)
                print(
                    f"[watch] whisper network error ({type(exc).__name__}: {exc}) — "
                    f"retrying in {delay:.1f}s (attempt {attempt + 2}/{MAX_ATTEMPTS})",
                    file=sys.stderr,
                )
                time.sleep(delay)
            continue

        try:
            return json.loads(payload)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Whisper returned non-JSON response: {exc}: {payload[:200]}")

    raise SystemExit(
        f"Whisper request failed after {MAX_ATTEMPTS} attempts: {last_exc}{last_detail}"
    )


def _read_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read()
    except Exception:
        return ""
    if not body:
        return ""
    try:
        return f" — {body.decode('utf-8', errors='replace')[:400]}"
    except Exception:
        return ""


def _retry_after(exc: urllib.error.HTTPError) -> float | None:
    header = exc.headers.get("Retry-After") if getattr(exc, "headers", None) else None
    if not header:
        return None
    try:
        return float(header)
    except ValueError:
        return None


def _segments_from_response(data: dict, time_offset: float = 0.0) -> list[dict]:
    """Convert Whisper verbose_json into our {start, end, text} segment format.

    `time_offset` is added to each segment's timestamps. Used when the audio
    was extracted from a window starting at t > 0 — Whisper's timestamps are
    relative to the audio file, but the rest of the pipeline expects them
    relative to the source video.
    """
    out: list[dict] = []
    for seg in data.get("segments") or []:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        out.append({
            "start": round(float(seg.get("start") or 0.0) + time_offset, 2),
            "end": round(float(seg.get("end") or 0.0) + time_offset, 2),
            "text": text,
        })

    if not out:
        full = (data.get("text") or "").strip()
        if full:
            out.append({"start": round(time_offset, 2), "end": round(time_offset, 2), "text": full})

    return out


def _probe_duration(video_path: str) -> float:
    """Return source media duration in seconds via ffprobe. Used to size chunks
    when the caller didn't provide an explicit window."""
    if shutil.which("ffprobe") is None:
        raise SystemExit("ffprobe is not installed. Install with: brew install ffmpeg")
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"ffprobe failed: {result.stderr.strip()}")
    try:
        return float(result.stdout.strip())
    except ValueError:
        raise SystemExit(f"ffprobe returned non-numeric duration: {result.stdout!r}")


def _chunk_windows(
    start: float, end: float, chunk_seconds: float, overlap: float = CHUNK_OVERLAP_SECONDS
) -> list[tuple[float, float]]:
    """Split [start, end) into windows of chunk_seconds, each starting `overlap`
    seconds before the previous one ended. The last may be shorter."""
    step = max(1.0, chunk_seconds - overlap)
    windows: list[tuple[float, float]] = []
    cursor = start
    while cursor < end:
        windows.append((cursor, min(cursor + chunk_seconds, end)))
        if cursor + chunk_seconds >= end:
            break
        cursor += step
    return windows


def _post_for_backend(
    backend: str, api_key: str, audio_path: Path, prompt: str | None = None
) -> dict:
    if backend == "groq":
        return _post_whisper(GROQ_ENDPOINT, api_key, GROQ_MODEL, audio_path, prompt)
    if backend == "openai":
        return _post_whisper(OPENAI_ENDPOINT, api_key, OPENAI_MODEL, audio_path, prompt)
    raise SystemExit(f"Unknown whisper backend: {backend}")


def _cache_key(
    video_path: str, win_start: float, win_end: float, backend: str, prompt: str | None = None
) -> str:
    """Stable hash for a (source file identity, window, backend, prompt) tuple.

    Includes file size + mtime so editing or replacing the source video
    invalidates entries automatically. The prompt is part of the key because it
    changes the output — a continuity tail is deterministic across re-runs
    (same prior chunks → same tail), so cached chunks still hit. Encoding
    params are hardcoded in extract_audio; bump CACHE_VERSION if those change.
    """
    p = Path(video_path).resolve()
    try:
        st = p.stat()
    except OSError:
        return ""
    ident = (
        f"v{CACHE_VERSION}|{p}|{st.st_size}|{st.st_mtime_ns}|"
        f"{win_start:.3f}|{win_end:.3f}|{backend}|{prompt or ''}"
    )
    return hashlib.sha256(ident.encode()).hexdigest()[:32]


def _cache_load(key: str) -> list[dict] | None:
    """Return cached segments for `key` if a valid entry exists, else None."""
    if not key:
        return None
    path = CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if data.get("version") != CACHE_VERSION:
        return None
    segments = data.get("segments")
    if not isinstance(segments, list):
        return None
    return segments


def _cache_store(key: str, segments: list[dict], meta: dict) -> None:
    """Best-effort write to the cache. Failures are silently swallowed —
    a broken cache must never break the transcription pipeline."""
    if not key:
        return
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (CACHE_DIR / f"{key}.json").write_text(
            json.dumps({"version": CACHE_VERSION, "segments": segments, "meta": meta}, indent=2)
        )
    except OSError:
        pass


def _transcribe_window(
    video_path: str,
    audio_path: Path,
    backend: str,
    api_key: str,
    win_start: float,
    win_end: float,
    label: str,
    prompt: str | None = None,
) -> list[dict]:
    """Transcribe audio in [win_start, win_end). Returns segments with
    timestamps already offset to source-video coordinates.

    Cache-aware: a successful prior run with identical inputs (same source
    file identity + window + backend + prompt) skips both extraction and upload.
    """
    key = _cache_key(video_path, win_start, win_end, backend, prompt)
    cached = _cache_load(key)
    if cached is not None:
        print(
            f"[watch] {label}: cache hit ({len(cached)} segments — skipped upload)",
            file=sys.stderr,
        )
        return cached

    print(f"[watch] {label}: extracting audio…", file=sys.stderr)
    extract_audio(
        video_path, audio_path,
        start_seconds=win_start if win_start > 0 else None,
        end_seconds=win_end,
    )
    size_kb = audio_path.stat().st_size / 1024
    print(f"[watch] {label}: {size_kb:.0f} kB — uploading…", file=sys.stderr)

    response = _post_for_backend(backend, api_key, audio_path, prompt)
    segments = _segments_from_response(response, time_offset=win_start)

    if segments:
        _cache_store(key, segments, {
            "video_path": str(Path(video_path).resolve()),
            "win_start": win_start,
            "win_end": win_end,
            "backend": backend,
        })

    return segments


def _combine_prompt(vocab: str | None, tail: str | None) -> str | None:
    """Vocab terms + continuity tail, tail last (the APIs weight the end of
    the prompt most and truncate from the front)."""
    parts = [p for p in (vocab, tail) if p]
    return " ".join(parts) if parts else None


def transcribe_video(
    video_path: str,
    audio_out: Path,
    backend: str | None = None,
    api_key: str | None = None,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
    vocab: str | None = None,
) -> tuple[list[dict], str, list[tuple[float, float, str]]]:
    """Run the full flow: extract audio → upload → parse segments.

    `vocab` is free text (typically comma-separated proper nouns) passed as
    the Whisper prompt to bias spelling — product names, people, tools that
    would otherwise be garbled.

    Audio over CHUNK_DURATION_SECONDS is split into overlapping chunks and
    uploaded independently; at each seam the overlap is split at its midpoint
    so no text duplicates and no sentence is cut cold. Each chunk also gets
    the tail of the previous chunk's text as prompt context, so recognition
    (and name spelling) stays consistent across boundaries. A chunk that
    fails is reported and skipped — the caller gets segments from the
    successful chunks plus a list of (start, end, reason) tuples for the
    failures. Total failure (no chunks succeeded) raises SystemExit.

    Returns (segments, backend_used, failures). `failures` is empty on a
    clean run.
    """
    if backend is None or api_key is None:
        detected_backend, detected_key = load_api_key()
        backend = backend or detected_backend
        api_key = api_key or detected_key

    if not backend or not api_key:
        setup_py = Path(__file__).resolve().parent / "setup.py"
        raise SystemExit(
            "No Whisper API key available. Set GROQ_API_KEY (preferred) or OPENAI_API_KEY "
            "in the environment or in ~/.config/watch/.env. "
            f"Run `python3 {setup_py}` to configure."
        )

    eff_start = float(start_seconds) if start_seconds is not None else 0.0
    eff_end = float(end_seconds) if end_seconds is not None else _probe_duration(video_path)
    eff_duration = max(0.0, eff_end - eff_start)
    audio_out.parent.mkdir(parents=True, exist_ok=True)

    # Single-upload path — preserves prior behavior for short audio.
    if eff_duration <= CHUNK_DURATION_SECONDS:
        focused = start_seconds is not None or end_seconds is not None
        label = f"single ({eff_start:.0f}s-{eff_end:.0f}s)" if focused else f"single ({backend})"
        segments = _transcribe_window(
            video_path, audio_out, backend, api_key, eff_start, eff_end, label,
            prompt=_combine_prompt(vocab, None),
        )
        if not segments:
            raise SystemExit("Whisper returned no transcript segments")
        print(f"[watch] transcribed {len(segments)} segments via {backend}", file=sys.stderr)
        return segments, backend, []

    # Chunked path — split audio into overlapping windows, upload each,
    # stitch at overlap midpoints.
    windows = _chunk_windows(eff_start, eff_end, CHUNK_DURATION_SECONDS)
    print(
        f"[watch] {eff_duration:.0f}s exceeds {CHUNK_DURATION_SECONDS}s — "
        f"splitting into {len(windows)} chunks of ≤{CHUNK_DURATION_SECONDS}s "
        f"({CHUNK_OVERLAP_SECONDS:.0f}s overlap)",
        file=sys.stderr,
    )

    all_segments: list[dict] = []
    failures: list[tuple[float, float, str]] = []
    prev_tail: str | None = None
    prev_succeeded = False

    for i, (chunk_start, chunk_end) in enumerate(windows, 1):
        label = f"chunk {i}/{len(windows)} ({chunk_start:.0f}s-{chunk_end:.0f}s)"
        chunk_audio = audio_out.parent / f"{audio_out.stem}_chunk_{i:03d}{audio_out.suffix}"
        try:
            chunk_segments = _transcribe_window(
                video_path, chunk_audio, backend, api_key, chunk_start, chunk_end, label,
                prompt=_combine_prompt(vocab, prev_tail),
            )
            if i > 1 and prev_succeeded:
                # Both sides heard the overlap; hand off at its midpoint.
                boundary = chunk_start + CHUNK_OVERLAP_SECONDS / 2
                all_segments = [s for s in all_segments if s["start"] < boundary]
                chunk_segments = [s for s in chunk_segments if s["start"] >= boundary]
            all_segments.extend(chunk_segments)
            chunk_text = " ".join(s["text"] for s in chunk_segments)
            prev_tail = chunk_text[-PROMPT_TAIL_CHARS:] if chunk_text else None
            prev_succeeded = True
            print(f"[watch] {label}: {len(chunk_segments)} segments", file=sys.stderr)
        except SystemExit as exc:
            failures.append((chunk_start, chunk_end, str(exc)))
            # No continuity across a hole — the next chunk starts cold.
            prev_tail = None
            prev_succeeded = False
            print(f"[watch] {label}: FAILED — {exc}", file=sys.stderr)
        finally:
            try:
                chunk_audio.unlink()
            except OSError:
                pass

    if not all_segments:
        details = "; ".join(f"{s:.0f}-{e:.0f}: {r[:120]}" for s, e, r in failures)
        raise SystemExit(f"All {len(windows)} chunks failed — {details}")

    succeeded = len(windows) - len(failures)
    print(
        f"[watch] transcribed {len(all_segments)} segments via {backend} "
        f"({succeeded}/{len(windows)} chunks succeeded)",
        file=sys.stderr,
    )
    return all_segments, backend, failures


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "usage: whisper.py <video-path> [<audio-out.mp3>] [--backend groq|openai] [--vocab \"terms\"]",
            file=sys.stderr,
        )
        raise SystemExit(2)

    video = sys.argv[1]
    audio_out = Path(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else Path("audio.mp3")
    backend_override = None
    if "--backend" in sys.argv:
        backend_override = sys.argv[sys.argv.index("--backend") + 1]
    vocab_arg = None
    if "--vocab" in sys.argv:
        vocab_arg = sys.argv[sys.argv.index("--vocab") + 1]

    segments, backend, failures = transcribe_video(
        video, audio_out, backend=backend_override, vocab=vocab_arg,
    )
    print(json.dumps({"backend": backend, "segments": segments, "failures": failures}, indent=2))
