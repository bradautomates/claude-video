#!/usr/bin/env python3
"""Transcribe a video via Groq or OpenAI Whisper API.

Strategy: extract audio (mono 16kHz mp3, tiny payload), upload to whichever
API has a key. Returns segments in the same shape as transcribe.parse_vtt so
the rest of the pipeline (filter_range, format_transcript) doesn't care where
the transcript came from.

Pure stdlib — no `pip install groq` or `pip install openai` needed.
"""
from __future__ import annotations

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


def _post_whisper(endpoint: str, api_key: str, model: str, audio_path: Path) -> dict:
    fields = {
        "model": model,
        "response_format": "verbose_json",
        "temperature": "0",
    }
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


def _chunk_windows(start: float, end: float, chunk_seconds: float) -> list[tuple[float, float]]:
    """Split [start, end) into windows of chunk_seconds. The last may be shorter."""
    windows: list[tuple[float, float]] = []
    cursor = start
    while cursor < end:
        windows.append((cursor, min(cursor + chunk_seconds, end)))
        cursor += chunk_seconds
    return windows


def _post_for_backend(backend: str, api_key: str, audio_path: Path) -> dict:
    if backend == "groq":
        return _post_whisper(GROQ_ENDPOINT, api_key, GROQ_MODEL, audio_path)
    if backend == "openai":
        return _post_whisper(OPENAI_ENDPOINT, api_key, OPENAI_MODEL, audio_path)
    raise SystemExit(f"Unknown whisper backend: {backend}")


def transcribe_video(
    video_path: str,
    audio_out: Path,
    backend: str | None = None,
    api_key: str | None = None,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
) -> tuple[list[dict], str, list[tuple[float, float, str]]]:
    """Run the full flow: extract audio → upload → parse segments.

    Audio over CHUNK_DURATION_SECONDS is split into chunks and uploaded
    independently. A chunk that fails is reported and skipped — the caller
    gets segments from the successful chunks plus a list of
    (start, end, reason) tuples for the failures. Total failure (no chunks
    succeeded) raises SystemExit.

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

    # Single-upload path — preserves prior behavior for short audio.
    if eff_duration <= CHUNK_DURATION_SECONDS:
        scope = f" ({eff_start:.0f}s-{eff_end:.0f}s)" if (start_seconds is not None or end_seconds is not None) else ""
        print(f"[watch] extracting audio for Whisper ({backend}){scope}…", file=sys.stderr)
        audio_path = extract_audio(video_path, audio_out, start_seconds, end_seconds)
        size_kb = audio_path.stat().st_size / 1024
        print(f"[watch] audio: {size_kb:.0f} kB — uploading to {backend} Whisper…", file=sys.stderr)

        response = _post_for_backend(backend, api_key, audio_path)
        offset = eff_start if eff_start > 0 else 0.0
        segments = _segments_from_response(response, time_offset=offset)
        if not segments:
            raise SystemExit("Whisper returned no transcript segments")
        print(f"[watch] transcribed {len(segments)} segments via {backend}", file=sys.stderr)
        return segments, backend, []

    # Chunked path — split audio, upload each window, stitch with offsets.
    windows = _chunk_windows(eff_start, eff_end, CHUNK_DURATION_SECONDS)
    print(
        f"[watch] {eff_duration:.0f}s exceeds {CHUNK_DURATION_SECONDS}s — "
        f"splitting into {len(windows)} chunks of ≤{CHUNK_DURATION_SECONDS}s",
        file=sys.stderr,
    )

    all_segments: list[dict] = []
    failures: list[tuple[float, float, str]] = []
    audio_out.parent.mkdir(parents=True, exist_ok=True)

    for i, (chunk_start, chunk_end) in enumerate(windows, 1):
        label = f"chunk {i}/{len(windows)} ({chunk_start:.0f}s-{chunk_end:.0f}s)"
        chunk_audio = audio_out.parent / f"{audio_out.stem}_chunk_{i:03d}{audio_out.suffix}"
        try:
            print(f"[watch] {label}: extracting audio…", file=sys.stderr)
            extract_audio(video_path, chunk_audio, chunk_start, chunk_end)
            size_kb = chunk_audio.stat().st_size / 1024
            print(f"[watch] {label}: {size_kb:.0f} kB — uploading…", file=sys.stderr)

            response = _post_for_backend(backend, api_key, chunk_audio)
            chunk_segments = _segments_from_response(response, time_offset=chunk_start)
            all_segments.extend(chunk_segments)
            print(f"[watch] {label}: {len(chunk_segments)} segments", file=sys.stderr)
        except SystemExit as exc:
            failures.append((chunk_start, chunk_end, str(exc)))
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
        print("usage: whisper.py <video-path> [<audio-out.mp3>] [--backend groq|openai]", file=sys.stderr)
        raise SystemExit(2)

    video = sys.argv[1]
    audio_out = Path(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else Path("audio.mp3")
    backend_override = None
    if "--backend" in sys.argv:
        backend_override = sys.argv[sys.argv.index("--backend") + 1]

    segments, backend = transcribe_video(video, audio_out, backend=backend_override)
    print(json.dumps({"backend": backend, "segments": segments}, indent=2))
