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
import re
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

# Chunking: Whisper APIs (Groq + OpenAI) cap uploads at 25 MB. Above the
# threshold we split at silence boundaries; the existing single-upload path
# is preserved byte-for-byte for everything below it.
WHISPER_CHUNK_THRESHOLD_MB = 22       # 3 MB headroom under the 25 MB server limit
MIN_CHUNK_FLOOR_MB = 5                # don't let the greedy planner pick a tiny first chunk
SILENCE_NOISE_DB = "-30dB"            # matches typical podcast/interview noise floor
SILENCE_MIN_DURATION = 0.5            # natural sentence-pause length


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
            for line in path.read_text(encoding="utf-8").splitlines():
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


def extract_audio(video_path: str, out_path: Path) -> Path:
    """Extract mono 16kHz 64kbps mp3 — ~480 kB/min. Hits the 22 MB chunking
    threshold around 46 min (server limit is 25 MB at ~52 min); the caller in
    transcribe_video() splits at silences past the threshold.
    """
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is not installed. Install with: brew install ffmpeg")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-i", str(Path(video_path).resolve()),
        "-vn",
        "-acodec", "libmp3lame",
        "-ar", "16000",
        "-ac", "1",
        "-b:a", "64k",
        str(out_path.resolve()),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"ffmpeg audio extraction failed: {result.stderr.strip()}")
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise SystemExit("ffmpeg produced no audio — video may have no audio track")
    return out_path


_SILENCE_START_RE = re.compile(r"silence_start:\s*(-?\d+(?:\.\d+)?)")


def _audio_duration(audio_path: Path) -> float:
    """Read the duration of an audio file in seconds via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "default=nw=1:nk=1",
            str(audio_path.resolve()),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise SystemExit(f"ffprobe failed to read audio duration: {result.stderr.strip()}")
    return float(result.stdout.strip())


def _detect_silences(audio_path: Path) -> list[float]:
    """Run ffmpeg's silencedetect filter; return ordered silence_start times (s).

    Returns an empty list if ffmpeg fails — the caller falls back to byte-target
    hard cuts, so a silencedetect failure degrades quality but doesn't abort
    chunking. The failure is logged to stderr so the user sees the real cause.
    """
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-i", str(audio_path.resolve()),
        "-af", f"silencedetect=noise={SILENCE_NOISE_DB}:duration={SILENCE_MIN_DURATION}",
        "-f", "null",
        "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        snippet = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "(no stderr)"
        print(
            f"[watch] silencedetect failed (rc={result.returncode}): {snippet} — "
            "falling back to byte-target cuts",
            file=sys.stderr,
        )
        return []
    silences: list[float] = []
    for line in result.stderr.splitlines():
        match = _SILENCE_START_RE.search(line)
        if not match:
            continue
        try:
            silences.append(float(match.group(1)))
        except ValueError:
            continue
    silences.sort()
    return silences


def _chunk_audio(audio_path: Path) -> list[tuple[Path, float]]:
    """Split audio at silence boundaries near a 22 MB target.

    Returns [(chunk_path, offset_seconds), ...] in chronological order. Falls
    back to hard byte-target cuts when no silence is available in the window —
    the resulting segment around such a cut may be mid-word but the rest of
    the transcript is unaffected.
    """
    file_size = audio_path.stat().st_size
    duration_s = _audio_duration(audio_path)
    if duration_s <= 0:
        raise SystemExit("Audio file has zero duration — cannot chunk")

    target_bytes = WHISPER_CHUNK_THRESHOLD_MB * 1024 * 1024
    floor_bytes = MIN_CHUNK_FLOOR_MB * 1024 * 1024
    bytes_per_sec = file_size / duration_s

    silences = _detect_silences(audio_path)

    cut_times: list[float] = []
    cursor_bytes = 0
    while file_size - cursor_bytes > target_bytes:
        lo = cursor_bytes + floor_bytes
        hi = cursor_bytes + target_bytes
        # Latest silence inside the (lo, hi] byte window — maximises chunk size
        # without exceeding the upload threshold.
        best_t: float | None = None
        for t in silences:
            byte_off = t * bytes_per_sec
            if byte_off <= lo:
                continue
            if byte_off > hi:
                break
            best_t = t  # keep iterating; we want the LAST candidate <= hi
        if best_t is not None:
            cut_bytes = best_t * bytes_per_sec
            cut_t = best_t
        else:
            cut_bytes = cursor_bytes + target_bytes
            cut_t = cut_bytes / bytes_per_sec
        cut_times.append(cut_t)
        cursor_bytes = int(cut_bytes)

    chunks_dir = audio_path.parent / "audio_chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    for stale in chunks_dir.glob("chunk_*.mp3"):
        stale.unlink()

    boundaries = [0.0] + cut_times + [duration_s]
    chunks: list[tuple[Path, float]] = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        delta = boundaries[i + 1] - start
        chunk_path = chunks_dir / f"chunk_{i:03d}.mp3"
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-ss", f"{start:.3f}",
            "-t", f"{delta:.3f}",
            "-i", str(audio_path.resolve()),
            "-c", "copy",
            str(chunk_path.resolve()),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise SystemExit(f"ffmpeg chunk split failed at chunk {i}: {result.stderr.strip()}")
        if not chunk_path.exists() or chunk_path.stat().st_size == 0:
            raise SystemExit(f"chunk {i} produced no output")
        chunks.append((chunk_path, start))

    return chunks


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


MAX_ATTEMPTS = 4       # initial + 3 retries
MAX_429_RETRIES = 2
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


def _segments_from_response(data: dict) -> list[dict]:
    """Convert Whisper verbose_json into our {start, end, text} segment format."""
    out: list[dict] = []
    for seg in data.get("segments") or []:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        out.append({
            "start": round(float(seg.get("start") or 0.0), 2),
            "end": round(float(seg.get("end") or 0.0), 2),
            "text": text,
        })

    if not out:
        full = (data.get("text") or "").strip()
        if full:
            out.append({"start": 0.0, "end": 0.0, "text": full})

    return out


def transcribe_video(
    video_path: str,
    audio_out: Path,
    backend: str | None = None,
    api_key: str | None = None,
) -> tuple[list[dict], str]:
    """Run the full flow: extract audio → upload → parse segments.

    Returns (segments, backend_used). Raises SystemExit on any failure.
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

    if backend == "groq":
        endpoint, model = GROQ_ENDPOINT, GROQ_MODEL
    elif backend == "openai":
        endpoint, model = OPENAI_ENDPOINT, OPENAI_MODEL
    else:
        raise SystemExit(f"Unknown whisper backend: {backend}")

    print(f"[watch] extracting audio for Whisper ({backend})…", file=sys.stderr)
    audio_path = extract_audio(video_path, audio_out)
    size_kb = audio_path.stat().st_size / 1024
    threshold_bytes = WHISPER_CHUNK_THRESHOLD_MB * 1024 * 1024

    if audio_path.stat().st_size <= threshold_bytes:
        print(f"[watch] audio: {size_kb:.0f} kB — uploading to {backend} Whisper…", file=sys.stderr)
        response = _post_whisper(endpoint, api_key, model, audio_path)
        segments = _segments_from_response(response)
    else:
        print(
            f"[watch] audio: {size_kb:.0f} kB > {WHISPER_CHUNK_THRESHOLD_MB} MB — "
            "splitting at silences before upload…",
            file=sys.stderr,
        )
        chunks = _chunk_audio(audio_path)
        print(f"[watch] chunked into {len(chunks)} pieces — transcribing serially…", file=sys.stderr)
        segments = []
        for i, (chunk_path, offset_s) in enumerate(chunks, 1):
            chunk_kb = chunk_path.stat().st_size / 1024
            print(
                f"[watch] transcribing chunk {i}/{len(chunks)} ({chunk_kb:.0f} kB, "
                f"offset {offset_s:.1f}s)…",
                file=sys.stderr,
            )
            response = _post_whisper(endpoint, api_key, model, chunk_path)
            for seg in _segments_from_response(response):
                seg["start"] = round(seg["start"] + offset_s, 2)
                seg["end"] = round(seg["end"] + offset_s, 2)
                if segments and segments[-1]["text"] == seg["text"]:
                    segments[-1]["end"] = seg["end"]
                    continue
                segments.append(seg)

    if not segments:
        raise SystemExit("Whisper returned no transcript segments")

    print(f"[watch] transcribed {len(segments)} segments via {backend}", file=sys.stderr)
    return segments, backend


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
