#!/usr/bin/env python3
"""Transcribe a video via Groq or OpenAI Whisper API.

Strategy: extract audio (mono 16kHz mp3). If the resulting file exceeds
Whisper's 25 MB upload limit, split into chunks via ffmpeg's segment muxer
and stitch transcripts with offset timestamps. Returns segments in the same
shape as transcribe.parse_vtt so the rest of the pipeline (filter_range,
format_transcript) doesn't care where the transcript came from.

Pure stdlib — no `pip install groq` or `pip install openai` needed.
"""
from __future__ import annotations

import io
import json
import math
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

# Whisper's upload ceiling is 25 MB; aim well below to leave headroom for the
# multipart envelope and to keep the last chunk from creeping over after the
# segment muxer's keyframe-snap padding.
MAX_UPLOAD_MB = 22


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
    """Extract mono 16kHz 64kbps mp3 — ~480 kB/min.

    Long videos may still exceed Whisper's 25 MB upload limit (~52 min at this
    bitrate); _post_whisper_chunked handles that case downstream.
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


def _audio_duration_seconds(audio_path: Path) -> float:
    """ffprobe the audio container for total duration."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(audio_path.resolve()),
        ],
        capture_output=True,
        text=True,
    )
    try:
        return float(result.stdout.strip() or 0)
    except ValueError:
        return 0.0


def _split_audio_into_chunks(
    audio_path: Path,
    chunks_dir: Path,
    max_chunk_mb: int = MAX_UPLOAD_MB,
) -> list[tuple[Path, float]]:
    """Split mp3 into ~max_chunk_mb-sized pieces.

    Returns [(chunk_path, offset_seconds)]. Uses ffmpeg's segment muxer with
    `-c copy` so the source isn't re-encoded. Segment length is computed from
    total bytes so each chunk lands close to the cap without spilling over.
    """
    duration = _audio_duration_seconds(audio_path)
    size_bytes = audio_path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)

    if duration <= 0 or size_mb <= max_chunk_mb:
        return [(audio_path, 0.0)]

    num_chunks = math.ceil(size_mb / max_chunk_mb)
    # Slight undershoot on segment length so the muxer's keyframe-snap padding
    # doesn't push a chunk past the cap.
    chunk_seconds = (duration / num_chunks) * 0.98

    chunks_dir.mkdir(parents=True, exist_ok=True)
    for existing in chunks_dir.glob("chunk_*.mp3"):
        existing.unlink()

    pattern = str(chunks_dir / "chunk_%03d.mp3")
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-i", str(audio_path.resolve()),
        "-f", "segment",
        "-segment_time", f"{chunk_seconds:.3f}",
        "-c", "copy",
        "-reset_timestamps", "1",
        pattern,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"audio chunking failed: {result.stderr.strip()}")

    chunks = sorted(chunks_dir.glob("chunk_*.mp3"))
    if not chunks:
        raise SystemExit("audio chunking produced no chunks")

    # Compute the actual offset per chunk by summing each chunk's measured
    # duration — relying on a uniform stride would drift if ffmpeg snaps to
    # a key/sync frame.
    out: list[tuple[Path, float]] = []
    cursor = 0.0
    for chunk in chunks:
        out.append((chunk, round(cursor, 3)))
        cursor += _audio_duration_seconds(chunk)
    return out


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


def _post_whisper_chunked(
    endpoint: str,
    api_key: str,
    model: str,
    audio_path: Path,
) -> dict:
    """Upload audio to Whisper, auto-splitting if it exceeds the 25 MB cap.

    For chunked uploads, transcripts are stitched with absolute timestamps so
    downstream code (filter_range, format_transcript) doesn't have to know
    chunking happened.
    """
    size_mb = audio_path.stat().st_size / (1024 * 1024)
    if size_mb <= MAX_UPLOAD_MB:
        return _post_whisper(endpoint, api_key, model, audio_path)

    print(
        f"[watch] audio is {size_mb:.0f} MB — splitting into ≤{MAX_UPLOAD_MB} MB chunks "
        f"for Whisper upload limit",
        file=sys.stderr,
    )
    chunks_dir = audio_path.parent / f"{audio_path.stem}-chunks"
    chunks = _split_audio_into_chunks(audio_path, chunks_dir)

    combined_segments: list[dict] = []
    combined_text_parts: list[str] = []
    for i, (chunk_path, offset) in enumerate(chunks):
        chunk_size_mb = chunk_path.stat().st_size / (1024 * 1024)
        print(
            f"[watch] chunk {i+1}/{len(chunks)} "
            f"({chunk_size_mb:.1f} MB, offset {offset:.1f}s) — uploading…",
            file=sys.stderr,
        )
        result = _post_whisper(endpoint, api_key, model, chunk_path)
        for seg in (result.get("segments") or []):
            seg = dict(seg)
            seg["start"] = float(seg.get("start", 0) or 0) + offset
            seg["end"] = float(seg.get("end", 0) or 0) + offset
            combined_segments.append(seg)
        if result.get("text"):
            combined_text_parts.append(result["text"])

    return {
        "segments": combined_segments,
        "text": " ".join(combined_text_parts),
    }


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
    """Run the full flow: extract audio → upload (chunking if needed) → parse segments.

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

    print(f"[watch] extracting audio for Whisper ({backend})…", file=sys.stderr)
    audio_path = extract_audio(video_path, audio_out)
    size_kb = audio_path.stat().st_size / 1024
    print(f"[watch] audio: {size_kb:.0f} kB — uploading to {backend} Whisper…", file=sys.stderr)

    if backend == "groq":
        response = _post_whisper_chunked(GROQ_ENDPOINT, api_key, GROQ_MODEL, audio_path)
    elif backend == "openai":
        response = _post_whisper_chunked(OPENAI_ENDPOINT, api_key, OPENAI_MODEL, audio_path)
    else:
        raise SystemExit(f"Unknown whisper backend: {backend}")

    segments = _segments_from_response(response)
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
