#!/usr/bin/env python3
"""Transcribe a video via Groq or OpenAI Whisper API.

Strategy: extract audio (mono 16kHz, Opus for Groq / MP3 for OpenAI — tiny
payload), upload to whichever API has a key. Returns segments in the same shape as transcribe.parse_vtt so
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

# Per-backend upload encoding. Opus at 24 kbps mono is transparent to Whisper
# for 16 kHz speech and ~2.7x smaller than 64 kbps MP3 (verified against Groq
# on real speech: identical transcript). OpenAI keeps MP3 — its Opus/ogg
# acceptance is unverified here. bytes_per_second sizes the upload budget math.
AUDIO_FORMATS = {
    "groq": {"suffix": ".ogg", "codec": ["-c:a", "libopus", "-b:a", "24k"], "bytes_per_second": 3000},
    "openai": {"suffix": ".mp3", "codec": ["-acodec", "libmp3lame", "-b:a", "64k"], "bytes_per_second": 8000},
}

# Chunk duration derives from the encoded bitrate against a safe upload budget
# (the APIs cap files at 25 MB) instead of a fixed 600s — most content becomes
# ONE seamless upload with zero stitch seams. The hour cap bounds how much of
# the per-hour audio quota a single failed-and-retried request can re-bill,
# and keeps server-side processing well inside the request timeout.
SAFE_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_CHUNK_SECONDS = 3600.0

# Failure recovery is bisection, not pre-chunking: when an upload fails after
# in-request retries, the span splits in half and each half retries
# independently, recursively down to a floor. The common path pays zero seams;
# the failure path isolates the bad region instead of losing the whole span.
# Depth-capped so a persistent failure can't cascade into unbounded re-billing
# (worst case ≤ MAX_BISECT_DEPTH extra passes over the failing span).
BISECT_FLOOR_SECONDS = 120.0
MAX_BISECT_DEPTH = 2

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
CACHE_VERSION = 3
CACHE_DIR = Path.home() / ".cache" / "watch" / "chunks"


def _max_chunk_seconds(backend: str) -> float:
    """Longest span whose encoded audio stays inside the upload budget."""
    fmt = AUDIO_FORMATS.get(backend) or AUDIO_FORMATS["openai"]
    return min(MAX_CHUNK_SECONDS, SAFE_UPLOAD_BYTES / fmt["bytes_per_second"])


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
    backend: str = "openai",
) -> Path:
    """Extract mono 16kHz speech audio in the backend's upload format
    (Opus ~180 kB/min for Groq, MP3 ~480 kB/min for OpenAI). The returned
    path carries the format's suffix — it may differ from `out_path`.

    When start_seconds/end_seconds are set, only that window is extracted —
    the rest never reaches Whisper. This keeps focused-mode runs cheap
    (smaller upload, less quota burn) and matches the frame extraction scope.
    """
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is not installed. Install with: brew install ffmpeg")

    fmt = AUDIO_FORMATS.get(backend) or AUDIO_FORMATS["openai"]
    out_path = out_path.with_suffix(fmt["suffix"])
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
        *fmt["codec"],
        "-ar", "16000",
        "-ac", "1",
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
    audio_path = extract_audio(
        video_path, audio_path,
        start_seconds=win_start if win_start > 0 else None,
        end_seconds=win_end,
        backend=backend,
    )
    size_kb = audio_path.stat().st_size / 1024
    print(f"[watch] {label}: {size_kb:.0f} kB — uploading…", file=sys.stderr)

    try:
        response = _post_for_backend(backend, api_key, audio_path, prompt)
    finally:
        try:
            audio_path.unlink()
        except OSError:
            pass
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


def _stitch(left: list[dict], right: list[dict], boundary: float) -> tuple[list[dict], list[dict]]:
    """Split ownership of an overlap region at `boundary`: both sides heard
    it, so each segment goes to the side its MIDPOINT falls on. Midpoints
    rather than starts — a segment straddling the boundary survives on
    exactly one side, so coverage can never gap (at worst a couple of
    seconds of speech appear on both sides when segment cuts align badly,
    which beats losing them)."""
    keep_left = [s for s in left if (s["start"] + s["end"]) / 2 < boundary]
    keep_right = [s for s in right if (s["start"] + s["end"]) / 2 >= boundary]
    return keep_left, keep_right


def _transcribe_span(
    video_path: str,
    audio_dir: Path,
    backend: str,
    api_key: str,
    start: float,
    end: float,
    vocab: str | None,
    incoming_tail: str | None,
    depth: int = 0,
) -> tuple[list[dict], list[tuple[float, float, str]], str | None]:
    """Transcribe [start, end), bisecting on failure.

    A failed upload (after _post_whisper's own in-request retries) splits the
    span at its midpoint — with CHUNK_OVERLAP_SECONDS shared across the cut so
    the seam can be stitched — and retries each half, recursively down to
    BISECT_FLOOR_SECONDS / MAX_BISECT_DEPTH. Rate-limit failures (429) are
    never bisected: more, smaller requests only digs the hole deeper.

    Returns (segments, failures, tail) where `tail` is the last
    PROMPT_TAIL_CHARS of the span's text for the next span's continuity
    prompt, or None if the span's end is a failure hole.
    """
    label = f"span {start:.0f}s-{end:.0f}s" + (f" [bisect {depth}]" if depth else "")
    audio_path = audio_dir / f"span_{int(start)}_{int(end)}.tmp"
    try:
        segments = _transcribe_window(
            video_path, audio_path, backend, api_key, start, end, label,
            prompt=_combine_prompt(vocab, incoming_tail),
        )
        text = " ".join(s["text"] for s in segments)
        return segments, [], (text[-PROMPT_TAIL_CHARS:] if text else None)
    except SystemExit as exc:
        reason = str(exc)
        if depth >= MAX_BISECT_DEPTH or (end - start) <= BISECT_FLOOR_SECONDS or "429" in reason:
            print(f"[watch] {label}: FAILED — {reason}", file=sys.stderr)
            return [], [(start, end, reason)], None

        mid = (start + end) / 2
        half_ov = CHUNK_OVERLAP_SECONDS / 2
        print(f"[watch] {label}: failed ({reason[:80]}) — bisecting", file=sys.stderr)
        left_segs, left_fails, left_tail = _transcribe_span(
            video_path, audio_dir, backend, api_key,
            start, min(end, mid + half_ov), vocab, incoming_tail, depth + 1,
        )
        right_segs, right_fails, right_tail = _transcribe_span(
            video_path, audio_dir, backend, api_key,
            max(start, mid - half_ov), end, vocab, left_tail, depth + 1,
        )
        left_segs, right_segs = _stitch(left_segs, right_segs, mid)
        return left_segs + right_segs, left_fails + right_fails, right_tail


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

    Upload strategy: the chunk duration is derived from the encoded bitrate
    against the 25 MB API cap, so most content goes up as ONE seamless
    request. Only audio longer than that is pre-split into overlapping
    windows stitched at overlap midpoints, each primed with the previous
    window's text tail. Failures recover by bisection (see _transcribe_span)
    — a bad region is isolated and reported as a (start, end, reason) tuple
    while the rest of the transcript survives. Total failure (no audio
    transcribed) raises SystemExit.

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

    max_chunk = _max_chunk_seconds(backend)

    # Common path: everything fits one upload — zero seams, one request.
    # Bisection inside _transcribe_span still recovers partial transcripts
    # if that single upload fails.
    if eff_duration <= max_chunk:
        segments, failures, _tail = _transcribe_span(
            video_path, audio_out.parent, backend, api_key, eff_start, eff_end, vocab, None,
        )
        if not segments:
            details = "; ".join(f"{s:.0f}-{e:.0f}: {r[:120]}" for s, e, r in failures)
            raise SystemExit(details or "Whisper returned no transcript segments")
        print(f"[watch] transcribed {len(segments)} segments via {backend}", file=sys.stderr)
        return segments, backend, failures

    # Pre-split only when a single upload would blow the size budget.
    windows = _chunk_windows(eff_start, eff_end, max_chunk)
    print(
        f"[watch] {eff_duration:.0f}s exceeds the {max_chunk:.0f}s upload budget — "
        f"splitting into {len(windows)} windows ({CHUNK_OVERLAP_SECONDS:.0f}s overlap)",
        file=sys.stderr,
    )

    all_segments: list[dict] = []
    failures: list[tuple[float, float, str]] = []
    prev_tail: str | None = None
    prev_produced = False

    for i, (win_start, win_end) in enumerate(windows, 1):
        win_segments, win_failures, win_tail = _transcribe_span(
            video_path, audio_out.parent, backend, api_key,
            win_start, win_end, vocab, prev_tail,
        )
        if i > 1 and prev_produced and win_segments:
            boundary = win_start + CHUNK_OVERLAP_SECONDS / 2
            all_segments, win_segments = _stitch(all_segments, win_segments, boundary)
        all_segments.extend(win_segments)
        failures.extend(win_failures)
        prev_tail = win_tail
        prev_produced = bool(win_segments)

    if not all_segments:
        details = "; ".join(f"{s:.0f}-{e:.0f}: {r[:120]}" for s, e, r in failures)
        raise SystemExit(f"All {len(windows)} windows failed — {details}")

    print(
        f"[watch] transcribed {len(all_segments)} segments via {backend}"
        + (f" ({len(failures)} failed region(s))" if failures else ""),
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
