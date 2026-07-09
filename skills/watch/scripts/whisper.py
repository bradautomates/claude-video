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
import math
import mimetypes
import shutil
import ssl
import subprocess
import sys
import time
import urllib.error
import uuid
from pathlib import Path
from urllib.request import Request, urlopen

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import read_trusted_setting  # noqa: E402


GROQ_ENDPOINT = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_MODEL = "whisper-large-v3"

OPENAI_ENDPOINT = "https://api.openai.com/v1/audio/transcriptions"
OPENAI_MODEL = "whisper-1"

# Both Groq's free tier and OpenAI whisper-1 cap uploads at 25 MB. We target a
# margin under that so multipart framing overhead never pushes a chunk over.
MAX_UPLOAD_BYTES = 24 * 1024 * 1024

# Local backend: whisper.cpp. Runs fully offline — no API key, and the audio
# never leaves the machine. `whisper-cli` is Homebrew's binary name; older
# builds ship it as `whisper-cpp`.
LOCAL_BACKEND = "local"
WHISPER_CPP_BINARIES = ("whisper-cli", "whisper-cpp")
LOCAL_MODEL_DIR = Path.home() / ".config" / "watch" / "models"


def find_whisper_cpp_binary() -> str | None:
    """Return a whisper.cpp CLI to run, or None if none is available.

    WHISPER_CPP_BIN overrides discovery (a name on PATH or an absolute path to
    an executable). Otherwise prefer `whisper-cli`, then the `whisper-cpp` alias.
    """
    override = read_trusted_setting("WHISPER_CPP_BIN")
    if override:
        return shutil.which(override)
    for name in WHISPER_CPP_BINARIES:
        found = shutil.which(name)
        if found:
            return found
    return None


def find_whisper_cpp_model() -> str | None:
    """Return a ggml model path for whisper.cpp, or None if none is found.

    WHISPER_CPP_MODEL wins if it points at a real file. Otherwise pick the
    first `*.bin` in ~/.config/watch/models (sorted for determinism).
    """
    override = read_trusted_setting("WHISPER_CPP_MODEL")
    if override:
        path = Path(override).expanduser()
        return str(path) if path.is_file() else None
    if LOCAL_MODEL_DIR.is_dir():
        models = sorted(LOCAL_MODEL_DIR.glob("*.bin"))
        if models:
            return str(models[0])
    return None


def plan_chunks(
    total_seconds: float,
    total_bytes: int,
    max_bytes: int = MAX_UPLOAD_BYTES,
) -> list[tuple[float, float]]:
    """Split a duration into contiguous (offset, duration) chunks under max_bytes.

    Size scales linearly with duration (constant-bitrate mono mp3), so an even
    time split yields evenly-sized chunks. Returns a single full-length chunk
    when the audio already fits.
    """
    if total_bytes <= max_bytes or total_seconds <= 0:
        return [(0.0, total_seconds)]

    n = math.ceil(total_bytes / max_bytes)
    chunk = total_seconds / n
    plan: list[tuple[float, float]] = []
    for i in range(n):
        offset = i * chunk
        # The last chunk absorbs any rounding remainder so durations sum exactly.
        duration = (total_seconds - offset) if i == n - 1 else chunk
        plan.append((round(offset, 3), round(duration, 3)))
    return plan


def load_api_key(preferred: str | None = None) -> tuple[str, str] | tuple[None, None]:
    """Return (backend, api_key). Prefers Groq, falls back to OpenAI.

    If `preferred` is "groq" or "openai", only that backend's key is considered.
    Keys come from trusted config only (environment or ~/.config/watch/.env); a
    project-local .env cannot supply one — see config.read_trusted_setting.
    """
    candidates = (("GROQ_API_KEY", "groq"), ("OPENAI_API_KEY", "openai"))
    if preferred is not None:
        candidates = tuple(c for c in candidates if c[1] == preferred)

    for key_name, backend in candidates:
        value = read_trusted_setting(key_name)
        if value:
            return backend, value

    return None, None


def resolve_backend(preferred: str | None = None) -> dict | None:
    """Pick a transcription backend and gather what it needs to run.

    Returns a descriptor {backend, api_key, binary, model} or None when the
    requested/available backend cannot run.

    preferred:
      "groq" | "openai" — that cloud backend only (needs its API key).
      "local"           — whisper.cpp only (needs a binary and a model).
      None (auto)       — a cloud key if one is set (Groq before OpenAI),
                          otherwise fall back to local whisper.cpp if available.
    """
    if preferred == LOCAL_BACKEND:
        binary = find_whisper_cpp_binary()
        model = find_whisper_cpp_model()
        if binary and model:
            return {"backend": LOCAL_BACKEND, "api_key": None, "binary": binary, "model": model}
        return None

    if preferred in ("groq", "openai"):
        backend, key = load_api_key(preferred)
        if backend and key:
            return {"backend": backend, "api_key": key, "binary": None, "model": None}
        return None

    backend, key = load_api_key()
    if backend and key:
        return {"backend": backend, "api_key": key, "binary": None, "model": None}

    binary = find_whisper_cpp_binary()
    model = find_whisper_cpp_model()
    if binary and model:
        return {"backend": LOCAL_BACKEND, "api_key": None, "binary": binary, "model": model}
    return None


def extract_audio(video_path: str, out_path: Path) -> Path:
    """Extract mono 16kHz 64kbps mp3 — ~480 kB/min, fits any Whisper limit."""
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


def extract_audio_wav(video_path: str, out_path: Path) -> Path:
    """Extract mono 16kHz s16 WAV — the format whisper.cpp reads natively.

    No upload here, so there is no size cap to respect: whisper.cpp streams the
    whole file locally.
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
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(out_path.resolve()),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"ffmpeg audio extraction failed: {result.stderr.strip()}")
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise SystemExit("ffmpeg produced no audio — video may have no audio track")
    return out_path


def audio_duration(audio_path: Path) -> float:
    """Return the duration of an audio file in seconds via ffprobe."""
    if shutil.which("ffprobe") is None:
        raise SystemExit("ffprobe is not installed. Install with: brew install ffmpeg")

    result = subprocess.run(
        [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            str(audio_path.resolve()),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(f"ffprobe failed: {result.stderr.strip()}")
    fmt = json.loads(result.stdout or "{}").get("format", {})
    return float(fmt.get("duration") or 0.0)


def split_audio(
    full_audio: Path,
    work_dir: Path,
    plan: list[tuple[float, float]],
) -> list[tuple[Path, float]]:
    """Slice full_audio into per-plan chunk files, returning (path, offset) pairs.

    Uses stream copy (`-c copy`) so there is no re-encode and no quality loss;
    mp3 frame boundaries are close enough for transcription's purposes.
    """
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is not installed. Install with: brew install ffmpeg")

    work_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[tuple[Path, float]] = []
    for index, (offset, duration) in enumerate(plan):
        out_path = work_dir / f"chunk_{index:03d}.mp3"
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-ss", f"{offset:.3f}",
            "-i", str(full_audio.resolve()),
            "-t", f"{duration:.3f}",
            "-c", "copy",
            str(out_path.resolve()),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 or not out_path.exists() or out_path.stat().st_size == 0:
            raise SystemExit(
                f"ffmpeg failed to split audio chunk {index + 1}: {result.stderr.strip()}"
            )
        chunks.append((out_path, offset))
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


def shift_segments(segments: list[dict], offset_seconds: float) -> list[dict]:
    """Return a copy of segments with start/end shifted by offset_seconds.

    Each chunk is transcribed in isolation, so Whisper returns 0-based timestamps
    per chunk; shifting by the chunk's offset stitches them into source time.
    """
    if offset_seconds == 0:
        return segments
    return [
        {
            "start": round(seg["start"] + offset_seconds, 2),
            "end": round(seg["end"] + offset_seconds, 2),
            "text": seg["text"],
        }
        for seg in segments
    ]


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


def _segments_from_whisper_cpp(data: dict) -> list[dict]:
    """Convert whisper.cpp --output-json into our {start, end, text} segments.

    whisper.cpp reports each line under `transcription` with millisecond
    `offsets` (from/to). We divide to seconds to match the caption/API shape.
    """
    out: list[dict] = []
    for item in data.get("transcription") or []:
        text = (item.get("text") or "").strip()
        if not text:
            continue
        offsets = item.get("offsets") or {}
        start_ms = float(offsets.get("from") or 0.0)
        end_ms = float(offsets.get("to") or 0.0)
        out.append({
            "start": round(start_ms / 1000.0, 2),
            "end": round(end_ms / 1000.0, 2),
            "text": text,
        })
    return out


def transcribe_local(
    audio_path: Path,
    binary: str,
    model: str,
    work_dir: Path | None = None,
) -> list[dict]:
    """Transcribe one WAV with whisper.cpp and return {start,end,text} segments.

    Fully offline: shells out to the whisper.cpp CLI with JSON output, then
    parses the sidecar JSON it writes. No network, no API key.
    """
    work_dir = work_dir or audio_path.parent
    work_dir.mkdir(parents=True, exist_ok=True)
    out_prefix = (work_dir / f"{audio_path.stem}.whisper").resolve()
    json_path = Path(f"{out_prefix}.json")

    cmd = [
        binary,
        "-m", str(Path(model).resolve()),
        "-f", str(audio_path.resolve()),
        "-oj",
        "-of", str(out_prefix),
        "-np",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"whisper.cpp failed: {result.stderr.strip()[:400]}")
    if not json_path.exists():
        raise SystemExit(f"whisper.cpp wrote no JSON output at {json_path}")
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"whisper.cpp JSON was unreadable: {exc}")
    return _segments_from_whisper_cpp(data)


def transcribe_local_video(
    video_path: str,
    audio_out: Path,
    binary: str | None = None,
    model: str | None = None,
) -> tuple[list[dict], str]:
    """Full offline flow: extract WAV → whisper.cpp → segments.

    Returns (segments, "local"). Raises SystemExit on any failure.
    """
    binary = binary or find_whisper_cpp_binary()
    if not binary:
        raise SystemExit(
            "whisper.cpp not found. Install it (macOS: `brew install whisper-cpp`) "
            "or set WHISPER_CPP_BIN to its path."
        )
    model = model or find_whisper_cpp_model()
    if not model:
        raise SystemExit(
            "No whisper.cpp model found. Put a ggml model in "
            f"{LOCAL_MODEL_DIR} or set WHISPER_CPP_MODEL. For example:\n"
            f"  mkdir -p {LOCAL_MODEL_DIR} && curl -L -o {LOCAL_MODEL_DIR}/ggml-base.en.bin \\\n"
            "    https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin"
        )

    wav_out = audio_out.with_suffix(".wav")
    print("[watch] extracting audio for whisper.cpp (local)…", file=sys.stderr)
    audio_path = extract_audio_wav(video_path, wav_out)

    print(
        f"[watch] transcribing locally with whisper.cpp ({Path(model).name})…",
        file=sys.stderr,
    )
    segments = transcribe_local(audio_path, binary, model)
    if not segments:
        raise SystemExit("whisper.cpp returned no transcript segments")

    print(f"[watch] transcribed {len(segments)} segments via local whisper.cpp", file=sys.stderr)
    return segments, LOCAL_BACKEND


def transcribe_chunks(
    chunks: list[tuple[Path, float]],
    transcribe_one,
) -> list[dict]:
    """Transcribe each chunk, shift its segments by the chunk offset, concatenate.

    A chunk that fails after its own retries is logged and skipped so one bad
    slice doesn't discard the whole transcript. Raises only if every chunk fails.
    """
    segments: list[dict] = []
    failures = 0
    for index, (path, offset) in enumerate(chunks):
        try:
            chunk_segments = transcribe_one(path)
        except SystemExit as exc:
            failures += 1
            print(
                f"[watch] chunk {index + 1}/{len(chunks)} failed — skipping ({exc})",
                file=sys.stderr,
            )
            continue
        segments.extend(shift_segments(chunk_segments, offset))
        print(
            f"[watch] chunk {index + 1}/{len(chunks)} → {len(chunk_segments)} segments",
            file=sys.stderr,
        )

    if failures == len(chunks):
        raise SystemExit("Whisper failed on every audio chunk")
    return segments


def _transcribe_file(backend: str, api_key: str, audio_path: Path) -> list[dict]:
    """Upload one audio file and return its 0-based segments."""
    if backend == "groq":
        response = _post_whisper(GROQ_ENDPOINT, api_key, GROQ_MODEL, audio_path)
    elif backend == "openai":
        response = _post_whisper(OPENAI_ENDPOINT, api_key, OPENAI_MODEL, audio_path)
    else:
        raise SystemExit(f"Unknown whisper backend: {backend}")
    return _segments_from_response(response)


def transcribe_video(
    video_path: str,
    audio_out: Path,
    backend: str | None = None,
    api_key: str | None = None,
    binary: str | None = None,
    model: str | None = None,
) -> tuple[list[dict], str]:
    """Run the full flow: extract audio → transcribe → parse segments.

    Cloud backends upload audio; the local backend runs whisper.cpp offline.
    Returns (segments, backend_used). Raises SystemExit on any failure.
    """
    if backend == LOCAL_BACKEND:
        return transcribe_local_video(video_path, audio_out, binary=binary, model=model)

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
    audio_bytes = audio_path.stat().st_size

    def transcribe_one(path: Path) -> list[dict]:
        return _transcribe_file(backend, api_key, path)

    if audio_bytes <= MAX_UPLOAD_BYTES:
        print(
            f"[watch] audio: {audio_bytes / 1024:.0f} kB — uploading to {backend} Whisper…",
            file=sys.stderr,
        )
        segments = transcribe_one(audio_path)
    else:
        duration = audio_duration(audio_path)
        plan = plan_chunks(duration, audio_bytes, MAX_UPLOAD_BYTES)
        print(
            f"[watch] audio: {audio_bytes / (1024 * 1024):.0f} MB exceeds "
            f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB — splitting into {len(plan)} chunks…",
            file=sys.stderr,
        )
        chunks = split_audio(audio_path, audio_out.parent / "chunks", plan)
        segments = transcribe_chunks(chunks, transcribe_one)

    if not segments:
        raise SystemExit("Whisper returned no transcript segments")

    print(f"[watch] transcribed {len(segments)} segments via {backend}", file=sys.stderr)
    return segments, backend


def transcribe_with_failover(
    video_path: str,
    audio_out: Path,
    choice: dict,
) -> tuple[list[dict], str]:
    """Transcribe with `choice`; if a cloud backend fails, retry locally.

    Local whisper.cpp is the safety net: when a cloud transcription attempt
    raises (bad key, rate limit, network), fall back to an installed local
    backend so the run still returns a transcript. Re-raises the original error
    when the failed backend was already local or no local backup is installed.
    """
    try:
        return transcribe_video(
            video_path,
            audio_out,
            backend=choice["backend"],
            api_key=choice.get("api_key"),
            binary=choice.get("binary"),
            model=choice.get("model"),
        )
    except SystemExit as cloud_error:
        if choice["backend"] == LOCAL_BACKEND:
            raise
        binary = find_whisper_cpp_binary()
        model = find_whisper_cpp_model()
        if not (binary and model):
            raise
        print(
            f"[watch] {choice['backend']} whisper failed ({cloud_error}); "
            "falling back to local whisper.cpp…",
            file=sys.stderr,
        )
        return transcribe_video(
            video_path,
            audio_out,
            backend=LOCAL_BACKEND,
            api_key=None,
            binary=binary,
            model=model,
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: whisper.py <video-path> [<audio-out.mp3>] [--backend groq|openai|local]", file=sys.stderr)
        raise SystemExit(2)

    video = sys.argv[1]
    audio_out = Path(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else Path("audio.mp3")
    backend_override = None
    if "--backend" in sys.argv:
        backend_override = sys.argv[sys.argv.index("--backend") + 1]

    choice = resolve_backend(backend_override)
    if not choice:
        raise SystemExit("no transcription backend available (no API key, no whisper.cpp)")
    segments, backend = transcribe_video(
        video,
        audio_out,
        backend=choice["backend"],
        api_key=choice.get("api_key"),
        binary=choice.get("binary"),
        model=choice.get("model"),
    )
    print(json.dumps({"backend": backend, "segments": segments}, indent=2))
