#!/usr/bin/env python3
"""Transcribe a video locally (whisper.cpp / mlx) or via Groq / OpenAI Whisper API.

Strategy: extract audio (mono 16kHz), then run it through the first available
backend in priority order — local → Groq → OpenAI. Returns segments in the same
shape as transcribe.parse_vtt so the rest of the pipeline (filter_range,
format_transcript) doesn't care where the transcript came from.

Pure stdlib — local backends are external binaries (like ffmpeg/yt-dlp), and the
cloud clients are hand-rolled multipart, so there's no `pip install` requirement.
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

# Both Groq's free tier and OpenAI whisper-1 cap uploads at 25 MB. We target a
# margin under that so multipart framing overhead never pushes a chunk over.
MAX_UPLOAD_BYTES = 24 * 1024 * 1024


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


# --- Local backends (auto-detected, no network, no API key) -------------------
# Two local engines are supported, in priority order:
#   1. whisper.cpp — needs a `whisper-cli` (or `whisper-cpp`/`main`) binary AND a
#      ggml-*.bin model on disk. Metal-accelerated on Apple Silicon.
#   2. mlx_whisper / openai-whisper — Python CLIs that download their own model.
# Set WHISPER_MODEL to point whisper.cpp at a specific ggml model file.
WHISPER_CPP_BINS = ("whisper-cli", "whisper-cpp", "main")
LOCAL_MODEL_DIRS = (
    "/opt/homebrew/share/whisper-cpp",
    "/usr/local/share/whisper-cpp",
    str(Path.home() / ".cache" / "whisper"),
)
# Preferred ggml models, best-first. Full large-v3-turbo is the default: fast and
# high-quality. Quantized variants are disk-frugal fallbacks; full large-v3 is the
# max-accuracy option (reach it with --accurate / --model large-v3, or WHISPER_MODEL).
LOCAL_MODEL_NAMES = (
    "ggml-large-v3-turbo.bin",
    "ggml-large-v3-turbo-q5_0.bin",
    "ggml-large-v3.bin",
    "ggml-large-v3-q5_0.bin",
    "ggml-medium.bin",
    "ggml-small.bin",
    "ggml-base.bin",
)

# Friendly names for --model / --accurate, mapped to ggml filenames.
MODEL_ALIASES = {
    "turbo": "ggml-large-v3-turbo.bin",
    "large-v3-turbo": "ggml-large-v3-turbo.bin",
    "accurate": "ggml-large-v3.bin",
    "large-v3": "ggml-large-v3.bin",
    "medium": "ggml-medium.bin",
    "small": "ggml-small.bin",
    "base": "ggml-base.bin",
}


def _find_ggml_model() -> str | None:
    """Locate a whisper.cpp ggml model: WHISPER_MODEL override, then known dirs."""
    override = os.environ.get("WHISPER_MODEL")
    if override and Path(override).expanduser().exists():
        return str(Path(override).expanduser())
    for directory in LOCAL_MODEL_DIRS:
        for name in LOCAL_MODEL_NAMES:
            candidate = Path(directory) / name
            if candidate.exists():
                return str(candidate)
    return None


def resolve_model_alias(name: str) -> str:
    """Map a friendly model name to a whisper.cpp ggml model path.

    Accepts an alias (turbo, large-v3/accurate, medium, small, base), a bare
    ggml filename, or a path to a `.bin`. Returns the resolved path; raises
    SystemExit if the model file isn't found in the known dirs.
    """
    p = Path(name).expanduser()
    if p.suffix == ".bin" and p.exists():
        return str(p)
    fname = MODEL_ALIASES.get(name.lower()) or (name if name.endswith(".bin") else f"ggml-{name}.bin")
    for directory in LOCAL_MODEL_DIRS:
        candidate = Path(directory) / fname
        if candidate.exists():
            return str(candidate)
    raise SystemExit(
        f"whisper model '{name}' not found (looked for {fname} in: "
        + ", ".join(LOCAL_MODEL_DIRS) + "). Download it there first."
    )


def detect_local_engine(model_override: str | None = None) -> dict | None:
    """Return {engine, bin, model} for an available local backend, else None.

    Priority: whisper.cpp (binary + ggml model) → mlx_whisper → openai-whisper.
    The Python CLIs are detected by name on PATH; they fetch their own models.
    Set WATCH_DISABLE_LOCAL_WHISPER to skip detection (forces cloud backends).
    """
    if os.environ.get("WATCH_DISABLE_LOCAL_WHISPER"):
        return None
    cpp_bin = next((shutil.which(b) for b in WHISPER_CPP_BINS if shutil.which(b)), None)
    model = model_override or _find_ggml_model()
    if cpp_bin and model:
        return {"engine": "whisper.cpp", "bin": cpp_bin, "model": model}

    if shutil.which("mlx_whisper"):
        return {
            "engine": "mlx",
            "bin": shutil.which("mlx_whisper"),
            "model": model_override or "mlx-community/whisper-large-v3-turbo",
        }
    if shutil.which("whisper"):  # openai-whisper CLI
        return {
            "engine": "openai-whisper",
            "bin": shutil.which("whisper"),
            "model": model_override or "large-v3-turbo",
        }
    return None


def _have_local_whisper() -> bool:
    return detect_local_engine() is not None


def load_api_key(preferred: str | None = None) -> tuple[str, str] | tuple[None, None]:
    """Resolve a transcription backend. Priority: local → Groq → OpenAI.

    Returns (backend, credential). For "local" the credential is the engine label
    (model path or model id — a sentinel; transcribe_video re-detects the engine).
    For "groq"/"openai" it's the API key. If `preferred` names a backend, only
    that one is considered. Returns (None, None) if nothing is available.
    """
    if preferred == "local" or (preferred is None and _have_local_whisper()):
        engine = detect_local_engine()
        if engine:
            return "local", (engine.get("model") or engine["engine"])
        if preferred == "local":
            return None, None

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
    """Extract mono 16kHz audio. `.wav` → PCM s16le (whisper.cpp's native input);
    anything else → 64kbps mp3 (~480 kB/min, fits any cloud Whisper upload limit).
    """
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is not installed. Install with: brew install ffmpeg")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".wav":
        codec = ["-acodec", "pcm_s16le"]
    else:
        codec = ["-acodec", "libmp3lame", "-b:a", "64k"]
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-i", str(Path(video_path).resolve()),
        "-vn",
        *codec,
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
            "No Whisper backend available. Install whisper.cpp locally "
            "(`brew install whisper-cpp` + a ggml model), or set GROQ_API_KEY / "
            "OPENAI_API_KEY in the environment or in ~/.config/watch/.env. "
            f"Run `python3 {setup_py}` to configure."
        )

    if backend == "local":
        engine = detect_local_engine()
        if not engine:
            raise SystemExit(
                "local Whisper requested but no engine found — install `whisper-cpp` "
                "+ a ggml model, or `pip install mlx-whisper`."
            )
        # whisper.cpp wants 16kHz WAV; the Python CLIs accept anything via ffmpeg.
        # No upload cap locally, so the chunking path below doesn't apply.
        wav_out = audio_out.with_suffix(".wav") if engine["engine"] == "whisper.cpp" else audio_out
        print(f"[watch] extracting audio for local Whisper ({engine['engine']})…", file=sys.stderr)
        audio_path = extract_audio(video_path, wav_out)
        segments = _transcribe_local(audio_path, engine)
    else:
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


def _transcribe_local(audio_path: Path, engine: dict) -> list[dict]:
    """Dispatch to the detected local engine. Returns {start, end, text} segments."""
    if engine["engine"] == "whisper.cpp":
        return _transcribe_whisper_cpp(audio_path, engine["bin"], engine["model"])
    return _transcribe_whisper_cli(audio_path, engine["bin"], engine["model"], engine["engine"])


def _transcribe_whisper_cpp(audio_path: Path, binary: str, model_path: str) -> list[dict]:
    """Run whisper.cpp (`whisper-cli`). Metal-accelerated on Apple Silicon."""
    out_base = audio_path.with_suffix("")
    cmd = [
        binary,
        "-m", model_path,
        "-oj",                       # write JSON
        "-of", str(out_base),        # output prefix (-> <out_base>.json)
        "-l", "auto",                # auto-detect language
        str(audio_path),
    ]
    print(f"[watch] running local whisper.cpp ({Path(model_path).name})…", file=sys.stderr)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"whisper-cli failed: {result.stderr.strip()[-400:]}")

    json_path = Path(f"{out_base}.json")
    if not json_path.exists():
        raise SystemExit(f"whisper-cli produced no JSON at {json_path}")

    data = json.loads(json_path.read_text(encoding="utf-8"))
    out: list[dict] = []
    for seg in data.get("transcription") or []:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        offsets = seg.get("offsets") or {}
        out.append({
            "start": round((offsets.get("from") or 0) / 1000.0, 2),  # ms → s
            "end": round((offsets.get("to") or 0) / 1000.0, 2),
            "text": text,
        })
    return out


def _transcribe_whisper_cli(audio_path: Path, binary: str, model: str, engine: str) -> list[dict]:
    """Run an openai-whisper-shaped CLI (`mlx_whisper` or `whisper`) → JSON segments.

    Both write `<audio-stem>.json` (verbose schema: {segments:[{start,end,text}]}).
    Flag spelling differs: openai-whisper uses underscores, mlx_whisper hyphens.
    """
    out_dir = audio_path.parent
    if engine == "mlx":
        flags = ["--model", model, "--output-dir", str(out_dir), "--output-format", "json"]
    else:  # openai-whisper
        flags = ["--model", model, "--output_dir", str(out_dir), "--output_format", "json", "--task", "transcribe"]
    cmd = [binary, str(audio_path), *flags]
    print(f"[watch] running local {engine} ({model})…", file=sys.stderr)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"{engine} failed: {result.stderr.strip()[-400:]}")

    json_path = audio_path.with_suffix(".json")
    if not json_path.exists():
        matches = sorted(out_dir.glob(f"{audio_path.stem}*.json"))
        if not matches:
            raise SystemExit(f"{engine} produced no JSON in {out_dir}")
        json_path = matches[0]
    return _segments_from_response(json.loads(json_path.read_text(encoding="utf-8")))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: whisper.py <video-path> [<audio-out.mp3>] [--backend local|groq|openai]", file=sys.stderr)
        raise SystemExit(2)

    video = sys.argv[1]
    audio_out = Path(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else Path("audio.mp3")
    backend_override = None
    if "--backend" in sys.argv:
        backend_override = sys.argv[sys.argv.index("--backend") + 1]

    segments, backend = transcribe_video(video, audio_out, backend=backend_override)
    print(json.dumps({"backend": backend, "segments": segments}, indent=2))
