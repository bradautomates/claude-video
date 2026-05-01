#!/usr/bin/env python3
"""Transcribe a video via local whisper-cli, Groq, or OpenAI Whisper API.

Strategy: prefer a local whisper-cli (whisper.cpp) when available — free,
private, no network needed. Fall back to Groq or OpenAI API when no local
binary or model is found. Returns segments in the same shape as
transcribe.parse_vtt so the rest of the pipeline (filter_range,
format_transcript) doesn't care where the transcript came from.

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

# Common locations where whisper.cpp GGML models are stored.
_MODEL_SEARCH_DIRS: list[Path] = [
    Path.home() / ".cache" / "whisper",
]


def _find_whisper_cli() -> str | None:
    """Return the whisper-cli binary path, or None if not installed."""
    return shutil.which("whisper-cli")


def _find_whisper_model() -> Path | None:
    """Return the best available GGML model, or None.

    Priority: WHISPER_MODEL env var > common search directories > brew cellar.
    Within a directory, prefers larger files (bigger model = better quality).
    """
    env_model = os.environ.get("WHISPER_MODEL", "").strip()
    if env_model:
        p = Path(env_model).expanduser()
        return p if p.is_file() else None

    for search_dir in _MODEL_SEARCH_DIRS:
        if not search_dir.is_dir():
            continue
        candidates = [
            f for f in search_dir.iterdir()
            if f.is_file() and (f.suffix == ".bin" or "ggml" in f.name)
        ]
        if candidates:
            return max(candidates, key=lambda p: p.stat().st_size)

    # Last resort: check the brew cellar for bundled models.
    try:
        result = subprocess.run(
            ["brew", "--prefix", "whisper-cpp"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            models_dir = Path(result.stdout.strip()) / "share" / "whisper-cpp" / "models"
            if models_dir.is_dir():
                candidates = [
                    f for f in models_dir.iterdir()
                    if f.is_file() and (f.suffix == ".bin" or "ggml" in f.name)
                ]
                if candidates:
                    return max(candidates, key=lambda p: p.stat().st_size)
    except (OSError, subprocess.TimeoutExpired):
        pass

    return None


def load_backend(preferred: str | None = None) -> tuple[str, str | None] | tuple[None, None]:
    """Return (backend, api_key). Prefers local, then Groq, then OpenAI.

    For the "local" backend, api_key is None (no key needed).
    If `preferred` is set, only that backend is considered.
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

    def _read_key(name: str) -> str | None:
        value = _from_env(name)
        if not value:
            for candidate in dotenv_paths:
                value = _from_dotenv(candidate, name)
                if value:
                    break
        return value

    # --- Local whisper-cli (preferred — free, private, no network) ---
    if preferred is None or preferred == "local":
        local_disabled = False
        val = _read_key("WHISPER_LOCAL")
        if val and val.lower() == "false":
            local_disabled = True

        if not local_disabled and _find_whisper_cli() and _find_whisper_model():
            return "local", None

        if preferred == "local":
            return None, None

    # --- API backends ---
    candidates = (("GROQ_API_KEY", "groq"), ("OPENAI_API_KEY", "openai"))
    if preferred is not None:
        candidates = tuple(c for c in candidates if c[1] == preferred)

    for key_name, backend in candidates:
        value = _read_key(key_name)
        if value:
            return backend, value

    return None, None


# Backward-compatible alias.
load_api_key = load_backend


def extract_audio(video_path: str, out_path: Path, *, wav: bool = False) -> Path:
    """Extract mono 16kHz audio from a video file.

    When *wav* is True, outputs PCM s16le WAV (native format for whisper.cpp).
    Otherwise outputs 64kbps mp3 (~480 kB/min, fits any Whisper API limit).
    """
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is not installed. Install with: brew install ffmpeg")

    if wav:
        out_path = out_path.with_suffix(".wav")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if wav:
        codec_args = ["-acodec", "pcm_s16le"]
    else:
        codec_args = ["-acodec", "libmp3lame", "-b:a", "64k"]

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-i", video_path,
        "-vn",
        *codec_args,
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


def _transcribe_local(audio_path: Path, model_path: Path) -> list[dict]:
    """Run whisper-cli locally and return parsed segments."""
    whisper_bin = _find_whisper_cli()
    if not whisper_bin:
        raise SystemExit("whisper-cli not found in PATH")

    cmd = [
        whisper_bin,
        "-m", str(model_path),
        "-oj",            # JSON output (writes <input>.json alongside)
        "--no-prints",     # suppress progress chatter
        "-f", str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise SystemExit(f"whisper-cli failed: {result.stderr.strip()}")

    # whisper-cli -oj writes a .json file next to the input audio.
    json_path = Path(str(audio_path) + ".json")
    if json_path.exists():
        data = json.loads(json_path.read_text())
    elif result.stdout.strip():
        data = json.loads(result.stdout)
    else:
        raise SystemExit("whisper-cli produced no JSON output")

    return _segments_from_local_json(data)


def _segments_from_local_json(data: dict) -> list[dict]:
    """Parse whisper.cpp JSON into our [{start, end, text}, ...] format."""
    out: list[dict] = []
    for item in data.get("transcription", []):
        text = (item.get("text") or "").strip()
        if not text:
            continue
        offsets = item.get("offsets", {})
        t_from = offsets.get("from", 0) / 1000.0
        t_to = offsets.get("to", 0) / 1000.0
        out.append({
            "start": round(t_from, 2),
            "end": round(t_to, 2),
            "text": text,
        })
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
    """Run the full flow: extract audio → transcribe → parse segments.

    Returns (segments, backend_used). Raises SystemExit on any failure.
    """
    if backend is None:
        detected_backend, detected_key = load_backend()
        backend = backend or detected_backend
        api_key = api_key or detected_key

    # --- Local whisper-cli (no API key needed) ---
    if backend == "local":
        model_path = _find_whisper_model()
        if not model_path:
            raise SystemExit(
                "whisper-cli found but no GGML model available. "
                "Set WHISPER_MODEL=/path/to/ggml-model.bin or place a model in "
                "~/.cache/whisper/"
            )
        print(f"[watch] extracting audio for local whisper…", file=sys.stderr)
        audio_path = extract_audio(video_path, audio_out, wav=True)
        size_kb = audio_path.stat().st_size / 1024
        print(
            f"[watch] audio: {size_kb:.0f} kB — running whisper-cli "
            f"({model_path.name})…",
            file=sys.stderr,
        )
        segments = _transcribe_local(audio_path, model_path)
        if not segments:
            raise SystemExit("whisper-cli returned no transcript segments")
        print(f"[watch] transcribed {len(segments)} segments via local whisper-cli", file=sys.stderr)
        return segments, "local"

    # --- API backends (need a key) ---
    if api_key is None and backend is not None:
        _, detected_key = load_backend(preferred=backend)
        api_key = detected_key

    if not backend or not api_key:
        setup_py = Path(__file__).resolve().parent / "setup.py"
        raise SystemExit(
            "No Whisper backend available. Install whisper-cpp for local transcription, "
            "or set GROQ_API_KEY / OPENAI_API_KEY in the environment or "
            f"~/.config/watch/.env. Run `python3 {setup_py}` to configure."
        )

    print(f"[watch] extracting audio for Whisper ({backend})…", file=sys.stderr)
    audio_path = extract_audio(video_path, audio_out)
    size_kb = audio_path.stat().st_size / 1024
    print(f"[watch] audio: {size_kb:.0f} kB — uploading to {backend} Whisper…", file=sys.stderr)

    if backend == "groq":
        response = _post_whisper(GROQ_ENDPOINT, api_key, GROQ_MODEL, audio_path)
    elif backend == "openai":
        response = _post_whisper(OPENAI_ENDPOINT, api_key, OPENAI_MODEL, audio_path)
    else:
        raise SystemExit(f"Unknown whisper backend: {backend}")

    segments = _segments_from_response(response)
    if not segments:
        raise SystemExit("Whisper returned no transcript segments")

    print(f"[watch] transcribed {len(segments)} segments via {backend}", file=sys.stderr)
    return segments, backend


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
