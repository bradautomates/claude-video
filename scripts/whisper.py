#!/usr/bin/env python3
"""Transcribe a video via Groq, OpenAI, or a local Whisper CLI.

Strategy: extract audio (mono 16kHz mp3, tiny payload), then either upload to
whichever cloud API has a key, or shell out to a local Whisper binary
(mlx_whisper on Apple Silicon, openai-whisper, or any drop-in that accepts
`--output-format json --output-dir`). Returns segments in the same shape as
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

# Local backend — searched in order; first present binary wins. mlx_whisper is
# Apple-Silicon-optimised and ships with the same CLI shape as openai-whisper,
# so a single call site handles both. Override with WATCH_LOCAL_WHISPER_BIN.
LOCAL_BIN_CANDIDATES = ("mlx_whisper", "whisper")
LOCAL_DEFAULT_MODEL_MLX = "mlx-community/whisper-large-v3-turbo"
LOCAL_DEFAULT_MODEL_OPENAI = "large-v3"


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


def _read_setting(name: str) -> str | None:
    """Read a config value from env first, then ~/.config/watch/.env, then ./.env."""
    value = _from_env(name)
    if value:
        return value
    for candidate in (Path.home() / ".config" / "watch" / ".env", Path.cwd() / ".env"):
        value = _from_dotenv(candidate, name)
        if value:
            return value
    return None


def _resolve_local_bin() -> str | None:
    """Return absolute path to a usable local Whisper binary, or None.

    Honours WATCH_LOCAL_WHISPER_BIN as an override, otherwise probes
    LOCAL_BIN_CANDIDATES in order.
    """
    override = _read_setting("WATCH_LOCAL_WHISPER_BIN")
    if override:
        path = shutil.which(override) or (override if Path(override).exists() else None)
        if path:
            return path
        return None
    for name in LOCAL_BIN_CANDIDATES:
        path = shutil.which(name)
        if path:
            return path
    return None


def load_api_key(preferred: str | None = None) -> tuple[str, str] | tuple[None, None]:
    """Return (backend, api_key). Prefers Groq, falls back to OpenAI.

    If `preferred` is "groq" or "openai", only that backend's key is considered.
    Local backend is handled by `resolve_backend`; this helper stays focused on
    the cloud-key flow so callers that only care about API keys don't change.
    """
    candidates = (("GROQ_API_KEY", "groq"), ("OPENAI_API_KEY", "openai"))
    if preferred is not None:
        candidates = tuple(c for c in candidates if c[1] == preferred)

    for key_name, backend in candidates:
        value = _read_setting(key_name)
        if value:
            return backend, value

    return None, None


def resolve_backend(preferred: str | None = None) -> tuple[str | None, str | None]:
    """Pick a Whisper backend.

    Returns (backend, credential):
      - ("groq",   api_key)
      - ("openai", api_key)
      - ("local",  bin_path)
      - (None, None)  no backend available

    Resolution order:
      1. Explicit `preferred` argument (from --whisper).
      2. WATCH_WHISPER_BACKEND env / .env (groq | openai | local).
      3. Cloud key auto-detect (Groq → OpenAI).
      4. Local binary auto-detect (mlx_whisper → whisper).
    """
    if preferred == "local":
        bin_path = _resolve_local_bin()
        return ("local", bin_path) if bin_path else (None, None)
    if preferred in ("groq", "openai"):
        backend, key = load_api_key(preferred)
        return (backend, key) if backend else (None, None)

    env_pref = _read_setting("WATCH_WHISPER_BACKEND")
    if env_pref:
        env_pref = env_pref.lower().strip()
        if env_pref in ("groq", "openai", "local"):
            return resolve_backend(env_pref)

    backend, key = load_api_key()
    if backend:
        return backend, key

    bin_path = _resolve_local_bin()
    if bin_path:
        return "local", bin_path

    return None, None


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


def _default_local_model(bin_path: str) -> str:
    """Pick a sensible default model id for the detected binary."""
    name = Path(bin_path).name
    if name == "mlx_whisper":
        return LOCAL_DEFAULT_MODEL_MLX
    return LOCAL_DEFAULT_MODEL_OPENAI


def _post_local(bin_path: str, audio_path: Path, work_dir: Path) -> dict:
    """Run a local Whisper binary and parse its JSON output.

    Both mlx_whisper and openai-whisper share a CLI shape:
      <bin> <audio> --output-format json --output-dir <dir> [--model <id>]

    They write `<audio_stem>.json` into <dir> with the same {text, segments}
    shape returned by the Whisper API. We parse that and reuse
    `_segments_from_response`.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    model = _read_setting("WATCH_LOCAL_WHISPER_MODEL") or _default_local_model(bin_path)

    cmd = [
        bin_path,
        str(audio_path.resolve()),
        "--output-format", "json",
        "--output-dir", str(work_dir.resolve()),
        "--model", model,
    ]
    print(
        f"[watch] running local whisper: {Path(bin_path).name} (model={model})…",
        file=sys.stderr,
    )
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "").strip().splitlines()[-5:]
        raise SystemExit(
            f"Local whisper ({Path(bin_path).name}) failed (exit {result.returncode}): "
            + " | ".join(tail)
        )

    json_path = work_dir / f"{audio_path.stem}.json"
    if not json_path.exists():
        # Some forks write the file using the resolved-symlink stem; fall back
        # to the first .json in work_dir.
        candidates = sorted(work_dir.glob("*.json"))
        if not candidates:
            raise SystemExit(f"Local whisper produced no JSON in {work_dir}")
        json_path = candidates[0]

    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Local whisper JSON parse failed: {exc}")


def transcribe_video(
    video_path: str,
    audio_out: Path,
    backend: str | None = None,
    api_key: str | None = None,
) -> tuple[list[dict], str]:
    """Run the full flow: extract audio → transcribe → parse segments.

    `api_key` carries the cloud API key for groq/openai backends, and the
    absolute path to the local Whisper binary for backend="local". Callers can
    pass `None` to trigger auto-resolution via `resolve_backend`.

    Returns (segments, backend_used). Raises SystemExit on any failure.
    """
    if backend is None or api_key is None:
        detected_backend, detected_credential = resolve_backend(backend)
        backend = backend or detected_backend
        api_key = api_key or detected_credential

    if not backend or not api_key:
        setup_py = Path(__file__).resolve().parent / "setup.py"
        raise SystemExit(
            "No Whisper backend available. Set GROQ_API_KEY (preferred) or "
            "OPENAI_API_KEY for cloud, or install mlx_whisper / openai-whisper "
            "for the local backend. "
            f"Run `python3 {setup_py}` to configure."
        )

    print(f"[watch] extracting audio for Whisper ({backend})…", file=sys.stderr)
    audio_path = extract_audio(video_path, audio_out)
    size_kb = audio_path.stat().st_size / 1024
    print(f"[watch] audio: {size_kb:.0f} kB — transcribing via {backend}…", file=sys.stderr)

    if backend == "groq":
        response = _post_whisper(GROQ_ENDPOINT, api_key, GROQ_MODEL, audio_path)
    elif backend == "openai":
        response = _post_whisper(OPENAI_ENDPOINT, api_key, OPENAI_MODEL, audio_path)
    elif backend == "local":
        response = _post_local(api_key, audio_path, audio_path.parent / "local-whisper")
    else:
        raise SystemExit(f"Unknown whisper backend: {backend}")

    segments = _segments_from_response(response)
    if not segments:
        raise SystemExit("Whisper returned no transcript segments")

    print(f"[watch] transcribed {len(segments)} segments via {backend}", file=sys.stderr)
    return segments, backend


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "usage: whisper.py <video-path> [<audio-out.mp3>] "
            "[--backend groq|openai|local]",
            file=sys.stderr,
        )
        raise SystemExit(2)

    video = sys.argv[1]
    audio_out = Path(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else Path("audio.mp3")
    backend_override = None
    if "--backend" in sys.argv:
        backend_override = sys.argv[sys.argv.index("--backend") + 1]

    segments, backend = transcribe_video(video, audio_out, backend=backend_override)
    print(json.dumps({"backend": backend, "segments": segments}, indent=2))
