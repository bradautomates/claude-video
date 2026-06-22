#!/usr/bin/env python3
"""TwelveLabs Pegasus client — on-the-fly video analysis (no indexing).

This is an alternative "parser" for /watch. Instead of extracting frames and a
Whisper transcript and shipping all of it into Claude's context (expensive in
image tokens, capped by context length), we hand the video to TwelveLabs'
Pegasus video-language model and get back *text*: a timestamped transcript plus
a scene-by-scene visual walkthrough. Claude then reads a few KB of text instead
of 80-100 JPEGs.

Pegasus handles audio itself (ASR) — so this path needs no Whisper key and no
captions. One upload, one analyze task, one report.

Flow (all on TwelveLabs API v1.3, auth via `x-api-key`):
  1. Upload the file as an asset:   POST /assets   (multipart, method=direct)
     - direct upload is capped at 200 MB; watch.py chunks larger inputs so each
       piece fits.
  2. Poll the asset until ready:     GET  /assets/{id}
  3. Run on-the-fly analysis:        POST /analyze/tasks  (video.asset_id)
  4. Poll the task until ready:      GET  /analyze/tasks/{id}  -> generated text

Pure stdlib — no `pip install twelvelabs` needed, same as whisper.py.
"""
from __future__ import annotations

import io
import json
import os
import ssl
import sys
import time
import urllib.error
import uuid
from pathlib import Path
from urllib.request import Request, urlopen


API_BASE = "https://api.twelvelabs.io/v1.3"
DEFAULT_MODEL = "pegasus1.5"

# Per-model max_tokens bounds for the analyze endpoint. Out-of-range values are
# rejected by the API with an opaque 400, so we clamp client-side.
MODEL_MAX_TOKENS = {
    "pegasus1.2": (1, 4096),
    "pegasus1.5": (2048, 32768),
}


def clamp_max_tokens(model: str, max_tokens: int) -> int:
    lo, hi = MODEL_MAX_TOKENS.get(model, (1, 32768))
    clamped = max(lo, min(int(max_tokens), hi))
    if clamped != max_tokens:
        print(
            f"[watch] --tl-max-tokens {max_tokens} out of range for {model}; "
            f"clamped to {clamped} (valid {lo}-{hi})",
            file=sys.stderr,
        )
    return clamped

# Direct upload (POST /assets, method=direct) is capped at 200 MB for video.
# Leave headroom so keyframe-snapped chunks that run slightly long still fit.
MAX_DIRECT_UPLOAD_BYTES = 190 * 1024 * 1024

# Pegasus accepts 4s-2h clips; analysis below the floor is rejected.
MIN_CLIP_SECONDS = 4.0

# Polling budgets (seconds).
ASSET_POLL_TIMEOUT = 900
ANALYZE_POLL_TIMEOUT = 1800
POLL_INTERVAL = 3.0

# HTTP retry policy — mirrors whisper.py.
MAX_ATTEMPTS = 4
MAX_429_RETRIES = 3
RETRY_BASE_DELAY = 2.0

USER_AGENT = "watch-skill/1.0 (+claude-code; twelvelabs-provider)"


# --------------------------------------------------------------------------- #
# API key loading — same precedence as whisper.load_api_key.
# --------------------------------------------------------------------------- #
def load_api_key() -> str | None:
    """Return the TwelveLabs API key from env or ~/.config/watch/.env (or cwd)."""
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

    name = "TWELVELABS_API_KEY"
    value = _from_env(name)
    if value:
        return value
    for candidate in (Path.home() / ".config" / "watch" / ".env", Path.cwd() / ".env"):
        value = _from_dotenv(candidate, name)
        if value:
            return value
    return None


# --------------------------------------------------------------------------- #
# HTTP plumbing — stdlib urllib with retry/backoff, like whisper._post_whisper.
# --------------------------------------------------------------------------- #
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


def _send(request: Request, timeout: float) -> dict:
    """Issue one request with retries; parse JSON. Raises SystemExit on failure."""
    context = ssl.create_default_context()
    rate_limit_hits = 0
    last_exc: Exception | None = None
    last_detail = ""

    for attempt in range(MAX_ATTEMPTS):
        try:
            with urlopen(request, timeout=timeout, context=context) as response:
                payload = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = _read_error_body(exc)
            last_exc, last_detail = exc, detail
            if 400 <= exc.code < 500 and exc.code != 429:
                raise SystemExit(f"TwelveLabs request failed: HTTP {exc.code}{detail}")
            if exc.code == 429:
                rate_limit_hits += 1
                if rate_limit_hits >= MAX_429_RETRIES:
                    raise SystemExit(f"TwelveLabs rate limited: HTTP 429{detail}")
                delay = _retry_after(exc) or (RETRY_BASE_DELAY * (2 ** attempt) + 1)
            else:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
            if attempt < MAX_ATTEMPTS - 1:
                print(
                    f"[watch] twelvelabs HTTP {exc.code} — retrying in {delay:.1f}s "
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
                    f"[watch] twelvelabs network error ({type(exc).__name__}: {exc}) — "
                    f"retrying in {delay:.1f}s (attempt {attempt + 2}/{MAX_ATTEMPTS})",
                    file=sys.stderr,
                )
                time.sleep(delay)
            continue

        try:
            return json.loads(payload) if payload.strip() else {}
        except json.JSONDecodeError as exc:
            raise SystemExit(f"TwelveLabs returned non-JSON response: {exc}: {payload[:200]}")

    raise SystemExit(
        f"TwelveLabs request failed after {MAX_ATTEMPTS} attempts: {last_exc}{last_detail}"
    )


def _get(path: str, api_key: str, timeout: float = 60) -> dict:
    request = Request(
        f"{API_BASE}{path}",
        headers={"x-api-key": api_key, "User-Agent": USER_AGENT},
        method="GET",
    )
    return _send(request, timeout)


def _post_json(path: str, api_key: str, body: dict, timeout: float = 120) -> dict:
    request = Request(
        f"{API_BASE}{path}",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        },
        method="POST",
    )
    return _send(request, timeout)


def _safe_filename(name: str) -> str:
    """Strip characters that would break out of the Content-Disposition header
    (quotes, backslashes, CR/LF). TwelveLabs only needs the bytes, not the name."""
    cleaned = name.replace("\\", "_").replace('"', "_").replace("\r", "").replace("\n", "")
    return cleaned or "video"


def _multipart(parts: list[tuple[str, str]], file_field: str | None, file_path: Path | None) -> tuple[bytes, str]:
    """Assemble a multipart/form-data body from text fields + an optional file."""
    boundary = f"----WatchTLBoundary{uuid.uuid4().hex}"
    eol = b"\r\n"
    buf = io.BytesIO()
    for name, value in parts:
        buf.write(f"--{boundary}".encode()); buf.write(eol)
        buf.write(f'Content-Disposition: form-data; name="{name}"'.encode()); buf.write(eol)
        buf.write(eol)
        buf.write(str(value).encode()); buf.write(eol)
    if file_field and file_path is not None:
        buf.write(f"--{boundary}".encode()); buf.write(eol)
        buf.write(
            f'Content-Disposition: form-data; name="{file_field}"; filename="{_safe_filename(file_path.name)}"'.encode()
        )
        buf.write(eol)
        buf.write(b"Content-Type: application/octet-stream"); buf.write(eol)
        buf.write(eol)
        buf.write(file_path.read_bytes())
        buf.write(eol)
    buf.write(f"--{boundary}--".encode()); buf.write(eol)
    return buf.getvalue(), boundary


# --------------------------------------------------------------------------- #
# Assets — upload a file (or register a public URL) and wait until ready.
# --------------------------------------------------------------------------- #
def _asset_id(payload: dict) -> str | None:
    return payload.get("_id") or payload.get("id") or payload.get("asset_id")


def _await_asset(asset_id: str, api_key: str) -> None:
    """Poll GET /assets/{id} until status is ready (no-op if already ready)."""
    deadline = time.monotonic() + ASSET_POLL_TIMEOUT
    while True:
        info = _get(f"/assets/{asset_id}", api_key)
        status = (info.get("status") or "").lower()
        if status == "ready":
            return
        if status == "failed":
            raise SystemExit(f"TwelveLabs asset {asset_id} processing failed")
        if time.monotonic() > deadline:
            raise SystemExit(
                f"TwelveLabs asset {asset_id} not ready after {ASSET_POLL_TIMEOUT}s (status={status})"
            )
        time.sleep(POLL_INTERVAL)


def _create_asset(parts: list[tuple[str, str]], file_path: Path | None, api_key: str, timeout: float) -> str:
    file_field = "file" if file_path is not None else None
    body, boundary = _multipart(parts, file_field, file_path)
    request = Request(
        f"{API_BASE}/assets",
        data=body,
        headers={
            "x-api-key": api_key,
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "User-Agent": USER_AGENT,
        },
        method="POST",
    )
    payload = _send(request, timeout)
    asset_id = _asset_id(payload)
    if not asset_id:
        raise SystemExit(f"TwelveLabs asset create returned no id: {json.dumps(payload)[:200]}")
    if (payload.get("status") or "").lower() != "ready":
        _await_asset(asset_id, api_key)
    return asset_id


def upload_asset(video_path: str, api_key: str) -> str:
    """Direct-upload a local file as an asset; return its id once ready.

    The tool always has a local file at this point (yt-dlp download or a local
    path), and honoring ">30 min ⇒ chunk" requires cutting that file, so we
    always upload rather than passing a source URL to TwelveLabs.
    """
    path = Path(video_path).resolve()
    size = path.stat().st_size
    if size > MAX_DIRECT_UPLOAD_BYTES:
        raise SystemExit(
            f"File {path.name} is {size / 1e6:.0f} MB — over the {MAX_DIRECT_UPLOAD_BYTES / 1e6:.0f} MB "
            "direct-upload limit. watch.py should have chunked it; pass a smaller --chunk-minutes."
        )
    print(f"[watch] uploading {path.name} ({size / 1e6:.0f} MB) to TwelveLabs…", file=sys.stderr)
    return _create_asset([("method", "direct")], path, api_key, timeout=600)


# --------------------------------------------------------------------------- #
# Analysis — on-the-fly Pegasus generation against an uploaded asset.
# --------------------------------------------------------------------------- #
def _task_id(payload: dict) -> str | None:
    return payload.get("task_id") or payload.get("_id") or payload.get("id")


def _extract_text(result: object) -> str:
    """Pull the generated text out of an analyze task result (shape-tolerant)."""
    if result is None:
        return ""
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, dict):
        for key in ("data", "text", "analysis", "summary", "generated_text", "output"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        # time_based_metadata / segment shapes: stitch any text-bearing fields.
        for key in ("segments", "results", "data"):
            seq = result.get(key)
            if isinstance(seq, list):
                parts = []
                for item in seq:
                    if isinstance(item, dict):
                        txt = item.get("text") or item.get("value") or item.get("data")
                        if isinstance(txt, str) and txt.strip():
                            parts.append(txt.strip())
                if parts:
                    return "\n".join(parts)
        return json.dumps(result, ensure_ascii=False)
    return str(result)


def analyze_asset(
    asset_id: str,
    prompt: str,
    api_key: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 16384,
) -> dict:
    """Run POST /analyze/tasks against an asset, poll to ready, return text + meta."""
    body = {
        "video": {"type": "asset_id", "asset_id": asset_id},
        "model_name": model,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": clamp_max_tokens(model, max_tokens),
    }
    created = _post_json("/analyze/tasks", api_key, body)
    task_id = _task_id(created)
    if not task_id:
        raise SystemExit(f"TwelveLabs analyze task returned no id: {json.dumps(created)[:200]}")

    print(f"[watch] pegasus analyzing (task {task_id})…", file=sys.stderr)
    deadline = time.monotonic() + ANALYZE_POLL_TIMEOUT
    while True:
        info = _get(f"/analyze/tasks/{task_id}", api_key)
        status = (info.get("status") or "").lower()
        if status == "ready":
            text = _extract_text(info.get("result"))
            if not text:
                raise SystemExit(f"TwelveLabs task {task_id} ready but produced no text")
            usage = info.get("usage")
            if usage is None and isinstance(info.get("result"), dict):
                usage = info["result"].get("usage")
            return {"text": text, "task_id": task_id, "model": model, "usage": usage}
        if status == "failed":
            raise SystemExit(
                f"TwelveLabs analyze task {task_id} failed: {json.dumps(info)[:300]}"
            )
        if time.monotonic() > deadline:
            raise SystemExit(
                f"TwelveLabs analyze task {task_id} not ready after {ANALYZE_POLL_TIMEOUT}s "
                f"(status={status})"
            )
        time.sleep(POLL_INTERVAL)


def analyze_file(
    video_path: str,
    prompt: str,
    api_key: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 16384,
) -> dict:
    """Upload a local file and analyze it. Returns the analyze_asset result dict."""
    asset_id = upload_asset(video_path, api_key)
    result = analyze_asset(asset_id, prompt, api_key, model, temperature, max_tokens)
    result["asset_id"] = asset_id
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: twelvelabs.py <video-path> [prompt]", file=sys.stderr)
        raise SystemExit(2)
    key = load_api_key()
    if not key:
        raise SystemExit("No TWELVELABS_API_KEY found (env or ~/.config/watch/.env).")
    user_prompt = sys.argv[2] if len(sys.argv) > 2 else "Describe what happens in this video with timestamps."
    out = analyze_file(sys.argv[1], user_prompt, key)
    print(json.dumps({k: v for k, v in out.items() if k != "text"}, indent=2))
    print()
    print(out["text"])
