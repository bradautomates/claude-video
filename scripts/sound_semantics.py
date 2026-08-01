#!/usr/bin/env python3
"""Optional learned semantic understanding for short-form audio.

The local cinematic pass in media_analysis.py measures when sound changes. This
module answers the complementary question: *what* is audible and what function
does it serve? It sends only the selected audio range (never video frames) to an
explicitly enabled audio-capable model, validates the response, aligns semantic
events with local signal/visual-change evidence, and caches the normalized JSON.

Pure stdlib; no OpenAI or Google SDK is required.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import shutil
import ssl
import subprocess
import time
import urllib.error
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen


SEMANTIC_VERSION = 1
PROMPT_VERSION = 1
MAX_SEMANTIC_SECONDS = 60.0
MAX_EVENTS = 60

OPENAI_ENDPOINT = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = "gpt-audio-1.5"
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GEMINI_MODEL = "gemini-3.6-flash"

CONFIG_FILE = Path.home() / ".config" / "watch" / ".env"
PROVIDERS = {"openai", "gemini"}
CATEGORIES = {"speech", "music", "sfx", "ambience", "foley", "silence", "unknown"}
DIEGETIC_VALUES = {"likely", "unlikely", "unclear"}


def _read_dotenv(path: Path, name: str) -> str | None:
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
            if len(value) >= 2 and value[0] in ("'", '"') and value[-1] == value[0]:
                value = value[1:-1]
            return value or None
    except OSError:
        return None
    return None


def read_config(name: str) -> str | None:
    """Read a setting from process env, watch config, then cwd .env."""
    value = os.environ.get(name)
    if value and value.strip():
        return value.strip()
    for path in (CONFIG_FILE, Path.cwd() / ".env"):
        value = _read_dotenv(path, name)
        if value:
            return value
    return None


def resolve_provider(
    mode: str,
    requested: str = "auto",
) -> tuple[str | None, str | None, str]:
    """Resolve provider + key without silently expanding upload behavior.

    In ``auto`` mode, a dedicated SOUND_SEMANTICS_PROVIDER setting is required.
    An OPENAI_API_KEY used for Whisper does not implicitly opt audio into this
    additional API request. ``api`` mode is itself explicit authorization, so it
    may select an available key when --sound-provider is left on auto.
    """
    if mode == "off":
        return None, None, "semantic sound understanding disabled"
    if mode not in {"auto", "api"}:
        return None, None, f"unsupported semantic sound mode: {mode}"

    configured = (read_config("SOUND_SEMANTICS_PROVIDER") or "").lower()
    if configured not in PROVIDERS:
        configured = ""

    provider = requested.lower()
    if provider not in PROVIDERS and provider != "auto":
        return None, None, f"unsupported semantic sound provider: {requested}"

    if mode == "auto" and not configured:
        return (
            None,
            None,
            "not enabled; set SOUND_SEMANTICS_PROVIDER=openai|gemini or pass "
            "--sound-semantics api",
        )

    if provider == "auto":
        provider = configured
    if provider == "auto" or not provider:
        # The explicit `api` flag authorizes selection of an existing key.
        provider = "gemini" if read_config("GEMINI_API_KEY") else "openai"

    key_name = "GEMINI_API_KEY" if provider == "gemini" else "OPENAI_API_KEY"
    key = read_config(key_name)
    if not key:
        return None, None, f"{key_name} is missing"
    return provider, key, "enabled"


def _model_for(provider: str) -> str:
    if provider == "gemini":
        return read_config("WATCH_GEMINI_AUDIO_MODEL") or GEMINI_MODEL
    return read_config("WATCH_OPENAI_AUDIO_MODEL") or OPENAI_MODEL


def _source_identity(video_path: str) -> str:
    path = Path(video_path).resolve()
    try:
        stat = path.stat()
        return f"{path}|{stat.st_size}|{stat.st_mtime_ns}"
    except OSError:
        return str(path)


def semantic_cache_key(
    video_path: str,
    start: float,
    end: float,
    provider: str,
    model: str,
    vocab: str | None,
    prompt: str = "",
) -> str:
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16] if prompt else ""
    raw = (
        f"v{SEMANTIC_VERSION}|p{PROMPT_VERSION}|{_source_identity(video_path)}|"
        f"{start:.3f}|{end:.3f}|{provider}|{model}|{vocab or ''}|{prompt_hash}"
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def extract_semantic_audio(
    video_path: str,
    out_path: Path,
    start: float,
    end: float,
) -> Path:
    """Create a compact 16 kHz mono MP3 containing only the selected range."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not installed")
    duration = max(0.0, end - start)
    if duration <= 0:
        raise RuntimeError("semantic audio range is empty")
    out_path = out_path.with_suffix(".mp3")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    if start > 0:
        command += ["-ss", f"{start:.3f}"]
    command += [
        "-i", video_path,
        "-t", f"{duration:.3f}",
        "-vn", "-acodec", "libmp3lame", "-b:a", "64k",
        "-ar", "16000", "-ac", "1", str(out_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed: {result.stderr.strip()}")
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError("ffmpeg produced no semantic audio")
    return out_path


def _relative_points(items: list[dict], start: float, key: str = "time") -> list[dict]:
    result = []
    for item in items[:20]:
        try:
            when = float(item[key]) - start
        except (KeyError, TypeError, ValueError):
            continue
        if when >= -0.01:
            result.append({**item, key: round(max(0.0, when), 3)})
    return result


def build_prompt(
    duration: float,
    audio_analysis: dict | None = None,
    video_analysis: dict | None = None,
    transcript: str | None = None,
    vocab: str | None = None,
    source_start: float = 0.0,
) -> str:
    """Build a film-sound prompt with local measurements as timing anchors."""
    audio_analysis = audio_analysis or {}
    video_analysis = video_analysis or {}
    evidence = {
        "audio_transient_candidates": _relative_points(
            audio_analysis.get("transient_peaks") or [], source_start
        ),
        "frequency_band_onset_candidates": _relative_points(
            audio_analysis.get("band_onsets") or [], source_start
        ),
        "loudness_peaks": _relative_points(
            audio_analysis.get("energy_peaks") or [], source_start
        ),
        "silences": [
            {
                "start": round(max(0.0, float(item.get("start", 0)) - source_start), 3),
                "end": round(max(0.0, float(item.get("end", 0)) - source_start), 3),
            }
            for item in (audio_analysis.get("silences") or [])[:20]
        ],
        "visual_change_candidates": _relative_points(
            video_analysis.get("change_peaks") or [], source_start
        ),
    }
    transcript_context = (transcript or "").strip()[:5000]
    vocabulary = (vocab or "").strip()[:1000]
    return f"""You are a film sound editor analyzing a {duration:.3f}-second audio clip.

Identify audible events beyond transcription: sound effects, foley, ambience,
music, vocalizations, silence, texture, rhythm, dynamics, and transitions. Explain
how each sound functions in editing or storytelling. Separate what is directly
audible from inference. Prefer a general label (for example, "low-frequency
impact") over an unsupported exact source ("car door slam"). Speech may be
marked as speech, but do not produce a transcript.

All event times MUST be seconds relative to the start of this supplied audio,
from 0.000 through {duration:.3f}. Use sub-second boundaries. Include overlapping
events when music, ambience, speech, or effects coexist. Confidence is confidence
in the audible label, not confidence in narrative interpretation.

Local measurements are timing hints, not semantic labels:
{json.dumps(evidence, separators=(',', ':'))}

Optional transcript context (may be empty; use only to disambiguate sound):
{transcript_context or '[none]'}

Optional user vocabulary for plausible sound labels:
{vocabulary or '[none]'}

Return JSON only with this shape:
{{
  "summary": "one concise overall sound-design description",
  "soundscape": "ambience, space, layering, and dynamics",
  "music": {{"present": true, "description": "style/instruments/texture", "tempo": "perceived pace", "mood": "audible mood cues"}},
  "events": [
    {{
      "start": 0.000,
      "end": 0.200,
      "label": "concise audible event",
      "category": "speech|music|sfx|ambience|foley|silence|unknown",
      "confidence": 0.0,
      "description": "audible qualities supporting the label",
      "story_function": "editing, emotional, spatial, or narrative role",
      "diegetic": "likely|unlikely|unclear",
      "evidence": ["short audible cue"]
    }}
  ],
  "caveats": ["important uncertainty"]
}}
"""


def _post_json(url: str, payload: dict, headers: dict[str, str], timeout: int = 180) -> dict:
    body = json.dumps(payload, separators=(",", ":")).encode()
    request_headers = {
        "Content-Type": "application/json",
        "User-Agent": "watch-skill/1.0 (+semantic-audio; python-urllib)",
        **headers,
    }
    context = ssl.create_default_context()
    last_error = "request failed"
    for attempt in range(3):
        request = Request(url, data=body, headers=request_headers, method="POST")
        try:
            with urlopen(request, timeout=timeout, context=context) as response:
                text = response.read().decode("utf-8", errors="replace")
            parsed = json.loads(text)
            if not isinstance(parsed, dict):
                raise RuntimeError("provider returned a non-object response")
            return parsed
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:1000]
            except Exception:
                detail = ""
            last_error = f"HTTP {exc.code}: {detail or exc.reason}"
            if exc.code not in {408, 429, 500, 502, 503, 504}:
                break
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = str(exc)
        if attempt < 2:
            time.sleep(1.5 * (2 ** attempt))
    raise RuntimeError(f"semantic audio provider failed: {last_error}")


def _response_schema() -> dict:
    event_properties = {
        "start": {"type": "NUMBER"},
        "end": {"type": "NUMBER"},
        "label": {"type": "STRING"},
        "category": {"type": "STRING", "enum": sorted(CATEGORIES)},
        "confidence": {"type": "NUMBER"},
        "description": {"type": "STRING"},
        "story_function": {"type": "STRING"},
        "diegetic": {"type": "STRING", "enum": sorted(DIEGETIC_VALUES)},
        "evidence": {"type": "ARRAY", "items": {"type": "STRING"}},
    }
    return {
        "type": "OBJECT",
        "properties": {
            "summary": {"type": "STRING"},
            "soundscape": {"type": "STRING"},
            "music": {
                "type": "OBJECT",
                "properties": {
                    "present": {"type": "BOOLEAN"},
                    "description": {"type": "STRING"},
                    "tempo": {"type": "STRING"},
                    "mood": {"type": "STRING"},
                },
                "required": ["present", "description", "tempo", "mood"],
            },
            "events": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": event_properties,
                    "required": list(event_properties),
                },
            },
            "caveats": {"type": "ARRAY", "items": {"type": "STRING"}},
        },
        "required": ["summary", "soundscape", "music", "events", "caveats"],
    }


def call_openai(audio_path: Path, prompt: str, api_key: str, model: str) -> str:
    audio = base64.b64encode(audio_path.read_bytes()).decode()
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "input_audio", "input_audio": {"data": audio, "format": "mp3"}},
                ],
            }
        ],
    }
    data = _post_json(
        OPENAI_ENDPOINT,
        payload,
        {"Authorization": f"Bearer {api_key}"},
    )
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("OpenAI response did not contain message content") from exc
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [
            str(item.get("text") or item.get("content") or "")
            for item in content if isinstance(item, dict)
        ]
        joined = "\n".join(text for text in texts if text)
        if joined:
            return joined
    raise RuntimeError("OpenAI response contained no semantic text")


def call_gemini(audio_path: Path, prompt: str, api_key: str, model: str) -> str:
    audio = base64.b64encode(audio_path.read_bytes()).decode()
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "audio/mp3", "data": audio}},
                ],
            }
        ],
        "generation_config": {
            "temperature": 0,
            "response_format": {"text": {"mime_type": "application/json"}},
            "response_schema": _response_schema(),
        },
    }
    url = GEMINI_ENDPOINT.format(model=quote(model, safe=""))
    data = _post_json(url, payload, {"x-goog-api-key": api_key})
    try:
        parts = data["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Gemini response did not contain candidate content") from exc
    text_parts = [str(part.get("text") or "") for part in parts if isinstance(part, dict)]
    content = "\n".join(text for text in text_parts if text)
    if not content:
        raise RuntimeError("Gemini response contained no semantic text")
    return content


def extract_json_object(text: str) -> dict:
    """Extract the first JSON object, tolerating Markdown fences or preamble."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    decoder = json.JSONDecoder()
    candidates = [0] if stripped.startswith("{") else []
    candidates.extend(i for i, char in enumerate(stripped) if char == "{")
    seen: set[int] = set()
    for index in candidates:
        if index in seen:
            continue
        seen.add(index)
        try:
            value, _ = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise RuntimeError("semantic provider did not return a JSON object")


def _short_string(value: Any, limit: int = 500) -> str:
    return str(value).strip()[:limit] if value is not None else ""


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return bool(value)


def _strings(value: Any, count: int = 8, limit: int = 200) -> list[str]:
    if not isinstance(value, list):
        return []
    return [text for text in (_short_string(item, limit) for item in value[:count]) if text]


def _nearest_time(items: list[dict], target: float, tolerance: float) -> dict | None:
    candidates = []
    for item in items:
        when = _number(item.get("time"), float("nan"))
        if math.isfinite(when):
            candidates.append((abs(when - target), when, item))
    if not candidates:
        return None
    distance, when, item = min(candidates, key=lambda value: value[0])
    if distance > tolerance:
        return None
    result = {"time": round(when, 3), "offset": round(when - target, 3)}
    if "score" in item:
        result["score"] = round(_number(item.get("score")), 3)
    if "label" in item:
        result["label"] = _short_string(item.get("label"), 100)
    return result


def normalize_result(
    raw: dict,
    source_start: float,
    source_end: float,
    provider: str,
    model: str,
    audio_analysis: dict | None = None,
    video_analysis: dict | None = None,
) -> dict:
    """Validate model output and convert clip-relative times to source time."""
    duration = max(0.0, source_end - source_start)
    audio_analysis = audio_analysis or {}
    video_analysis = video_analysis or {}
    transients = (
        (audio_analysis.get("transient_peaks") or [])
        + (audio_analysis.get("band_onsets") or [])
    )
    visual_changes = video_analysis.get("change_peaks") or []
    events = []
    raw_events = raw.get("events") if isinstance(raw.get("events"), list) else []
    for item in raw_events[:MAX_EVENTS]:
        if not isinstance(item, dict):
            continue
        label = _short_string(item.get("label"), 140)
        if not label:
            continue
        relative_start = min(duration, max(0.0, _number(item.get("start"))))
        relative_end = min(duration, max(relative_start, _number(item.get("end"), relative_start)))
        category = _short_string(item.get("category"), 30).lower()
        if category not in CATEGORIES:
            category = "unknown"
        diegetic = _short_string(item.get("diegetic"), 20).lower()
        if diegetic not in DIEGETIC_VALUES:
            diegetic = "unclear"
        absolute_start = round(source_start + relative_start, 3)
        absolute_end = round(source_start + relative_end, 3)
        event = {
            "start": absolute_start,
            "end": absolute_end,
            "label": label,
            "category": category,
            "confidence": round(min(1.0, max(0.0, _number(item.get("confidence"), 0.5))), 3),
            "description": _short_string(item.get("description"), 500),
            "story_function": _short_string(item.get("story_function"), 500),
            "diegetic": diegetic,
            "evidence": _strings(item.get("evidence")),
            "grounding": {},
        }
        transient = _nearest_time(transients, absolute_start, 0.25)
        visual = _nearest_time(visual_changes, absolute_start, 0.25)
        if transient:
            event["grounding"]["audio_change"] = transient
        if visual:
            event["grounding"]["visual_change"] = visual
        events.append(event)
    events.sort(key=lambda item: (item["start"], item["end"], item["label"]))

    music_raw = raw.get("music") if isinstance(raw.get("music"), dict) else {}
    music = {
        "present": _boolean(music_raw.get("present", False)),
        "description": _short_string(music_raw.get("description"), 500),
        "tempo": _short_string(music_raw.get("tempo"), 160),
        "mood": _short_string(music_raw.get("mood"), 200),
    }
    return {
        "version": SEMANTIC_VERSION,
        "status": "ok",
        "provider": provider,
        "model": model,
        "time_base": "source_seconds",
        "range": {"start": round(source_start, 3), "end": round(source_end, 3)},
        "summary": _short_string(raw.get("summary"), 800),
        "soundscape": _short_string(raw.get("soundscape"), 800),
        "music": music,
        "events": events,
        "caveats": _strings(raw.get("caveats"), count=12, limit=300),
        "cache_hit": False,
    }


def analyze_sound_semantics(
    video_path: str,
    work_dir: Path,
    cache_root: Path,
    start: float,
    end: float,
    mode: str = "auto",
    provider: str = "auto",
    audio_analysis: dict | None = None,
    video_analysis: dict | None = None,
    transcript: str | None = None,
    vocab: str | None = None,
    use_cache: bool = True,
) -> dict:
    """Run or load semantic sound analysis for one short selected range."""
    resolved, api_key, reason = resolve_provider(mode, provider)
    if not resolved or not api_key:
        return {"status": "skipped", "reason": reason, "events": []}
    duration = max(0.0, end - start)
    if duration <= 0:
        return {"status": "error", "reason": "selected audio range is empty", "events": []}
    if duration > MAX_SEMANTIC_SECONDS + 0.001:
        return {
            "status": "skipped" if mode == "auto" else "error",
            "reason": (
                f"semantic sound analysis is limited to {MAX_SEMANTIC_SECONDS:g}s; "
                "use --start/--end to focus the range"
            ),
            "events": [],
        }

    model = _model_for(resolved)
    prompt = build_prompt(
        duration,
        audio_analysis=audio_analysis,
        video_analysis=video_analysis,
        transcript=transcript,
        vocab=vocab,
        source_start=start,
    )
    cache_key = semantic_cache_key(video_path, start, end, resolved, model, vocab, prompt)
    cache_path = cache_root / cache_key / "semantics.json"
    if use_cache and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
            if cached.get("version") == SEMANTIC_VERSION and cached.get("status") == "ok":
                cached["cache_hit"] = True
                return cached
        except (OSError, json.JSONDecodeError, AttributeError):
            pass

    work_dir.mkdir(parents=True, exist_ok=True)
    audio_path: Path | None = None
    try:
        audio_path = extract_semantic_audio(video_path, work_dir / "semantic-audio", start, end)
        if resolved == "gemini":
            response_text = call_gemini(audio_path, prompt, api_key, model)
        else:
            response_text = call_openai(audio_path, prompt, api_key, model)
        raw = extract_json_object(response_text)
        result = normalize_result(
            raw,
            start,
            end,
            resolved,
            model,
            audio_analysis=audio_analysis,
            video_analysis=video_analysis,
        )
        if use_cache:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(result, indent=2))
            except OSError:
                pass
        return result
    except Exception as exc:
        return {
            "status": "error",
            "provider": resolved,
            "model": model,
            "reason": str(exc),
            "events": [],
        }
    finally:
        if audio_path is not None:
            try:
                audio_path.unlink(missing_ok=True)
            except OSError:
                pass
