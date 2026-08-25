#!/usr/bin/env python3
"""Transcribe audio on-device with faster-whisper — no API, no key, no upload.

Strategy: reuse the same mono 16 kHz mp3 that the API path extracts, and run it
through faster-whisper (CTranslate2). Returns segments in the same
``{start, end, text}`` shape as :func:`whisper._segments_from_response` and
:func:`transcribe.parse_vtt`, so nothing downstream cares where the transcript
came from.

Why faster-whisper rather than openai-whisper: it is a CTranslate2 reimplementation,
so it runs several times faster at the same accuracy, has no torch dependency
(~2.5 GB saved), and quantizes to int8 so ``large-v3`` fits on a 4 GB GPU or a
laptop CPU. Model weights download on first use to the Hugging Face cache; later loads
still make a revision check against huggingface.co unless HF_HUB_OFFLINE=1.

Nothing leaves the machine: no network call at transcription time, no key, no upload.
"""
from __future__ import annotations

import ctypes
import glob
import os
import site
import sys
from pathlib import Path


# Default model. Overridable via WATCH_WHISPER_MODEL — accepts a faster-whisper
# size alias ("tiny", "base", "small", "medium", "large-v3", "distil-large-v3"),
# a Hugging Face repo id, or a path to a local CTranslate2 model directory.
# large-v3 is a 2.9 GB one-time download and matches the API backends' quality;
# "small" (464 MB) is the usual choice when disk or CPU time is tight.
DEFAULT_MODEL = "large-v3"

# GPU compute type. int8_float16 rather than float16 on purpose: it halves VRAM
# (~1.5 GB vs ~3.1 GB for large-v3) with no accuracy loss worth measuring, which
# is the difference between fitting and OOM-ing on a 4 GB card.
GPU_COMPUTE = "int8_float16"
GPU_COMPUTE_FALLBACK = "float16"
CPU_COMPUTE = "int8"


def is_available() -> bool:
    """True if faster-whisper is importable (i.e. the user opted into local)."""
    try:
        import faster_whisper  # noqa: F401
    except Exception:
        return False
    return True


def resolve_runtime(
    device: str | None = None,
    compute_type: str | None = None,
) -> tuple[str, str]:
    """Pick (device, compute_type), honouring explicit overrides.

    ``device`` of None/"auto" probes for a usable CUDA device via CTranslate2 and
    falls back to CPU. Detection is best-effort only — a machine can report a CUDA
    device and still fail at load time on missing cuDNN (common under WSL), which
    is why :func:`transcribe_local` also retries on CPU at runtime.
    """
    want_device = (device or os.environ.get("WATCH_WHISPER_DEVICE") or "auto").strip().lower()
    want_compute = (compute_type or os.environ.get("WATCH_WHISPER_COMPUTE") or "auto").strip().lower()

    resolved_device = "cpu"
    supported: set[str] = set()
    if want_device in ("auto", "cuda"):
        try:
            import ctranslate2

            if ctranslate2.get_cuda_device_count() > 0:
                resolved_device = "cuda"
                supported = set(ctranslate2.get_supported_compute_types("cuda"))
        except Exception:
            resolved_device = "cpu"
    if want_device == "cpu":
        resolved_device = "cpu"
    elif want_device == "cuda":
        # Explicit request: honour it even if probing failed, and let the load
        # error (with its CPU retry) be the thing that reports the real problem.
        resolved_device = "cuda"

    if want_compute != "auto":
        return resolved_device, want_compute

    if resolved_device == "cuda":
        if not supported or GPU_COMPUTE in supported:
            return resolved_device, GPU_COMPUTE
        if GPU_COMPUTE_FALLBACK in supported:
            return resolved_device, GPU_COMPUTE_FALLBACK
        return resolved_device, "float32"
    return resolved_device, CPU_COMPUTE


def _preload_cuda_libs() -> int:
    """Make pip-installed CUDA libraries loadable by CTranslate2. Returns count loaded.

    `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12` drops its .so files under
    site-packages/nvidia/*/lib, which is not on the dynamic loader's search path.
    CTranslate2 then reports "Library libcublas.so.12 is not found" on a machine
    where the library is demonstrably installed — the single most common reason
    GPU transcription silently falls back to CPU. Loading them RTLD_GLOBAL here
    puts them in the process's global symbol table before the model is built.

    Discovery is by glob, never by hardcoded soname, so this keeps working across
    CUDA major versions. It is best-effort: a system-wide CUDA install needs none
    of it, and every failure is ignored in favour of the CPU retry.
    """
    bases = list(site.getsitepackages())
    try:
        bases.append(site.getusersitepackages())
    except Exception:
        pass

    libs = []
    for base in bases:
        libs.extend(glob.glob(os.path.join(base, "nvidia", "*", "lib", "*.so*")))
    if not libs:
        return 0

    # Several passes: these libraries depend on each other and glob order is
    # arbitrary, so one that fails on pass 1 often loads once its dependency has.
    loaded: set[str] = set()
    for _ in range(3):
        progressed = False
        for lib in libs:
            if lib in loaded:
                continue
            try:
                ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                continue
            loaded.add(lib)
            progressed = True
        if not progressed:
            break
    return len(loaded)


def _load_model(model: str, device: str, compute_type: str):
    from faster_whisper import WhisperModel

    return WhisperModel(model, device=device, compute_type=compute_type)


def _looks_like_vad_problem(exc: Exception) -> bool:
    """True if `exc` reads like a missing VAD dependency rather than a real failure.

    Matching on message text is crude, but the alternative — treating *any*
    exception from a VAD-enabled transcribe as a VAD problem — misreports GPU
    and OOM failures as "VAD unavailable" and sends the user down the wrong path.
    """
    blob = f"{type(exc).__name__}: {exc}".lower()
    return any(token in blob for token in ("onnx", "vad", "silero"))


def _collect(loaded, audio_path: Path, language: str | None, vad: bool) -> list[dict]:
    """Transcribe and fully drain the segment generator.

    Draining matters: faster-whisper's transcribe() returns lazily, so the actual
    compute — and therefore any GPU failure — happens in this loop, not at the
    call above it. Anything that consumes segments outside a try/except would let
    those errors escape the fallback that is supposed to catch them.
    """
    segments, info = loaded.transcribe(str(audio_path), language=language, vad_filter=vad)

    out: list[dict] = []
    total = getattr(info, "duration", 0.0) or 0.0
    next_mark = 60.0
    for seg in segments:
        text = (seg.text or "").strip()
        if text:
            out.append({
                "start": round(float(seg.start or 0.0), 2),
                "end": round(float(seg.end or 0.0), 2),
                "text": text,
            })
        # Report progress so a long clip doesn't look like a hang.
        if total and seg.end and seg.end >= next_mark:
            pct = min(100, int(100 * seg.end / total))
            print(f"[watch] local whisper: {pct}% ({seg.end:.0f}s/{total:.0f}s)", file=sys.stderr)
            while next_mark <= seg.end:
                next_mark += 60.0

    detected = getattr(info, "language", None)
    if detected and language is None:
        print(f"[watch] local whisper detected language: {detected}", file=sys.stderr)
    return out


def _run(loaded, audio_path: Path, language: str | None) -> list[dict]:
    """Transcribe, dropping the VAD filter if (and only if) VAD is what broke.

    vad_filter needs onnxruntime and the bundled Silero weights; when either is
    missing faster-whisper raises. VAD only trims silence, so losing it costs
    speed, never correctness. Every other failure propagates so the caller's
    device fallback can see it.
    """
    try:
        return _collect(loaded, audio_path, language, vad=True)
    except Exception as exc:
        if not _looks_like_vad_problem(exc):
            raise
        print(
            f"[watch] VAD filter unavailable ({type(exc).__name__}) — "
            "transcribing without it",
            file=sys.stderr,
        )
        return _collect(loaded, audio_path, language, vad=False)


def transcribe_local(
    audio_path: Path,
    model: str | None = None,
    device: str | None = None,
    compute_type: str | None = None,
    language: str | None = None,
) -> list[dict]:
    """Transcribe `audio_path` on-device and return {start, end, text} segments.

    `language` of None lets Whisper auto-detect. Raises SystemExit on failure so
    the caller's existing `except SystemExit` fallback path keeps working.
    """
    if not is_available():
        raise SystemExit(
            "faster-whisper is not installed. Install the local backend with:\n"
            "  pip install \"faster-whisper>=1.0\"\n"
            "…or set GROQ_API_KEY / OPENAI_API_KEY to use an API backend instead."
        )

    model_name = model or os.environ.get("WATCH_WHISPER_MODEL") or DEFAULT_MODEL
    resolved_device, resolved_compute = resolve_runtime(device, compute_type)

    # A visible CUDA device is not a working one. CTranslate2 resolves cuBLAS and
    # cuDNN lazily, so a machine can enumerate a GPU, construct the model without
    # complaint, and only fail on the first matrix multiply. _preload_cuda_libs()
    # removes the most common cause; OOM, driver mismatches and unsupported compute
    # types remain, and only show up mid-transcode. Deliberately no library-presence
    # pre-flight beyond that preload: hardcoding
    # sonames (libcublas.so.12, libcudnn_ops.so.9) would silently demote a working
    # GPU as soon as CTranslate2 moves to the next CUDA major. Retrying the whole
    # load-and-transcode on CPU is version-proof and costs one wasted model load.
    attempts = [(resolved_device, resolved_compute)]
    if resolved_device == "cuda":
        attempts.append(("cpu", CPU_COMPUTE))

    last_error: Exception | None = None
    for index, (attempt_device, attempt_compute) in enumerate(attempts):
        if attempt_device == "cuda":
            _preload_cuda_libs()
        print(
            f"[watch] loading local whisper model '{model_name}' "
            f"({attempt_device}/{attempt_compute})…",
            file=sys.stderr,
        )
        try:
            loaded = _load_model(model_name, attempt_device, attempt_compute)
            return _run(loaded, audio_path, language)
        except Exception as exc:
            last_error = exc
            if index + 1 < len(attempts):
                print(
                    f"[watch] {attempt_device} backend failed "
                    f"({type(exc).__name__}: {exc}) — falling back to CPU",
                    file=sys.stderr,
                )

    raise SystemExit(f"Local whisper failed ('{model_name}'): {last_error}")


if __name__ == "__main__":
    import json

    if len(sys.argv) < 2:
        print("usage: local_whisper.py <audio-path> [--model M] [--device cpu|cuda]", file=sys.stderr)
        raise SystemExit(2)

    argv = sys.argv[1:]
    audio = Path(argv[0])
    model_arg = argv[argv.index("--model") + 1] if "--model" in argv else None
    device_arg = argv[argv.index("--device") + 1] if "--device" in argv else None

    result = transcribe_local(audio, model=model_arg, device=device_arg)
    print(json.dumps({"backend": "local", "segments": result}, indent=2))
