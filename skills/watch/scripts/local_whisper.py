#!/usr/bin/env python3
"""Local Whisper fallback via faster-whisper (CTranslate2) — no API key needed.

Kicks in when no cloud (Groq/OpenAI) key is configured. Reuses whisper.extract_audio
for the same ffmpeg audio extraction, then transcribes on-device with
faster-whisper large-v3. Returns {start, end, text} segments identical in shape to
the cloud path, so filter_range / format_transcript downstream don't care.

Requires `pip install faster-whisper` (imports lazily, only when actually used).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from whisper import extract_audio  # same ffmpeg mono-16k extraction as the cloud path

# ponytail: large-v3 default (the "v3" model); override with WATCH_FW_MODEL=small
# for speed on a slow CPU. int8 keeps it torch-free; bump to float16 if on GPU.
DEFAULT_MODEL = os.environ.get("WATCH_FW_MODEL", "large-v3")

# ponytail: whisper's own doubt cutoffs. Segments past either are MARKED, never
# dropped — silently deleting weak audio is how the tail got eaten before, and a
# reader can discount a marked line but not a removed one.
#
# Know what this does NOT do. It catches degraded audio, not invention. Measured on
# the confabulated "yes, ma'am" passage that motivated it, the made-up lines scored
# no_speech_prob 0.04 / avg_logprob -0.35 — more confident than the real dialogue
# beside them, because these numbers describe decoding certainty, not truth, and a
# fluent hallucination off a strong language prior decodes cleanly. The values are
# also per decode window, not per segment: every line in that 30-second stretch
# carried identical figures. Frames remain the only check on whether a quiet scene
# said what the transcript claims.
NO_SPEECH_MAX = 0.6
AVG_LOGPROB_MIN = -1.0

_model_cache: dict[str, object] = {}


def _get_model(name: str):
    from faster_whisper import WhisperModel  # heavy import — only when transcribing

    if name not in _model_cache:
        # device="auto" uses CUDA if present else CPU; int8 needs neither torch nor GPU.
        _model_cache[name] = WhisperModel(name, device="auto", compute_type="int8")
    return _model_cache[name]


def transcribe_video_local(
    video_path: str,
    audio_out: Path,
    model_name: str | None = None,
) -> tuple[list[dict], str]:
    """Extract audio → transcribe locally. Returns (segments, backend_label)."""
    model_name = model_name or DEFAULT_MODEL
    print(f"[watch] extracting audio for local faster-whisper ({model_name})…", file=sys.stderr)
    audio_path = extract_audio(video_path, audio_out)

    print(
        f"[watch] transcribing locally with faster-whisper {model_name} "
        "(first run downloads the model)…",
        file=sys.stderr,
    )
    model = _get_model(model_name)
    # ponytail: temperature=0.0 alone + condition_on_previous_text (the default True)
    # makes whisper loop — it repeats one token or sentence from some point to the end
    # of the file and the tail of the transcript is silently destroyed. These two
    # settings are the fix: no conditioning on prior text, and a temperature ladder so
    # a degenerate decode is retried instead of accepted.
    #
    # vad_filter is deliberately OFF, and this is a trade rather than a free win.
    # ON, a 114-minute film lost its last 21 minutes outright (785 segments vs 1543),
    # and the truncation point moved depending on whether the audio was passed whole
    # or sliced, so the loss scales with input length. OFF, silence reaches the model
    # and it confabulates fluent dialogue over it — on that same film it invented a
    # cleanup crew answering "yes, ma'am" over a scene where the frames show one
    # person alone. Losing 21 real minutes is worse than gaining a few invented lines
    # that the confidence marking below flags, so OFF stands. Do not flip it back
    # without re-measuring segment counts on a long file.
    segments_iter, _info = model.transcribe(
        str(audio_path.resolve()),
        beam_size=5,
        condition_on_previous_text=False,
        temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    )

    out: list[dict] = []
    n_unsure = 0
    for seg in segments_iter:
        text = (seg.text or "").strip()
        if not text:
            continue
        record = {"start": round(seg.start, 2), "end": round(seg.end, 2), "text": text}
        # Fail-open: a build that does not report these is treated as confident.
        no_speech = getattr(seg, "no_speech_prob", 0.0)
        avg_logprob = getattr(seg, "avg_logprob", 0.0)
        if no_speech > NO_SPEECH_MAX or avg_logprob < AVG_LOGPROB_MIN:
            record["uncertain"] = True
            n_unsure += 1
        out.append(record)

    if not out:
        raise SystemExit("Local faster-whisper returned no transcript segments")

    note = f", {n_unsure} low-confidence (marked [?])" if n_unsure else ""
    print(
        f"[watch] transcribed {len(out)} segments via faster-whisper {model_name}{note}",
        file=sys.stderr,
    )
    return out, f"faster-whisper ({model_name})"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: local_whisper.py <video-path> [audio-out.mp3] [model]", file=sys.stderr)
        raise SystemExit(2)
    import json

    video = sys.argv[1]
    audio_out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("audio.mp3")
    model = sys.argv[3] if len(sys.argv) > 3 else None
    segs, backend = transcribe_video_local(video, audio_out, model)
    print(json.dumps({"backend": backend, "segments": segs}, indent=2))
