"""Subprocess-only WhisperX 3.8.6 adapter. Never imports Torch into watch."""
from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from config import ConfigError, get_config, positive_number
from runtime import diagnostic
from transcribe import normalize_segments

MODELS = {'tiny', 'base', 'small', 'medium', 'large', 'large-v1', 'large-v2', 'large-v3',
          'large-v3-turbo', 'turbo', 'tiny.en', 'base.en', 'small.en', 'medium.en'}


def settings(cfg=None) -> dict:
    cfg = dict(get_config(backend_override='whisperx') if cfg is None else cfg)
    if cfg['whisperx_model'] not in MODELS:
        raise ConfigError('WATCH_WHISPERX_MODEL must be a supported Whisper model name (default small).')
    if cfg['whisperx_device'] not in ('cpu', 'cuda'):
        raise ConfigError('WATCH_WHISPERX_DEVICE must be cpu or cuda (CUDA is untested).')
    if cfg['whisperx_compute_type'] not in ('default', 'int8', 'float16', 'float32'):
        raise ConfigError('WATCH_WHISPERX_COMPUTE_TYPE must be default, int8, float16, or float32.')
    batch = positive_number(cfg['whisperx_batch_size'], 'WATCH_WHISPERX_BATCH_SIZE')
    if batch != int(batch):
        raise ConfigError('WATCH_WHISPERX_BATCH_SIZE must be a positive integer.')
    cfg['whisperx_batch_size'] = int(batch)
    cfg['whisperx_timeout'] = positive_number(cfg['whisperx_timeout'], 'WATCH_WHISPERX_TIMEOUT') if cfg['whisperx_timeout'] else None
    if cfg['whisperx_language'] and not re.fullmatch('[a-z]{2}', cfg['whisperx_language']):
        raise ConfigError('WATCH_WHISPERX_LANGUAGE must be a two-letter lowercase code.')
    return cfg


def sentinel_for(executable: Path) -> Path:
    return executable.parent.parent / '.deps-ok'


def executable_path(cfg: dict, *, require_ready=True) -> Path:
    value = cfg['whisperx_bin']
    if not value or not Path(value).is_absolute() or not Path(value).is_file():
        raise SystemExit('WhisperX executable missing. Run setup.py --install-whisperx.')
    executable = Path(value)
    if require_ready and not sentinel_for(executable).is_file():
        raise SystemExit('WhisperX venv is incomplete (no .deps-ok). Run setup.py --install-whisperx to rebuild it.')
    return executable


def command(executable: Path, audio: Path, output: Path, cfg: dict) -> list[str]:
    cmd = [str(executable), str(audio.resolve()), '--model', cfg['whisperx_model'],
           '--device', cfg['whisperx_device'], '--compute_type', cfg['whisperx_compute_type'],
           '--batch_size', str(cfg['whisperx_batch_size']), '--output_dir', str(output.resolve()),
           '--output_format', 'json', '--task', 'transcribe', '--no_align', '--vad_method', 'silero', '--verbose', 'False']
    if cfg['whisperx_language']:
        cmd += ['--language', cfg['whisperx_language']]
    return cmd


def child_env() -> dict[str, str]:
    env = dict(os.environ)
    env.update(PYTHONIOENCODING='utf-8', PYTHONWARNINGS='ignore', PYANNOTE_METRICS_ENABLED='0')
    return env


def run_inference(cmd: list[str], timeout: float | None = None) -> None:
    """Stream bounded byte diagnostics while enforcing an optional wall-clock timeout."""
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=child_env())
    except OSError as exc:
        raise SystemExit(f'WhisperX executable is not runnable ({type(exc).__name__}); rerun setup.py --install-whisperx.') from None
    messages = queue.Queue(maxsize=64)
    def read():
        try:
            while chunk := process.stdout.read1(4096):
                messages.put(chunk)
        finally:
            messages.put(None)
    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    deadline = time.monotonic() + timeout if timeout is not None else None
    tail = b''
    try:
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError()
            try:
                chunk = messages.get(timeout=0.1)
            except queue.Empty:
                continue
            if chunk is None:
                break
            tail = (tail + chunk)[-16000:]
            text = diagnostic(chunk, 4096)
            if text:
                print(f'[whisperx] {text}', file=sys.stderr, flush=True)
        process.wait(timeout=max(0.01, deadline - time.monotonic()) if deadline is not None else None)
    except (TimeoutError, subprocess.TimeoutExpired):
        process.kill()
        process.wait()
        raise SystemExit('WhisperX timed out and was stopped. Increase WATCH_WHISPERX_TIMEOUT or leave it unset.') from None
    except KeyboardInterrupt:
        process.kill()
        process.wait()
        raise SystemExit('WhisperX cancelled and stopped; no cloud fallback was attempted.') from None
    finally:
        # Drain queued diagnostics so the reader cannot remain blocked after cancellation.
        while reader.is_alive():
            try:
                messages.get(timeout=0.1)
            except queue.Empty:
                pass
        reader.join()
        process.stdout.close()
    if process.returncode != 0:
        detail = diagnostic(tail)
        lower = tail.decode('utf-8', errors='replace').lower()
        if any(token in lower for token in ('out of memory', 'bad_alloc', 'cuda', 'cudnn', 'cublas')) or process.returncode in (-9, 137):
            hint = 'Memory or device failure; use cpu/int8, a smaller model, or lower batch size. Small needs at least 8 GB system RAM.'
        elif any(token in lower for token in ('huggingface', 'connection', 'offline', 'download', 'localentrynotfound')):
            hint = 'Model download/cache failure; reconnect and rerun setup.py --install-whisperx to warm the caches.'
        else:
            hint = 'Check the error below and rerun setup.py --install-whisperx if dependencies are broken.'
        raise SystemExit(f'WhisperX failed (exit {process.returncode}). {hint}\n{detail}')


def transcribe_audio(audio: Path, cfg=None, *, require_ready=True):
    cfg = settings(cfg)
    executable = executable_path(cfg, require_ready=require_ready)
    print(f"[watch] local WhisperX {cfg['whisperx_model']} transcription; CPU runs may take several minutes.", file=sys.stderr, flush=True)
    with tempfile.TemporaryDirectory(prefix='watch-whisperx-', dir=audio.parent) as temporary:
        output = Path(temporary)
        run_inference(command(executable, audio, output, cfg), cfg['whisperx_timeout'])
        try:
            data = json.loads((output / f'{audio.stem}.json').read_text(encoding='utf-8'))
            segments = normalize_segments(data)
        except (OSError, ValueError, TypeError) as exc:
            raise SystemExit(f'WhisperX output missing or malformed ({type(exc).__name__}).') from None
    # 3.8.6 overwrites JSON language with alignment language even with --no_align.
    if segments.no_speech:
        print('[watch] WhisperX detected no speech.', file=sys.stderr)
    return segments
