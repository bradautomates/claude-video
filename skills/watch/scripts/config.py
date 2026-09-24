#!/usr/bin/env python3
"""Shared configuration: literal dotenv values, explicit precedence, atomic writes."""
from __future__ import annotations

import math
import os
import re
import tempfile
from pathlib import Path

CONFIG_DIR = Path.home() / '.config' / 'watch'
CONFIG_FILE = CONFIG_DIR / '.env'
DEFAULT_DETAIL = 'balanced'
DETAILS = {'transcript', 'efficient', 'balanced', 'token-burner'}
BACKENDS = {'auto', 'groq', 'openai', 'whisperx', 'none'}
ENGINES = {'auto', 'gemini', 'local'}
DEFAULT_GEMINI_MODEL = 'gemini-3.7-flash'


class ConfigError(ValueError):
    """A safe diagnostic that never includes a configuration value."""


def read_env_text(path: Path) -> str:
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return ''
    except OSError:
        raise ConfigError(f'Cannot read config {path}; check file permissions.') from None
    try:
        text = data.decode('utf-16' if data.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig')
        if '\x00' in text:
            raise UnicodeError()
        return text
    except UnicodeError:
        raise ConfigError(f'Cannot decode config {path}; save it as UTF-8 (or BOM-marked UTF-16).') from None


def parse_env(text: str, path: Path) -> dict[str, str]:
    """Last assignment wins. No expansion or escape interpretation (Windows paths work)."""
    values = {}
    for number, line in enumerate(text.splitlines(), 1):
        raw = line.strip()
        if not raw or raw.startswith('#') or '=' not in raw:
            continue
        key, value = raw.split('=', 1)
        key, value = key.strip(), value.strip()
        if not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', key):
            continue
        if value.startswith(('"', "'")):
            close = value.find(value[0], 1)
            if close < 0 or (value[close + 1:].strip() and not value[close + 1:].strip().startswith('#')):
                raise ConfigError(f'Malformed quoted value in {path}, line {number}; fix the quotes.')
            value = value[1:close]
        else:
            value = re.split(r'\s+#', value, maxsplit=1)[0].rstrip()
        values[key] = value
    return values


def read_env_file(path: Path | None = None) -> dict[str, str]:
    path = CONFIG_FILE if path is None else path
    return parse_env(read_env_text(path), path)


def write_settings(values: dict[str, str], path: Path | None = None) -> None:
    """Replace all assignments for these keys, preserving other lines and comments."""
    path = CONFIG_FILE if path is None else path
    text = read_env_text(path)
    parse_env(text, path)  # Never overwrite unreadable or malformed configuration.
    lines = []
    for line in text.splitlines():
        key = line.partition('=')[0].strip()
        if key not in values:
            lines.append(line)
    for key, value in values.items():
        if not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', key) or any(c in value for c in '\r\n'):
            raise ConfigError('Invalid configuration assignment.')
        quote = '"' if '"' not in value else "'"
        if quote in value:
            raise ConfigError('Configuration values cannot contain both kinds of quotes.')
        lines.append(f'{key}={quote}{value}{quote}')
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.watch-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as stream:
            stream.write('\n'.join(lines) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def load_api_key(preferred: str | None = None) -> tuple[str | None, str | None]:
    candidates = [('groq', 'GROQ_API_KEY'), ('openai', 'OPENAI_API_KEY')]
    for backend, name in candidates:
        if preferred is not None and backend != preferred:
            continue
        value = os.environ.get(name, '').strip()
        if not value:
            for path in (CONFIG_FILE, Path.cwd() / '.env'):
                value = read_env_file(path).get(name, '').strip()
                if value:
                    break
        if value:
            return backend, value
    return None, None


def load_gemini_key() -> str | None:
    """Environment, then the user config, then a project .env; never logged."""
    value = os.environ.get('GEMINI_API_KEY', '').strip()
    if not value:
        for path in (CONFIG_FILE, Path.cwd() / '.env'):
            value = read_env_file(path).get('GEMINI_API_KEY', '').strip()
            if value:
                break
    return value or None


def resolve_engine(choice: str, has_key: bool) -> str:
    if choice not in ENGINES:
        raise ConfigError('WATCH_ENGINE must be auto, gemini, or local.')
    if choice == 'gemini' and not has_key:
        raise ConfigError('The gemini engine needs GEMINI_API_KEY in the environment or '
                          f'{CONFIG_FILE}. Add it, or rerun with --engine local.')
    return 'gemini' if choice == 'gemini' or (choice == 'auto' and has_key) else 'local'


def get_config(*, backend_override: str | None = None) -> dict:
    file_values = read_env_file()
    def setting(name, default=''):
        return os.environ.get(name, file_values.get(name, default)).strip()
    detail = setting('WATCH_DETAIL', DEFAULT_DETAIL)
    backend = backend_override if backend_override is not None else setting('WATCH_WHISPER_BACKEND', 'auto')
    if backend not in BACKENDS:
        raise ConfigError('WATCH_WHISPER_BACKEND must be auto, groq, openai, whisperx, or none.')
    engine = setting('WATCH_ENGINE', 'auto')
    if engine not in ENGINES:
        raise ConfigError('WATCH_ENGINE must be auto, gemini, or local.')
    return {
        'detail': detail if detail in DETAILS else DEFAULT_DETAIL,
        'config_file': str(CONFIG_FILE),
        'whisper_backend': backend,
        'engine': engine,
        'gemini_model': setting('WATCH_GEMINI_MODEL', DEFAULT_GEMINI_MODEL) or DEFAULT_GEMINI_MODEL,
        'gemini_timeout': positive_number(setting('WATCH_GEMINI_TIMEOUT', '600'), 'WATCH_GEMINI_TIMEOUT'),
        'sub_lang': setting('WATCH_SUB_LANG', 'auto'),
        'cookies_file': setting('WATCH_COOKIES_FILE'),
        'cookies_from_browser': setting('WATCH_COOKIES_FROM_BROWSER'),
        'whisperx_bin': setting('WATCH_WHISPERX_BIN'),
        'whisperx_model': setting('WATCH_WHISPERX_MODEL', 'small'),
        'whisperx_device': setting('WATCH_WHISPERX_DEVICE', 'cpu'),
        'whisperx_compute_type': setting('WATCH_WHISPERX_COMPUTE_TYPE', 'int8'),
        'whisperx_batch_size': setting('WATCH_WHISPERX_BATCH_SIZE', '8'),
        'whisperx_language': setting('WATCH_WHISPERX_LANGUAGE'),
        'whisperx_timeout': setting('WATCH_WHISPERX_TIMEOUT'),
    }


def positive_number(value, name: str) -> float:
    try:
        number = float(value)
        if not math.isfinite(number) or number <= 0:
            raise ValueError()
        return number
    except (TypeError, ValueError):
        raise ConfigError(f'{name} must be a finite positive number.') from None


def frame_cap(detail: str) -> int | None:
    return {'efficient': 50, 'balanced': 100, 'token-burner': None, 'transcript': None}.get(detail, 100)
