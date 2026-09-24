#!/usr/bin/env python3
"""Gemini engine: Google watches the video (agentic processing) and answers.

Standard library only. The key travels in the x-goog-api-key header and is
scrubbed from every diagnostic.
"""
from __future__ import annotations

import json
import math
import mimetypes
import re
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from runtime import diagnostic

API = 'https://generativelanguage.googleapis.com'
INTERACTIONS = f'{API}/v1beta/interactions'
_YOUTUBE = re.compile(
    r'^https?://(?:(?:www\.|m\.|music\.)?youtube\.com/(?:watch\?\S*\bv=|shorts/|live/)|youtu\.be/)\S+$', re.I)
_HINTS = {
    'auth': 'check GEMINI_API_KEY (create one at https://aistudio.google.com/apikey)',
    'quota': 'the key is rate-limited or out of quota; wait or check its plan',
    'rejected': 'Gemini refused this request; the video may be private, unsupported, or too long',
    'service': 'Gemini had a server error; retry shortly',
    'network': 'could not reach generativelanguage.googleapis.com; check network, proxy, or egress rules',
    'response': 'Gemini returned something this version cannot read',
    'upload': 'the video could not be uploaded to or processed by the Gemini Files API',
}
DEFAULT_QUESTION = ('Give a thorough summary of this video: what is shown, any on-screen text or code, '
                    'and what is said, in chronological order.')
PROMPT_SUFFIX = ('\n\nCite an MM:SS (or H:MM:SS) timestamp for every claim about a specific moment. '
                 'Separate what is seen from what is heard when it matters. If the video does not '
                 'show or say something needed to answer, say so plainly instead of guessing.')


def is_youtube(source: str) -> bool:
    return bool(_YOUTUBE.match(source or ''))


def build_prompt(question: str | None) -> str:
    return ((question or '').strip() or DEFAULT_QUESTION) + PROMPT_SUFFIX


def _fail(category: str, detail: str, key: str) -> SystemExit:
    detail = diagnostic(detail, 600).replace(key, '[redacted]') if key else diagnostic(detail, 600)
    return SystemExit(f'Gemini {category}: {_HINTS[category]}. Detail: {detail or "none"}. '
                      'No local fallback was attempted; rerun with --engine local to use frames + transcript.')


def _error_message(body: bytes) -> str:
    text = body.decode('utf-8', errors='replace')
    try:
        data = json.loads(text)
        if isinstance(data, list) and data:  # Auth errors arrive wrapped in a one-element array.
            data = data[0]
        return str(data['error']['message'])
    except (ValueError, KeyError, TypeError):
        return text


def _call(method: str, url: str, key: str, *, data=None, headers=None, timeout: float = 60.0):
    request = Request(url, data=data, method=method, headers={'x-goog-api-key': key, **(headers or {})})
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.status, dict(response.headers), response.read()
    except HTTPError as exc:
        message = _error_message(exc.read())
        # The Files API answers a malformed key with 400 "API key not valid", not 401.
        category = ('auth' if exc.code in (401, 403) or 'api key' in message.lower() else 'quota' if exc.code == 429
                    else 'service' if exc.code >= 500 else 'rejected')
        raise _fail(category, f'HTTP {exc.code}: {message}', key) from None
    except (URLError, TimeoutError, OSError) as exc:
        raise _fail('network', f'{type(exc).__name__}: {getattr(exc, "reason", exc)}', key) from None


def _clock(seconds: int) -> str:
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f'{hours}:{minutes:02d}:{secs:02d}' if hours else f'{minutes:02d}:{secs:02d}'


def _processing(clip):
    if not clip or (clip[0] is None and clip[1] is None):
        return 'agentic', 'agentic'
    start, end = clip
    config = {'type': 'static'}  # Offsets are only accepted in static mode, as duration strings.
    if start is not None:
        config['start_offset'] = f'{int(math.floor(start))}s'
    if end is not None:
        config['end_offset'] = f'{int(math.ceil(end))}s'
    label = (f'static clip {_clock(int(math.floor(start or 0)))}–'
             f'{_clock(int(math.ceil(end))) if end is not None else "end"}')
    return config, label


def ask(video: dict, question: str | None, *, model: str, key: str, clip=None, timeout: float = 600.0) -> dict:
    processing, label = _processing(clip)
    payload = {'model': model, 'input': [{'type': 'video', **video, 'processing': processing},
                                         {'type': 'text', 'text': build_prompt(question)}]}
    _, _, body = _call('POST', INTERACTIONS, key, data=json.dumps(payload).encode('utf-8'),
                       headers={'Content-Type': 'application/json'}, timeout=timeout)
    try:
        data = json.loads(body)
        parts = [part['text'] for step in data['steps'] if step.get('type') == 'model_output'
                 for part in step.get('content', []) if part.get('type') == 'text' and part.get('text')]
        tokens = (data.get('usage') or {}).get('total_tokens')
    except (ValueError, KeyError, TypeError, AttributeError):
        raise _fail('response', body[:300].decode('utf-8', errors='replace'), key) from None
    if not parts:
        raise _fail('response', f'no answer text (status {data.get("status")})', key)
    return {'text': '\n'.join(parts).strip(), 'model': model, 'processing': label,
            'total_tokens': tokens if isinstance(tokens, int) else None}


def upload_file(path: Path, key: str, *, poll_seconds: float = 2.0, max_wait: float = 900.0, sleep=time.sleep) -> dict:
    path = Path(path)
    size = path.stat().st_size
    if size == 0:
        raise _fail('upload', f'{path.name} is empty', key)
    guessed = mimetypes.guess_type(path.name)[0] or ''
    mime = guessed if guessed.startswith('video/') else 'video/mp4'
    _, headers, _ = _call('POST', f'{API}/upload/v1beta/files', key,
                          data=json.dumps({'file': {'display_name': 'watch-upload'}}).encode('utf-8'),
                          headers={'X-Goog-Upload-Protocol': 'resumable', 'X-Goog-Upload-Command': 'start',
                                   'X-Goog-Upload-Header-Content-Length': str(size),
                                   'X-Goog-Upload-Header-Content-Type': mime, 'Content-Type': 'application/json'})
    session = {name.lower(): value for name, value in headers.items()}.get('x-goog-upload-url')
    if not session:
        raise _fail('upload', 'the Files API did not return an upload session URL', key)
    with path.open('rb') as stream:  # Streamed in blocks; never held in memory.
        _, _, body = _call('POST', session, key, data=stream,
                           headers={'Content-Length': str(size), 'X-Goog-Upload-Offset': '0',
                                    'X-Goog-Upload-Command': 'upload, finalize'}, timeout=3600.0)
    try:
        info = json.loads(body)['file']
        waited = 0.0
        while info.get('state') == 'PROCESSING':
            if waited >= max_wait:
                delete_file(info['name'], key)
                raise _fail('upload', f'still processing after {int(max_wait)}s', key)
            sleep(poll_seconds)
            waited += poll_seconds
            info = json.loads(_call('GET', f"{API}/v1beta/{info['name']}", key)[2])
        if info.get('state') != 'ACTIVE':
            delete_file(info['name'], key)
            raise _fail('upload', f"file state {info.get('state')}", key)
        return {'name': info['name'], 'uri': info['uri'], 'mime_type': info.get('mimeType') or mime}
    except (ValueError, KeyError, TypeError):
        raise _fail('upload', 'unreadable Files API response', key) from None


def delete_file(name: str, key: str) -> str | None:
    try:
        _call('DELETE', f'{API}/v1beta/{name}', key)
        return None
    except (SystemExit, Exception) as exc:  # Cleanup must never mask the real result or error.
        return (f'Could not delete uploaded {name} from Google ({str(exc).split(". Detail:")[0]}); '
                'it expires on its own within 48 hours.')
