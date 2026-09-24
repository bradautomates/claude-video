"""Gemini adapter: payloads, parsing, and failures against a fake urlopen. No network."""
from __future__ import annotations

import io
import json
from urllib.error import HTTPError, URLError

import pytest

import gemini

KEY = 'unit-test-key-DO-NOT-LEAK'


class FakeResponse:
    def __init__(self, body=b'{}', status=200, headers=None):
        self.status, self.headers, self._body = status, headers or {}, body
    def read(self):
        return self._body
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        return False


def http_error(code, body):
    return HTTPError('https://example.invalid', code, 'error', {}, io.BytesIO(body))


def answer(*texts):
    steps = [{'type': 'processing_call', 'id': 'c1'}, {'type': 'processing_result', 'call_id': 'c1'}, {'type': 'thought'}]
    steps.append({'type': 'model_output', 'content': [{'type': 'text', 'text': t} for t in texts]})
    return json.dumps({'status': 'completed', 'steps': steps, 'usage': {'total_tokens': 4348}}).encode()


@pytest.fixture
def calls(monkeypatch):
    """Record every request; replies are popped from calls.replies in order."""
    class Calls(list):
        replies = []
    recorded = Calls()
    def fake(request, timeout=None):
        recorded.append(request)
        reply = recorded.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply
    monkeypatch.setattr(gemini, 'urlopen', fake)
    return recorded


@pytest.mark.parametrize('source,expected', [
    ('https://www.youtube.com/watch?v=rlOpbu3Enkw', True),
    ('https://youtu.be/rlOpbu3Enkw', True),
    ('https://m.youtube.com/shorts/abc123', True),
    ('https://www.youtube.com/playlist?list=PL1', False),
    ('https://vimeo.com/123', False),
    ('/tmp/clip.mp4', False),
    ('https://notyoutube.com/watch?v=x', False),
])
def test_is_youtube(source, expected):
    assert gemini.is_youtube(source) is expected


def test_prompt_always_demands_timestamps():
    assert 'What is on the whiteboard?' in gemini.build_prompt('What is on the whiteboard?')
    assert 'MM:SS' in gemini.build_prompt('What is on the whiteboard?')
    default = gemini.build_prompt(None)
    assert 'summary' in default.lower() and 'MM:SS' in default


def test_agentic_youtube_payload_and_header_only_key(calls):
    calls.replies = [FakeResponse(answer('At 00:10 the screen turns blue.'))]
    result = gemini.ask({'uri': 'https://youtu.be/abc'}, 'When does it change?', model='gemini-3.7-flash', key=KEY)
    request = calls[0]
    assert request.full_url == 'https://generativelanguage.googleapis.com/v1beta/interactions'
    assert KEY not in request.full_url
    assert request.get_header('X-goog-api-key') == KEY
    payload = json.loads(request.data)
    assert payload['model'] == 'gemini-3.7-flash'
    assert payload['input'][0] == {'type': 'video', 'uri': 'https://youtu.be/abc', 'processing': 'agentic'}
    assert payload['input'][1]['type'] == 'text' and 'When does it change?' in payload['input'][1]['text']
    assert result == {'text': 'At 00:10 the screen turns blue.', 'model': 'gemini-3.7-flash',
                      'processing': 'agentic', 'total_tokens': 4348}


def test_clip_uses_static_processing_with_duration_strings(calls):
    calls.replies = [FakeResponse(answer('Blue.'))]
    result = gemini.ask({'uri': 'https://generativelanguage.googleapis.com/v1beta/files/x', 'mime_type': 'video/mp4'},
                        None, model='m', key=KEY, clip=(1200.4, 1500.2))
    video = json.loads(calls[0].data)['input'][0]
    assert video['mime_type'] == 'video/mp4'
    assert video['processing'] == {'type': 'static', 'start_offset': '1200s', 'end_offset': '1501s'}
    assert result['processing'] == 'static clip 20:00–25:01'


def test_open_ended_clip_omits_the_missing_bound(calls):
    calls.replies = [FakeResponse(answer('ok'))]
    gemini.ask({'uri': 'https://youtu.be/abc'}, None, model='m', key=KEY, clip=(90.0, None))
    assert json.loads(calls[0].data)['input'][0]['processing'] == {'type': 'static', 'start_offset': '90s'}


def test_multiple_text_parts_are_joined(calls):
    calls.replies = [FakeResponse(answer('First.', 'Second.'))]
    assert gemini.ask({'uri': 'https://youtu.be/abc'}, None, model='m', key=KEY)['text'] == 'First.\nSecond.'


def test_empty_answer_is_a_failure(calls):
    calls.replies = [FakeResponse(json.dumps({'steps': [{'type': 'thought'}]}).encode())]
    with pytest.raises(SystemExit, match='Gemini response'):
        gemini.ask({'uri': 'https://youtu.be/abc'}, None, model='m', key=KEY)


@pytest.mark.parametrize('reply,category', [
    (http_error(401, b'[{"error":{"code":401,"message":"Request had invalid authentication credentials.","status":"UNAUTHENTICATED"}}]'), 'auth'),
    (http_error(403, b'{"error":{"message":"API key not valid","code":"permission_denied"}}'), 'auth'),
    (http_error(429, b'{"error":{"message":"Quota exceeded","code":"resource_exhausted"}}'), 'quota'),
    (http_error(400, b'{"error":{"message":"Invalid input at \'input[0].processing\'.","code":"invalid_request"}}'), 'rejected'),
    (http_error(503, b'upstream unavailable'), 'service'),
    (URLError('certificate verify failed'), 'network'),
    (TimeoutError('timed out'), 'network'),
    (FakeResponse(b'<html>not json</html>'), 'response'),
])
def test_failures_are_categorized_and_never_leak_the_key(calls, reply, category):
    calls.replies = [reply]
    with pytest.raises(SystemExit) as caught:
        gemini.ask({'uri': 'https://youtu.be/abc'}, None, model='m', key=KEY)
    message = str(caught.value)
    assert message.startswith(f'Gemini {category}:')
    assert KEY not in message
    assert '--engine local' in message


def test_key_echoed_by_the_server_is_redacted(calls):
    calls.replies = [http_error(400, json.dumps({'error': {'message': f'bad key {KEY}'}}).encode())]
    with pytest.raises(SystemExit) as caught:
        gemini.ask({'uri': 'https://youtu.be/abc'}, None, model='m', key=KEY)
    assert KEY not in str(caught.value) and '[redacted]' in str(caught.value)


def file_state(state, name='files/abc'):
    return FakeResponse(json.dumps({'name': name, 'state': state, 'mimeType': 'video/mp4',
                                    'uri': f'https://generativelanguage.googleapis.com/v1beta/{name}'}).encode())


def test_upload_streams_file_then_polls_until_active(calls, tmp_path):
    clip = tmp_path / 'clip.mp4'
    clip.write_bytes(b'x' * 1000)
    calls.replies = [
        FakeResponse(headers={'X-Goog-Upload-URL': 'https://upload.invalid/session?upload_id=1'}),
        FakeResponse(json.dumps({'file': {'name': 'files/abc', 'state': 'PROCESSING', 'mimeType': 'video/mp4',
                                          'uri': 'https://generativelanguage.googleapis.com/v1beta/files/abc'}}).encode()),
        file_state('PROCESSING'), file_state('ACTIVE'),
    ]
    naps = []
    result = gemini.upload_file(clip, KEY, sleep=naps.append)
    start, body, poll = calls[0], calls[1], calls[2]
    assert start.full_url == 'https://generativelanguage.googleapis.com/upload/v1beta/files'
    assert start.get_header('X-goog-upload-command') == 'start'
    assert start.get_header('X-goog-upload-header-content-length') == '1000'
    assert start.get_header('X-goog-upload-header-content-type') == 'video/mp4'
    assert body.full_url == 'https://upload.invalid/session?upload_id=1'
    assert body.get_header('X-goog-upload-command') == 'upload, finalize'
    assert body.get_header('Content-length') == '1000'
    assert hasattr(body.data, 'read'), 'the body must be streamed, not loaded into memory'
    assert poll.full_url.endswith('/v1beta/files/abc') and poll.get_method() == 'GET'
    assert naps == [2.0, 2.0]
    assert result == {'name': 'files/abc', 'mime_type': 'video/mp4',
                      'uri': 'https://generativelanguage.googleapis.com/v1beta/files/abc'}


def test_upload_failed_state_and_timeout(calls, tmp_path):
    clip = tmp_path / 'clip.mkv'
    clip.write_bytes(b'x')
    first = [FakeResponse(headers={'x-goog-upload-url': 'https://upload.invalid/s'}),
             FakeResponse(json.dumps({'file': {'name': 'files/abc', 'state': 'PROCESSING', 'uri': 'u'}}).encode())]
    calls.replies = [*first, file_state('FAILED')]
    with pytest.raises(SystemExit, match='Gemini upload:.*FAILED'):
        gemini.upload_file(clip, KEY, sleep=lambda s: None)
    calls.replies = [*first, *[file_state('PROCESSING') for _ in range(3)]]
    with pytest.raises(SystemExit, match='Gemini upload:.*still processing'):
        gemini.upload_file(clip, KEY, poll_seconds=2.0, max_wait=6.0, sleep=lambda s: None)


def test_upload_without_session_url_or_with_empty_file(calls, tmp_path):
    empty = tmp_path / 'empty.mp4'
    empty.write_bytes(b'')
    with pytest.raises(SystemExit, match='Gemini upload:.*empty'):
        gemini.upload_file(empty, KEY)
    clip = tmp_path / 'clip.mp4'
    clip.write_bytes(b'x')
    calls.replies = [FakeResponse(headers={})]
    with pytest.raises(SystemExit, match='Gemini upload:.*session'):
        gemini.upload_file(clip, KEY)


def test_unknown_extension_falls_back_to_mp4(calls, tmp_path):
    clip = tmp_path / 'clip.unknownext'
    clip.write_bytes(b'x')
    calls.replies = [FakeResponse(headers={'x-goog-upload-url': 'https://upload.invalid/s'}),
                     FakeResponse(json.dumps({'file': {'name': 'files/a', 'state': 'ACTIVE', 'uri': 'u'}}).encode())]
    assert gemini.upload_file(clip, KEY)['mime_type'] == 'video/mp4'
    assert calls[0].get_header('X-goog-upload-header-content-type') == 'video/mp4'


def test_delete_is_best_effort(calls):
    calls.replies = [FakeResponse(b'{}')]
    assert gemini.delete_file('files/abc', KEY) is None
    assert calls[0].get_method() == 'DELETE' and calls[0].full_url.endswith('/v1beta/files/abc')
    calls.replies = [http_error(500, b'boom')]
    warning = gemini.delete_file('files/abc', KEY)
    assert 'files/abc' in warning and '48 hours' in warning and KEY not in warning


def test_invalid_key_reported_as_http_400_is_still_auth(calls):
    """Observed live: the Files API answers a malformed key with 400, not 401."""
    calls.replies = [http_error(400, b'{"error":{"code":400,"message":"API key not valid. Please pass a valid API key.","status":"INVALID_ARGUMENT"}}')]
    with pytest.raises(SystemExit, match='^Gemini auth:'):
        gemini.ask({'uri': 'https://youtu.be/abc'}, None, model='m', key=KEY)
