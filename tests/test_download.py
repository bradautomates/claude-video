"""Bounded caption selection, argv inheritance, and download ownership."""
import json
import subprocess
from pathlib import Path

import pytest
import download

URL = 'https://example.com/video'


def _tracks(*keys):
    return {k: [{'url': 'https://example.com/sub.vtt', 'ext': 'vtt'}] for k in keys}


@pytest.mark.parametrize('language', ['en', 'ko', 'pt-BR'])
def test_original_track_overrides_dubbed_metadata(language):
    info = {'extractor_key': 'Youtube', 'language': 'en', 'subtitles': _tracks(language),
            'automatic_captions': _tracks(f'{language}-orig', 'en', 'ko')}
    track = download.select_caption(info)
    assert track['key'] == language and track['kind'] == 'manual' and track['provenance'] == 'original'
    info['subtitles'] = _tracks('fr')
    assert download.select_caption(info)['key'] == f'{language}-orig'


def test_unknown_language_prefers_english():
    # "Me at the zoo" reports no language and has manual de + en tracks.
    picked = download.select_caption({'extractor_key': 'Youtube', 'subtitles': _tracks('de', 'en')})
    assert (picked['key'], picked['provenance']) == ('en', 'unknown')
    assert download.select_caption({'automatic_captions': _tracks('af', 'en-GB', 'zu')})['key'] == 'en-GB'
    assert download.select_caption({'subtitles': _tracks('de', 'fr')})['key'] == 'de'


def test_requested_translation_and_unknown_source():
    info = {'extractor_key': 'Youtube', 'automatic_captions': _tracks('ko-orig', 'en')}
    assert download.select_caption(info, 'en')['provenance'] == 'requested translation'
    assert download.select_caption({'subtitles': _tracks('es')})['provenance'] == 'unknown'
    assert download.select_caption(info, 'de') is None


def test_cookie_modes_and_conflict():
    assert download.auth_args(cookies_from_browser='firefox:profile with spaces') == ['--no-cookies', '--cookies-from-browser', 'firefox:profile with spaces']
    assert download.auth_args(cookies_file='jar.txt')[:2] == ['--no-cookies-from-browser', '--cookies']
    assert download.auth_args() == []
    with pytest.raises(SystemExit, match='either'):
        download.auth_args('jar', 'browser')


def _fake_stages(monkeypatch, *, fail_media=False, media_path=None):
    calls = []
    def run(cmd):
        calls.append(cmd)
        template = cmd[cmd.index('-o') + 1]
        directory = Path(template.rsplit('/video.', 1)[0].replace('%%', '%'))
        if '--load-info-json' not in cmd and '--skip-download' in cmd:
            (directory / 'video.info.json').write_text(json.dumps({'id': 'test', 'language': 'ko', 'subtitles': _tracks('ko')}))
        elif '--sub-langs' in cmd:
            (directory / 'video.ko.vtt').write_text('WEBVTT\n\n00:00.000 --> 00:01.000\nhello\n')
        else:
            if fail_media:
                raise SystemExit('network failure')
            path = Path(media_path) if media_path else directory / 'video.mp4'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'completed')
            record = Path(cmd[cmd.index('--print-to-file') + 2].replace('%%', '%'))
            record.write_text(json.dumps(str(path)) + '\n')
        return subprocess.CompletedProcess(cmd, 0, '', '')
    monkeypatch.setattr(download, '_run', run)
    return calls


def test_fetch_captions_requests_one_exact_track(monkeypatch, tmp_path):
    calls = _fake_stages(monkeypatch)
    result = download.fetch_captions(URL, tmp_path / '100% 한 글', cookies_from_browser='firefox:profile')
    download.download_url(URL, tmp_path, context=result, cookies_from_browser='firefox:profile')
    metadata, captions, media = calls
    assert '--no-write-subs' in metadata and '--no-write-auto-subs' in metadata
    assert metadata[-2:] == ['--', URL]
    assert captions[captions.index('--sub-langs') + 1] == '-all,^ko$'
    assert '--load-info-json' in captions and URL not in captions
    assert '--no-write-subs' in media and '--no-write-auto-subs' in media and '--sub-langs' not in media
    assert all('--cookies-from-browser' in cmd and '--no-cookies' in cmd for cmd in calls)
    assert Path(result['subtitle_path']).exists()


def test_failed_download_never_returns_old_media(monkeypatch, tmp_path):
    (tmp_path / 'video.mp4').write_bytes(b'old source')
    (tmp_path / 'video.f137.mp4').write_bytes(b'partial')
    _fake_stages(monkeypatch, fail_media=True)
    with pytest.raises(SystemExit, match='network'):
        download.download_url(URL, tmp_path)
    assert (tmp_path / 'video.mp4').read_bytes() == b'old source'


@pytest.mark.parametrize('name', ['video.f137.mp4', 'video.part'])
def test_rejects_partial_final_paths(monkeypatch, tmp_path, name):
    run = tmp_path / 'run'
    run.mkdir()
    _fake_stages(monkeypatch, media_path=run / name)
    with pytest.raises(SystemExit, match='completed'):
        download.download_url(URL, tmp_path, context={'run_dir': str(run), 'info': {'title': 'test'}})


def test_rejects_outside_final_path(monkeypatch, tmp_path):
    _fake_stages(monkeypatch, media_path=tmp_path / 'outside.mp4')
    with pytest.raises(SystemExit, match='inside'):
        download.download_url(URL, tmp_path)


@pytest.mark.parametrize('message,hint', [
    ('HTTP Error 403: Forbidden', 'latest release'), ('unexpected extractor failure', 'original error'), ('HTTP 429', 'rate limiting'),
    ('Sign in to continue', 'authentication'), ('No JavaScript runtime', 'Deno/EJS'),
    ('egress denied', 'network settings'), ('certificate verify failed', 'trusted CA'),
])
def test_network_diagnostics(message, hint):
    assert hint in download.network_diagnostic(message)


def test_actual_ytdlp_exact_track_with_inherited_all(monkeypatch, tmp_path, static_clip):
    """Use the installed stable CLI offline for caption selection and final media paths."""
    import shutil
    if not shutil.which('yt-dlp'):
        pytest.skip('yt-dlp required for offline CLI acceptance')
    subs = tmp_path / 'source.vtt'
    subs.write_text('WEBVTT\n\n00:00.000 --> 00:01.000\n안녕\n', encoding='utf-8')
    user_cfg = Path.home() / '.config' / 'yt-dlp'
    user_cfg.mkdir(parents=True)
    # Redirecting typed templates must not divert watch's output.
    user_cfg.joinpath('config').write_text('--enable-file-urls\n--sub-langs all\n-o "subtitle:elsewhere/sub.%(ext)s"\n', encoding='utf-8')
    raw = {'id': 'offline', 'title': 'Offline', 'extractor': 'generic', 'extractor_key': 'Generic',
           'webpage_url': URL, 'url': static_clip.as_uri(), 'ext': 'mp4', 'duration': 3,
           'language': 'ko', 'subtitles': {'ko': [{'url': subs.as_uri(), 'ext': 'vtt'}],
                                        'en': [{'url': (tmp_path / 'must-not-read.vtt').as_uri(), 'ext': 'vtt'}]}}
    real_run = download._run
    def stages(cmd):
        if '--load-info-json' not in cmd and '--skip-download' in cmd:
            input_info = tmp_path / 'input.info.json'
            input_info.write_text(json.dumps(raw))
            # Retain the actual metadata-stage flags/templates, only substitute
            # an offline source for the network extractor.
            return real_run([*cmd[:-2], '--load-info-json', str(input_info)])
        return real_run(cmd)
    monkeypatch.setattr(download, '_run', stages)
    result = download.fetch_captions(URL, tmp_path / '100% 한 글')
    assert not result['errors'], result['errors']
    assert Path(result['subtitle_path']).read_text(encoding='utf-8').endswith('안녕\n')
    assert len(list(Path(result['run_dir']).glob('*.vtt'))) == 1
    assert Path(result['info_path']).name == 'video.info.json'
    assert 'requested_subtitles' not in json.loads(Path(result['info_path']).read_text())
    result = download.download_url(URL, tmp_path, context=result)
    assert Path(result['video_path']).read_bytes() == static_clip.read_bytes()
    assert not Path('elsewhere').exists()


def test_explicit_regional_language_is_not_silently_replaced():
    assert download.select_caption({'subtitles': _tracks('en-US')}, 'en-GB') is None
    assert download.select_caption({'subtitles': _tracks('en-US')}, 'en')['key'] == 'en-US'
