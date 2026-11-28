"""Apple SpeechAnalyzer backend: language precedence, and an opt-in real run on macOS 26.

The real run may download an OS speech model, so it only runs with WATCH_TEST_APPLE_SPEECH=1.
"""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys

import pytest

import config
import whisper


posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell stand-in for the helper")


@pytest.fixture
def fake_helper(monkeypatch, tmp_path):
    """Stand-in for the compiled helper: echoes the language hint it received."""
    helper = tmp_path / "apple_speech"
    helper.write_text(
        '#!/bin/sh\nif [ -n "$2" ]; then lang="$2"; src=given; else lang=pt-BR; src=system; fi\n'
        'printf \'{"language":"%s","language_source":"%s","segments":[{"start":0,"end":1.5,"text":"oi"}]}\' "$lang" "$src"\n'
    )
    helper.chmod(helper.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setattr(whisper, "apple_speech_binary", lambda: helper)
    monkeypatch.setattr(whisper, "audio_duration", lambda _: 1.5)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "missing.env")
    monkeypatch.delenv("WATCH_APPLE_LOCALE", raising=False)
    return tmp_path / "audio.mp3"


@posix_only
def test_metadata_language_reaches_helper(fake_helper):
    segments, used = whisper.transcribe_apple(fake_helper, "en", "ignored title")
    assert used == "apple"
    assert segments == [{"start": 0.0, "end": 1.5, "text": "oi"}]
    assert (segments.language, segments.language_source) == ("en", "declared by the video")


@posix_only
def test_configured_locale_wins_over_metadata(fake_helper, monkeypatch):
    monkeypatch.setenv("WATCH_APPLE_LOCALE", "pt-BR")
    segments, _ = whisper.transcribe_apple(fake_helper, "en")
    assert (segments.language, segments.language_source) == ("pt-BR", "set by WATCH_APPLE_LOCALE")


@posix_only
def test_no_hint_is_reported_as_unverified(fake_helper):
    segments, _ = whisper.transcribe_apple(fake_helper, None)
    assert segments.language_source == "system default, unverified"


@pytest.mark.skipif(os.environ.get("WATCH_TEST_APPLE_SPEECH") != "1" or not whisper.apple_speech_available()
                    or shutil.which("say") is None,
                    reason="opt-in (WATCH_TEST_APPLE_SPEECH=1); needs macOS 26+, swiftc and say")
def test_transcribes_real_speech(tmp_path, monkeypatch):
    monkeypatch.setattr(whisper, "APPLE_SPEECH_CACHE", tmp_path / "cache")
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "missing.env")
    monkeypatch.delenv("WATCH_APPLE_LOCALE", raising=False)
    audio = tmp_path / "speech.aiff"
    voice = subprocess.run(["say", "-v", "Samantha", "-o", str(audio),
                            "The quick brown fox jumps over the lazy dog. [[slnc 2000]] Where did the fox go?"],
                           capture_output=True)
    if voice.returncode != 0:
        pytest.skip("Samantha voice unavailable")
    # No declared language: it must come from the title text, whatever the system locale is.
    segments, used = whisper.transcribe_apple(audio, None, "NASA ScienceCasts: Unlocking the Origins of the Universe")
    assert (used, segments.language, segments.language_source) == ("apple", "en-US", "detected from title/description")
    assert len(segments) >= 2  # split at the pause, not one block per result
    assert "fox" in " ".join(s["text"] for s in segments).lower()
