"""Local/self-hosted Whisper backend: endpoint resolution, precedence, auth.

No network — these exercise config resolution and the request builder only.
"""
from __future__ import annotations

import pytest
from pathlib import Path

import whisper


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch, tmp_path):
    """Neutralize ambient config so a real ~/.config/watch/.env can't leak in."""
    for var in (
        whisper.LOCAL_BASE_URL_VAR,
        whisper.LOCAL_MODEL_VAR,
        whisper.LOCAL_KEY_VAR,
        "GROQ_API_KEY",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(whisper.config, "CONFIG_FILE", tmp_path / "absent.env")
    monkeypatch.chdir(tmp_path)


def test_unset_base_url_yields_no_local_backend():
    assert whisper.local_endpoint() is None


def test_bare_origin_gets_transcriptions_route_appended(monkeypatch):
    monkeypatch.setenv(whisper.LOCAL_BASE_URL_VAR, "http://localhost:8080")
    endpoint, model, key = whisper.local_endpoint()
    assert endpoint == "http://localhost:8080/v1/audio/transcriptions"
    assert model == whisper.LOCAL_MODEL_DEFAULT
    assert key == ""


def test_trailing_slash_does_not_double_up(monkeypatch):
    monkeypatch.setenv(whisper.LOCAL_BASE_URL_VAR, "http://localhost:8080/")
    assert whisper.local_endpoint()[0] == "http://localhost:8080/v1/audio/transcriptions"


def test_full_url_is_used_verbatim(monkeypatch):
    full = "http://127.0.0.1:9000/v1/audio/transcriptions"
    monkeypatch.setenv(whisper.LOCAL_BASE_URL_VAR, full)
    assert whisper.local_endpoint()[0] == full


def test_model_and_key_overrides(monkeypatch):
    monkeypatch.setenv(whisper.LOCAL_BASE_URL_VAR, "http://localhost:8080")
    monkeypatch.setenv(whisper.LOCAL_MODEL_VAR, "large-v3")
    monkeypatch.setenv(whisper.LOCAL_KEY_VAR, "sk-local")
    _, model, key = whisper.local_endpoint()
    assert (model, key) == ("large-v3", "sk-local")


def test_local_wins_over_groq_by_default(monkeypatch):
    """Configuring a local server is the opt-in that keeps audio off the network."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk-real")
    monkeypatch.setenv(whisper.LOCAL_BASE_URL_VAR, "http://localhost:8080")
    assert whisper.load_api_key() == ("local", "")


def test_explicit_groq_still_overrides_local(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-real")
    monkeypatch.setenv(whisper.LOCAL_BASE_URL_VAR, "http://localhost:8080")
    assert whisper.load_api_key("groq") == ("groq", "gsk-real")


def test_preferred_local_without_config_returns_nothing(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-real")
    assert whisper.load_api_key("local") == (None, None)


def test_groq_openai_precedence_unchanged_when_local_absent(monkeypatch):
    """Regression guard: the patch must not disturb existing behavior."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk-real")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real")
    assert whisper.load_api_key() == ("groq", "gsk-real")


def test_base_url_readable_from_dotenv(monkeypatch, tmp_path):
    # read_env_value resolves ~/.config/watch/.env from Path.home() per call,
    # so point HOME at a scratch dir rather than patching CONFIG_FILE.
    cfg = tmp_path / ".config" / "watch"
    cfg.mkdir(parents=True)
    (cfg / ".env").write_text(f"{whisper.LOCAL_BASE_URL_VAR}=http://localhost:1234  # lm studio\n")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.chdir(tmp_path)
    assert whisper.local_endpoint()[0] == "http://localhost:1234/v1/audio/transcriptions"


def test_auth_header_omitted_when_key_is_empty(monkeypatch, tmp_path):
    """An empty bearer breaks some local servers — the header must be absent."""
    captured = {}

    class _FakeResponse:
        def read(self):
            return b'{"segments": []}'

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def _fake_urlopen(request, **kwargs):
        captured["headers"] = dict(request.headers)
        return _FakeResponse()

    monkeypatch.setattr(whisper, "urlopen", _fake_urlopen)
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"\x00" * 16)

    whisper._post_whisper("http://localhost:8080/v1/audio/transcriptions", "", "whisper-1", audio)
    assert not any(k.lower() == "authorization" for k in captured["headers"])

    whisper._post_whisper("http://localhost:8080/v1/audio/transcriptions", "k", "whisper-1", audio)
    assert captured["headers"].get("Authorization") == "Bearer k"
