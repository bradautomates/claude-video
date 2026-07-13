"""WHISPER_BASE_URL: point `watch` at a local/self-hosted OpenAI-compatible server."""
from __future__ import annotations

import pytest

import whisper


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    """Isolate from the developer's real environment and ~/.config/watch/.env."""
    for name in ("WHISPER_BASE_URL", "WHISPER_MODEL", "WHISPER_API_KEY",
                 "GROQ_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    # Point HOME at an empty dir so ~/.config/watch/.env cannot leak in,
    # and cwd at an empty dir so ./.env cannot either.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)


class TestResolveEndpoint:
    def test_unset_keeps_hosted_endpoint(self):
        assert whisper.resolve_endpoint(whisper.GROQ_ENDPOINT) == whisper.GROQ_ENDPOINT
        assert whisper.resolve_endpoint(whisper.OPENAI_ENDPOINT) == whisper.OPENAI_ENDPOINT

    def test_base_url_overrides_endpoint(self, monkeypatch):
        monkeypatch.setenv("WHISPER_BASE_URL", "http://127.0.0.1:8178/v1")
        assert (
            whisper.resolve_endpoint(whisper.GROQ_ENDPOINT)
            == "http://127.0.0.1:8178/v1/audio/transcriptions"
        )

    def test_trailing_slash_is_tolerated(self, monkeypatch):
        monkeypatch.setenv("WHISPER_BASE_URL", "http://127.0.0.1:8178/v1/")
        assert (
            whisper.resolve_endpoint(whisper.GROQ_ENDPOINT)
            == "http://127.0.0.1:8178/v1/audio/transcriptions"
        )

    def test_base_url_can_come_from_dotenv(self, monkeypatch, tmp_path):
        env_dir = tmp_path / ".config" / "watch"
        env_dir.mkdir(parents=True)
        (env_dir / ".env").write_text("WHISPER_BASE_URL=http://192.168.1.5:9000/v1\n")
        assert (
            whisper.resolve_endpoint(whisper.OPENAI_ENDPOINT)
            == "http://192.168.1.5:9000/v1/audio/transcriptions"
        )

    def test_env_wins_over_dotenv(self, monkeypatch, tmp_path):
        env_dir = tmp_path / ".config" / "watch"
        env_dir.mkdir(parents=True)
        (env_dir / ".env").write_text("WHISPER_BASE_URL=http://from-dotenv/v1\n")
        monkeypatch.setenv("WHISPER_BASE_URL", "http://from-env/v1")
        assert whisper.resolve_endpoint("x") == "http://from-env/v1/audio/transcriptions"


class TestBackendLabel:
    """Progress output must never claim a cloud provider when it's not being used."""

    def test_unset_shows_the_real_backend(self):
        assert whisper._backend_label("groq") == "groq"
        assert whisper._backend_label("openai") == "openai"

    def test_loopback_shows_local_not_groq(self, monkeypatch):
        monkeypatch.setenv("WHISPER_BASE_URL", "http://127.0.0.1:8178/v1")
        assert whisper._backend_label("groq") == "local"

    def test_localhost_shows_local(self, monkeypatch):
        monkeypatch.setenv("WHISPER_BASE_URL", "http://localhost:8178/v1")
        assert whisper._backend_label("groq") == "local"

    def test_remote_self_hosted_shows_the_host(self, monkeypatch):
        monkeypatch.setenv("WHISPER_BASE_URL", "http://whisper.lan:9000/v1")
        assert whisper._backend_label("openai") == "whisper.lan"


class TestLoadApiKeyWithBaseUrl:
    def test_no_key_needed_when_self_hosted(self, monkeypatch):
        """A local server needs no auth — don't hard-exit demanding a cloud key."""
        monkeypatch.setenv("WHISPER_BASE_URL", "http://127.0.0.1:8178/v1")
        backend, key = whisper.load_api_key()
        assert backend == "groq"
        assert key  # a placeholder, but truthy so the caller proceeds

    def test_explicit_key_is_used_when_provided(self, monkeypatch):
        monkeypatch.setenv("WHISPER_BASE_URL", "http://127.0.0.1:8178/v1")
        monkeypatch.setenv("WHISPER_API_KEY", "secret-token")
        _, key = whisper.load_api_key()
        assert key == "secret-token"

    def test_without_base_url_a_cloud_key_is_still_required(self):
        assert whisper.load_api_key() == (None, None)

    def test_without_base_url_groq_key_still_wins(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk_x")
        monkeypatch.setenv("OPENAI_API_KEY", "sk_y")
        assert whisper.load_api_key() == ("groq", "gsk_x")
