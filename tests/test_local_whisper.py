"""On-device local Whisper backend: detection, aliasing, dispatch, and setup readiness.

conftest's autouse `isolated_user` sets WATCH_DISABLE_LOCAL_WHISPER=1 for hermeticity,
so the rest of the suite never depends on a developer's real whisper.cpp install. Every
test here opts back in via `enable_local` and fakes PATH / model dirs, so it exercises
the feature deterministically instead of the kill switch or the host machine.
"""
import json
import subprocess
from pathlib import Path

import pytest

import whisper
import setup


@pytest.fixture(autouse=True)
def enable_local(monkeypatch):
    monkeypatch.delenv("WATCH_DISABLE_LOCAL_WHISPER", raising=False)
    monkeypatch.delenv("WHISPER_MODEL", raising=False)


def _fake_which(available):
    """Stub for shutil.which: a fake path for names in `available`, else None."""
    return lambda binary: f"/usr/bin/{binary}" if binary in available else None


# --- detect_local_engine: priority order ------------------------------------

def test_detection_prefers_whisper_cpp(monkeypatch):
    monkeypatch.setattr(whisper.shutil, "which", _fake_which({"whisper-cli", "mlx_whisper", "whisper"}))
    monkeypatch.setattr(whisper, "_find_ggml_model", lambda: "/models/ggml-large-v3-turbo.bin")
    assert whisper.detect_local_engine() == {
        "engine": "whisper.cpp",
        "bin": "/usr/bin/whisper-cli",
        "model": "/models/ggml-large-v3-turbo.bin",
    }


def test_detection_skips_cpp_without_a_model_and_uses_mlx(monkeypatch):
    # cpp binary present but no ggml model on disk → cpp is unusable; mlx wins over openai-whisper.
    monkeypatch.setattr(whisper.shutil, "which", _fake_which({"whisper-cli", "mlx_whisper", "whisper"}))
    monkeypatch.setattr(whisper, "_find_ggml_model", lambda: None)
    engine = whisper.detect_local_engine()
    assert engine["engine"] == "mlx" and engine["bin"] == "/usr/bin/mlx_whisper"


def test_detection_falls_back_to_openai_whisper(monkeypatch):
    monkeypatch.setattr(whisper.shutil, "which", _fake_which({"whisper"}))
    monkeypatch.setattr(whisper, "_find_ggml_model", lambda: None)
    engine = whisper.detect_local_engine()
    assert engine["engine"] == "openai-whisper" and engine["bin"] == "/usr/bin/whisper"


def test_detection_none_when_nothing_is_installed(monkeypatch):
    monkeypatch.setattr(whisper.shutil, "which", _fake_which(set()))
    monkeypatch.setattr(whisper, "_find_ggml_model", lambda: None)
    assert whisper.detect_local_engine() is None


# --- Generic `main` disambiguation ------------------------------------------
# `main` is whisper.cpp's legacy CLI name but a generic program name: a `main`
# earlier on PATH must be verified before it's accepted (and invoked) as the
# transcription engine, so an unrelated `main` isn't run as whisper.cpp.

def _fake_help(*, stdout="", stderr="", exc=None):
    """Stub for whisper.subprocess.run used by _looks_like_whisper_cpp."""
    def run(cmd, *a, **k):
        if exc is not None:
            raise exc
        return subprocess.CompletedProcess(cmd, 0, stdout, stderr)
    return run


# A trimmed-but-representative whisper.cpp `--help` banner.
_WHISPER_CPP_HELP = (
    "usage: ./main [options] file0.wav file1.wav ...\n"
    "options:\n"
    "  -m FNAME,  --model FNAME   [models/ggml-base.en.bin] model path\n"
    "  -oj,       --output-json   [false  ] output result in a JSON file\n"
)


def test_probe_accepts_real_whisper_cpp_help(monkeypatch):
    monkeypatch.setattr(whisper.subprocess, "run", _fake_help(stdout=_WHISPER_CPP_HELP))
    assert whisper._looks_like_whisper_cpp("/usr/bin/main") is True


def test_probe_rejects_unrelated_main_help(monkeypatch):
    unrelated = "usage: main [build|run|clean]\nA toy build tool. See --help.\n"
    monkeypatch.setattr(whisper.subprocess, "run", _fake_help(stdout=unrelated))
    assert whisper._looks_like_whisper_cpp("/usr/bin/main") is False


def test_probe_fails_closed_on_launch_error(monkeypatch):
    monkeypatch.setattr(whisper.subprocess, "run", _fake_help(exc=OSError("boom")))
    assert whisper._looks_like_whisper_cpp("/usr/bin/main") is False
    monkeypatch.setattr(
        whisper.subprocess, "run",
        _fake_help(exc=subprocess.TimeoutExpired(cmd="main", timeout=10)),
    )
    assert whisper._looks_like_whisper_cpp("/usr/bin/main") is False


def test_detection_rejects_unverified_bare_main(monkeypatch):
    # Only `main` is on PATH (plus a ggml model), and probing it yields help
    # that isn't whisper.cpp — so it must NOT be detected/invoked. Before the
    # fix, bare `main` was trusted unconditionally and this returned a
    # whisper.cpp engine bound to the unrelated binary.
    monkeypatch.setattr(whisper.shutil, "which", _fake_which({"main"}))
    monkeypatch.setattr(whisper, "_find_ggml_model", lambda: "/models/ggml-base.bin")
    unrelated = "usage: main [build|run|clean]\nA toy build tool.\n"
    monkeypatch.setattr(whisper.subprocess, "run", _fake_help(stdout=unrelated))
    assert whisper.detect_local_engine() is None


def test_detection_accepts_verified_main(monkeypatch):
    monkeypatch.setattr(whisper.shutil, "which", _fake_which({"main"}))
    monkeypatch.setattr(whisper, "_find_ggml_model", lambda: "/models/ggml-base.bin")
    monkeypatch.setattr(whisper.subprocess, "run", _fake_help(stdout=_WHISPER_CPP_HELP))
    assert whisper.detect_local_engine() == {
        "engine": "whisper.cpp",
        "bin": "/usr/bin/main",
        "model": "/models/ggml-base.bin",
    }


def test_specific_cpp_names_are_trusted_without_probing(monkeypatch):
    # whisper-cli/whisper-cpp are specific enough to trust straight off PATH;
    # the probe subprocess must never run for them.
    def boom(*a, **k):
        raise AssertionError("specific cpp names must not be probed")
    monkeypatch.setattr(whisper.shutil, "which", _fake_which({"whisper-cpp"}))
    monkeypatch.setattr(whisper, "_find_ggml_model", lambda: "/models/ggml-base.bin")
    monkeypatch.setattr(whisper.subprocess, "run", boom)
    engine = whisper.detect_local_engine()
    assert engine["engine"] == "whisper.cpp" and engine["bin"] == "/usr/bin/whisper-cpp"


# --- WATCH_DISABLE_LOCAL_WHISPER kill switch --------------------------------

def test_kill_switch_disables_detection_and_reaches_cloud_key(monkeypatch):
    # A full local install is present...
    monkeypatch.setattr(whisper.shutil, "which", _fake_which({"whisper-cli"}))
    monkeypatch.setattr(whisper, "_find_ggml_model", lambda: "/models/ggml-base.bin")
    monkeypatch.setenv("WATCH_DISABLE_LOCAL_WHISPER", "1")
    # ...but the flag forces detection off...
    assert whisper.detect_local_engine() is None
    # ...so auto resolution falls through to the cloud key resolver.
    monkeypatch.setenv("GROQ_API_KEY", "test-only")
    assert whisper.load_api_key(None) == ("groq", "test-only")


# --- load_api_key: local branch ---------------------------------------------

def test_load_api_key_auto_and_explicit_prefer_local(monkeypatch):
    engine = {"engine": "whisper.cpp", "bin": "/usr/bin/whisper-cli", "model": "/m/ggml-small.bin"}
    monkeypatch.setattr(whisper, "detect_local_engine", lambda *a, **k: engine)
    assert whisper.load_api_key(None) == ("local", "/m/ggml-small.bin")
    assert whisper.load_api_key("local") == ("local", "/m/ggml-small.bin")


def test_load_api_key_local_requested_but_unavailable(monkeypatch):
    monkeypatch.setattr(whisper, "detect_local_engine", lambda *a, **k: None)
    assert whisper.load_api_key("local") == (None, None)


# --- resolve_model_alias -----------------------------------------------------

def test_resolve_model_alias_variants(monkeypatch, tmp_path):
    monkeypatch.setattr(whisper, "LOCAL_MODEL_DIRS", (str(tmp_path),))
    (tmp_path / "ggml-large-v3-turbo.bin").write_bytes(b"x")
    (tmp_path / "ggml-base.bin").write_bytes(b"x")
    custom = tmp_path / "custom.bin"
    custom.write_bytes(b"x")

    # friendly alias → ggml filename in a known dir
    assert whisper.resolve_model_alias("turbo") == str(tmp_path / "ggml-large-v3-turbo.bin")
    # bare ggml filename → known dir
    assert whisper.resolve_model_alias("ggml-base.bin") == str(tmp_path / "ggml-base.bin")
    # full path to an existing .bin → returned as-is
    assert whisper.resolve_model_alias(str(custom)) == str(custom)
    # unknown → SystemExit with a helpful message
    with pytest.raises(SystemExit, match="not found"):
        whisper.resolve_model_alias("no-such-model")


# --- _transcribe_whisper_cpp: JSON parsing ----------------------------------

def test_transcribe_whisper_cpp_converts_ms_and_skips_empty(monkeypatch, tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF")
    fixture = {
        "transcription": [
            {"text": " Hello world ", "offsets": {"from": 1500, "to": 3200}},
            {"text": "   ", "offsets": {"from": 3200, "to": 4000}},          # empty → dropped
            {"text": "second", "offsets": {"from": 4000, "to": 5500}},
        ]
    }

    def fake_run(cmd, capture_output, text):
        out_base = cmd[cmd.index("-of") + 1]   # whisper.cpp writes <out_base>.json
        Path(f"{out_base}.json").write_text(json.dumps(fixture), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(whisper.subprocess, "run", fake_run)
    segments = whisper._transcribe_whisper_cpp(audio, "/usr/bin/whisper-cli", "/m/ggml-small.bin")
    assert segments == [
        {"start": 1.5, "end": 3.2, "text": "Hello world"},   # ms → s, trimmed
        {"start": 4.0, "end": 5.5, "text": "second"},
    ]


def test_transcribe_whisper_cpp_raises_when_cli_fails(monkeypatch, tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF")
    monkeypatch.setattr(whisper.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a[0], 1, "", "boom"))
    with pytest.raises(SystemExit, match="whisper-cli failed"):
        whisper._transcribe_whisper_cpp(audio, "/usr/bin/whisper-cli", "/m/ggml-small.bin")


# --- extract_audio: codec by output extension -------------------------------

def test_extract_audio_uses_wav_pcm_or_mp3(monkeypatch, tmp_path):
    monkeypatch.setattr(whisper.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    captured = {}

    def fake_run_text(cmd, *a, **k):
        captured["cmd"] = cmd
        Path(cmd[-1]).write_bytes(b"audio")   # satisfy the non-empty-output check
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(whisper, "run_text", fake_run_text)

    whisper.extract_audio("video.mp4", tmp_path / "a.wav")
    assert "pcm_s16le" in captured["cmd"] and "libmp3lame" not in captured["cmd"]

    whisper.extract_audio("video.mp4", tmp_path / "a.mp3")
    assert "libmp3lame" in captured["cmd"] and "pcm_s16le" not in captured["cmd"]


# --- setup._status: backend_ready for the auto-detected local backend -------

def test_setup_status_backend_ready_for_local(monkeypatch):
    present = {"whisper-cli", "ffmpeg", "ffprobe", "yt-dlp"}
    monkeypatch.setattr(setup, "_which", lambda name: f"/usr/bin/{name}" if name in present else None)
    monkeypatch.setattr(setup, "_have_local_whisper", lambda: True)
    status = setup._status()
    assert status["whisper_backend"] == "local"
    assert status["has_api_key"] is True
    assert status["backend_ready"] is True


def test_setup_status_backend_not_ready_without_local_or_key(monkeypatch):
    monkeypatch.setattr(setup, "_which", lambda name: f"/usr/bin/{name}" if name in {"ffmpeg", "ffprobe", "yt-dlp"} else None)
    monkeypatch.setattr(setup, "_have_local_whisper", lambda: False)
    status = setup._status()
    assert status["has_api_key"] is False
    assert status["backend_ready"] is False
