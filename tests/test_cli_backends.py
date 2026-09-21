"""On-device CLI transcription backends: `parakeet` (parakeet-mlx preset) and
the generic `cli` (WATCH_TRANSCRIBE_CMD). Both are exercised with a stub
executable that writes a subtitle file, so no model or MLX is needed."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import config  # noqa: E402
import whisper  # noqa: E402

VTT = "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nHallo und willkommen.\n\n00:00:02.000 --> 00:00:04.500\nHeute geht es um Parakeet.\n"
SRT = "1\n00:00:00,000 --> 00:00:02,000\nHallo und willkommen.\n\n2\n00:00:02,000 --> 00:00:04,500\nHeute geht es um Parakeet.\n"


def _stub(bin_dir: Path, name: str, ext: str, body: str) -> None:
    """A fake transcriber: writes <audio stem><ext> into the --output-dir/{out_dir} arg."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    script = f'''#!/usr/bin/env python3
import sys
from pathlib import Path
args = sys.argv[1:]
audio = Path(args[0])
out_dir = Path(args[args.index("--output-dir") + 1]) if "--output-dir" in args else Path(args[-1])
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / (audio.stem + {ext!r})).write_text({body!r}, encoding="utf-8")
'''
    if os.name == "nt":
        (bin_dir / f"{name}.py").write_text(script, encoding="utf-8")
        (bin_dir / f"{name}.bat").write_text(f'@"{sys.executable}" "%~dp0{name}.py" %*\r\n', encoding="utf-8")
    else:
        p = bin_dir / name
        p.write_text(script, encoding="utf-8")
        p.chmod(0o755)


@pytest.fixture
def audio(tmp_path: Path) -> Path:
    a = tmp_path / "audio.mp3"
    a.write_bytes(b"\x00" * 10)
    return a


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for v in ("WATCH_PARAKEET_CMD", "WATCH_PARAKEET_MODEL", "WATCH_TRANSCRIBE_CMD", "WATCH_WHISPER_BACKEND"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(config, "read_env_value", lambda name, **k: os.environ.get(name))
    monkeypatch.setattr(whisper, "read_env_value", lambda name, **k: os.environ.get(name))


def test_parakeet_runs_cli_and_parses_vtt(tmp_path, audio, monkeypatch):
    _stub(tmp_path / "bin", "parakeet-mlx", ".vtt", VTT)
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    segs = whisper._transcribe_via_cli("parakeet", audio)
    assert [s["text"] for s in segs] == ["Hallo und willkommen.", "Heute geht es um Parakeet."]
    assert segs[1]["start"] == 2.0 and segs[1]["end"] == 4.5


def test_parakeet_default_model_is_tdt_v3(monkeypatch):
    assert "parakeet-tdt-0.6b-v3" in whisper._parakeet_template()
    monkeypatch.setenv("WATCH_PARAKEET_MODEL", "mlx-community/parakeet-tdt-0.6b-v2")
    assert "parakeet-tdt-0.6b-v2" in whisper._parakeet_template()


def test_parakeet_missing_binary_names_the_install(monkeypatch, audio, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path))  # nothing on it
    with pytest.raises(SystemExit) as exc:
        whisper._transcribe_via_cli("parakeet", audio)
    assert "parakeet-mlx" in str(exc.value) and "install" in str(exc.value).lower()


def test_generic_cli_accepts_srt(tmp_path, audio, monkeypatch):
    _stub(tmp_path / "bin", "mytranscriber", ".srt", SRT)
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("WATCH_TRANSCRIBE_CMD", "mytranscriber {audio} --output-dir {out_dir}")
    segs = whisper._transcribe_via_cli("cli", audio)
    assert len(segs) == 2 and segs[0]["text"] == "Hallo und willkommen."


def test_generic_cli_requires_placeholders(monkeypatch, audio):
    monkeypatch.setenv("WATCH_TRANSCRIBE_CMD", "mytranscriber {audio}")
    ok, hint = whisper.cli_backend_available("cli")
    assert not ok and "{out_dir}" in hint


def test_cli_backends_need_no_key():
    assert whisper.load_api_key(preferred="parakeet") == ("parakeet", "")
    assert whisper.load_api_key(preferred="cli") == ("cli", "")


def test_transcribe_video_skips_chunking_for_cli(tmp_path, monkeypatch):
    """A 'huge' audio must not be split for an on-device tool."""
    _stub(tmp_path / "bin", "parakeet-mlx", ".vtt", VTT)
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr(whisper, "extract_audio", lambda v, out, *a: (out.write_bytes(b"x" * 10), out)[1])
    monkeypatch.setattr(whisper, "MAX_UPLOAD_BYTES", 1)
    monkeypatch.setattr(whisper, "split_audio", lambda *a, **k: pytest.fail("must not chunk"))
    segs, used = whisper.transcribe_video("video.mp4", tmp_path / "audio.mp3", backend="parakeet", api_key="")
    assert used == "parakeet" and len(segs) == 2


def test_config_default_backend(monkeypatch):
    monkeypatch.setenv("WATCH_WHISPER_BACKEND", "parakeet")
    assert config.get_config()["whisper_backend"] == "parakeet"
    monkeypatch.setenv("WATCH_WHISPER_BACKEND", "bogus")
    assert config.get_config()["whisper_backend"] is None


def test_focused_run_extracts_only_the_window_and_shifts_timestamps(tmp_path, monkeypatch):
    """--start/--end must trim the audio before transcription (a 30 s window
    of a 30 min video should not transcribe 30 min) and return source time."""
    import subprocess
    clip = tmp_path / "tone.mp4"  # the shared fixtures have no audio track
    subprocess.run(["ffmpeg", "-v", "quiet", "-y", "-f", "lavfi", "-i", "testsrc=duration=12:size=64x64:rate=5",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=12", "-shortest", str(clip)], check=True)
    seen = {}
    real_extract = whisper.extract_audio

    def spy(video, out, start=None, end=None):
        seen["range"] = (start, end)
        return real_extract(video, out, start, end)

    monkeypatch.setattr(whisper, "extract_audio", spy)
    _stub(tmp_path / "bin", "parakeet-mlx", ".vtt", VTT)
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    segs, _ = whisper.transcribe_video(str(clip), tmp_path / "audio.mp3", backend="parakeet", api_key="",
                                       start_seconds=5.0, end_seconds=8.0)
    assert seen["range"] == (5.0, 8.0)
    assert segs[0]["start"] == 5.0 and segs[1]["end"] == 9.5  # VTT 0.0/4.5 shifted by 5 s
    dur = whisper.audio_duration(tmp_path / "audio.mp3")
    assert 2.5 < dur < 3.6
