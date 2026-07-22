"""Bounded visual presentation and complete frame-index contracts."""
from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import pytest

import presentation


def _frames(tmp_path: Path, count: int) -> list[dict]:
    frames = []
    for index in range(count):
        path = tmp_path / "frames" / f"frame_{index:04d}.jpg"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(f"frame-{index}".encode())
        frames.append({
            "index": index,
            "timestamp_seconds": index * 0.5,
            "path": str(path),
            "reason": "transcript-cue" if index in {7, 31} else "scene-change",
        })
    return frames


@pytest.mark.parametrize(
    ("count", "page_sizes"),
    [(0, []), (1, [1]), (20, [20]), (21, [20, 1]), (100, [20, 20, 20, 20, 20])],
)
def test_overview_pages_are_bounded_complete_and_non_mutating(
    tmp_path: Path, count: int, page_sizes: list[int],
):
    frames = _frames(tmp_path, count)
    original = copy.deepcopy(frames)

    pages = presentation.paginate_frames(frames)

    assert [len(page) for page in pages] == page_sizes
    assert [frame for page in pages for frame in page] == original
    assert frames == original


def test_manifest_preserves_all_frames_and_maps_every_tile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    frames = _frames(tmp_path, 53)
    original = copy.deepcopy(frames)

    def fake_contact_sheet(page: list[dict], output_path: Path) -> Path:
        assert 0 < len(page) <= 20
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"overview")
        return output_path

    monkeypatch.setattr(presentation, "create_contact_sheet", fake_contact_sheet)
    result = presentation.prepare_frame_presentation(
        tmp_path,
        source_path="/retained/source.mp4",
        source_meta={"duration_seconds": 26.5, "width": 320, "height": 240, "codec": "h264"},
        frames=frames,
    )

    manifest = json.loads(Path(result["index_path"]).read_text())
    pages = manifest["overview"]["pages"]
    assert manifest["schema_version"] == 1
    assert manifest["frames"] == original
    assert frames == original
    assert manifest["overview"]["page_count"] == 3
    assert [page["page"] for page in pages] == [1, 2, 3]
    assert [tile["frame_index"] for page in pages for tile in page["tiles"]] == list(range(53))
    assert all(Path(page["path"]).is_file() for page in pages)


def test_contact_sheet_uses_shell_free_subprocess(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    frames = _frames(tmp_path, 2)
    output_path = tmp_path / "overview.jpg"
    calls = []

    class Result:
        returncode = 0
        stderr = ""

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        output_path.write_bytes(b"overview")
        return Result()

    monkeypatch.setattr(presentation.subprocess, "run", fake_run)
    assert presentation.create_contact_sheet(frames, output_path) == output_path
    command, kwargs = calls[0]
    assert isinstance(command, list)
    assert kwargs["shell"] is False


def test_presentation_failure_invalidates_manifest_and_preserves_frames(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    frames = _frames(tmp_path, 25)
    paths = [Path(frame["path"]) for frame in frames]
    (tmp_path / "frame-index.json").write_text('{"stale": true}')

    def fail_contact_sheet(*_args, **_kwargs):
        raise presentation.PresentationError("render failed")

    monkeypatch.setattr(presentation, "create_contact_sheet", fail_contact_sheet)

    with pytest.raises(presentation.PresentationError, match="render failed"):
        presentation.prepare_frame_presentation(
            tmp_path, source_path=None, source_meta={}, frames=frames,
        )

    assert not (tmp_path / "frame-index.json").exists()
    assert all(path.read_bytes() for path in paths)


def test_mixed_dimensions_render_one_overview(tmp_path: Path):
    frames = []
    for index, size in enumerate(("320x240", "640x180")):
        path = tmp_path / "frames" / f"frame_{index:04d}.jpg"
        path.parent.mkdir(exist_ok=True)
        result = presentation.subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"color=c=blue:s={size}",
            "-frames:v", "1", str(path),
        ])
        assert result.returncode == 0
        frames.append({
            "index": index,
            "timestamp_seconds": float(index),
            "path": str(path),
            "reason": "scene-change",
        })

    output = presentation.create_contact_sheet(frames, tmp_path / "overview.jpg")

    assert output.is_file()


def test_rejects_symlinked_overview_directory(tmp_path: Path):
    frames = _frames(tmp_path, 1)
    external = tmp_path / "external"
    external.mkdir()
    (tmp_path / "overview").symlink_to(external, target_is_directory=True)

    with pytest.raises(presentation.PresentationError, match="must not be a symlink"):
        presentation.prepare_frame_presentation(tmp_path, None, {}, frames)

    assert list(external.iterdir()) == []


def test_paginate_rejects_zero_page_size():
    with pytest.raises(ValueError, match="greater than zero"):
        presentation.paginate_frames([], page_size=0)


def test_contact_sheet_rejects_empty_page(tmp_path: Path):
    with pytest.raises(presentation.PresentationError, match="empty overview"):
        presentation.create_contact_sheet([], tmp_path / "overview.jpg")


def test_contact_sheet_rejects_missing_frame(tmp_path: Path):
    frames = [{
        "index": 0,
        "timestamp_seconds": 0.0,
        "path": str(tmp_path / "missing.jpg"),
        "reason": "scene-change",
    }]

    with pytest.raises(presentation.PresentationError, match="unavailable"):
        presentation.create_contact_sheet(frames, tmp_path / "overview.jpg")


@pytest.mark.parametrize(
    "error",
    [OSError("missing ffmpeg"), subprocess.TimeoutExpired("ffmpeg", 120)],
)
def test_contact_sheet_wraps_subprocess_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, error: Exception,
):
    frames = _frames(tmp_path, 1)
    monkeypatch.setattr(presentation.subprocess, "run", lambda *_args, **_kwargs: (_ for _ in ()).throw(error))

    with pytest.raises(presentation.PresentationError, match="ffmpeg overview generation failed"):
        presentation.create_contact_sheet(frames, tmp_path / "overview.jpg")


def test_contact_sheet_wraps_nonzero_ffmpeg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    frames = _frames(tmp_path, 1)

    class Result:
        returncode = 1
        stderr = "invalid image"

    monkeypatch.setattr(presentation.subprocess, "run", lambda *_args, **_kwargs: Result())

    with pytest.raises(presentation.PresentationError, match="invalid image"):
        presentation.create_contact_sheet(frames, tmp_path / "overview.jpg")


def test_atomic_write_failure_removes_temporary_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    target = tmp_path / "frame-index.json"
    monkeypatch.setattr(presentation.os, "replace", lambda *_args: (_ for _ in ()).throw(OSError("replace failed")))

    with pytest.raises(OSError, match="replace failed"):
        presentation._write_json_atomic(target, {"ok": True})

    assert list(tmp_path.glob(".frame-index.json.*")) == []


def test_manifest_write_failure_is_wrapped(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    frames = _frames(tmp_path, 1)

    def fake_contact_sheet(_frames: list[dict], path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"overview")
        return path

    monkeypatch.setattr(presentation, "create_contact_sheet", fake_contact_sheet)
    monkeypatch.setattr(presentation, "_write_json_atomic", lambda *_args: (_ for _ in ()).throw(OSError("disk full")))

    with pytest.raises(presentation.PresentationError, match="could not write"):
        presentation.prepare_frame_presentation(tmp_path, None, {}, frames)
