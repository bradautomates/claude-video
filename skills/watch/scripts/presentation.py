"""Bounded visual handoff artifacts for /watch reports."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
TILES_PER_PAGE = 20
TILE_COLUMNS = 4
TILE_WIDTH = 320
TILE_HEIGHT = 180
FFMPEG_TIMEOUT_SECONDS = 120


class PresentationError(RuntimeError):
    """Raised when bounded media presentation cannot be prepared."""


def paginate_frames(frames: list[dict], page_size: int = TILES_PER_PAGE) -> list[list[dict]]:
    """Return contiguous chronological pages without changing frame dictionaries."""
    if page_size < 1:
        raise ValueError("page_size must be greater than zero")
    return [list(frames[start:start + page_size]) for start in range(0, len(frames), page_size)]


def _frame_record(frame: dict) -> dict[str, Any]:
    return {
        "index": frame["index"],
        "timestamp_seconds": frame["timestamp_seconds"],
        "path": frame["path"],
        "reason": frame.get("reason", "selected"),
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def create_contact_sheet(frames: list[dict], output_path: Path) -> Path:
    """Render one page of existing JPEGs with shell-free ffmpeg arguments."""
    if not frames:
        raise PresentationError("cannot render an empty overview page")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    inputs = [str(Path(frame["path"]).resolve()) for frame in frames]
    for path in inputs:
        if not Path(path).is_file():
            raise PresentationError(f"frame artifact is unavailable: {path}")
    rows = (len(inputs) + TILE_COLUMNS - 1) // TILE_COLUMNS
    normalized = []
    for index in range(len(inputs)):
        normalized.append(
            f"[{index}:v]scale={TILE_WIDTH}:{TILE_HEIGHT}:force_original_aspect_ratio=decrease,"
            f"pad={TILE_WIDTH}:{TILE_HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=black[v{index}]"
        )
    labels = "".join(f"[v{index}]" for index in range(len(inputs)))
    filter_graph = ";".join(normalized) + ";" + (
        f"{labels}concat=n={len(inputs)}:v=1:a=0,"
        f"tile={TILE_COLUMNS}x{rows}:nb_frames={len(inputs)}"
    )
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for path in inputs:
        command += ["-i", path]
    command += [
        "-filter_complex", filter_graph,
        "-frames:v", "1",
        "-q:v", "4",
        str(output_path),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            shell=False,
            timeout=FFMPEG_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        try:
            output_path.unlink()
        except OSError:
            pass
        raise PresentationError(f"ffmpeg overview generation failed: {exc}") from exc
    if result.returncode != 0 or not output_path.exists():
        try:
            output_path.unlink()
        except OSError:
            pass
        detail = result.stderr.strip() or "no output image was produced"
        raise PresentationError(f"ffmpeg overview generation failed: {detail}")
    return output_path


def prepare_frame_presentation(
    work: Path,
    source_path: str | None,
    source_meta: dict[str, Any],
    frames: list[dict],
) -> dict[str, Any]:
    """Write complete frame index and bounded full-coverage overview pages."""
    pages = paginate_frames(frames)
    index_path = work / "frame-index.json"
    try:
        index_path.unlink()
    except OSError:
        pass
    overview_dir = work / "overview"
    if overview_dir.is_symlink():
        raise PresentationError(f"overview directory must not be a symlink: {overview_dir}")
    for stale in overview_dir.glob("overview_*.jpg"):
        try:
            stale.unlink()
        except OSError:
            pass
    overview_pages: list[dict[str, Any]] = []
    for page_number, page in enumerate(pages, start=1):
        path = overview_dir / f"overview_{page_number:04d}.jpg"
        rendered = create_contact_sheet(page, path)
        page_record: dict[str, Any] = {
            "page": page_number,
            "frame_start": page[0]["index"],
            "frame_end": page[-1]["index"],
            "tiles": [
                {
                    "tile": tile_number,
                    "frame_index": frame["index"],
                    "timestamp_seconds": frame["timestamp_seconds"],
                    "reason": frame.get("reason", "selected"),
                }
                for tile_number, frame in enumerate(page)
            ],
        }
        page_record["kind"] = "image"
        page_record["path"] = str(rendered)
        overview_pages.append(page_record)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source": {"path": source_path, "metadata": dict(source_meta)},
        "frame_count": len(frames),
        "frames": [_frame_record(frame) for frame in frames],
        "overview": {
            "page_count": len(overview_pages),
            "page_size": TILES_PER_PAGE,
            "tile_width": TILE_WIDTH,
            "tile_height": TILE_HEIGHT,
            "pages": overview_pages,
        },
    }
    try:
        _write_json_atomic(index_path, manifest)
    except Exception as exc:
        raise PresentationError(f"could not write {index_path}: {exc}") from exc
    return {"index_path": index_path, "pages": overview_pages, "manifest": manifest}
