"""Pinned, bounded Watch input validation."""

from __future__ import annotations

import hashlib
import os
import stat
import subprocess  # nosec B404
from pathlib import Path
from typing import Any, Callable


class InputValidationError(RuntimeError):
    pass


InputError = InputValidationError

MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_TRANSCRIPT_BYTES = 1024 * 1024
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_AGGREGATE_IMAGE_BYTES = 256 * 1024 * 1024
EXPECTED_PAGE_SIZE = 20
EXPECTED_TILE_WIDTH = 320
EXPECTED_TILE_HEIGHT = 180
TILE_COLUMNS = 4
FFMPEG_TIMEOUT_SECONDS = 120
MAX_RENDERED_OVERVIEW_BYTES = 10 * 1024 * 1024


def _render_overview(frame_bytes: list[bytes], rows: int) -> bytes:
    missing = TILE_COLUMNS * rows - len(frame_bytes)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "-framerate",
        "1",
        "-i",
        "pipe:0",
        "-vf",
        (
            f"scale={EXPECTED_TILE_WIDTH}:{EXPECTED_TILE_HEIGHT}:"
            "force_original_aspect_ratio=decrease,"
            f"pad={EXPECTED_TILE_WIDTH}:{EXPECTED_TILE_HEIGHT}:"
            "(ow-iw)/2:(oh-ih)/2:color=black,"
            f"tile={TILE_COLUMNS}x{rows}:nb_frames={len(frame_bytes)}:"
            f"init_padding={missing}"
        ),
        "-frames:v",
        "1",
        "-q:v",
        "4",
        "-f",
        "image2pipe",
        "-c:v",
        "mjpeg",
        "pipe:1",
    ]
    try:
        result = subprocess.run(  # nosec B603
            command,
            input=b"".join(frame_bytes),
            capture_output=True,
            timeout=FFMPEG_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise InputError("overview source-frame verification failed") from exc
    if (
        result.returncode != 0
        or not result.stdout
        or len(result.stdout) > MAX_RENDERED_OVERVIEW_BYTES
    ):
        raise InputError("overview source-frame verification failed")
    return result.stdout


def confined_file(
    work: Path,
    raw_path: Any,
    *,
    label: str,
    string: Callable[[Any, str, int], str],
) -> Path:
    if isinstance(raw_path, Path):
        candidate = raw_path
    else:
        path = Path(string(raw_path, label, 4096))
        candidate = path if path.is_absolute() else work / path
    if not candidate.is_absolute():
        candidate = work / candidate
    try:
        relative = candidate.relative_to(work)
    except ValueError as exc:
        raise InputError(f"{label} is outside work directory") from exc
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise InputError(f"{label} is outside work directory")
    return work.joinpath(*relative.parts)


def _open_relative_regular(root: Path, root_fd: int, path: Path) -> int:
    """Open a confined file through a pinned directory FD."""
    try:
        parts = path.relative_to(root).parts
    except ValueError as exc:
        raise InputError("file is outside work directory") from exc
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise InputError("file path is invalid")
    directory_fd = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            next_fd = os.open(
                part,
                os.O_RDONLY
                | os.O_DIRECTORY
                | os.O_NOFOLLOW
                | getattr(os, "O_NONBLOCK", 0),
                dir_fd=directory_fd,
            )
            os.close(directory_fd)
            directory_fd = next_fd
        return os.open(
            parts[-1],
            os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0),
            dir_fd=directory_fd,
        )
    finally:
        os.close(directory_fd)


def read_regular(
    path: Path,
    *,
    max_bytes: int,
    label: str,
    root: Path,
    root_fd: int,
) -> bytes:
    try:
        descriptor = _open_relative_regular(root, root_fd, path)
    except OSError as exc:
        if exc.errno in {getattr(os, "ELOOP", 40), getattr(os, "ENOTDIR", 20)}:
            raise InputError(f"{label} must be a regular non-symlink file") from exc
        raise InputError(f"{label} could not be opened safely") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > max_bytes:
            raise InputError(f"{label} is not a bounded regular file")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, max_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > max_bytes:
                raise InputError(f"{label} exceeds byte limit")
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise InputError(f"{label} changed during read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def jpeg_dimensions(data: bytes, label: str) -> tuple[int, int]:
    if not data.startswith(b"\xff\xd8\xff"):
        raise InputError(f"{label} is not JPEG data")
    offset = 2
    while offset + 4 <= len(data):
        if data[offset] != 0xFF:
            offset += 1
            continue
        marker = data[offset + 1]
        offset += 2
        if marker in {0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
            continue
        if offset + 2 > len(data):
            break
        length = int.from_bytes(data[offset : offset + 2], "big")
        if length < 2 or offset + length > len(data):
            break
        if marker in {
            0xC0,
            0xC1,
            0xC2,
            0xC3,
            0xC5,
            0xC6,
            0xC7,
            0xC9,
            0xCA,
            0xCB,
            0xCD,
            0xCE,
            0xCF,
        }:
            if length < 7:
                break
            height = int.from_bytes(data[offset + 3 : offset + 5], "big")
            width = int.from_bytes(data[offset + 5 : offset + 7], "big")
            if width < 1 or height < 1:
                break
            return width, height
        offset += length
    raise InputError(f"{label} JPEG dimensions are invalid")


def validated_manifest(
    work: Path,
    root_fd: int | None = None,
    *,
    parse_json: Callable[..., dict[str, Any]],
    exact_keys: Callable[[dict[str, Any], set[str], str], None],
    integer: Callable[[Any, str], int],
    number: Callable[[Any, str], float],
    string: Callable[[Any, str, int], str],
    bounded_list: Callable[[Any, str, int], list[Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], bytes, bytes | None]:
    if root_fd is None:
        if work.is_symlink() or not work.is_dir():
            raise InputError("work directory must be a regular non-symlink directory")
        work = work.resolve(strict=True)
        descriptor = os.open(
            work,
            os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            return validated_manifest(
                work,
                descriptor,
                parse_json=parse_json,
                exact_keys=exact_keys,
                integer=integer,
                number=number,
                string=string,
                bounded_list=bounded_list,
            )
        finally:
            os.close(descriptor)
    manifest_path = confined_file(
        work, work / "frame-index.json", label="frame index", string=string
    )
    manifest_raw = read_regular(
        manifest_path,
        max_bytes=MAX_MANIFEST_BYTES,
        label="frame index",
        root=work,
        root_fd=root_fd,
    )
    manifest = parse_json(
        manifest_raw, label="frame index", max_bytes=MAX_MANIFEST_BYTES
    )
    exact_keys(
        manifest,
        {"schema_version", "source", "frame_count", "frames", "overview"},
        "frame index",
    )
    if integer(manifest["schema_version"], "schema_version") != 1:
        raise InputError("frame index schema version is unsupported")
    frame_count = integer(manifest["frame_count"], "frame_count")
    frames = bounded_list(manifest["frames"], "frames", 100000)
    if frame_count != len(frames) or frame_count < 0:
        raise InputError("frame coverage is invalid")
    frame_by_index: dict[int, dict[str, Any]] = {}
    frame_bytes_by_index: dict[int, bytes] = {}
    paths: set[Path] = set()
    total_image_bytes = 0
    for expected_index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise InputError("frame record is invalid")
        exact_keys(
            frame, {"index", "timestamp_seconds", "path", "reason"}, "frame record"
        )
        index = integer(frame["index"], "frame index")
        if index != expected_index:
            raise InputError("frame indices must be contiguous")
        number(frame["timestamp_seconds"], "frame timestamp")
        string(frame["reason"], "frame reason", 128)
        path = confined_file(
            work,
            frame["path"],
            label="frame path",
            string=string,
        )
        if path in paths:
            raise InputError("image paths must be unique")
        paths.add(path)
        remaining = MAX_AGGREGATE_IMAGE_BYTES - total_image_bytes
        data = read_regular(
            path,
            max_bytes=min(MAX_IMAGE_BYTES, remaining),
            label="frame image",
            root=work,
            root_fd=root_fd,
        )
        jpeg_dimensions(data, "frame image")
        total_image_bytes += len(data)
        if total_image_bytes > MAX_AGGREGATE_IMAGE_BYTES:
            raise InputError("aggregate images exceed byte limit")
        frame_by_index[index] = {
            **frame,
            "_path": path,
            "_sha256": hashlib.sha256(data).hexdigest(),
        }
        frame_bytes_by_index[index] = data
    overview = manifest["overview"]
    if not isinstance(overview, dict):
        raise InputError("overview is invalid")
    exact_keys(
        overview,
        {"page_count", "page_size", "tile_width", "tile_height", "pages"},
        "overview",
    )
    if (
        integer(overview["page_size"], "overview page_size") != EXPECTED_PAGE_SIZE
        or integer(overview["tile_width"], "overview tile_width") != EXPECTED_TILE_WIDTH
        or integer(overview["tile_height"], "overview tile_height")
        != EXPECTED_TILE_HEIGHT
    ):
        raise InputError("overview metadata is invalid")
    pages = bounded_list(overview["pages"], "overview pages", 10000)
    if integer(overview["page_count"], "overview page_count") != len(pages) or bool(
        pages
    ) != bool(frame_count):
        raise InputError("overview page count is invalid")
    expected_indices: list[int] = []
    validated_pages: list[dict[str, Any]] = []
    for expected_page, page in enumerate(pages, 1):
        if not isinstance(page, dict):
            raise InputError("overview page is invalid")
        exact_keys(
            page,
            {
                "page",
                "frame_start",
                "frame_end",
                "frame_sha256",
                "tiles",
                "kind",
                "path",
            },
            "overview page",
        )
        if (
            integer(page["page"], "page number") != expected_page
            or page["kind"] != "image"
        ):
            raise InputError("overview page sequence is invalid")
        tiles = bounded_list(page["tiles"], "overview tiles", 20)
        if not tiles:
            raise InputError("overview page has no tiles")
        page_indices: list[int] = []
        timestamps: list[float] = []
        for expected_tile, tile in enumerate(tiles):
            if not isinstance(tile, dict):
                raise InputError("overview tile is invalid")
            exact_keys(
                tile,
                {"tile", "frame_index", "timestamp_seconds", "reason"},
                "overview tile",
            )
            if integer(tile["tile"], "tile number") != expected_tile:
                raise InputError("overview tile sequence is invalid")
            index = integer(tile["frame_index"], "tile frame index")
            if index not in frame_by_index:
                raise InputError("overview tile references unknown frame")
            timestamp = number(tile["timestamp_seconds"], "tile timestamp")
            if timestamp != number(
                frame_by_index[index]["timestamp_seconds"], "frame timestamp"
            ):
                raise InputError("overview timestamp mismatch")
            page_indices.append(index)
            timestamps.append(timestamp)
        if page_indices != list(range(page_indices[0], page_indices[-1] + 1)):
            raise InputError("overview page frame range is not contiguous")
        if (
            page["frame_start"] != page_indices[0]
            or page["frame_end"] != page_indices[-1]
        ):
            raise InputError("overview page range mismatch")
        frame_sha256 = bounded_list(
            page["frame_sha256"], "overview source frame digests", 20
        )
        if len(frame_sha256) != len(page_indices) or any(
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for digest in frame_sha256
        ):
            raise InputError("overview source frame digest is invalid")
        expected_sha256 = [frame_by_index[index]["_sha256"] for index in page_indices]
        if frame_sha256 != expected_sha256:
            raise InputError("overview source frame digest mismatch")
        path = confined_file(
            work,
            page["path"],
            label="overview path",
            string=string,
        )
        if path in paths:
            raise InputError("image paths must be unique")
        paths.add(path)
        remaining = MAX_AGGREGATE_IMAGE_BYTES - total_image_bytes
        data = read_regular(
            path,
            max_bytes=min(MAX_IMAGE_BYTES, remaining),
            label="overview image",
            root=work,
            root_fd=root_fd,
        )
        width, height = jpeg_dimensions(data, "overview image")
        expected_width = EXPECTED_TILE_WIDTH * TILE_COLUMNS
        expected_height = EXPECTED_TILE_HEIGHT * ((len(tiles) + 3) // 4)
        if (width, height) != (expected_width, expected_height):
            raise InputError("overview image dimensions are invalid")
        canonical = _render_overview(
            [frame_bytes_by_index[index] for index in page_indices],
            (len(tiles) + TILE_COLUMNS - 1) // TILE_COLUMNS,
        )
        if hashlib.sha256(data).digest() != hashlib.sha256(canonical).digest():
            raise InputError("overview image is not derived from source frames")
        total_image_bytes += len(data)
        if total_image_bytes > MAX_AGGREGATE_IMAGE_BYTES:
            raise InputError("aggregate images exceed byte limit")
        validated_pages.append(
            {
                "page": expected_page,
                "frame_start": page_indices[0],
                "frame_end": page_indices[-1],
                "indices": page_indices,
                "timestamps": timestamps,
                "path": path,
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
                "width": width,
                "height": height,
                "canonical": canonical,
                "frames": [frame_by_index[index] for index in page_indices],
            }
        )
        expected_indices.extend(page_indices)
    if expected_indices != list(range(frame_count)):
        raise InputError("overview coverage is incomplete")
    if total_image_bytes > MAX_AGGREGATE_IMAGE_BYTES:
        raise InputError("aggregate images exceed byte limit")
    transcript_path = work / "transcript.txt"
    transcript_raw: bytes | None = None
    try:
        transcript = confined_file(
            work,
            transcript_path,
            label="transcript",
            string=string,
        )
        transcript_raw = read_regular(
            transcript,
            max_bytes=MAX_TRANSCRIPT_BYTES,
            label="transcript",
            root=work,
            root_fd=root_fd,
        )
    except InputError as exc:
        if exc.__cause__ is None or getattr(exc.__cause__, "errno", None) != getattr(
            os, "ENOENT", 2
        ):
            raise
    if transcript_raw is not None:
        try:
            transcript_text = transcript_raw.decode("utf-8", "strict")
        except UnicodeDecodeError as exc:
            raise InputError("transcript is not UTF-8") from exc
    else:
        transcript_text = ""
    if frame_count == 0 and not transcript_text.strip():
        raise InputError("zero-frame review requires transcript evidence")
    return manifest, validated_pages, manifest_raw, transcript_raw
