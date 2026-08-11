#!/usr/bin/env python3
"""Descriptor-relative immutable evidence publication."""

from __future__ import annotations

import errno
import hashlib
import os
import stat
from contextlib import contextmanager
from typing import Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover - trusted publication is POSIX-only
    fcntl = None  # type: ignore[assignment]

from _visual_runtime import HarnessError, PinnedDirectory

TARGET_NAME = "watch-review-v2.json"
MAX_BUNDLE_BYTES = 256 * 1024


def check_runtime_support() -> None:
    """Prove locking, anonymous publication, and receipt verification work."""
    import tempfile
    from pathlib import Path

    raw = b"{}\n"
    try:
        with tempfile.TemporaryDirectory(prefix="watch-runtime-") as temporary:
            root_path = Path(temporary)
            descriptor = os.open(
                root_path,
                os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                info = os.fstat(descriptor)
                root = PinnedDirectory(
                    root_path,
                    descriptor,
                    (info.st_dev, info.st_ino),
                )
                root.assert_path_identity()
                with review_lock(root):
                    receipt = publish_bundle(root, raw)
                    if verify_bundle(root, receipt) != raw:
                        raise HarnessError("evidence publication preflight failed")
            finally:
                os.close(descriptor)
    except HarnessError:
        raise
    except OSError as exc:
        raise HarnessError("evidence publication preflight failed") from exc


def committed_bundle_exists(root: PinnedDirectory) -> bool:
    try:
        os.stat(TARGET_NAME, dir_fd=root.descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise HarnessError("committed evidence state is unavailable") from exc
    return True


@contextmanager
def review_lock(root: PinnedDirectory) -> Iterator[None]:
    if fcntl is None:
        raise HarnessError("trusted review locking is unsupported")
    try:
        info = os.fstat(root.descriptor)
    except OSError as exc:
        raise HarnessError("review lock failed") from exc
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.geteuid()
        or info.st_mode & 0o022
    ):
        raise HarnessError("work directory is unsafe for evidence")
    try:
        fcntl.flock(root.descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        if exc.errno in {errno.EACCES, errno.EAGAIN}:
            raise HarnessError("review is already in progress") from exc
        raise HarnessError("review lock failed") from exc
    try:
        yield
    finally:
        fcntl.flock(root.descriptor, fcntl.LOCK_UN)


@contextmanager
def locked_evidence(root: PinnedDirectory) -> Iterator[PinnedDirectory]:
    root.assert_path_identity()
    with review_lock(root):
        root.assert_path_identity()
        if committed_bundle_exists(root):
            raise HarnessError("committed evidence already exists")
        yield root


def _open_anonymous(root: PinnedDirectory) -> int:
    flags = os.O_RDWR | getattr(os, "O_TMPFILE", 0)
    if not getattr(os, "O_TMPFILE", 0):
        raise HarnessError("anonymous staging is unavailable")
    try:
        descriptor = os.open(".", flags, 0o600, dir_fd=root.descriptor)
    except OSError as exc:
        raise HarnessError("anonymous staging is unavailable") from exc
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.geteuid()
            or info.st_nlink != 0
            or info.st_mode & 0o777 != 0o600
        ):
            raise HarnessError("anonymous staging is unsafe")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _write_all(descriptor: int, raw: bytes) -> None:
    if not raw or len(raw) > MAX_BUNDLE_BYTES:
        raise HarnessError("evidence bundle size is invalid")
    view = memoryview(raw)
    while view:
        count = os.write(descriptor, view)
        if count < 1:
            raise HarnessError("evidence write failed")
        view = view[count:]
    os.fchmod(descriptor, 0o400)
    os.fsync(descriptor)
    info = os.fstat(descriptor)
    if info.st_size != len(raw) or info.st_nlink != 0 or info.st_mode & 0o777 != 0o400:
        raise HarnessError("anonymous evidence identity changed")


def _link_anonymous(descriptor: int, root: PinnedDirectory) -> None:
    try:
        os.link(
            f"/proc/self/fd/{descriptor}",
            TARGET_NAME,
            dst_dir_fd=root.descriptor,
            follow_symlinks=True,
        )
    except FileExistsError as exc:
        raise HarnessError("committed evidence already exists") from exc
    except OSError as exc:
        raise HarnessError("evidence publication failed") from exc
    info = os.fstat(descriptor)
    if info.st_nlink != 1 or info.st_mode & 0o777 != 0o400:
        raise HarnessError("committed evidence identity changed")


def _receipt(raw: bytes) -> dict[str, str | int]:
    return {
        "schema_version": 1,
        "path": TARGET_NAME,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


def _validate_receipt(value: dict[str, object]) -> tuple[str, int]:
    if set(value) != {"schema_version", "path", "sha256", "bytes"}:
        raise HarnessError("evidence receipt is invalid")
    digest = value["sha256"]
    size = value["bytes"]
    if (
        value["schema_version"] != 1
        or value["path"] != TARGET_NAME
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or type(size) is not int
        or not 1 <= size <= MAX_BUNDLE_BYTES
    ):
        raise HarnessError("evidence receipt is invalid")
    return digest, size


def _read_exact(descriptor: int, size: int) -> bytes:
    chunks: list[bytes] = []
    offset = 0
    while offset < size:
        chunk = os.pread(descriptor, min(64 * 1024, size - offset), offset)
        if not chunk:
            raise HarnessError("committed evidence changed")
        chunks.append(chunk)
        offset += len(chunk)
    return b"".join(chunks)


def _identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
        stat.S_IMODE(info.st_mode),
        info.st_uid,
        info.st_nlink,
    )


def verify_bundle(
    root: PinnedDirectory,
    receipt: dict[str, object],
) -> bytes:
    digest, size = _validate_receipt(receipt)
    try:
        descriptor = os.open(
            TARGET_NAME,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
            dir_fd=root.descriptor,
        )
    except OSError as exc:
        raise HarnessError("committed evidence is unavailable") from exc
    failure: BaseException | None = None
    raw = b""
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.geteuid()
            or before.st_nlink != 1
            or before.st_mode & 0o777 != 0o400
            or before.st_size != size
        ):
            raise HarnessError("committed evidence identity changed")
        raw = _read_exact(descriptor, size)
        after = os.fstat(descriptor)
        if _identity(before) != _identity(after):
            raise HarnessError("committed evidence identity changed")
        if hashlib.sha256(raw).hexdigest() != digest:
            raise HarnessError("committed evidence digest changed")
    except BaseException as exc:
        failure = exc
    try:
        os.close(descriptor)
    except OSError as exc:
        if failure is None:
            raise HarnessError("committed evidence cleanup failed") from exc
    if failure is not None:
        raise failure
    return raw


def publish_bundle(
    root: PinnedDirectory,
    raw: bytes,
) -> dict[str, str | int]:
    """Commit once linked; report any post-link durability/cleanup failure."""
    if committed_bundle_exists(root):
        raise HarnessError("committed evidence already exists")
    descriptor = _open_anonymous(root)
    failure: BaseException | None = None
    try:
        _write_all(descriptor, raw)
        _link_anonymous(descriptor, root)
        os.fsync(root.descriptor)
    except BaseException as exc:
        failure = exc
    try:
        os.close(descriptor)
    except OSError as exc:
        if failure is None:
            raise HarnessError("anonymous staging cleanup failed") from exc
    if failure is not None:
        if isinstance(failure, HarnessError):
            raise failure
        if isinstance(failure, OSError):
            raise HarnessError("evidence publication failed") from failure
        raise failure
    receipt = _receipt(raw)
    verify_bundle(root, receipt)
    return receipt
