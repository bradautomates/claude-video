"""Descriptor-relative publication regressions."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import visual_harness


def test_directory_fsync_failure_leaves_recoverable_committed_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "review"
    root.mkdir(mode=0o700)
    raw = b"{}\n"
    original_fsync = visual_harness.publication.os.fsync
    failed = False

    def fail_first_directory_fsync(descriptor: int) -> None:
        nonlocal failed
        if not failed and descriptor == pinned.descriptor:
            failed = True
            raise OSError("do-not-leak")
        original_fsync(descriptor)

    with visual_harness.runtime.pin_directory(root) as pinned:
        monkeypatch.setattr(
            visual_harness.publication.os,
            "fsync",
            fail_first_directory_fsync,
        )
        with visual_harness.publication.locked_evidence(pinned):
            with pytest.raises(
                visual_harness.HarnessError,
                match="evidence publication failed",
            ):
                visual_harness.publication.publish_bundle(pinned, raw)

        target = root / "watch-review-v2.json"
        assert target.read_bytes() == raw
        assert target.stat().st_mode & 0o777 == 0o400
        assert visual_harness.publication.committed_bundle_exists(pinned)
        with pytest.raises(
            visual_harness.HarnessError,
            match="committed evidence already exists",
        ):
            with visual_harness.publication.locked_evidence(pinned):
                pass


def test_primary_write_failure_precedes_stage_close_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "review"
    root.mkdir(mode=0o700)
    original_open = visual_harness.publication.os.open
    original_close = visual_harness.publication.os.close
    stage_descriptor: int | None = None

    def record_open(path: object, flags: int, *args: object, **kwargs: object) -> int:
        nonlocal stage_descriptor
        descriptor = original_open(path, flags, *args, **kwargs)
        if path == "." and flags & getattr(os, "O_TMPFILE", 0):
            stage_descriptor = descriptor
        return descriptor

    def fail_write(_descriptor: int, _raw: bytes | memoryview) -> int:
        raise OSError("primary-secret")

    def fail_close(descriptor: int) -> None:
        original_close(descriptor)
        if descriptor == stage_descriptor:
            raise OSError("cleanup-secret")

    monkeypatch.setattr(visual_harness.publication.os, "open", record_open)
    monkeypatch.setattr(visual_harness.publication.os, "write", fail_write)
    monkeypatch.setattr(visual_harness.publication.os, "close", fail_close)
    with visual_harness.runtime.pin_directory(root) as pinned:
        with visual_harness.publication.locked_evidence(pinned):
            with pytest.raises(
                visual_harness.HarnessError,
                match="evidence publication failed",
            ) as caught:
                visual_harness.publication.publish_bundle(pinned, b"{}\n")

    assert "primary-secret" not in str(caught.value)
    assert "cleanup-secret" not in str(caught.value)
    assert not (root / "watch-review-v2.json").exists()


def test_publication_close_failure_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "review"
    root.mkdir(mode=0o700)
    original_open = visual_harness.publication.os.open
    original_close = visual_harness.publication.os.close
    stage_descriptor: int | None = None

    def record_stage_open(
        path: object,
        flags: int,
        *args: object,
        **kwargs: object,
    ) -> int:
        nonlocal stage_descriptor
        descriptor = original_open(path, flags, *args, **kwargs)
        if path == "." and flags & getattr(os, "O_TMPFILE", 0):
            stage_descriptor = descriptor
        return descriptor

    def fail_stage_close(descriptor: int) -> None:
        if descriptor == stage_descriptor:
            original_close(descriptor)
            raise OSError("cleanup-failed")
        original_close(descriptor)

    monkeypatch.setattr(visual_harness.publication.os, "open", record_stage_open)
    monkeypatch.setattr(visual_harness.publication.os, "close", fail_stage_close)
    with visual_harness.runtime.pin_directory(root) as pinned:
        with visual_harness.publication.locked_evidence(pinned) as evidence:
            with pytest.raises(
                visual_harness.HarnessError,
                match="anonymous staging cleanup failed",
            ):
                visual_harness.publication.publish_bundle(evidence, b"{}\n")

    target = root / "watch-review-v2.json"
    assert target.exists()
    assert target.stat().st_mode & 0o777 == 0o400


def test_verify_bundle_reports_descriptor_close_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "review"
    root.mkdir(mode=0o700)
    raw = b"{}\n"
    with visual_harness.runtime.pin_directory(root) as pinned:
        with visual_harness.publication.locked_evidence(pinned):
            receipt = visual_harness.publication.publish_bundle(pinned, raw)
        original_close = visual_harness.publication.os.close
        target_descriptor: int | None = None
        original_open = visual_harness.publication.os.open

        def record_open(
            path: object, flags: int, *args: object, **kwargs: object
        ) -> int:
            nonlocal target_descriptor
            descriptor = original_open(path, flags, *args, **kwargs)
            if path == visual_harness.publication.TARGET_NAME:
                target_descriptor = descriptor
            return descriptor

        def fail_close(descriptor: int) -> None:
            original_close(descriptor)
            if descriptor == target_descriptor:
                raise OSError("do-not-leak")

        monkeypatch.setattr(visual_harness.publication.os, "open", record_open)
        monkeypatch.setattr(visual_harness.publication.os, "close", fail_close)
        with pytest.raises(
            visual_harness.HarnessError,
            match="committed evidence cleanup failed",
        ):
            visual_harness.publication.verify_bundle(pinned, receipt)


@pytest.mark.parametrize("failure_point", ["file_fsync", "link"])
def test_precommit_publication_failures_leave_no_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure_point: str,
) -> None:
    root = tmp_path / "review"
    root.mkdir(mode=0o700)
    original_fsync = visual_harness.publication.os.fsync

    with visual_harness.runtime.pin_directory(root) as pinned:
        if failure_point == "file_fsync":
            monkeypatch.setattr(
                visual_harness.publication.os,
                "fsync",
                lambda descriptor: (
                    (_ for _ in ()).throw(OSError("do-not-leak"))
                    if descriptor != pinned.descriptor
                    else original_fsync(descriptor)
                ),
            )
        else:
            monkeypatch.setattr(
                visual_harness.publication.os,
                "link",
                lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("do-not-leak")),
            )
        with visual_harness.publication.locked_evidence(pinned):
            with pytest.raises(
                visual_harness.HarnessError,
                match="evidence publication failed",
            ):
                visual_harness.publication.publish_bundle(pinned, b"{}\n")

    assert not (root / "watch-review-v2.json").exists()
