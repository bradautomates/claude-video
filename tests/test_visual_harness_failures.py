"""Focused failure diagnostics regressions for the visual harness."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

import visual_harness
from test_visual_harness import _fake, _request, _work


SECRET = "RAW_PROVIDER_BODY_SHOULD_NOT_ESCAPE"


def _patch_fake(claude: Path, old: str, new: str) -> None:
    script = claude.parent / "fake-claude.py"
    source = script.read_text(encoding="utf-8")
    assert old in source
    script.write_text(source.replace(old, new, 1), encoding="utf-8")


def _add_api_404_failure(claude: Path) -> None:
    marker = 'if mode == "result_only":\n'
    injected = (
        'if mode == "api_404":\n'
        '    result["subtype"] = "error"\n'
        '    result["is_error"] = True\n'
        '    result["api_error_status"] = 404\n'
        f'    result["error"] = {{"message": {json.dumps(SECRET)}}}\n'
    )
    _patch_fake(claude, marker, injected + marker)


def _add_process_failure(claude: Path, mode: str) -> None:
    marker = 'if mode == "timeout":\n'
    action = (
        "    raise SystemExit(23)\n"
        if mode == "exit"
        else "    os.kill(os.getpid(), 2)\n"
    )
    injected = f'if mode == "{mode}":\n' + action

    _patch_fake(claude, marker, injected + marker)


def _assert_no_evidence(work: Path) -> None:
    assert not (work / "watch-review-v2.json").exists()
    evidence = work / "evidence"
    if evidence.exists():
        assert not list(evidence.glob("page-*.json"))
        assert not (evidence / "final.json").exists()
        assert not (evidence / "lineage.json").exists()
        assert not list(evidence.glob(".review-v2.stage.*"))


def _launch_count(log: Path) -> int:
    if not log.exists():
        return 0
    return len(log.read_text(encoding="utf-8").splitlines())


def test_inspect_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    work = _work(tmp_path)
    image = work / "overview/overview_0001.jpg"
    image.unlink()
    os.mkfifo(image)
    started = time.monotonic()

    with pytest.raises(visual_harness.HarnessError, match="bounded regular file"):
        visual_harness.inspect_work(work)

    assert time.monotonic() - started < 1


def test_api_404_is_bounded_without_raw_output_leakage(tmp_path: Path) -> None:
    work = _work(tmp_path)
    inspection = visual_harness.inspect_work(work)
    claude, _log = _fake(tmp_path, "api_404")
    _add_api_404_failure(claude)

    with pytest.raises(visual_harness.ChildFailure) as caught:
        visual_harness.review_work(
            work,
            _request(inspection, claude),
            claude_bin=claude,
        )

    failure = caught.value
    assert failure.category == "api_not_found"
    assert failure.envelope["api_status"] == 404
    assert SECRET not in str(failure)
    assert SECRET not in json.dumps(failure.envelope)
    _assert_no_evidence(work)


@pytest.mark.parametrize(
    ("per_child", "review", "match"),
    (
        (visual_harness.MIN_PER_CHILD_TIMEOUT_SECONDS - 0.001, 1.0, "outside"),
        (1.0, visual_harness.MIN_REVIEW_TIMEOUT_SECONDS - 0.001, "outside"),
        (0.2, 0.1, "shorter"),
    ),
)
def test_request_timeout_bounds_are_enforced(
    tmp_path: Path,
    per_child: float,
    review: float,
    match: str,
) -> None:
    work = _work(tmp_path)
    inspection = visual_harness.inspect_work(work)
    claude, _log = _fake(tmp_path, "valid")
    request = _request(inspection, claude)
    request = replace(
        request,
        per_child_timeout_seconds=per_child,
        review_timeout_seconds=review,
    )

    with pytest.raises(visual_harness.HarnessError, match=match):
        visual_harness.review_work(work, request, claude_bin=claude)

    _assert_no_evidence(work)


def test_oversized_question_is_rejected_before_child_launch(tmp_path: Path) -> None:
    work = _work(tmp_path)
    inspection = visual_harness.inspect_work(work)
    claude, log = _fake(tmp_path)
    request = replace(
        _request(inspection, claude),
        question="q" * (visual_harness.MAX_QUESTION_BYTES + 1),
    )

    with pytest.raises(visual_harness.HarnessError, match="bounded string"):
        visual_harness.review_work(work, request, claude_bin=claude)

    assert not log.exists()
    _assert_no_evidence(work)


def test_bidi_timing_failure_publishes_no_evidence(tmp_path: Path) -> None:
    work = _work(tmp_path)
    inspection = visual_harness.inspect_work(work)
    claude, log = _fake(tmp_path, "page_bidi_timing_literal")

    with pytest.raises(visual_harness.ChildFailure) as caught:
        visual_harness.review_work(
            work,
            _request(inspection, claude),
            claude_bin=claude,
        )

    assert caught.value.category == "protocol"
    assert caught.value.envelope["protocol_stage"] == "page_report"
    assert _launch_count(log) == 1
    _assert_no_evidence(work)


def test_review_deadline_failure_is_bounded_and_publishes_no_evidence(
    tmp_path: Path,
) -> None:
    work = _work(tmp_path)
    inspection = visual_harness.inspect_work(work)
    claude, _log = _fake(tmp_path, "timeout")
    request = _request(inspection, claude)
    request = replace(
        request,
        per_child_timeout_seconds=0.15,
        review_timeout_seconds=0.15,
    )

    with pytest.raises(visual_harness.ChildFailure) as caught:
        visual_harness.review_work(work, request, claude_bin=claude)

    assert caught.value.category == "review_deadline"
    assert caught.value.envelope["phase"] == "page"
    _assert_no_evidence(work)


def test_process_exit_is_a_bounded_failure(tmp_path: Path) -> None:
    work = _work(tmp_path)
    inspection = visual_harness.inspect_work(work)
    claude, _log = _fake(tmp_path, "exit")
    _add_process_failure(claude, "exit")

    with pytest.raises(visual_harness.ChildFailure) as caught:
        visual_harness.review_work(
            work,
            _request(inspection, claude),
            claude_bin=claude,
        )

    assert caught.value.category == "process_exit"
    assert caught.value.envelope["exit_code"] == 23
    _assert_no_evidence(work)


def test_prompt_too_long_is_bounded_without_raw_diagnostic_leakage(
    tmp_path: Path,
) -> None:
    work = _work(tmp_path)
    inspection = visual_harness.inspect_work(work)
    claude, log = _fake(tmp_path, "context_limit_stderr")

    with pytest.raises(visual_harness.ChildFailure) as caught:
        visual_harness.review_work(
            work,
            _request(inspection, claude),
            claude_bin=claude,
        )

    failure = caught.value
    assert failure.category == "context_limit"
    assert failure.envelope["reason"] == "Claude child context limit was exceeded"
    assert "Prompt is too long" not in str(failure)
    assert "RAW_CONTEXT_DIAGNOSTIC" not in json.dumps(failure.envelope)
    assert _launch_count(log) == 1
    _assert_no_evidence(work)


def test_prompt_too_long_before_settings_is_context_limit(tmp_path: Path) -> None:
    work = _work(tmp_path)
    inspection = visual_harness.inspect_work(work)
    claude, log = _fake(tmp_path, "context_limit_before_settings")

    with pytest.raises(visual_harness.ChildFailure) as caught:
        visual_harness.review_work(
            work,
            _request(inspection, claude),
            claude_bin=claude,
        )

    assert caught.value.category == "context_limit"
    assert _launch_count(log) == 1
    _assert_no_evidence(work)


def test_oversized_serialized_child_input_is_rejected_before_launch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    work = _work(tmp_path)
    inspection = visual_harness.inspect_work(work)
    claude, log = _fake(tmp_path)
    monkeypatch.setattr(visual_harness.runtime, "MAX_CHILD_INPUT_BYTES", 128)

    with pytest.raises(visual_harness.ChildFailure) as caught:
        visual_harness.review_work(
            work,
            _request(inspection, claude),
            claude_bin=claude,
        )

    assert caught.value.category == "context_limit"
    assert not log.exists()
    _assert_no_evidence(work)


def test_unknown_stderr_remains_generic_and_is_not_exposed(tmp_path: Path) -> None:
    work = _work(tmp_path)
    inspection = visual_harness.inspect_work(work)
    claude, log = _fake(tmp_path, "unknown_stderr")

    with pytest.raises(visual_harness.ChildFailure) as caught:
        visual_harness.review_work(
            work,
            _request(inspection, claude),
            claude_bin=claude,
        )

    failure = caught.value
    assert failure.category == "process_exit"
    assert failure.envelope["exit_code"] == 23
    assert "RAW_CONTEXT_DIAGNOSTIC" not in json.dumps(failure.envelope)
    assert _launch_count(log) == 1
    _assert_no_evidence(work)


def test_capture_distinguishes_direct_signal_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = visual_harness.runtime._Capture(
        stdout=b"",
        stderr=b"",
        returncode=-15,
        elapsed_seconds=0.01,
        timed_out=False,
        overflow=False,
        input_failed=False,
    )
    monkeypatch.setattr(
        visual_harness.runtime, "_execute", lambda *args, **kwargs: capture
    )

    with pytest.raises(visual_harness.ChildFailure) as caught:
        visual_harness.runtime.run_child(
            object(),
            object(),
            model="claude-test",
            effort="high",
            budget=1.0,
            schema={},
            system_prompt="test",
            message={},
            timeout_seconds=1.0,
            phase="page",
            child_ordinal=1,
        )

    assert caught.value.category == "signal"
    assert caught.value.envelope["signal"] == 15


def test_failure_launches_exactly_once_without_retry(tmp_path: Path) -> None:
    work = _work(tmp_path)
    inspection = visual_harness.inspect_work(work)
    claude, log = _fake(tmp_path, "exit")
    _add_process_failure(claude, "exit")

    with pytest.raises(visual_harness.ChildFailure):
        visual_harness.review_work(
            work,
            _request(inspection, claude),
            claude_bin=claude,
        )

    assert _launch_count(log) == 1
    _assert_no_evidence(work)


def test_descendant_escaping_process_group_is_killed(
    tmp_path: Path,
) -> None:
    work = _work(tmp_path)
    inspection = visual_harness.inspect_work(work)
    claude, _log = _fake(tmp_path, "detached")
    marker = tmp_path / "escaped-marker"
    script = claude.parent / "fake-claude.py"
    source = script.read_text(encoding="utf-8")
    anchor = 'if mode == "no_read":\n'
    injected = (
        'if mode == "detached":\n'
        "    child = os.fork()\n"
        "    if child == 0:\n"
        "        os.setsid()\n"
        "        time.sleep(0.6)\n"
        f'        Path({str(marker)!r}).write_text("escaped")\n'
        "        raise SystemExit(0)\n"
        "    time.sleep(30)\n"
    )
    assert anchor in source
    script.write_text(source.replace(anchor, injected + anchor, 1), encoding="utf-8")

    with pytest.raises(visual_harness.ChildFailure, match="timed out"):
        visual_harness.review_work(
            work,
            _request(inspection, claude, per_child_timeout_seconds=0.15),
            claude_bin=claude,
        )

    time.sleep(0.8)
    assert not marker.exists()
    _assert_no_evidence(work)


def test_publish_bundle_removes_anonymous_stage_after_write_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "review"
    root.mkdir(mode=0o700)
    original_write = visual_harness.publication.os.write
    calls = 0

    def fail_second_write(descriptor: int, raw: bytes | memoryview) -> int:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("do-not-leak")
        return original_write(descriptor, raw[:1])

    monkeypatch.setattr(visual_harness.publication.os, "write", fail_second_write)
    with visual_harness.runtime.pin_directory(root) as pinned:
        with visual_harness.publication.locked_evidence(pinned) as evidence:
            with pytest.raises(visual_harness.HarnessError, match="publication failed"):
                visual_harness.publication.publish_bundle(
                    evidence,
                    b"{}\n",
                )

    assert not (root / "watch-review-v2.json").exists()


def test_publish_bundle_fails_closed_when_anonymous_staging_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "review"
    root.mkdir(mode=0o700)
    original_open = visual_harness.publication.os.open

    def fail_anonymous_open(
        path: object, flags: int, *args: object, **kwargs: object
    ) -> int:
        if path == "." and flags & getattr(os, "O_TMPFILE", 0):
            raise OSError("do-not-leak")
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(visual_harness.publication.os, "open", fail_anonymous_open)
    with visual_harness.runtime.pin_directory(root) as pinned:
        with visual_harness.publication.locked_evidence(pinned) as evidence:
            with pytest.raises(
                visual_harness.HarnessError,
                match="anonymous staging is unavailable",
            ):
                visual_harness.publication.publish_bundle(evidence, b"{}\n")

    assert not (root / "watch-review-v2.json").exists()


def test_concurrent_lock_rejects_before_publish(tmp_path: Path) -> None:
    root = tmp_path / "review"
    root.mkdir(mode=0o700)
    entered = threading.Event()
    release = threading.Event()
    errors: list[BaseException] = []

    def hold_lock() -> None:
        try:
            with visual_harness.runtime.pin_directory(root) as pinned:
                with visual_harness.publication.locked_evidence(pinned):
                    entered.set()
                    release.wait(2)
        except BaseException as exc:  # pragma: no cover - test handoff
            errors.append(exc)

    thread = threading.Thread(target=hold_lock)
    thread.start()
    assert entered.wait(2)
    try:
        with visual_harness.runtime.pin_directory(root) as pinned:
            with pytest.raises(visual_harness.HarnessError, match="in progress"):
                with visual_harness.publication.locked_evidence(pinned):
                    pass
    finally:
        release.set()
        thread.join(2)
    assert not errors
    assert not thread.is_alive()


@pytest.mark.parametrize(
    "failure",
    [
        OSError(13, "denied", "/tmp/secret-token-value"),
        visual_harness.HarnessError("denied /tmp/secret-token-value"),
    ],
)
def test_cli_normalizes_expected_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: Exception,
) -> None:
    secret = "/tmp/secret-token-value"

    def fail(_path: Path) -> dict[str, object]:
        raise failure

    monkeypatch.setattr(visual_harness, "inspect_work", fail)

    assert visual_harness.main(["inspect", "--work-dir", "/tmp/review"]) == 1
    error = json.loads(capsys.readouterr().err)
    assert error == {
        "schema_version": 1,
        "status": "blocked",
        "visual_status": "unknown",
        "category": "protocol",
        "reason": "Claude child protocol validation failed",
    }
    assert secret not in json.dumps(error)
    assert "denied" not in json.dumps(error)


def test_cli_preserves_bounded_child_failure_envelope(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    failure = visual_harness.ChildFailure(
        "api_not_found",
        phase="page",
        child_ordinal=1,
        model="claude-test",
        effort="max",
        api_status=404,
    )

    def fail(_path: Path) -> dict[str, object]:
        raise failure

    monkeypatch.setattr(visual_harness, "inspect_work", fail)

    assert visual_harness.main(["inspect", "--work-dir", "/tmp/review"]) == 1
    assert json.loads(capsys.readouterr().err) == failure.envelope


def test_duplicate_json_key_is_not_echoed() -> None:
    secret_key = "SECRET_DUPLICATE_KEY"
    raw = json.dumps({secret_key: 1})[:-1].encode() + (f',"{secret_key}":2}}'.encode())

    with pytest.raises(visual_harness.HarnessError) as caught:
        visual_harness._json_no_duplicates(
            raw,
            label="fixture",
            max_bytes=1024,
        )

    assert secret_key not in str(caught.value)


def test_inspect_runtime_cli_invokes_publication_preflight(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    claude, _log = _fake(tmp_path)
    marker = tmp_path / "publication-checked"
    launcher = tmp_path / "inspect-runtime.py"
    launcher.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(Path(visual_harness.__file__).parent)!r})\n"
        "import visual_harness\n"
        f"visual_harness.publication.check_runtime_support = lambda: "
        f"open({str(marker)!r}, 'w').write('checked')\n"
        f"raise SystemExit(visual_harness.main(['inspect-runtime', '--claude-bin', {str(claude)!r}]))\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(launcher)],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert marker.read_text(encoding="utf-8") == "checked"
    assert json.loads(result.stdout)["schema_version"] == 1


def test_runtime_preflight_checks_evidence_publication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    claude, _log = _fake(tmp_path)
    checked: list[str] = []

    def check_publication() -> None:
        checked.append("publication")

    monkeypatch.setattr(
        visual_harness.publication,
        "check_runtime_support",
        check_publication,
    )

    visual_harness.runtime.inspect_runtime(
        claude,
        publication_check=visual_harness.publication.check_runtime_support,
    )

    assert checked == ["publication"]


def test_runtime_preflight_fails_when_evidence_publication_is_unsupported(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    claude, _log = _fake(tmp_path)

    def reject_publication() -> None:
        raise visual_harness.HarnessError("unsupported publication")

    monkeypatch.setattr(
        visual_harness.publication,
        "check_runtime_support",
        reject_publication,
    )

    with pytest.raises(visual_harness.HarnessError, match="unsupported publication"):
        visual_harness.runtime.inspect_runtime(
            claude,
            publication_check=visual_harness.publication.check_runtime_support,
        )


def test_runtime_digest_redacts_credentials_and_binds_routing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    claude, _log = _fake(tmp_path)
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "first-secret")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://route-one.invalid")
    first = visual_harness.runtime.inspect_runtime(claude)["runtime_digest"]
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "second-secret")
    second = visual_harness.runtime.inspect_runtime(claude)["runtime_digest"]
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://route-two.invalid")
    third = visual_harness.runtime.inspect_runtime(claude)["runtime_digest"]

    assert first == second
    assert second != third
    assert "first-secret" not in first
    assert "second-secret" not in second
    assert "route-one" not in first


def test_runtime_digest_binds_executable_privilege_state(
    tmp_path: Path,
) -> None:
    claude, _log = _fake(tmp_path)
    first = visual_harness.runtime.inspect_runtime(claude)["runtime_digest"]
    claude.chmod(0o711)
    second = visual_harness.runtime.inspect_runtime(claude)["runtime_digest"]

    assert first != second


def test_child_environment_excludes_unapproved_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WATCH_RAW_SECRET", "do-not-forward")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "required-secret")

    environment = visual_harness.runtime.child_environment()

    assert "WATCH_RAW_SECRET" not in environment
    assert environment["ANTHROPIC_AUTH_TOKEN"] == "required-secret"
    assert environment["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "1"


@pytest.mark.parametrize(
    "model",
    (
        "claude-opus-5",
        "claude-haiku-4-5-20251001",
        "cx/gpt-5.6-luna(max)",
    ),
)
def test_exact_model_identifier_accepts_approved_forms(
    tmp_path: Path,
    model: str,
) -> None:
    work = _work(tmp_path)
    inspection = visual_harness.inspect_work(work)
    claude, _log = _fake(tmp_path)
    request = replace(_request(inspection, claude), model=model)

    visual_harness._validate_request(
        request,
        inspection,
        request.runtime_digest,
    )


def test_model_identifier_rejects_url_and_never_leaks_credentials(
    tmp_path: Path,
) -> None:
    work = _work(tmp_path)
    inspection = visual_harness.inspect_work(work)
    claude, log = _fake(tmp_path)
    credential = "user:RAW_PASSWORD"
    request = replace(
        _request(inspection, claude),
        model=f"https://{credential}@example.invalid/model",
    )

    with pytest.raises(visual_harness.HarnessError) as caught:
        visual_harness.review_work(work, request, claude_bin=claude)

    assert credential not in str(caught.value)
    assert not log.exists()
    _assert_no_evidence(work)


def test_publish_bundle_uses_no_named_staging_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "review"
    root.mkdir(mode=0o700)
    raw = b'{"schema_version":1}\n'

    with visual_harness.runtime.pin_directory(root) as pinned:
        with visual_harness.publication.locked_evidence(pinned) as evidence:
            original_mkdir = visual_harness.publication.os.mkdir
            original_open = visual_harness.publication.os.open

            def reject_stage_mkdir(
                path: object, *args: object, **kwargs: object
            ) -> None:
                if isinstance(path, str) and path.startswith(".review-v2.stage."):
                    raise AssertionError("named staging directory used")
                original_mkdir(path, *args, **kwargs)

            def reject_stage_open(path: object, *args: object, **kwargs: object) -> int:
                if isinstance(path, str) and path.startswith(".review-v2.stage."):
                    raise AssertionError("named staging directory used")
                return original_open(path, *args, **kwargs)

            monkeypatch.setattr(
                visual_harness.publication.os,
                "mkdir",
                reject_stage_mkdir,
            )
            monkeypatch.setattr(
                visual_harness.publication.os,
                "open",
                reject_stage_open,
            )
            published = visual_harness.publication.publish_bundle(evidence, raw)

    assert published["path"] == "watch-review-v2.json"
    assert published["bytes"] == len(raw)
    assert (root / "watch-review-v2.json").read_bytes() == raw


def test_verify_bundle_rejects_post_publish_replacement(
    tmp_path: Path,
) -> None:
    root = tmp_path / "review"
    root.mkdir(mode=0o700)
    raw = b"{}\n"
    with visual_harness.runtime.pin_directory(root) as pinned:
        with visual_harness.publication.locked_evidence(pinned):
            receipt = visual_harness.publication.publish_bundle(pinned, raw)
        target = root / "watch-review-v2.json"
        target.unlink()
        target.write_bytes(b'{"replacement":true}\n')
        target.chmod(0o400)

        with pytest.raises(
            visual_harness.HarnessError,
            match="identity changed|digest changed",
        ):
            visual_harness.publication.verify_bundle(pinned, receipt)

    assert target.read_bytes() == b'{"replacement":true}\n'


def test_publish_bundle_loses_target_race_without_overwrite(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "review"
    root.mkdir(mode=0o700)
    foreign = b'{"foreign":true}\n'
    original_link = visual_harness.publication.os.link

    def race_link(
        source: object,
        target: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        directory_fd = kwargs["dst_dir_fd"]
        descriptor = os.open(
            target,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=directory_fd,
        )
        try:
            os.write(descriptor, foreign)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        original_link(source, target, *args, **kwargs)

    monkeypatch.setattr(visual_harness.publication.os, "link", race_link)
    with visual_harness.runtime.pin_directory(root) as pinned:
        with visual_harness.publication.locked_evidence(pinned) as evidence:
            with pytest.raises(
                visual_harness.HarnessError,
                match="already exists",
            ):
                visual_harness.publication.publish_bundle(
                    evidence,
                    b'{"ours":true}\n',
                )

    target = root / "watch-review-v2.json"
    assert target.read_bytes() == foreign


def test_runtime_digest_binds_containment_helper_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    claude, _log = _fake(tmp_path)
    with (
        visual_harness.runtime.audit_executable(claude) as executable,
        visual_harness.runtime.audit_containment_helpers() as helpers,
    ):
        first = visual_harness.runtime.runtime_digest(executable, helpers)
        original = visual_harness.runtime._runtime_executable_record

        def changed_record(
            target: visual_harness.runtime.AuditedExecutable,
        ) -> dict[str, object]:
            record = original(target)
            if target is helpers[0]:
                return {**record, "sha256": "0" * 64}
            return record

        monkeypatch.setattr(
            visual_harness.runtime,
            "_runtime_executable_record",
            changed_record,
        )
        second = visual_harness.runtime.runtime_digest(executable, helpers)

    assert first != second


def test_execute_uses_sealed_snapshot_and_parent_death_signal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    claude, _log = _fake(tmp_path)
    with (
        visual_harness.runtime.audit_executable(claude) as executable,
        visual_harness.runtime.audit_containment_helpers() as helpers,
    ):
        captured: dict[str, object] = {}
        armed: list[int] = []

        class Process:
            stdin = None
            stdout = subprocess.PIPE
            stderr = subprocess.PIPE
            returncode = 0
            pid = os.getpid()

            def wait(self, timeout: float) -> int:
                return 0

        def fake_popen(command: list[str], **kwargs: object) -> Process:
            captured["command"] = command
            captured["pass_fds"] = kwargs["pass_fds"]
            captured["preexec_fn"] = kwargs["preexec_fn"]
            raise OSError("stop after launch inspection")

        monkeypatch.setattr(
            visual_harness.runtime,
            "_arm_parent_death_signal",
            lambda parent_pid: armed.append(parent_pid),
        )
        monkeypatch.setattr(
            visual_harness.runtime.subprocess,
            "Popen",
            fake_popen,
        )
        with pytest.raises(
            visual_harness.HarnessError,
            match="could not start",
        ):
            visual_harness.runtime._execute(
                executable,
                helpers,
                ["--version"],
                raw_input=None,
                timeout_seconds=1,
                stdout_limit=1024,
                stderr_limit=1024,
            )

    command = captured["command"]
    preexec_fn = captured["preexec_fn"]
    assert isinstance(command, list)
    assert callable(preexec_fn)
    preexec_fn()
    assert armed == [os.getpid()]
    assert "--pdeathsig" in command
    assert command[command.index("--pdeathsig") + 1] == "KILL"
    assert all(
        str(path) not in command
        for path in (claude, Path("/usr/bin/setpriv"), Path("/usr/bin/unshare"))
    )
    assert len(captured["pass_fds"]) == 3


@pytest.mark.parametrize(
    ("cleanup_ok", "expected_error"),
    ((True, KeyboardInterrupt), (False, visual_harness.ChildFailure)),
)
def test_execute_cleans_up_when_wait_is_interrupted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    cleanup_ok: bool,
    expected_error: type[BaseException],
) -> None:
    claude, _log = _fake(tmp_path)
    with (
        visual_harness.runtime.audit_executable(claude) as executable,
        visual_harness.runtime.audit_containment_helpers() as helpers,
    ):

        class Stream:
            def read(self, _size: int) -> bytes:
                return b""

            def close(self) -> None:
                pass

        class Process:
            stdin = None
            stdout = Stream()
            stderr = Stream()
            returncode = None
            pid = 424242

            def wait(self, timeout: float) -> int:
                raise KeyboardInterrupt

        stopped: list[int] = []
        monkeypatch.setattr(
            visual_harness.runtime.subprocess,
            "Popen",
            lambda *_args, **_kwargs: Process(),
        )
        monkeypatch.setattr(
            visual_harness.runtime,
            "_stop_and_reap",
            lambda process: stopped.append(process.pid) or cleanup_ok,
        )

        with pytest.raises(expected_error) as caught:
            visual_harness.runtime._execute(
                executable,
                helpers,
                ["--version"],
                raw_input=None,
                timeout_seconds=1,
                stdout_limit=1024,
                stderr_limit=1024,
            )

    assert stopped == [424242]
    if not cleanup_ok:
        assert isinstance(caught.value, visual_harness.ChildFailure)
        assert caught.value.category == "cleanup"
