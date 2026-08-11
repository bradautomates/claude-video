"""Focused input and child-identity trust regressions."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pytest

import _visual_stream
import visual_harness
from test_visual_harness import _fake, _request, _transcript_work, _work


@pytest.mark.parametrize(
    "model",
    [
        "openrouter/auto",
        "claude-sonnet-latest",
        "claude-sonnet",
        "claude-sonnet-4-5",
        "openrouter/sonnet",
        "openrouter/sonnet4",
        "cx/fable",
        "cx/fable5",
        "cx/sonnet(max)",
    ],
)
def test_review_request_rejects_routing_model_aliases(model: str) -> None:
    digest = "a" * 64
    runtime_digest = "b" * 64
    inspection = {
        "digest": digest,
        "page_count": 1,
        "maximum_child_count": 2,
        "approval_required": False,
    }
    request = visual_harness.ReviewRequest(
        digest=digest,
        runtime_digest=runtime_digest,
        approved_page_count=1,
        approved_child_count=2,
        model=model,
        effort="max",
        per_child_budget_usd=0.1,
        aggregate_budget_usd=0.2,
        per_child_timeout_seconds=1.0,
        review_timeout_seconds=2.0,
        question="Describe the visual sequence.",
    )

    with pytest.raises(visual_harness.HarnessError, match="exact model ID"):
        visual_harness._validate_request(request, inspection, runtime_digest)


@pytest.mark.parametrize(
    "mode,category,dimension",
    [
        ("bad_effort", "identity", "effort"),
        ("missing_effort", "identity", "effort"),
        ("malformed_effort", "identity", "effort"),
        ("wrong_request_id", "protocol", None),
        ("duplicate_key_response", "protocol", None),
        ("duplicate_response", "protocol", None),
        ("oversized_settings_line", "output_limit", None),
        ("missing_applied_model", "identity", "model"),
        ("malformed_applied_model", "identity", "model"),
        ("bad_applied_model", "identity", "model"),
        ("missing_effective", "protocol", None),
        ("malformed_effective", "protocol", None),
        ("missing_sources", "protocol", None),
        ("malformed_sources", "protocol", None),
    ],
)
def test_review_rejects_unverified_effective_effort(
    tmp_path: Path,
    mode: str,
    category: str,
    dimension: str | None,
) -> None:
    work = _work(tmp_path)
    inspection = visual_harness.inspect_work(work)
    claude, _log = _fake(tmp_path, mode)

    with pytest.raises(visual_harness.ChildFailure) as caught:
        visual_harness.review_work(
            work,
            _request(inspection, claude),
            claude_bin=claude,
        )

    assert caught.value.category == category
    assert caught.value.metadata.get("identity_dimension") == dimension
    expected_keys = {
        "schema_version",
        "status",
        "visual_status",
        "category",
        "reason",
        "phase",
        "child_ordinal",
        "model",
        "effort",
        "elapsed_seconds",
    }
    if dimension:
        expected_keys.add("identity_dimension")
    assert set(caught.value.envelope) == expected_keys
    assert not _log.exists() or all(
        json.loads(line)["inputs"] == ["control_request"]
        for line in _log.read_text().splitlines()
    )
    assert not (work / "watch-review-v2.json").exists()


def test_delayed_duplicate_is_protocol_failure_after_dispatch(tmp_path: Path) -> None:
    work = _work(tmp_path)
    inspection = visual_harness.inspect_work(work)
    claude, log = _fake(tmp_path, "delayed_duplicate_response")

    with pytest.raises(visual_harness.ChildFailure) as caught:
        visual_harness.review_work(
            work,
            _request(inspection, claude),
            claude_bin=claude,
        )

    assert caught.value.category == "protocol"
    if log.exists() and log.stat().st_size:
        calls = [json.loads(line) for line in log.read_text().splitlines()]
        assert calls[0]["inputs"] == ["control_request", "user"]
    assert not (work / "watch-review-v2.json").exists()


def test_invalid_settings_failure_stops_hung_child_promptly(tmp_path: Path) -> None:
    work = _work(tmp_path)
    inspection = visual_harness.inspect_work(work)
    claude, log = _fake(tmp_path, "invalid_response_hang")

    started = visual_harness.time.monotonic()
    with pytest.raises(visual_harness.ChildFailure) as caught:
        visual_harness.review_work(
            work,
            _request(
                inspection,
                claude,
                per_child_timeout_seconds=2,
            ),
            claude_bin=claude,
        )
    elapsed = visual_harness.time.monotonic() - started

    assert caught.value.category == "identity"
    assert caught.value.metadata.get("identity_dimension") == "effort"
    assert elapsed < 1.5
    assert not log.exists() or all(
        json.loads(line)["inputs"] == ["control_request"]
        for line in log.read_text().splitlines()
    )
    assert not (work / "watch-review-v2.json").exists()


def test_settings_timeout_sends_no_user_frame_or_evidence(tmp_path: Path) -> None:
    work = _work(tmp_path)
    inspection = visual_harness.inspect_work(work)
    claude, log = _fake(tmp_path, "settings_timeout")

    with pytest.raises(visual_harness.ChildFailure) as caught:
        visual_harness.review_work(
            work,
            _request(
                inspection,
                claude,
                per_child_timeout_seconds=0.15,
            ),
            claude_bin=claude,
        )

    assert caught.value.category == "timeout"
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert calls[0]["inputs"] == ["control_request"]
    assert not (work / "watch-review-v2.json").exists()


def test_gate_timeout_atomically_blocks_late_settings_response() -> None:
    gate = _visual_stream.EffortGate("claude-test", "high")

    assert gate.wait(0) is False
    response = {
        "type": "control_response",
        "response": {
            "subtype": "success",
            "request_id": _visual_stream.REQUEST_ID,
            "response": {
                "effective": {},
                "sources": [],
                "applied": {"model": "claude-test", "effort": "high"},
            },
        },
    }
    gate.feed(json.dumps(response).encode() + b"\n")

    reader, writer = os.pipe()
    os.set_blocking(reader, False)
    try:
        assert gate.dispatch(writer, b'{"type":"user"}\n') is False
        with pytest.raises(BlockingIOError):
            os.read(reader, 1)
    finally:
        os.close(reader)
        os.close(writer)
    assert gate.failure == ("timeout", None)


def test_pre_dispatch_result_blocks_user_frame_and_evidence(tmp_path: Path) -> None:
    work = _work(tmp_path)
    inspection = visual_harness.inspect_work(work)
    claude, log = _fake(tmp_path, "pre_dispatch_result")

    with pytest.raises(visual_harness.ChildFailure) as caught:
        visual_harness.review_work(
            work,
            _request(inspection, claude),
            claude_bin=claude,
        )

    assert caught.value.category == "protocol"
    assert not log.exists() or all(
        json.loads(line)["inputs"] == ["control_request"]
        for line in log.read_text().splitlines()
    )
    assert not (work / "watch-review-v2.json").exists()


def test_pre_settings_overflow_is_prompt_and_output_limited(tmp_path: Path) -> None:
    work = _work(tmp_path)
    inspection = visual_harness.inspect_work(work)
    claude, log = _fake(tmp_path, "overflow_before_settings")

    started = visual_harness.time.monotonic()
    with pytest.raises(visual_harness.ChildFailure) as caught:
        visual_harness.review_work(
            work,
            _request(
                inspection,
                claude,
                per_child_timeout_seconds=2,
            ),
            claude_bin=claude,
        )
    elapsed = visual_harness.time.monotonic() - started

    assert caught.value.category == "output_limit"
    assert elapsed < 1.5
    assert not log.exists() or all(
        json.loads(line)["inputs"] == ["control_request"]
        for line in log.read_text().splitlines()
    )
    assert not (work / "watch-review-v2.json").exists()


def test_invalid_settings_values_are_not_exposed(tmp_path: Path) -> None:
    work = _work(tmp_path)
    inspection = visual_harness.inspect_work(work)
    claude, log = _fake(tmp_path, "settings_leak")

    with pytest.raises(visual_harness.ChildFailure) as caught:
        visual_harness.review_work(
            work,
            _request(inspection, claude),
            claude_bin=claude,
        )

    encoded = json.dumps(caught.value.envelope, sort_keys=True)
    assert caught.value.category == "identity"
    assert caught.value.envelope["identity_dimension"] == "effort"
    assert "SETTINGS_SECRET" not in encoded
    assert "private.invalid" not in encoded
    assert not log.exists() or all(
        json.loads(line)["inputs"] == ["control_request"]
        for line in log.read_text().splitlines()
    )
    assert not (work / "watch-review-v2.json").exists()


def test_cancel_before_approval_is_a_protocol_failure() -> None:
    gate = _visual_stream.EffortGate("claude-test", "high")

    gate.cancel()

    assert gate.approved is False
    assert gate.failure == ("protocol", None)
    assert gate.ready.is_set()


def test_cancel_revokes_approved_gate() -> None:
    gate = _visual_stream.EffortGate("claude-test", "high")
    response = {
        "type": "control_response",
        "response": {
            "subtype": "success",
            "request_id": _visual_stream.REQUEST_ID,
            "response": {
                "effective": {},
                "sources": [],
                "applied": {"model": "claude-test", "effort": "high"},
            },
        },
    }
    gate.feed(json.dumps(response).encode() + b"\n")
    assert gate.approved is True

    gate.cancel()

    assert gate.approved is False
    assert gate.failure == ("protocol", None)


def _approve_gate() -> _visual_stream.EffortGate:
    gate = _visual_stream.EffortGate("claude-test", "high")
    response = {
        "type": "control_response",
        "response": {
            "subtype": "success",
            "request_id": _visual_stream.REQUEST_ID,
            "response": {
                "effective": {},
                "sources": [],
                "applied": {"model": "claude-test", "effort": "high"},
            },
        },
    }
    gate.feed(json.dumps(response).encode() + b"\n")
    return gate


def test_gate_dispatch_is_single_use() -> None:
    gate = _approve_gate()
    raw = b'{"type":"user"}\n'
    reader, writer = os.pipe()
    try:
        assert gate.dispatch(writer, raw) is True
        assert gate.dispatch(writer, raw) is False
        assert os.read(reader, len(raw)) == raw
        assert os.get_blocking(writer) is False
    finally:
        os.close(reader)
        os.close(writer)
    gate.feed(b'{"type":"system","subtype":"init"}\n')
    assert gate.approved is False


def test_cancel_does_not_block_behind_dispatch_io() -> None:
    gate = _approve_gate()
    reader, writer = os.pipe()
    os.set_blocking(writer, False)
    fill = b"x" * 65536
    while True:
        try:
            os.write(writer, fill)
        except BlockingIOError:
            break
    blocked = threading.Event()
    release = threading.Event()
    result: list[bool] = []
    errors: list[BaseException] = []

    def wait_writable(_descriptor: int) -> None:
        blocked.set()
        assert release.wait(1)

    def dispatch() -> None:
        try:
            result.append(gate.dispatch(writer, b'{"type":"user"}\n', wait_writable))
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=dispatch)
    thread.start()
    assert blocked.wait(1)

    cancel = threading.Thread(target=gate.cancel)
    cancel.start()
    cancel.join(1)
    assert not cancel.is_alive()
    release.set()
    os.close(reader)
    thread.join(1)
    os.close(writer)

    assert not thread.is_alive()
    assert result == [False]
    assert errors == []
    assert gate.failure == ("protocol", None)
    assert gate.consumed is True


def test_cancelled_gate_never_dispatches_user_frame() -> None:
    gate = _approve_gate()
    gate.cancel()
    reader, writer = os.pipe()
    os.set_blocking(reader, False)
    try:
        assert gate.dispatch(writer, b'{"type":"user"}\n') is False
        with pytest.raises(BlockingIOError):
            os.read(reader, 1)
    finally:
        os.close(reader)
        os.close(writer)


def test_dispatch_wins_when_gate_state_transition_linearizes_first() -> None:
    gate = _approve_gate()
    raw = b'{"type":"user"}\n'
    reader, writer = os.pipe()
    try:
        assert gate.dispatch(writer, raw) is True
        gate.cancel()
        assert os.read(reader, len(raw)) == raw
    finally:
        os.close(reader)
        os.close(writer)

    assert gate.consumed is True
    assert gate.failure == ("protocol", None)


def test_split_pre_dispatch_result_revokes_approval() -> None:
    gate = _approve_gate()

    gate.feed(b'{"type":"res')

    assert gate.approved is False
    reader, writer = os.pipe()
    os.set_blocking(reader, False)
    try:
        assert gate.dispatch(writer, b'{"type":"user"}\n') is False
        with pytest.raises(BlockingIOError):
            os.read(reader, 1)
    finally:
        os.close(reader)
        os.close(writer)

    gate.feed(b'ult","subtype":"success"}\n')
    assert gate.failure == ("protocol", None)


def test_gate_rejects_event_racing_single_write_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = _approve_gate()
    reader, writer = os.pipe()
    entered_write = threading.Event()
    release_write = threading.Event()
    original_write = _visual_stream.os.write
    result: list[bool] = []

    def paused_write(descriptor: int, raw: bytes | memoryview) -> int:
        if descriptor == writer:
            entered_write.set()
            assert release_write.wait(1)
        return original_write(descriptor, raw)

    monkeypatch.setattr(_visual_stream.os, "write", paused_write)
    thread = threading.Thread(
        target=lambda: result.append(gate.dispatch(writer, b'{"type":"user"}\n'))
    )
    thread.start()
    assert entered_write.wait(1)

    feed = threading.Thread(
        target=lambda: gate.feed(b'{"type":"result","subtype":"success"}\n')
    )
    feed.start()
    feed.join(1)
    assert not feed.is_alive()

    release_write.set()
    thread.join(1)
    os.close(reader)
    os.close(writer)

    assert not thread.is_alive()
    assert result == [False]
    assert gate.failure == ("protocol", None)
    assert gate.dispatch_complete is False


def test_gate_rejects_event_during_incomplete_dispatch() -> None:
    gate = _approve_gate()
    reader, writer = os.pipe()
    os.set_blocking(writer, False)
    while True:
        try:
            os.write(writer, b"x" * 65536)
        except BlockingIOError:
            break
    blocked = threading.Event()
    release = threading.Event()

    def wait_writable(_descriptor: int) -> None:
        blocked.set()
        assert release.wait(1)

    results: list[bool] = []
    thread = threading.Thread(
        target=lambda: results.append(
            gate.dispatch(writer, b'{"type":"user"}\n', wait_writable)
        )
    )
    thread.start()
    assert blocked.wait(1)

    gate.feed(b'{"type":"result","subtype":"success"}\n')
    assert gate.failure == ("protocol", None)

    os.read(reader, 65536)
    release.set()
    thread.join(1)
    os.close(reader)
    os.close(writer)
    assert not thread.is_alive()
    assert results == [False]
    assert gate.dispatch_complete is False


def test_split_event_started_during_dispatch_remains_protocol_failure() -> None:
    gate = _approve_gate()
    reader, writer = os.pipe()
    os.set_blocking(writer, False)
    while True:
        try:
            os.write(writer, b"x" * 65536)
        except BlockingIOError:
            break
    blocked = threading.Event()
    release = threading.Event()
    result: list[bool] = []

    def wait_writable(_descriptor: int) -> None:
        blocked.set()
        assert release.wait(1)

    thread = threading.Thread(
        target=lambda: result.append(
            gate.dispatch(writer, b'{"type":"user"}\n', wait_writable)
        )
    )
    thread.start()
    assert blocked.wait(1)

    gate.feed(b'{"type":"res')
    os.read(reader, 65536)
    release.set()
    thread.join(1)
    assert result == [False]

    gate.feed(b'ult","subtype":"success"}\n')
    assert gate.failure == ("protocol", None)

    os.close(reader)
    os.close(writer)


def test_first_identity_failure_precedes_later_oversized_output() -> None:
    gate = _visual_stream.EffortGate("claude-test", "high")
    response = {
        "type": "control_response",
        "response": {
            "subtype": "success",
            "request_id": _visual_stream.REQUEST_ID,
            "response": {
                "effective": {},
                "sources": [],
                "applied": {"model": "claude-test", "effort": "low"},
            },
        },
    }
    raw = (
        json.dumps(response).encode()
        + b"\n"
        + b"x" * (_visual_stream.MAX_PRE_GATE_LINE_BYTES + 1)
    )

    gate.feed(raw)

    assert gate.failure == ("identity", "effort")


def test_oversized_event_after_dispatch_is_output_limit() -> None:
    gate = _approve_gate()
    reader, writer = os.pipe()
    try:
        assert gate.dispatch(writer, b'{"type":"user"}\n') is True
    finally:
        os.close(reader)
        os.close(writer)

    gate.feed(b"x" * (_visual_stream.MAX_PRE_GATE_LINE_BYTES + 1))

    assert gate.failure == ("output_limit", None)


def test_gate_preserves_first_failure() -> None:
    gate = _visual_stream.EffortGate("claude-test", "high")
    response = {
        "type": "control_response",
        "response": {
            "subtype": "success",
            "request_id": _visual_stream.REQUEST_ID,
            "response": {
                "effective": {},
                "sources": [],
                "applied": {"model": "claude-test", "effort": "low"},
            },
        },
    }
    gate.feed(json.dumps(response).encode() + b"\ntrailing")
    gate.finish()

    assert gate.failure == ("identity", "effort")


def test_nested_settings_json_is_a_bounded_protocol_failure() -> None:
    gate = _visual_stream.EffortGate("claude-test", "high")

    gate.feed(b"[" * 10000 + b"]" * 10000 + b"\n")

    assert gate.approved is False
    assert gate.failure == ("protocol", None)


def test_nested_review_request_json_is_a_bounded_harness_failure() -> None:
    raw = b'{"nested":' + b"[" * 10000 + b"]" * 10000 + b"}"

    with pytest.raises(
        visual_harness.HarnessError,
        match="review request is invalid JSON",
    ):
        visual_harness._json_no_duplicates(
            raw,
            label="review request",
            max_bytes=64 * 1024,
        )


def test_review_sends_user_frame_only_after_exact_effort_response(
    tmp_path: Path,
) -> None:
    work = _work(tmp_path)
    inspection = visual_harness.inspect_work(work)
    claude, log = _fake(tmp_path)

    visual_harness.review_work(
        work,
        _request(inspection, claude),
        claude_bin=claude,
    )

    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert calls[0]["inputs"] == ["control_request", "user"]
    assert calls[0]["response_sent"] is True
    assert calls[0]["user_ready_before_response"] is False


@pytest.mark.parametrize(
    "mode,dimension",
    [
        ("bad_model", "model"),
        ("bad_tools", "tools"),
        ("missing_init", "init_count"),
        ("duplicate_init", "init_count"),
    ],
)
def test_review_attributes_identity_failure_without_child_values(
    tmp_path: Path,
    mode: str,
    dimension: str,
) -> None:
    work = _work(tmp_path)
    inspection = visual_harness.inspect_work(work)
    claude, _log = _fake(tmp_path, mode)

    with pytest.raises(visual_harness.ChildFailure) as caught:
        visual_harness.review_work(
            work,
            _request(inspection, claude),
            claude_bin=claude,
        )

    envelope = caught.value.envelope
    assert envelope["category"] == "identity"
    assert envelope["identity_dimension"] == dimension
    encoded = json.dumps(envelope, sort_keys=True)
    assert "wrong-model" not in encoded
    assert "Read" not in encoded
    assert not (work / "watch-review-v2.json").exists()


@pytest.mark.parametrize(
    "category,dimension",
    [
        ("identity", "child-controlled-value"),
        ("identity", ["effort"]),
        ("identity", {"value": "effort"}),
        ("protocol", "effort"),
    ],
)
def test_child_failure_rejects_invalid_identity_dimension_context(
    category: str,
    dimension: object,
) -> None:
    failure = visual_harness.ChildFailure(
        category,
        identity_dimension=dimension,
    )

    assert "identity_dimension" not in failure.metadata
    assert "identity_dimension" not in failure.envelope


@pytest.mark.parametrize(
    "mode,dimension",
    [
        ("model_effort_tools", "model"),
        ("effort_tools", "tools"),
    ],
)
def test_review_reports_first_identity_failure(
    tmp_path: Path,
    mode: str,
    dimension: str,
) -> None:
    work = _work(tmp_path)
    inspection = visual_harness.inspect_work(work)
    claude, _log = _fake(tmp_path, mode)

    with pytest.raises(visual_harness.ChildFailure) as caught:
        visual_harness.review_work(
            work,
            _request(inspection, claude),
            claude_bin=claude,
        )

    assert caught.value.envelope["identity_dimension"] == dimension
    assert not (work / "watch-review-v2.json").exists()


def test_cli_preserves_fixed_identity_dimension(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    failure = visual_harness.ChildFailure(
        "identity",
        phase="page",
        child_ordinal=1,
        model="cx/gpt-5.6-luna(max)",
        effort="max",
        identity_dimension="effort",
    )

    def fail(_path: Path) -> dict[str, object]:
        raise failure

    monkeypatch.setattr(visual_harness, "inspect_work", fail)

    assert visual_harness.main(["inspect", "--work-dir", "/tmp/review"]) == 1
    error = json.loads(capsys.readouterr().err)
    assert error == failure.envelope
    assert error["identity_dimension"] == "effort"


def test_inspect_stops_at_first_aggregate_image_overage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    work = _work(tmp_path)
    original = visual_harness.inputs.read_regular
    labels: list[str] = []

    def tracked_read(
        path: Path,
        *,
        max_bytes: int,
        label: str,
        root: Path,
        root_fd: int,
    ) -> bytes:
        labels.append(label)
        return original(
            path,
            max_bytes=max_bytes,
            label=label,
            root=root,
            root_fd=root_fd,
        )

    monkeypatch.setattr(visual_harness.inputs, "read_regular", tracked_read)
    monkeypatch.setattr(visual_harness.inputs, "MAX_AGGREGATE_IMAGE_BYTES", 0)

    with pytest.raises(
        visual_harness.HarnessError,
        match="bounded regular file",
    ):
        visual_harness.inspect_work(work)

    assert labels == ["frame index", "frame image"]


@pytest.mark.parametrize("transcript", [b"", b" \n"])
def test_zero_frame_manifest_requires_transcript_evidence(
    tmp_path: Path,
    transcript: bytes,
) -> None:
    work = _transcript_work(tmp_path)
    (work / "transcript.txt").write_bytes(transcript)

    with pytest.raises(
        visual_harness.HarnessError,
        match="transcript evidence",
    ):
        visual_harness.inspect_work(work)
