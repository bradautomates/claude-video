"""Bounded protocol-stage attribution regressions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import visual_harness
from test_visual_harness import _fake, _request, _work

SECRET = "RAW_PROTOCOL_DETAIL_MUST_NOT_ESCAPE"


def _assert_no_evidence(work: Path) -> None:
    assert not (work / "watch-review-v2.json").exists()


def _assert_protocol_stage(
    failure: visual_harness.ChildFailure,
    stage: str,
    *,
    phase: str,
) -> None:
    assert failure.category == "protocol"
    assert failure.envelope["protocol_stage"] == stage
    assert failure.envelope["phase"] == phase
    assert failure.envelope["status"] == "blocked"
    assert failure.envelope["visual_status"] == "unknown"
    assert SECRET not in json.dumps(failure.envelope)


def test_malformed_child_stream_has_bounded_host_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = visual_harness.runtime._Capture(
        stdout=(SECRET + "\n").encode(),
        stderr=b"",
        returncode=0,
        elapsed_seconds=0.125,
        timed_out=False,
        overflow=False,
        input_failed=False,
    )
    monkeypatch.setattr(
        visual_harness.runtime,
        "_execute",
        lambda *_args, **_kwargs: capture,
    )

    with pytest.raises(visual_harness.ChildFailure) as caught:
        visual_harness.runtime.run_child(
            object(),
            object(),
            model="cx/gpt-5.6-luna(max)",
            effort="max",
            budget=0.1,
            schema={},
            system_prompt="bounded fixture",
            message={},
            timeout_seconds=1,
            phase="page",
            child_ordinal=2,
        )

    _assert_protocol_stage(caught.value, "stream_parse", phase="page")
    assert caught.value.envelope["child_ordinal"] == 2
    assert caught.value.envelope["model"] == "cx/gpt-5.6-luna(max)"
    assert caught.value.envelope["effort"] == "max"
    assert caught.value.envelope["elapsed_seconds"] == 0.125


def test_invalid_page_report_has_bounded_stage(tmp_path: Path) -> None:
    work = _work(tmp_path)
    inspection = visual_harness.inspect_work(work)
    claude, log = _fake(tmp_path, "invalid_page_report")

    with pytest.raises(visual_harness.ChildFailure) as caught:
        visual_harness.review_work(
            work,
            _request(inspection, claude),
            claude_bin=claude,
        )

    _assert_protocol_stage(caught.value, "page_report", phase="page")
    assert len(log.read_text(encoding="utf-8").splitlines()) == 1
    _assert_no_evidence(work)


def test_invalid_final_report_has_bounded_stage(tmp_path: Path) -> None:
    work = _work(tmp_path)
    inspection = visual_harness.inspect_work(work)
    claude, log = _fake(tmp_path, "invalid_final_report")

    with pytest.raises(visual_harness.ChildFailure) as caught:
        visual_harness.review_work(
            work,
            _request(inspection, claude),
            claude_bin=claude,
        )

    _assert_protocol_stage(caught.value, "final_report", phase="reduce")
    assert len(log.read_text(encoding="utf-8").splitlines()) == 2
    _assert_no_evidence(work)


@pytest.mark.parametrize(
    ("mode", "stage", "phase", "call_count"),
    [
        ("page_timing_literal", "page_report", "page", 1),
        ("final_timing_literal", "final_report", "reduce", 2),
    ],
)
def test_model_authored_timing_literal_fails_boundedly_without_publication(
    tmp_path: Path,
    mode: str,
    stage: str,
    phase: str,
    call_count: int,
) -> None:
    work = _work(tmp_path)
    inspection = visual_harness.inspect_work(work)
    claude, log = _fake(tmp_path, mode)

    with pytest.raises(visual_harness.ChildFailure) as caught:
        visual_harness.review_work(
            work,
            _request(inspection, claude),
            claude_bin=claude,
        )

    _assert_protocol_stage(caught.value, stage, phase=phase)
    assert "RAW_TIMING_SECRET" not in json.dumps(caught.value.envelope)
    assert len(log.read_text(encoding="utf-8").splitlines()) == call_count
    _assert_no_evidence(work)


def test_publication_failure_has_bounded_stage_without_retry_or_leakage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    work = _work(tmp_path)
    inspection = visual_harness.inspect_work(work)
    claude, log = _fake(tmp_path)

    def fail_publish(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise OSError(13, SECRET, "/tmp/private-provider-path")

    monkeypatch.setattr(
        visual_harness.publication,
        "publish_bundle",
        fail_publish,
    )

    with pytest.raises(visual_harness.ChildFailure) as caught:
        visual_harness.review_work(
            work,
            _request(inspection, claude),
            claude_bin=claude,
        )

    _assert_protocol_stage(caught.value, "host_publication", phase="publication")
    assert len(log.read_text(encoding="utf-8").splitlines()) == 2
    _assert_no_evidence(work)


@pytest.mark.parametrize(
    ("category", "stage"),
    (
        ("protocol", "child-controlled-stage"),
        ("protocol", ["stream_parse"]),
        ("identity", "stream_parse"),
    ),
)
def test_protocol_stage_rejects_unapproved_metadata(
    category: str,
    stage: object,
) -> None:
    failure = visual_harness.ChildFailure(category, protocol_stage=stage)

    assert "protocol_stage" not in failure.metadata
    assert "protocol_stage" not in failure.envelope
