#!/usr/bin/env python3
"""Run bounded, tool-less visual review outside the coordinator context."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import _visual_contract as contract
import _visual_inputs as inputs
import _visual_publication as publication
import _visual_reports as reports
import _visual_runtime as runtime

SCHEMA_VERSION = contract.SCHEMA_VERSION
DEFAULT_PAGE_LIMIT = 5
MAX_CHILD_COUNT = 64
MAX_PAGE_REPORT_BYTES = reports.MAX_PAGE_REPORT_BYTES
MAX_FINAL_REPORT_BYTES = reports.MAX_FINAL_REPORT_BYTES
MAX_QUESTION_BYTES = 8 * 1024
MAX_PER_CHILD_BUDGET_USD = 5.0
MAX_AGGREGATE_BUDGET_USD = MAX_CHILD_COUNT * MAX_PER_CHILD_BUDGET_USD
MIN_PER_CHILD_TIMEOUT_SECONDS = 0.05
MAX_PER_CHILD_TIMEOUT_SECONDS = 900.0
MIN_REVIEW_TIMEOUT_SECONDS = 0.1
MAX_REVIEW_TIMEOUT_SECONDS = 3600.0
MAX_EVIDENCE_BUNDLE_BYTES = 256 * 1024
REDUCER_FAN_IN = 8
ALLOWED_EFFORTS = {"low", "medium", "high", "xhigh", "max"}
MODEL_ALIASES = {"default", "inherit", "opus", "sonnet", "haiku", "fable"}
MODEL_ALIAS_PATTERN = re.compile(
    r"(?:^|[./_()-])(?:auto|automatic|current|default|free|inherit|latest|recommended|stable)(?:$|[./_()-])",
    re.IGNORECASE,
)
CLAUDE_MODEL_ID_PATTERN = re.compile(
    r"claude-[A-Za-z0-9][A-Za-z0-9.]*-\d+(?:-\d+-\d{8})?"
)
ROUTED_MODEL_ID_PATTERN = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9.-]{0,63}/"
    r"[A-Za-z0-9][A-Za-z0-9()_-]{0,95}(?:\d+\.\d+|\d{8})"
    r"[A-Za-z0-9.()_-]{0,31}"
)

PAGE_SCHEMA = contract.PAGE_SCHEMA
FINAL_SCHEMA = contract.FINAL_SCHEMA
ReviewRequest = contract.ReviewRequest

VISUAL_SYSTEM_PROMPT = contract.VISUAL_SYSTEM_PROMPT
REDUCER_SYSTEM_PROMPT = contract.REDUCER_SYSTEM_PROMPT


HarnessError = runtime.HarnessError
ChildFailure = runtime.ChildFailure


def _json_no_duplicates(raw: bytes, *, label: str, max_bytes: int) -> dict[str, Any]:
    if len(raw) > max_bytes:
        raise HarnessError(f"{label} exceeds byte limit")
    try:
        text = raw.decode("utf-8", "strict")
    except UnicodeDecodeError as exc:
        raise HarnessError(f"{label} is not UTF-8") from exc

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise HarnessError(f"{label} contains a duplicate key")
            output[key] = value
        return output

    try:
        value = json.loads(
            text,
            object_pairs_hook=unique,
            parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)),
        )
    except (json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise HarnessError(f"{label} is invalid JSON") from exc
    try:
        json.dumps(value, ensure_ascii=False).encode("utf-8")
    except (RecursionError, UnicodeEncodeError) as exc:
        raise HarnessError(f"{label} contains invalid Unicode") from exc
    if not isinstance(value, dict):
        raise HarnessError(f"{label} root must be an object")
    return value


def _exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise HarnessError(f"{label} keys are invalid")


def _integer(value: Any, label: str) -> int:
    if type(value) is not int:
        raise HarnessError(f"{label} must be an integer")
    return value


def _number(value: Any, label: str) -> float:
    if type(value) not in {int, float} or not math.isfinite(value):
        raise HarnessError(f"{label} must be a finite number")
    return float(value)


def _string(value: Any, label: str, max_bytes: int = 2048) -> str:
    if not isinstance(value, str):
        raise HarnessError(f"{label} must be a bounded string")
    try:
        size = len(value.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise HarnessError(f"{label} must be a bounded string") from exc
    if size > max_bytes:
        raise HarnessError(f"{label} must be a bounded string")
    return value


def _list(value: Any, label: str, max_items: int = 1000) -> list[Any]:
    if not isinstance(value, list) or len(value) > max_items:
        raise HarnessError(f"{label} must be a bounded array")
    return value


def _tree_calls(input_count: int) -> int:
    count = input_count
    total = 0
    while True:
        count = max(1, math.ceil(count / REDUCER_FAN_IN))
        total += count
        if count == 1:
            return total


def _maximum_child_count(page_count: int, transcript_chunks: int) -> int:
    reducer_inputs = page_count + transcript_chunks
    return page_count + _tree_calls(max(1, reducer_inputs))


def _validated_manifest(
    work: Path,
    root_fd: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], bytes, bytes | None]:
    try:
        return inputs.validated_manifest(
            work,
            root_fd,
            parse_json=_json_no_duplicates,
            exact_keys=_exact_keys,
            integer=_integer,
            number=_number,
            string=_string,
            bounded_list=_list,
        )
    except inputs.InputValidationError as exc:
        raise HarnessError(str(exc)) from exc


def _inspection_digest(
    manifest_raw: bytes, pages: list[dict[str, Any]], transcript_raw: bytes | None
) -> str:
    digest = hashlib.sha256()
    digest.update(b"watch-visual-harness-v1\0")
    digest.update(hashlib.sha256(manifest_raw).digest())
    for page in pages:
        digest.update(page["page"].to_bytes(8, "big"))
        digest.update(bytes.fromhex(page["sha256"]))
        for frame in page["frames"]:
            digest.update(frame["index"].to_bytes(8, "big"))
            digest.update(bytes.fromhex(frame["_sha256"]))
    if transcript_raw is not None:
        digest.update(hashlib.sha256(transcript_raw).digest())
    return digest.hexdigest()


def inspect_work(work: Path, root_fd: int | None = None) -> dict[str, Any]:
    if root_fd is None and work.is_symlink():
        raise HarnessError("work directory must be a regular non-symlink directory")
    manifest, pages, manifest_raw, transcript_raw = _validated_manifest(
        work.resolve(strict=True) if root_fd is None else work,
        root_fd,
    )
    page_count = len(pages)
    transcript_chunks = len(_transcript_evidence(transcript_raw))
    maximum_child_count = _maximum_child_count(page_count, transcript_chunks)
    if maximum_child_count > MAX_CHILD_COUNT:
        raise HarnessError("review requires too many child calls")
    maximum_bundle_bytes = (
        page_count * MAX_PAGE_REPORT_BYTES + MAX_FINAL_REPORT_BYTES + 16 * 1024
    )
    if maximum_bundle_bytes > MAX_EVIDENCE_BUNDLE_BYTES:
        raise HarnessError("review exceeds evidence bundle limit")
    return {
        "schema_version": SCHEMA_VERSION,
        "digest": _inspection_digest(manifest_raw, pages, transcript_raw),
        "page_count": page_count,
        "frame_count": manifest["frame_count"],
        "frame_range": (
            [0, manifest["frame_count"] - 1] if manifest["frame_count"] else []
        ),
        "transcript_chunk_count": transcript_chunks,
        "maximum_child_count": maximum_child_count,
        "approval_required": page_count > DEFAULT_PAGE_LIMIT,
    }


def _validate_request(
    request: ReviewRequest,
    inspection: dict[str, Any],
    current_runtime_digest: str,
) -> None:
    if (
        not re.fullmatch(r"[0-9a-f]{64}", request.digest)
        or request.digest != inspection["digest"]
    ):
        raise HarnessError("approval digest mismatch")
    if (
        not re.fullmatch(r"[0-9a-f]{64}", request.runtime_digest)
        or request.runtime_digest != current_runtime_digest
    ):
        raise HarnessError("approved runtime identity changed")
    if (
        type(request.approved_page_count) is not int
        or request.approved_page_count != inspection["page_count"]
    ):
        raise HarnessError("approved page count mismatch")
    if (
        type(request.approved_child_count) is not int
        or request.approved_child_count != inspection["maximum_child_count"]
    ):
        raise HarnessError("approved child count mismatch")
    if inspection["approval_required"] and request.allow_large_run is not True:
        raise HarnessError("large run approval is required")
    model = _string(request.model, "model", 256)
    if (
        model.casefold() in MODEL_ALIASES
        or MODEL_ALIAS_PATTERN.search(model)
        or (
            CLAUDE_MODEL_ID_PATTERN.fullmatch(model) is None
            and ROUTED_MODEL_ID_PATTERN.fullmatch(model) is None
        )
    ):
        raise HarnessError("exact model ID is required")
    effort = _string(request.effort, "effort", 32)
    if effort not in ALLOWED_EFFORTS:
        raise HarnessError("exact effort is invalid")
    per_child = _number(request.per_child_budget_usd, "per-child budget")
    aggregate = _number(request.aggregate_budget_usd, "aggregate budget")
    if per_child <= 0 or aggregate < per_child * inspection["maximum_child_count"]:
        raise HarnessError("aggregate budget is insufficient")
    if per_child > MAX_PER_CHILD_BUDGET_USD or aggregate > MAX_AGGREGATE_BUDGET_USD:
        raise HarnessError("review budget ceiling exceeded")
    per_child_timeout = _number(request.per_child_timeout_seconds, "per-child timeout")
    review_timeout = _number(request.review_timeout_seconds, "review timeout")
    if (
        not MIN_PER_CHILD_TIMEOUT_SECONDS
        <= per_child_timeout
        <= MAX_PER_CHILD_TIMEOUT_SECONDS
    ):
        raise HarnessError("per-child timeout is outside the approved bounds")
    if not MIN_REVIEW_TIMEOUT_SECONDS <= review_timeout <= MAX_REVIEW_TIMEOUT_SECONDS:
        raise HarnessError("review timeout is outside the approved bounds")
    if review_timeout < per_child_timeout:
        raise HarnessError("review timeout is shorter than the per-child timeout")
    _string(request.question, "question", MAX_QUESTION_BYTES)


def check_claude_cli(executable: Path) -> None:
    with (
        runtime.audit_executable(executable) as audited,
        runtime.audit_containment_helpers() as helpers,
    ):
        runtime.check_claude_cli(audited, helpers)


def _run_child(
    executable: runtime.AuditedExecutable,
    helpers: runtime.ContainmentHelpers,
    *,
    model: str,
    effort: str,
    budget: float,
    schema: dict[str, Any],
    system_prompt: str,
    message: dict[str, Any],
    timeout_seconds: float,
    phase: str = "page",
    child_ordinal: int = 1,
    deadline_limited: bool = False,
) -> runtime.ChildResult:
    return runtime.run_child(
        executable,
        helpers,
        model=model,
        effort=effort,
        budget=budget,
        schema=schema,
        system_prompt=system_prompt,
        message=message,
        timeout_seconds=timeout_seconds,
        phase=phase,
        child_ordinal=child_ordinal,
        deadline_limited=deadline_limited,
    )


def _image_message(image: bytes, request: dict[str, Any]) -> dict[str, Any]:
    return reports.image_message(image, request)


def _text_message(request: dict[str, Any]) -> dict[str, Any]:
    return reports.text_message(request)


def decode_image_block(block: dict[str, Any]) -> bytes:
    return reports.decode_image_block(block, error=HarnessError)


def validate_page_report(
    value: dict[str, Any],
    *,
    page: int,
    indices: list[int],
    timestamps: list[float],
) -> dict[str, Any]:
    return reports.validate_page_report(
        value,
        page=page,
        indices=indices,
        timestamps=timestamps,
        error=HarnessError,
    )


def validate_final_report(
    value: dict[str, Any],
    *,
    expected_indices: list[int],
    timestamps: list[float] | None = None,
) -> dict[str, Any]:
    return reports.validate_final_report(
        value,
        expected_indices=expected_indices,
        timestamps=timestamps,
        error=HarnessError,
    )


def _private_json_bytes(value: dict[str, Any], max_bytes: int) -> bytes:
    try:
        raw = reports.canonical_json_bytes(value)
    except ValueError as exc:
        raise HarnessError("evidence contains invalid Unicode") from exc
    if len(raw) > max_bytes:
        raise HarnessError("evidence exceeds byte limit")
    return raw


_page_schema = reports.page_schema


def _page_request(page: dict[str, Any], question: str) -> dict[str, Any]:
    return {
        "kind": "page",
        "page": page["page"],
        "frame_start": page["frame_start"],
        "frame_end": page["frame_end"],
        "frame_indices": page["indices"],
        "timestamps": page["timestamps"],
        "question": question,
    }


def _transcript_evidence(transcript_raw: bytes | None) -> list[dict[str, Any]]:
    try:
        return reports.transcript_evidence(transcript_raw)
    except (UnicodeDecodeError, ValueError) as exc:
        raise HarnessError(str(exc)) from exc


def _approved_timeout(request: ReviewRequest, deadline: float) -> tuple[float, bool]:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ChildFailure(
            "review_deadline", elapsed_seconds=request.review_timeout_seconds
        )
    return min(
        request.per_child_timeout_seconds, remaining
    ), remaining < request.per_child_timeout_seconds


def _reduce_reports(
    report_items: list[dict[str, Any]],
    *,
    expected_indices: list[int],
    timestamp_by_index: dict[int, float],
    request: ReviewRequest,
    executable: runtime.AuditedExecutable,
    helpers: runtime.ContainmentHelpers,
    deadline: float,
    initial_ordinal: int,
    initial_cost: float,
) -> tuple[dict[str, Any], int, float]:
    if not report_items:
        raise HarnessError("reducer evidence is empty")
    current: list[dict[str, Any]] = report_items
    calls = 0
    total_cost = initial_cost
    while True:
        next_level: list[dict[str, Any]] = []
        for start in range(0, len(current), REDUCER_FAN_IN):
            chunk = current[start : start + REDUCER_FAN_IN]
            chunk_indices = sorted(
                {
                    index
                    for item in chunk
                    for index in item["coverage"]["inspected_frame_indices"]
                }
            )
            child_request = {
                "kind": "reduce",
                "question": request.question,
                "expected_indices": chunk_indices,
                "evidence": chunk,
            }
            timeout, deadline_limited = _approved_timeout(request, deadline)
            result = _run_child(
                executable,
                helpers,
                model=request.model,
                effort=request.effort,
                budget=request.per_child_budget_usd,
                schema=reports.final_schema(chunk_indices),
                system_prompt=REDUCER_SYSTEM_PROMPT,
                message=_text_message(child_request),
                timeout_seconds=timeout,
                phase="reduce",
                child_ordinal=initial_ordinal + calls,
                deadline_limited=deadline_limited,
            )
            total_cost += result.cost_usd
            if total_cost > request.aggregate_budget_usd + 1e-9:
                raise ChildFailure(
                    "budget",
                    phase="reduce",
                    child_ordinal=initial_ordinal + calls,
                    model=request.model,
                    effort=request.effort,
                    reported_cost_usd=round(total_cost, 6),
                )
            try:
                report = validate_final_report(
                    result.structured_output,
                    expected_indices=chunk_indices,
                    timestamps=[timestamp_by_index[index] for index in chunk_indices],
                )
            except HarnessError as exc:
                raise ChildFailure(
                    "protocol",
                    phase="reduce",
                    child_ordinal=initial_ordinal + calls,
                    model=request.model,
                    effort=request.effort,
                    protocol_stage="final_report",
                ) from exc
            next_level.append(report)
            calls += 1
        if len(next_level) == 1:
            final = next_level[0]
            if final["coverage"]["inspected_frame_indices"] != expected_indices:
                raise HarnessError("reducer coverage mismatch")
            return final, calls, total_cost
        current = next_level


def _current_review_state(
    work: Path,
    root_fd: int,
    expected_digest: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], bytes | None]:
    manifest, pages, manifest_raw, transcript_raw = _validated_manifest(work, root_fd)
    if _inspection_digest(manifest_raw, pages, transcript_raw) != expected_digest:
        raise HarnessError("approved media changed before review")
    return manifest, pages, transcript_raw


def _lineage(
    request: ReviewRequest,
    *,
    calls: int,
    total_cost: float,
    reports: list[dict[str, Any]],
    final: dict[str, Any],
) -> dict[str, Any]:
    report_bytes = [
        _private_json_bytes(report, MAX_PAGE_REPORT_BYTES) for report in reports
    ]
    final_bytes = _private_json_bytes(final, MAX_FINAL_REPORT_BYTES)
    return {
        "schema_version": 2,
        "media_digest": request.digest,
        "request_sha256": request.sha256,
        "runtime_digest": request.runtime_digest,
        "model": request.model,
        "effort": request.effort,
        "approved_page_count": request.approved_page_count,
        "approved_child_count": request.approved_child_count,
        "per_child_budget_usd": request.per_child_budget_usd,
        "aggregate_budget_usd": request.aggregate_budget_usd,
        "per_child_timeout_seconds": request.per_child_timeout_seconds,
        "review_timeout_seconds": request.review_timeout_seconds,
        "actual_child_count": calls,
        "actual_cost_usd": round(total_cost, 6),
        "records": [
            {
                "kind": "page",
                "page": index,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
            }
            for index, raw in enumerate(report_bytes, start=1)
        ]
        + [
            {
                "kind": "final",
                "sha256": hashlib.sha256(final_bytes).hexdigest(),
                "bytes": len(final_bytes),
            }
        ],
    }


def _evidence_bundle(
    request: ReviewRequest,
    *,
    calls: int,
    total_cost: float,
    reports: list[dict[str, Any]],
    final: dict[str, Any],
) -> bytes:
    lineage = _lineage(
        request,
        calls=calls,
        total_cost=total_cost,
        reports=reports,
        final=final,
    )
    value = {
        "schema_version": 2,
        "kind": "watch-review-evidence",
        "pages": reports,
        "final": final,
        "lineage": lineage,
    }
    return _private_json_bytes(value, MAX_EVIDENCE_BUNDLE_BYTES)


def review_work(
    work: Path,
    request: ReviewRequest,
    *,
    claude_bin: Path | None = None,
) -> dict[str, Any]:
    if claude_bin is None:
        discovered = shutil.which("claude")
        if discovered is None:
            raise HarnessError("Claude CLI is unavailable")
        executable_path = Path(discovered).resolve(strict=True)
    else:
        executable_path = claude_bin
    with (
        runtime.pin_directory(work) as pinned,
        runtime.audit_executable(executable_path) as executable,
        runtime.audit_containment_helpers() as helpers,
    ):
        inspection = inspect_work(pinned.path, pinned.descriptor)
        _validate_request(
            request,
            inspection,
            runtime.runtime_digest(executable, helpers),
        )
        manifest, pages, transcript_raw = _current_review_state(
            pinned.path,
            pinned.descriptor,
            request.digest,
        )
        runtime.check_claude_cli(executable, helpers)
        pinned.assert_path_identity()
        executable.assert_identity()
        manifest, pages, transcript_raw = _current_review_state(
            pinned.path,
            pinned.descriptor,
            request.digest,
        )
        if runtime.runtime_digest(executable, helpers) != request.runtime_digest:
            raise HarnessError("approved runtime identity changed")
        deadline = time.monotonic() + request.review_timeout_seconds
        expected_indices = list(range(manifest["frame_count"]))
        timestamp_by_index = {
            frame["index"]: float(frame["timestamp_seconds"])
            for frame in manifest["frames"]
        }
        page_reports: list[dict[str, Any]] = []
        calls = 0
        total_cost = 0.0
        with publication.locked_evidence(pinned) as evidence:
            for approved_page in pages:
                pinned.assert_path_identity()
                executable.assert_identity()
                current_manifest, current_pages, current_transcript_raw = (
                    _current_review_state(
                        pinned.path,
                        pinned.descriptor,
                        request.digest,
                    )
                )
                page = current_pages[approved_page["page"] - 1]
                if page["indices"] != list(
                    range(page["frame_start"], page["frame_end"] + 1)
                ):
                    raise HarnessError("overview page coverage changed before review")
                image = page["canonical"]
                if hashlib.sha256(image).hexdigest() != page["sha256"]:
                    raise HarnessError("overview changed before review")
                timeout, deadline_limited = _approved_timeout(request, deadline)
                result = _run_child(
                    executable,
                    helpers,
                    model=request.model,
                    effort=request.effort,
                    budget=request.per_child_budget_usd,
                    schema=_page_schema(page),
                    system_prompt=VISUAL_SYSTEM_PROMPT,
                    message=_image_message(
                        image, _page_request(page, request.question)
                    ),
                    timeout_seconds=timeout,
                    phase="page",
                    child_ordinal=calls + 1,
                    deadline_limited=deadline_limited,
                )
                total_cost += result.cost_usd
                if total_cost > request.aggregate_budget_usd + 1e-9:
                    raise ChildFailure(
                        "budget",
                        phase="page",
                        child_ordinal=calls + 1,
                        model=request.model,
                        effort=request.effort,
                        reported_cost_usd=round(total_cost, 6),
                    )
                try:
                    report = validate_page_report(
                        result.structured_output,
                        page=page["page"],
                        indices=page["indices"],
                        timestamps=page["timestamps"],
                    )
                except HarnessError as exc:
                    raise ChildFailure(
                        "protocol",
                        phase="page",
                        child_ordinal=calls + 1,
                        model=request.model,
                        effort=request.effort,
                        protocol_stage="page_report",
                    ) from exc
                if report["drilldown_needed"]:
                    indices = json.dumps(
                        report["drilldown_needed"], separators=(",", ":")
                    )
                    raise HarnessError(
                        "exact-frame drilldown requires a separately approved focused pass; "
                        f"validated frame indices: {indices}"
                    )
                page_reports.append(report)
                calls += 1
                manifest = current_manifest
                pages = current_pages
                transcript_raw = current_transcript_raw
                expected_indices = list(range(manifest["frame_count"]))
                timestamp_by_index = {
                    frame["index"]: float(frame["timestamp_seconds"])
                    for frame in manifest["frames"]
                }
            evidence_reports = [*page_reports, *_transcript_evidence(transcript_raw)]
            final, reducer_calls, total_cost = _reduce_reports(
                evidence_reports,
                expected_indices=expected_indices,
                timestamp_by_index=timestamp_by_index,
                request=request,
                executable=executable,
                helpers=helpers,
                deadline=deadline,
                initial_ordinal=calls + 1,
                initial_cost=total_cost,
            )
            calls += reducer_calls
            if calls != request.approved_child_count:
                raise HarnessError("actual child count differs from approval")
            pinned.assert_path_identity()
            executable.assert_identity()
            _current_review_state(pinned.path, pinned.descriptor, request.digest)
            if runtime.runtime_digest(executable, helpers) != request.runtime_digest:
                raise HarnessError("approved runtime identity changed")
            bundle = _evidence_bundle(
                request,
                calls=calls,
                total_cost=total_cost,
                reports=page_reports,
                final=final,
            )
            try:
                receipt = publication.publish_bundle(evidence, bundle)
            except (HarnessError, OSError) as exc:
                raise ChildFailure(
                    "protocol",
                    phase="publication",
                    model=request.model,
                    effort=request.effort,
                    protocol_stage="host_publication",
                ) from exc
            return {"final": final, "evidence": receipt}


def _read_stdin_json(max_bytes: int = 64 * 1024) -> dict[str, Any]:
    raw = sys.stdin.buffer.read(max_bytes + 1)
    return _json_no_duplicates(raw, label="review request", max_bytes=max_bytes)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("--work-dir", required=True, type=Path)
    runtime_parser = subparsers.add_parser("inspect-runtime")
    runtime_parser.add_argument("--claude-bin", required=True, type=Path)
    review_parser = subparsers.add_parser("review")
    review_parser.add_argument("--work-dir", required=True, type=Path)
    review_parser.add_argument("--claude-bin", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            result = inspect_work(args.work_dir)
        elif args.command == "inspect-runtime":
            result = runtime.inspect_runtime(
                args.claude_bin,
                publication_check=publication.check_runtime_support,
            )
        else:
            request = ReviewRequest.from_dict(_read_stdin_json(), error=HarnessError)
            result = review_work(args.work_dir, request, claude_bin=args.claude_bin)
    except ChildFailure as exc:
        error = exc.envelope
    except (HarnessError, OSError):
        error = ChildFailure("protocol").envelope
    else:
        print(json.dumps(result, separators=(",", ":"), ensure_ascii=False))
        return 0
    print(json.dumps(error, separators=(",", ":"), ensure_ascii=False), file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
