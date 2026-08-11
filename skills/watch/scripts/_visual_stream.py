"""Response-gated stream-JSON effort validation."""

from __future__ import annotations

import json
import math
import os
import select
import threading
from typing import Any, Callable

from _visual_failures import ChildFailure

REQUEST_ID = "watch-effort-gate-v1"
MAX_PRE_GATE_LINE_BYTES = 64 * 1024
SAFE_TERMINAL_REASONS = {
    "api_error",
    "authentication_error",
    "authorization_error",
    "budget_exceeded",
    "max_budget_usd",
    "rate_limit",
    "rate_limit_error",
    "provider_error",
}


def _event(raw: bytes) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate")
            value[key] = item
        return value

    value = json.loads(
        raw.decode("utf-8", "strict"),
        object_pairs_hook=unique,
        parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)),
    )
    if not isinstance(value, dict):
        raise ValueError("root")
    return value


def _parse_events(raw: bytes, context: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        return [_event(line) for line in raw.splitlines() if line.strip()]
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        RecursionError,
        ValueError,
    ) as exc:
        raise ChildFailure(
            "protocol",
            **context,
            protocol_stage="stream_parse",
        ) from exc


def _safe_result_metadata(result: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    status = result.get("api_error_status")
    if type(status) is int and 100 <= status <= 599:
        metadata["api_status"] = status
    reason = result.get("terminal_reason")
    if isinstance(reason, str) and reason in SAFE_TERMINAL_REASONS:
        metadata["terminal_reason"] = reason
    cost = result.get("total_cost_usd")
    if type(cost) in {int, float} and math.isfinite(cost) and cost >= 0:
        metadata["reported_cost_usd"] = round(float(cost), 6)
    return metadata


def _failure_category(metadata: dict[str, Any]) -> str:
    status = metadata.get("api_status")
    reason = metadata.get("terminal_reason")
    if status == 404:
        return "api_not_found"
    if status in {401, 403} or reason in {
        "authentication_error",
        "authorization_error",
    }:
        return "auth"
    if status == 429 or reason in {"rate_limit", "rate_limit_error"}:
        return "rate_limit"
    if reason in {"budget_exceeded", "max_budget_usd"}:
        return "budget"
    return "provider"


def validate_child_events(
    raw: bytes,
    *,
    context: dict[str, Any],
    model: str,
    budget: float,
    returncode: int,
    input_failed: bool,
) -> tuple[dict[str, Any], float]:
    events = _parse_events(raw, context)
    if not events:
        if returncode:
            raise ChildFailure("process_exit", **context, exit_code=returncode)
        raise ChildFailure("protocol", **context)
    for event in events:
        event_type = event.get("type")
        if event_type == "tool_use" and event.get("name") != "StructuredOutput":
            raise ChildFailure("permission", **context)
        if event_type == "tool_result" and event.get("name") not in {
            None,
            "StructuredOutput",
        }:
            raise ChildFailure("permission", **context)
        if event_type == "assistant" and isinstance(event.get("message"), dict):
            content = event["message"].get("content")
            if isinstance(content, list) and any(
                isinstance(block, dict)
                and block.get("type") == "tool_use"
                and block.get("name") != "StructuredOutput"
                for block in content
            ):
                raise ChildFailure("permission", **context)
    if returncode and not any(event.get("type") == "result" for event in events):
        raise ChildFailure("process_exit", **context, exit_code=returncode)
    init_events = [
        event
        for event in events
        if event.get("type") == "system" and event.get("subtype") == "init"
    ]
    if len(init_events) != 1:
        raise ChildFailure("identity", **context, identity_dimension="init_count")
    init = init_events[0]
    if init.get("model") != model:
        raise ChildFailure("identity", **context, identity_dimension="model")
    if init.get("tools") != ["StructuredOutput"]:
        raise ChildFailure("identity", **context, identity_dimension="tools")
    results = [event for event in events if event.get("type") == "result"]
    if len(results) != 1 or events[-1] is not results[0]:
        raise ChildFailure("protocol", **context)
    result = results[0]
    metadata = _safe_result_metadata(result)
    if result.get("subtype") != "success" or result.get("is_error") is not False:
        raise ChildFailure(_failure_category(metadata), **context, **metadata)
    if returncode:
        raise ChildFailure("process_exit", **context, exit_code=returncode, **metadata)
    if input_failed:
        raise ChildFailure("protocol", **context)
    if type(result.get("num_turns")) is not int or not 1 <= result["num_turns"] <= 2:
        raise ChildFailure("protocol", **context)
    if result.get("permission_denials") != []:
        raise ChildFailure("permission", **context)
    model_usage = result.get("modelUsage")
    if not isinstance(model_usage, dict) or not model_usage:
        raise ChildFailure("protocol", **context)
    if set(model_usage) != {model}:
        raise ChildFailure("identity", **context, identity_dimension="model")
    if not isinstance(model_usage[model], dict):
        raise ChildFailure("protocol", **context)
    cost = result.get("total_cost_usd")
    if type(cost) not in {int, float} or not math.isfinite(cost) or cost < 0:
        raise ChildFailure("protocol", **context)
    cost = float(cost)
    if cost > budget + 1e-9:
        raise ChildFailure("budget", **context, reported_cost_usd=round(cost, 6))
    structured = result.get("structured_output")
    if not isinstance(structured, dict):
        raise ChildFailure("protocol", **context)
    return structured, cost


class EffortGate:
    """Permit one user frame only after exact host-applied effort is reported."""

    def __init__(self, model: str, effort: str) -> None:
        self.request = (
            json.dumps(
                {
                    "type": "control_request",
                    "request_id": REQUEST_ID,
                    "request": {"subtype": "get_settings"},
                },
                separators=(",", ":"),
            )
            + "\n"
        ).encode()
        self.model = model
        self.effort = effort
        self.ready = threading.Event()
        self._lock = threading.Lock()
        self._pending = bytearray()
        self._response_valid = False
        self.approved = False
        self.consumed = False
        self.dispatch_complete = False
        self._pending_during_dispatch = False
        self.failure: tuple[str, str | None] | None = None

    def _fail(self, category: str, dimension: str | None = None) -> None:
        if self.failure is None:
            self.failure = (category, dimension)
        self.approved = False
        self.ready.set()

    def cancel(self, category: str = "protocol") -> None:
        with self._lock:
            self._fail(category)

    def wait(self, timeout_seconds: float) -> bool:
        if self.ready.wait(timeout_seconds):
            return True
        with self._lock:
            self._fail("timeout")
        return False

    def dispatch(
        self,
        descriptor: int,
        raw: bytes,
        wait_writable: Callable[[int], None] | None = None,
    ) -> bool:
        """Claim one authorized frame; later output can fail, not revoke, the review."""
        if not raw or not raw.endswith(b"\n") or b"\n" in raw[:-1]:
            with self._lock:
                self._fail("protocol")
            return False
        wait = wait_writable or self._wait_writable
        with self._lock:
            if not self.approved or self.failure or self.consumed:
                return False
            os.set_blocking(descriptor, False)
            self.approved = False
            self.consumed = True
        pending = memoryview(raw)
        while pending:
            with self._lock:
                if self.failure:
                    return False
            try:
                written = os.write(descriptor, pending)
            except BlockingIOError:
                written = None
            if written is not None:
                if written < 1:
                    raise OSError("user-frame write made no progress")
                pending = pending[written:]
                if not pending:
                    with self._lock:
                        if self.failure or self._pending_during_dispatch:
                            self._fail("protocol")
                            return False
                        self.dispatch_complete = True
                        return True
            wait(descriptor)
        return False

    @staticmethod
    def _wait_writable(descriptor: int) -> None:
        select.select([], [descriptor], [])

    def feed(self, chunk: bytes) -> None:
        with self._lock:
            if self.failure:
                return
            self._pending.extend(chunk)
            self.approved = False
            if self.consumed and not self.dispatch_complete:
                self._pending_during_dispatch = True
            while b"\n" in self._pending:
                line, _, rest = self._pending.partition(b"\n")
                self._pending = bytearray(rest)
                if not line.strip():
                    continue
                if len(line) > MAX_PRE_GATE_LINE_BYTES:
                    self._fail("output_limit")
                    return
                try:
                    self._accept(_event(line))
                except (
                    UnicodeDecodeError,
                    json.JSONDecodeError,
                    RecursionError,
                    ValueError,
                ):
                    self._fail("protocol")
                    return
                if self.failure:
                    return
            if len(self._pending) > MAX_PRE_GATE_LINE_BYTES:
                self._fail("output_limit")
                return
            if self._response_valid and not self._pending and not self.consumed:
                self.approved = True
                self.ready.set()

    def _accept(self, event: dict[str, Any]) -> None:
        event_type = event.get("type")
        if self.consumed:
            if (
                event_type == "control_response"
                or not self.dispatch_complete
                or self._pending_during_dispatch
            ):
                self._fail("protocol")
            return
        if self._response_valid:
            self._fail("protocol")
            return
        if event_type == "system" and event.get("subtype") == "init":
            if event.get("model") != self.model:
                self._fail("identity", "model")
                return
            if event.get("tools") != ["StructuredOutput"]:
                self._fail("identity", "tools")
                return
            return
        if event_type != "control_response":
            self._fail("protocol")
            return
        response = event.get("response")
        if (
            not isinstance(response, dict)
            or response.get("subtype") != "success"
            or response.get("request_id") != REQUEST_ID
        ):
            self._fail("protocol")
            return
        settings = response.get("response")
        if (
            not isinstance(settings, dict)
            or not isinstance(settings.get("effective"), dict)
            or not isinstance(settings.get("sources"), list)
        ):
            self._fail("protocol")
            return
        applied = settings.get("applied")
        if not isinstance(applied, dict):
            self._fail("identity", "effort")
            return
        if applied.get("model") != self.model:
            self._fail("identity", "model")
            return
        if applied.get("effort") != self.effort:
            self._fail("identity", "effort")
            return
        self._response_valid = True

    def finish(self, *, incomplete: bool = False) -> None:
        with self._lock:
            if incomplete:
                return
            if not self.ready.is_set() or self._pending.strip():
                self._fail("protocol")
