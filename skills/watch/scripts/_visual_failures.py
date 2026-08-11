"""Bounded failures shared by the trusted visual-review boundary."""

from __future__ import annotations

import json
from typing import Any

FAILURE_REASONS = {
    "timeout": "Claude child timed out",
    "review_deadline": "review deadline expired",
    "api_not_found": "Claude child API model was not found",
    "auth": "Claude child authentication failed",
    "rate_limit": "Claude child rate limit was reached",
    "provider": "Claude child provider request failed",
    "budget": "Claude child budget was exceeded",
    "signal": "Claude child terminated by signal",
    "process_exit": "Claude child failed with process exit",
    "context_limit": "Claude child context limit was exceeded",
    "output_limit": "Claude child output limit exceeded",
    "permission": "Claude child reported a permission denial",
    "protocol": "Claude child protocol validation failed",
    "identity": "Claude child model identity or tool boundary is invalid",
    "cleanup": "Claude child cleanup failed",
}
IDENTITY_DIMENSIONS = {"init_count", "model", "effort", "tools"}
PROTOCOL_STAGES = {
    "stream_parse",
    "page_report",
    "final_report",
    "host_publication",
}


class HarnessError(RuntimeError):
    """The visual review boundary failed closed."""


class ChildFailure(HarnessError):
    """A bounded, coordinator-safe child failure."""

    def __init__(self, category: str, **metadata: Any) -> None:
        if category not in FAILURE_REASONS:
            category = "protocol"
        allowed = {
            "phase",
            "child_ordinal",
            "model",
            "effort",
            "api_status",
            "terminal_reason",
            "reported_cost_usd",
            "exit_code",
            "signal",
            "elapsed_seconds",
            "identity_dimension",
            "protocol_stage",
        }
        safe = {key: value for key, value in metadata.items() if key in allowed}
        identity_dimension = safe.get("identity_dimension")
        if (
            category != "identity"
            or not isinstance(identity_dimension, str)
            or identity_dimension not in IDENTITY_DIMENSIONS
        ):
            safe.pop("identity_dimension", None)
        protocol_stage = safe.get("protocol_stage")
        if (
            category != "protocol"
            or not isinstance(protocol_stage, str)
            or protocol_stage not in PROTOCOL_STAGES
        ):
            safe.pop("protocol_stage", None)
        self.category = category
        self.metadata = safe
        self.envelope = {
            "schema_version": 1,
            "status": "blocked",
            "visual_status": "unknown",
            "category": category,
            "reason": FAILURE_REASONS[category],
            **safe,
        }
        super().__init__(
            json.dumps(self.envelope, sort_keys=True, separators=(",", ":"))
        )
