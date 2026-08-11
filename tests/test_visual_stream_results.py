"""Terminal child stream validation regressions."""

from __future__ import annotations

import json

import pytest

import _visual_stream as visual_stream
from _visual_failures import ChildFailure


def test_child_stream_rejects_events_after_terminal_result() -> None:
    model = "claude-test"
    events = [
        {
            "type": "system",
            "subtype": "init",
            "model": model,
            "tools": ["StructuredOutput"],
        },
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "num_turns": 1,
            "permission_denials": [],
            "modelUsage": {model: {}},
            "structured_output": {},
            "total_cost_usd": 0.01,
        },
        {"type": "assistant", "message": {"content": []}},
    ]
    raw = b"".join((json.dumps(event) + "\n").encode() for event in events)

    with pytest.raises(ChildFailure) as caught:
        visual_stream.validate_child_events(
            raw,
            context={},
            model=model,
            budget=0.1,
            returncode=0,
            input_failed=False,
        )

    assert caught.value.category == "protocol"


@pytest.mark.parametrize("reason", [{}, []])
def test_child_stream_contains_malformed_terminal_reason(reason: object) -> None:
    assert visual_stream._safe_result_metadata({"terminal_reason": reason}) == {}
