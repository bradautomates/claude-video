"""Unicode-obfuscated citation regressions."""

from __future__ import annotations

import pytest

import visual_harness

BIDI_TIMING_CLAIM = "Damage remains visible for one ‮sm‬."


@pytest.mark.parametrize(
    "claim",
    [
        "Damage remains visible for one ｍｓ.",
        "Damage remains visible for one m​s.",
        "Damage remains visible for one m͏s.",
        "Damage remains visible for one m️s.",
        BIDI_TIMING_CLAIM,
    ],
)
def test_page_validator_rejects_unicode_obfuscated_timing(claim: str) -> None:
    payload = {
        "schema_version": 1,
        "coverage": {
            "page": 1,
            "frame_start": 0,
            "frame_end": 0,
            "inspected_frame_indices": [0],
            "omitted_ranges": [],
        },
        "observations": [
            {
                "claim": claim,
                "frame_indices": [0],
                "basis": "visual",
                "status": "observed",
                "confidence": "high",
            }
        ],
        "ambiguities": [],
        "drilldown_needed": [],
        "limitations": [],
    }

    match = (
        "invalid Unicode control"
        if "​" in claim or claim == BIDI_TIMING_CLAIM
        else "host-owned citation"
    )
    with pytest.raises(visual_harness.HarnessError, match=match):
        visual_harness.validate_page_report(
            payload,
            page=1,
            indices=[0],
            timestamps=[0.0],
        )


def test_final_validator_rejects_bidi_obfuscated_timing() -> None:
    payload = {
        "schema_version": 1,
        "status": "complete",
        "answer": "Validated observations use host-owned citations.",
        "observations": [
            {
                "claim": BIDI_TIMING_CLAIM,
                "frame_indices": [0],
                "basis": "visual",
                "status": "observed",
                "confidence": "high",
            }
        ],
        "coverage": {"inspected_frame_indices": [0], "omitted_ranges": []},
        "uncertainty": [],
        "limitations": [],
    }

    with pytest.raises(visual_harness.HarnessError, match="invalid Unicode control"):
        visual_harness.validate_final_report(
            payload,
            expected_indices=[0],
            timestamps=[0.0],
        )
