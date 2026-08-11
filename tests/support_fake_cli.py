"""Local fake Claude CLI used by trusted visual-harness tests."""

FAKE_CLAUDE = r"""#!/usr/bin/env python3
import json
import os
import sys
import time
from pathlib import Path

if "--version" in sys.argv:
    print("2.1.220 (Claude Code)")
    raise SystemExit(0)
if "--help" in sys.argv:
    print("--safe-mode --tools --permission-mode --disable-slash-commands --input-format --output-format --json-schema --no-session-persistence --model --effort --max-budget-usd --system-prompt")
    raise SystemExit(0)

mode = __FAKE_MODE__
log = Path(__FAKE_LOG__)
logged = False
pending = b""
inputs = []
response_sent = False
user_ready_before_response = False


def append_record(user_raw="", *, final=False):
    global logged
    if final and logged:
        return
    record = {
        "argv": sys.argv[1:],
        "stdin": user_raw,
        "inputs": list(inputs),
        "response_sent": response_sent,
        "user_ready_before_response": user_ready_before_response,
    }
    with log.open("ab") as handle:
        handle.write(json.dumps(record).encode() + b"\n")
    if final:
        logged = True


def emit(value):
    os.write(1, json.dumps(value, separators=(",", ":")).encode() + b"\n")


def read_line():
    global pending
    while b"\n" not in pending:
        chunk = os.read(0, 65536)
        if not chunk:
            line, pending = pending, b""
            return line
        pending += chunk
    line, pending = pending.split(b"\n", 1)
    return line


def receive_value():
    raw = read_line()
    if not raw:
        return raw, None
    value = json.loads(raw.decode("utf-8"))
    inputs.append(value.get("type") if isinstance(value, dict) else None)
    return raw, value


if mode == "detached":
    pass
if mode == "context_limit_before_settings":
    append_record(final=True)
    os.write(2, b"API Error: Prompt is too long. RAW_CONTEXT_DIAGNOSTIC\n")
    raise SystemExit(1)
if mode == "no_read":
    time.sleep(30)
if mode == "close_stdin":
    os.close(0)
    time.sleep(0.2)
    raise SystemExit(0)
model = sys.argv[sys.argv.index("--model") + 1]
effort = sys.argv[sys.argv.index("--effort") + 1]
first_raw, first = receive_value()
if not isinstance(first, dict):
    append_record(final=True)
    raise SystemExit(2)

message = None
if first.get("type") == "user":
    message = first
    append_record(first_raw.decode("utf-8"), final=True)
else:
    request_id = first.get("request_id")
    if pending:
        user_ready_before_response = True
    applied = {"model": model, "effort": effort}
    if mode in {"bad_effort", "settings_leak"}:
        applied["effort"] = "low"
    elif mode == "missing_effort":
        applied.pop("effort")
    elif mode == "malformed_effort":
        applied = []
    elif mode == "missing_applied_model":
        applied.pop("model")
    elif mode == "malformed_applied_model":
        applied["model"] = []
    elif mode == "bad_applied_model":
        applied["model"] = "wrong-model"
    settings = {"effective": {}, "sources": [], "applied": applied}
    if mode == "missing_effective":
        settings.pop("effective")
    elif mode == "malformed_effective":
        settings["effective"] = []
    elif mode == "missing_sources":
        settings.pop("sources")
    elif mode == "malformed_sources":
        settings["sources"] = {}
    elif mode == "settings_leak":
        settings["effective"] = {"credential": "SETTINGS_SECRET"}
        settings["sources"] = ["https://private.invalid/raw-provider-body"]
    response_id = "wrong-id" if mode == "wrong_request_id" else request_id
    response = {
        "type": "control_response",
        "response": {
            "subtype": "success",
            "request_id": response_id,
            "response": settings,
        },
    }
    response_sent = True
    if mode == "settings_timeout":
        append_record(final=True)
        time.sleep(30)
    elif mode == "duplicate_key_response":
        raw = json.dumps(response, separators=(",", ":"))
        raw = raw.replace('"effort":"' + effort + '"', '"effort":"' + effort + '","effort":"' + effort + '"', 1)
        os.write(1, raw.encode() + b"\n")
    elif mode == "duplicate_response":
        raw = json.dumps(response, separators=(",", ":")).encode() + b"\n"
        os.write(1, raw + raw)
    elif mode == "delayed_duplicate_response":
        emit(response)
        user_raw, message = receive_value()
        time.sleep(0.05)
        emit(response)
    elif mode == "invalid_response_hang":
        response["response"]["response"]["applied"]["effort"] = "low"
        emit(response)
        time.sleep(30)
    elif mode == "pre_dispatch_result":
        emit(response)
        emit({
            "type": "result", "subtype": "success", "is_error": False,
            "num_turns": 1, "permission_denials": [],
            "modelUsage": {model: {}}, "structured_output": {},
            "total_cost_usd": 0.001,
        })
    elif mode == "overflow_before_settings":
        os.write(1, b"x" * 700000)
        time.sleep(30)
    elif mode == "oversized_settings_line":
        os.write(1, b'{"type":"control_response","padding":"' + b"x" * 70000 + b'"}\n')
    else:
        emit(response)
    invalid_settings_modes = {
        "bad_effort", "missing_effort", "malformed_effort",
        "missing_applied_model", "malformed_applied_model", "bad_applied_model",
        "missing_effective", "malformed_effective",
        "missing_sources", "malformed_sources", "settings_leak",
        "wrong_request_id", "duplicate_key_response", "duplicate_response",
        "pre_dispatch_result", "overflow_before_settings", "oversized_settings_line",
    }
    if mode in invalid_settings_modes:
        append_record(final=True)
        raise SystemExit(0)
    user_raw, message = receive_value()
    if isinstance(message, dict) and message.get("type") == "user":
        append_record(user_raw.decode("utf-8"), final=True)
    else:
        append_record(final=True)
        raise SystemExit(2)

if mode == "timeout":
    time.sleep(30)
if mode == "context_limit_stderr":
    os.write(2, b"API Error: Prompt is too long. RAW_CONTEXT_DIAGNOSTIC\n")
    raise SystemExit(1)
if mode == "unknown_stderr":
    os.write(2, b"Unknown failure. RAW_CONTEXT_DIAGNOSTIC\n")
    raise SystemExit(23)
if mode == "overflow":
    sys.stdout.write("x" * 700000)
    raise SystemExit(0)
content = message["message"]["content"]
text = next(item["text"] for item in content if item["type"] == "text")
request = json.loads(text.split("\nHOST_REQUEST_JSON=", 1)[1])
init_model = "wrong-model" if mode in {"bad_model", "model_effort_tools"} else model
init_tools = ["Read"] if mode in {"bad_tools", "model_effort_tools", "effort_tools"} else ["StructuredOutput"]
init = {"type": "system", "subtype": "init", "model": init_model, "tools": init_tools}
if mode != "missing_init":
    emit(init)
if mode == "duplicate_init":
    emit(init)
if mode == "bad_tools":
    raise SystemExit(0)
if mode == "tool_event":
    emit({"type": "tool_use", "name": "Read"})
if request["kind"] == "page":
    start, end = request["frame_start"], request["frame_end"]
    indices = request["frame_indices"]
    structured = {
        "schema_version": 1,
        "coverage": {"page": request["page"], "frame_start": start, "frame_end": end,
                     "inspected_frame_indices": indices, "omitted_ranges": []},
        "observations": [{"claim": "Synthetic overview inspected",
                          "frame_indices": indices, "basis": "visual",
                          "status": "observed", "confidence": "high"}],
        "ambiguities": [],
        "drilldown_needed": [indices[-1]] if mode == "drilldown" else [],
        "limitations": []
    }
    if mode == "invalid_page_report":
        structured["coverage"]["inspected_frame_indices"] = []
    elif mode == "page_timing_literal":
        structured["observations"][0]["claim"] = "RAW_TIMING_SECRET at 09:59"
    elif mode == "page_bidi_timing_literal":
        structured["observations"][0]["claim"] = "RAW_TIMING_SECRET after one ‮sm‬"
else:
    expected = request["expected_indices"]
    structured = {
        "schema_version": 1, "status": "complete",
        "answer": ("Validated observations use host-owned citations."
                   if expected else "Validated transcript evidence."),
        "observations": ([{"claim": "Synthetic overview inspected",
                           "frame_indices": [expected[0]], "basis": "visual",
                           "status": "observed", "confidence": "high"}]
                         if expected else []),
        "coverage": {"inspected_frame_indices": expected, "omitted_ranges": []},
        "uncertainty": [], "limitations": []
    }
    if mode == "invalid_final_report":
        structured["coverage"]["inspected_frame_indices"] = []
    elif mode == "final_timing_literal":
        structured["observations"][0]["claim"] = "RAW_TIMING_SECRET after 42 seconds"
result = {
    "type": "result", "subtype": "success", "is_error": False, "num_turns": 1,
    "permission_denials": [], "modelUsage": {model: {}}, "structured_output": structured,
    "total_cost_usd": 0.001
}
if mode == "result_only":
    result.pop("structured_output")
    result["result"] = json.dumps(structured)
if mode == "permission":
    result["permission_denials"] = [{"tool_name": "Read"}]
emit(result)
"""
