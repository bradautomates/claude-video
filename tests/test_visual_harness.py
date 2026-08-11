"""Host-enforced, tool-less visual review harness contracts."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import presentation
import visual_harness
from support_fake_cli import FAKE_CLAUDE


def _jpeg(path: Path, color: str, size: str = "320x180") -> None:
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c={color}:s={size}",
            "-frames:v",
            "1",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def _work(tmp_path: Path, page_count: int = 1, frames_per_page: int = 2) -> Path:
    work = tmp_path / "review"
    frames_dir = work / "frames"
    overview_dir = work / "overview"
    frames_dir.mkdir(parents=True)
    overview_dir.mkdir()
    frames = []
    pages = []
    frame_index = 0
    for page in range(1, page_count + 1):
        page_path = overview_dir / f"overview_{page:04d}.jpg"
        tiles = []
        start = frame_index
        frame_limit = frames_per_page if page < page_count else min(frames_per_page, 10)
        for _ in range(frame_limit):
            path = frames_dir / f"frame_{frame_index:04d}.jpg"
            _jpeg(path, "red" if frame_index % 2 else "green")
            frames.append(
                {
                    "index": frame_index,
                    "timestamp_seconds": frame_index * 0.5,
                    "path": str(path),
                    "reason": "scene-change",
                }
            )
            tiles.append(
                {
                    "tile": len(tiles),
                    "frame_index": frame_index,
                    "timestamp_seconds": frame_index * 0.5,
                    "reason": "scene-change",
                }
            )
            frame_index += 1
        presentation.create_contact_sheet(frames[start:frame_index], page_path)
        pages.append(
            {
                "page": page,
                "frame_start": start,
                "frame_end": frame_index - 1,
                "tiles": tiles,
                "kind": "image",
                "path": str(page_path),
            }
        )
    for page in pages:
        page["frame_sha256"] = [
            hashlib.sha256(Path(frames[index]["path"]).read_bytes()).hexdigest()
            for index in range(page["frame_start"], page["frame_end"] + 1)
        ]
    manifest = {
        "schema_version": 1,
        "source": {
            "path": str(work / "source.mp4"),
            "metadata": {"duration_seconds": frame_index * 0.5},
        },
        "frame_count": len(frames),
        "frames": frames,
        "overview": {
            "page_count": len(pages),
            "page_size": 20,
            "tile_width": 320,
            "tile_height": 180,
            "pages": pages,
        },
    }
    (work / "frame-index.json").write_text(json.dumps(manifest), encoding="utf-8")
    (work / "transcript.txt").write_text(
        "[00:00] synthetic transcript", encoding="utf-8"
    )
    return work


def _compile_fake(tmp_path: Path, script: Path) -> Path:
    launcher = tmp_path / "claude"
    source = tmp_path / "fake-claude.c"
    source.write_text(
        "#include <stdlib.h>\n"
        "#include <unistd.h>\n"
        "int main(int argc, char **argv) {\n"
        "  char **args = calloc((size_t) argc + 2, sizeof(char *));\n"
        "  if (!args) return 127;\n"
        "  args[0] = " + json.dumps(sys.executable) + ";\n"
        "  args[1] = " + json.dumps(str(script)) + ";\n"
        "  for (int i = 1; i < argc; i++) args[i + 1] = argv[i];\n"
        "  execv(args[0], args);\n"
        "  return 127;\n"
        "}\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        ["cc", "-O2", "-o", str(launcher), str(source)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    return launcher


def _fake(tmp_path: Path, mode: str = "valid") -> tuple[Path, Path]:
    script = tmp_path / "fake-claude.py"
    log = tmp_path / "claude.log"
    source = FAKE_CLAUDE.replace("__FAKE_MODE__", repr(mode)).replace(
        "__FAKE_LOG__",
        repr(str(log)),
    )
    script.write_text(source, encoding="utf-8")
    return _compile_fake(tmp_path, script), log


def _request(
    inspection: dict,
    claude: Path,
    *,
    allow_large_run: bool = False,
    per_child_timeout_seconds: float = 2.0,
) -> visual_harness.ReviewRequest:
    runtime_identity = visual_harness.runtime.inspect_runtime(claude)
    return visual_harness.ReviewRequest(
        digest=inspection["digest"],
        runtime_digest=runtime_identity["runtime_digest"],
        approved_page_count=inspection["page_count"],
        approved_child_count=inspection["maximum_child_count"],
        model="cx/gpt-5.6-sol",
        effort="xhigh",
        per_child_budget_usd=0.10,
        aggregate_budget_usd=inspection["maximum_child_count"] * 0.10,
        per_child_timeout_seconds=per_child_timeout_seconds,
        review_timeout_seconds=max(
            per_child_timeout_seconds,
            inspection["maximum_child_count"] * per_child_timeout_seconds,
        ),
        question="Describe the visual sequence.",
        allow_large_run=allow_large_run,
    )


def _transcript_work(tmp_path: Path) -> Path:
    work = tmp_path / "transcript-review"
    work.mkdir()
    manifest = {
        "schema_version": 1,
        "source": {"path": None, "metadata": {"duration_seconds": 1.0}},
        "frame_count": 0,
        "frames": [],
        "overview": {
            "page_count": 0,
            "page_size": 20,
            "tile_width": 320,
            "tile_height": 180,
            "pages": [],
        },
    }
    (work / "frame-index.json").write_text(json.dumps(manifest), encoding="utf-8")
    (work / "transcript.txt").write_text("[00:00] transcript only", encoding="utf-8")
    return work


def test_inspect_binds_complete_visual_input(tmp_path: Path) -> None:
    work = _work(tmp_path, page_count=2)

    result = visual_harness.inspect_work(work)

    assert result["page_count"] == 2
    assert result["frame_count"] == 4
    assert result["frame_range"] == [0, 3]
    assert result["approval_required"] is False
    assert result["maximum_child_count"] > result["page_count"]
    assert len(result["digest"]) == 64


def test_transcript_only_inspect_and_review_use_one_toolless_reducer(
    tmp_path: Path,
) -> None:
    work = _transcript_work(tmp_path)
    inspection = visual_harness.inspect_work(work)
    claude, log = _fake(tmp_path)

    assert inspection["page_count"] == 0
    assert inspection["frame_count"] == 0
    assert inspection["frame_range"] == []
    assert inspection["transcript_chunk_count"] == 1
    assert inspection["maximum_child_count"] == 1

    result = visual_harness.review_work(
        work,
        _request(inspection, claude),
        claude_bin=claude,
    )

    assert result["final"]["coverage"]["inspected_frame_indices"] == []
    assert result["final"]["status"] == "partial"
    assert result["final"]["evidence_mode"] == "transcript_only"
    assert result["final"]["answer"] == "Validated transcript evidence."
    assert result["final"]["limitations"] == []
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(calls) == 1
    assert calls[0]["inputs"] == ["control_request", "user"]
    message = json.loads(calls[0]["stdin"])
    assert all(item["type"] == "text" for item in message["message"]["content"])


def test_inspect_rejects_raster_change(tmp_path: Path) -> None:
    work = _work(tmp_path)
    visual_harness.inspect_work(work)
    _jpeg(work / "overview/overview_0001.jpg", "yellow", "1280x180")

    with pytest.raises(visual_harness.HarnessError, match="source frames"):
        visual_harness.inspect_work(work)


def test_inspect_rejects_manifest_traversal(tmp_path: Path) -> None:
    work = _work(tmp_path)
    manifest_path = work / "frame-index.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["overview"]["pages"][0]["path"] = str(work / "../outside.jpg")
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(visual_harness.HarnessError, match="outside work directory"):
        visual_harness.inspect_work(work)


def test_inspect_rejects_invalid_source_frame_digest_shape(tmp_path: Path) -> None:
    work = _work(tmp_path)
    manifest_path = work / "frame-index.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["overview"]["pages"][0]["frame_sha256"] = ["not-a-digest"]
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(visual_harness.HarnessError, match="digest is invalid"):
        visual_harness.inspect_work(work)


def test_inspect_rejects_overview_not_derived_from_source_frames(tmp_path: Path) -> None:
    work = _work(tmp_path)
    _jpeg(work / "overview/overview_0001.jpg", "yellow", "1280x180")

    with pytest.raises(visual_harness.HarnessError, match="source frames"):
        visual_harness.inspect_work(work)


def test_inspect_rejects_overview_with_wrong_dimensions(tmp_path: Path) -> None:
    work = _work(tmp_path)
    _jpeg(work / "overview/overview_0001.jpg", "yellow", "640x180")

    with pytest.raises(visual_harness.HarnessError, match="overview image dimensions"):
        visual_harness.inspect_work(work)


def test_inspect_rejects_symlinked_image(tmp_path: Path) -> None:
    work = _work(tmp_path)
    image = work / "overview/overview_0001.jpg"
    outside = tmp_path / "outside.jpg"
    image.replace(outside)
    image.symlink_to(outside)

    with pytest.raises(visual_harness.HarnessError, match="regular non-symlink"):
        visual_harness.inspect_work(work)


def test_large_run_requires_explicit_approval(tmp_path: Path) -> None:
    work = _work(tmp_path, page_count=6)
    inspection = visual_harness.inspect_work(work)
    claude, log = _fake(tmp_path)

    assert inspection["approval_required"] is True
    with pytest.raises(visual_harness.HarnessError, match="large run approval"):
        visual_harness.review_work(
            work, _request(inspection, claude), claude_bin=claude
        )


def test_review_uses_toolless_safe_child_and_private_structured_output(
    tmp_path: Path,
) -> None:
    work = _work(tmp_path, page_count=2)
    inspection = visual_harness.inspect_work(work)
    claude, log = _fake(tmp_path)

    result = visual_harness.review_work(
        work,
        _request(inspection, claude),
        claude_bin=claude,
    )

    assert result["final"]["status"] == "complete"
    assert result["final"]["coverage"]["inspected_frame_indices"] == [0, 1, 2, 3]
    assert result["evidence"]["path"] == "watch-review-v2.json"
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(calls) >= 3
    for call in calls:
        argv = call["argv"]
        assert "--safe-mode" in argv
        assert argv[argv.index("--tools") + 1] == ""
        assert argv[argv.index("--permission-mode") + 1] == "dontAsk"
        assert "--disable-slash-commands" in argv
        assert "--no-session-persistence" in argv
        assert argv[argv.index("--model") + 1] == "cx/gpt-5.6-sol"
        assert argv[argv.index("--effort") + 1] == "xhigh"
        assert "--json-schema" in argv
        assert "frame-index.json" not in call["stdin"]
        assert str(work) not in call["stdin"]
    assert "source" not in json.dumps(result).casefold()


def test_review_passes_page_and_reducer_specific_schemas(tmp_path: Path) -> None:
    work = _work(tmp_path)
    inspection = visual_harness.inspect_work(work)
    claude, log = _fake(tmp_path)

    visual_harness.review_work(work, _request(inspection, claude), claude_bin=claude)

    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(calls) == 2
    page_schema = json.loads(
        calls[0]["argv"][calls[0]["argv"].index("--json-schema") + 1]
    )
    final_schema = json.loads(
        calls[1]["argv"][calls[1]["argv"].index("--json-schema") + 1]
    )
    assert page_schema["properties"]["observations"]["minItems"] == 1
    assert page_schema["properties"]["drilldown_needed"]["maxItems"] == 4
    assert final_schema["properties"]["status"] == {
        "type": "string",
        "const": "complete",
    }
    coverage = final_schema["properties"]["coverage"]["properties"]
    assert coverage["inspected_frame_indices"]["const"] == [0, 1]
    assert coverage["omitted_ranges"]["const"] == []


def test_inspect_rejects_review_that_cannot_fit_evidence_bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    work = _work(tmp_path, page_count=2)
    monkeypatch.setattr(visual_harness, "MAX_EVIDENCE_BUNDLE_BYTES", 1)

    with pytest.raises(visual_harness.HarnessError, match="evidence bundle limit"):
        visual_harness.inspect_work(work)


def test_reducer_rejects_empty_evidence_before_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_range(*_args: object) -> None:
        raise AssertionError("empty evidence entered reducer loop")

    monkeypatch.setattr(visual_harness, "range", unexpected_range, raising=False)

    with pytest.raises(visual_harness.HarnessError, match="reducer evidence is empty"):
        visual_harness._reduce_reports(
            [],
            expected_indices=[],
            timestamp_by_index={},
            request=None,
            executable=None,
            helpers=None,
            deadline=0.0,
            initial_ordinal=1,
            initial_cost=0.0,
        )


def test_review_preserves_image_bytes_in_stream_input(tmp_path: Path) -> None:
    work = _work(tmp_path)
    inspection = visual_harness.inspect_work(work)
    claude, log = _fake(tmp_path)

    visual_harness.review_work(work, _request(inspection, claude), claude_bin=claude)

    first = json.loads(log.read_text().splitlines()[0])
    message = json.loads(first["stdin"])
    image = next(
        item for item in message["message"]["content"] if item["type"] == "image"
    )
    assert (
        visual_harness.decode_image_block(image)
        == (work / "overview/overview_0001.jpg").read_bytes()
    )


@pytest.mark.parametrize(
    "mode,match",
    [
        ("bad_model", "model identity"),
        ("result_only", "protocol validation"),
        ("permission", "permission denial"),
        ("tool_event", "permission denial"),
        ("overflow", "output limit"),
    ],
)
def test_review_fails_closed_on_invalid_child(
    tmp_path: Path,
    mode: str,
    match: str,
) -> None:
    work = _work(tmp_path)
    inspection = visual_harness.inspect_work(work)
    claude, log = _fake(tmp_path, mode)

    with pytest.raises(visual_harness.HarnessError, match=match):
        visual_harness.review_work(
            work,
            _request(inspection, claude),
            claude_bin=claude,
        )

    assert not (work / "watch-review-v2.json").exists()


def test_review_surfaces_validated_drilldown_indices_without_publishing(
    tmp_path: Path,
) -> None:
    work = _work(tmp_path)
    inspection = visual_harness.inspect_work(work)
    claude, log = _fake(tmp_path, "drilldown")

    with pytest.raises(
        visual_harness.HarnessError,
        match=r"separately approved focused pass; validated frame indices: \[1\]",
    ):
        visual_harness.review_work(
            work,
            _request(inspection, claude),
            claude_bin=claude,
        )

    assert not (work / "watch-review-v2.json").exists()


def test_review_times_out_and_leaves_no_report(tmp_path: Path) -> None:
    work = _work(tmp_path)
    inspection = visual_harness.inspect_work(work)
    claude, log = _fake(tmp_path, "timeout")

    with pytest.raises(visual_harness.HarnessError, match="timed out"):
        visual_harness.review_work(
            work,
            _request(inspection, claude, per_child_timeout_seconds=0.2),
            claude_bin=claude,
        )

    assert not (work / "watch-review-v2.json").exists()


def test_review_does_not_block_on_child_stderr_or_stdout_pipe(tmp_path: Path) -> None:
    work = _work(tmp_path)
    inspection = visual_harness.inspect_work(work)
    claude, log = _fake(tmp_path, "no_read")
    with pytest.raises(visual_harness.HarnessError, match="timed out"):
        visual_harness.review_work(
            work,
            _request(inspection, claude, per_child_timeout_seconds=0.2),
            claude_bin=claude,
        )


def test_review_rejects_stale_approval_before_child_call(tmp_path: Path) -> None:
    work = _work(tmp_path)
    inspection = visual_harness.inspect_work(work)
    claude, log = _fake(tmp_path)
    _jpeg(work / "overview/overview_0001.jpg", "black", "1280x180")

    with pytest.raises(
        visual_harness.HarnessError,
        match="not derived from source frames|digest mismatch",
    ):
        visual_harness.review_work(
            work, _request(inspection, claude), claude_bin=claude
        )

    assert not log.exists()


def test_stale_approval_is_rejected_before_cli_probe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    work = _work(tmp_path)
    inspection = visual_harness.inspect_work(work)
    claude, log = _fake(tmp_path)
    request = _request(inspection, claude)
    _jpeg(work / "overview/overview_0001.jpg", "black", "1280x180")
    probed = False

    def reject_probe(*_args: object, **_kwargs: object) -> None:
        nonlocal probed
        probed = True
        raise AssertionError("CLI probe preceded approval validation")

    monkeypatch.setattr(visual_harness.runtime, "check_claude_cli", reject_probe)

    with pytest.raises(
        visual_harness.HarnessError,
        match="not derived from source frames|digest mismatch",
    ):
        visual_harness.review_work(work, request, claude_bin=claude)

    assert probed is False
    assert not log.exists()


def test_page_contract_binds_partial_second_page_to_global_indices(
    tmp_path: Path,
) -> None:
    work = _work(tmp_path, page_count=2, frames_per_page=20)
    inspection = visual_harness.inspect_work(work)
    claude, log = _fake(tmp_path)

    result = visual_harness.review_work(
        work,
        _request(inspection, claude),
        claude_bin=claude,
    )

    assert result["final"]["coverage"]["inspected_frame_indices"] == list(range(30))
    page_calls = []
    for line in log.read_text(encoding="utf-8").splitlines():
        call = json.loads(line)
        message = json.loads(call["stdin"])
        text = next(
            item["text"]
            for item in message["message"]["content"]
            if item["type"] == "text"
        )
        request = json.loads(text.split("\nHOST_REQUEST_JSON=", 1)[1])
        if request["kind"] == "page":
            page_calls.append(request)
    assert page_calls[1]["frame_indices"] == list(range(20, 30))


def test_review_revalidates_complete_digest_before_each_page(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    work = _work(tmp_path, page_count=2)
    inspection = visual_harness.inspect_work(work)
    claude, log = _fake(tmp_path)
    original_run_child = visual_harness._run_child
    page_calls = 0

    def mutate_after_first_page(*args: object, **kwargs: object):
        nonlocal page_calls
        result = original_run_child(*args, **kwargs)
        if kwargs.get("phase") == "page":
            page_calls += 1
            if page_calls == 1:
                _jpeg(work / "frames/frame_0003.jpg", "yellow")
        return result

    monkeypatch.setattr(visual_harness, "_run_child", mutate_after_first_page)

    with pytest.raises(
        visual_harness.HarnessError,
        match="approved media changed|source frame digest mismatch",
    ):
        visual_harness.review_work(
            work,
            _request(inspection, claude),
            claude_bin=claude,
        )

    assert page_calls == 1
    assert len(log.read_text(encoding="utf-8").splitlines()) == 1
    assert not (work / "watch-review-v2.json").exists()


def test_reports_are_private_and_never_overwritten(tmp_path: Path) -> None:
    if os.name != "posix":
        pytest.skip("POSIX mode contract")
    work = _work(tmp_path)
    inspection = visual_harness.inspect_work(work)
    claude, log = _fake(tmp_path)
    request = _request(inspection, claude)

    result = visual_harness.review_work(work, request, claude_bin=claude)

    bundle = work / "watch-review-v2.json"
    assert bundle.stat().st_mode & 0o777 == 0o400
    before = bundle.read_bytes()
    payload = json.loads(before)
    assert payload["schema_version"] == 2
    assert payload["pages"][0]["coverage"]["page"] == 1
    assert payload["final"]["status"] == "complete"
    assert payload["lineage"]["schema_version"] == 2
    with visual_harness.runtime.pin_directory(work) as pinned:
        verified = visual_harness.publication.verify_bundle(
            pinned,
            result["evidence"],
        )
    assert verified == before
    with pytest.raises(visual_harness.HarnessError, match="already exists"):
        visual_harness.review_work(work, request, claude_bin=claude)
    assert bundle.read_bytes() == before


def test_cli_check_rejects_old_binary(tmp_path: Path) -> None:
    script = tmp_path / "old-claude.py"
    script.write_text("print('2.1.100 (Claude Code)')\n", encoding="utf-8")
    executable = _compile_fake(tmp_path, script)

    with pytest.raises(visual_harness.HarnessError, match="2.1.220"):
        visual_harness.check_claude_cli(executable)


def test_inspect_rejects_manifest_metadata_mismatch(tmp_path: Path) -> None:
    work = _work(tmp_path)
    manifest_path = work / "frame-index.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["overview"]["page_size"] = 99
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(visual_harness.HarnessError, match="overview metadata"):
        visual_harness.inspect_work(work)


def test_review_rejects_mismatched_provider_accounting_model(tmp_path: Path) -> None:
    work = _work(tmp_path)
    inspection = visual_harness.inspect_work(work)
    claude, _log = _fake(tmp_path)

    script = claude.parent / "fake-claude.py"
    fake = script.read_text(encoding="utf-8").replace(
        '"modelUsage": {model: {}},',
        '"modelUsage": {"provider-accounting-model": {}},',
    )
    script.write_text(fake, encoding="utf-8")

    with pytest.raises(visual_harness.ChildFailure) as caught:
        visual_harness.review_work(
            work,
            _request(inspection, claude),
            claude_bin=claude,
        )

    assert caught.value.category == "identity"
    assert caught.value.metadata["identity_dimension"] == "model"


def test_review_rejects_invalid_model_usage_record(tmp_path: Path) -> None:
    work = _work(tmp_path)
    inspection = visual_harness.inspect_work(work)
    claude, log = _fake(tmp_path)

    script = claude.parent / "fake-claude.py"
    fake = script.read_text(encoding="utf-8").replace(
        '"modelUsage": {model: {}},',
        '"modelUsage": {model: []},',
    )
    script.write_text(fake, encoding="utf-8")

    with pytest.raises(visual_harness.HarnessError, match="protocol validation"):
        visual_harness.review_work(
            work,
            _request(inspection, claude),
            claude_bin=claude,
        )


def test_review_rejects_budget_above_approved_ceiling(tmp_path: Path) -> None:
    work = _work(tmp_path)
    inspection = visual_harness.inspect_work(work)
    claude, log = _fake(tmp_path)
    request = _request(inspection, claude)
    request = visual_harness.ReviewRequest(
        **{
            **request.__dict__,
            "per_child_budget_usd": 10_000.0,
            "aggregate_budget_usd": 20_000.0,
        },
    )

    with pytest.raises(visual_harness.HarnessError, match="budget ceiling"):
        visual_harness.review_work(
            work,
            request,
            claude_bin=claude,
        )

    assert not log.exists()


def test_review_resolves_default_claude_symlink(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    work = _work(tmp_path)
    inspection = visual_harness.inspect_work(work)
    claude, _log = _fake(tmp_path)
    link = tmp_path / "claude-link"
    link.symlink_to(claude)
    monkeypatch.setattr(visual_harness.shutil, "which", lambda _name: str(link))

    result = visual_harness.review_work(work, _request(inspection, claude))

    assert result["final"]["status"] == "complete"


def test_review_rejects_symlinked_claude_binary(tmp_path: Path) -> None:
    work = _work(tmp_path)
    inspection = visual_harness.inspect_work(work)
    claude, log = _fake(tmp_path)
    link = tmp_path / "claude-link"
    link.symlink_to(claude)

    with pytest.raises(visual_harness.HarnessError, match="executable"):
        visual_harness.review_work(
            work,
            _request(inspection, claude),
            claude_bin=link,
        )

    assert not log.exists()


def test_review_rejects_script_claude_binary(tmp_path: Path) -> None:
    work = _work(tmp_path)
    inspection = visual_harness.inspect_work(work)
    script = tmp_path / "claude-script"
    script.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    script.chmod(0o755)

    with pytest.raises(visual_harness.HarnessError, match="native executable"):
        visual_harness.review_work(
            work,
            _request(inspection, _fake(tmp_path)[0]),
            claude_bin=script,
        )


def test_verify_bundle_returns_the_descriptor_verified_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "review"
    root.mkdir(mode=0o700)
    raw = b'{"verified":true}\n'
    replacement = b'{"replacement":true}\n'

    with visual_harness.runtime.pin_directory(root) as pinned:
        with visual_harness.publication.locked_evidence(pinned):
            receipt = visual_harness.publication.publish_bundle(pinned, raw)
        original_close = visual_harness.publication.os.close
        replaced = False

        def replace_after_close(descriptor: int) -> None:
            nonlocal replaced
            original_close(descriptor)
            if not replaced:
                target = root / "watch-review-v2.json"
                target.unlink()
                target.write_bytes(replacement)
                target.chmod(0o400)
                replaced = True

        monkeypatch.setattr(
            visual_harness.publication.os,
            "close",
            replace_after_close,
        )
        verified = visual_harness.publication.verify_bundle(pinned, receipt)

    assert verified == raw
    assert (root / "watch-review-v2.json").read_bytes() == replacement
