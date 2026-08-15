"""Cross-platform process output setup."""
from __future__ import annotations

import runtime


class FakeStream:
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.calls: list[dict] = []

    def reconfigure(self, **kwargs):
        if self.fail:
            raise ValueError("stream already detached")
        self.calls.append(kwargs)


def test_windows_streams_are_reconfigured_to_utf8(monkeypatch):
    monkeypatch.setattr(runtime.os, "name", "nt")
    stdout = FakeStream()
    stderr = FakeStream()

    runtime.configure_utf8_output((stdout, stderr))

    expected = {"encoding": "utf-8", "errors": "backslashreplace"}
    assert stdout.calls == [expected]
    assert stderr.calls == [expected]


def test_replacement_stream_without_reconfigure_is_ignored(monkeypatch):
    monkeypatch.setattr(runtime.os, "name", "nt")
    runtime.configure_utf8_output((object(), FakeStream(fail=True)))
