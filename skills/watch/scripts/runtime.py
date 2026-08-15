"""Cross-platform runtime helpers shared by the bundled scripts."""
from __future__ import annotations

import os
import sys


def configure_utf8_output(streams=None) -> None:
    """Write reports as UTF-8 on Windows instead of the legacy code page.

    Video titles, paths, captions, and the report's punctuation are user-facing
    Unicode. A replacement stream may not implement ``reconfigure`` (tests and
    some agent harnesses), so unsupported streams are left untouched.
    """
    if os.name != "nt":
        return
    targets = streams if streams is not None else (sys.stdout, sys.stderr)
    for stream in targets:
        try:
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except (AttributeError, ValueError):
            pass
