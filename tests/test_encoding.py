"""Regression test: watch.py must not crash printing non-ASCII text (emoji,
accented chars, CJK, etc.) to stdout.

On Windows, the console defaults to cp1252, which cannot encode most
Unicode — a title like "She Created a Glowing Aquarium ✨❤️" previously
crashed watch.py's `print(f"- **Title:** {info['title']}")` with
UnicodeEncodeError, after download + frame extraction had already
succeeded. This mirrors the emoji-crash class of bug fixed in 0.1.2 (a
different call site: a long-video warning) that regressed in 0.2.0 at
this one (the title print).

Run as a real subprocess (not an in-process import) so the test exercises
watch.py's own stdout, not pytest's capture wrapper, which reinitializes
sys.stdout per test and would otherwise mask the fix.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

WATCH = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts" / "watch.py"


def test_prints_emoji_without_crashing():
    env = dict(os.environ)
    env.pop("PYTHONIOENCODING", None)  # exercise watch.py's own reconfigure, not an override
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import sys; sys.path.insert(0, r'{WATCH.parent}'); "
            "import watch; "
            "print('- **Title:** She Created a Glowing Aquarium ✨❤️')",
        ],
        capture_output=True,
        env=env,
    )
    assert proc.returncode == 0, f"crashed: {proc.stderr.decode('utf-8', 'replace')}"
    assert b"Aquarium" in proc.stdout
