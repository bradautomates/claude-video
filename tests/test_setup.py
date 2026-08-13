"""setup.py --json surfaces the resolved watch detail and binary-presence status."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

SETUP = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts" / "setup.py"


def _run(args, *, home=None, extra_env=None):
    env = dict(os.environ)
    env.pop("WATCH_DETAIL", None)
    env.pop("SETUP_COMPLETE", None)
    if home is not None:
        env["HOME"] = str(home)
        env["USERPROFILE"] = str(home)  # Windows
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, str(SETUP), *args],
        capture_output=True, text=True, env=env,
    )


def test_json_reports_watch_detail():
    proc = _run(["--json"])
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["watch_detail"] == "balanced"


def test_check_passes_when_binaries_present():
    """ffmpeg/ffprobe/yt-dlp are dev-machine prerequisites — --check is silent when present."""
    chk = _run(["--check"])
    assert chk.returncode == 0, chk.stderr
    assert chk.stdout == "" and chk.stderr == ""

    js = json.loads(_run(["--json"]).stdout)
    assert js["status"] == "ready"
    assert js["can_proceed"] is True
    assert js["missing_binaries"] == []
