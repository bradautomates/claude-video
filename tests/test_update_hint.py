"""The 403 update hint names the package manager that owns the yt-dlp we run.

`yt-dlp -U` only updates yt-dlp's own release binaries; for a copy installed by a
package manager its source refuses ("You installed yt-dlp ... with a package
manager; Use that to update"). So the hint is only useful if it identifies the
owner. The old check looked for the substring "pipx" in the PATH entry, but pipx
and `uv tool` publish a symlink in ~/.local/bin, so the fork's own recommended
Linux install got `yt-dlp -U` — a command that cannot work there.
"""
from __future__ import annotations

import os
import shlex
import sys

import pytest

import download


PIP_SHEBANG = b"#!/home/u/venv/bin/python3\n"


@pytest.mark.parametrize("real_path,head,expected", [
    # Package managers are identified by a directory SEGMENT of the resolved path.
    ("/opt/homebrew/Cellar/yt-dlp/2026.8.19_1/libexec/bin/yt-dlp", b"#!/opt/homebrew/x/python\n", "brew upgrade yt-dlp"),
    ("/home/linuxbrew/.linuxbrew/bin/yt-dlp", b"#!/x/python\n", "brew upgrade yt-dlp"),
    ("/home/u/.local/pipx/venvs/yt-dlp/bin/yt-dlp", b"#!/home/u/.local/pipx/venvs/yt-dlp/bin/python\n", "pipx upgrade yt-dlp"),
    ("/home/u/.local/share/uv/tools/yt-dlp/bin/yt-dlp", b"#!/home/u/.local/share/uv/tools/yt-dlp/bin/python\n", "uv tool upgrade yt-dlp"),
    (r"C:\Users\u\AppData\Local\Microsoft\WinGet\Packages\yt-dlp.yt-dlp_x\yt-dlp.exe", b"MZ\x90\x00", "winget upgrade yt-dlp.yt-dlp"),
    (r"C:\Users\u\scoop\apps\yt-dlp\current\yt-dlp.exe", b"MZ\x90\x00", "scoop update yt-dlp"),
    (r"C:\ProgramData\chocolatey\lib\yt-dlp\tools\yt-dlp.exe", b"MZ\x90\x00", "choco upgrade yt-dlp"),
])
def test_package_manager_installs_name_their_manager(real_path, head, expected):
    assert download._classify_ytdlp(real_path, head) == expected


def test_segment_match_is_not_a_substring_match():
    """A folder merely *called* pipx-something must not be read as a pipx install."""
    hint = download._classify_ytdlp("/home/u/pipx-notes/venv/bin/yt-dlp", PIP_SHEBANG)
    assert "pipx upgrade" not in hint
    assert "-m pip install -U" in hint


def test_pip_install_is_updated_through_its_own_interpreter():
    """pip wrote the shebang, so that interpreter's pip reaches the right environment."""
    hint = download._classify_ytdlp("/home/u/venv/bin/yt-dlp", PIP_SHEBANG)
    assert hint == "/home/u/venv/bin/python3 -m pip install -U 'yt-dlp[default]'"


@pytest.mark.parametrize("head", [
    b"#!/usr/bin/env python3\nPK\x03\x04rest-of-zip",   # Unix release zipapp
    b"\x7fELF\x02\x01\x01",                             # yt-dlp_linux
    b"MZ\x90\x00\x03\x00",                              # yt-dlp.exe
    b"\xcf\xfa\xed\xfe\x07\x00",                        # yt-dlp_macos
])
def test_release_binaries_self_update(head):
    """Only yt-dlp's own release binaries accept -U."""
    assert download._classify_ytdlp("/home/u/bin/yt-dlp", head) == "/home/u/bin/yt-dlp -U"


def test_package_manager_beats_binary_format():
    """A winget-installed yt-dlp.exe is a PE binary but must go through winget."""
    hint = download._classify_ytdlp(r"C:\x\WinGet\Packages\yt-dlp\yt-dlp.exe", b"MZ\x90\x00")
    assert hint.startswith("winget")


def test_distro_package_points_at_the_system_manager():
    hint = download._classify_ytdlp("/usr/bin/yt-dlp", b"#! /usr/bin/python3\n")
    assert "system package manager" in hint and "pip install" not in hint


def test_unrecognised_install_does_not_guess_a_command():
    hint = download._classify_ytdlp("/opt/weird/yt-dlp", b"#!/bin/sh\nexec something\n")
    assert hint == download._UNKNOWN_OWNER
    assert "-U" not in hint


def test_paths_with_spaces_are_shell_quoted():
    hint = download._classify_ytdlp("/Users/a b/bin/yt-dlp", b"\x7fELF")
    assert hint == "'/Users/a b/bin/yt-dlp' -U"


@pytest.mark.skipif(os.name == "nt", reason="symlink fixture")
def test_real_pipx_symlink_layout_is_resolved(tmp_path_factory, monkeypatch):
    """End to end on a real on-disk layout: the regression this change fixes.

    pipx puts the script in ~/.local/pipx/venvs/yt-dlp/bin and a symlink in
    ~/.local/bin. The PATH entry contains no "pipx"; only the resolved path does.
    A neutral root, not tmp_path: tmp_path is named after this test, whose name
    contains "pipx", which would hand the old substring check the answer.
    """
    home = tmp_path_factory.mktemp("home")
    venv_bin = home / ".local" / "pipx" / "venvs" / "yt-dlp" / "bin"
    venv_bin.mkdir(parents=True)
    script = venv_bin / "yt-dlp"
    script.write_bytes(f"#!{venv_bin / 'python'}\n".encode())
    script.chmod(0o755)
    local_bin = home / ".local" / "bin"
    local_bin.mkdir(parents=True)
    (local_bin / "yt-dlp").symlink_to(script)

    monkeypatch.setattr(download, "ytdlp_cmd", lambda: [str(local_bin / "yt-dlp")])
    assert "pipx" not in str(local_bin / "yt-dlp"), "fixture must not leak the answer into the PATH entry"
    assert download._update_hint() == "pipx upgrade yt-dlp"


def test_watch_ytdlp_python_module_form_uses_that_interpreter(monkeypatch):
    """WATCH_YTDLP="python -m yt_dlp" is honoured: the hint follows the copy we run."""
    monkeypatch.setattr(download, "ytdlp_cmd", lambda: [sys.executable, "-m", "yt_dlp"])
    assert download._update_hint() == f"{shlex.quote(sys.executable)} -m pip install -U 'yt-dlp[default]'"


def test_unreadable_binary_degrades_to_the_generic_hint(monkeypatch, tmp_path):
    monkeypatch.setattr(download, "ytdlp_cmd", lambda: [str(tmp_path / "missing" / "yt-dlp")])
    assert download._update_hint() == download._UNKNOWN_OWNER


@pytest.mark.parametrize("output,is_403", [
    ("ERROR: unable to download video data: HTTP Error 403: Forbidden\n", True),
    ("ERROR: [youtube] abc: Forbidden\n", True),
    ("ERROR: [generic] 'x4031y' is not a valid URL\n", False),   # "403" inside an id
    ("downloaded 14032 bytes then failed\n", False),            # "403" inside a number
])
def test_403_is_matched_as_a_word(output, is_403, tmp_path, monkeypatch):
    """A substring test blamed yt-dlp's age for any failure whose log contained "403"."""
    monkeypatch.setattr(download, "_yt_dlp_version", lambda: None)
    monkeypatch.setattr(download, "_update_hint", lambda: "<cmd>")
    msg = download._download_failure_message(output, 1, tmp_path, None)
    assert ("Update and retry once" in msg) is is_403
