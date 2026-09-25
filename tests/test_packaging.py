"""SKILL.md frontmatter stays inside the Agent Skills spec.

claude.ai's skill upload rejects frontmatter keys outside the spec
(https://agentskills.io/specification), and the fork ships `dist/watch.skill`
for exactly that upload. Claude Code tolerates extra keys, so a regression here
is invisible in the most common host and only surfaces when someone uploads the
bundle — which is why it is pinned by a test rather than left to review.

Parsed with the stdlib on purpose: the skill has no runtime dependencies and
the test suite should not grow a YAML library just to read eleven lines.
"""
from __future__ import annotations

import re
from pathlib import Path

SKILL_MD = Path(__file__).resolve().parent.parent / "skills" / "watch" / "SKILL.md"
SPEC_KEYS = {"name", "description", "license", "compatibility", "metadata", "allowed-tools"}


def _frontmatter() -> tuple[dict[str, str], dict[str, str]]:
    """Return (top-level scalars, metadata children) from SKILL.md's frontmatter."""
    lines = SKILL_MD.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "---", "SKILL.md must open with a frontmatter fence"
    end = lines.index("---", 1)
    top: dict[str, str] = {}
    meta: dict[str, str] = {}
    in_meta = False
    for line in lines[1:end]:
        if not line.strip():
            continue
        if line.startswith((" ", "\t")):
            assert in_meta, f"indented line outside metadata: {line!r}"
            key, _, value = line.strip().partition(":")
            meta[key] = value.strip()
            continue
        key, _, value = line.partition(":")
        top[key] = value.strip()
        in_meta = key == "metadata"
    return top, meta


def test_only_spec_keys_at_top_level():
    top, _ = _frontmatter()
    extra = set(top) - SPEC_KEYS
    assert not extra, f"non-spec frontmatter keys (claude.ai upload rejects these): {sorted(extra)}"


def test_required_keys_present():
    top, _ = _frontmatter()
    assert top.get("name") and top.get("description")


def test_name_constraints():
    name = _frontmatter()[0]["name"]
    assert len(name) <= 64
    assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", name), name


def test_description_within_limit():
    assert 0 < len(_frontmatter()[0]["description"]) <= 1024


def test_allowed_tools_is_space_separated():
    """The spec says space-separated; Claude Code accepts commas too, others may not."""
    tools = _frontmatter()[0].get("allowed-tools", "")
    assert "," not in tools, f"allowed-tools must be space-separated: {tools!r}"


def test_metadata_values_are_quoted_strings():
    """The spec types metadata as string -> string. An unquoted 0.4 would be a float."""
    _, meta = _frontmatter()
    for key, value in meta.items():
        assert value.startswith('"') and value.endswith('"'), f"metadata.{key} must be a quoted string: {value}"


def test_version_lives_under_metadata():
    top, meta = _frontmatter()
    assert "version" not in top
    assert re.fullmatch(r'"\d+\.\d+\.\d+"', meta.get("version", "")), meta.get("version")


def test_upstream_attribution_is_preserved():
    """CONTRIBUTING #6: the author field keeps crediting Bradley Bonanno."""
    _, meta = _frontmatter()
    assert "Bradley Bonanno" in meta.get("author", "")
