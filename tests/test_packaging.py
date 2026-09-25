"""Cross-host layout and version consistency are release invariants."""
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_frontmatter_uses_only_agent_skills_keys():
    # claude.ai uploads reject any other top-level key.
    text = (ROOT / 'skills/watch/SKILL.md').read_text(encoding='utf-8')
    frontmatter = text.split('---\n')[1]
    keys = {line.split(':')[0] for line in frontmatter.splitlines() if line and not line.startswith(' ')}
    assert keys <= {'name', 'description', 'license', 'allowed-tools', 'metadata', 'compatibility'}


def test_release_versions_match_canonical_skill():
    skill = ROOT / 'skills/watch/SKILL.md'
    version = re.search(r'^  version: "([^"]+)"', skill.read_text(encoding='utf-8'), re.M).group(1)
    for relative in ('.claude-plugin/plugin.json', '.codex-plugin/plugin.json'):
        assert json.loads((ROOT / relative).read_text(encoding='utf-8'))['version'] == version
    assert json.loads((ROOT / '.codex-plugin/plugin.json').read_text(encoding='utf-8'))['skills'] == './skills/'
    assert not (ROOT / 'commands').exists()


def test_copied_skill_runs_outside_repository(tmp_path, static_clip):
    target = tmp_path / 'other host' / 'watch'
    shutil.copytree(ROOT / 'skills/watch', target, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    result = subprocess.run([sys.executable, str(target / 'scripts/watch.py'), str(static_clip), '--no-whisper', '--out-dir', str(tmp_path / 'output')],
                            capture_output=True, encoding='utf-8', env=dict(os.environ))
    assert result.returncode == 0, result.stderr
    assert '**Detail:** balanced' in result.stdout and 'reason=uniform' in result.stdout


def test_skill_documents_and_bundles_the_gemini_engine():
    skill = (ROOT / 'skills/watch/SKILL.md').read_text(encoding='utf-8')
    assert (ROOT / 'skills/watch/scripts/gemini.py').is_file()
    for needle in ('`gemini.py`', '--engine', '--question', 'GEMINI_API_KEY', '--engine local',
                   'uploaded to Google', 'Answer (from Gemini)'):
        assert needle in skill, needle
    assert 'CLAUDE_SKILL_DIR' not in skill
    runtime = sorted(p.name for p in (ROOT / 'skills/watch/scripts').glob('*.py'))
    assert runtime == ['config.py', 'download.py', 'frames.py', 'gemini.py', 'local_whisperx.py',
                       'runtime.py', 'setup.py', 'transcribe.py', 'watch.py', 'whisper.py']


def test_copied_skill_runs_gemini_engine_check_outside_repository(tmp_path):
    target = tmp_path / 'other host' / 'watch'
    shutil.copytree(ROOT / 'skills/watch', target, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    env = {k: v for k, v in os.environ.items() if k != 'GEMINI_API_KEY'}
    result = subprocess.run([sys.executable, str(target / 'scripts/watch.py'), 'https://youtu.be/abc', '--engine', 'gemini'],
                            capture_output=True, encoding='utf-8', env=env)
    assert result.returncode != 0 and 'GEMINI_API_KEY' in result.stderr
