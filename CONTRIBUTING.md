# Contributing

Pull requests are welcome — this fork exists because a queue of good ones had nowhere to land.

## The short version

1. Branch from `main`, open a PR against `main`. CI must be green on all five targets (Ubuntu ffmpeg 4.4 / 6.1, macOS, Windows 8.x / 9.x).
2. **Tests with the change.** The suite is offline (`python -m pytest`); stub binaries and ffmpeg-synthesized clips are in `tests/conftest.py`. Nothing in `tests/` may read the developer's real `~/.config/watch/.env` or need a network.
3. **No new runtime dependencies** in `skills/watch/scripts/` — the skill is stdlib + `ffmpeg` + `yt-dlp` and must stay installable by copying a directory. Optional tools (Parakeet, cookies, local servers) are reached by shelling out or by an env var, never by `import`.
4. **One problem per PR.** A bundle that touches five concerns gets deferred, not reviewed (see `UPSTREAM.md` for how that went upstream).
5. Add a line to `CHANGELOG.md` under *Unreleased*. If the idea or code comes from an upstream PR or issue, cite it (`from #215`) and add or update the row in `UPSTREAM.md`; you'll be credited in `AUTHORS.md`.
6. Keep the `author` fields on Bradley Bonanno and the license notice untouched.

## Labels

- `fork-feature` — a feature that originates here, not upstream. One issue per feature.
- `upstream-pr` / `deferred` — an upstream PR we could not take as-is; the issue says what is still wanted.
- `roadmap` — planned work.

## Conventions the code relies on

- Every `subprocess.run(..., text=True)` passes `encoding="utf-8", errors="replace"` (Windows consoles).
- Resolve a binary with `shutil.which()` and run the resolved path (`config.ytdlp_cmd()`), never the bare name.
- Settings are read through `config.read_env_value()` — env first, then `~/.config/watch/.env`, then `./.env`.
- `setup.py --check` must print **nothing** on success; anything handled automatically is not a warning.
- The report footer names the working dir as *temporary* or *user-supplied*; never delete a user-supplied one.

## Releasing (maintainers)

Bump `version` in `skills/watch/SKILL.md`, `.claude-plugin/plugin.json` and `.codex-plugin/plugin.json` — Claude Code keys plugin updates on that string — move *Unreleased* to a dated section, tag `vX.Y.Z`. The release workflow attaches `watch.skill`.
