# Setup preflight

Trusted visual review currently requires Linux with `/proc/self/fd`, a native ELF Claude CLI 2.1.220+, `/usr/bin/setpriv`, `/usr/bin/unshare`, user/PID namespaces, sealed `memfd`, and `O_TMPFILE`. macOS, Windows, claude.ai, and non-Claude hosts may extract media but cannot complete trusted review; report `BLOCKED/UNKNOWN`, never inspect pixels in the coordinator.

Before any download, extraction, or provider call, resolve the native Claude binary and run:

```bash
python3 "$SKILL_DIR/scripts/visual_harness.py" inspect-runtime \
  --claude-bin "<resolved native Claude CLI>"
```

A bounded failure means unsupported runtime. Stop before media processing.

Run once per session:

```bash
python3 "$SKILL_DIR/scripts/setup.py" --json
```

- `can_proceed=true`, `first_run=false`: continue silently.
- `first_run=true`: install missing binaries first; rerun setup to scaffold `~/.config/watch/.env`; offer Whisper key choice; ask detail preference; set `SETUP_COMPLETE=true`.
- `can_proceed=false`, `first_run=false`: repair missing binaries; do not re-ask preferences.

Later invocations:

```bash
python3 "$SKILL_DIR/scripts/setup.py" --check
```

Exit 0: silent. Exit 2: missing `ffmpeg`/`ffprobe`/`yt-dlp`. Exit 3: first-run missing key. Exit 4: both. Run the idempotent installer when needed:

```bash
python3 "$SKILL_DIR/scripts/setup.py"
```

Linux/Windows installer output gives user-run commands. Do not mutate host packages without approval. macOS Homebrew setup may install dependencies only within granted authority.

A Whisper key is encouraged, not required. Ask Groq vs OpenAI only on genuine first run. Store only in `~/.config/watch/.env` mode `0600`:

```text
GROQ_API_KEY=...
# or OPENAI_API_KEY=...
WATCH_DETAIL=balanced
SETUP_COMPLETE=true
```

Never add trailing inline comments to values. Keyless choice: use `--no-whisper`; report frames-only when captions are absent.

Ask detail preference in this exact order:

1. `transcript` — transcript only; no ordinary frames
2. `efficient` — keyframes, cap 50
3. `balanced` (recommended) — scene-aware, cap 100
4. `token-burner` — scene-aware, uncapped

Do not ask again after `SETUP_COMPLETE=true`.
