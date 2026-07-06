#!/usr/bin/env bash
# SessionStart hook for /watch — one-line status so users know what's wired up.
# Silent on ready state to avoid spam. Points at the installer when something
# is missing.
set -euo pipefail

CONFIG_FILE="$HOME/.config/watch/.env"

# Warn if the secrets file has loose permissions.
if [[ -f "$CONFIG_FILE" ]]; then
  perms=$(stat -c '%a' "$CONFIG_FILE" 2>/dev/null || stat -f '%Lp' "$CONFIG_FILE" 2>/dev/null || echo "")
  if [[ -n "$perms" && "$perms" != "600" && "$perms" != "400" ]]; then
    echo "/watch: WARNING — $CONFIG_FILE has permissions $perms (should be 600)."
    echo "  Fix: chmod 600 $CONFIG_FILE"
  fi
fi

# Load API keys from the config file without exporting them.
read_key() {
  local name="$1"
  if [[ -n "${!name:-}" ]]; then
    echo "${!name}"
    return
  fi
  if [[ -f "$CONFIG_FILE" ]]; then
    awk -F= -v k="$name" '
      /^[[:space:]]*#/ { next }
      $1 == k {
        sub(/^[[:space:]]*/, "", $2); sub(/[[:space:]]*$/, "", $2);
        gsub(/^["'\'']|["'\'']$/, "", $2);
        print $2; exit
      }
    ' "$CONFIG_FILE"
  fi
}

HAS_FFMPEG=""
HAS_YTDLP=""
HAS_UV=""
command -v ffmpeg >/dev/null 2>&1 && HAS_FFMPEG="yes"
command -v yt-dlp >/dev/null 2>&1 && HAS_YTDLP="yes"
command -v uv >/dev/null 2>&1 && HAS_UV="yes"  # drives the local faster-whisper backend

SETUP_COMPLETE="$(read_key SETUP_COMPLETE)"

# Fully configured → silent (Claude can surface status on demand via --check).
# No API key is required: transcription runs fully local via faster-whisper.
if [[ "$SETUP_COMPLETE" == "true" && -n "$HAS_FFMPEG" && -n "$HAS_YTDLP" && -n "$HAS_UV" ]]; then
  exit 0
fi

# First-run / partially-configured → one-line hint.
if [[ -z "$HAS_FFMPEG" || -z "$HAS_YTDLP" || -z "$HAS_UV" ]]; then
  echo "/watch: needs ffmpeg + yt-dlp + uv. Run \`python3 \$CLAUDE_PLUGIN_ROOT/skills/watch/scripts/setup.py\` once to install and scaffold config."
else
  echo "/watch: ready. Transcription runs fully local (faster-whisper, no API key needed)."
fi
