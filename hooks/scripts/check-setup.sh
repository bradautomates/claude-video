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

# Report only WHETHER a key is set — never read its value into a variable.
# The single question asked here is "is one configured", and answering it with
# the secret itself puts it in the shell's memory and in awk's output for no gain.
have_key() {
  local name="$1"
  if [[ -n "${!name:-}" ]]; then
    echo "yes"
    return
  fi
  if [[ -f "$CONFIG_FILE" ]]; then
    awk -F= -v k="$name" '
      /^[[:space:]]*#/ { next }
      $1 == k {
        sub(/^[[:space:]]*/, "", $2); sub(/[[:space:]]*$/, "", $2);
        gsub(/^["'\'']|["'\'']$/, "", $2);
        if ($2 != "") print "yes";
        exit
      }
    ' "$CONFIG_FILE"
  fi
}

# SETUP_COMPLETE is a marker, not a secret, so its value is read directly.
read_flag() {
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

# find_spec rather than a real import: this runs at every SessionStart, and
# importing faster-whisper pulls in CTranslate2 and its CUDA bindings. The
# trade-off is that find_spec answers "the module is on the path", not "it
# imports cleanly" — a broken install reads as present here. That is fine for
# a one-line status hint; setup.py --check and watch.py both do a real import.
has_local_whisper() {
  command -v python3 >/dev/null 2>&1 || return 1
  python3 -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('faster_whisper') else 1)" 2>/dev/null
}

HAS_FFMPEG=""
HAS_YTDLP=""
command -v ffmpeg >/dev/null 2>&1 && HAS_FFMPEG="yes"
command -v yt-dlp >/dev/null 2>&1 && HAS_YTDLP="yes"

HAS_GROQ="$(have_key GROQ_API_KEY)"
HAS_OPENAI="$(have_key OPENAI_API_KEY)"
SETUP_COMPLETE="$(read_flag SETUP_COMPLETE)"

# Fully configured → silent (Claude can surface status on demand via --check).
if [[ "$SETUP_COMPLETE" == "true" && -n "$HAS_FFMPEG" && -n "$HAS_YTDLP" ]]; then
  exit 0
fi

# First-run / partially-configured → one-line hint.
if [[ -z "$HAS_FFMPEG" || -z "$HAS_YTDLP" ]]; then
  echo "/watch: needs ffmpeg + yt-dlp. Run \`python3 \$CLAUDE_PLUGIN_ROOT/skills/watch/scripts/setup.py\` once to install and scaffold config."
elif [[ -n "$HAS_GROQ" || -n "$HAS_OPENAI" ]]; then
  echo "/watch: ready."
elif has_local_whisper; then
  echo "/watch: ready — transcription runs on this machine via faster-whisper, no API key needed."
else
  echo "/watch: ready for videos with native captions. For the rest, either \`pip install \"faster-whisper>=1.0\"\` (runs locally, no key) or add GROQ_API_KEY / OPENAI_API_KEY to ~/.config/watch/.env."
fi
