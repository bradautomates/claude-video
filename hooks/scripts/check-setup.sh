#!/usr/bin/env bash
# SessionStart hook for /watch — one-line status so users know what's wired up.
# Silent on ready state to avoid spam. Points at the installer when something
# is missing.
set -euo pipefail

CONFIG_FILE="$HOME/.config/watch/.env"

# Warn if the secrets file has loose permissions.
if [[ -f "$CONFIG_FILE" ]]; then
  case "$(uname -s 2>/dev/null)" in
    MINGW*|MSYS*|CYGWIN*)
      # NTFS: Git Bash stat bits are synthetic (non-executable files always
      # report 644), so a chmod-600 check can never pass here. Inspect the
      # real ACL instead and warn only when broad-access principals hold an
      # entry. SIDs keep this locale-independent (group names are localized
      # on non-English Windows).
      if command -v icacls >/dev/null 2>&1 && command -v cygpath >/dev/null 2>&1; then
        win_path=$(cygpath -w "$CONFIG_FILE")
        broad_sid=""
        # S-1-1-0 Everyone · S-1-5-32-545 BUILTIN\Users · S-1-5-11 Authenticated Users
        for sid in S-1-1-0 S-1-5-32-545 S-1-5-11; do
          if MSYS_NO_PATHCONV=1 icacls "$win_path" /findsid "*$sid" 2>/dev/null | grep -qiF "$win_path"; then
            broad_sid="$sid"
            break
          fi
        done
        if [[ -n "$broad_sid" ]]; then
          echo "/watch: WARNING — $CONFIG_FILE ACL grants broad access (SID $broad_sid)."
          echo "  Fix (Git Bash): MSYS_NO_PATHCONV=1 icacls \"\$(cygpath -w $CONFIG_FILE)\" /inheritance:r /grant:r \"\$USERNAME:(R,W)\""
        fi
      fi
      ;;
    *)
      perms=$(stat -c '%a' "$CONFIG_FILE" 2>/dev/null || stat -f '%Lp' "$CONFIG_FILE" 2>/dev/null || echo "")
      if [[ -n "$perms" && "$perms" != "600" && "$perms" != "400" ]]; then
        echo "/watch: WARNING — $CONFIG_FILE has permissions $perms (should be 600)."
        echo "  Fix: chmod 600 $CONFIG_FILE"
      fi
      ;;
  esac
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
command -v ffmpeg >/dev/null 2>&1 && HAS_FFMPEG="yes"
command -v yt-dlp >/dev/null 2>&1 && HAS_YTDLP="yes"

HAS_GROQ="$(read_key GROQ_API_KEY)"
HAS_OPENAI="$(read_key OPENAI_API_KEY)"
SETUP_COMPLETE="$(read_key SETUP_COMPLETE)"

# Fully configured → silent (Claude can surface status on demand via --check).
if [[ "$SETUP_COMPLETE" == "true" && -n "$HAS_FFMPEG" && -n "$HAS_YTDLP" ]]; then
  exit 0
fi

# First-run / partially-configured → one-line hint.
if [[ -z "$HAS_FFMPEG" || -z "$HAS_YTDLP" ]]; then
  echo "/watch: needs ffmpeg + yt-dlp. Run \`python3 \$CLAUDE_PLUGIN_ROOT/scripts/setup.py\` once to install and scaffold config."
elif [[ -z "$HAS_GROQ" && -z "$HAS_OPENAI" ]]; then
  echo "/watch: ready for videos with native captions. Add GROQ_API_KEY (preferred) or OPENAI_API_KEY to ~/.config/watch/.env to unlock Whisper fallback."
else
  echo "/watch: ready."
fi
