#!/usr/bin/env bash
# Extract mono 16kHz mp3 audio from a video/audio-only download, for the
# Voicebox transcription fallback (see SKILL.md "Transcript resolution").
# Usage: extract-audio.sh <video-path> <audio-out.mp3>
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$SCRIPT_DIR/whisper.py" "$1" "$2"
