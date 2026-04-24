---
description: Watch a video (URL or local path). Downloads with yt-dlp, extracts frames with ffmpeg, transcribes from captions or Whisper, and answers questions about what's in the video.
argument-hint: <video-url-or-path> [question]
allowed-tools: [Bash, Read, AskUserQuestion]
---

Invoke the `watch` skill (defined in SKILL.md) with the user's arguments: $ARGUMENTS

Follow the skill's full pipeline: preflight setup check → download via yt-dlp → extract frames at auto-scaled fps → pull captions or Whisper transcript → Read each frame → answer the user grounded in frames and transcript. If the user provided no arguments, ask them for a video URL or local path before proceeding.
