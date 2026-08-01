---
description: Watch a video or audio recording. Uses scene-aware frames, motion and sound evidence, optional semantic sound understanding, captions or Whisper, and focused multi-pass analysis.
argument-hint: <video-or-audio-url-or-path> [question]
allowed-tools: [Bash, Read, AskUserQuestion, Task]
---

Invoke the `watch` skill (defined in SKILL.md) with the user's arguments: $ARGUMENTS

Follow the skill's full pipeline: preflight setup check → download/cache → extract coverage-preserving scene-aware frames and short-form cinematic evidence → pull and clean captions or use Whisper → add semantic sound analysis when the user asks for sound identification/design → use source chapters when available → Read each visual artifact → answer grounded in the aligned visual, sound, and transcript timeline. Use the focused or deep multi-pass protocol when needed. If the user provided no arguments, ask for a video/audio URL or local path.
