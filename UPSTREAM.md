# Upstream ledger

Disposition of every item open on [bradautomates/claude-video](https://github.com/bradautomates/claude-video) when this fork was cut (2026-09-21). Statuses: **merged** (PR merged as authored), **adapted** (idea or parts re-implemented, author credited), **duplicate** (same fix, another PR taken), **deferred** (tracked as a fork issue), **declined**, **n/a**; for issues **fixed**, **docs**, **info**, **invalid**.

Everything taken is offered back upstream in [bradautomates/claude-video#237](https://github.com/bradautomates/claude-video/pull/237). Per-item detail: [CHANGELOG.md](CHANGELOG.md); contributor credits: [AUTHORS.md](AUTHORS.md).

## Pull requests (99 open at fork time)

| PR | Author | Title | Status | Landed as / reason |
|---|---|---|---|---|
| [#97](https://github.com/bradautomates/claude-video/pull/97) | @sainbayare-net | Fix efficient-detail crash on keyframe-less ranges; silence Windows perms warning | **merged** | frames part merged (keyframe-less range → uniform fallback); Windows-permission part via #206 |
| [#147](https://github.com/bradautomates/claude-video/pull/147) | @JMAL1988 | Use a sidecar .vtt as the transcript for local video files | **merged** | merged as authored (conflicts resolved) |
| [#154](https://github.com/bradautomates/claude-video/pull/154) | @thomaswillner | docs(skills): make watch description state when to invoke | **merged** | merged as authored (conflicts resolved) |
| [#175](https://github.com/bradautomates/claude-video/pull/175) | @Daily-AC | Compare dedup thumbnails in RGB, not grayscale | **merged** | merged as authored (conflicts resolved) |
| [#176](https://github.com/bradautomates/claude-video/pull/176) | @Daily-AC | Correct the image-token numbers in the /watch skill | **merged** | merged as authored (conflicts resolved) |
| [#182](https://github.com/bradautomates/claude-video/pull/182) | @eltokhy | Fix rolling-caption duplication and dropped cues in VTT parsing | **merged** | merged as authored (conflicts resolved) |
| [#192](https://github.com/bradautomates/claude-video/pull/192) | @IsaiahCalvo | Fix UnicodeEncodeError crashing the report on Windows | **merged** | merged as authored (conflicts resolved) |
| [#199](https://github.com/bradautomates/claude-video/pull/199) | @jayvee6 | Add optional local/self-hosted Whisper backend | **merged** | merged as authored (conflicts resolved) |
| [#206](https://github.com/bradautomates/claude-video/pull/206) | @Jordan-Zhu | Make Windows file-permission handling honest instead of a no-op | **merged** | merged as authored (conflicts resolved) |
| [#208](https://github.com/bradautomates/claude-video/pull/208) | @Jordan-Zhu | Match frame report lines regardless of path separator | **merged** | merged as authored (conflicts resolved) |
| [#210](https://github.com/bradautomates/claude-video/pull/210) | @Jordan-Zhu | Fall back to uniform when scene changes cluster, not just when they are few | **merged** | merged as authored (conflicts resolved) |
| [#214](https://github.com/bradautomates/claude-video/pull/214) | @nbkwabi | Fix: don't send one provider's API key to the other provider's endpoint | **merged** | merged as authored (conflicts resolved) |
| [#219](https://github.com/bradautomates/claude-video/pull/219) | @bagalutenGregor | fix(frames): ffmpeg 8+ compatibility — -vsync was removed | **merged** | merged as authored (conflicts resolved) |
| [#223](https://github.com/bradautomates/claude-video/pull/223) | @oheewono | Flag Whisper transcripts that look hallucinated | **merged** | merged as authored (conflicts resolved) |
| [#224](https://github.com/bradautomates/claude-video/pull/224) | @dsp407 | Fix uniform fallback reporting more frames selected than candidates | **merged** | merged as authored (conflicts resolved) |
| [#226](https://github.com/bradautomates/claude-video/pull/226) | @OpenClawLinda | Fix uniform extract() sampling the head instead of spreading across the range | **merged** | merged as authored (conflicts resolved) |
| [#227](https://github.com/bradautomates/claude-video/pull/227) | @OpenClawLinda | Warn when yt-dlp is stale enough to be the cause of a silent 403 | **merged** | merged as authored (conflicts resolved) |
| [#228](https://github.com/bradautomates/claude-video/pull/228) | @OpenClawLinda | Explain a 403, and keep the transcript when only the video stream fails | **merged** | merged as authored (conflicts resolved) |
| [#231](https://github.com/bradautomates/claude-video/pull/231) | @charles98601-sg | Isolate setup/download tests from host binaries | **merged** | merged as authored (conflicts resolved) |
| [#232](https://github.com/bradautomates/claude-video/pull/232) | @charles98601-sg | Read .env through a single parser | **merged** | merged as authored (conflicts resolved) |
| [#233](https://github.com/bradautomates/claude-video/pull/233) | @charles98601-sg | Upload Whisper chunks concurrently | **merged** | merged as authored (conflicts resolved) |
| [#236](https://github.com/bradautomates/claude-video/pull/236) | @gth-spec | docs(watch): flag transcript proper nouns as unverified | **merged** | merged as authored (conflicts resolved) |
| [#119](https://github.com/bradautomates/claude-video/pull/119) | @stickersfxlab | fix(windows): read .env in the encodings Windows actually writes | **adapted** | `decode_env_bytes` (UTF-16 ± BOM, UTF-8 BOM, ANSI) behind the single `.env` parser, with tests |
| [#127](https://github.com/bradautomates/claude-video/pull/127) | @caleb436 | Fix YouTube returning nothing: retry with the android player client (SABR) | **adapted** | `android` added to the media-fetch client fallbacks; SABR-specific handling not taken |
| [#136](https://github.com/bradautomates/claude-video/pull/136) | @gqbeerman | fix(setup): preflight must read cwd/.env like whisper.py does | **adapted** | setup preflight now uses the same config → `./.env` search order as whisper.py |
| [#179](https://github.com/bradautomates/claude-video/pull/179) | @pornthepp | watch: fall back to android client (format 18) on download 403 | **adapted** | `android` client fallback; forced format-18 (360p) downgrade not taken |
| [#184](https://github.com/bradautomates/claude-video/pull/184) | @endiaye677 | Add opt-in yt-dlp cookie support | **adapted** | `WATCH_COOKIES_FILE` / `WATCH_COOKIES_FROM_BROWSER`, via the shared parser, all yt-dlp calls |
| [#187](https://github.com/bradautomates/claude-video/pull/187) | @Rchardd | Add `--lang` so non-English videos get their native captions | **adapted** | `--lang` (single preferred language, overrides detection) |
| [#200](https://github.com/bradautomates/claude-video/pull/200) | @drsandeeprana00-bit | fix(youtube): survive datacenter-IP bot gate; add storyboard frame fallback | **adapted** | `player_client` retry for the media fetch only; storyboard fallback (Pillow) not taken |
| [#204](https://github.com/bradautomates/claude-video/pull/204) | @mrrobotbuilder | Fix frame extraction on ffmpeg 8.x, which removed -vsync | **adapted** | `OSError` guard folded into #219's probe; `_frame_lines` via #208 |
| [#205](https://github.com/bradautomates/claude-video/pull/205) | @Jordan-Zhu | Frame-coverage fixes and an on-device Whisper fallback | **adapted** | 2 of 7 commits cherry-picked (time-spaced selection, `WATCH_MAX_FPS`); `-frames:v` via #226, UTF-8 via #192; faster-whisper commits superseded by #199 |
| [#207](https://github.com/bradautomates/claude-video/pull/207) | @D0mD0mD0m | Add --force-whisper to bypass native captions | **adapted** | `--force-whisper` as written |
| [#212](https://github.com/bradautomates/claude-video/pull/212) | @AlexLenovo | Prefer the video's own caption language over English | **adapted** | `language` field → native-track fetch is the core of the caption logic |
| [#218](https://github.com/bradautomates/claude-video/pull/218) | @jvdurian-pixel | Fix quoted .env values with trailing comments and duplicate parsers | **adapted** | `_parse_value` + tests folded onto #232's parser |
| [#220](https://github.com/bradautomates/claude-video/pull/220) | @Rasmus257 | feat(download): Support for TikTok photo slideshows | **adapted** | audio-only crash fixed; gallery-dl slideshow support deferred → fork issue |
| [#221](https://github.com/bradautomates/claude-video/pull/221) | @oheewono | Prefer human-authored captions over auto-generated ones | **adapted** | human > auto ranking is the top rule of the new caption picker |
| [#225](https://github.com/bradautomates/claude-video/pull/225) | @OpenClawLinda | Fix YouTube rolling auto-captions shipping every line twice | **adapted** | #182 taken as the dedupe implementation; identical result on a real track |
| [#234](https://github.com/bradautomates/claude-video/pull/234) | @charles98601-sg | Fall back to the video's own caption language | **adapted** | combined with #212/#221 into the caption-language implementation |
| [#100](https://github.com/bradautomates/claude-video/pull/100) | @fabio-pisoni-hw | fix(setup): skip POSIX permission warning on Windows | **duplicate** | Windows permission check; #206 taken |
| [#102](https://github.com/bradautomates/claude-video/pull/102) | @PollxTroy-create | Fix frame extraction on ffmpeg builds without -vsync | **duplicate** | same `-vsync` fix; #219 taken |
| [#103](https://github.com/bradautomates/claude-video/pull/103) | @varunsahni18 | fix: replace deprecated -vsync with -fps_mode in ffmpeg frame extraction | **duplicate** | same `-vsync` fix; #219 taken |
| [#104](https://github.com/bradautomates/claude-video/pull/104) | @weekly100million | Fix UnicodeDecodeError on Windows / non-UTF-8 locales (cp949) | **duplicate** | UnicodeDecodeError on subprocess output — fixed in 0.3.0 by UTF-8 decoding at every call site (#108) |
| [#105](https://github.com/bradautomates/claude-video/pull/105) | @Maktorin | Collapse partial overlap in roll-up captions | **duplicate** | rolling-caption dedupe; #182 taken |
| [#106](https://github.com/bradautomates/claude-video/pull/106) | @greekr4 | Isolate tests from the developer's real config and env (#96) | **duplicate** | test isolation; #231 taken |
| [#110](https://github.com/bradautomates/claude-video/pull/110) | @redonto-007 | Fix UnicodeEncodeError on Windows: force UTF-8 stdout/stderr | **duplicate** | Windows console encoding; #192 taken |
| [#112](https://github.com/bradautomates/claude-video/pull/112) | @Diterex | Add a local whisper.cpp backend for offline transcription | **duplicate** | local transcription; superseded by #199's dependency-free OpenAI-compatible backend |
| [#113](https://github.com/bradautomates/claude-video/pull/113) | @tomimoyano15-byte | Captions: fetch the video's own language track, not English only | **duplicate** | native-language captions; landed via #212/#221/#234/#187 |
| [#114](https://github.com/bradautomates/claude-video/pull/114) | @nacho-marin | Fix caption downloads for source languages | **duplicate** | native-language captions; landed via #212/#221/#234/#187 |
| [#116](https://github.com/bradautomates/claude-video/pull/116) | @ai-websites-poland | Fix: non-English videos silently yield no transcript | **duplicate** | native-language captions; landed via #212/#221/#234/#187 |
| [#118](https://github.com/bradautomates/claude-video/pull/118) | @stickersfxlab | fix(windows): permission warning, console encoding, and two test-suite bugs | **duplicate** | permissions → #206, console encoding → #192, test fixes → #231/#208 |
| [#123](https://github.com/bradautomates/claude-video/pull/123) | @Nicopatron | Stop requesting YouTube's auto-translated caption tracks (HTTP 429) | **duplicate** | native-language captions; landed via #212/#221/#234/#187 |
| [#124](https://github.com/bradautomates/claude-video/pull/124) | @Ydiouri | Decode HTML entities in WebVTT captions | **duplicate** | HTML entity unescape is part of #182 |
| [#125](https://github.com/bradautomates/claude-video/pull/125) | @vgrosetti-maker | fix: ffmpeg 9 removed -vsync, use -fps_mode in frames.py | **duplicate** | same `-vsync` fix; #219 taken |
| [#129](https://github.com/bradautomates/claude-video/pull/129) | @drlee91 | Fix duplicate lines from YouTube's rolling auto-caption windows | **duplicate** | rolling-caption dedupe; #182 taken |
| [#130](https://github.com/bradautomates/claude-video/pull/130) | @victoropp | Fix two Windows/ffmpeg9 breakages in /watch (frame extraction + UnicodeEncodeError) | **duplicate** | `-vsync` → #219, console encoding → #192 |
| [#132](https://github.com/bradautomates/claude-video/pull/132) | @tiff4183 | Fix two Windows crashes: dropped -vsync flag, cp1252 console encoding | **duplicate** | `-vsync` → #219, console encoding → #192 |
| [#133](https://github.com/bradautomates/claude-video/pull/133) | @apalm8 | Fix frame extraction on ffmpeg 9.0+ (-vsync was removed) | **duplicate** | same `-vsync` fix; #219 taken |
| [#138](https://github.com/bradautomates/claude-video/pull/138) | @syrusdigital | Fix ffmpeg 9 incompatibility: probe for -fps_mode vs -vsync | **duplicate** | same `-vsync` fix; #219 taken |
| [#139](https://github.com/bradautomates/claude-video/pull/139) | @thetimlee1 | Fix ffmpeg 9 and Windows compatibility | **duplicate** | `-vsync` → #219, permissions → #206, test helper → #208 |
| [#140](https://github.com/bradautomates/claude-video/pull/140) | @xiaoqian289-foece | feat: add cookie support for login-walled sites (Douyin, Bilibili, etc.) | **duplicate** | cookie support; #184's smaller opt-in version taken |
| [#146](https://github.com/bradautomates/claude-video/pull/146) | @dungartoriaaa | Fix FFmpeg compatibility, Windows Unicode handling, and caption selection | **duplicate** | bundle: `-vsync`/encoding/captions all landed via the PRs above; CI added separately |
| [#148](https://github.com/bradautomates/claude-video/pull/148) | @utkarshbindal-wq | Fix frame extraction on ffmpeg 9 (-vsync removed) | **duplicate** | same `-vsync` fix; #219 taken |
| [#151](https://github.com/bradautomates/claude-video/pull/151) | @waxandwires | transcribe: dedupe YouTube rolling-caption cues at parse time | **duplicate** | rolling-caption dedupe; #182 taken |
| [#159](https://github.com/bradautomates/claude-video/pull/159) | @khcho98-maker | Fix Windows: UTF-8 stdio, platform-correct install and interpreter hints | **duplicate** | UTF-8 stdio → #192; Windows install-hint wording partly already in 0.1.2 |
| [#162](https://github.com/bradautomates/claude-video/pull/162) | @dustymurph | Fix frame extraction on ffmpeg 8+ where -vsync was removed | **duplicate** | same `-vsync` fix; #219 taken |
| [#164](https://github.com/bradautomates/claude-video/pull/164) | @djhammer20k | Fix: non-English videos return no captions at all | **duplicate** | native-language captions; landed via #212/#221/#234/#187 |
| [#166](https://github.com/bradautomates/claude-video/pull/166) | @vakogogu-coder | Fix frame extraction on ffmpeg 9.0 (-vsync removed) | **duplicate** | same `-vsync` fix; #219 taken |
| [#168](https://github.com/bradautomates/claude-video/pull/168) | @sauveteur71 | Fix /watch on Windows + ffmpeg 9.0 (-vsync removed) and cp1252 consoles | **duplicate** | `-vsync` → #219, console encoding → #192 |
| [#169](https://github.com/bradautomates/claude-video/pull/169) | @androsland | Add an on-device Whisper backend (faster-whisper) | **duplicate** | local transcription; superseded by #199's dependency-free OpenAI-compatible backend |
| [#171](https://github.com/bradautomates/claude-video/pull/171) | @ELpistolero21 | fix: use -fps_mode instead of deprecated -vsync for ffmpeg frame extraction | **duplicate** | same `-vsync` fix; #219 taken |
| [#172](https://github.com/bradautomates/claude-video/pull/172) | @ZiCoreDom | fix(frames): replace removed -vsync with -fps_mode for ffmpeg 8+ | **duplicate** | same `-vsync` fix; #219 taken |
| [#173](https://github.com/bradautomates/claude-video/pull/173) | @ZiCoreDom | fix: two Windows portability issues (spurious chmod warning, POSIX-only test helper) | **duplicate** | Windows permission check; #206 taken |
| [#177](https://github.com/bradautomates/claude-video/pull/177) | @SVSOnderwijs | Fix frame extraction on ffmpeg 8+ (-vsync was removed) | **duplicate** | same `-vsync` fix; #219 taken |
| [#181](https://github.com/bradautomates/claude-video/pull/181) | @MaCeeeee | Fix frame extraction failing on ffmpeg 9 (-vsync removed) | **duplicate** | same `-vsync` fix; #219 taken |
| [#183](https://github.com/bradautomates/claude-video/pull/183) | @endiaye677 | Fix frame extraction on ffmpeg 9 (-vsync removed) | **duplicate** | same `-vsync` fix; #219 taken |
| [#185](https://github.com/bradautomates/claude-video/pull/185) | @vanlieropf-dot | Fix: skip POSIX permission check on Windows (false positive) | **duplicate** | Windows permission check; #206 taken |
| [#186](https://github.com/bradautomates/claude-video/pull/186) | @Rchardd | Replace removed `-vsync` with `-fps_mode` for ffmpeg 8+ | **duplicate** | same `-vsync` fix; #219 taken |
| [#188](https://github.com/bradautomates/claude-video/pull/188) | @amipcoaching2027 | Fix ffmpeg frame extraction: replace deprecated -vsync with -fps_mode | **duplicate** | same `-vsync` fix; #219 taken |
| [#190](https://github.com/bradautomates/claude-video/pull/190) | @ppradeep123-ops | Skip .env permission check on Windows Git Bash/Cygwin (noacl) | **duplicate** | Windows permission check; #206 taken |
| [#191](https://github.com/bradautomates/claude-video/pull/191) | @Rchardd | Fix two Windows-only test failures (permissions warning, path separator) | **duplicate** | Windows permission check; #206 taken |
| [#194](https://github.com/bradautomates/claude-video/pull/194) | @aromat24 | Windows: -fps_mode for ffmpeg 9, skip the POSIX-mode check on noacl mounts, test helper accepts native paths | **duplicate** | `-vsync` → #219, permissions → #206, path helper → #208 |
| [#197](https://github.com/bradautomates/claude-video/pull/197) | @crybbyforreal | Fix frame extraction on ffmpeg 8+: replace removed -vsync with -fps_mode | **duplicate** | same `-vsync` fix; #219 taken |
| [#198](https://github.com/bradautomates/claude-video/pull/198) | @dd58mk72wv-stack | Use -fps_mode where ffmpeg no longer accepts -vsync | **duplicate** | same `-vsync` fix; #219 taken |
| [#201](https://github.com/bradautomates/claude-video/pull/201) | @frankkeil | fix(hooks): skip .env permission check on Windows | **duplicate** | Windows permission check; #206 taken |
| [#202](https://github.com/bradautomates/claude-video/pull/202) | @tradersc2020-oss | Fix frame extraction on ffmpeg 9: replace removed -vsync with -fps_mode | **duplicate** | same `-vsync` fix; #219 taken |
| [#203](https://github.com/bradautomates/claude-video/pull/203) | @tradersc2020-oss | Fix UnicodeEncodeError on Windows cp1252 consoles by forcing UTF-8 stdio | **duplicate** | Windows console encoding; #192 taken |
| [#211](https://github.com/bradautomates/claude-video/pull/211) | @SatishGs01 | Fix frame extraction on ffmpeg 9.x (removed -vsync flag) | **duplicate** | same `-vsync` fix; #219 taken |
| [#213](https://github.com/bradautomates/claude-video/pull/213) | @blkzera | Fix frame extraction on ffmpeg 8+ where -vsync was removed | **duplicate** | same `-vsync` fix; #219 taken |
| [#216](https://github.com/bradautomates/claude-video/pull/216) | @oheewono | Use -fps_mode instead of the removed -vsync flag | **duplicate** | same `-vsync` fix; #219 taken |
| [#217](https://github.com/bradautomates/claude-video/pull/217) | @oheewono | Don't crash on non-UTF-8 Windows consoles | **duplicate** | Windows console encoding; #192 taken |
| [#230](https://github.com/bradautomates/claude-video/pull/230) | @rainervianaprocurador | Fix frame extraction on ffmpeg 8+ by probing for -fps_mode | **duplicate** | same `-vsync` fix; #219 taken |
| [#115](https://github.com/bradautomates/claude-video/pull/115) | @yejoopapa | Restore YouTube caption retrieval past the PO-token gate | **deferred** | forces `player_client=default,tv,web_safari` on every caption fetch; needs evidence it helps beyond the media retry |
| [#152](https://github.com/bradautomates/claude-video/pull/152) | @binyangzhu000-sudo | feat: add Atlas Cloud ASR transcription backend | **deferred** | Atlas Cloud ASR vendor service; same |
| [#193](https://github.com/bradautomates/claude-video/pull/193) | @Anil-matcha | feat: add MuAPI Whisper fallback | **deferred** | MuAPI vendor service; same |
| [#196](https://github.com/bradautomates/claude-video/pull/196) | @AureliaRen | Add DashScope (Alibaba Bailian) transcription backend for networks where Groq/OpenAI are unreachable | **deferred** | DashScope SDK dependency; generic local backend covers self-hosting |
| [#215](https://github.com/bradautomates/claude-video/pull/215) | @Verohomie | Make /watch usable on screen recordings: fix macOS downloads, lift the 720p cap, size frames to the source | **deferred** | 2k-line bundle (quality dial, screen detection, manifest); yt-dlp capability part covered by preflight warnings; fork issue |
| [#235](https://github.com/bradautomates/claude-video/pull/235) | @charles98601-sg | Cache downloads so a repeat run skips the fetch | **deferred** | download cache — design question (growth/cleanup); fork issue |
| [#142](https://github.com/bradautomates/claude-video/pull/142) | @dluxcru | Replace paid Whisper API fallback with local-first transcript resolution (youtube-data MCP + Voicebox) | **declined** | removes the Whisper API path (−946 lines) in favour of an MCP + Voicebox dependency |
| [#158](https://github.com/bradautomates/claude-video/pull/158) | @OctoBored | Fix broken star history chart | **n/a** | upstream README star-history chart; the fork's README differs |

## Issues (41 open at fork time)

| Issue | Author | Title | Status | Resolution |
|---|---|---|---|---|
| [#37](https://github.com/bradautomates/claude-video/issues/37) | @Opus721 | Sub-second / fast-action clips: 2 fps hard cap yields 1–2 frames, no all-frames mode | **fixed** | `WATCH_MAX_FPS` |
| [#47](https://github.com/bradautomates/claude-video/issues/47) | @willrenzo39 | SessionStart permission check false-positives on Windows when bash isn't Git-Bash/MSYS | **fixed** | Windows permission check skipped (PR #206) |
| [#48](https://github.com/bradautomates/claude-video/issues/48) | @max1andyco | Browser Session Cookie needed | **fixed** | opt-in cookies (from #184) |
| [#51](https://github.com/bradautomates/claude-video/issues/51) | @smuchow1962 | watch.py crashes on Windows when a video title contains emoji/non-cp1252 chars | **fixed** | UTF-8 stdout (PR #192) |
| [#67](https://github.com/bradautomates/claude-video/issues/67) | @cwinvestments | Two Windows findings: setup.py --check passes but YouTube fails without deno/curl_cffi, and focused mode crashes on cp1252 consoles | **fixed** | preflight warnings for no impersonation / no JS runtime; `WATCH_YTDLP`; 403 message names causes |
| [#80](https://github.com/bradautomates/claude-video/issues/80) | @Lukeini | watch.py wipes a pre-existing source file when it lives inside --out-dir (breaks the documented local-file re-run flow) | **fixed** | temp vs user-supplied work dir; SKILL.md forbids deleting user dirs |
| [#93](https://github.com/bradautomates/claude-video/issues/93) | @amobe464 | macOS: frames never extract — brew's yt-dlp has no impersonation targets (403 on video, captions still work) | **fixed** | preflight warnings for no impersonation / no JS runtime; `WATCH_YTDLP`; 403 message names causes |
| [#96](https://github.com/bradautomates/claude-video/issues/96) | @jacobroberts0102-collab | Test suite cannot pass on a developer machine: real user config leaks in, and 3 tests are POSIX-only | **fixed** | test isolation (PRs #231 #208) |
| [#99](https://github.com/bradautomates/claude-video/issues/99) | @spikefcz | frames.py: `-vsync` was removed from modern ffmpeg — both frame paths fail, zero frames extracted | **fixed** | ffmpeg `-fps_mode` probe (PR #219) |
| [#101](https://github.com/bradautomates/claude-video/issues/101) | @PollxTroy-create | ffmpeg frame extraction fails on builds without -vsync ("Unrecognized option 'vsync'") | **fixed** | ffmpeg `-fps_mode` probe (PR #219) |
| [#107](https://github.com/bradautomates/claude-video/issues/107) | @Elenics | Windows: secrets permission check always warns (st_mode does not reflect NTFS ACLs) | **fixed** | Windows permission check skipped (PR #206) |
| [#108](https://github.com/bradautomates/claude-video/issues/108) | @SamuelCarvajal210408 | Windows: UnicodeDecodeError in subprocess output breaks /watch on any non-ASCII ffmpeg output | **fixed** | UTF-8 subprocess decoding everywhere |
| [#109](https://github.com/bradautomates/claude-video/issues/109) | @redonto-007 | UnicodeEncodeError on Windows: report output crashes on cp1252 console (→ arrow) | **fixed** | UTF-8 stdout (PR #192) |
| [#117](https://github.com/bradautomates/claude-video/issues/117) | @skankykiwi | ffmpeg 9.0 removes -vsync, breaking scene/keyframe extraction (Unrecognized option 'vsync') | **fixed** | ffmpeg `-fps_mode` probe (PR #219) |
| [#120](https://github.com/bradautomates/claude-video/issues/120) | @Jochen-1979 | check-setup.sh: SessionStart hook aborts when .env exists but is unreadable (read_key + set -e) | **fixed** | hook checks `-r`, tolerates awk failure |
| [#122](https://github.com/bradautomates/claude-video/issues/122) | @Captain-Tayternuts | Frame extraction fails on ffmpeg 8.0+ (-vsync was removed) | **fixed** | ffmpeg `-fps_mode` probe (PR #219) |
| [#126](https://github.com/bradautomates/claude-video/issues/126) | @MidiService-stack | frames.py: -vsync removed in FFmpeg 9 - use -fps_mode | **fixed** | ffmpeg `-fps_mode` probe (PR #219) |
| [#128](https://github.com/bradautomates/claude-video/issues/128) | @vgrosetti-maker | Windows: /watch aborts when App Control blocks ffprobe.exe but allows ffmpeg.exe | **fixed** | `ffmpeg -i` metadata fallback when ffprobe is blocked |
| [#134](https://github.com/bradautomates/claude-video/issues/134) | @noameoscar29-ship-it | Two Windows blockers in v0.2.0: `-vsync` removed in ffmpeg 9, and cp1252 crash on non-latin1 titles | **fixed** | `-vsync` (#219) + cp1252 (#192) |
| [#137](https://github.com/bradautomates/claude-video/issues/137) | @gqbeerman | Local whisper.cpp as a keyless fallback — plus a punctuation gotcha worth documenting either way | **fixed** | self-hosted OpenAI-compatible backend (PR #199) |
| [#141](https://github.com/bradautomates/claude-video/issues/141) | @chpark-ML | Frame extraction fails on ffmpeg >= 8: -vsync was removed (-fps_mode fixes it) | **fixed** | ffmpeg `-fps_mode` probe (PR #219) |
| [#143](https://github.com/bradautomates/claude-video/issues/143) | @Vinyo2112 | Frame extraction broken on ffmpeg 8+: '-vsync' was removed (use -fps_mode) | **fixed** | ffmpeg `-fps_mode` probe (PR #219) |
| [#144](https://github.com/bradautomates/claude-video/issues/144) | @seonghun8476-sudo | Non-English videos: hardcoded --sub-langs "en.*" fetches the auto-translated track (429s, and worse transcripts when it succeeds) | **fixed** | native-language captions (from #212 #221 #234 #187) |
| [#149](https://github.com/bradautomates/claude-video/issues/149) | @LittlePirate58 | ffmpeg 'vsync' option removed in ffmpeg 9.x breaks frame extraction | **fixed** | ffmpeg `-fps_mode` probe (PR #219) |
| [#150](https://github.com/bradautomates/claude-video/issues/150) | @mkgaskin-ops | UnicodeEncodeError on Windows: report printing crashes on cp1252 in focused mode | **fixed** | UTF-8 stdout (PR #192) |
| [#153](https://github.com/bradautomates/claude-video/issues/153) | @miguexis | Captions are hardcoded to English (--sub-langs "en.*"), so non-English videos return machine-translated transcripts | **fixed** | native-language captions (from #212 #221 #234 #187) |
| [#156](https://github.com/bradautomates/claude-video/issues/156) | @realsalestalk | YouTube media 403 on Windows despite deno + curl_cffi present: only player_client=mweb works, but mweb alone kills subtitles | **fixed** | media retry via `mweb`/`tv`/`web_embedded`/`android` (from #200 #127 #179) + preflight warnings |
| [#161](https://github.com/bradautomates/claude-video/issues/161) | @badwookie208 | Frame extraction fails on ffmpeg 9.0: -vsync was removed (use -fps_mode) | **fixed** | ffmpeg `-fps_mode` probe (PR #219) |
| [#163](https://github.com/bradautomates/claude-video/issues/163) | @dmitrykarabutov | Frame extraction fails on ffmpeg 7+: `-vsync` was removed in favour of `-fps_mode` | **fixed** | ffmpeg `-fps_mode` probe (PR #219) |
| [#174](https://github.com/bradautomates/claude-video/issues/174) | @KenHROI | Frame extraction fails on ffmpeg 9: '-vsync vfr' no longer recognized | **fixed** | ffmpeg `-fps_mode` probe (PR #219) |
| [#178](https://github.com/bradautomates/claude-video/issues/178) | @Jordan-Zhu | --fps override silently truncates frame coverage to the head of the video | **fixed** | `--fps` spread (PR #226) |
| [#180](https://github.com/bradautomates/claude-video/issues/180) | @Beatkim96 | Frame extraction fails on FFmpeg 9: `-vsync` was removed | **fixed** | ffmpeg `-fps_mode` probe (PR #219) |
| [#189](https://github.com/bradautomates/claude-video/issues/189) | @ppradeep123-ops | check-setup.sh: false "permissions 644 (should be 600)" warning on Windows Git Bash (includes fix) | **fixed** | Windows permission check skipped (PR #206) |
| [#195](https://github.com/bradautomates/claude-video/issues/195) | @cheeann18 | ffmpeg 9.x: -vsync removed, breaks scene-aware frame extraction | **fixed** | ffmpeg `-fps_mode` probe (PR #219) |
| [#222](https://github.com/bradautomates/claude-video/issues/222) | @oheewono | Whisper invents dialogue on videos with no speech, and it is presented as a normal transcript | **fixed** | Whisper hallucination flag (PR #223) |
| [#229](https://github.com/bradautomates/claude-video/issues/229) | @SheepsSuck | frames.py fails on newer ffmpeg: removed -vsync option (use -fps_mode instead) | **fixed** | ffmpeg `-fps_mode` probe (PR #219) |
| [#83](https://github.com/bradautomates/claude-video/issues/83) | @shivamg194-lab | Youtube is not reachable | **docs** | sandbox egress allowlist — environmental; documented in README Limits and SKILL.md |
| [#135](https://github.com/bradautomates/claude-video/issues/135) | @CatsLurkerer | /watch` fails on YouTube — egress block | **docs** | sandbox egress allowlist — environmental; documented in README Limits and SKILL.md |
| [#31](https://github.com/bradautomates/claude-video/issues/31) | @pheistman | Fork notification: pheistman/claude-watch — multi-provider transcription + transcript-first pipeline | **info** | fork notification (pheistman/claude-watch) — no action |
| [#54](https://github.com/bradautomates/claude-video/issues/54) | @Artemonim | AskVLM | **info** | pointer to AskVLM — no action |
| [#98](https://github.com/bradautomates/claude-video/issues/98) | @ekaterinakozina3-commits | Мои | **invalid** | empty issue body |

## Keeping this current

`.github/workflows/upstream-watch.yml` runs weekly, lists upstream PRs, issues and commits with activity in the past 8 days, and opens a tracking issue here. Triaged items get a row above.
