[English](README.md) | **简体中文** | [日本語](README.ja.md)

# /watch

**让 Claude 能够观看任何视频。**

Claude Code（推荐——通过市场自动更新）：
```
/plugin marketplace add bradautomates/claude-video
/plugin install watch@claude-video
```

Codex、Cursor、Copilot、Gemini CLI，或其他 50 多种 [Agent Skills](https://agentskills.io) 宿主：
```bash
npx skills add bradautomates/claude-video -g
```
（`-g` 会为当前用户全局安装，使其可用于所有项目。去掉该参数即可限定在单个项目内。）

更多安装方式（claude.ai 网页版、手动安装）请参阅下方的[安装](#安装)部分。

无需配置即可开始——首次运行时，`yt-dlp` 和 `ffmpeg` 会通过 macOS 上的 `brew` 自动安装（Linux/Windows 会输出准确的安装命令）。大多数公开视频都能通过字幕免费处理。仅当视频没有字幕时才需要 Whisper API 密钥。

---

Claude 可以读取网页、运行脚本、浏览仓库。但它开箱即用时做不到的是*观看视频*。你粘贴一个 YouTube 链接，它只能根据标题猜测，或者提取一份遗漏了 90% 画面信息的转录文本。

借助 Claude Video `/watch`，你可以粘贴 URL 或本地路径并提出问题。Claude 会先获取字幕，只下载所需内容，提取帧（场景感知；或在 `efficient` 细节模式下使用快速关键帧），生成带时间戳的转录文本（有字幕时免费使用字幕，否则回退到 Whisper API），再将每一帧作为图片交给 `Read`。等它回答时，它已经*看过*视频，也已经*听过*音频。

```
/watch https://youtu.be/dQw4w9WgXcQ what happens at the 30 second mark?
```

## 人们实际如何使用它

**分析他人的内容。** `/watch https://youtu.be/<viral-video> what hook did they open with?` Claude 会查看最开始的几帧，读取开场转录文本，并拆解内容结构。广告创意、竞争对手发布会、播客开场，以及任何*如何呈现*和*呈现什么*同样重要的内容都适用。

**通过视频诊断缺陷。** 有人给你发来一段故障录屏。`/watch bug-repro.mov what's going wrong?` Claude 会观看录屏，找出问题出现的画面，描述屏幕上的内容，而且通常无需你亲自打开文件就能发现原因。

**总结视频。** `/watch https://youtu.be/<long-thing> summarize this` 会完成显而易见的工作——提取结构、关键时刻，以及实际说了什么、展示了什么。比用 2 倍速观看更快。

**去掉更新视频中的夸大宣传。** `/watch https://youtu.be/<launch-video> what's actually new — skip the hype` 会把某个被称为“颠覆性”的功能发布压缩为真正重要的几项内容，让你无需忍受十分钟的开场和过度推销就能获得实质信息。

**把播放列表转换成笔记。** `/watch https://youtu.be/<video> summarize this to a note` 对系列视频逐个运行，并为每个视频保存一份摘要，这样频道或课程就会变成可搜索的笔记集，而不是你必须花数小时看完的内容。

## 工作原理

1. **你粘贴视频并提出问题。** 可以是 URL（yt-dlp 支持的任何来源——YouTube、Loom、TikTok、X、Instagram，以及其他数百个平台）或本地路径（`.mp4`、`.mov`、`.mkv`、`.webm`）。
2. **`yt-dlp` 首先检查字幕。** 在 `transcript` 细节模式下，有字幕的 URL 无需下载视频即可返回。其他模式下，或 Whisper 需要音频时，只下载本次运行需要的内容。
3. **`ffmpeg` 按所选细节模式提取帧。** `efficient` 仅解码关键帧（几乎瞬间完成）；`balanced`/`token-burner` 优先使用场景变化帧，若数量不足则回退到依据时长调整的均匀采样器。JPEG 默认宽 512 像素，并将高度限制在 1998 像素以内，以兼容 Claude Read。
4. **转录文本来自两个来源之一。** 首选方式：`yt-dlp` 从源视频中提取原生字幕（人工或自动生成）。免费、即时，准确度尚可。回退方式：提取单声道 16 kHz、64 kbps 的 mp3 音频片段（约 480 kB/分钟）并发送给 Whisper——Groq 的 `whisper-large-v3`（首选——更便宜、更快）或 OpenAI 的 `whisper-1`。
5. **帧和转录文本会交给 Claude。** 脚本输出带有 `t=MM:SS` 标记的帧路径，以及带时间戳的转录文本。Claude 会并行 `Read` 每一帧——JPEG 会直接作为图片呈现在它的上下文中。
6. **Claude 根据屏幕和音频中的实际内容作答。** 不是“根据描述”，也不是“根据标题”。它看过这些帧，听过转录文本，并像真正看过视频的人一样回答。
7. **清理。** 脚本会在最后输出工作目录。如果你不再追问，Claude 会将其删除。

## 帧预算——为何重要

令牌成本主要由帧决定。每一帧都是图片；图片令牌会迅速累积。脚本的自动 fps 逻辑可避免你把上下文预算浪费在对 30 分钟视频的稀疏扫描上，因为这种问题通常更适合聚焦于某个 30 秒片段。

| 时长 | 默认帧预算 | 效果 |
|----------|---------------------|--------------|
| ≤30 秒 | 约 30 帧 | 密集——基本覆盖每个关键时刻 |
| 30 秒 - 1 分钟 | 约 40 帧 | 仍然密集 |
| 1 - 3 分钟 | 约 60 帧 | 从容覆盖 |
| 3 - 10 分钟 | 约 80 帧 | 稀疏但可用 |
| > 10 分钟 | 100 帧（有限额模式） | 发出“稀疏扫描”警告——聚焦后重新运行，或使用 `--detail token-burner` 获得完整的无限额覆盖 |

当用户指出某个时刻（“2:30 左右”“最后 30 秒”“从 0:45 到 1:00”）时，请传入 `--start` / `--end`。聚焦模式会获得更密集的每秒预算，上限为 2 fps，比对整个视频进行稀疏扫描有用得多。

## 帧去重

帧选择——关键帧（`efficient`）、场景变化检测（`balanced`/`token-burner`），或两者在不足时回退到的均匀采样器——仍可能产生近乎相同的帧：一段屏幕录制可能在同一张幻灯片上停留 90 秒，结果产生十几帧，而每一帧都会按独立图片计费。在帧交给 Claude 之前，去重过程会将这些帧丢弃。该过程默认在每种帧模式下运行（使用 `--no-dedup` 可关闭）：

1. 调用一次 `ffmpeg`，将每张已提取的 JPEG 缩放为 16×16 灰度缩略图。此后的所有处理都使用 Python 标准库——无需图片库。
2. 对每一帧，计算它与*上一个保留帧*之间的**平均绝对差值**（每像素亮度变化的平均值，范围为 0–255）。
3. 如果差值小于或等于阈值（`2.0`），该帧会被视为近似重复并丢弃；否则予以保留，并成为新的参照帧。
4. 帧预算上限在去重*之后*应用，因此预算会花在不同的帧上。

与上一个*保留的*帧比较（而不是与前一帧比较）可以捕捉那些逐帧变化始终达不到阈值的缓慢淡入淡出。阈值被刻意设得很低，并且衡量绝对亮度而非结构，因此一行代码差异、终端滚动一行，或两张颜色不同的纯色幻灯片都能保留下来。

**Frames** 行会报告折叠情况，例如 `6 selected from 14 candidates (… 8 near-duplicates dropped …)`。对于始终运动的画面，不会丢弃任何内容，成本也与原来相同。

## 细节模式——实测数据

`--detail` 旋钮在速度、令牌成本与视觉保真度之间进行取舍。下表数字来自对一段真实的 **49:08** YouTube 视频（1280×720、英文自动字幕）的测试——这是一段很长、画面大多静止的屏幕录制，最能考验帧上限。提取时间是在本地 CPU 上针对预先下载的副本测得；一次性的下载耗时约 **37 秒** / 76 MB，三种帧模式共用这份副本。

| 模式 | 引擎 | 帧数 | 上限 | 提取时间 | 时间覆盖 | 预计图片令牌 |
|------|--------|--------|-----|-----------------|-------------------|-------------------|
| `transcript` | 无（字幕） | 0 | — | **约 4.5 秒**（一次 yt-dlp 调用，无下载） | 完整（文本） | 0（约 26.6k 文本令牌） |
| `efficient` | 关键帧（`-skip_frame nokey`） | 50 | 50 | **约 0.5 秒** | 0:00 → 49:04（完整） | **约 9.8k** |
| `balanced` | 场景变化 | 100 | 100 | **约 20.9 秒** | 0:00 → 48:38（完整） | **约 19.7k** |
| `token-burner` | 场景变化 | 116 | 无上限 | **约 21.0 秒** | 0:00 → 48:38（完整） | **约 22.8k** |

- **图片令牌**采用 Anthropic 的 `(width × height) / 750`——在默认 512 像素宽度下，这些 720p 帧为 512×288，约 **197 令牌/帧**；使用 `--resolution 1024` 时约为其 4 倍。每种有字幕的模式都会提供转录文本；对于长视频，文本往往成本更高。
- **所有帧模式采用同一采样规则。** 每种模式先检测完整范围内的所有候选帧，再均匀采样（始终保留第一帧和最后一帧）至其上限。模式之间只有候选帧*来源*（关键帧或场景切换）和上限不同，覆盖的分布方式始终相同——因此最后一帧总会落在视频末尾，而不是中途。
- **`efficient` 是速度档**（约 0.5 秒）——它只重建关键帧，因此比需要解码每一帧以查找切换点的场景模式快约 40 倍。在低动态画面中，它返回的帧数也可能比 `balanced` *更多*（关键帧多于场景切换）；“efficient”指快速提取，并不表示帧数更少。
- **`token-burner` 只有在超过上限时才与 `balanced` 不同。** 该片段有 116 个切换点，因此 `balanced` 采样 100 个，而 `token-burner` 保留全部 116 个。对于有数百个切换点的高动态视频，`token-burner` 会保留所有内容（并触发超过 250 帧的令牌警告），而 `balanced` 会稀释至 100 帧。

从一个冷启动 URL 完成端到端处理时，`transcript` 是成本最低的模式；帧模式除了提取时间外，还需要共同的约 37 秒下载时间。

## 安装

| 平台 | 安装方式 |
|---------|---------|
| **Claude Code** | `/plugin marketplace add bradautomates/claude-video`，然后运行 `/plugin install watch@claude-video` |
| **Codex、Cursor、Copilot、Gemini CLI 及其他 50 多种宿主** | `npx skills add bradautomates/claude-video -g` |
| **claude.ai**（网页版） | [下载 `watch.skill`](https://github.com/bradautomates/claude-video/releases/latest) → Settings → Capabilities → Skills → `+` |
| **手动 / 开发** | 运行 `git clone`，然后将 `skills/watch` 符号链接到宿主的 skills 目录（见下文） |

### Claude Code

```
/plugin marketplace add bradautomates/claude-video
/plugin install watch@claude-video
```

以后可使用 `/plugin update watch@claude-video` 更新。

### Codex、Cursor、Copilot、Gemini CLI 及其他 50 多种宿主

[Agent Skills](https://agentskills.io) CLI 会将该 skill 安装到它检测到的任意 agent 中：

```bash
npx skills add bradautomates/claude-video -g
```

`-g` 会为当前用户全局安装（`~/.codex/skills`、`~/.cursor/skills` 等）；去掉它即可安装到当前项目。常用参数：

- `-a, --agent <names…>`——指定宿主，例如 `-a codex -a cursor`
- `-l, --list`——列出此仓库中的 skill，但不安装
- `--copy`——复制文件而不是创建符号链接（适用于不支持符号链接的文件系统）

CLI 会从 `skills/watch/SKILL.md` 发现该 skill，并将整个文件夹——`SKILL.md` 及其 `scripts/` 运行时——作为一个自包含单元复制。`SKILL.md` 会相对于自身安装位置解析其脚本，因此在每种宿主上都以相同方式运行。

以后可使用 `npx skills update watch -g` 更新。

### claude.ai（网页版）

1. 从最新版本中[下载 `watch.skill`](https://github.com/bradautomates/claude-video/releases/latest)。
2. 前往 Settings → Capabilities → Skills。
3. 点击 `+` 并放入该文件。

请先在 Capabilities 下启用“Code execution and file creation”——该 skill 会调用 `ffmpeg` 和 `yt-dlp`，不启用就无法运行。

### 手动安装（开发者）

克隆仓库，并将自包含的 skill 文件夹符号链接到宿主的 skills 目录——符号链接能让你编辑工作树时，安装内容与其保持同步：

```bash
git clone https://github.com/bradautomates/claude-video.git
ln -s "$(pwd)/claude-video/skills/watch" ~/.claude/skills/watch   # or ~/.codex/skills/watch
```

若用于 claude.ai，请从源代码构建 `.skill` 包：运行 `bash skills/watch/scripts/build-skill.sh` 会生成 `dist/watch.skill`。

## 首次运行

第一次调用 `/watch` 时，该 skill 会运行 `scripts/setup.py --check`。如果 `ffmpeg` / `yt-dlp` 不在 PATH 中，或未设置 Whisper API 密钥，它会引导你修复：

- **macOS**——自动运行 `brew install ffmpeg yt-dlp`。
- **Linux**——输出准确的 `apt` / `dnf` / `pipx` 命令。
- **Windows**——输出准确的 `winget` / `pip` 命令。
- **API 密钥**——在 `~/.config/watch/.env` 中创建脚手架文件（权限模式为 `0600`），并包含 `GROQ_API_KEY`（首选）与 `OPENAI_API_KEY` 的注释占位符。

设置完成后，预检不会输出任何内容，`/watch` 将直接运行。该检查只是一次不到 100 毫秒的查找，因此不会拖慢后续运行。

## 使用自己的密钥

大多数公开视频都能通过字幕免费处理。仅当视频确实没有字幕轨道时才会启用 Whisper 回退——通常是本地文件、TikTok、部分 Vimeo，以及偶尔没有字幕的 YouTube 视频。

| 功能 | 所需内容 | 成本 |
|------------|---------------|------|
| 下载 + 原生字幕 | `yt-dlp` + `ffmpeg` | 免费 |
| Whisper 回退（首选） | [Groq API 密钥](https://console.groq.com/keys)——`whisper-large-v3` | 便宜、快速 |
| Whisper 回退（备用） | [OpenAI API 密钥](https://platform.openai.com/api-keys)——`whisper-1` | 标准定价 |
| 完全禁用 Whisper | `--no-whisper` | 免费；无字幕时仅使用帧 |

## 用法

```
/watch https://youtu.be/dQw4w9WgXcQ what happens at the 30 second mark?
/watch https://www.tiktok.com/@user/video/123 summarize this
/watch ~/Movies/screen-recording.mp4 when does the UI break?
/watch https://vimeo.com/123 what tools does she mention?
```

聚焦于特定片段——帧预算更密集、令牌成本更低：
```
/watch https://youtu.be/abc --start 2:15 --end 2:45
/watch video.mp4 --start 50 --end 60
/watch "$URL" --start 1:12:00            # from 1h12m to end
```

其他调整项（传给 `scripts/watch.py`）：

- `--detail transcript|efficient|balanced|token-burner`——保真度/速度旋钮。`transcript` 跳过帧（仅转录文本）；`efficient` 使用快速关键帧（上限 50）；`balanced` 进行场景感知（上限 100）；`token-burner` 进行场景感知且没有上限。
- `--timestamps T1,T2,…`——在每个绝对时间戳（`SS`/`MM:SS`/`HH:MM:SS`）抓取一帧。Claude 会先读取转录文本，再定位演示者提示的时刻（“看这里”“如你所见”）。这些帧会添加到细节模式帧之外（但会预留上限名额）；聚焦模式会丢弃窗口外的提示；与 `--detail transcript` 一起使用时，它们会成为唯一的帧。
- `--max-frames N`——降低帧上限，以减少令牌预算。
- `--resolution W`——当 Claude 需要读取屏幕文字（幻灯片、终端、代码）时，将帧宽提升到 1024 像素。
- `--fps F`——覆盖自动 fps 计算（仍然限制在 2 fps）。
- `--whisper groq|openai`——强制指定 Whisper 后端。
- `--no-whisper`——完全禁用转录；无字幕时仅使用帧。
- `--no-dedup`——保留近似重复帧。默认情况下，帧差异检测会丢弃与上一帧视觉上近乎相同的帧（停留的幻灯片、静态屏幕录制、暂停的视频），从而把帧预算用在不同的内容上；该参数会关闭这一行为。
- `--out-dir DIR`——将工作文件保留在指定位置（默认使用自动生成的临时目录）。

## 限制

- **长视频的准确性取决于细节模式。** 在有限额模式（`efficient`、默认的 `balanced`）下，超过约 10 分钟后覆盖会变得稀疏——帧上限会分散到整段视频中，因此脚本会输出“稀疏扫描”警告，更好的做法是使用 `--start`/`--end` 聚焦后重新运行。`token-burner` 会解除上限，并保留整个视频的*每一个*场景变化帧，因此在长视频上仍能保持完整，代价是使用更多图片令牌。10 分钟只是有限额模式的指导值，并非硬性上限。
- **细节只有一个旋钮。** 默认值保持平衡：场景感知帧、最高 2 fps、100 帧上限。需要快速的 50 帧关键帧扫描时使用 `--detail efficient`，需要无上限场景候选帧时使用 `--detail token-burner`。设置 `WATCH_DETAIL`（位于 `~/.config/watch/.env` 中）可更改默认值。

## 结构

```
.
├── skills/watch/                 # self-contained skill — copied as a unit by every installer
│   ├── SKILL.md                  # skill contract — the source of truth across all surfaces
│   └── scripts/
│       ├── watch.py              # entry point — orchestrates download → frames → transcript
│       ├── download.py           # yt-dlp wrapper
│       ├── frames.py             # ffmpeg frame extraction + auto-fps logic
│       ├── transcribe.py         # VTT parsing + dedupe + Whisper orchestration
│       ├── whisper.py            # Groq / OpenAI clients (pure stdlib)
│       ├── config.py             # shared config (~/.config/watch/.env)
│       ├── setup.py              # preflight + installer
│       └── build-skill.sh        # build dist/watch.skill for claude.ai upload (dev-only)
├── hooks/                        # SessionStart status hook (Claude Code only)
├── .claude-plugin/               # plugin.json + marketplace.json (Claude Code)
├── .codex-plugin/                # plugin.json — Codex/agents manifest ("skills": "./skills/")
├── .agents/plugins/              # marketplace.json — Agent Skills marketplace listing
├── AGENTS.md → CLAUDE.md         # generic-agent entry point
├── tests/                        # pytest suite (ffmpeg-synthesized clips, no network)
└── .github/workflows/            # release.yml — auto-builds watch.skill on tag push
```

## 开发

```bash
# Run the test suite (stdlib + pytest; ffmpeg required for frame tests):
python3 -m pytest -q

# Build the claude.ai upload bundle:
bash skills/watch/scripts/build-skill.sh      # → dist/watch.skill
```

发布：创建 `vX.Y.Z` 标签并推送。工作流会构建 `dist/watch.skill` 并将其附加到 GitHub Release。请让 `skills/watch/SKILL.md`、`.claude-plugin/plugin.json` 和 `.codex-plugin/plugin.json` 中的版本保持同步。

版本历史请参阅 [CHANGELOG.md](CHANGELOG.md)。

## 开源

MIT 许可证。

基于 `yt-dlp`、`ffmpeg` 和 Claude 的多模态 `Read` 工具构建。Whisper 转录由 [Groq](https://groq.com) 或 [OpenAI](https://openai.com) 提供。

由 Brad Bonanno 构建——我会在 [YouTube（@bradbonanno）](https://www.youtube.com/@bradbonanno)分享关于使用 AI 构建产品的内容，也会在 [Solaris Automation](https://www.solarisautomation.io/) 为企业构建 AI 操作系统。如果 `/watch` 能让你不必在视频中来回拖动查找内容，欢迎到频道里打个招呼。

## Star 历史

<a href="https://www.star-history.com/?repos=bradautomates%2Fclaude-video&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=bradautomates/claude-video&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=bradautomates/claude-video&type=date&legend=top-left" />
   <img alt="Star 历史图表" src="https://api.star-history.com/chart?repos=bradautomates/claude-video&type=date&legend=top-left" />
 </picture>
</a>

---

[github.com/bradautomates/claude-video](https://github.com/bradautomates/claude-video) · [@bradbonanno](https://www.youtube.com/@bradbonanno) · [Solaris Automation](https://www.solarisautomation.io/) · [许可证](LICENSE)
