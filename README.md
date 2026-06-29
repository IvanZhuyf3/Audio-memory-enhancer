# Audio-memory-enhancer

Plaud wearable → cloud → Qwen3-ASR → DeepSeek classifier → Obsidian vault.

Pulls recordings from your Plaud cloud account, transcribes them (Plaud cloud
transcript for short memos, local Qwen3-ASR for long meetings), classifies the
content, and routes each note to the right folder in your Obsidian vault.

## Pipeline at a glance

```
Plaud device ──(auto)──► Plaud cloud ──► this pipeline ──► Obsidian vault
                                            │
                       ┌────────────────────┴───────────────────┐
                       ▼                                         ▼
              < 15 min (short)                            ≥ 15 min (long)
              Plaud cloud transcript                      local Qwen3-ASR
                       │                                         │
                       ▼                                         ▼
              DeepSeek sub-type classify              DeepSeek project + theme
              (time-sensitive / long-term /           + summary + action items
               project-snippet)                                │
                       │                                         ▼
                       ▼                              Meeting Notes/<date>_<theme>.md
              Obsmem/raw/<date>_<id>.md               (full transcript, speakers,
              (raw inbox; Phase 5 will                summary, action items,
               consolidate into Obsmem/digest/)       project link via frontmatter)
```

**State**: idempotent + crash-recoverable via `state.json` (per-recording state machine).

## Setup

### 1. One-time Plaud login (interactive)

```powershell
cd C:\Users\Yifan\OneDrive\Opencode_workspace\Plaud-toolkit
npx tsx packages/cli/bin/plaud.ts login
```

Enter your Plaud email + password + region (`us`/`eu`). Tokens last ~300 days and
auto-refresh. Credentials are stored at `~/.plaud/config.json`.

### 2. Python environment — **do NOT create a new venv**

This project reuses the existing proven venv at:

```
C:\Users\Yifan\venvs\audio_transcribe\
```

which has every dependency (torch+CUDA, qwen-asr, openai, pyyaml, jinja2, pyannote,
silero-vad, librosa, Pillow). Reinstalling risks breaking the shared ASR pipeline.
See `requirements.txt` for the verified version list.

### 3. Secrets

The DeepSeek API key + HuggingFace token are read from the shared config:

```
C:\Users\Yifan\OneDrive\Opencode_workspace\audio_transcribe_notes\config.ini
```

Ensure `deepseek_api_key` is set there (used for classification + AI cleaning).

### 4. Task Scheduler (optional, for periodic auto-sync)

Point a Windows Scheduled Task at:

- **Program**: `C:\Users\Yifan\OneDrive\Opencode_workspace\Audio-memory-enhancer\run_pipeline.bat`
- **Arguments**: `sync`
- **Start in**: `C:\Users\Yifan\OneDrive\Opencode_workspace\Audio-memory-enhancer`
- **Trigger**: every 30 minutes (configurable)

## Usage

All commands via the launcher (activates the shared venv automatically):

```powershell
.\run_pipeline.bat sync --dry-run       # show what would be processed (no writes)
.\run_pipeline.bat sync                 # process new recordings
.\run_pipeline.bat sync --only <id>     # one recording
.\run_pipeline.bat sync --limit 3       # cap this run to 3 recordings
.\run_pipeline.bat list                 # show all Plaud recordings + state
.\run_pipeline.bat bootstrap-projects   # rescan vault Projects/ → projects.yaml
.\run_pipeline.bat reprocess <id>       # delete old note, reset to DISCOVERED
.\run_pipeline.bat digest               # (Phase 5 stub) consolidate raw memos
```

Or directly with the venv active:

```powershell
& C:\Users\Yifan\venvs\audio_transcribe\Scripts\activate.bat
python pipeline.py sync --dry-run
```

## Configuration

Edit `config.yaml` for thresholds, paths, and model names. Key fields:

| Field | Default | Purpose |
|---|---|---|
| `transcription.short_threshold_min` | `15` | `<15min` → cloud; `≥15min` → local Qwen3-ASR |
| `vault.memo_raw_folder` | `Obsmem/raw` | Short memos land here |
| `vault.meeting_notes_folder` | `Meeting Notes` | Long recordings land here |
| `audio_archive.root` | `E:/Audio-arxiv` | Audio files archived by month |
| `llm.model` | `deepseek-v4-flash` | DeepSeek model for classification |
| `projects_registry` | `projects.yaml` | Project registry for routing |

Edit `projects.yaml` to add aliases + keywords for each project so the
classifier can match recordings to projects by content. Run
`bootstrap-projects` to re-scan the vault for new project folders.

## Content types & routing

| Path | Trigger | Vault target | Note shape |
|---|---|---|---|
| Short | duration < `short_threshold_min` | `Obsmem/raw/<date>_<plaud_id>.md` | Cloud transcript + sub-type tag |
| Long | duration ≥ `short_threshold_min` | `Meeting Notes/meeting_<date>_<theme>.md` | Full transcript, speakers, summary, action items |

Every note gets an `#unreviewed` tag at intake. Remove the tag once you've
spot-checked the routing.

### Short memo sub-types (DeepSeek-classified)

- `time-sensitive` — deadlines, reminders, follow-ups
- `long-term` — reference knowledge, ideas, reflections
- `project-snippet` — relates to a registered project

Phase 5's digest pass will consolidate `Obsmem/raw/` → `Obsmem/digest/`.

## Status

| Phase | Status |
|---|---|
| 1 — Skeleton + Plaud cloud sync + dry-run | ✅ Done |
| 2 — Short-memo path end-to-end | ✅ Wired (untested with real data) |
| 3 — Long-recording path (port Qwen3-ASR) | ⏳ `transcribe_local.py` is a stub |
| 4 — Polish (reprocess, inbox dashboard, Task Scheduler docs) | ⏳ Partial |
| 5 — Digest pass + opencode skill wrapper | ⏳ Deferred |

## Architecture

- `pipeline.py` — CLI + orchestrator (`sync`, `list`, `bootstrap-projects`, ...)
- `plaud_sync.py` — Direct Plaud REST API client (reads `~/.plaud/config.json`,
  calls the API directly, auto-refreshes tokens). No subprocess, no text parsing.
- `state.py` — Atomic `state.json`, per-recording state machine, crash recovery.
- `routing.py` — Duration split, vault target paths, audio archive, project registry.
- `classify.py` — DeepSeek classifiers (short sub-type + long project/theme/summary).
- `note_templates.py` — Markdown + frontmatter builders.
- `transcribe_local.py` — Phase 3: local Qwen3-ASR (stub until ported).

See `AGENTS.md` for the architectural deep-dive.

## Related

- Upstream Plaud API toolkit: `../Plaud-toolkit/` (used only for `plaud login`)
- ASR pipeline source (for Phase 3 port): `../audio_transcribe_notes/transcribe.py`
