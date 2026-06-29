# AGENTS.md — Audio-memory-enhancer

## What This Is

Procedural Python pipeline: Plaud cloud → transcribe → DeepSeek classify →
Obsidian vault. Windows-native, GPU-optional (only the long-recording path uses
CUDA). Six focused modules, no classes (matches the sibling
`audio_transcribe_notes` style). State machine + atomic state.json make it
idempotent and crash-recoverable.

## Critical Path Facts

### Venv — DO NOT create a new one

- **Reuse** `C:\Users\Yifan\venvs\audio_transcribe\` (shared with audio_transcribe_notes).
- Has: torch 2.11.0+cu126, torchaudio, qwen-asr 0.0.6, openai 2.30.0,
  pyannote-audio 4.0.4, silero-vad, librosa, Pillow+pillow-heif, PyYAML, Jinja2.
- `requirements.txt` is a **reference**, NOT an install manifest. Do not pip install
  into this venv without checking what's already there.

### Secrets — read from the sibling project

- DeepSeek API key + HF token live in `audio_transcribe_notes/config.ini` `[defaults]`.
- `config.yaml` → `secrets_source` points there. Never duplicate secrets into this repo.
- Plaud credentials live in `~/.plaud/config.json` (managed by upstream `plaud login`).

### Windows Encoding

Every `open()` / `read_text()` / `write_text()` uses `encoding="utf-8"`. Windows
defaults to GBK on Chinese locale. `run_pipeline.bat` sets `PYTHONIOENCODING=utf-8`.
Don't break this when editing.

### Path resolution

`load_config()` resolves `state_file` + `projects_registry` against the config
file's directory (not CWD). This was a bug once — bootstrap-projects wrote
`projects.yaml` to the wrong dir. Don't reintroduce it; always go through
`load_config()` for these paths.

## Architecture — Data Flow

```
[Task Scheduler / manual]
  └─ run_pipeline.bat sync
       └─ pipeline.cmd_sync()
            ├─ state.reset_in_flight()          # crash recovery
            ├─ state.prune_done()                # cleanup
            ├─ plaud_sync.list_recordings()      # GET /file/simple/web
            ├─ diff vs state.json → DISCOVERED
            ├─ routing.decide_path(duration)     # short (<15m) | long (≥15m)
            └─ per recording:
                 ├─ SHORT: process_short()
                 │    ├─ plaud_sync.get_recording(id)     # cloud transcript
                 │    ├─ classify.classify_short()         # DeepSeek JSON
                 │    ├─ note_templates.render_short_memo()
                 │    ├─ _atomic_write → Obsmem/raw/
                 │    └─ plaud_sync.download_audio() → archive to E:/Audio-arxiv
                 └─ LONG: process_long()  [Phase 3 — currently raises]
                      └─ transcribe_local.transcribe() → NotImplementedError
```

## Key Modules

| File | Role | Status |
|---|---|---|
| `pipeline.py` | CLI + orchestrator. `sync`/`list`/`bootstrap-projects`/`reprocess`/`digest`. | Working (Phase 1) |
| `plaud_sync.py` | Direct Plaud REST client. Reads `~/.plaud/config.json`, auto-refreshes JWT, handles region mismatch. | Working |
| `state.py` | Atomic state.json, state machine (DISCOVERED→...→DONE), crash recovery, pruning. | Working |
| `routing.py` | Duration split, vault target paths, audio archive, project registry (yaml load/save/bootstrap). | Working |
| `classify.py` | DeepSeek classifiers. `classify_short` (sub-type), `classify_long` (project+theme+summary+actions). | Working (short proven, long untested) |
| `note_templates.py` | Markdown + frontmatter builders for memos + meetings. Pure string templating. | Working |
| `transcribe_local.py` | Local Qwen3-ASR pipeline (ported from audio_transcribe_notes). GPU lock + diarization + AI clean. | Working (Phase 3, validated on GPU) |

## Plaud API (reverse-engineered) — VERIFIED against real data

Endpoints (from upstream `packages/core/src/client.ts`, confirmed working):
- `POST /auth/access-token` — form-urlencoded login (username=email, password)
- `GET /file/simple/web` → `data_file_list` (filter `is_trash=false`). Each item
  has full **32-char hex** `id` (e.g. `7f4baff36798c5038e1f969cae5aa804`).
- `GET /file/detail/<id>` → `data` object with the fields below.
- `GET /file/temp-url/<id>?is_opus=false` → pre-signed MP3 URL.
- `GET /file/download/<id>` → opus bytes (fallback, rarely needed).
- `GET /user/me` → account info.

Base URLs: `us` → `https://api.plaud.ai`, `eu` → `https://api-euc1.plaud.ai`.
**Browser User-Agent required** (default urllib UA gets 403). Region mismatch
returns `{status:-302, data:{domains:{api:"..."}}}` — handled in
`_resolve_region_mismatch()`.

`duration` and `start_time` are **epoch-milliseconds** (not seconds). Common bug
source — always divide by 1000.

### Detail endpoint shape (verified)

The `/file/detail/<id>` response `data` object contains:
- `file_id`, `file_name`, `duration` (ms), `start_time` (epoch ms), `is_trash`
- `pre_download_content_list[].data_content` — **inline content**. For SHORT
  recordings Plaud inlines the raw transcript here inside an auto_sum preamble
  (e.g. `转写内容较短，无需生成总结。音频转写原文如下：\n> [Speaker 1]...`). For
  longer recordings this becomes the AI summary (raw transcript NOT inlined).
  → Use `plaud_sync.parse_transcript()` to strip the preamble + extract the
    `[Speaker N]`-tagged body. Returns `{text, speakers, had_preamble}`.
- `content_list[]` — S3 pre-signed gzipped links, four types:
  - `transaction` → `trans_result.json.gz` (raw transcript with segment timing)
  - `outline` → `outline.json.gz`
  - `transaction_polish` → polished/smoothed transcript
  - `auto_sum_note` → `ai_content.md.gz` (AI summary)
  These are the cleanest sources for the Phase 3 long path (fetch + gunzip + parse).
- `extra_data.tranConfig` → `{language: "zh-0", diarization: 1, llm: "auto", ...}`
  — valuable classification metadata.
- `embeddings.{Speaker N}` → 256-dim speaker embedding vectors (unused for now).

### Skipping un-transcribed recordings

Recordings with `is_trans=false` (Plaud hasn't processed them yet) return empty
transcripts. `process_short()` auto-triggers cloud transcription via the
captured endpoints and defers to the next sync.

**Known limitation:** very short clips (<30s) may never get transcribed by Plaud
even after triggering. Observed on a 26-second test clip: `tranConfig` was set
(PATCH succeeded) but `content_list` stayed empty indefinitely. For these, the
pipeline should eventually fall back to local Qwen3-ASR (future enhancement).

### Plaud summary vs raw transcript

For short recordings, `pre_download_content_list[0].data_content` contains:
- **<1 min clips**: raw transcript with `[Speaker N]` tags (prefixed by a
  Chinese preamble that `parse_transcript()` strips)
- **~2 min clips**: full AI summary with markdown formatting (`##`, `**bold**`,
  checkboxes) — NOT a raw transcript. This summary gets appended as a very long
  bullet to the weekly raw file. The markdown formatting may render oddly inside
  a bullet list. The raw transcript for these is available via
  `content_list[0].data_link` (gzipped S3 JSON) but not currently fetched on
  the short path. Acceptable tradeoff — the summary is useful content for the
  digest pass, and the user can edit in Obsidian.

## State Machine

```
DISCOVERED → DOWNLOADING → TRANSCRIBING → ROUTED → DONE
                              │                        │
                              ▼                        ▼
                           FAILED (retries < max → DISCOVERED)
```

- `reset_in_flight()` on every startup moves DOWNLOADING/TRANSCRIBING → DISCOVERED.
- `mark_failed()` increments retries; at `max_retries` (3) the recording sticks at FAILED.
- DONE entries pruned after `prune_done_after_days` (60).
- State persisted after EACH recording (crash-safe progress).

## Per-Recording Record Shape (state.json)

```json
{
  "state": "DONE",
  "first_seen": "2026-06-28T15:05:00",
  "duration_s": 1832.4,
  "recorded_at": "2026-06-28T14:30:00+00:00",
  "plaud_filename": "Meeting.m4a",
  "path_chosen": "long",
  "transcript_source": "local-qwen3-asr",
  "vault_path": "Meeting Notes/meeting_2026-06-28_14h30_Standup",
  "archive_path": "E:/Audio-arxiv/2026-06/abc123.mp3",
  "classification": {"sub_type": "...", "project": "...", "confidence": 0.9},
  "done_at": "2026-06-28T15:06:11",
  "retries": 0
}
```

## Phase 3 — done (validated on GPU)

`transcribe_local.py` ports the proven ASR pipeline from
`audio_transcribe_notes/transcribe.py`:

- `_vad_split` — silero-vad chunking for audio >180s (ForcedAligner limit)
- `_build_sentence_segments` — char-level timestamps → sentence segments
- `_assign_speakers` — pyannote diarization → speaker labels
- `_run_asr` — Qwen3-ASR + ForcedAligner, per-chunk 10min timeout, chunk log
- `ai_clean` — DeepSeek dictionary correction (operates on segments directly)
- `transcribe()` — main entry, returns `{segments, transcript_markdown, transcript_text, speakers, language}`

### GPU lock (GpuLock class)

File-based cross-process lock at `config.gpu_lock` (`C:/Users/Yifan/venvs/.gpu_lock`).
Uses `O_CREAT | O_EXCL` for atomic creation + PID-in-file for stale-lock recovery.
Acquired around the ASR + diarization phase only (not the whole pipeline).
**Purpose**: serialize against `audio_transcribe_notes/monitor.py` so two Qwen3-ASR
jobs never contend for GPU memory. Short path (no GPU) does NOT acquire it.

### torchcodec warnings are benign

pyannote emits `Could not load libtorchcodec` warnings on every run. These are
harmless — audio is passed as an in-memory `{'waveform': tensor, 'sample_rate': int}`
dict, so pyannote never needs torchcodec's FFmpeg decoder. Don't waste time
"fixing" this.

### Long-path validation result

Tested by forcing the 0.7-min test clip through the long path (threshold→0):
GPU lock acquired/released ✓, Qwen3-ASR produced 12 segments ✓, pyannote found
1 speaker ✓, DeepSeek classify_long generated theme + summary ✓, note written
to Meeting Notes/ ✓. The torchcodec warning appeared but did not affect output.

## Design Decisions (locked with user)

1. **Duration split @ 15min** (configurable) decides short vs long path.
2. **Short** → Plaud cloud transcript → `Obsmem/raw/` (intake inbox). Phase 5
   digests into `Obsmem/digest/`.
3. **Long** → local Qwen3-ASR → `Meeting Notes/` with full treatment.
4. **Project routing**: single home + frontmatter link (no duplication).
   `project: <name>` field enables Obsidian backlinks/Dataview.
5. **Time-sensitivity**: type flag only; manual dates (no auto-extraction).
6. **DeepSeek** for all LLM calls (reuse existing key).
7. **Audio archive**: `E:/Audio-arxiv/<YYYY-MM>/<plaud_id>.<ext>` (outside OneDrive).
8. **Trigger**: Windows Task Scheduler, ~30 min cadence.
9. **Review**: `#unreviewed` tag on every note at intake; user removes after spot-check.

## What Not To Do

- Don't create a new venv or pip-install into the shared one without checking versions.
- Don't duplicate secrets — read from `audio_transcribe_notes/config.ini`.
- Don't use relative paths for state/projects without going through `load_config()`.
- Don't drop the `encoding="utf-8"` on any file I/O.
- Don't shell out to the TS `plaud` CLI for normal operation — call the REST API
  directly via `plaud_sync.py` (the CLI is login-only).
- Don't run local Qwen3-ASR without acquiring the GPU lock.
- Don't add classes — keep it procedural (matches sibling project style).
