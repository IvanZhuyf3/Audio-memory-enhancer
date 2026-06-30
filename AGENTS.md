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
  pyannote-audio 4.0.4, silero-vad, librosa, Pillow+pillow-heif, PyYAML, Jinja2,
  python-docx, PyMuPDF (fitz).
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
                 │    ├─ plaud_sync.parse_transcript()    # strip preamble
                 │    ├─ note_templates.append_clip_to_weekly()
                 │    ├─ _atomic_append → Obsmem/raw/YYYY-W##.md
                 │    └─ plaud_sync.download_audio() → archive to E:/Audio-arxiv
                 │
                 └─ LONG: process_long()
                      ├─ transcribe_local.build_asr_context()   # dictionary → context
                      ├─ transcribe_local.transcribe()
                      │    ├─ _vad_split()              # silero-vad chunking
                      │    ├─ _run_asr()                # Qwen3-ASR per chunk
                      │    ├─ _build_sentence_segments() # timestamps → sentences
                      │    ├─ _assign_speakers()         # pyannote diarization
                      │    └─ ai_clean()                 # DeepSeek post-correction
                      ├─ classify.classify_long()        # project + theme + summary
                      ├─ note_templates.render_long_meeting()
                      ├─ _atomic_write → Meeting Notes/
                      └─ plaud_sync.download_audio() → archive to E:/Audio-arxiv
```

## Key Modules

| File | Role | Status |
|---|---|---|
| `pipeline.py` | CLI + orchestrator. `sync`/`list`/`bootstrap-projects`/`reprocess`/`digest`. | Working (digest is stub) |
| `plaud_sync.py` | Direct Plaud REST client. JWT auto-refresh, transcription trigger, transcript parser. | Working |
| `state.py` | Atomic state.json, state machine (DISCOVERED→...→DONE), crash recovery, pruning. | Working |
| `routing.py` | Duration split, vault target paths, audio archive, project registry. | Working |
| `classify.py` | DeepSeek classifiers. `classify_long` (project+theme+summary+actions). Short-path classification deferred to digest pass. | Working |
| `note_templates.py` | Weekly raw appender + Meeting Notes renderer. | Working |
| `transcribe_local.py` | Qwen3-ASR + diarization + AI clean + GPU lock + `build_asr_context()`. | Working (validated on GPU) |

## Domain Dictionary Subsystem

### Overview

`dictionary.md` is a bilingual (English | Chinese) domain glossary that biases
Qwen3-ASR toward correct terminology. It feeds into the model's `context`
parameter (system message) via `build_asr_context()`, providing pre-correction
before transcription even happens. This complements the post-correction
`ai_clean()` DeepSeek pass.

### Dictionary format

```
- stimulated Raman scattering (SRS)        ← extracts: full term + "SRS"
- lipid droplet | 脂滴                      ← extracts: "lipid droplet" + "脂滴"
- thermal lensing                          ← extracts: "thermal lensing" (rare, prefer bilingual)
```

Three forms per line:
1. **Abbreviation**: `- full term (ABBR)` → parser extracts both full + abbreviation
2. **Bilingual**: `- english term | 中文翻译` → parser splits on ` | ` and extracts both
3. **Plain**: `- term` → single entry (rare; most are bilingual)

### build_asr_context() details

- Located in `transcribe_local.py`
- `max_chars=15000` (was 500; bumped for 354-term dictionary)
- Output: `"Domain-specific terms that may appear in this audio: term1, term2, ..."`
- Also appends project keywords (from `projects.yaml`) and recording title
- Context budget: ~45K tokens after audio (model max_position_embeddings=65536,
  audio rate=12.5 tokens/sec, max 20-min chunks = 15K audio tokens)

### Current dictionary stats

- **354 entries** from two vibrational microscopy review papers
- **297 bilingual** (non-abbr, translated to Chinese)
- **57 abbreviation** (universal, no translation needed)
- **708 individual terms** reaching Qwen3-ASR (~2,548 tokens)

### Dictionary maintenance scripts

| Script | Purpose |
|---|---|
| `scripts/extract_paper_text.py` | Extract text from .docx (python-docx) |
| `scripts/extract_dictionary.py` | Feed texts to DeepSeek → structured glossary |
| `scripts/translate_dictionary.py` | Batch-translate non-abbr terms to Chinese (idempotent) |
| `scripts/dict_extraction_raw.txt` | Raw DeepSeek response from extraction (audit trail) |
| `scripts/dict_translate_raw.txt` | Raw DeepSeek response from translation (audit trail) |

Documented in the `plaud-sync` opencode skill (`~/.config/opencode/skills/plaud-sync/SKILL.md`),
under "Maintain dictionary" sub-function.

### Measured impact

On a 75-min Chinese-English code-switched meeting:

| Metric | Without dict (7 terms) | With dict (354 terms) |
|---|---|---|
| Domain terms in transcript | 28/707 (4.0%) | **399/707 (56.4%)** |
| `photothermal` occurrences | 0 | 21 |
| `mid-infrared` occurrences | 0 | 20 |
| `脂滴` (lipid droplet) | 0 | 5 |
| `光热` (photothermal) | 0 | 16 |

## Testing & Comparison Infrastructure

### Blind A/B comparison (`scripts/blind_compare.py`)

Generates an HTML page with shuffled A/B pairs (Qwen3 vs Plaud) for blind
evaluation. DeepSeek filters to pairs where meaning actually differs. After the
user picks winners, `--reveal` maps picks back to sources.

**Results across 3 rounds** (20 pairs each, same 75-min meeting):

| Round | Config | Plaud wins | Qwen3 wins | Ties |
|---|---|---|---|---|
| 1 | `language=auto`, broken alignment | 4 | 0 | 36 |
| 2 | `language=zh`, char-bigram align | 10 | 6 | 4 |
| 3 | `language=zh` + 354-term bilingual dict | 7 | 4 | 9 |

Plaud remains the better overall transcription, but the dictionary dramatically
closed the gap in domain vocabulary. Ties jumped 4→9, meaning the dictionary
eliminated segments where Qwen3 was previously clearly worse.

### Other comparison tools

- `scripts/compare_dict_effect.py` — domain term coverage + Plaud similarity
- `scripts/asr_test.py` — standalone Qwen3-ASR runner (no vault writes)
- `scripts/compare_transcripts.py` — Qwen3 vs Plaud stats + WER

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
- `extra_data.tranConfig` → `{language: "zh-0", diarization: 1, llm: "auto", ...}`

### Transcription trigger (CDP-captured)

Two API calls on the consumer API (`api.plaud.ai`):
1. `PATCH /file/<id>` — set `tranConfig` (language, diarization, etc.)
2. `POST /ai/transsumm/<id>` — fire transcription task

No partner API keys needed. Poll `/ai/trans-status` for completion.

**Known limitation:** very short clips (<30s) may never get transcribed by Plaud
even after triggering.

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

## Phase 3 — Qwen3-ASR Pipeline (validated)

`transcribe_local.py` ports the proven ASR pipeline from
`audio_transcribe_notes/transcribe.py`:

- `_vad_split` — silero-vad chunking for audio >180s (ForcedAligner limit)
- `_build_sentence_segments` — char-level timestamps → sentence segments
- `_assign_speakers` — pyannote diarization → speaker labels
- `_run_asr` — Qwen3-ASR + ForcedAligner, per-chunk 10min timeout, chunk log
- `ai_clean` — DeepSeek dictionary correction (operates on segments directly)
- `build_asr_context` — dictionary → system message (pre-correction, see above)
- `transcribe()` — main entry, returns `{segments, transcript_markdown, transcript_text, speakers, language}`

### GPU lock (GpuLock class)

File-based cross-process lock at `config.gpu_lock` (`C:/Users/Yifan/venvs/.gpu_lock`).
Uses `O_CREAT | O_EXCL` for atomic creation + PID-in-file for stale-lock recovery.
Acquired around the ASR + diarization phase only (not the whole pipeline).
**Purpose**: serialize against `audio_transcribe_notes/monitor.py` so two Qwen3-ASR
jobs never contend for GPU memory. Short path (no GPU) does NOT acquire it.

### Qwen3-ASR context budget

- Model: `Qwen/Qwen3-ASR-1.7B`, `max_position_embeddings=65536` tokens
- Audio token rate: 12.5 tokens/sec (Whisper mel at 100Hz → 3× Conv2d stride-2)
- `MAX_ASR_INPUT_SECONDS=1200` (20 min per chunk, auto-split by VAD)
- `max_new_tokens=4096` for generated text
- **Context budget after audio: ~45K tokens** — ample room for a 354-term dictionary (~2.5K tokens)

### torchcodec warnings are benign

pyannote emits `Could not load libtorchcodec` warnings on every run. These are
harmless — audio is passed as an in-memory `{'waveform': tensor, 'sample_rate': int}`
dict, so pyannote never needs torchcodec's FFmpeg decoder. Don't waste time
"fixing" this.

### `language: zh` is critical

Force `language=zh` in config. Auto-detect (`language=auto`) causes severe
repetition hallucination on Chinese-English code-switching content. This was the
key finding from blind comparison Round 1 (Qwen3 0 wins with auto).

## Phase 5 — Digest Pass (next up)

The `pipeline.py digest` command is currently a stub. The goal:

1. Read accumulated `Obsmem/raw/YYYY-W##.md` weekly files
2. Use DeepSeek to classify + consolidate into themed notes
3. Write to `Obsmem/digest/` as structured reference notes
4. Remove `#unreviewed` tags from consolidated items

The raw inbox is accumulating correctly. The digest pass makes sense once there
are a few weeks of memos to consolidate.

## Design Decisions (locked with user)

1. **Duration split @ 15min** (configurable) decides short vs long path.
2. **Short** → Plaud cloud transcript → `Obsmem/raw/` (intake inbox). Phase 5
   digests into `Obsmem/digest/`.
3. **Long** → local Qwen3-ASR → `Meeting Notes/` with full treatment.
4. **Project routing**: single home + frontmatter link (no duplication).
5. **DeepSeek** for all LLM calls (classification, AI cleaning, dictionary extraction/translation).
6. **Audio archive**: `E:/Audio-arxiv/<YYYY-MM>/<plaud_id>.<ext>` (outside OneDrive).
7. **`language: zh`** forced — prevents repetition hallucination on code-switched content.
8. **Bilingual dictionary** — non-abbr terms have `| 中文翻译` for Chinese-English code-switching.
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
- Don't use `language=auto` — it causes repetition hallucination on Chinese content.
- Don't truncate `build_asr_context()` below `max_chars=15000` — the dictionary needs room.
