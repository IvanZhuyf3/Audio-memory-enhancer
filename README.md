# Audio-memory-enhancer

Plaud wearable → cloud → Qwen3-ASR → DeepSeek classifier → Obsidian vault.

Pulls recordings from your Plaud cloud account, transcribes them (Plaud cloud
transcript for short memos, local Qwen3-ASR for long meetings), and routes
each note to the right folder in your Obsidian vault.

## Pipeline at a glance

```
Plaud device ──(auto)──► Plaud cloud ──► this pipeline ──► Obsidian vault
                                            │
                       ┌────────────────────┴───────────────────┐
                       ▼                                         ▼
              < 15 min (short)                            ≥ 15 min (long)
              Plaud cloud transcript                      local Qwen3-ASR
                       │                                 + bilingual dictionary
                       │                                 + pyannote diarization
                       │                                 + DeepSeek AI clean
                       │                                         │
                       ▼                                         ▼
              Obsmem/raw/YYYY-W##.md                   Meeting Notes/<date>_<theme>.md
              (weekly accumulator,                     (full transcript, speakers,
               one bullet per clip)                     summary, action items)
                       │
                       ▼
              Phase 5: digest pass
              Obsmem/raw/ → Obsmem/digest/
              (DeepSeek consolidation into themed notes)
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

Ensure `deepseek_api_key` is set there (used for classification + AI cleaning +
dictionary translation).

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
| `transcription.language` | `zh` | Force Chinese — avoids repetition hallucination on code-switched content |
| `vault.memo_raw_folder` | `Obsmem/raw` | Short memos land here |
| `vault.meeting_notes_folder` | `Meeting Notes` | Long recordings land here |
| `audio_archive.root` | `E:/Audio-arxiv` | Audio files archived by month |
| `llm.model` | `deepseek-v4-flash` | DeepSeek model for classification |
| `projects_registry` | `projects.yaml` | Project registry for routing |

Edit `projects.yaml` to add aliases + keywords for each project so the
classifier can match recordings to projects by content. Run
`bootstrap-projects` to re-scan the vault for new project folders.

## Domain Dictionary (`dictionary.md`)

Biilingual (English | Chinese) domain glossary that biases Qwen3-ASR toward
correct terminology via the `context` parameter (system message). Built from
two vibrational microscopy review papers, currently **354 terms** (708
individual terms including abbreviations + Chinese translations).

### Dictionary format

```
- stimulated Raman scattering (SRS)        ← abbreviation: extracts full + abbr
- lipid droplet | 脂滴                      ← bilingual: extracts en + zh
- thermal lensing | 热透镜效应              ← bilingual
```

`build_asr_context()` in `transcribe_local.py` parses this and feeds all forms
into the ASR context string (~2.5K tokens, well within the ~45K token model budget).

### Maintaining the dictionary

Three scripts handle the full lifecycle (see `plaud-sync` opencode skill for details):

```powershell
# 1. Extract text from papers
#    .docx: python-docx (in venv)
#    .pdf: MinerU via mineru-pdf2md skill (far better for academic layouts)

# 2. Extract domain terms via DeepSeek
python scripts/extract_dictionary.py    # → outputs 150-400 terms to dictionary.md

# 3. Auto-translate non-abbreviation terms to Chinese
python scripts/translate_dictionary.py  # → adds "| 中文翻译" to non-abbr entries
```

The translation script is idempotent — skips entries that already have ` | ` separator.

### Measured impact

Dictionary injection raised domain term recognition from **4% → 56%** (399/707
terms appearing in a 75-min meeting transcript). Key terms like `photothermal`,
`mid-infrared`, `lipid droplet/脂滴` went from zero to 5-21 occurrences.

## Content types & routing

| Path | Trigger | Vault target | Note shape |
|---|---|---|---|
| Short | duration < `short_threshold_min` | `Obsmem/raw/YYYY-W##.md` | One bullet per clip, cloud transcript |
| Long | duration ≥ `short_threshold_min` | `Meeting Notes/meeting_<date>_<theme>.md` | Full transcript, speakers, summary, action items |

Every note gets an `#unreviewed` tag at intake. Remove the tag once you've
spot-checked the routing.

Short-path classification (time-sensitive / long-term / project-snippet) is
deferred to Phase 5's digest pass.

## Status

| Phase | Status |
|---|---|
| 1 — Skeleton + Plaud cloud sync + dry-run | ✅ Done |
| 2 — Short-memo path (weekly raw files) | ✅ Done + validated |
| 3 — Long-recording path (local Qwen3-ASR + bilingual dictionary) | ✅ Done + validated |
| 4 — Polish (Task Scheduler docs, opencode skill, dictionary toolchain) | ✅ Done |
| 5 — Digest pass (consolidate raw → themed notes) | ⏳ Next — stub in place |

## Architecture

### Core modules

| File | Role |
|---|---|
| `pipeline.py` | CLI + orchestrator (`sync`, `list`, `bootstrap-projects`, `reprocess`, `digest`) |
| `plaud_sync.py` | Direct Plaud REST API client (JWT auto-refresh, transcription trigger, transcript parser) |
| `state.py` | Atomic `state.json`, per-recording state machine, crash recovery |
| `routing.py` | Duration split, vault target paths, audio archive, project registry |
| `classify.py` | DeepSeek classifiers (long: project + theme + summary + actions) |
| `note_templates.py` | Markdown + frontmatter builders (weekly raw appender + meeting notes renderer) |
| `transcribe_local.py` | Qwen3-ASR pipeline: VAD chunking, ForcedAligner, pyannote diarization, DeepSeek AI clean, GPU lock, `build_asr_context()` |
| `config.yaml` | All thresholds, paths, model names |
| `projects.yaml` | Project registry for routing (aliases + keywords) |
| `dictionary.md` | 354-term bilingual domain glossary |

### Scripts (`scripts/`)

| File | Purpose |
|---|---|
| `extract_paper_text.py` | Extract text from .docx (python-docx) for term mining |
| `extract_dictionary.py` | Feed paper texts to DeepSeek → structured domain glossary |
| `translate_dictionary.py` | Auto-translate non-abbr terms to Chinese (idempotent) |
| `compare_dict_effect.py` | Measure domain term coverage + Plaud similarity (old vs new dict) |
| `blind_compare.py` | Blind A/B comparison HTML generator + reveal (Qwen3 vs Plaud) |
| `asr_test.py` | Standalone Qwen3-ASR runner (no vault writes, saves JSON) |
| `compare_transcripts.py` | Qwen3 vs Plaud stats + WER |
| `debug_detail.py` | Raw Plaud API response dumper |
| `cdp_capture.mjs` | Chrome DevTools Protocol network capture (endpoint discovery) |

See `AGENTS.md` for the architectural deep-dive and `work_report.md` for session history.

## Related

- Upstream Plaud API toolkit: `../Plaud-toolkit/` (used only for `plaud login`)
- ASR pipeline source (for Phase 3 port): `../audio_transcribe_notes/transcribe.py`
- opencode skill: `~/.config/opencode/skills/plaud-sync/SKILL.md`
