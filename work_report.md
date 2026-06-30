# Work Report — Audio-memory-enhancer Pipeline

## Task

Build a Plaud wearable → cloud → Qwen3-ASR → Obsidian vault pipeline. Short
recordings (<15 min) use Plaud cloud transcripts and accumulate as weekly raw
memos. Long recordings (>=15 min) use local Qwen3-ASR with speaker diarization
and land as full meeting notes. A bilingual domain dictionary biases the ASR
toward correct terminology. Phase 5 will consolidate raw memos into themed
digest notes.

## Status: Phases 1-4 complete, Phase 5 (digest) is next

| Phase | Status | Highlights |
|---|---|---|
| 1 — Skeleton + Plaud cloud sync | ✅ Done | Direct REST API client, state machine, dry-run |
| 2 — Short-memo path | ✅ Done + validated | Weekly raw file accumulator, auto-trigger cloud transcription |
| 3 — Long-recording path | ✅ Done + validated | Qwen3-ASR + bilingual dictionary (354 terms) + diarization + AI clean |
| 4 — Polish | ✅ Done | Dictionary toolchain, blind comparison tooling, opencode skill, docs |
| 5 — Digest pass | ⏳ Next | `pipeline.py digest` is a stub; raw data accumulating in `Obsmem/raw/` |

## What was built

### Core pipeline (`Audio-memory-enhancer/`)

- **`plaud_sync.py`** — Direct Plaud REST API client. JWT auto-refresh.
  Endpoints: list, detail, download, temp-url, trigger_transcription,
  parse_transcript. No subprocess.
- **`state.py`** — Atomic state.json with per-recording state machine.
  Crash recovery resets in-flight → DISCOVERED. Pruning after 60 days.
- **`routing.py`** — Duration split (<15min/≥15min), ISO-week target path,
  Meeting Notes path, audio archive to `E:\Audio-arxiv\`, project registry.
- **`classify.py`** — DeepSeek `classify_long` (project + theme + summary +
  action items). Short-path classification deferred to digest pass.
- **`note_templates.py`** — Weekly raw appender (one bullet per clip) +
  Meeting Notes renderer (frontmatter + transcript + summary + actions).
- **`transcribe_local.py`** — Full Qwen3-ASR pipeline: VAD chunking,
  ForcedAligner, pyannote diarization, DeepSeek AI cleaning, GPU lock,
  `build_asr_context()` (dictionary → system message).
- **`pipeline.py`** — CLI orchestrator: `sync`, `list`, `bootstrap-projects`,
  `reprocess`, `digest` (stub).

### Domain dictionary subsystem

- **`dictionary.md`** — 354 bilingual terms (English | Chinese) extracted from
  two vibrational microscopy review papers. 708 individual terms (including
  abbreviations + Chinese translations) reaching Qwen3-ASR's context parameter.
- **`scripts/extract_paper_text.py`** — Extract text from .docx (python-docx).
- **`scripts/extract_dictionary.py`** — Feed paper texts to DeepSeek →
  structured domain glossary (150-400 terms).
- **`scripts/translate_dictionary.py`** — Batch-translate non-abbr terms to
  Chinese via DeepSeek. Idempotent (skips already-translated).
- **`build_asr_context()`** in transcribe_local.py — Parses dictionary, extracts
  all forms (en + zh + abbreviations), builds context string. `max_chars=15000`.

### Testing & comparison infrastructure

- **`scripts/blind_compare.py`** — Blind A/B comparison HTML generator + reveal.
  Aligns Qwen3 vs Plaud segments by text similarity, filters to meaning-different
  pairs via DeepSeek, generates shuffled HTML for human evaluation.
- **`scripts/compare_dict_effect.py`** — Domain term coverage analysis (how many
  dictionary terms appear in each transcript).
- **`scripts/asr_test.py`** — Standalone Qwen3-ASR runner (no vault writes).
- **`scripts/compare_transcripts.py`** — Qwen3 vs Plaud stats + WER.
- **`scripts/cdp_capture.mjs`** — Chrome DevTools Protocol network capture tool
  (used to reverse-engineer Plaud's transcription trigger API).

### Configuration

- **`config.yaml`** — `language: zh` (critical), `short_threshold_min: 15`,
  all paths, model names. Resolves relative paths against config file dir.
- **`projects.yaml`** — Bootstrapped from 6 vault project folders.
- **`run_pipeline.bat`** — Task Scheduler entry point.
- **opencode skill** at `~/.config/opencode/skills/plaud-sync/SKILL.md` —
  covers sync, dry-run, list, reprocess, bootstrap-projects, maintain dictionary.

## Validation results

### Short path (4 test clips, 0.4–0.7 min each)
- 3 processed into `Obsmem/raw/2026-W26.md` as weekly bullets ✓
- 1 auto-triggered for cloud transcription ✓
- Plaud preamble correctly stripped ✓
- Named speakers (`[朱一凡]`) and generic speakers handled ✓

### Long path (75-min Chinese-English meeting)
- 24 VAD chunks, all transcribed without timeouts ✓
- 436 segments, 2 speakers detected by pyannote ✓
- DeepSeek AI cleaning: 16 segments corrected ✓
- Domain term recognition: **399/707 (56.4%)** with dictionary vs 28/707 (4.0%) without ✓
- Key terms `photothermal`, `mid-infrared`, `脂滴`, `光热` all recognized (0 without dict) ✓

### Blind A/B comparison (3 rounds, same 75-min meeting)

| Round | Config | Plaud | Qwen3 | Ties | Qwen3 win rate |
|---|---|---|---|---|---|
| 1 | `language=auto` | 4 | 0 | 36 | 0% (repetition hallucination) |
| 2 | `language=zh` | 10 | 6 | 4 | 37.5% |
| 3 | `language=zh` + 354-term bilingual dict | 7 | 4 | 9 | 36.4% |

**Key findings:**
- `language=zh` was the critical fix (Round 1→2): eliminated repetition hallucination
- Dictionary (Round 2→3): domain term coverage 4%→56%, ties 4→9 (closed gap in
  many segments), but didn't flip the overall win rate. Plaud remains better on
  fluency and readability.
- **Bottom line**: Plaud cloud is the better transcription; Qwen3-ASR + dictionary
  is the capable local fallback for long recordings where Plaud quota is a concern.

## Key discoveries (reverse-engineered)

1. **Plaud recording IDs are 32-char hex** (not 26).
2. **Plaud detail endpoint**: `pre_download_content_list[].data_content` for inline
   transcripts (short), `content_list[].data_link` for S3 gzipped JSON (all types).
3. **Plaud cloud transcript** has a Chinese preamble before the actual transcript
   for short recordings. `parse_transcript()` strips this.
4. **Transcription trigger** (CDP-captured): `PATCH /file/<id>` + `POST /ai/transsumm/<id>`.
5. **Qwen3-ASR has no "effort" setting** — generation is hardcoded greedy. The
   `context` parameter is the only quality lever. Audio rate: 12.5 tokens/sec.
   Context budget: ~45K tokens after audio.
6. **`language=zh` is critical** — auto-detect causes repetition hallucination on
   Chinese-English code-switching content.
7. **pyannote over-splits speakers** — known issue; speaker naming/clustering
   is a future refinement.
8. **`os.rename` fails across drives** (C:→E:) on Windows. `shutil.move` handles it.

## Pending items (require user action)

1. **Set up Windows Task Scheduler** — point at `run_pipeline.bat sync`, every ~30 min.
2. **Fill in `projects.yaml`** — add aliases + keywords for the 6 registered projects.
3. **Grow `dictionary.md`** — add more papers in the field to cover remaining ~40%
   of terms. Use the `extract_dictionary.py` + `translate_dictionary.py` workflow.

## Phase 5 — Digest pass (for the next agent)

The `pipeline.py digest` command is a stub. What it should do:

1. Read accumulated `Obsmem/raw/YYYY-W##.md` weekly files
2. Use DeepSeek to classify each bullet: `time-sensitive`, `long-term`, `project-snippet`
3. Consolidate into themed `Obsmem/digest/` notes (one per theme/project)
4. Cross-reference with `projects.yaml` for routing
5. Remove `#unreviewed` tags from consolidated items

The raw inbox format is one bullet per clip:
```
- [ ] YYYY-MM-DD HH:MM — transcript text here
```

The digest pass should produce structured reference notes, not just re-bullets.
Think of it as a weekly/monthly journal consolidation — grouping related memos,
extracting action items, and creating a searchable knowledge base.

## What's deliberately deferred

- **Speaker naming/clustering** — pyannote gives generic labels (`SPEAKER_00`).
  A speaker registry mapping `SPEAKER_00 → "Name"` across recordings would
  improve readability. Future enhancement.
- **Repetition penalty monkey-patch** — Qwen3-ASR's generation is hardcoded to
  greedy (temperature=0, no beam search, no repetition_penalty). If repetition
  recurs on other recordings, a monkey-patch could add repetition_penalty.
  Not needed with `language=zh` for now.
