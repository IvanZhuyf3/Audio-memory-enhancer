# Work Report — Audio-memory-enhancer Pipeline

## Task

Build a Plaud wearable → cloud → Qwen3-ASR → Obsidian vault pipeline. Short
recordings (<15 min) use Plaud cloud transcripts and accumulate as weekly raw
memos. Long recordings (>=15 min) use local Qwen3-ASR with speaker diarization
and land as full meeting notes. Classification deferred to a future digest pass.

## Status: all core phases complete

| Phase | Status | Highlights |
|---|---|---|
| 1 — Skeleton + Plaud cloud sync | Done | Direct REST API client, state machine, dry-run |
| 2 — Short-memo path | Done + validated | Weekly raw file accumulator, auto-trigger cloud transcription |
| 3 — Long-recording path | Done + validated | Qwen3-ASR on 81-min Steve Jobs recording (26 chunks, 714 segments) |
| 4 — Polish | Partial | opencode skill deployed; Task Scheduler docs ready (user to set up) |
| 5 — Digest pass | Deferred | Waiting for accumulated raw data |

## What was built

### Core pipeline (`Audio-memory-enhancer/`)

- **`plaud_sync.py`** — Direct Plaud REST API client. Reads token from
  `~/.plaud/config.json` (written by upstream `plaud login`). Auto-refreshes
  JWT. Endpoints: list, detail, download, temp-url, trigger_transcription.
  No subprocess, no text-table parsing.
- **`state.py`** — Atomic state.json with per-recording state machine
  (DISCOVERED → DOWNLOADING → TRANSCRIBING → ROUTED → DONE + FAILED).
  Crash recovery resets in-flight → DISCOVERED. Pruning after 60 days.
- **`routing.py`** — Duration split (<15min short / >=15min long), ISO-week
  target path for short memos, Meeting Notes path for long, audio archive
  to `E:\Audio-arxiv\YYYY-MM\`, project registry (yaml load/save/bootstrap).
- **`classify.py`** — DeepSeek classifiers. `classify_long` (project + theme
  + summary + action items). Short-path classifier removed at intake (deferred
  to digest pass). Both fall back to keyword heuristics on API failure.
- **`note_templates.py`** — Weekly raw file appender (one bullet per clip,
  matching existing Obsmem convention) + Meeting Notes renderer (frontmatter
  + transcript + summary + action items + `#unreviewed` tag).
- **`transcribe_local.py`** — Full Qwen3-ASR pipeline ported from
  `audio_transcribe_notes/transcribe.py`. VAD chunking, ForcedAligner,
  pyannote diarization, DeepSeek AI cleaning, GPU lock (cross-process file
  lock to serialize vs `monitor.py`). ASR context parameter wired to
  `build_asr_context()` (dictionary terms → system message for pre-correction).
- **`pipeline.py`** — CLI orchestrator: `sync`, `list`, `bootstrap-projects`,
  `reprocess`, `digest` (stub). Handles auto-trigger of cloud transcription
  for untranscribed recordings.

### Supporting infrastructure

- **`scripts/cdp_capture.mjs`** — Chrome DevTools Protocol network capture
  tool. Used to reverse-engineer the Plaud web app's transcription trigger
  (PATCH `/file/<id>` + POST `/ai/transsumm/<id>`). Reusable for future
  endpoint discovery.
- **`scripts/debug_detail.py`** — Dumps raw Plaud API responses for
  endpoint-shape investigation.
- **`scripts/compare_transcripts.py`** — Compares Qwen3-ASR output against
  Plaud cloud transcript (ground truth). Stats, WER, side-by-side samples.
- **`config.yaml`** — All thresholds, paths, model names. Resolves relative
  paths against the config file's directory (not CWD).
- **`projects.yaml`** — Bootstrapped from 6 vault project folders. User
  needs to add aliases/keywords.
- **`run_pipeline.bat`** — Task Scheduler entry point. Activates shared
  venv, sets `PYTHONIOENCODING=utf-8`, runs `pipeline.py`.

### opencode skill (`~/.config/opencode/skills/plaud-sync/`)

Wraps the pipeline so the user can say "sync my Plaud" in opencode instead
of opening a terminal. Covers sync, dry-run, list, reprocess, bootstrap.

## Key discoveries (reverse-engineered)

1. **Plaud recording IDs are 32-char hex** (not 26). The upstream toolkit's
   CLI truncates them in display, causing detail-endpoint "file not found".
2. **Plaud detail endpoint** returns transcript in
   `pre_download_content_list[].data_content` (inline, for short recordings)
   and `content_list[].data_link` (S3 gzipped JSON, for raw/polished/summary).
   `extra_data.tranConfig` exposes language/diarization metadata.
3. **Plaud cloud transcript for short recordings** has a Chinese preamble
  ("转写内容较短，无需生成总结。音频转写原文如下：") before the actual
   transcript. `parse_transcript()` strips this, extracting `[Speaker N]`
   or `[named speaker]` tagged lines.
4. **Transcription trigger** (CDP-captured): two API calls on the consumer
   API (`api.plaud.ai`) — `PATCH /file/<id>` (set tranConfig) + 
   `POST /ai/transsumm/<id>` (fire task). No partner API keys needed.
5. **Qwen3-ASR has no "effort" setting** — generation is hardcoded to greedy
   (temperature=0, no beam search). The `context` parameter (system message)
   is the only quality lever beyond model size (0.6B vs 1.7B). Wired to
   feed dictionary terms as pre-correction.
6. **pyannote over-splits speakers** — 9 speakers detected vs Plaud's 6
   named speakers on the same recording. Known issue; speaker registry /
   clustering is a future refinement.
7. **`os.rename` fails across drives** (C:→E:) on Windows. `shutil.move`
   handles it correctly.

## Validation results

### Short path (4 test clips, 0.4–0.7 min each)
- 3 processed into `Obsmem/raw/2026-W26.md` as weekly bullets ✓
- 1 auto-triggered for cloud transcription ✓
- Plaud preamble correctly stripped ✓
- Named speakers (`[朱一凡]`) and generic speakers (`[Speaker 1]`) both handled ✓

### Long path (Steve Jobs & Bill Gates, 81 min)
- 26 VAD chunks, all transcribed without timeouts ✓
- 714 segments, 9 pyannote speakers vs Plaud's 269 segments, 6 named speakers
- DeepSeek AI cleaning: 2 corrections + 1 new dictionary term
- DeepSeek classification: theme "Apple Microsoft rivalry collaboration" + accurate summary
- Audio archived to `E:\Audio-arxiv\2026-06\aecd6667....mp3`
- Ground truth comparison report saved at `Temp/opencode/asr_comparison_report.md`

## Pending items (require user action)

1. **Grow `dictionary.md`** — currently 7 terms. Add real domain terminology
   for both ASR context (pre-correction) and DeepSeek ai_clean (post-correction)
   to work effectively.
2. **Fill in `projects.yaml`** — add aliases + keywords for the 6 registered
   projects so recordings auto-route by content.
3. **Set up Windows Task Scheduler** — point at `run_pipeline.bat sync`,
   every ~30 min. Instructions in README.
4. **Record real content** — test clips are mic-distance tests. The pipeline
   is ready for real memos and meetings.

## What's deliberately deferred

- **Phase 5 digest pass** — consolidates `Obsmem/raw/` weekly bullets into
  themed `Obsmem/digest/` notes. Doesn't make sense until there's enough
  raw data (a few weeks of memos). The raw inbox is accumulating correctly.
- **Speaker naming/clustering** — pyannote gives generic labels
  (`SPEAKER_06`). A speaker registry that maps `SPEAKER_06 → "Steve Jobs"`
  across recordings would improve readability. Future enhancement.
