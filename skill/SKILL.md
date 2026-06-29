---
name: plaud-sync
description: Sync and process Plaud wearable recordings into the Obsidian vault. Use when the user wants to pull new recordings from Plaud cloud, transcribe memos or meetings, check recording status, or reprocess a specific clip. Short clips (<15min) use Plaud cloud transcript and land in Obsmem/raw/ weekly files. Long recordings (>=15min) use local Qwen3-ASR with speaker diarization and land in Meeting Notes/. Untranscribed recordings are auto-triggered for cloud transcription.
---

## Rules

- Pipeline lives at `C:\Users\Yifan\OneDrive\Opencode_workspace\Audio-memory-enhancer\`
- Always set `PYTHONIOENCODING=utf-8` before running (Windows GBK breaks CJK output)
- The shared venv (`C:\Users\Yifan\venvs\audio_transcribe\`) is pre-configured — never pip install into it
- The pipeline is idempotent + crash-recoverable: `state.json` tracks every recording, so re-running `sync` is always safe
- Short path (<15 min): Plaud cloud transcript → `Obsmem/raw/YYYY-W##.md` (one bullet per clip, no classification)
- Long path (>=15 min): local Qwen3-ASR + pyannote diarization + DeepSeek cleaning → `Meeting Notes/meeting_<date>_<theme>.md`
- Audio archived to `E:\Audio-arxiv\YYYY-MM\<plaud_id>.mp3`
- If a recording lacks a cloud transcript (`is_trans=False`), the pipeline auto-triggers one via the Plaud API and defers to the next sync
- Long recordings take ~30 min GPU time per hour of audio — warn the user before processing one

## Workflow

### Default: sync (pull + process new recordings)

```
& "C:\Users\Yifan\OneDrive\Opencode_workspace\Audio-memory-enhancer\run_pipeline.bat" sync
```

After running, summarize the output for the user:
- How many recordings processed (ok), skipped (pending cloud transcription), failed
- Which vault files were written or appended to (e.g. `Obsmem/raw/2026-W26.md`, `Meeting Notes/meeting_..._theme.md`)
- Any recordings that were auto-triggered for cloud transcription

### Preview without writing (dry-run)

```
& "C:\Users\Yifan\OneDrive\Opencode_workspace\Audio-memory-enhancer\run_pipeline.bat" sync --dry-run
```

### List all recordings + pipeline state

```
& "C:\Users\Yifan\OneDrive\Opencode_workspace\Audio-memory-enhancer\run_pipeline.bat" list
```

The table shows: ID (32-char hex), state (DISCOVERED/DONE/FAILED), date, duration, path (short/long), filename. Use `[TS]` vs `[--]` to tell the user which recordings have cloud transcripts.

### Reprocess a specific recording

```
& "C:\Users\Yifan\OneDrive\Opencode_workspace\Audio-memory-enhancer\run_pipeline.bat" reprocess <plaud_id>
& "C:\Users\Yifan\OneDrive\Opencode_workspace\Audio-memory-enhancer\run_pipeline.bat" sync --only <plaud_id>
```

Reprocess resets state to DISCOVERED but does NOT delete old vault notes (the user may have edited them). For weekly raw files, the reprocessed clip appends as a new bullet — remove the old bullet manually if needed.

### Rescan vault for new project folders

```
& "C:\Users\Yifan\OneDrive\Opencode_workspace\Audio-memory-enhancer\run_pipeline.bat" bootstrap-projects
```

Merges any new `Projects/` subfolders into `projects.yaml`. The user should then hand-edit aliases/keywords for better matching.
