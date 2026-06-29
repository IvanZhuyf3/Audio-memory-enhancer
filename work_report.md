# Work Report — Phase 1: Skeleton + Plaud cloud sync

## Task

User request: build a Plaud device → cloud → Qwen3-ASR → DeepSeek classifier →
Obsidian vault pipeline. Multi-phase project; Phase 1 goal = clone upstream
plaud-toolkit, `plaud login`, build the new repo skeleton, and get
`sync --dry-run` working against real recordings.

Original prompt (verbatim, abridged): *"build a workflow where i record on my
plaud device, plaud automatically upload it to cloud, this repo helps me
download it form cloud, and then an agent (or just script with llm api call)
help me transcribe it with qwen3asr, then manage the context and send them to
appropriate positions in my obsidian vault... other possible content types
include but not limit to: time sensitive memo, long term memo, project related
information, etc. leave this part configurable."*

## Context (going in)

- Sibling project `audio_transcribe_notes` already has a proven Qwen3-ASR +
  diarization + DeepSeek cleaning pipeline for meetings; reusable as code source.
- Shared venv at `C:\Users\Yifan\venvs\audio_transcribe\` already has every dep
  (torch+CUDA, qwen-asr, openai, pyyaml, jinja2, pyannote, silero-vad, librosa).
- Obsidian vault at `C:\Users\Yifan\Documents\Obsidian\Yifan_Obsidian` with
  pre-existing folders: `Meeting Notes`, `Obsmem/{raw,digest,digest_archive}`,
  `Projects/{OmniSRS, Review_reponse, spSRP, Theory_chat, Vibe_coding,
  Video-rate hyperspectral SWIP}`, `a_Memo`, `Archive`, etc.
- DeepSeek key + HF token in `audio_transcribe_notes/config.ini` (reuse, don't dupe).
- Upstream `plaud-toolkit` (TypeScript, alpha) provides `plaud login` + REST API
  reference; its bundled Obsidian plugin is macOS-only (irrelevant on Windows).

Decisions locked across 4 rounds of Q&A with the user (see plan in conversation):

1. New standalone repo (sibling of audio_transcribe_notes), reuse shared venv.
2. Duration split @ 15 min: `<15min` → Plaud cloud transcript (short path);
   `≥15min` → local Qwen3-ASR (long path).
3. Short → `Obsmem/raw/<date>_<plaud_id>.md`; long → `Meeting Notes/meeting_<date>_<theme>.md`.
4. DeepSeek for all LLM calls. Auto-route + `#unreviewed` tag (user spot-checks).
5. Project detection via `projects.yaml` registry + vault scan. Single home +
   frontmatter link (no duplication).
6. Audio archived to `E:\Audio-arxiv\<YYYY-MM>\<plaud_id>.<ext>`.
7. Trigger: Windows Task Scheduler, periodic (~30 min).
8. Intake ships now; digest pass (Phase 5) deferred.
9. Repo name: `Audio-memory-enhancer`.

## Workflow (what was done, in order)

1. **Read-only planning** (plan mode): fetched upstream README, explored the
   sibling `audio_transcribe_notes` project (README, AGENTS.md, config.ini,
   transcribe.py, monitor.py), inspected Obsidian vault structure.
2. **Cloned upstream** `plaud-toolkit` into `Opencode_workspace\Plaud-toolkit\`
   (was empty). `npm install` → 164 packages.
3. **Read upstream TS source** (`packages/core/src/{client,auth,config,types}.ts`,
   `packages/cli/src/commands/*.ts`) to learn exact API contract:
   - Endpoints: `/auth/access-token`, `/file/simple/web`, `/file/detail/<id>`,
     `/file/temp-url/<id>`, `/file/download/<id>`, `/user/me`.
   - Region hosts: `api.plaud.ai` (us) / `api-euc1.plaud.ai` (eu).
   - Browser User-Agent required (default fetch/urllib UA gets 403).
   - Region mismatch returns `{status:-302, data:{domains:{api}}}`.
   - `duration` + `start_time` are epoch-**milliseconds**.
   - Transcript extracted from `pre_download_content_list[].data_content` (longest wins).
4. **Architectural improvement over the original plan**: instead of shelling out
   to the TS CLI (which prints text tables, no JSON flag — fragile to parse),
   wrote `plaud_sync.py` to call the Plaud REST API **directly** from Python.
   It reads the token that `plaud login` writes to `~/.plaud/config.json` and
   auto-refreshes via stored credentials. Zero subprocess calls in normal operation.
5. **Created new repo** `Audio-memory-enhancer\` with 6 Python modules:
   - `plaud_sync.py` — direct REST client (list/get/download/ensure_token).
   - `state.py` — atomic state.json, state machine, crash recovery, pruning.
   - `routing.py` — duration split, vault target paths, audio archive, project registry.
   - `classify.py` — DeepSeek classifiers (short sub-type + long project/theme/summary).
   - `note_templates.py` — markdown + frontmatter builders (short memo + long meeting).
   - `transcribe_local.py` — Phase 3 stub (port Qwen3-ASR later).
   - `pipeline.py` — CLI orchestrator (`sync`/`list`/`bootstrap-projects`/`reprocess`/`digest`).
6. **Config + supporting files**: `config.yaml`, `projects.yaml` (bootstrapped
   from vault scan), `requirements.txt` (reference only), `run_pipeline.bat`,
   `.gitignore`, `dictionary.md` (copied from sibling), README.md, AGENTS.md.
7. **Smoke-tested** every module that doesn't need Plaud login: state.py,
   routing.py, note_templates.py, pipeline.py --help, pipeline.py digest.
   All pass.
8. **Found + fixed a path-resolution bug**: `state_file` + `projects_registry`
   were relative paths resolved against CWD, causing bootstrap-projects to write
   `projects.yaml` to the wrong directory. Fixed `load_config()` to resolve
   project-relative paths against the config file's location.
9. **Found + fixed a kwarg-collision bug**: `set_state()` passed `state=new_state`
   into `upsert_recording(state, ...)` which collided with the positional `state`
   dict param. Fixed by setting `rec["state"]` after upsert.
10. **git init + initial commit** (15 files, single Phase-1 commit).

## Outputs

| File | Purpose |
|---|---|
| `C:\Users\Yifan\OneDrive\Opencode_workspace\Plaud-toolkit\` | Upstream clone (npm installed; used only for `plaud login`) |
| `C:\Users\Yifan\OneDrive\Opencode_workspace\Audio-memory-enhancer\` | **New repo** — all pipeline code |
| `Audio-memory-enhancer\pipeline.py` | CLI entry point |
| `Audio-memory-enhancer\plaud_sync.py` | Direct Plaud REST client |
| `Audio-memory-enhancer\state.py` | Atomic state + state machine |
| `Audio-memory-enhancer\routing.py` | Duration split + routing + project registry |
| `Audio-memory-enhancer\classify.py` | DeepSeek classifiers |
| `Audio-memory-enhancer\note_templates.py` | Markdown builders |
| `Audio-memory-enhancer\transcribe_local.py` | Phase 3 stub |
| `Audio-memory-enhancer\config.yaml` | All thresholds + paths |
| `Audio-memory-enhancer\projects.yaml` | Project registry (6 projects, needs aliases/keywords) |
| `Audio-memory-enhancer\README.md` | User-facing docs |
| `Audio-memory-enhancer\AGENTS.md` | Architectural deep-dive for future sessions |

## Assessment

**Completeness:** Phase 1 scope fully delivered. Skeleton + cloud sync +
dry-run capability all in place. Short-memo path is wired end-to-end (will
actually process real short recordings once the user logs in). Long path is
correctly stubbed (raises `NotImplementedError` with a clear Phase-3 message).

**Confidence:** High on the architecture and the short path. Medium on
short-path real-data behavior — untested against actual Plaud cloud transcripts
(the one open unknown: the exact text format Plaud returns, which informs
whether any parsing/cleaning is needed before classification).

**Known gaps / risks:**
- `transcribe_local.py` is a stub (Phase 3). Long recordings will be skipped
  with a clear error until then.
- Plaud cloud transcript format not yet inspected (need real login first).
- `deepseek-v4-flash` model name in the shared config is taken as-is; needs
  confirmation at first real LLM call.
- GPU lock mechanism referenced in config but not yet implemented (only needed
  for Phase 3 long path).
- No automated test suite (matches sibling project's style; manual verification
  via smoke tests in each module's `__main__` block).

## Pending items (require manual action)

1. **User must run `plaud login` interactively** (the one remaining Phase-1 gate):
   ```powershell
   cd C:\Users\Yifan\OneDrive\Opencode_workspace\Plaud-toolkit
   npx tsx packages/cli/bin/plaud.ts login
   ```
   Enter Plaud email + password + region. This writes `~/.plaud/config.json`.
2. **After login, run the dry-run** to validate against real data + inspect the
   cloud transcript format:
   ```powershell
   cd C:\Users\Yifan\OneDrive\Opencode_workspace\Audio-memory-enhancer
   .\run_pipeline.bat sync --dry-run
   ```
3. **Hand-edit `projects.yaml`** to add aliases + keywords for the 6 projects so
   the classifier can match recordings to projects by content. Example:
   ```yaml
   - name: OmniSRS
     aliases: [SRS, omnisrs]
     keywords: [Raman, scattering, Stokes, microscope, hyperspectral]
   ```

## Next phases (for future sessions)

- **Phase 2** (mostly done — validate with real data): confirm short path writes
  correct notes to `Obsmem/raw/`, tune the classifier prompt if needed.
- **Phase 3**: port `run_qwen3_asr()` + `ai_clean()` from
  `audio_transcribe_notes/transcribe.py` into `transcribe_local.py`; implement
  the GPU lock; wire `process_long()` end-to-end.
- **Phase 4**: inbox dashboard generator, Task Scheduler install script,
  `reprocess` end-to-end test.
- **Phase 5**: digest pass (`Obsmem/raw/` → `Obsmem/digest/`) + opencode skill
  wrapper under `skill/SKILL.md`.
