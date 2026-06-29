"""Audio-memory-enhancer — Plaud → transcribe → classify → Obsidian pipeline.

CLI entry point. Run via run_pipeline.bat (which activates the shared venv):

    run_pipeline.bat sync --dry-run       # list what would be processed
    run_pipeline.bat sync                 # full run
    run_pipeline.bat sync --only <id>     # one recording
    run_pipeline.bat list                 # show Plaud recordings + state
    run_pipeline.bat bootstrap-projects   # rescan vault Projects/ folder
    run_pipeline.bat reprocess <id>       # delete old note, re-run
    run_pipeline.bat digest               # (stub, Phase 5)

State machine + atomic state.json (see state.py). Pipeline is idempotent and
crash-recoverable: any recording stuck in DOWNLOADING/TRANSCRIBING on startup
is reset to DISCOVERED.
"""

from __future__ import annotations

import argparse
import configparser
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

import plaud_sync
import state as state_mod
import routing
import classify
import note_templates
import transcribe_local

CONFIG_PATH = Path(__file__).parent / "config.yaml"


# ── Config loading ──────────────────────────────────────────────────

def load_config(path: str | Path = CONFIG_PATH) -> dict:
    """Load config.yaml. Returns a flat dict with project-relative paths resolved.

    Relative paths (state_file, projects_registry) are resolved against the
    config file's directory so the pipeline works regardless of CWD.
    """
    p = Path(path).resolve()
    if not p.exists():
        raise RuntimeError(f"config.yaml not found at {p}")
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    project_root = p.parent
    # Resolve project-relative paths so CWD never matters.
    for key in ("state_file", "projects_registry"):
        val = cfg.get(key)
        if val:
            resolved = (project_root / val).resolve() if not Path(val).is_absolute() else Path(val)
            cfg[key] = str(resolved)
    return cfg


def _vault(cfg: dict) -> Path:
    return Path(cfg["vault"]["root"])


# ── Secrets (from shared audio_transcribe_notes/config.ini) ─────────

def load_secrets(cfg: dict) -> dict:
    """Read hf_token + deepseek_api_key from the shared config.ini."""
    src = Path(cfg["secrets_source"])
    if not src.exists():
        return {"hf_token": "", "deepseek_api_key": "", "deepseek_model": "deepseek-v4-flash"}
    cp = configparser.ConfigParser()
    cp.read(src, encoding="utf-8")
    d = cp["defaults"] if "defaults" in cp else {}
    return {
        "hf_token": d.get("hf_token", "").strip(),
        "deepseek_api_key": d.get("deepseek_api_key", "").strip(),
        "deepseek_model": d.get("deepseek_model", "deepseek-v4-flash").strip(),
    }


# ── Per-recording processing ────────────────────────────────────────

def _recording_duration_s(rec: dict) -> float:
    """Plaud 'duration' is epoch-ms; return seconds."""
    return (rec.get("duration") or 0) / 1000


def _recording_recorded_at_epoch_ms(rec: dict) -> int:
    return rec.get("start_time") or 0


def _recording_summary(rec: dict, path_decision: str) -> str:
    """One-line summary for logging."""
    dur_min = _recording_duration_s(rec) / 60
    dt = datetime.fromtimestamp(
        _recording_recorded_at_epoch_ms(rec) / 1000, tz=timezone.utc
    ).astimezone()
    date = dt.strftime("%Y-%m-%d %H:%M")
    flags = ("T" if rec.get("is_trans") else "-") + ("S" if rec.get("is_summary") else "-")
    fname = (rec.get("filename") or rec.get("id"))[:40]
    return f"{date}  {dur_min:6.1f}m  [{flags}]  {path_decision:5s}  {fname}"


def process_short(rec: dict, cfg: dict, state: dict, dry_run: bool = False) -> str:
    """Process a short recording: append one bullet to the week's raw file.

    No per-recording file, no LLM classification (that's the digest pass's job).
    Matches the existing Obsmem/raw/YYYY-W##.md convention: one bullet per clip.
    Returns a status string. Writes nothing if dry_run.
    """
    plaud_id = rec["id"]
    recorded_at = _recording_recorded_at_epoch_ms(rec)
    duration_s = _recording_duration_s(rec)

    # If Plaud hasn't transcribed yet, skip gracefully (leave DISCOVERED for retry).
    if not rec.get("is_trans"):
        return "SKIP:not-yet-transcribed"

    state_mod.set_state(state, plaud_id, "TRANSCRIBING",
                        path_chosen="short", transcript_source="plaud-cloud")
    detail = plaud_sync.get_recording(plaud_id)
    raw_transcript = detail.get("transcript") or ""
    if not raw_transcript.strip():
        return "SKIP:no-inline-transcript"

    parsed = plaud_sync.parse_transcript(raw_transcript)
    transcript_text = parsed["text"]
    if not transcript_text:
        return "SKIP:empty-transcript-after-parse"

    weekly_path = routing.weekly_target_path(
        _vault(cfg), cfg["vault"]["memo_raw_folder"], recorded_at
    )
    if dry_run:
        return f"would append to {weekly_path.name} ({len(transcript_text)} chars, {parsed['speakers']} speaker(s))"

    # Append the bullet (no classification — digest pass handles that later).
    bullet = note_templates.append_clip_to_weekly(
        weekly_path, recorded_at, transcript_text
    )

    # Archive audio (non-fatal on failure — the bullet is already written).
    audio_dest = routing.audio_archive_path(
        cfg["audio_archive"]["root"], plaud_id, recorded_at, "mp3"
    )
    if not audio_dest.exists():
        try:
            cache_dir = Path(cfg["state_file"]).parent / "cache"
            cache_path = cache_dir / f"{plaud_id}.mp3"
            plaud_sync.download_audio(plaud_id, cache_path)
            routing.move_to_archive(cache_path, audio_dest)
        except Exception as e:
            print(f"  [{plaud_id}] audio archive skipped: {e}")

    state_mod.set_state(
        state, plaud_id, "ROUTED",
        vault_path=str(weekly_path.relative_to(_vault(cfg)).with_suffix("")),
        archive_path=str(audio_dest),
    )
    return f"→ {weekly_path.relative_to(_vault(cfg))}  [{len(transcript_text)} chars]"


def process_long(rec: dict, cfg: dict, state: dict, dry_run: bool = False) -> str:
    """Process a long recording via local Qwen3-ASR → Meeting Notes/.

    Downloads audio, acquires the GPU lock, transcribes with diarization,
    classifies (project + theme + summary + action items), writes the note,
    and archives the audio.
    """
    plaud_id = rec["id"]
    recorded_at = _recording_recorded_at_epoch_ms(rec)
    duration_s = _recording_duration_s(rec)

    if dry_run:
        return f"would transcribe locally ({duration_s/60:.1f}m → Meeting Notes/)"

    project_root = Path(__file__).parent
    cache_dir = project_root / "cache"
    cache_dir.mkdir(exist_ok=True)
    audio_path = cache_dir / f"{plaud_id}.mp3"

    # 1. Download audio (reuse cache if present).
    state_mod.set_state(state, plaud_id, "DOWNLOADING",
                        path_chosen="long", transcript_source="local-qwen3-asr")
    if not audio_path.exists():
        print(f"  [{plaud_id}] downloading audio...")
        plaud_sync.download_audio(plaud_id, audio_path)

    # 2. Transcribe (GPU lock + ASR + diarization + AI clean).
    state_mod.set_state(state, plaud_id, "TRANSCRIBING")
    secrets = load_secrets(cfg)
    tcfg = cfg["transcription"]
    result = transcribe_local.transcribe(
        audio_path,
        language=tcfg.get("language", "auto"),
        qwen_model=tcfg.get("qwen_model", "Qwen/Qwen3-ASR-1.7B"),
        forced_aligner=tcfg.get("forced_aligner", "Qwen/Qwen3-ForcedAligner-0.6B"),
        device=tcfg.get("device", "cuda"),
        hf_token=secrets["hf_token"],
        deepseek_api_key=secrets["deepseek_api_key"],
        deepseek_model=secrets["deepseek_model"],
        dictionary_path=project_root / "dictionary.md",
        gpu_lock=cfg.get("gpu_lock"),
        gpu_lock_timeout_s=cfg.get("gpu_lock_timeout_s", 3600),
        log_dir=cache_dir,
    )

    if not result["segments"]:
        raise RuntimeError("transcription produced no segments")

    # 3. Classify (project + theme + summary + action items).
    projects = routing.load_projects(cfg["projects_registry"])
    classification = classify.classify_long(
        result["transcript_text"], rec, projects,
        secrets_source=cfg["secrets_source"],
        temperature=cfg["llm"].get("temperature_classify", 0.0),
    )

    # 4. Render + write note.
    target = routing.long_target_path(
        _vault(cfg), cfg["vault"]["meeting_notes_folder"], recorded_at,
        classification["theme"],
    )
    md = note_templates.render_long_meeting(
        plaud_id=plaud_id,
        recorded_at_epoch_ms=recorded_at,
        duration_s=duration_s,
        theme=classification["theme"],
        transcript_markdown=result["transcript_markdown"],
        summary=classification["summary"],
        action_items=classification["action_items"],
        project=classification["project"],
        speakers=result["speakers"] or None,
        unreviewed_tag=cfg["vault"].get("unreviewed_tag", "unreviewed"),
    )
    _atomic_write(target, md)

    # 5. Archive audio.
    audio_dest = routing.audio_archive_path(
        cfg["audio_archive"]["root"], plaud_id, recorded_at, "mp3"
    )
    try:
        routing.move_to_archive(audio_path, audio_dest)
    except Exception as e:
        print(f"  [{plaud_id}] audio archive skipped: {e}")

    # 6. Update state.
    state_mod.set_state(
        state, plaud_id, "ROUTED",
        vault_path=str(target.relative_to(_vault(cfg)).with_suffix("")),
        archive_path=str(audio_dest),
        classification=classification,
        speakers=result["speakers"],
        language=result.get("language"),
    )
    rel = target.relative_to(_vault(cfg))
    proj = f" [{classification['project']}]" if classification.get("project") else ""
    return f"→ {rel}  ({len(result['segments'])} segs, {result['speakers']} speakers){proj}"


# ── Main sync ───────────────────────────────────────────────────────

def cmd_sync(args, cfg: dict) -> int:
    state_path = Path(cfg.get("state_file", "state.json"))
    state = state_mod.load_state(state_path)

    # Crash recovery.
    reset = state_mod.reset_in_flight(state)
    if reset:
        print(f"[recovery] reset {reset} in-flight recording(s) to DISCOVERED")

    # Prune old DONE entries.
    pruned = state_mod.prune_done(state, cfg.get("prune_done_after_days", 60))
    if pruned:
        print(f"[cleanup] pruned {pruned} old DONE entries")

    # Pull recordings from Plaud cloud.
    print("[plaud] listing recordings...")
    try:
        recordings = plaud_sync.list_recordings()
    except Exception as e:
        print(f"[error] failed to list recordings: {e}")
        return 1
    print(f"[plaud] {len(recordings)} recording(s) on cloud")

    # Diff against state — register new ones.
    new_count = 0
    for rec in recordings:
        if rec.get("id") and not state_mod.get_recording(state, rec["id"]):
            state_mod.upsert_recording(
                state, rec["id"],
                duration_s=_recording_duration_s(rec),
                recorded_at=datetime.fromtimestamp(
                    _recording_recorded_at_epoch_ms(rec) / 1000, tz=timezone.utc
                ).isoformat(),
                plaud_filename=rec.get("filename"),
            )
            new_count += 1
    print(f"[state] {new_count} new recording(s) registered")

    # Filter to the ones we should process this run.
    threshold_min = cfg["transcription"]["short_threshold_min"]
    if args.only:
        recordings = [r for r in recordings if r.get("id") == args.only]
        if not recordings:
            print(f"[error] no recording with id {args.only}")
            return 1
    elif args.all:
        # process every non-DONE recording (re-process DISCOVERED + FAILED)
        target_ids = set(state_mod.pending_recordings(state))
        recordings = [r for r in recordings if r.get("id") in target_ids]
    else:
        # default: only DISCOVERED (new) ones
        recordings = [
            r for r in recordings
            if (rec_state := state_mod.get_recording(state, r.get("id", "")))
            and rec_state.get("state") == "DISCOVERED"
        ]
    if args.limit:
        recordings = recordings[: args.limit]

    # Print plan (always — dry run or not).
    print()
    print(f"{'ID':<34} {'Date':<17} {'Dur':>7}  {'Path':<5} {'Filename'}")
    print("-" * 96)
    for rec in recordings:
        path_decision = routing.decide_path(
            _recording_duration_s(rec), threshold_min
        )
        rid = rec.get("id") or ""
        print(f"{rid:<34} " + _recording_summary(rec, path_decision))
    print()

    if args.dry_run:
        print(f"[dry-run] would process {len(recordings)} recording(s); no writes.")
        state_mod.save_state(state, state_path)  # persist newly-discovered
        return 0

    if not recordings:
        print("[sync] nothing to process; all caught up.")
        state_mod.save_state(state, state_path)
        return 0

    # Process each.
    ok, skipped, failed = 0, 0, 0
    for rec in recordings:
        plaud_id = rec["id"]
        duration_s = _recording_duration_s(rec)
        path_decision = routing.decide_path(duration_s, threshold_min)
        try:
            if path_decision == "short":
                msg = process_short(rec, cfg, state)
            else:
                msg = process_long(rec, cfg, state)
            # SKIP return → leave as DISCOVERED so next sync retries (no DONE).
            if msg.startswith("SKIP:"):
                reason = msg.split(":", 1)[1]
                print(f"  [skip] {plaud_id}: {reason} (will retry next sync)")
                # Reset to DISCOVERED so it's picked up again next time.
                state_mod.set_state(state, plaud_id, "DISCOVERED")
                skipped += 1
            else:
                print(f"  [ok] {plaud_id}: {msg}")
                state_mod.set_state(state, plaud_id, "DONE")
                ok += 1
        except Exception as e:
            print(f"  [FAIL] {plaud_id}: {e}")
            state_mod.mark_failed(state, plaud_id, str(e), cfg.get("max_retries", 3))
            failed += 1
        # Persist after each recording (crash-safe progress).
        state_mod.save_state(state, state_path)

    print(f"\n[sync] done: {ok} ok, {skipped} skipped, {failed} failed")
    return 0 if failed == 0 else 2


# ── Other commands ──────────────────────────────────────────────────

def cmd_list(args, cfg: dict) -> int:
    try:
        recordings = plaud_sync.list_recordings()
    except Exception as e:
        print(f"[error] {e}")
        return 1
    state = state_mod.load_state(cfg.get("state_file", "state.json"))
    threshold_min = cfg["transcription"]["short_threshold_min"]
    print(f"{'ID':<34} {'State':<13} {'Date':<17} {'Dur':>7}  {'Path':<5} {'Filename'}")
    print("-" * 106)
    for rec in recordings:
        rid = rec.get("id", "")
        st = (state_mod.get_recording(state, rid) or {}).get("state", "new")
        path_decision = routing.decide_path(_recording_duration_s(rec), threshold_min)
        print(f"{rid:<34} {st:<13} " + _recording_summary(rec, path_decision))
    print(f"\n{len(recordings)} recording(s)")
    return 0


def cmd_bootstrap_projects(args, cfg: dict) -> int:
    yaml_path = cfg["projects_registry"]
    existing = routing.load_projects(yaml_path)
    before = len(existing)
    existing = routing.bootstrap_projects_from_vault(
        _vault(cfg) / cfg["vault"]["projects_folder"], existing
    )
    routing.save_projects(yaml_path, existing)
    print(f"[bootstrap] {len(existing)} projects ({len(existing) - before} new) → {yaml_path}")
    for p in existing:
        print(f"  - {p['name']}  aliases={p.get('aliases')}  keywords={p.get('keywords')}")
    print("\nEdit projects.yaml to add aliases/keywords for better matching.")
    return 0


def cmd_reprocess(args, cfg: dict) -> int:
    """Reset a recording to DISCOVERED so the next sync reprocesses it.

    Does NOT delete the old note — for weekly raw files that would destroy
    other clips' bullets, and for Meeting Notes the user may have edited.
    Manually remove the old bullet/note if you want a clean re-render.
    """
    plaud_id = args.id
    state_path = Path(cfg.get("state_file", "state.json"))
    state = state_mod.load_state(state_path)
    rec = state_mod.get_recording(state, plaud_id)
    if not rec:
        print(f"[error] {plaud_id} not in state; run `sync --dry-run` first to discover it.")
        return 1
    # Reset state so sync picks it up. Leave old note in place.
    rec["state"] = "DISCOVERED"
    rec["retries"] = 0
    rec.pop("vault_path", None)
    rec.pop("done_at", None)
    state_mod.save_state(state, state_path)
    print(f"[reprocess] {plaud_id} reset to DISCOVERED.")
    if rec.get("vault_path"):
        print(f"           old note left in place: {rec.get('vault_path')}  (edit/remove manually if needed)")
    print(f"           Run `sync --only {plaud_id}` next.")
    return 0


def cmd_digest(args, cfg: dict) -> int:
    print("[digest] Phase 5 (deferred). Consolidates Obsmem/raw/ → Obsmem/digest/.")
    print("         Not implemented yet. Memos accumulate in Obsmem/raw/ until then.")
    return 0


# ── Helpers ─────────────────────────────────────────────────────────

def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    import os
    os.replace(tmp, path)


# ── CLI ─────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pipeline",
        description="Plaud → transcribe → classify → Obsidian pipeline.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_sync = sub.add_parser("sync", help="Pull new recordings and process them.")
    p_sync.add_argument("--dry-run", action="store_true", help="Show plan, write nothing.")
    p_sync.add_argument("--only", metavar="ID", help="Process one recording by Plaud ID.")
    p_sync.add_argument("--all", action="store_true",
                        help="Process all non-DONE (including FAILED) recordings.")
    p_sync.add_argument("--limit", type=int, default=None,
                        help="Process at most N recordings this run.")
    p_sync.set_defaults(func=cmd_sync)

    p_list = sub.add_parser("list", help="List Plaud recordings + state.")
    p_list.set_defaults(func=cmd_list)

    p_boot = sub.add_parser("bootstrap-projects", help="Scan vault Projects/ → projects.yaml.")
    p_boot.set_defaults(func=cmd_bootstrap_projects)

    p_re = sub.add_parser("reprocess", help="Reset one recording to DISCOVERED.")
    p_re.add_argument("id", help="Plaud recording ID.")
    p_re.set_defaults(func=cmd_reprocess)

    p_dig = sub.add_parser("digest", help="(Phase 5) Consolidate raw memos into digest notes.")
    p_dig.set_defaults(func=cmd_digest)

    args = parser.parse_args(argv)
    cfg = load_config()
    return args.func(args, cfg)


if __name__ == "__main__":
    sys.exit(main())
