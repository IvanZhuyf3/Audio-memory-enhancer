"""Per-recording state management.

state.json tracks every Plaud recording this pipeline has seen so that re-runs
are idempotent and crashes are recoverable. Atomic writes (.tmp + os.replace),
mirrors the pattern in audio_transcribe_notes/monitor.py.

State machine per recording:
    DISCOVERED → DOWNLOADING → TRANSCRIBING → ROUTED → DONE
                                       │          │        │
                                       ▼          │        ▼
                                    FAILED ◄──────┘   (pruned after N days)
                  ▲
                  └── (crash recovery: TRANSCRIBING/DOWNLOADING → DISCOVERED)
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

STATE_VERSION = 1
# Any of these states are considered "in flight" and reset on crash recovery.
IN_FLIGHT_STATES = ("DOWNLOADING", "TRANSCRIBING")

STATES = ("DISCOVERED", "DOWNLOADING", "TRANSCRIBING", "ROUTED", "DONE", "FAILED")


def empty_state() -> dict:
    return {"version": STATE_VERSION, "recordings": {}, "last_sync": None}


def load_state(path: str | Path) -> dict:
    """Load state.json. Returns a fresh empty state on missing/corrupt file."""
    p = Path(path)
    if not p.exists():
        return empty_state()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if "recordings" in data:
            if "version" not in data:
                data["version"] = STATE_VERSION
            return data
    except (json.JSONDecodeError, ValueError):
        pass
    print(f"  [state] Warning: {p} corrupt or unexpected shape, starting fresh")
    return empty_state()


def save_state(state: dict, path: str | Path) -> None:
    """Atomic write: serialize to .tmp then os.replace (crash-safe)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    state["last_sync"] = datetime.now().isoformat(timespec="seconds")
    tmp.write_text(
        json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(tmp, p)


def get_recording(state: dict, plaud_id: str) -> dict | None:
    return state.get("recordings", {}).get(plaud_id)


def upsert_recording(state: dict, plaud_id: str, **fields) -> dict:
    """Insert or update a recording record. Returns the updated record."""
    rec = state.setdefault("recordings", {}).setdefault(
        plaud_id, {"state": "DISCOVERED", "first_seen": _now_iso(), "retries": 0}
    )
    rec.update(fields)
    return rec


def set_state(state: dict, plaud_id: str, new_state: str, **fields) -> dict:
    """Transition a recording to a new state with optional extra fields."""
    if new_state not in STATES:
        raise ValueError(f"Unknown state: {new_state}")
    # Don't pass state= as a kwarg (collides with the dict param). Set it after.
    rec = upsert_recording(state, plaud_id, **fields)
    rec["state"] = new_state
    if new_state == "DONE":
        rec["done_at"] = _now_iso()
    return rec


def mark_failed(state: dict, plaud_id: str, error: str, max_retries: int = 3) -> dict:
    rec = upsert_recording(state, plaud_id)
    rec["retries"] = rec.get("retries", 0) + 1
    rec["last_error"] = error
    rec["last_error_at"] = _now_iso()
    rec["state"] = "FAILED" if rec["retries"] >= max_retries else "DISCOVERED"
    return rec


def reset_in_flight(state: dict) -> int:
    """Crash recovery: move any DOWNLOADING/TRANSCRIBING back to DISCOVERED.
    Returns the count reset."""
    count = 0
    for rec in state.get("recordings", {}).values():
        if rec.get("state") in IN_FLIGHT_STATES:
            rec["state"] = "DISCOVERED"
            count += 1
    return count


def prune_done(state: dict, days: int = 60) -> int:
    """Remove DONE entries older than N days. Returns count pruned."""
    cutoff = datetime.now() - timedelta(days=days)
    to_remove = []
    for pid, rec in state.get("recordings", {}).items():
        if rec.get("state") == "DONE":
            done_at = rec.get("done_at")
            if done_at:
                try:
                    if datetime.fromisoformat(done_at) < cutoff:
                        to_remove.append(pid)
                except ValueError:
                    pass
    for pid in to_remove:
        del state["recordings"][pid]
    return len(to_remove)


def pending_recordings(state: dict) -> list[str]:
    """Recording IDs not yet DONE or permanently FAILED."""
    return [
        pid
        for pid, rec in state.get("recordings", {}).items()
        if rec.get("state") not in ("DONE",)
    ]


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


if __name__ == "__main__":
    # Smoke test: create, mutate, save, reload.
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "state.json"
        s = empty_state()
        upsert_recording(s, "abc", duration_s=1800, recorded_at="2026-06-28T14:30:00")
        set_state(s, "abc", "TRANSCRIBING")
        save_state(s, p)
        s2 = load_state(p)
        assert s2["recordings"]["abc"]["state"] == "TRANSCRIBING", s2
        n = reset_in_flight(s2)
        assert n == 1 and s2["recordings"]["abc"]["state"] == "DISCOVERED"
        print("state.py smoke test: OK")
