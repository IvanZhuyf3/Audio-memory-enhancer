"""Routing: duration-based path decision, vault target paths, audio archive.

The duration split decides short (Plaud cloud transcript) vs. long (local Qwen3-ASR).
Short memos land in Obsmem/raw/; long recordings land in Meeting Notes/.
Audio is archived to E:/Audio-arxiv/<YYYY-MM>/ regardless of path.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import yaml  # PyYAML — already in the shared venv


# ── Duration-based path decision ────────────────────────────────────

def decide_path(duration_s: float, threshold_min: float = 15.0) -> str:
    """Return 'short' (<threshold) or 'long' (>=threshold)."""
    return "short" if duration_s < threshold_min * 60 else "long"


# ── Vault target paths ──────────────────────────────────────────────

def _sanitize_filename(name: str, fallback: str = "untitled") -> str:
    """Make a string safe for use as a filename. Truncate to 80 chars."""
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return (name[:80] or fallback)


def _local_dt(epoch_ms: int) -> datetime:
    """Plaud start_time is epoch-ms UTC; render in local time for filenames."""
    return datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc).astimezone()


def short_target_path(
    vault_root: str | Path,
    memo_raw_folder: str,
    recorded_at_epoch_ms: int,
    plaud_id: str,
) -> Path:
    """Obsmem/raw/<YYYY-MM-DD_HHhMM>_<plaud_id>.md — sortable + unique via plaud_id."""
    dt = _local_dt(recorded_at_epoch_ms)
    stamp = dt.strftime("%Y-%m-%d_%Hh%M")
    name = f"{stamp}_{plaud_id}.md"
    return Path(vault_root) / memo_raw_folder / name


def long_target_path(
    vault_root: str | Path,
    meeting_notes_folder: str,
    recorded_at_epoch_ms: int,
    theme: str,
) -> Path:
    """Meeting Notes/meeting_<YYYY-MM-DD_HHhMM>_<theme>.md (matches existing pipeline)."""
    dt = _local_dt(recorded_at_epoch_ms)
    stamp = dt.strftime("%Y-%m-%d_%Hh%M")
    safe_theme = _sanitize_filename(theme, "meeting")
    name = f"meeting_{stamp}_{safe_theme}.md"
    return Path(vault_root) / meeting_notes_folder / name


# ── Audio archive ───────────────────────────────────────────────────

def audio_archive_path(
    archive_root: str | Path,
    plaud_id: str,
    recorded_at_epoch_ms: int,
    ext: str,
) -> Path:
    """E:/Audio-arxiv/<YYYY-MM>/<plaud_id>.<ext> (by_month layout)."""
    dt = _local_dt(recorded_at_epoch_ms)
    month = dt.strftime("%Y-%m")
    ext = ext.lstrip(".")
    return Path(archive_root) / month / f"{plaud_id}.{ext}"


def move_to_archive(src_path: str | Path, archive_path: str | Path) -> Path:
    """Move an audio file to its archive location. Returns the archive path.

    Uses shutil.move (not os.rename) because the archive is often on a different
    drive (C: → E:) and rename fails across volumes with WinError 17.
    """
    import shutil
    src = Path(src_path)
    dst = Path(archive_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()  # idempotent overwrite
    shutil.move(str(src), str(dst))
    return dst


# ── Project registry ────────────────────────────────────────────────

def load_projects(yaml_path: str | Path) -> list[dict]:
    """Load projects.yaml. Returns list of {name, aliases[], keywords[]}."""
    p = Path(yaml_path)
    if not p.exists():
        return []
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return data.get("projects", []) or []


def save_projects(yaml_path: str | Path, projects: list[dict]) -> None:
    """Write projects.yaml."""
    p = Path(yaml_path)
    p.write_text(
        yaml.safe_dump(
            {"projects": projects},
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        ),
        encoding="utf-8",
    )


def bootstrap_projects_from_vault(
    vault_projects_folder: str | Path, existing: list[dict] | None = None
) -> list[dict]:
    """Scan vault Projects/ folder for subfolders, merge into the project list.
    Existing project names are preserved (their aliases/keywords untouched).
    """
    existing = existing or []
    existing_names = {p["name"] for p in existing}
    folder = Path(vault_projects_folder)
    if not folder.is_dir():
        return existing
    for sub in sorted(folder.iterdir()):
        if sub.is_dir() and not sub.name.startswith(".") and sub.name != "z_Attachments":
            if sub.name not in existing_names:
                existing.append(
                    {"name": sub.name, "aliases": [], "keywords": []}
                )
                existing_names.add(sub.name)
    return existing


if __name__ == "__main__":
    # Smoke test
    print("decide_path(1800, 15):", decide_path(1800, 15))  # 30min → long
    print("decide_path(600, 15):", decide_path(600, 15))    # 10min → short
    print(
        "short_target:",
        short_target_path("VAULT", "Obsmem/raw", 1751110200000, "abc123"),
    )
    print(
        "long_target:",
        long_target_path("VAULT", "Meeting Notes", 1751110200000, "Project Standup"),
    )
    print(
        "archive:",
        audio_archive_path("E:/Audio-arxiv", "abc123", 1751110200000, "mp3"),
    )
