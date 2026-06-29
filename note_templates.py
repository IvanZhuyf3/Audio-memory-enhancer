"""Markdown note templates for short memos and long meetings.

Pure string builders — no LLM, no GPU. Produce the full note body the pipeline
writes to the Obsidian vault. Frontmatter is hand-emitted (key: value) to match
Obsidian/YAML conventions and the upstream plaud-toolkit sync.ts style.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path


def _local_dt(epoch_ms: int) -> datetime:
    return datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc).astimezone()


def _iso_local(epoch_ms: int) -> str:
    return _local_dt(epoch_ms).isoformat(timespec="seconds")


def _atomic_write_text(path: Path, content: str) -> None:
    """Atomic write: serialize to .tmp then os.replace (crash-safe)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


# ── Short memo — weekly accumulator (Obsmem/raw/YYYY-W##.md) ────────

WEEKLY_HEADER = "# {year}-W{week:02d}\n"

WEEKLY_BULLET_TEMPLATE = "- [ ] {ts} — {transcript}\n"


def weekly_file_stamp(recorded_at_epoch_ms: int) -> tuple[str, str, str]:
    """Return (iso_year_week_str 'YYYY-W##', header_line, bullet_timestamp 'YYYY-MM-DD HH:MM')."""
    dt = _local_dt(recorded_at_epoch_ms)
    iso_year, iso_week, _ = dt.isocalendar()
    year_week = f"{iso_year}-W{iso_week:02d}"
    header = WEEKLY_HEADER.format(year=iso_year, week=iso_week)
    bullet_ts = dt.strftime("%Y-%m-%d %H:%M")
    return year_week, header, bullet_ts


def append_clip_to_weekly(
    weekly_path: str | Path,
    recorded_at_epoch_ms: int,
    transcript: str,
) -> str:
    """Append one bullet line for a clip to the weekly raw file.

    Matches the existing Obsmem convention:
        - [ ] 2026-06-28 21:46 — <transcript>

    Creates the file with a `# YYYY-W##` header if missing. The clip is always
    `[ ]` (unchecked) at intake — the digest pass flips it to `[v]`.

    Returns the text of the bullet line that was appended (for logging).
    """
    p = Path(weekly_path)
    _year_week, header, bullet_ts = weekly_file_stamp(recorded_at_epoch_ms)
    # Collapse the transcript to a single line (existing convention: one line per memo).
    one_line = " ".join(transcript.split())
    bullet = WEEKLY_BULLET_TEMPLATE.format(ts=bullet_ts, transcript=one_line)

    if p.exists():
        content = p.read_text(encoding="utf-8")
        # Append + ensure single trailing newline before the bullet.
        if not content.endswith("\n"):
            content += "\n"
        content += bullet
    else:
        content = header + "\n" + bullet

    _atomic_write_text(p, content)
    return bullet.rstrip()


# ── Long meeting (Meeting Notes/) ───────────────────────────────────

def render_long_meeting(
    *,
    plaud_id: str,
    recorded_at_epoch_ms: int,
    duration_s: float,
    theme: str,
    transcript_markdown: str,     # rendered speaker/timestamp body
    summary: str | None = None,
    action_items: list[str] | None = None,
    project: str | None = None,
    speakers: int | None = None,
    unreviewed_tag: str = "unreviewed",
) -> str:
    """Render a long-meeting note for Meeting Notes/."""
    tags = ["meeting", unreviewed_tag]
    if project:
        tags.append("project")

    lines = ["---"]
    lines.append("type: meeting")
    lines.append(f"plaud_id: {plaud_id}")
    lines.append(f"recorded_at: {_iso_local(recorded_at_epoch_ms)}")
    lines.append(f"duration_s: {duration_s:.0f}")
    lines.append("transcript_source: local-qwen3-asr")
    lines.append(f"project: {project if project else 'null'}")
    lines.append(f'theme: "{theme}"')
    if speakers is not None:
        lines.append(f"speakers: {speakers}")
    lines.append(f"tags: [{', '.join(tags)}]")
    lines.append("---")
    lines.append("")
    lines.append(f"# {theme}")
    lines.append("")
    lines.append(f"**Date:** {_local_dt(recorded_at_epoch_ms).strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append("## Transcript")
    lines.append("")
    lines.append(transcript_markdown.strip())
    lines.append("")
    if summary:
        lines.append("## Summary")
        lines.append("")
        lines.append(summary.strip())
        lines.append("")
    if action_items:
        lines.append("## Action items")
        lines.append("")
        for item in action_items:
            clean = item.strip().lstrip("-").strip()
            if clean:
                lines.append(f"- [ ] {clean}")
        lines.append("")
    lines.append(f"#{unreviewed_tag}")
    lines.append("")
    return "\n".join(lines)


# ── Inbox dashboard ─────────────────────────────────────────────────
# (Phase 4 — will be redesigned around the weekly raw files + digest workflow.)


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        weekly = Path(d) / "2026-W26.md"
        bullet = append_clip_to_weekly(
            weekly,
            recorded_at_epoch_ms=1782697590000,  # 2026-06-28 21:46 local
            transcript="[Speaker 1]在测试一下录音距离。哎，现在在录了吗？\n[Speaker 1]三米远，测试结束。",
        )
        print("appended:", bullet)
        print("---file---")
        print(weekly.read_text(encoding="utf-8"))
        # Append a second clip to verify accumulation.
        append_clip_to_weekly(
            weekly,
            recorded_at_epoch_ms=1782697800000,
            transcript="Second clip of the same week.",
        )
        print("---after 2nd clip---")
        print(weekly.read_text(encoding="utf-8"))
