"""Markdown note templates for short memos and long meetings.

Pure string builders — no LLM, no GPU. Produce the full note body the pipeline
writes to the Obsidian vault. Frontmatter is hand-emitted (key: value) to match
Obsidian/YAML conventions and the upstream plaud-toolkit sync.ts style.
"""

from __future__ import annotations

from datetime import datetime, timezone


def _local_dt(epoch_ms: int) -> datetime:
    return datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc).astimezone()


def _iso_local(epoch_ms: int) -> str:
    return _local_dt(epoch_ms).isoformat(timespec="seconds")


# ── Short memo (Obsmem/raw/) ────────────────────────────────────────

def render_short_memo(
    *,
    plaud_id: str,
    recorded_at_epoch_ms: int,
    duration_s: float,
    transcript: str,
    sub_type: str,                # time-sensitive | long-term | project-snippet
    project: str | None = None,
    plaud_title: str | None = None,
    unreviewed_tag: str = "unreviewed",
) -> str:
    """Render a short-memo note for Obsmem/raw/.

    Body is the Plaud cloud transcript verbatim; frontmatter carries the
    classification fields the future digest pass will consume.
    """
    tags = ["memo", unreviewed_tag, sub_type]
    if project:
        tags.append("project")

    lines = ["---"]
    lines.append("type: memo")
    lines.append(f"sub_type: {sub_type}")
    lines.append(f"plaud_id: {plaud_id}")
    lines.append(f"recorded_at: {_iso_local(recorded_at_epoch_ms)}")
    lines.append(f"duration_s: {duration_s:.0f}")
    lines.append("transcript_source: plaud-cloud")
    lines.append(f"project: {project if project else 'null'}")
    if plaud_title:
        # Quote in case the title has colons / special chars.
        safe = plaid_title_escaped = plaud_title.replace('"', '\\"')
        lines.append(f'plaud_title: "{safe}"')
    lines.append(f"tags: [{', '.join(tags)}]")
    lines.append("---")
    lines.append("")
    if plaud_title:
        lines.append(f"# {plaud_title}")
        lines.append("")
    lines.append("## Transcript")
    lines.append("")
    lines.append(transcript.strip() or "*(no transcript available)*")
    lines.append("")
    lines.append(f"#{unreviewed_tag}")
    lines.append("")
    return "\n".join(lines)


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

def render_inbox_dashboard(unreviewed_paths: list[str]) -> str:
    """A simple Dataview-backed dashboard note listing open unreviewed items."""
    lines = [
        "---",
        "type: dashboard",
        "---",
        "",
        "# Plaud Inbox",
        "",
        "Auto-generated list of unreviewed Plaud notes. Remove the `#unreviewed`",
        "tag from a note to drop it off this list.",
        "",
        "```dataview",
        "TABLE sub_type AS Type, project AS Project, recorded_at AS Recorded",
        f'FROM "{unreviewed_paths[0].split("/")[0] if unreviewed_paths else "Obsmem/raw"}" OR "Meeting Notes"',
        'WHERE contains(tags, "unreviewed")',
        "SORT recorded_at DESC",
        "```",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    # Smoke test
    print(render_short_memo(
        plaud_id="abc123",
        recorded_at_epoch_ms=1751110200000,
        duration_s=180,
        transcript="Remember to email the vendor about the quote.",
        sub_type="time-sensitive",
        project=None,
        plaud_title="Vendor followup",
    )[:300])
    print("---")
    print(render_long_meeting(
        plaud_id="def456",
        recorded_at_epoch_ms=1751110200000,
        duration_s=1832,
        theme="Project Standup",
        transcript_markdown="**Speaker 1** (00:00:05)\nLet's start with the SRS update.\n",
        summary="Discussed SRS timeline and next steps.",
        action_items=["Send timeline to advisor", "Book microscope time"],
        project="OmniSRS",
        speakers=3,
    )[:400])
