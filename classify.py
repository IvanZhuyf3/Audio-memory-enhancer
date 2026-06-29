"""DeepSeek-powered content classification.

Two entry points:
    classify_short(transcript, metadata, projects, ...) -> dict
        For short memos (<threshold). Returns {sub_type, project, confidence}.

    classify_long(transcript_text, metadata, projects, ...) -> dict
        For long recordings. Returns {project, theme, summary, action_items}.

Both use the DeepSeek API (OpenAI-compatible). The API key + model are read
from the shared audio_transcribe_notes/config.ini to avoid duplicating secrets.

sub_type values for memos: time-sensitive | long-term | project-snippet
project values: a project name from projects.yaml, or None.
"""

from __future__ import annotations

import configparser
import json
import re
from pathlib import Path


# ── Secrets + client ────────────────────────────────────────────────

def load_llm_config(secrets_source: str | Path) -> tuple[str, str]:
    """Read (api_key, model) from the shared audio_transcribe_notes/config.ini.

    Falls back gracefully if the file/section is missing.
    """
    p = Path(secrets_source)
    if not p.exists():
        raise RuntimeError(
            f"Secrets source not found: {p}. "
            f"Set deepseek_api_key in audio_transcribe_notes/config.ini."
        )
    cp = configparser.ConfigParser()
    cp.read(p, encoding="utf-8")
    if "defaults" not in cp:
        raise RuntimeError(f"No [defaults] section in {p}")
    key = cp["defaults"].get("deepseek_api_key", "").strip()
    model = cp["defaults"].get("deepseek_model", "deepseek-v4-flash").strip()
    if not key:
        raise RuntimeError(
            f"deepseek_api_key is empty in {p}. Fill it in and re-run."
        )
    return key, model


def _client(api_key: str):
    """OpenAI client pointed at DeepSeek's OpenAI-compatible endpoint."""
    from openai import OpenAI  # already in the shared venv
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


# ── Project matching context ────────────────────────────────────────

def _projects_context(projects: list[dict]) -> str:
    """Render projects.yaml as a reference block for the LLM."""
    if not projects:
        return "(no projects registered)"
    lines = []
    for p in projects:
        aliases = ", ".join(p.get("aliases", []) or [])
        keywords = ", ".join(p.get("keywords", []) or [])
        parts = [p["name"]]
        if aliases:
            parts.append(f"aliases: {aliases}")
        if keywords:
            parts.append(f"keywords: {keywords}")
        lines.append("- " + " | ".join(parts))
    return "\n".join(lines)


def _match_project_name(text: str, projects: list[dict]) -> str | None:
    """Cheap keyword/alias pre-match. Returns a project name or None.

    Used as a hint / fallback alongside the LLM call.
    """
    lower = text.lower()
    for p in projects:
        candidates = [p["name"].lower()] + [a.lower() for a in (p.get("aliases") or [])]
        for c in candidates:
            if c and c in lower:
                return p["name"]
        for kw in (p.get("keywords") or []):
            if kw and kw.lower() in lower:
                return p["name"]
    return None


# ── Short memo classification ───────────────────────────────────────

SHORT_SUB_TYPES = ("time-sensitive", "long-term", "project-snippet")


def classify_short(
    transcript: str,
    metadata: dict,
    projects: list[dict],
    *,
    secrets_source: str | Path,
    temperature: float = 0.0,
) -> dict:
    """Classify a short memo. Returns {sub_type, project, confidence, raw}.

    Falls back to heuristics if the LLM call fails (so intake never blocks).
    """
    if not transcript.strip():
        return {"sub_type": "long-term", "project": None, "confidence": 0.0, "raw": ""}

    hint_project = _match_project_name(transcript, projects)
    system = (
        "You classify short voice memos (under 15 minutes, single speaker) into one of:\n"
        "- time-sensitive: a deadline, reminder, or task that expires (meetings to book, "
        "follow-ups, due dates, 'remember to...').\n"
        "- long-term: reference knowledge, ideas, reflections, journaling — no deadline.\n"
        "- project-snippet: relates to a specific ongoing project.\n\n"
        f"Registered projects (if the memo relates to one, name it exactly):\n"
        f"{_projects_context(projects)}\n\n"
        "Respond as STRICT JSON only, no prose:\n"
        '{"sub_type": "<one of time-sensitive|long-term|project-snippet>", '
        '"project": "<project name or null>", '
        '"confidence": <0.0-1.0>}'
    )

    try:
        key, model = load_llm_config(secrets_source)
        resp = _client(key).chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": transcript[:4000]},
            ],
            temperature=temperature,
            response_format={"type": "json_object"},
            extra_body={"thinking": {"type": "disabled"}},
        )
        raw = resp.choices[0].message.content.strip()
        parsed = _parse_json_loose(raw)
        sub_type = parsed.get("sub_type", "long-term")
        if sub_type not in SHORT_SUB_TYPES:
            sub_type = "long-term"
        project = parsed.get("project") or hint_project
        confidence = float(parsed.get("confidence", 0.5))
        return {"sub_type": sub_type, "project": project, "confidence": confidence, "raw": raw}
    except Exception as e:
        # Never block intake on classifier failure — fall back to heuristics.
        print(f"  [classify_short] LLM call failed ({e}); using heuristic fallback")
        sub_type = "time-sensitive" if hint_project else "long-term"
        return {"sub_type": sub_type, "project": hint_project, "confidence": 0.3, "raw": ""}


# ── Long recording classification ───────────────────────────────────

def classify_long(
    transcript_text: str,
    metadata: dict,
    projects: list[dict],
    *,
    secrets_source: str | Path,
    temperature: float = 0.0,
) -> dict:
    """For long recordings: detect project, generate theme + summary + action items.

    Returns {project, theme, summary, action_items[], confidence, raw}.
    """
    hint_project = _match_project_name(transcript_text, projects)
    text_for_llm = transcript_text[:8000]  # truncate to control token cost
    system = (
        "You analyze a meeting/conference/experiment recording transcript. Produce:\n"
        "1. theme: a 2-4 word title (no quotes, no punctuation).\n"
        "2. summary: 2-4 sentences capturing the key discussion.\n"
        "3. action_items: a JSON array of short imperative to-dos (empty array if none).\n"
        "4. project: the project name this relates to, or null.\n\n"
        f"Registered projects:\n{_projects_context(projects)}\n\n"
        "Respond as STRICT JSON only:\n"
        '{"theme": "...", "summary": "...", "action_items": ["..."], '
        '"project": "<name or null>", "confidence": <0.0-1.0>}'
    )
    try:
        key, model = load_llm_config(secrets_source)
        resp = _client(key).chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": text_for_llm},
            ],
            temperature=temperature,
            response_format={"type": "json_object"},
            extra_body={"thinking": {"type": "disabled"}},
        )
        raw = resp.choices[0].message.content.strip()
        parsed = _parse_json_loose(raw)
        theme = _sanitize_theme(parsed.get("theme") or "Meeting")
        project = parsed.get("project") or hint_project
        summary = parsed.get("summary") or ""
        action_items = parsed.get("action_items") or []
        if not isinstance(action_items, list):
            action_items = []
        return {
            "project": project,
            "theme": theme,
            "summary": summary,
            "action_items": [str(a) for a in action_items],
            "confidence": float(parsed.get("confidence", 0.5)),
            "raw": raw,
        }
    except Exception as e:
        print(f"  [classify_long] LLM call failed ({e}); using heuristic fallback")
        return {
            "project": hint_project,
            "theme": _sanitize_theme(hint_project or "Meeting"),
            "summary": "",
            "action_items": [],
            "confidence": 0.3,
            "raw": "",
        }


def generate_theme(transcript_text: str, *, secrets_source: str | Path, temperature: float = 0.3) -> str:
    """Standalone theme generator (used when classify_long isn't called yet)."""
    try:
        key, model = load_llm_config(secrets_source)
        resp = _client(key).chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Generate a very short title (2-4 words) for this transcript. "
                        "Output ONLY the title, no quotes, no punctuation. "
                        "Examples: 'Project Review', 'Lab Meeting', 'Interview Prep'."
                    ),
                },
                {"role": "user", "content": transcript_text[:2000]},
            ],
            temperature=temperature,
            extra_body={"thinking": {"type": "disabled"}},
        )
        return _sanitize_theme(resp.choices[0].message.content.strip())
    except Exception as e:
        print(f"  [generate_theme] failed ({e})")
        return "Meeting"


# ── Helpers ─────────────────────────────────────────────────────────

def _parse_json_loose(raw: str) -> dict:
    """Parse JSON, stripping markdown code fences if present."""
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # Last-ditch: try to find a {...} block.
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        return {}


def _sanitize_theme(theme: str) -> str:
    theme = re.sub(r'[\\/*?:"<>|]', "", theme or "")
    theme = re.sub(r"\s+", " ", theme).strip()
    return theme[:50] or "Meeting"


if __name__ == "__main__":
    # Smoke test — requires secrets_source to be valid.
    import sys
    if len(sys.argv) < 2:
        print("Usage: python classify.py <secrets_path>")
        sys.exit(0)
    out = classify_short(
        "Remember to email the SRS vendor about the quote tomorrow.",
        {},
        [{"name": "OmniSRS", "aliases": ["SRS"], "keywords": ["Raman"]}],
        secrets_source=sys.argv[1],
    )
    print("short:", out)
