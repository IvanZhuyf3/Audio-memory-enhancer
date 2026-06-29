r"""Compare Qwen3-ASR output against Plaud cloud transcript (ground truth).

Usage:
    python scripts/compare_transcripts.py

Reads:
    - Plaud ground truth: Temp/opencode/steve_jobs_plaud_raw.json
    - Qwen3-ASR output:   Meeting Notes/meeting_..._Apple Microsoft...md

Produces:
    - Side-by-side stats (word count, speakers, segments)
    - Approximate Word Error Rate (WER) via normalized text comparison
    - Sample side-by-side excerpts
    - Saves report to Temp/opencode/asr_comparison_report.md
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

TEMP = Path(r"C:\Users\Yifan\AppData\Local\Temp\opencode")
PLAUD_JSON = TEMP / "steve_jobs_plaud_raw.json"
VAULT = Path(r"C:\Users\Yifan\Documents\Obsidian\Yifan_Obsidian\Meeting Notes")
REPORT = TEMP / "asr_comparison_report.md"


def find_qwen_note() -> Path:
    """Find the generated Meeting Notes markdown for the Steve Jobs recording."""
    for f in VAULT.glob("meeting_*Apple Microsoft*.md"):
        return f
    # Broader search
    for f in VAULT.glob("meeting_*.md"):
        text = f.read_text(encoding="utf-8")
        if "aecd6667db85237be327a2772d3a3fc7" in text:
            return f
    raise FileNotFoundError("Could not find the Qwen3-ASR Meeting Notes file")


def extract_plaud_text(data: list[dict]) -> tuple[str, list[str]]:
    """Extract plain text + speaker list from Plaud raw JSON."""
    texts = []
    speakers = set()
    for seg in data:
        sp = seg.get("speaker", "").strip()
        content = seg.get("content", "").strip()
        if content:
            texts.append(content)
        if sp:
            speakers.add(sp)
    return " ".join(texts), sorted(speakers)


def extract_qwen_text(md_path: Path) -> tuple[str, list[str]]:
    """Extract plain text + speaker list from the Meeting Notes markdown."""
    text = md_path.read_text(encoding="utf-8")
    # Find the Transcript section
    parts = text.split("## Transcript", 1)
    if len(parts) < 2:
        return "", []
    body = parts[1].split("## Summary", 1)[0] if "## Summary" in parts[1] else parts[1]

    texts = []
    speakers = set()
    for line in body.split("\n"):
        line = line.strip()
        # Speaker lines: **SPEAKER_06** (00:03:13)
        m = re.match(r"\*\*(.+?)\*\*\s*\(\d{2}:\d{2}:\d{2}\)", line)
        if m:
            speakers.add(m.group(1).strip())
        elif line and not line.startswith("<!--") and not line.startswith("#"):
            texts.append(line)
    return " ".join(texts), sorted(speakers)


def normalize(text: str) -> list[str]:
    """Normalize text for WER: lowercase, strip punctuation, split into words."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.split()


def wer(reference: list[str], hypothesis: list[str]) -> float:
    """Approximate Word Error Rate using Levenshtein distance on word sequences.

    WER = (substitutions + deletions + insertions) / reference_words
    """
    r, h = reference, hypothesis
    # DP table
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1]) + 1
    edits = d[len(r)][len(h)]
    return edits / max(len(r), 1)


def main() -> int:
    # Load Plaud ground truth
    plaud_data = json.loads(PLAUD_JSON.read_text(encoding="utf-8"))
    plaud_text, plaud_speakers = extract_plaud_text(plaud_data)

    # Load Qwen3-ASR output
    qwen_path = find_qwen_note()
    qwen_text, qwen_speakers = extract_qwen_text(qwen_path)

    # Normalize for WER
    plaud_words = normalize(plaud_text)
    qwen_words = normalize(qwen_text)

    # Compute WER (on a truncated sample for speed if very long)
    sample_size = min(len(plaud_words), 5000)
    wer_val = wer(plaud_words[:sample_size], qwen_words[:sample_size])

    # Build report
    lines = []
    lines.append("# ASR Comparison: Qwen3-ASR vs Plaud Cloud (Ground Truth)")
    lines.append("")
    lines.append(f"**Recording**: Steve Jobs & Bill Gates: A Conversation (81 min)")
    lines.append(f"**Qwen3-ASR note**: `{qwen_path.name}`")
    lines.append("")

    lines.append("## Stats")
    lines.append("")
    lines.append("| Metric | Plaud (ground truth) | Qwen3-ASR |")
    lines.append("|---|---|---|")
    lines.append(f"| Segments | {len(plaud_data)} | ~{len(qwen_speakers)} speaker groups |")
    lines.append(f"| Words | {len(plaud_words)} | {len(qwen_words)} |")
    lines.append(f"| Speakers | {len(plaud_speakers)} | {len(qwen_speakers)} |")
    lines.append(f"| Speaker labels | Named (Jobs, Gates, ...) | Generic (SPEAKER_04, ...) |")
    lines.append("")

    lines.append(f"**Word Error Rate (first {sample_size} words): {wer_val:.1%}**")
    lines.append(f"*(Lower is better. Typical good ASR: 5-15%. This compares Qwen3-ASR")
    lines.append(f"against Plaud's cloud whisper — differences include both real errors")
    lines.append(f"and legitimate wording variations.)")
    lines.append("")

    lines.append("## Plaud speakers (named)")
    for s in plaud_speakers:
        lines.append(f"- {s}")
    lines.append("")

    lines.append("## Qwen3-ASR speakers (generic)")
    for s in qwen_speakers:
        lines.append(f"- {s}")
    lines.append("")

    # Side-by-side sample: first 500 chars of each
    lines.append("## Sample comparison (first ~400 chars)")
    lines.append("")
    lines.append("### Plaud ground truth")
    lines.append("```")
    lines.append(plaud_text[:400])
    lines.append("```")
    lines.append("")
    lines.append("### Qwen3-ASR")
    lines.append("```")
    lines.append(qwen_text[:400])
    lines.append("```")
    lines.append("")

    # Another sample from the middle
    mid = len(plaud_text) // 2
    lines.append("## Sample comparison (middle ~400 chars)")
    lines.append("")
    lines.append("### Plaud ground truth")
    lines.append("```")
    lines.append(plaud_text[mid:mid + 400])
    lines.append("```")
    lines.append("")
    qmid = len(qwen_text) // 2
    lines.append("### Qwen3-ASR")
    lines.append("```")
    lines.append(qwen_text[qmid:qmid + 400])
    lines.append("```")
    lines.append("")

    report = "\n".join(lines)
    REPORT.write_text(report, encoding="utf-8")
    print(report[:3000])
    print(f"\n[full report saved to {REPORT}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
