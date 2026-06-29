r"""Blind A/B comparison between Plaud cloud transcript and Qwen3-ASR output.

Aligns both transcripts by timestamp, uses DeepSeek to find pairs with
genuinely different meanings, then generates a self-contained HTML where
the user picks winners without knowing which is which.

Usage:
    python scripts/blind_compare.py
    # (reads Temp/opencode/meeting_plaud.json + meeting_qwen.json)
    # → generates Temp/opencode/blind_comparison.html
    # → saves reveal key to Temp/opencode/blind_reveal_key.json

After the user picks winners in the HTML and saves results:
    python scripts/blind_compare.py --reveal <results.json>
"""

from __future__ import annotations

import html
import json
import random
import sys
import configparser
from pathlib import Path

TEMP = Path(r"C:\Users\Yifan\AppData\Local\Temp\opencode")
PLAUD_JSON = TEMP / "meeting_plaud.json"
QWEN_JSON = TEMP / "meeting_qwen.json"
HTML_OUT = TEMP / "blind_comparison.html"
REVEAL_KEY = TEMP / "blind_reveal_key.json"

SECRETS_SRC = Path(r"C:\Users\Yifan\OneDrive\Opencode_workspace\audio_transcribe_notes\config.ini")
MAX_PAIRS = 40  # cap to avoid overwhelming the user


# ── Load + normalize ────────────────────────────────────────────────

def load_plaud(path: Path) -> list[dict]:
    """Plaud format: [{start_time (ms), end_time (ms), content, speaker}, ...]"""
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for seg in data:
        text = (seg.get("content") or "").strip()
        if not text:
            continue
        out.append({
            "start": seg.get("start_time", 0) / 1000,  # ms → s
            "end": seg.get("end_time", 0) / 1000,
            "text": text,
            "speaker": seg.get("speaker", ""),
            "source": "plaud",
        })
    return out


def load_qwen(path: Path) -> list[dict]:
    """Qwen format: [{start (s), end (s), text, speaker}, ...]"""
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for seg in data:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        out.append({
            "start": seg.get("start", 0),
            "end": seg.get("end", 0),
            "text": text,
            "speaker": seg.get("speaker", ""),
            "source": "qwen",
        })
    return out


# ── Time alignment ──────────────────────────────────────────────────

def align_by_time(plaud: list[dict], qwen: list[dict]) -> list[dict]:
    """For each Plaud segment, find overlapping Qwen3 segment(s) and pair them."""
    pairs = []
    for p in plaud:
        p_start, p_end = p["start"], p["end"]
        overlapping = []
        for q in qwen:
            overlap = min(p_end, q["end"]) - max(p_start, q["start"])
            if overlap > 0:
                overlapping.append(q)
        if not overlapping:
            continue
        # Merge overlapping Qwen segments into one text block.
        overlapping.sort(key=lambda q: q["start"])
        merged_text = " ".join(q["text"] for q in overlapping)
        pairs.append({
            "timestamp": p_start,
            "plaud_text": p["text"],
            "plaud_speaker": p["speaker"],
            "qwen_text": merged_text,
            "qwen_speakers": ", ".join(sorted(set(q["speaker"] for q in overlapping))),
        })
    return pairs


# ── DeepSeek filtering ──────────────────────────────────────────────

def filter_different(pairs: list[dict], api_key: str, model: str) -> list[dict]:
    """Use DeepSeek to find pairs with genuinely different meanings.

    Returns pairs tagged with a difference_score (1-5) for ranking.
    """
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    different = []
    batch_size = 8

    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]
        prompt_lines = []
        for j, p in enumerate(batch):
            prompt_lines.append(f"=== Pair {j+1} ===")
            prompt_lines.append(f"Version A: {p['plaud_text'][:400]}")
            prompt_lines.append(f"Version B: {p['qwen_text'][:400]}")
            prompt_lines.append("")

        system = (
            "You compare pairs of ASR transcript segments covering the same audio. "
            "For each pair, judge:\n"
            "1. judgment: SAME if they convey the same meaning (despite minor wording "
            "differences, synonyms, or paraphrasing). DIFFERENT if there's a substantive "
            "difference in content, names, numbers, or meaning.\n"
            "2. score: 1-5 indicating HOW different (5 = completely different, 1 = nearly identical).\n\n"
            "Respond as a JSON array: "
            '[{"pair": 1, "judgment": "SAME"|"DIFFERENT", "score": 1-5}, ...]'
        )

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": "\n".join(prompt_lines)},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
                extra_body={"thinking": {"type": "disabled"}},
            )
            raw = resp.choices[0].message.content.strip()
            parsed = _parse_json_loose(raw)
            # Response might be {"data": [...]} or [...]
            items = parsed if isinstance(parsed, list) else parsed.get("data", parsed.get("pairs", []))
            if isinstance(items, dict):
                items = [items]
        except Exception as e:
            print(f"  [DeepSeek] batch {i//batch_size+1} failed ({e}), skipping")
            continue

        for item in items:
            if not isinstance(item, dict):
                continue
            idx = item.get("pair", 0) - 1
            if 0 <= idx < len(batch) and item.get("judgment", "").upper() == "DIFFERENT":
                p = batch[idx]
                p["difference_score"] = int(item.get("score", 3))
                different.append(p)

        print(f"  [DeepSeek] batch {i//batch_size+1}/{(len(pairs)+batch_size-1)//batch_size}: "
              f"{len(different)} different so far")

    # Sort by difference score (most different first), cap at MAX_PAIRS.
    different.sort(key=lambda p: -p.get("difference_score", 3))
    return different[:MAX_PAIRS]


def _parse_json_loose(raw: str):
    import re
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"[\[{].*[\]}]", s, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {}


# ── HTML generation ─────────────────────────────────────────────────

def _fmt_timestamp(seconds: float) -> str:
    h = int(seconds) // 3600
    m = (int(seconds) % 3600) // 60
    s = int(seconds) % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def generate_html(different_pairs: list[dict], reveal_key: list[dict]) -> str:
    """Generate self-contained blind comparison HTML. Returns HTML string."""
    cards = []
    for i, p in enumerate(different_pairs):
        # Randomly shuffle which is Option A vs B.
        if random.random() < 0.5:
            opt_a, opt_b = p["plaud_text"], p["qwen_text"]
            reveal_key.append({"pair": i + 1, "A": "plaud", "B": "qwen"})
        else:
            opt_a, opt_b = p["qwen_text"], p["plaud_text"]
            reveal_key.append({"pair": i + 1, "A": "qwen", "B": "plaud"})

        ts = _fmt_timestamp(p["timestamp"])
        score = p.get("difference_score", 3)
        cards.append(f"""
        <div class="pair" data-pair="{i+1}">
            <div class="ts">Pair {i+1} — timestamp {ts} — difference score {score}/5</div>
            <div class="opt">
                <label><input type="radio" name="p{i+1}" value="A"> <strong>Option A:</strong></label>
                <div class="txt">{html.escape(opt_a)}</div>
            </div>
            <div class="opt">
                <label><input type="radio" name="p{i+1}" value="B"> <strong>Option B:</strong></label>
                <div class="txt">{html.escape(opt_b)}</div>
            </div>
            <div class="opt">
                <label><input type="radio" name="p{i+1}" value="TIE" checked> Tie / both wrong</label>
            </div>
        </div>""")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ASR Blind Comparison</title>
<style>
  body {{ font-family: -apple-system, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; line-height: 1.6; }}
  h1 {{ margin-bottom: 5px; }}
  .intro {{ color: #555; margin-bottom: 30px; }}
  .pair {{ border: 1px solid #ddd; padding: 15px; margin: 20px 0; border-radius: 8px; background: #fafafa; }}
  .ts {{ color: #666; font-size: 0.85em; margin-bottom: 10px; font-family: monospace; }}
  .opt {{ margin: 8px 0; padding: 10px; background: white; border-radius: 4px; border: 1px solid #eee; }}
  .opt label {{ cursor: pointer; display: flex; align-items: center; gap: 8px; }}
  .txt {{ margin-top: 8px; padding: 8px; color: #333; }}
  .actions {{ text-align: center; margin: 30px 0; position: sticky; bottom: 0; background: white; padding: 15px; }}
  button {{ padding: 12px 40px; font-size: 1.1em; cursor: pointer; border: 2px solid #333; background: white; border-radius: 8px; }}
  button:hover {{ background: #333; color: white; }}
</style>
</head>
<body>
<h1>ASR Blind Comparison</h1>
<p class="intro">
  For each pair, find the timestamp in your audio recording and listen.
  Pick <strong>Option A</strong> or <strong>Option B</strong> based on which
  transcript is more accurate. Choose <strong>Tie</strong> if they're equally
  good/bad. Focus on meaning — don't overthink minor wording differences.
</p>

{''.join(cards)}

<div class="actions">
  <button onclick="save()">Save Results & Reveal</button>
</div>

<script>
function save() {{
  const results = [];
  document.querySelectorAll('.pair').forEach(el => {{
    const pairNum = parseInt(el.dataset.pair);
    const checked = el.querySelector('input[type=radio]:checked');
    results.push({{ pair: pairNum, choice: checked ? checked.value : 'TIE' }});
  }});
  const blob = new Blob([JSON.stringify(results, null, 2)], {{type: 'application/json'}});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'blind_comparison_results.json';
  a.click();
  alert('Results saved! Tell the agent the file is ready.');
}}
</script>
</body>
</html>"""


# ── Reveal + tally ──────────────────────────────────────────────────

def reveal(results_path: Path, reveal_key: list[dict]) -> int:
    """Read user picks + reveal key, tally winners."""
    # utf-8-sig strips BOM if present (PowerShell/tools sometimes add it).
    picks = json.loads(results_path.read_text(encoding="utf-8-sig"))
    key_map = {k["pair"]: k for k in reveal_key}

    plaud_wins, qwen_wins, ties = 0, 0, 0
    details = []

    for pick in picks:
        pair_num = pick["pair"]
        choice = pick["choice"]
        key = key_map.get(pair_num, {})
        a_src = key.get("A", "?")
        b_src = key.get("B", "?")

        if choice == "TIE":
            ties += 1
            winner = "tie"
        elif choice == "A":
            winner = a_src
        elif choice == "B":
            winner = b_src
        else:
            ties += 1
            winner = "tie"

        if winner == "plaud":
            plaud_wins += 1
        elif winner == "qwen":
            qwen_wins += 1
        else:
            ties += 1

        details.append(f"  Pair {pair_num}: chose {choice} → {a_src} was A, {b_src} was B → winner: {winner}")

    print(f"\n{'='*60}")
    print(f"BLIND COMPARISON RESULTS")
    print(f"{'='*60}")
    print(f"  Plaud cloud wins:  {plaud_wins}")
    print(f"  Qwen3-ASR wins:    {qwen_wins}")
    print(f"  Ties / both wrong: {ties}")
    print(f"  Total pairs:       {len(picks)}")
    print()
    for d in details:
        print(d)
    print()

    if plaud_wins > qwen_wins:
        print(f"  → Plaud cloud wins ({plaud_wins} vs {qwen_wins})")
    elif qwen_wins > plaud_wins:
        print(f"  → Qwen3-ASR wins ({qwen_wins} vs {plaud_wins})")
    else:
        print(f"  → Tie ({plaud_wins} each)")

    return 0


# ── Main ────────────────────────────────────────────────────────────

def main() -> int:
    if "--reveal" in sys.argv:
        idx = sys.argv.index("--reveal")
        results_path = Path(sys.argv[idx + 1])
        reveal_key = json.loads(REVEAL_KEY.read_text(encoding="utf-8"))
        return reveal(results_path, reveal_key)

    # Normal flow: generate HTML.
    if not PLAUD_JSON.exists() or not QWEN_JSON.exists():
        print(f"Missing transcript files. Need both:")
        print(f"  {PLAUD_JSON}")
        print(f"  {QWEN_JSON}")
        return 1

    print("Loading transcripts...")
    plaud = load_plaud(PLAUD_JSON)
    qwen = load_qwen(QWEN_JSON)
    print(f"  Plaud: {len(plaud)} segments")
    print(f"  Qwen3: {len(qwen)} segments")

    print("\nAligning by timestamp...")
    pairs = align_by_time(plaud, qwen)
    print(f"  {len(pairs)} aligned pairs")

    # Load DeepSeek config.
    cp = configparser.ConfigParser()
    cp.read(SECRETS_SRC, encoding="utf-8")
    d = cp["defaults"] if "defaults" in cp else {}
    api_key = d.get("deepseek_api_key", "").strip()
    model = d.get("deepseek_model", "deepseek-v4-flash").strip()

    if api_key:
        print(f"\nFiltering with DeepSeek (finding meaning-different pairs)...")
        different = filter_different(pairs, api_key, model)
        print(f"  {len(different)} pairs with different meanings (capped at {MAX_PAIRS})")
    else:
        print("\n[warning] No DeepSeek key — skipping meaning filter, showing all pairs")
        different = pairs[:MAX_PAIRS]

    if not different:
        print("\nNo different pairs found — transcripts are very similar!")
        return 0

    print(f"\nGenerating blind comparison HTML...")
    reveal_key: list[dict] = []
    html_content = generate_html(different, reveal_key)
    HTML_OUT.write_text(html_content, encoding="utf-8")
    REVEAL_KEY.write_text(json.dumps(reveal_key, indent=2), encoding="utf-8")

    print(f"\nDone! Open this in your browser:")
    print(f"  {HTML_OUT}")
    print(f"\nAfter picking winners and saving the results JSON, run:")
    print(f"  python scripts/blind_compare.py --reveal <results.json>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
