"""Compare old vs new Qwen3-ASR results + Plaud, focusing on domain term accuracy."""
import json
import re
import sys
from pathlib import Path

OUT_DIR = Path(r"C:\Users\Yifan\AppData\Local\Temp\opencode")
DICT_PATH = Path(r"C:\Users\Yifan\OneDrive\Opencode_workspace\Audio-memory-enhancer\dictionary.md")


def load_segments(path):
    """Load JSON segments and return full text."""
    data = json.loads(path.read_text(encoding="utf-8"))
    # Handle both Qwen3 ('text') and Plaud ('content') field names
    parts = []
    for s in data:
        t = s.get("text") or s.get("content") or ""
        parts.append(t)
    return " ".join(parts), len(data)


def extract_domain_terms(dict_path):
    """Extract all domain terms (en + zh + abbr) from dictionary.md."""
    terms = set()
    abbr_re = re.compile(r"^(.+?)\s*\(([A-Z][A-Za-z0-9-]+)\)$")
    for line in dict_path.read_text(encoding="utf-8").split("\n"):
        line = line.strip()
        if not line.startswith("- "):
            continue
        entry = line[2:].strip()
        for part in entry.split(" | "):
            part = part.strip()
            m = abbr_re.match(part)
            if m:
                terms.add(m.group(1).strip().lower())
                terms.add(m.group(2).strip().lower())
            elif part:
                terms.add(part.lower())
    return terms


def count_domain_terms_present(text, domain_terms):
    """Count how many domain terms appear in the text (case-insensitive)."""
    text_lower = text.lower()
    found = []
    missing = []
    for term in sorted(domain_terms):
        if term in text_lower:
            found.append(term)
        else:
            missing.append(term)
    return found, missing


def char_bigram_jaccard(text_a, text_b):
    """Character bigram Jaccard similarity (used in blind comparison)."""
    def bigrams(t):
        chars = list(t)
        return set(zip(chars, chars[1:]))
    bg_a = bigrams(text_a)
    bg_b = bigrams(text_b)
    if not bg_a or not bg_b:
        return 0.0
    return len(bg_a & bg_b) / len(bg_a | bg_b)


def main():
    # Paths
    qwen_old = OUT_DIR / "meeting_qwen_old_dict.json"
    qwen_new = OUT_DIR / "meeting_qwen.json"
    plaud = OUT_DIR / "meeting_plaud.json"

    if not qwen_new.exists():
        print("ERROR: meeting_qwen.json not found. Run asr_test.py first.")
        return 1

    # Load texts
    texts = {}
    seg_counts = {}
    for label, path in [("Qwen3 (new)", qwen_new), ("Plaud", plaud)]:
        if path.exists():
            texts[label], seg_counts[label] = load_segments(path)
    if qwen_old.exists():
        texts["Qwen3 (old)"], seg_counts["Qwen3 (old)"] = load_segments(qwen_old)

    # Print basic stats
    print("=" * 70)
    print("BASIC STATS")
    print("=" * 70)
    for label, text in texts.items():
        print(f"  {label:20s}: {len(text):,} chars, {seg_counts[label]} segments")

    # Domain term coverage
    domain_terms = extract_domain_terms(DICT_PATH)
    print(f"\n  Dictionary terms (en+zh+abbr): {len(domain_terms)}")

    print("\n" + "=" * 70)
    print("DOMAIN TERM COVERAGE (how many dict terms appear in transcript)")
    print("=" * 70)
    for label, text in texts.items():
        found, missing = count_domain_terms_present(text, domain_terms)
        pct = len(found) / len(domain_terms) * 100
        print(f"  {label:20s}: {len(found):3d} / {len(domain_terms)} ({pct:.1f}%)")

    # Similarity to Plaud (reference)
    if "Plaud" in texts:
        print("\n" + "=" * 70)
        print("SIMILARITY TO PLAUD (char-bigram Jaccard)")
        print("=" * 70)
        for label, text in texts.items():
            if label == "Plaud":
                continue
            sim = char_bigram_jaccard(text, texts["Plaud"])
            print(f"  {label:20s} vs Plaud: {sim:.4f}")

    # Sample differences (first 500 chars of each)
    print("\n" + "=" * 70)
    print("FIRST 500 CHARS OF EACH")
    print("=" * 70)
    for label, text in texts.items():
        print(f"\n--- {label} ---")
        print(text[:500])

    # Check specific high-value domain terms
    print("\n" + "=" * 70)
    print("KEY DOMAIN TERM SPOT-CHECK")
    print("=" * 70)
    key_terms = [
        "stimulated raman scattering", "srs", "cars",
        "coherent anti-stokes", "photothermal", "mid-infrared",
        "lipid droplet", "脂滴", "受激拉曼", "光漂白", "光热",
        "numerical aperture", "wavenumber", "photobleaching",
        " Stimulated Raman", "SRS", "CARS", "MIP", "VIP",
    ]
    for term in key_terms:
        row = f"  '{term}'"
        for label in texts:
            count = texts[label].lower().count(term.lower())
            row += f" | {label}: {count}"
        print(row)

    return 0


if __name__ == "__main__":
    sys.exit(main())
