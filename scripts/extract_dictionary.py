"""Extract domain terms from papers via DeepSeek for dictionary.md.

Feeds both paper texts to DeepSeek with a structured prompt asking for a
glossary of domain-specific terms, abbreviations, and jargon that an ASR
system might get wrong. Output is formatted for dictionary.md.
"""
import os
import sys
import configparser
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# ── Paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SECRETS_SRC = Path(r"C:\Users\Yifan\OneDrive\Opencode_workspace\audio_transcribe_notes\config.ini")

DOCX_TEXT = Path(os.environ["TEMP"]) / "opencode" / "paper_text" / "aop_review.txt"
PDF_MD = Path(os.environ["USERPROFILE"]) / "Downloads" / "s41592-025-02655-w" / "s41592-025-02655-w.md"
OUTPUT = PROJECT_ROOT / "dictionary.md"


def load_secrets() -> tuple[str, str]:
    cp = configparser.ConfigParser()
    cp.read(SECRETS_SRC, encoding="utf-8")
    key = cp["defaults"].get("deepseek_api_key", "").strip()
    model = cp["defaults"].get("deepseek_model", "deepseek-chat").strip()
    if not key:
        raise RuntimeError(f"deepseek_api_key empty in {SECRETS_SRC}")
    return key, model


def extract_terms(text_a: str, text_b: str, api_key: str, model: str) -> str:
    """Send both paper texts to DeepSeek, get back a formatted glossary."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    system_prompt = """You are a domain terminology extractor for ASR (automatic speech recognition) pre-correction.

Your job: read scientific papers and extract a comprehensive glossary of domain-specific terms that an ASR system might misrecognize. These are terms whose SPELLING matters and that a general-purpose ASR would likely get wrong.

INCLUDE:
1. Technical terms with abbreviations: format as "- full term (ABBR)" e.g. "- stimulated Raman scattering (SRS)"
2. Method/modality names: "- stimulated Raman photothermal microscopy"
3. Instrument components: "- photomultiplier tube (PMT)"
4. Physical/optical/chemical concepts: "- near-infrared", "- wavenumber"
5. Units and parameters specific to the field: "- inverse centimeter"
6. Biological/biochemical terms that are domain-specific (not common words)
7. Software tools, databases, algorithms: "- ImageJ", "- principal component analysis (PCA)"
8. Compound hyphenated terms: "- epi-detected", "- phase-matching"
9. Named methods/eponymous techniques: "- Kramers-Kronig"
10. Chemical bonds/functional groups mentioned in context: "- alkyne", "- azide"

EXCLUDE:
- Common English words any ASR gets right (cell, tissue, light, image, etc.)
- Author names unless used as eponymous method names (e.g. include "Kramers-Kronig" but not "Cheng")
- Generic scientific words (experiment, result, figure, table, etc.)
- Single common abbreviations that are universally known (DNA, RNA, pH)

RULES:
- One term per line, markdown list format: "- term" or "- full term (ABBR)"
- Deduplicate (case-insensitive)
- Sort alphabetically by the full term
- Extract EVERY relevant term — be thorough, not selective. Aim for 150-400 terms.
- If a term appears with and without an abbreviation, use the form WITH abbreviation.
- For terms with multiple accepted spellings, use the most standard one."""

    user_prompt = f"""Below are two scientific papers about vibrational microscopy / Raman spectroscopy / photothermal microscopy. Extract the complete domain glossary.

=== PAPER 1: Stimulated Raman Photothermal Microscopy: Theory and Implementation (review) ===

{text_a}

=== PAPER 2: Advanced vibrational microscopes for life science (Nature Methods review) ===

{text_b}

Now output the glossary. One term per line, sorted alphabetically, in markdown list format."""

    print(f"Sending {len(text_a) + len(text_b):,} chars to DeepSeek ({model})...")
    print("(this may take 1-3 minutes)\n")

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=8000,
        stream=False,
    )

    content = resp.choices[0].message.content.strip()
    tokens_in = resp.usage.prompt_tokens
    tokens_out = resp.usage.completion_tokens
    print(f"Done. Tokens: {tokens_in:,} in, {tokens_out:,} out\n")
    return content


def parse_terms(raw: str) -> list[str]:
    """Parse DeepSeek output into clean list items."""
    terms = []
    seen = set()
    for line in raw.split("\n"):
        line = line.strip()
        if not line.startswith("- "):
            continue
        entry = line[2:].strip()
        if not entry:
            continue
        key = entry.lower()
        if key not in seen:
            seen.add(key)
            terms.append(entry)

    return sorted(terms, key=str.lower)


def main():
    # Load texts
    text_a = DOCX_TEXT.read_text(encoding="utf-8")
    text_b = PDF_MD.read_text(encoding="utf-8")
    print(f"Paper 1 (docx): {len(text_a):,} chars")
    print(f"Paper 2 (pdf):  {len(text_b):,} chars")

    # Load secrets
    api_key, model = load_secrets()
    print(f"Model: {model}")

    # Extract
    raw = extract_terms(text_a, text_b, api_key, model)

    # Parse
    terms = parse_terms(raw)
    print(f"Extracted {len(terms)} unique terms\n")

    # Print preview
    print("=" * 60)
    print("EXTRACTED TERMS (first 50):")
    print("=" * 60)
    for t in terms[:50]:
        print(f"  {t}")
    if len(terms) > 50:
        print(f"  ... and {len(terms) - 50} more")
    print()

    # Save raw response for review
    raw_path = PROJECT_ROOT / "scripts" / "dict_extraction_raw.txt"
    raw_path.write_text(raw, encoding="utf-8")
    print(f"Raw response saved: {raw_path}")

    # Write dictionary.md
    header = """# Domain Dictionary

Terms extracted from:
- "Stimulated Raman Photothermal Microscopy: Theory and Implementation" (review)
- "Advanced vibrational microscopes for life science" (Nature Methods, 2025)

One entry per line: `- full term (ABBR)` or `- term`.
Used by build_asr_context() to bias Qwen3-ASR toward correct domain spellings.

"""

    body = "\n".join(f"- {t}" for t in terms)
    OUTPUT.write_text(header + body + "\n", encoding="utf-8")
    print(f"\nDictionary written: {OUTPUT} ({len(terms)} terms)")


if __name__ == "__main__":
    main()
