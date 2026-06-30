"""Auto-translate non-abbreviation dictionary terms to Chinese via DeepSeek.

Reads dictionary.md, identifies terms WITHOUT abbreviations, sends them to
DeepSeek for Chinese translation, and updates the file in-place with bilingual
entries: "- english term | 中文翻译"

Terms WITH abbreviations (e.g. "- stimulated Raman scattering (SRS)") are
left untouched — abbreviations are universal and don't need translation.
"""
import os
import re
import sys
import configparser
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DICT_PATH = PROJECT_ROOT / "dictionary.md"
SECRETS_SRC = Path(r"C:\Users\Yifan\OneDrive\Opencode_workspace\audio_transcribe_notes\config.ini")

ABBR_RE = re.compile(r"^(.+?)\s*\(([A-Z][A-Za-z0-9-]+)\)$")


def load_secrets() -> tuple[str, str]:
    cp = configparser.ConfigParser()
    cp.read(SECRETS_SRC, encoding="utf-8")
    key = cp["defaults"].get("deepseek_api_key", "").strip()
    model = cp["defaults"].get("deepseek_model", "deepseek-chat").strip()
    if not key:
        raise RuntimeError(f"deepseek_api_key empty in {SECRETS_SRC}")
    return key, model


def parse_dictionary(text: str) -> list[dict]:
    """Parse dictionary.md into structured entries.

    Returns list of {line, entry, has_abbr, is_term} dicts.
    Non-term lines (headers, blanks) have is_term=False.
    """
    entries = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped.startswith("- "):
            entries.append({"raw": line, "is_term": False})
            continue
        entry = stripped[2:].strip()
        entries.append({
            "raw": line,
            "entry": entry,
            "is_term": True,
            "has_abbr": bool(ABBR_RE.match(entry)),
            "has_translation": " | " in entry,
        })
    return entries


def collect_translatable(entries: list[dict]) -> list[tuple[int, str]]:
    """Return [(index, english_term)] for terms that need translation."""
    result = []
    for i, e in enumerate(entries):
        if not e.get("is_term"):
            continue
        if e.get("has_abbr"):
            continue  # abbreviation terms don't need translation
        if e.get("has_translation"):
            continue  # already has Chinese translation
        result.append((i, e["entry"]))
    return result


def translate_batch(terms: list[str], api_key: str, model: str) -> dict[str, str]:
    """Send terms to DeepSeek, return {english: chinese} mapping.

    Terms without standard Chinese translations get "SKIP".
    """
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(terms))

    system_prompt = (
        "You are a scientific terminology translator specializing in biophotonics, "
        "vibrational microscopy, Raman spectroscopy, and photothermal imaging. "
        "Translate English scientific terms to their standard Chinese equivalents "
        "used in Chinese academic discussions.\n\n"
        "Output ONE line per term: english term || chinese translation\n"
        "Use the STANDARD Chinese scientific term, not literal translation.\n"
        "If no Chinese equivalent exists, output: english term || SKIP"
    )

    user_prompt = (
        f"Translate these {len(terms)} terms. Output exactly {len(terms)} lines, "
        f"same order, one per line:\n\n{numbered}"
    )

    print(f"Sending {len(terms)} terms to DeepSeek ({model})...")
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
    print(f"Tokens: {resp.usage.prompt_tokens:,} in, {resp.usage.completion_tokens:,} out")
    print(f"finish_reason: {resp.choices[0].finish_reason}")
    print(f"Response lines: {len(content.split(chr(10)))}")

    # Save raw for debugging
    raw_path = PROJECT_ROOT / "scripts" / "dict_translate_raw.txt"
    raw_path.write_text(content, encoding="utf-8")

    # Parse response — DeepSeek may prepend numbers; strip them
    translations = {}
    for line in content.split("\n"):
        line = re.sub(r"^\d+\.\s*", "", line.strip())
        if "||" not in line:
            continue
        parts = line.split("||", 1)
        en = parts[0].strip()
        zh = parts[1].strip()
        if zh.upper() == "SKIP" or not zh:
            continue
        translations[en] = zh

    print(f"Parsed translations: {len(translations)}\n")
    return translations


def main():
    text = DICT_PATH.read_text(encoding="utf-8")
    entries = parse_dictionary(text)
    translatable = collect_translatable(entries)

    print(f"Dictionary: {sum(1 for e in entries if e.get('is_term'))} terms total")
    print(f"  With abbreviation (skip): {sum(1 for e in entries if e.get('has_abbr'))}")
    print(f"  Already translated (skip): {sum(1 for e in entries if e.get('has_translation'))}")
    print(f"  Need translation: {len(translatable)}")
    print()

    if not translatable:
        print("Nothing to translate. All terms already have translations or abbreviations.")
        return

    # Load secrets + translate
    api_key, model = load_secrets()
    terms_only = [t for _, t in translatable]
    translations = translate_batch(terms_only, api_key, model)

    # Apply translations
    updated_count = 0
    skipped_count = 0
    for idx, en_term in translatable:
        zh = translations.get(en_term)
        if zh:
            entries[idx]["entry"] = f"{en_term} | {zh}"
            updated_count += 1
        else:
            skipped_count += 1

    print(f"Translated: {updated_count}")
    print(f"Skipped (no Chinese equivalent): {skipped_count}")
    print()

    # Rebuild file
    lines = []
    for e in entries:
        if e.get("is_term"):
            lines.append(f"- {e['entry']}")
        else:
            lines.append(e["raw"])
    DICT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Updated: {DICT_PATH}")

    # Preview some translations
    print("\n=== Sample translations ===")
    shown = 0
    for idx, en_term in translatable:
        zh = translations.get(en_term)
        if zh and shown < 20:
            print(f"  {en_term}  →  {zh}")
            shown += 1


if __name__ == "__main__":
    main()
