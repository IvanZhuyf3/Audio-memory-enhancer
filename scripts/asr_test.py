r"""Standalone Qwen3-ASR test runner — transcribes a Plaud recording and saves
raw segments as JSON for comparison. Does NOT write to the Obsidian vault.

Usage:
    & "C:\Users\Yifan\venvs\audio_transcribe\Scripts\python.exe" scripts\asr_test.py <plaud_id>

Output:
    Temp/opencode/meeting_qwen.json  — raw segments [{start, end, text, speaker}, ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plaud_sync
import transcribe_local
import routing
import classify

OUT_DIR = Path(r"C:\Users\Yifan\AppData\Local\Temp\opencode")
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/asr_test.py <plaud_id>")
        return 1
    rec_id = sys.argv[1]

    # 1. Fetch recording metadata + download audio.
    print(f"[asr_test] Fetching metadata for {rec_id}...")
    recordings = plaud_sync.list_recordings()
    rec = next((r for r in recordings if r["id"] == rec_id), None)
    if not rec:
        print(f"[error] recording {rec_id} not found")
        return 1
    duration_min = rec.get("duration", 0) / 60000
    print(f"  Title: {rec.get('filename')}")
    print(f"  Duration: {duration_min:.1f} min")

    cache_dir = PROJECT_ROOT / "cache"
    cache_dir.mkdir(exist_ok=True)
    audio_path = cache_dir / f"{rec_id}.mp3"
    if not audio_path.exists():
        print(f"[asr_test] Downloading audio...")
        plaud_sync.download_audio(rec_id, audio_path)
    else:
        print(f"[asr_test] Audio cached at {audio_path}")

    # 2. Build ASR context from dictionary + title.
    # Load secrets from sibling config for DeepSeek cleaning.
    import configparser
    secrets_src = PROJECT_ROOT.parent / "audio_transcribe_notes" / "config.ini"
    cp = configparser.ConfigParser()
    cp.read(secrets_src, encoding="utf-8")
    d = cp["defaults"] if "defaults" in cp else {}
    hf_token = d.get("hf_token", "").strip()
    deepseek_key = d.get("deepseek_api_key", "").strip()
    deepseek_model = d.get("deepseek_model", "deepseek-v4-flash").strip()

    asr_context = transcribe_local.build_asr_context(
        dictionary_path=PROJECT_ROOT / "dictionary.md",
        recording_title=rec.get("filename"),
    )

    # 3. Run Qwen3-ASR.
    print(f"[asr_test] Starting Qwen3-ASR transcription ({duration_min:.1f} min audio)...")
    result = transcribe_local.transcribe(
        audio_path,
        language="zh",  # force Chinese to avoid code-switching confusion
        qwen_model="Qwen/Qwen3-ASR-1.7B",
        forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
        device="cuda",
        hf_token=hf_token,
        deepseek_api_key=deepseek_key,
        deepseek_model=deepseek_model,
        dictionary_path=PROJECT_ROOT / "dictionary.md",
        gpu_lock=str(Path.home() / "venvs" / ".gpu_lock"),
        log_dir=cache_dir,
        context=asr_context,
    )

    # 4. Save raw segments.
    out_path = OUT_DIR / "meeting_qwen.json"
    out_path.write_text(
        json.dumps(result["segments"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    speakers = set(s.get("speaker", "") for s in result["segments"])
    total_chars = sum(len(s.get("text", "")) for s in result["segments"])
    print(f"\n[asr_test] Done:")
    print(f"  Segments: {len(result['segments'])}")
    print(f"  Speakers: {len(speakers)} ({', '.join(sorted(speakers))})")
    print(f"  Total text: {total_chars} chars")
    print(f"  Language: {result.get('language', 'unknown')}")
    print(f"  Saved to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
