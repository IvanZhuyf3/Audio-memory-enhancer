"""Local Qwen3-ASR transcription for long recordings (>= short_threshold_min).

PHASE 3 MODULE — currently a stub. The real implementation will port
run_qwen3_asr() + ai_clean() + helpers from:
    C:/Users/Yifan/OneDrive/Opencode_workspace/audio_transcribe_notes/transcribe.py

The upstream pipeline is proven (Qwen3-ASR + ForcedAligner + silero-vad chunking
+ pyannote speaker diarization + DeepSeek dictionary cleaning). Port plan:
  - copy: _vad_split, _build_sentence_segments, _assign_speakers, _map_language,
           run_qwen3_asr, ai_clean, load_dictionary, format_timestamp
  - copy: dictionary.md (already in this repo's root)
  - adapt: emit segments in the shape note_templates.render_long_meeting() expects
  - coordinate: acquire GPU lock before loading models (serialize vs monitor.py)

Public interface (stable — Phase 3 fills the body):
    transcribe(audio_path, config) -> {
        segments:   [{start, end, text, speaker}, ...],
        transcript_markdown: str,    # rendered **Speaker** (HH:MM:SS)\\ntext blocks
        transcript_text: str,        # flat text for classify_long()
        speakers:   int,
    }
"""

from __future__ import annotations

from pathlib import Path


def transcribe(
    audio_path: str | Path,
    *,
    language: str = "auto",
    qwen_model: str = "Qwen/Qwen3-ASR-1.7B",
    forced_aligner: str = "Qwen/Qwen3-ForcedAligner-0.6B",
    device: str = "cuda",
    hf_token: str = "",
    deepseek_api_key: str = "",
    deepseek_model: str = "deepseek-v4-flash",
    dictionary_path: str | Path | None = None,
) -> dict:
    """Transcribe a long recording with speaker diarization + AI cleaning.

    Phase 3 will implement this by porting run_qwen3_asr() from
    audio_transcribe_notes/transcribe.py. Returns the dict described in the
    module docstring.
    """
    raise NotImplementedError(
        "transcribe_local.transcribe() is a Phase 3 stub. "
        "Port run_qwen3_asr() from audio_transcribe_notes/transcribe.py to enable "
        "the long-recording path."
    )


def segments_to_markdown(segments: list[dict]) -> str:
    """Render [{start, end, text, speaker}, ...] as Meeting-Notes-style markdown:
        **Speaker 1** (HH:MM:SS)
        text...
    """
    lines = []
    for seg in segments:
        speaker = seg.get("speaker") or "Speaker ?"
        ts = _format_timestamp(seg.get("start", 0))
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        lines.append(f"**{speaker}** ({ts})")
        lines.append(text)
        lines.append("")
    return "\n".join(lines).strip()


def segments_to_text(segments: list[dict]) -> str:
    """Flatten segments to plain text for the classifier."""
    return " ".join((s.get("text") or "").strip() for s in segments if (s.get("text") or "").strip())


def _format_timestamp(seconds: float) -> str:
    h = int(seconds) // 3600
    m = (int(seconds) % 3600) // 60
    s = int(seconds) % 60
    return f"{h:02d}:{m:02d}:{s:02d}"
