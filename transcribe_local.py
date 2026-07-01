"""Local Qwen3-ASR transcription for long recordings (>= short_threshold_min).

Ports the proven ASR pipeline from audio_transcribe_notes/transcribe.py:
  Qwen3-ASR + ForcedAligner (char-level timestamps, 180s chunk limit)
  → silero-vad chunking for longer audio
  → pyannote speaker diarization
  → DeepSeek dictionary-based AI cleaning

Public interface:
    transcribe(audio_path, ...) -> {
        segments:            [{start, end, text, speaker}, ...],
        transcript_markdown: str,    # **Speaker** (HH:MM:SS)\\ntext blocks
        transcript_text:     str,    # flat text for classify_long()
        speakers:            int,
        language:            str,
    }

GPU coordination: acquires a file lock (config.gpu_lock) so this pipeline and
audio_transcribe_notes/monitor.py never run Qwen3-ASR concurrently.
"""

from __future__ import annotations

import gc
import json
import os
import re
import threading
import time
from datetime import timedelta
from pathlib import Path

CHUNK_TIMEOUT_S = 600        # per-chunk transcribe timeout (10 min)
MAX_CHUNK_SECONDS = 180.0    # ForcedAligner char-timestamp limit


# ── GPU file lock ───────────────────────────────────────────────────

class GpuLock:
    """Simple cross-process file lock so only one GPU job runs at a time.

    Uses atomic file creation (O_CREAT | O_EXCL). On conflict, polls until
    the holder PID is gone (stale-lock recovery) or timeout expires.
    """

    def __init__(self, path: str | Path, timeout_s: float = 3600.0):
        self.path = Path(path)
        self.timeout_s = timeout_s
        self._fd: int | None = None

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()

    def acquire(self) -> None:
        start = time.monotonic()
        while True:
            try:
                self._fd = os.open(
                    str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY
                )
                os.write(self._fd, str(os.getpid()).encode("ascii"))
                return
            except FileExistsError:
                # Check if the holder is still alive (stale-lock recovery).
                try:
                    content = self.path.read_text(encoding="utf-8").strip()
                    holder_pid = int(content) if content else 0
                    if holder_pid and not _pid_alive(holder_pid):
                        self.path.unlink(missing_ok=True)
                        continue
                except (ValueError, OSError):
                    pass
                if time.monotonic() - start > self.timeout_s:
                    raise TimeoutError(
                        f"Could not acquire GPU lock {self.path} within "
                        f"{self.timeout_s}s. Another GPU job may be running."
                    )
                time.sleep(5)

    def release(self) -> None:
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None
        self.path.unlink(missing_ok=True)


def _pid_alive(pid: int) -> bool:
    """Check if a process is running (cross-platform best-effort)."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


# ── Language mapping ────────────────────────────────────────────────

def _map_language(code: str | None) -> str | None:
    """Map short CLI language codes to Qwen3-ASR full names."""
    mapping = {
        "zh": "Chinese", "en": "English", "ja": "Japanese",
        "ko": "Korean", "fr": "French", "de": "German", "es": "Spanish",
    }
    if not code or code == "auto":
        return None
    return mapping.get(code, code.title())


# ── VAD chunking (port of _vad_split) ───────────────────────────────

def _vad_split(wav, max_seconds: float = MAX_CHUNK_SECONDS):
    """Split audio at VAD speech boundaries into chunks <= max_seconds.

    Returns list of (start_sample, end_sample, wav_chunk).
    """
    import numpy as np  # noqa: F401
    from silero_vad import load_silero_vad, get_speech_timestamps

    sr = 16000
    max_samples = int(max_seconds * sr)

    vad_model = load_silero_vad(onnx=True)
    speech_ts = get_speech_timestamps(
        wav, vad_model, sampling_rate=sr,
        min_speech_duration_ms=1500, min_silence_duration_ms=500,
    )

    if not speech_ts:
        return [(0, len(wav), wav)]

    split_points = [0]
    for seg in speech_ts:
        split_points.append(seg["start"])
        split_points.append(seg["end"])
    split_points.append(len(wav))
    split_points = sorted(set(split_points))

    chunks = []
    chunk_start = split_points[0]
    for sp in split_points[1:]:
        if sp - chunk_start >= max_samples and sp > chunk_start:
            chunks.append((chunk_start, sp, wav[chunk_start:sp]))
            chunk_start = sp
    if chunk_start < len(wav):
        remaining = wav[chunk_start:]
        if chunks and len(remaining) < sr * 2:
            last_s, last_e, last_w = chunks[-1]
            chunks[-1] = (last_s, len(wav), wav[last_s:len(wav)])
        else:
            chunks.append((chunk_start, len(wav), remaining))

    return chunks if chunks else [(0, len(wav), wav)]


# ── Sentence segmentation (port of _build_sentence_segments) ────────

SENTENCE_END = set("。！？!?.")
PUNCT = set("，。！？、；：""''（）【】《》…—,.!?;:'\"()[]{}/\u3000 ")


def _build_sentence_segments(text: str, time_stamps: list, offset_s: float = 0.0) -> list[dict]:
    """Split transcription text into sentence-level segments with timing.

    text: full transcription with punctuation (from result.text).
    time_stamps: word-level ForcedAlignItem list (each has .text, .start_time, .end_time).
    offset_s: seconds offset for this chunk within the full audio.
    """
    word_timing = []
    ts_idx = 0
    text_pos = 0

    while ts_idx < len(time_stamps) and text_pos < len(text):
        if text[text_pos] in PUNCT:
            text_pos += 1
            continue

        ts = time_stamps[ts_idx]
        ts_word = ts.text
        ts_word_len = len(ts_word)

        if text[text_pos:text_pos + ts_word_len].lower() == ts_word.lower():
            word_timing.append((
                text_pos,
                text_pos + ts_word_len,
                ts.start_time + offset_s,
                ts.end_time + offset_s,
            ))
            text_pos += ts_word_len
            ts_idx += 1
        else:
            word_timing.append((
                text_pos,
                text_pos + 1,
                ts.start_time + offset_s,
                ts.end_time + offset_s,
            ))
            text_pos += 1
            ts_idx += 1

    if not word_timing:
        return []

    segments = []
    seg_start = 0

    for i in range(len(text)):
        if text[i] in SENTENCE_END or i == len(text) - 1:
            seg_text = text[seg_start:i + 1].strip()
            if not seg_text:
                seg_start = i + 1
                continue

            first_ts = None
            last_ts = None
            for ws, we, wt_start, wt_end in word_timing:
                if ws >= seg_start and ws <= i:
                    if first_ts is None:
                        first_ts = (wt_start, wt_end)
                    last_ts = (wt_start, wt_end)

            if first_ts and last_ts:
                segments.append({
                    "start": first_ts[0],
                    "end": last_ts[1],
                    "text": seg_text,
                })
            seg_start = i + 1

    return segments


# ── Speaker assignment (port of _assign_speakers) ───────────────────

def _assign_speakers(
    segments: list[dict],
    speaker_turns: list[tuple[float, float, str]],
) -> list[dict]:
    """Assign speaker labels to segments based on time overlap."""
    if not speaker_turns:
        for seg in segments:
            seg["speaker"] = "Speaker ?"
        return segments

    for seg in segments:
        best_speaker = "Speaker ?"
        best_overlap = 0.0
        for sp_start, sp_end, sp_label in speaker_turns:
            overlap = max(0, min(seg["end"], sp_end) - max(seg["start"], sp_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = sp_label
        seg["speaker"] = best_speaker

    return segments


# ── Core ASR pipeline (adapted from run_qwen3_asr) ──────────────────

def _run_asr(
    audio_path: Path,
    hf_token: str,
    language: str | None = None,
    model_name: str = "Qwen/Qwen3-ASR-1.7B",
    forced_aligner: str = "Qwen/Qwen3-ForcedAligner-0.6B",
    device: str = "cuda",
    log_dir: Path | None = None,
    context: str = "",
) -> tuple[list[dict], int, str]:
    """Run Qwen3-ASR + diarization. Returns (segments, speaker_count, language)."""
    import torch
    from qwen_asr import Qwen3ASRModel

    import librosa
    print("  Loading audio...")
    wav, sr = librosa.load(str(audio_path), sr=16000, mono=True)
    duration = len(wav) / sr
    print(f"  Duration: {duration:.1f}s")

    # VAD split if needed.
    if duration > MAX_CHUNK_SECONDS:
        print(f"  Splitting audio into <= {MAX_CHUNK_SECONDS:.0f}s chunks (VAD)...")
        chunks = _vad_split(wav, max_seconds=MAX_CHUNK_SECONDS)
        print(f"  Split into {len(chunks)} chunks")
    else:
        chunks = [(0, len(wav), wav)]

    # Load model.
    print(f"  Loading model '{model_name}' with ForcedAligner on {device}...")
    model = Qwen3ASRModel.from_pretrained(
        model_name,
        forced_aligner=forced_aligner,
        forced_aligner_kwargs=dict(dtype=torch.bfloat16, device_map=device),
        dtype=torch.bfloat16,
        device_map=device,
        max_new_tokens=2048,
    )

    lang_param = _map_language(language)
    all_segments = []
    failed_chunks = []
    log_lines = []
    detected_lang = language or "unknown"

    for i, (start_s, end_s, chunk_wav) in enumerate(chunks):
        offset_s = start_s / sr
        chunk_dur = (end_s - start_s) / sr
        chunk_tag = f"  Chunk {i+1}/{len(chunks)}" if len(chunks) > 1 else "  Transcribing"
        print(f"{chunk_tag} ({chunk_dur:.1f}s)...")

        chunk_result = [None]
        chunk_error = [None]

        def _transcribe_chunk():
            try:
                chunk_result[0] = model.transcribe(
                    audio=(chunk_wav, sr),
                    context=context,
                    language=lang_param,
                    return_time_stamps=True,
                )
            except Exception as exc:
                chunk_error[0] = exc

        t = threading.Thread(target=_transcribe_chunk)
        t.start()
        t.join(timeout=CHUNK_TIMEOUT_S)

        if t.is_alive():
            log_lines.append(
                f"CHUNK {i+1}/{len(chunks)} | offset={offset_s:.1f}s | "
                f"dur={chunk_dur:.1f}s | TIMEOUT after {CHUNK_TIMEOUT_S}s"
            )
            failed_chunks.append(i + 1)
            print(f"    TIMEOUT after {CHUNK_TIMEOUT_S}s, skipping")
            continue

        if chunk_error[0] is not None:
            log_lines.append(
                f"CHUNK {i+1}/{len(chunks)} | offset={offset_s:.1f}s | "
                f"dur={chunk_dur:.1f}s | ERROR: {chunk_error[0]}"
            )
            failed_chunks.append(i + 1)
            print(f"    Error: {chunk_error[0]}, skipping")
            continue

        results = chunk_result[0]
        seg_count = 0
        for r in results:
            if r.text.strip() and r.time_stamps:
                segs = _build_sentence_segments(r.text, r.time_stamps, offset_s)
                all_segments.extend(segs)
                seg_count += len(segs)
        log_lines.append(
            f"CHUNK {i+1}/{len(chunks)} | offset={offset_s:.1f}s | "
            f"dur={chunk_dur:.1f}s | OK | {seg_count} segments"
        )
        if results:
            detected_lang = results[0].language

    print(f"  Detected language: {detected_lang}")

    # Write chunk log.
    if log_lines and log_dir:
        log_path = log_dir / f"chunk_log_{audio_path.stem}.txt"
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"# Chunk log for {audio_path.name}\n")
            f.write(f"# {len(chunks)} chunks, {len(failed_chunks)} failed\n\n")
            for line in log_lines:
                f.write(line + "\n")
        if failed_chunks:
            print(f"  {len(failed_chunks)} chunk(s) failed: {failed_chunks}")
            print(f"  Chunk log: {log_path}")
        else:
            print(f"  All {len(chunks)} chunks OK. Log: {log_path}")

    # Free ASR model from GPU.
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    if not all_segments:
        print("  Warning: No transcription produced")
        return [], 0, detected_lang

    # Speaker diarization.
    speaker_turns = []
    print("  Running speaker diarization...")
    try:
        from pyannote.audio import Pipeline as PyannotePipeline
        diarize_pipeline = PyannotePipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            token=hf_token,
        ).to(torch.device(device))
        audio_data = {
            "waveform": torch.from_numpy(wav[None, :]),
            "sample_rate": sr,
        }
        diarization = diarize_pipeline(audio_data)
        for turn, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
            speaker_turns.append((turn.start, turn.end, speaker))
        n_speakers = len(set(s for _, _, s in speaker_turns))
        print(f"  Detected {n_speakers} speaker(s)")

        # Try voiceprint matching against registry.
        try:
            import speaker_id as sid
            embed_model = sid.load_embedding_model(hf_token, device)
            embeds_by_speaker = sid.extract_speaker_embeddings(
                wav, sr, speaker_turns, embed_model,
            )
            registry = sid.load_registry()
            name_map = sid.identify_speakers(embeds_by_speaker, registry)
            # Accumulate embeddings for future matching.
            sid.accumulate_embeddings(embeds_by_speaker, name_map, registry)
            sid.save_registry(registry)
            # Remap speaker labels.
            new_turns = []
            for start, end, label in speaker_turns:
                new_turns.append((start, end, name_map.get(label, label)))
            renamed = sum(1 for k, v in name_map.items() if v != k)
            if renamed:
                print(f"  Voiceprint matched {renamed} speaker(s):")
                for old, new in sorted(name_map.items()):
                    if old != new:
                        print(f"    {old} → {new}")
            speaker_turns = new_turns
        except Exception as e:
            print(f"  Voiceprint matching skipped: {e}")
        del diarize_pipeline
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    except Exception as e:
        n_speakers = 0
        print(f"  Warning: Diarization failed ({e}), skipping speaker labels")

    segments = _assign_speakers(all_segments, speaker_turns)
    print(f"  Produced {len(segments)} segments")
    return segments, n_speakers, detected_lang


# ── AI cleaning (adapted from ai_clean) ─────────────────────────────

def load_dictionary(dictionary_path: Path) -> str:
    """Read the domain dictionary file. Returns its full text."""
    if not dictionary_path.exists():
        return ""
    return dictionary_path.read_text(encoding="utf-8").strip()


def build_asr_context(
    dictionary_path: str | Path | None = None,
    project_keywords: list[str] | None = None,
    recording_title: str | None = None,
    max_chars: int = 15000,
) -> str:
    """Build a Qwen3-ASR context string from dictionary terms + project keywords + title.

    The context becomes the model's system message, biasing recognition toward
    domain-specific terms BEFORE transcription (pre-correction). This complements
    the post-correction DeepSeek ai_clean step — context prevents errors, ai_clean
    catches the rest.

    Dictionary format: one entry per line, `- full term (ABBR)` or `- term`.
    Both the full form and abbreviation are extracted and included.
    """
    terms: list[str] = []

    # 1. Dictionary terms (both full forms + abbreviations)
    if dictionary_path:
        p = Path(dictionary_path)
        if p.exists():
            for line in p.read_text(encoding="utf-8").split("\n"):
                line = line.strip()
                if not line.startswith("- "):
                    continue
                entry = line[2:].strip()
                # Bilingual entries use " | " separator:
                #   "- lipid droplet | 脂滴"
                # Each part is processed independently for abbreviations.
                for part in entry.split(" | "):
                    part = part.strip()
                    if not part:
                        continue
                    m = re.match(r"^(.+?)\s*\(([A-Z][A-Za-z0-9-]+)\)$", part)
                    if m:
                        terms.append(m.group(1).strip())  # full term
                        terms.append(m.group(2).strip())  # abbreviation
                    else:
                        terms.append(part)

    # 2. Project keywords + aliases (deduped, case-insensitive)
    if project_keywords:
        seen = {t.lower() for t in terms}
        for kw in project_keywords:
            kw = kw.strip()
            if kw and kw.lower() not in seen:
                terms.append(kw)
                seen.add(kw.lower())

    # 3. Recording title (names/topics from Plaud metadata)
    if recording_title:
        title = recording_title.strip()
        if title and len(title) < 100:
            terms.append(title)

    if not terms:
        return ""

    context = "Domain-specific terms that may appear in this audio: " + ", ".join(terms)
    if len(context) > max_chars:
        context = context[:max_chars].rsplit(", ", 1)[0]
    return context


def ai_clean(
    segments: list[dict],
    api_key: str,
    dictionary_text: str,
    model: str = "deepseek-v4-flash",
    dictionary_path: Path | None = None,
) -> list[dict]:
    """Use DeepSeek to correct mis-transcribed terms in the transcript.

    Operates directly on segments (simpler than the sibling project's item-list).
    If dictionary_path is provided, newly identified terms are appended to it.
    """
    if not api_key or not segments:
        return segments
    from openai import OpenAI

    numbered_lines = []
    seg_indices = []
    for i, seg in enumerate(segments):
        text = (seg.get("text") or "").strip()
        if text:
            seg_indices.append(i)
            numbered_lines.append(f"[{len(numbered_lines) + 1}] {text}")

    if not numbered_lines:
        return segments

    transcript = "\n".join(numbered_lines)
    dict_section = (
        f"Reference dictionary:\n{dictionary_text}\n"
        if dictionary_text
        else "No reference dictionary provided.\n"
    )

    system_prompt = (
        "You are correcting ASR transcription errors in a professional transcript. "
        "The speaker uses domain-specific terms and abbreviations that are often mis-transcribed.\n\n"
        f"{dict_section}\n"
        "The dictionary lists the correct forms of terms. Use it to spot near-miss ASR errors: "
        "words that sound similar to a dictionary term are likely mis-transcriptions "
        '(e.g. "Ramen" -> "Raman", "S R S" -> "SRS").\n\n'
        "Below is the transcript with numbered segments. Correct any mis-transcribed words or phrases. "
        "Output the corrected transcript using the same numbered format: "
        "[1] corrected text\\n[2] corrected text\\n...\n"
        "Only change words that are clearly errors. Do not rewrite or paraphrase. "
        "Output ALL segments, not just the corrected ones.\n\n"
        "After the corrected transcript, list any DOMAIN-SPECIFIC terms, abbreviations, or named concepts "
        "you encountered that are NOT already in the reference dictionary. "
        "Use this exact format:\n"
        "---NEW TERMS---\n"
        "- full term (ABBR)\n\n"
        "If no new terms found, omit the ---NEW TERMS--- section."
    )

    print("  AI cleaning with DeepSeek...")
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": transcript},
            ],
            temperature=0.0,
            extra_body={"thinking": {"type": "disabled"}},
        )
    except Exception as e:
        print(f"  Warning: AI cleaning failed ({e}), using raw transcript")
        return segments

    raw = response.choices[0].message.content.strip()

    new_terms_section = ""
    if "---NEW TERMS---" in raw:
        parts = raw.split("---NEW TERMS---", 1)
        corrected_text = parts[0].strip()
        new_terms_section = parts[1].strip()
    else:
        corrected_text = raw

    corrections = {}
    for line in corrected_text.split("\n"):
        line = line.strip()
        match = re.match(r"\[(\d+)\]\s*(.*)", line)
        if match:
            seg_num = int(match.group(1))
            text = match.group(2).strip()
            corrections[seg_num] = text

    changed = 0
    for line_num, seg_idx in enumerate(seg_indices, 1):
        if line_num in corrections:
            new_text = corrections[line_num]
            if new_text != segments[seg_idx].get("text"):
                segments[seg_idx]["text"] = new_text
                changed += 1

    print(f"  AI cleaning done: {changed} segment(s) corrected")

    # Save new terms to dictionary.
    if new_terms_section and dictionary_path:
        _append_new_terms(dictionary_path, new_terms_section, dictionary_text)

    return segments


def _append_new_terms(dictionary_path: Path, new_terms_section: str, existing_dict_text: str) -> None:
    """Append new terms to dictionary.md, skipping duplicates."""
    existing_terms = set()
    for line in existing_dict_text.split("\n"):
        line = line.strip()
        if line.startswith("- "):
            existing_terms.add(line[2:].strip().lower())

    to_add = []
    for line in new_terms_section.split("\n"):
        line = line.strip()
        if line.startswith("- "):
            key = line[2:].strip().lower()
            if key and key not in existing_terms:
                to_add.append(line)
                existing_terms.add(key)

    if not to_add:
        return

    existing = dictionary_path.read_text(encoding="utf-8").rstrip("\n")
    dictionary_path.write_text(existing + "\n" + "\n".join(to_add) + "\n", encoding="utf-8")
    print(f"  Added {len(to_add)} new term(s) to {dictionary_path.name}")


# ── Main entry point ────────────────────────────────────────────────

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
    gpu_lock: str | Path | None = None,
    gpu_lock_timeout_s: float = 3600.0,
    log_dir: str | Path | None = None,
    skip_clean: bool = False,
    context: str = "",
) -> dict:
    """Transcribe a long recording with speaker diarization + optional AI cleaning.

    The `context` string is passed to Qwen3-ASR's system message, biasing
    recognition toward domain terms (pre-correction). Build it via
    `build_asr_context()` from the dictionary + project keywords + title.

    Returns {segments, transcript_markdown, transcript_text, speakers, language}.
    """
    audio_path = Path(audio_path)
    dict_path = Path(dictionary_path) if dictionary_path else None

    if context:
        print(f"  ASR context: {context[:120]}{'...' if len(context) > 120 else ''}")

    # Acquire GPU lock for the whole ASR + diarization phase.
    lock = GpuLock(gpu_lock, timeout_s=gpu_lock_timeout_s) if gpu_lock else None
    if lock:
        print(f"  Acquiring GPU lock ({gpu_lock})...")
        lock.acquire()
        try:
            segments, speakers, detected_lang = _run_asr(
                audio_path, hf_token, language, qwen_model, forced_aligner, device,
                log_dir=Path(log_dir) if log_dir else None,
                context=context,
            )
        finally:
            lock.release()
    else:
        segments, speakers, detected_lang = _run_asr(
            audio_path, hf_token, language, qwen_model, forced_aligner, device,
            log_dir=Path(log_dir) if log_dir else None,
            context=context,
        )

    if not segments:
        return {
            "segments": [],
            "transcript_markdown": "",
            "transcript_text": "",
            "speakers": 0,
            "language": detected_lang,
        }

    # Optional AI cleaning.
    if not skip_clean and deepseek_api_key and dict_path:
        dict_text = load_dictionary(dict_path)
        segments = ai_clean(
            segments, deepseek_api_key, dict_text,
            model=deepseek_model, dictionary_path=dict_path,
        )

    return {
        "segments": segments,
        "transcript_markdown": segments_to_markdown(segments),
        "transcript_text": segments_to_text(segments),
        "speakers": speakers,
        "language": detected_lang,
    }


# ── Renderers ───────────────────────────────────────────────────────

def segments_to_markdown(segments: list[dict]) -> str:
    """Render [{start, end, text, speaker}, ...] as Meeting-Notes-style markdown:
        **Speaker 1** (HH:MM:SS)
        text...
    """
    lines = []
    for seg in segments:
        speaker = seg.get("speaker") or "Speaker ?"
        ts = format_timestamp(seg.get("start", 0))
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        lines.append(f"**{speaker}** ({ts})")
        lines.append(text)
        lines.append("")
    return "\n".join(lines).strip()


def segments_to_text(segments: list[dict]) -> str:
    """Flatten segments to plain text for the classifier."""
    return " ".join(
        (s.get("text") or "").strip()
        for s in segments
        if (s.get("text") or "").strip()
    )


def format_timestamp(seconds: float) -> str:
    h = int(seconds) // 3600
    m = (int(seconds) % 3600) // 60
    s = int(seconds) % 60
    return f"{h:02d}:{m:02d}:{s:02d}"
