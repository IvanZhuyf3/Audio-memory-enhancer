"""Speaker voiceprint registry + matching for Qwen3-ASR pipeline.

Uses pyannote/embedding for voiceprint extraction. Maintains a
speaker_registry.json with {name: {embeddings: [...], sample_count, last_seen}}.

During diarization: extract embeddings per speaker, match against registry,
replace generic "SPEAKER_00" labels with real names. Unknown speakers
accumulate embeddings under "_unlabeled_*" keys for later enrollment.

Also provides CLI for enrollment from audio clips:
  python speaker_id.py enroll --name "程继新" --wav path/to/sample.wav
  python speaker_id.py list
  python speaker_id.py rename "_unlabeled_SPEAKER_00" "朱一凡"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# numpy is imported lazily inside functions (only available in GPU venv).

# Default cosine similarity threshold for matching.
# pyannote community-1 model: 0.6-0.7 is conservative.
MATCH_THRESHOLD = 0.65
# Minimum seconds of audio to extract per speaker for embedding.
MIN_EMBED_AUDIO_S = 3.0
# Max embeddings stored per speaker (FIFO).
MAX_EMBEDDINGS_PER_SPEAKER = 10

DEFAULT_REGISTRY = Path(__file__).parent / "speaker_registry.json"


# ── Embedding extraction ────────────────────────────────────────────

def load_embedding_model(hf_token: str, device: str = "cuda"):
    """Load pyannote/embedding model. Returns a callable that takes
    {waveform: tensor, sample_rate: int} -> embedding tensor (512-dim).
    """
    import torch
    from pyannote.audio import Inference as EmbeddingInference
    model = EmbeddingInference(
        "pyannote/embedding",
        window="whole",
        token=hf_token,
    )
    model.to(torch.device(device))
    return model


def extract_embedding(
    waveform: "np.ndarray",
    sample_rate: int,
    model,
    start_s: float = 0.0,
    duration_s: float | None = None,
) -> "np.ndarray | None":
    """Extract speaker embedding from an audio segment.

    Args:
        waveform: 1-D float32 numpy array at sample_rate Hz.
        sample_rate: sample rate of waveform.
        model: loaded pyannote embedding model.
        start_s: start time in seconds (0-based within waveform).
        duration_s: length to extract. If None, uses rest of waveform.
    Returns: 1-D numpy float32 array (512-dim), or None if segment too short.
    """
    import numpy as np
    import torch

    start_sample = int(start_s * sample_rate)
    if duration_s is not None:
        end_sample = start_sample + int(duration_s * sample_rate)
    else:
        end_sample = len(waveform)

    segment = waveform[start_sample:end_sample]
    if len(segment) < int(MIN_EMBED_AUDIO_S * sample_rate):
        return None

    input_dict = {
        "waveform": torch.from_numpy(segment[None, :]).float(),
        "sample_rate": sample_rate,
    }
    embedding: torch.Tensor = model(input_dict)
    return embedding.detach().cpu().numpy().squeeze()


def extract_speaker_embeddings(
    wav: np.ndarray,
    sr: int,
    speaker_turns: list[tuple[float, float, str]],
    embedding_model,
    max_per_speaker: int = 3,
) -> dict[str, list[np.ndarray]]:
    """Extract embeddings for each speaker from diarized audio.

    Groups segments by speaker label, picks up to max_per_speaker
    non-overlapping 3-5s clips from each speaker's total speaking time.

    Returns: {speaker_label: [embedding_array, ...]}
    """
    by_speaker: dict[str, list[tuple[float, float]]] = {}
    for start, end, label in speaker_turns:
        by_speaker.setdefault(label, []).append((start, end))

    result: dict[str, list[np.ndarray]] = {}
    for label, segments in by_speaker.items():
        # Pick representative clips: spread across total speaking time.
        clips = _pick_embed_clips(segments, max_per_speaker)
        embeddings = []
        for clip_start, clip_end in clips:
            dur = clip_end - clip_start
            emb = extract_embedding(wav, sr, embedding_model, clip_start, dur)
            if emb is not None:
                embeddings.append(emb)
        if embeddings:
            result[label] = embeddings
    return result


def _pick_embed_clips(
    segments: list[tuple[float, float]],
    n: int,
    clip_duration: float = 4.0,
) -> list[tuple[float, float]]:
    """Pick up to n non-overlapping clips from segments for embedding."""
    # Sort by start time.
    sorted_segs = sorted(segments, key=lambda s: s[0])
    clips = []
    for seg_start, seg_end in sorted_segs:
        seg_dur = seg_end - seg_start
        if seg_dur < MIN_EMBED_AUDIO_S:
            continue
        # Take a clip from the middle of the segment.
        clip_start = seg_start + (seg_dur - clip_duration) / 2
        clip_start = max(seg_start, min(clip_start, seg_end - clip_duration))
        clip_end = clip_start + clip_duration
        clips.append((clip_start, clip_end))
        if len(clips) >= n:
            break
    return clips


# ── Registry I/O ─────────────────────────────────────────────────────

def load_registry(path: Path | None = None) -> dict:
    path = path or DEFAULT_REGISTRY
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_registry(registry: dict, path: Path | None = None):
    path = path or DEFAULT_REGISTRY
    path.parent.mkdir(parents=True, exist_ok=True)
    # Convert numpy arrays to lists for JSON serialization.
    serializable = {}
    for name, info in registry.items():
        entry = dict(info)
        if "embeddings" in entry:
            entry["embeddings"] = [
                emb.tolist() if isinstance(emb, np.ndarray) else emb
                for emb in entry["embeddings"]
            ]
        serializable[name] = entry
    path.write_text(
        json.dumps(serializable, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ── Matching ─────────────────────────────────────────────────────────

def _cosine_similarity(a: "np.ndarray", b: "np.ndarray") -> float:
    import numpy as np
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def match_speaker(
    embedding: np.ndarray,
    registry: dict,
    threshold: float = MATCH_THRESHOLD,
) -> tuple[str | None, float]:
    """Match an embedding against the registry.

    Returns (name, score). name is None if no match > threshold.
    Compares against the centroid of each speaker's stored embeddings.
    """
    import numpy as np
    best_name = None
    best_score = 0.0
    for name, info in registry.items():
        if name.startswith("_unlabeled"):
            continue  # skip unlabeled pending entries
        stored = info.get("embeddings", [])
        if not stored:
            continue
        stored_arr = np.array(stored)  # shape: (n, 512)
        centroid = stored_arr.mean(axis=0)
        score = _cosine_similarity(embedding, centroid)
        if score > best_score:
            best_score = score
            best_name = name
    if best_score >= threshold:
        return best_name, best_score
    return None, best_score


def identify_speakers(
    speaker_embeddings: dict[str, list[np.ndarray]],
    registry: dict,
    threshold: float = MATCH_THRESHOLD,
) -> dict[str, str]:
    """Match each diarization speaker label to a registry name.

    Args:
        speaker_embeddings: {label: [embedding_arrays]} from diarization.
        registry: loaded speaker registry.

    Returns: {label: display_name} — mapped names or original labels.
    """
    mapping: dict[str, str] = {}
    for label, emb_list in speaker_embeddings.items():
        if not emb_list:
            mapping[label] = label
            continue
        # Average embeddings for this speaker.
        avg_emb = np.mean(emb_list, axis=0)
        matched_name, score = match_speaker(avg_emb, registry, threshold)
        if matched_name:
            mapping[label] = matched_name
        else:
            # Store as unlabeled for future enrollment.
            mapping[label] = label  # keep original label for now
    return mapping


# ── Registry update ──────────────────────────────────────────────────

def accumulate_embeddings(
    speaker_embeddings: dict[str, list[np.ndarray]],
    speaker_names: dict[str, str],
    registry: dict,
):
    """Add new embeddings to the registry.

    For known speakers, append to their embedding list (FIFO capped).
    For unknown speakers, store under "_unlabeled_<label>" key.
    """
    from datetime import datetime, timezone

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    for label, emb_list in speaker_embeddings.items():
        name = speaker_names.get(label, label)
        is_unlabeled = name == label and label.startswith("SPEAKER_")
        storage_key = f"_unlabeled_{label}" if is_unlabeled else name

        if storage_key not in registry:
            registry[storage_key] = {
                "embeddings": [],
                "sample_count": 0,
                "first_seen": today,
                "last_seen": today,
            }
            if is_unlabeled:
                registry[storage_key]["pending_label"] = True

        entry = registry[storage_key]
        for emb in emb_list:
            entry["embeddings"].append(emb)
        # FIFO cap.
        if len(entry["embeddings"]) > MAX_EMBEDDINGS_PER_SPEAKER:
            entry["embeddings"] = entry["embeddings"][-MAX_EMBEDDINGS_PER_SPEAKER:]
        entry["sample_count"] += 1
        entry["last_seen"] = today


# ── CLI ──────────────────────────────────────────────────────────────

def cmd_enroll(args):
    """Enroll a speaker from an audio clip."""
    import librosa
    import configparser

    # Load HF token (same source as rest of pipeline).
    from pathlib import Path as _P
    config_path = _P(__file__).parent / "config.yaml"
    import yaml
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    secrets_src = _P(cfg.get("secrets_source", ""))
    cp = configparser.ConfigParser()
    with open(secrets_src, "r", encoding="utf-8-sig") as f:
        cp.read_file(f)
    hf_token = cp["defaults"]["hf_token"].strip()

    wav_path = Path(args.wav)
    if not wav_path.exists():
        print(f"Error: {wav_path} not found")
        sys.exit(1)

    print(f"Loading {wav_path} ...")
    wav, sr = librosa.load(str(wav_path), sr=16000, mono=True)
    dur = len(wav) / sr
    print(f"  Duration: {dur:.1f}s")

    if dur < 3:
        print("Error: audio too short (<3s), need at least 3 seconds")
        sys.exit(1)

    model = load_embedding_model(hf_token, "cpu")
    # Extract from the first 4 seconds.
    emb = extract_embedding(wav, sr, model, start_s=1.0, duration_s=3.0)
    if emb is None:
        print("Error: failed to extract embedding")
        sys.exit(1)

    registry = load_registry()
    if args.name in registry:
        print(f"  {args.name}: adding embedding (existing, {registry[args.name]['sample_count']+1} samples)")
        registry[args.name]["embeddings"].append(emb)
        if len(registry[args.name]["embeddings"]) > MAX_EMBEDDINGS_PER_SPEAKER:
            registry[args.name]["embeddings"] = registry[args.name]["embeddings"][-MAX_EMBEDDINGS_PER_SPEAKER:]
        registry[args.name]["sample_count"] += 1
    else:
        print(f"  {args.name}: new enrollment (1 sample)")
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        registry[args.name] = {
            "embeddings": [emb],
            "sample_count": 1,
            "first_seen": today,
            "last_seen": today,
        }

    save_registry(registry)
    print(f"  Saved to {DEFAULT_REGISTRY}")


def cmd_list(args):
    """List registered speakers."""
    registry = load_registry()
    if not registry:
        print("Registry is empty.")
        return
    print(f"{'Name':<30} {'Samples':>8}  {'First':>10}  {'Last':>10}")
    print("-" * 65)
    for name, info in sorted(registry.items()):
        pending = " ⚠" if info.get("pending_label") else ""
        print(f"{name:<30} {info['sample_count']:>8}  {info.get('first_seen','?'):>10}  {info.get('last_seen','?'):>10}{pending}")


def cmd_rename(args):
    """Rename an unlabeled speaker entry."""
    registry = load_registry()
    old = args.old_name
    new = args.new_name
    if old not in registry:
        print(f"Error: '{old}' not in registry")
        sys.exit(1)
    if new in registry:
        print(f"Error: '{new}' already exists")
        sys.exit(1)
    registry[new] = registry.pop(old)
    registry[new].pop("pending_label", None)
    save_registry(registry)
    print(f"Renamed '{old}' → '{new}'")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(prog="speaker_id")
    sub = parser.add_subparsers(dest="command", required=True)

    p_enroll = sub.add_parser("enroll", help="Enroll a speaker from audio clip")
    p_enroll.add_argument("--name", required=True, help="Speaker name")
    p_enroll.add_argument("--wav", required=True, help="Path to .wav file (3s+ mono 16kHz)")
    p_enroll.set_defaults(func=cmd_enroll)

    p_list = sub.add_parser("list", help="List registered speakers")
    p_list.set_defaults(func=cmd_list)

    p_rename = sub.add_parser("rename", help="Rename an unlabeled speaker")
    p_rename.add_argument("old_name", help="Current registry key")
    p_rename.add_argument("new_name", help="New speaker name")
    p_rename.set_defaults(func=cmd_rename)

    args = parser.parse_args()
    args.func(args)
