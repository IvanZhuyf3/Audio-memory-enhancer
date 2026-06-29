r"""Debug: dump the raw /file/detail/<id> or /file/simple/web JSON.

Usage:
    python scripts\debug_detail.py detail <recording-id> [--out raw.json]
    python scripts\debug_detail.py list [--out raw.json]

Used to discover the actual field layout of Plaud's endpoints, since the
upstream TS client's assumption (transcript in pre_download_content_list[].data_content,
and list items having an `id` field) did not match real responses.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from anywhere — add project root to path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import plaud_sync  # noqa: E402


def _walk(obj, prefix="", out=None):
    """Walk a nested dict/list and record (path, type, preview) for each leaf."""
    if out is None:
        out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, (dict, list)):
                _walk(v, p, out)
            else:
                preview = repr(v)[:120]
                out.append((p, type(v).__name__, preview))
    elif isinstance(obj, list):
        out.append((prefix, f"list[{len(obj)}]", ""))
        for i, v in enumerate(obj[:3]):  # only first 3 items
            _walk(v, f"{prefix}[{i}]", out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["detail", "list"], help="endpoint to dump")
    ap.add_argument("recording_id", nargs="?", help="recording id (detail mode only)")
    ap.add_argument("--out", default=None, help="write full JSON to this file")
    args = ap.parse_args()

    if args.mode == "detail":
        if not args.recording_id:
            ap.error("detail mode requires a recording_id")
        raw = plaud_sync._api_request(f"/file/detail/{args.recording_id}")
    else:
        raw = plaud_sync._api_request("/file/simple/web")

    # Save full JSON.
    if args.out:
        Path(args.out).write_text(
            json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"full JSON → {args.out}")

    # Print a flat field map so we can spot where the transcript / IDs live.
    print("\n--- field map ---")
    for path, typ, preview in _walk(raw):
        print(f"{path:<55} {typ:<6} {preview}")

    # For list mode: also print every top-level key on each item so we can spot
    # the real ID field (the upstream client assumes 'id', which may be wrong).
    if args.mode == "list":
        items = raw.get("data_file_list") or raw.get("data") or []
        if items:
            print(f"\n--- list item keys ({len(items)} items) ---")
            print(" ", sorted(items[0].keys()))
            print("\n--- first item id-like fields ---")
            for k in sorted(items[0].keys()):
                if "id" in k.lower() or "_id" in k.lower() or "file" in k.lower():
                    print(f"  {k} = {items[0][k]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
