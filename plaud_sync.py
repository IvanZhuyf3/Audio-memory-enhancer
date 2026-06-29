"""Direct Plaud cloud API client.

Reads credentials + token from ~/.plaud/config.json (written by the upstream
plaud-toolkit's `plaud login` command). Calls the Plaud REST API directly — no
subprocess, no text-table parsing. Auto-refreshes the token using stored
credentials when within 30 days of expiry, mirroring the upstream TS library's
behaviour.

Procedural API:
    list_recordings()          -> [{id, filename, duration, start_time, ...}]
    get_recording(rec_id)      -> {... + transcript}
    get_mp3_url(rec_id)        -> pre-signed URL string | None
    download_audio(rec_id, p)  -> Path  (saves mp3/opus to disk)
    get_user_info()            -> {id, nickname, email, membership_type, ...}

All file I/O uses encoding="utf-8" (Windows-safe).
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# Plaud's API rejects the default urllib User-Agent with 403; send a browser UA.
# Mirrors packages/core/src/types.ts USER_AGENT in the upstream toolkit.
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
BASE_URLS = {"us": "https://api.plaud.ai", "eu": "https://api-euc1.plaud.ai"}
TOKEN_REFRESH_BUFFER_S = 30 * 24 * 3600  # refresh if <30 days left
HTTP_TIMEOUT_S = 60

# In-memory cache of the config file; reset by `reload_config()`.
_config_cache: dict | None = None
_config_path: Path | None = None


# ── Config loading ──────────────────────────────────────────────────

def default_config_path() -> Path:
    return Path.home() / ".plaud" / "config.json"


def load_config(explicit_path: str | Path | None = None) -> dict:
    """Load ~/.plaud/config.json (or an explicit path). Caches in memory."""
    global _config_cache, _config_path
    path = Path(explicit_path) if explicit_path else default_config_path()
    if _config_cache is None or _config_path != path:
        if not path.exists():
            raise RuntimeError(
                f"Plaud config not found at {path}.\n"
                f"Run `plaud login` once in the upstream toolkit:\n"
                f"  cd <plaud-toolkit> && npx tsx packages/cli/bin/plaud.ts login"
            )
        _config_cache = json.loads(path.read_text(encoding="utf-8"))
        _config_path = path
    return _config_cache


def reload_config() -> dict:
    """Force re-read from disk (use after login/refresh)."""
    global _config_cache
    _config_cache = None
    return load_config(_config_path)


# ── JWT + token management ───────────────────────────────────────────

def _decode_jwt_expiry(jwt_str: str) -> tuple[int, int]:
    """Return (iat, exp) epoch seconds from a JWT. (0, 0) if unparseable."""
    parts = jwt_str.split(".")
    if len(parts) != 3:
        return 0, 0
    payload_b64 = parts[1] + "=" * (-len(parts[1]) % 4)  # pad to multiple of 4
    try:
        payload = json.loads(base64.urlsafe_b64decode(payload_b64).decode("utf-8"))
        return int(payload.get("iat", 0)), int(payload.get("exp", 0))
    except (ValueError, json.JSONDecodeError):
        return 0, 0


def ensure_token() -> str:
    """Return a valid access token, refreshing (re-login) if expiring soon."""
    cfg = load_config()
    token = cfg.get("token") or {}
    access = token.get("accessToken")
    if access:
        # expiresAt is epoch-ms (TS format); fall back to decoding the JWT.
        exp_ms = token.get("expiresAt") or 0
        exp_s = exp_ms / 1000 if exp_ms else _decode_jwt_expiry(access)[1]
        now_s = datetime.now(timezone.utc).timestamp()
        if exp_s - now_s > TOKEN_REFRESH_BUFFER_S:
            return access
    return _login_and_cache()


def _login_and_cache() -> str:
    """POST /auth/access-token with stored email/password, cache the new token."""
    cfg = load_config()
    creds = cfg.get("credentials")
    if not creds:
        raise RuntimeError("No Plaud credentials stored. Run `plaud login` first.")
    region = creds.get("region", "us")
    base = BASE_URLS.get(region, BASE_URLS["us"])
    form = urllib.parse.urlencode(
        {"username": creds["email"], "password": creds["password"]}
    )
    req = urllib.request.Request(
        base + "/auth/access-token",
        data=form.encode("utf-8"),
        method="POST",
        headers={
            "User-Agent": USER_AGENT,
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if data.get("status") != 0 or not data.get("access_token"):
        raise RuntimeError(
            f"Plaud login failed: status={data.get('status')} msg={data.get('msg')}"
        )
    jwt = data["access_token"]
    iat, exp = _decode_jwt_expiry(jwt)
    # Update the on-disk config.json so the TS CLI and this module stay in sync.
    path = _config_path or default_config_path()
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    existing["token"] = {
        "accessToken": jwt,
        "tokenType": data.get("token_type", "Bearer"),
        "issuedAt": iat * 1000,
        "expiresAt": exp * 1000,
    }
    path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    global _config_cache
    _config_cache = existing
    return jwt


# ── HTTP helpers ────────────────────────────────────────────────────

def _api_request(
    path: str,
    method: str = "GET",
    body: str | None = None,
    content_type: str = "application/json",
    region: str | None = None,
) -> dict:
    """Call a Plaud API path (starting with /) and return parsed JSON."""
    cfg = load_config()
    r = region or (cfg.get("credentials") or {}).get("region", "us")
    base = BASE_URLS.get(r, BASE_URLS["us"])
    url = base + path
    headers = {
        "User-Agent": USER_AGENT,
        "Authorization": f"Bearer {ensure_token()}",
    }
    if content_type:
        headers["Content-Type"] = content_type
    data = body.encode("utf-8") if isinstance(body, str) else body
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        snippet = e.read().decode("utf-8", "replace")[:500]
        raise RuntimeError(f"Plaud API HTTP {e.code} for {path}: {snippet}") from None


def _http_get_bytes(url: str) -> bytes:
    """Plain GET for a (possibly pre-signed) URL. Returns raw bytes."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S * 2) as resp:
        return resp.read()


def _resolve_region_mismatch(data: dict) -> str | None:
    """Plaud returns {status:-302, data:{domains:{api:"..."}}} on wrong-region calls."""
    if data.get("status") == -302:
        domain = (data.get("data") or {}).get("domains", {}).get("api", "")
        if domain:
            return "eu" if "euc1" in domain else "us"
    return None


# ── Public API ──────────────────────────────────────────────────────

def list_recordings(region: str | None = None) -> list[dict]:
    """List all non-trashed recordings. Each item has fields from PlaudRecording:
    id, filename, fullname, filesize, duration (ms), start_time (epoch ms),
    end_time, is_trash, is_trans, is_summary, keywords[], serial_number.
    """
    data = _api_request("/file/simple/web", region=region)
    new_region = _resolve_region_mismatch(data)
    if new_region and new_region != region:
        return list_recordings(region=new_region)
    lst = data.get("data_file_list") or data.get("data") or []
    return [r for r in lst if not r.get("is_trash")]


def get_recording(rec_id: str) -> dict:
    """Full recording detail with transcript + metadata extracted.

    Returns the raw detail dict plus:
        id, filename, transcript (raw data_content — use parse_transcript()
        to clean it), content_list (S3 data-links for raw/polish/outline/summary),
        tran_config (language/diarization/llm from extra_data).
    """
    data = _api_request(f"/file/detail/{rec_id}")
    raw = data.get("data") or data

    # Inline content: for short recordings Plaud inlines the raw transcript
    # inside an auto_sum preamble. For longer recordings this is the AI summary.
    # Use the longest data_content as the inline transcript (matches upstream).
    transcript = ""
    for item in raw.get("pre_download_content_list") or []:
        content = item.get("data_content") or ""
        if len(content) > len(transcript):
            transcript = content

    raw["id"] = raw.get("file_id") or rec_id
    raw["filename"] = raw.get("file_name") or raw.get("filename") or rec_id
    raw["transcript"] = transcript
    raw["content_list"] = raw.get("content_list") or []
    raw["tran_config"] = (raw.get("extra_data") or {}).get("tranConfig") or {}
    return raw


# ── Transcript parsing ──────────────────────────────────────────────

import re as _re

# Matches "[Speaker 1]", "[Speaker 2]", "> [Speaker 1]", etc.
_SPEAKER_LINE_RE = _re.compile(r"^(?:>\s*)?\[Speaker\s+(\d+)\]")


def parse_transcript(content: str) -> dict:
    """Clean a Plaud inline transcript.

    Short-recording summaries start with a preamble like
    "转写内容较短，无需生成总结。音频转写原文如下：" followed by
    "[Speaker N] ..." lines. This extracts just the speaker-tagged portion.

    Returns {text, speakers, had_preamble}:
        text       — cleaned transcript (speaker-tagged lines, preamble stripped)
        speakers   — count of distinct speakers (0 if none tagged)
        had_preamble — True if a preamble was detected and stripped
    """
    if not content or not content.strip():
        return {"text": "", "speakers": 0, "had_preamble": False}

    lines = content.split("\n")
    # Find the first line that starts with a [Speaker N] marker.
    first_speaker_idx = None
    for i, line in enumerate(lines):
        if _SPEAKER_LINE_RE.match(line.strip()):
            first_speaker_idx = i
            break

    if first_speaker_idx is None:
        # No speaker markers — return the content as-is (likely a pure summary).
        return {"text": content.strip(), "speakers": 0, "had_preamble": False}

    had_preamble = first_speaker_idx > 0
    body_lines = lines[first_speaker_idx:]
    # Strip leading "> " blockquote markers (Plaud wraps transcript lines).
    cleaned = []
    for line in body_lines:
        s = line.strip()
        m = _SPEAKER_LINE_RE.match(s)
        if m:
            # Normalize: remove "> " prefix, keep "[Speaker N]" + text.
            s = _re.sub(r"^(?:>\s*)", "", line).rstrip()
        cleaned.append(s)

    text = "\n".join(cleaned).strip()
    # Count distinct speaker numbers.
    speaker_nums = set()
    for line in body_lines:
        m = _SPEAKER_LINE_RE.match(line.strip())
        if m:
            speaker_nums.add(m.group(1))
    return {"text": text, "speakers": len(speaker_nums), "had_preamble": had_preamble}


def get_mp3_url(rec_id: str) -> str | None:
    """Pre-signed MP3 URL (short-lived). None if unavailable."""
    try:
        data = _api_request(f"/file/temp-url/{rec_id}?is_opus=false")
    except RuntimeError:
        return None
    return (
        data.get("url")
        or (data.get("data") or {}).get("url")
        or data.get("data")
        or data.get("temp_url")
    )


def download_audio(rec_id: str, dest_path: str | Path) -> Path:
    """Download audio for a recording. Prefers the temp MP3 URL; falls back to
    /file/download/<id> (opus). Returns the destination Path.
    """
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    mp3_url = get_mp3_url(rec_id)
    if mp3_url:
        buf = _http_get_bytes(mp3_url)
        if buf:
            out = dest.with_suffix(".mp3") if dest.suffix == "" else dest
            out.write_bytes(buf)
            return out

    # Fallback: direct download endpoint (returns opus bytes).
    cfg = load_config()
    region = (cfg.get("credentials") or {}).get("region", "us")
    base = BASE_URLS.get(region, BASE_URLS["us"])
    req = urllib.request.Request(
        f"{base}/file/download/{rec_id}",
        headers={"User-Agent": USER_AGENT, "Authorization": f"Bearer {ensure_token()}"},
    )
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S * 2) as resp:
        buf = resp.read()
    out = dest.with_suffix(".opus") if dest.suffix == "" else dest
    out.write_bytes(buf)
    return out


def get_user_info() -> dict:
    """Account info: id, nickname, email, country, membership_type."""
    data = _api_request("/user/me")
    user = data.get("data_user") or data.get("data") or data
    return {
        "id": user.get("id"),
        "nickname": user.get("nickname"),
        "email": user.get("email"),
        "country": user.get("country"),
        "membership_type": (data.get("data_state") or {}).get("membership_type", "unknown"),
    }


# ── CLI smoke-test entry: python -m plaud_sync <list|info> ───────────

if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    if cmd == "list":
        recs = list_recordings()
        print(f"{len(recs)} recording(s):")
        for r in recs:
            dur_min = (r.get("duration") or 0) / 60000
            date = datetime.fromtimestamp(
                (r.get("start_time") or 0) / 1000, tz=timezone.utc
            ).strftime("%Y-%m-%d %H:%M")
            flags = ("T" if r.get("is_trans") else "-") + ("S" if r.get("is_summary") else "-")
            print(f"  {r.get('id')}  {date}  {dur_min:6.1f}m  {flags}  {r.get('filename')}")
    elif cmd == "info":
        info = get_user_info()
        print(json.dumps(info, indent=2, ensure_ascii=False))
    elif cmd == "detail" and len(sys.argv) > 2:
        d = get_recording(sys.argv[2])
        t = d.get("transcript") or ""
        print(f"id={d.get('id')}  filename={d.get('filename')}")
        print(f"transcript length: {len(t)} chars")
        print(f"first 300 chars: {t[:300]!r}")
    else:
        print("Usage: python plaud_sync.py [list|info|detail <id>]")
