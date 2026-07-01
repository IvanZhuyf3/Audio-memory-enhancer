"""Phase 5 — Digest pass: consolidate raw memos into themed digest notes.

Pipeline:
  1. Parse Obsmem/raw/*.md unchecked (- [ ]) bullets
  2. LLM garbage filter (device tests, noise, empty)
  3. Load existing digest files as context
  4. LLM classify: target_file + lifetime + title + content + relation + confidence
  5. Apply: high/medium conf → write digest + mark raw; low conf → leave [ ]

Three-tier confidence:
  ≥ high_threshold   → write, raw [v], silent in report
  ≥ low_threshold    → write, raw [v?], listed in report
  <  low_threshold   → leave raw [ ], listed in report for interactive

Procedural, no classes (matches sibling modules). All I/O encoding="utf-8".
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from classify import load_llm_config, _client, _parse_json_loose


# ── Raw parsing ─────────────────────────────────────────────────────

# Matches: - [STATUS] YYYY-MM-DD HH:MM — text
# Status: space=unchecked, v/V/x/X=checked, ?=low-confidence-processed
_RAW_ENTRY_RE = re.compile(
    r'^- \[([ xvX?])\]\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})\s*[—–-]\s*(.*)$'
)


def parse_raw_entries(raw_dir: Path) -> list[dict]:
    """Parse all weekly raw files. Return list of entry dicts:
    {week_file, week_stem, timestamp_str, text, raw_line, checkbox}
    Only returns unchecked ([ ]) entries.
    """
    entries = []
    for md_path in sorted(raw_dir.glob("*.md")):
        content = md_path.read_text(encoding="utf-8")
        for line in content.splitlines():
            m = _RAW_ENTRY_RE.match(line.strip())
            if not m:
                continue
            checkbox = m.group(1)
            if checkbox != " ":
                continue  # skip checked
            entries.append({
                "week_file": str(md_path),
                "week_stem": md_path.stem,  # e.g. "2026-W23"
                "timestamp_str": m.group(2),
                "text": m.group(3).strip(),
                "raw_line": line,
            })
    return entries


def count_all_entries(raw_dir: Path) -> dict:
    """Count entries by checkbox status across all raw files."""
    counts = {"unchecked": 0, "done": 0, "low_conf": 0}
    for md_path in sorted(raw_dir.glob("*.md")):
        content = md_path.read_text(encoding="utf-8")
        for line in content.splitlines():
            m = _RAW_ENTRY_RE.match(line.strip())
            if not m:
                continue
            cb = m.group(1)
            if cb == " ":
                counts["unchecked"] += 1
            elif cb == "?":
                counts["low_conf"] += 1
            else:
                counts["done"] += 1
    return counts


# ── Garbage filter ──────────────────────────────────────────────────

def filter_garbage(
    entries: list[dict],
    api_key: str,
    model: str,
    temperature: float = 0.0,
) -> tuple[list[dict], list[dict]]:
    """LLM filters device tests / noise / empty recordings.

    Returns (valid_entries, invalid_entries).
    Invalid entries get a 'skip_reason' field.
    """
    if not entries:
        return [], []

    # Build the entry list for the prompt.
    entry_lines = []
    for i, e in enumerate(entries):
        entry_lines.append(f"[{i}] {e['timestamp_str']} — {e['text'][:300]}")
    entry_block = "\n".join(entry_lines)

    system = (
        "你过滤来自可穿戴录音仪(Plaud)的语音备忘。有些录音是设备测试、噪音测试、"
        "距离测试或空白——这些不是真正的备忘，应该过滤掉。\n\n"
        "无效录音示例：\n"
        "- 测试录音仪能否从不同距离听到声音\n"
        "- 按圆珠笔测试噪音敏感度\n"
        "- '测试测试，能听到吗？'\n"
        "- 内容完全关于录音设备本身\n"
        "- 纯噪音或无意义重复\n\n"
        "有效备忘示例：\n"
        "- 想法、提醒、实验记录、会议总结\n"
        "- 个人感悟、观察、知识\n"
        "- 用户将来会想记住的任何内容\n\n"
        "对每条，输出JSON：{\"is_valid\": true/false, \"reason\": \"简短原因\"}"
    )

    user = f"以下 {len(entries)} 条录音转写文本，请判断哪些是有效备忘：\n\n{entry_block}"

    try:
        resp = _client(api_key).chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            response_format={"type": "json_object"},
            extra_body={"thinking": {"type": "disabled"}},
        )
        raw = resp.choices[0].message.content.strip()
        parsed = _parse_json_loose(raw)
        # Handle both dict {"decisions": [...]} and bare list [...]
        if isinstance(parsed, list):
            decisions = parsed
        elif isinstance(parsed, dict):
            decisions = parsed.get("decisions") or parsed.get("results") or []
            if isinstance(decisions, dict):
                decisions = [{"index": int(k), **v} for k, v in decisions.items()]
        else:
            decisions = []
        # Ensure each decision has an index (use array position if missing)
        for i_, d_ in enumerate(decisions):
            if "index" not in d_:
                d_["index"] = i_
    except Exception as e:
        print(f"  [filter_garbage] LLM failed ({e}); treating all as valid")
        return entries, []

    valid, invalid = [], []
    for i, entry in enumerate(entries):
        # Find matching decision
        dec = None
        for d in decisions:
            if d.get("index") == i:
                dec = d
                break
        if dec and not dec.get("is_valid", True):
            entry["skip_reason"] = dec.get("reason", "filtered")
            invalid.append(entry)
        else:
            valid.append(entry)
    return valid, invalid


# ── Digest context ──────────────────────────────────────────────────

def load_digest_context(digest_dir: Path) -> dict:
    """Read all existing digest files. Return {filename_stem: content}.
    Also returns available category names for the classifier.
    """
    context = {}
    for md_path in sorted(digest_dir.glob("*.md")):
        context[md_path.stem] = md_path.read_text(encoding="utf-8")
    return context


def build_categories_context(
    digest_context: dict,
    projects: list[dict],
) -> str:
    """Build a description of available target files for the classifier."""
    lines = []
    if digest_context:
        lines.append("现有 digest 文件：")
        for name in sorted(digest_context.keys()):
            lines.append(f"  - {name}.md")
    if projects:
        lines.append("\n已注册的项目（如果内容属于某个项目，用项目名做文件名）：")
        for p in projects:
            aliases = ", ".join(p.get("aliases", []) or [])
            kw = ", ".join(p.get("keywords", []) or [])
            extra = f" (别名: {aliases})" if aliases else ""
            extra += f" (关键词: {kw})" if kw else ""
            lines.append(f"  - {p['name']}{extra}")
    lines.append(
        "\n如果内容不属于以上任何文件或项目，可以建议一个新文件名"
        "（如 Experiment_notes、Journal 等）。"
    )
    return "\n".join(lines)


def build_digest_content_block(digest_context: dict) -> str:
    """Format existing digest file contents for the classifier prompt."""
    if not digest_context:
        return "(暂无 digest 内容)"
    blocks = []
    for name, content in sorted(digest_context.items()):
        blocks.append(f"### {name}.md\n```\n{content}\n```")
    return "\n\n".join(blocks)


# ── Classification ──────────────────────────────────────────────────

LIFETIME_TAGS = {
    "Recurring": "[永久·{freq}]",
    "Permanent": "[永久]",
    "Ephemeral": "[exp: {date}]",
    "One-shot": "[一次性]",
}


def classify_entries(
    entries: list[dict],
    digest_context: dict,
    projects: list[dict],
    api_key: str,
    model: str,
    temperature: float = 0.0,
) -> list[dict]:
    """LLM classifies valid entries against existing digest context.

    Returns list of decision dicts:
    {index, target_file, lifetime, lifetime_tag, title, content,
     relation, related_title, confidence, week_ref, original_entry}
    """
    if not entries:
        return []

    categories_ctx = build_categories_context(digest_context, projects)
    digest_block = build_digest_content_block(digest_context)

    # Build entry list for prompt
    entry_lines = []
    for i, e in enumerate(entries):
        entry_lines.append(
            f"[{i}] ({e['week_stem']}) {e['timestamp_str']} — {e['text'][:500]}"
        )
    entry_block = "\n".join(entry_lines)

    system = (
        "你管理一个来自语音备忘的个人知识库。对每条新备忘，判断：\n\n"
        "1. **target_file**: 写入哪个 digest 文件（不带.md后缀）。\n"
        f"{categories_ctx}\n\n"
        "2. **lifetime**: 按以下决策树判断（顺序不可乱）：\n"
        "   Q1: 这条信息未来会在同一条件下再次有效吗？（每年/每月/每周，或条件触发）\n"
        "       YES → \"Recurring\"（如纪念日、固定日程）\n"
        "       NO → 进入 Q2\n"
        "   Q2: 有明确的失效日期吗？（会议、截止、行程）\n"
        "       YES → \"Ephemeral\"\n"
        "       NO → 进入 Q3\n"
        "   Q3: 是事实/偏好，还是一次性任务？\n"
        "       事实 → \"Permanent\"\n"
        "       任务 → \"One-shot\"\n"
        "   关键：Q1 优先于 Q2。有日期不等于会过期。"
        "纪念日有日期但每年都过 → Recurring。\n\n"
        "3. **title**: 2-6个词的粗体标题\n"
        "4. **content**: 压缩精炼的内容（1-2句）。保留关键事实。不要注水。\n"
        "   如果这条和已有条目相关（补充/推翻/合并），在content里明确引用旧条目"
        "（如\"此前6月5日规划的XX方案取消\"）。\n"
        "5. **relation**: 与 target_file 中已有条目的关系：\n"
        "   - \"new\": 新增，无关联\n"
        "   - \"updates\": 补充已有条目的信息\n"
        "   - \"supersedes\": 推翻/否定已有条目\n"
        "   - \"merges_with\": 应与已有条目合并\n"
        "6. **related_title**: 如果 relation 不是 new，引用的已有条目标题（否则 null）\n"
        "7. **confidence**: 0.0-1.0（你对这个分类的把握）\n\n"
        "**拆分规则**：一条录音可能包含多个不相关的意思（比如先说实验，中间转折聊到"
        "另一个话题）。如果前后内容在主题、用途或生命周期上明显不同，拆成多条 "
        "decision（相同的 index）。"
        "但只有前后内容**真正独立**时才拆——对同一件事的补充说明、因果延伸、"
        "细节展开都应合并为一条，不要过度拆分。\n\n"
        "现有 digest 内容（用于判断 relation）：\n"
        f"{digest_block}\n\n"
        "输出 STRICT JSON：\n"
        '{"decisions": [{"index": 0, "target_file": "...", "lifetime": "...", '
        '"title": "...", "content": "...", "relation": "...", '
        '"related_title": null, "confidence": 0.9}]}\n'
        "注意：同一个 index 可以出现多次（拆分），也可以不出现（被过滤）。"
    )

    user = f"以下 {len(entries)} 条有效备忘，请分类：\n\n{entry_block}"

    try:
        resp = _client(api_key).chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            response_format={"type": "json_object"},
            extra_body={"thinking": {"type": "disabled"}},
        )
        raw = resp.choices[0].message.content.strip()
        parsed = _parse_json_loose(raw)
        if isinstance(parsed, list):
            decisions_raw = parsed
        elif isinstance(parsed, dict):
            decisions_raw = parsed.get("decisions") or []
        else:
            decisions_raw = []
        for i_, d_ in enumerate(decisions_raw):
            if "index" not in d_:
                d_["index"] = i_
    except Exception as e:
        print(f"  [classify_entries] LLM failed ({e}); all low confidence")
        return [
            {
                "index": i,
                "target_file": None,
                "lifetime": None,
                "title": "",
                "content": "",
                "relation": "new",
                "related_title": None,
                "confidence": 0.0,
                "week_ref": e_["week_stem"],
                "original_entry": e_,
                "error": str(e),
            }
            for i, e_ in enumerate(entries)
        ]

    # Merge decisions with original entries
    decisions = []
    for i, entry in enumerate(entries):
        dec = None
        for d in decisions_raw:
            if d.get("index") == i:
                dec = d
                break
        if not dec:
            dec = {"index": i, "confidence": 0.0}

        # Build lifetime tag
        lifetime = dec.get("lifetime", "Permanent")
        if lifetime not in LIFETIME_TAGS:
            lifetime = "Permanent"
        tag_template = LIFETIME_TAGS[lifetime]
        if lifetime == "Recurring":
            tag = tag_template.format(freq=dec.get("freq", "条件"))
        elif lifetime == "Ephemeral":
            tag = tag_template.format(date=dec.get("expiry_date", ""))
        else:
            tag = tag_template

        decisions.append({
            "index": i,
            "target_file": dec.get("target_file"),
            "lifetime": lifetime,
            "lifetime_tag": tag,
            "title": dec.get("title", "").strip(),
            "content": dec.get("content", "").strip(),
            "relation": dec.get("relation", "new"),
            "related_title": dec.get("related_title"),
            "confidence": float(dec.get("confidence", 0.5)),
            "week_ref": entry["week_stem"],
            "original_entry": entry,
        })
    return decisions


# ── Apply decisions ─────────────────────────────────────────────────

# Section order in digest files (top to bottom)
SECTION_ORDER = ["固定事件", "长期事实"]


def _format_entry(decision: dict) -> str:
    """Format a decision as a markdown bullet line."""
    title = decision["title"]
    content = decision["content"]
    tag = decision["lifetime_tag"]
    week = decision["week_ref"]
    return f"- **{title}**：{content} {tag} [[{week}]]"


def _resolve_target_section(lifetime: str, timestamp_str: str) -> str:
    """Determine which section header a lifetime maps to."""
    if lifetime == "Recurring":
        return "固定事件"
    elif lifetime == "Permanent":
        return "长期事实"
    else:
        # Ephemeral + One-shot → month section (YYYY-MM)
        # Extract from timestamp_str "YYYY-MM-DD HH:MM"
        month = timestamp_str[:7]  # "2026-06"
        return month


def _insert_into_digest(
    content: str,
    section: str,
    entry_line: str,
    file_title: str,
) -> str:
    """Insert an entry line into the correct section of a digest file.

    Creates sections / file header as needed. Maintains section order.
    Returns the new file content.
    """
    lines = content.splitlines()

    # Ensure file header exists
    if not lines or not lines[0].startswith("# "):
        header = f"# {file_title}\n"
        content = header + "\n" + content
        lines = content.splitlines()

    # Find or create the target section
    section_header = f"## {section}"
    section_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Match exact or prefix (e.g. "## 固定事件" matches "## 固定事件（每年，不归档）")
        if stripped == section_header or stripped.startswith(section_header):
            section_idx = i
            break

    if section_idx is None:
        # Need to insert the section in the right position.
        # Find all existing ## sections.
        existing_sections = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("## "):
                sname = stripped[3:]
                # Normalize: strip parenthetical suffixes for ordering
                sname_base = sname.split("（")[0].split("(")[0].strip()
                existing_sections.append((i, sname, sname_base))

        # Determine insertion point based on section order.
        # Fixed sections (固定事件, 长期事实) go first in their order.
        # Date sections (2026-06) go after, chronologically.
        insert_idx = len(lines)
        section_base = section.split("（")[0].split("(")[0].strip()
        if section_base in SECTION_ORDER:
            my_order = SECTION_ORDER.index(section_base)
            for idx, sname, sname_base in existing_sections:
                if sname_base in SECTION_ORDER:
                    their_order = SECTION_ORDER.index(sname_base)
                    if their_order > my_order:
                        insert_idx = idx
                        break
                else:
                    # Hit a date section — insert before it
                    insert_idx = idx
                    break
        else:
            # Date section — insert in chronological order (newer dates go below
            # older ones, matching existing convention).
            for idx, sname, sname_base in existing_sections:
                if sname_base not in SECTION_ORDER and sname_base < section:
                    continue
                if sname_base not in SECTION_ORDER and sname_base > section:
                    insert_idx = idx
                    break

        # Insert: section header + blank + entry + blank
        new_lines = ["", section_header, "", entry_line, ""]
        lines = lines[:insert_idx] + new_lines + lines[insert_idx:]
    else:
        # Section exists — append entry at the end of this section.
        # Find the next section or end of file.
        insert_idx = len(lines)
        for i in range(section_idx + 1, len(lines)):
            if lines[i].strip().startswith("## "):
                insert_idx = i
                break
        # Insert before the next section, with a blank line if needed.
        # Strip trailing blanks, add entry, add blank.
        while insert_idx > 0 and lines[insert_idx - 1].strip() == "":
            insert_idx -= 1
        lines.insert(insert_idx, entry_line)
        lines.insert(insert_idx + 1, "")  # blank after

    return "\n".join(lines)


def _mark_raw_done(
    raw_content: str,
    timestamp_str: str,
    status: str = "v",
) -> str:
    """Replace '- [ ] TIMESTAMP' with '- [STATUS] TIMESTAMP' in raw content."""
    old = f"- [ ] {timestamp_str}"
    if status == "v":
        new = f"- [v] {timestamp_str}"
    elif status == "?":
        new = f"- [?] {timestamp_str}"
    else:
        new = f"- [{status}] {timestamp_str}"
    return raw_content.replace(old, new, 1)


def _atomic_write(path: Path, content: str) -> None:
    import time
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    import os
    # Retry on Windows/OneDrive file locks (WinError 5)
    for attempt in range(5):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt < 4:
                time.sleep(1)
            else:
                raise


def apply_decisions(
    decisions: list[dict],
    raw_dir: Path,
    digest_dir: Path,
    high_threshold: float = 0.7,
    low_threshold: float = 0.4,
    dry_run: bool = False,
) -> dict:
    """Apply classification decisions to digest + raw files.

    Supports 1:N split: multiple decisions with the same raw entry
    (same (week_file, timestamp)) produce multiple digest entries but
    only one raw checkbox mark. The mark uses the MIN confidence across
    all splits of the same raw entry, so a low-confidence split prevents
    the entry from being marked completely resolved.

    Returns report dict: {done, low_conf, skipped, details[]}
    """
    report = {"done": 0, "low_conf": 0, "skipped": 0, "details": []}

    # Track raw updates keyed by (week_path, timestamp) -> {status, confidence}
    # so splits of the same entry use the worst (lowest) status.
    raw_update_map: dict[tuple[str, str], dict] = {}

    for dec in decisions:
        entry = dec["original_entry"]
        ts = entry["timestamp_str"]
        week_path = Path(entry["week_file"])
        conf = dec["confidence"]
        raw_key = (str(week_path), ts)

        if conf < low_threshold:
            # Low confidence — leave [ ], add to report
            report["low_conf"] += 1
            report["details"].append({
                "status": "low_conf",
                "timestamp": ts,
                "text_snippet": entry["text"][:80],
                "suggested_file": dec.get("target_file"),
                "suggested_title": dec.get("title"),
                "confidence": conf,
            })
            # Track as lowest confidence for this raw entry
            if raw_key not in raw_update_map or conf < raw_update_map[raw_key]["confidence"]:
                raw_update_map[raw_key] = {"status": " ", "confidence": conf}
            continue

        # High or medium confidence — write to digest
        status = "v" if conf >= high_threshold else "?"

        # Determine target file
        target_name = dec.get("target_file") or "Miscellaneous"
        target_name = re.sub(r'[\\/*?:"<>|]', "", target_name).strip() or "Miscellaneous"
        target_path = digest_dir / f"{target_name}.md"

        # Read existing content
        existing = ""
        if target_path.exists():
            existing = target_path.read_text(encoding="utf-8")

        # Determine section
        section = _resolve_target_section(dec["lifetime"], ts)

        # Format entry
        entry_line = _format_entry(dec)

        # Insert into digest
        new_content = _insert_into_digest(existing, section, entry_line, target_name)

        if not dry_run:
            _atomic_write(target_path, new_content)

        # Track status for this raw entry — use WORST (lowest) confidence
        if raw_key not in raw_update_map or conf < raw_update_map[raw_key]["confidence"]:
            # Update: still track confidence but derive status from it
            raw_update_map[raw_key] = {"status": status, "confidence": conf}

        report["done"] += 1
        report["details"].append({
            "status": "done" if status == "v" else "low_conf_done",
            "timestamp": ts,
            "target_file": target_name,
            "title": dec["title"],
            "lifetime": dec["lifetime"],
            "relation": dec.get("relation", "new"),
            "confidence": conf,
        })

    # Apply raw updates (batch per file, deduplicated by raw_key)
    if not dry_run:
        # Group by week_path
        by_file: dict[Path, list[tuple[str, str]]] = {}
        for (wp, ts), info in raw_update_map.items():
            p = Path(wp)
            if p not in by_file:
                by_file[p] = []
            by_file[p].append((ts, info["status"]))
        for week_path, updates in by_file.items():
            content = week_path.read_text(encoding="utf-8")
            for ts, status in updates:
                content = _mark_raw_done(content, ts, status)
            _atomic_write(week_path, content)

    return report


def apply_skips(
    invalid_entries: list[dict],
    dry_run: bool = False,
) -> int:
    """Mark garbage-filtered entries as [v] in raw files. Returns count."""
    if not invalid_entries:
        return 0

    # Group by week file
    by_file: dict[Path, list[str]] = {}
    for entry in invalid_entries:
        week_path = Path(entry["week_file"])
        if week_path not in by_file:
            by_file[week_path] = []
        by_file[week_path].append(entry["timestamp_str"])

    if dry_run:
        return len(invalid_entries)

    for week_path, timestamps in by_file.items():
        content = week_path.read_text(encoding="utf-8")
        for ts in timestamps:
            content = _mark_raw_done(content, ts, "v")
        _atomic_write(week_path, content)

    return len(invalid_entries)


# ── Report formatting ───────────────────────────────────────────────

def format_report(
    report: dict,
    skip_count: int,
    total_unchecked: int,
    dry_run: bool,
) -> str:
    """Format the digest report for stdout (cron delivery)."""
    lines = []
    prefix = "[DRY-RUN] " if dry_run else ""

    done = report["done"]
    low = report["low_conf"]
    skip_details = report["details"]

    if done == 0 and low == 0 and skip_count == 0:
        return f"{prefix}[digest] 没有待处理的录音备忘。"

    lines.append(f"{prefix}[digest] 处理了 {total_unchecked} 条录音备忘：")
    if skip_count:
        lines.append(f"  🗑 过滤 {skip_count} 条（设备测试/噪音）")
    if done:
        high = sum(1 for d in skip_details if d["status"] == "done")
        med = sum(1 for d in skip_details if d["status"] == "low_conf_done")
        lines.append(f"  ✅ {done} 条已归档（{high} 高置信, {med} 中置信标 [v?]）")
    if low:
        lines.append(f"  ⚠️ {low} 条低置信度，待你确认：")
        for d in skip_details:
            if d["status"] == "low_conf":
                snippet = d["text_snippet"]
                suggested = d.get("suggested_file") or "?"
                title = d.get("suggested_title") or "?"
                lines.append(
                    f"    • {d['timestamp']} \"{snippet}...\" → {suggested}/{title} "
                    f"(conf {d['confidence']:.2f})"
                )

    # Also list medium-confidence ones briefly
    med_details = [d for d in skip_details if d["status"] == "low_conf_done"]
    if med_details:
        lines.append(f"  📋 中置信度（已归档，扫一眼）：")
        for d in med_details:
            lines.append(
                f"    • {d['timestamp']} → {d['target_file']}/{d['title']} "
                f"(conf {d['confidence']:.2f})"
            )

    return "\n".join(lines)


# ── Orchestrator ────────────────────────────────────────────────────

def run_digest(
    cfg: dict,
    dry_run: bool = False,
    high_threshold: float = 0.7,
    low_threshold: float = 0.4,
) -> str:
    """Main entry point. Returns formatted report string for cron output."""
    vault_root = Path(cfg["vault"]["root"])
    raw_dir = vault_root / cfg["vault"]["memo_raw_folder"]
    digest_dir = vault_root / cfg["vault"].get("memo_digest_folder", "Obsmem/digest")

    if not raw_dir.exists():
        return f"[digest] raw 目录不存在: {raw_dir}"

    # 1. Parse raw entries
    entries = parse_raw_entries(raw_dir)
    total = len(entries)
    if total == 0:
        counts = count_all_entries(raw_dir)
        return (
            f"[digest] 没有待处理的录音备忘。"
            f"(已归档 {counts['done']}, 低置信 {counts['low_conf']})"
        )

    print(f"[digest] 发现 {total} 条未处理备忘")

    # 2. Load secrets
    from pipeline import load_secrets
    secrets = load_secrets(cfg)
    api_key = secrets["deepseek_api_key"]
    model = secrets["deepseek_model"]
    if not api_key:
        return "[digest] DeepSeek API key 未配置，跳过。"

    # 3. Garbage filter
    valid, invalid = filter_garbage(entries, api_key, model)
    if invalid:
        print(f"[digest] 过滤 {len(invalid)} 条无效录音")
    skip_count = apply_skips(invalid, dry_run=dry_run)

    if not valid:
        return format_report(
            {"done": 0, "low_conf": 0, "details": []},
            skip_count, total, dry_run,
        )

    # 4. Load digest context + projects
    digest_context = load_digest_context(digest_dir)
    projects = []
    projects_path = cfg.get("projects_registry")
    if projects_path and Path(projects_path).exists():
        import yaml
        data = yaml.safe_load(Path(projects_path).read_text(encoding="utf-8"))
        projects = data.get("projects", []) if data else []

    # 5. Classify
    decisions = classify_entries(valid, digest_context, projects, api_key, model)

    # 6. Apply
    report = apply_decisions(
        decisions, raw_dir, digest_dir,
        high_threshold=high_threshold,
        low_threshold=low_threshold,
        dry_run=dry_run,
    )

    # 7. Format report
    return format_report(report, skip_count, total, dry_run)
