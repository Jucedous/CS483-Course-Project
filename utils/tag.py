'''
好的，我们把方案改为“开放标签集”：LLM 可以从已有标签集合中选，也可以自由创造新标签；脚本会把所有出现过的标签汇总到一个全局标签集合文件，后续你可以再做标签归并。

改动要点
- 标签集为开放集合：在运行期间维护一个全局 tag set（持久化为一个 JSON 文件，如 tags_set.json）。LLM 可复用其中标签，也可新增任意新标签。
- 并行分析与进度条：同前，async 并发 + tqdm 进度条。
- 跳过空内容：content 为 null/空白的邮件跳过不送 LLM。
- JSON 异常文件移动：解析失败或结构不符的 JSON 移到 bad_dir。
- LLM 输出：严格 JSON，包括 messages 的 index、tags（可多选，可新建）、notes（打标签理由），以及事件级概述 event_summary。
- 标签写回：每个 message 写 tags 和 tag_notes；事件级写 event_summary。事件额外记录 new_tags_in_event（在本次分析前不在全局集合中的新标签）。
- 全局标签集合：脚本从 tag_set_path 读取已有集合，作为“候选建议”提供给 LLM；每处理完一个事件，合并新标签进集合；运行结束写回集合文件。为并发安全使用锁。
- 可选轻度规范化：提供 --normalize_tags 开关（默认 false）。开启后将标签做低强度规范化（lower、去首尾空白、合并连续空白为单空格；保持自由度，避免过度变形）

依赖与运行
- 安装：pip install openai tqdm
- 设置环境变量：export OPENAI_API_KEY="你的key"
- 示例运行：
  python tag_lkml_events_open_set.py --input_dir ./events --output_dir ./tagged --bad_dir ./bad --model gpt-4o-mini --concurrency 4 --tag_set_path ./tagged/tags_set.json

脚本：tag_lkml_events_open_set.py

使用建议
- 后续合并标签：处理完成后，可遍历 output_dir 中的事件文件，统计所有 tags 字段，结合 tags_set.json 做归并映射（同义词、大小写、分词差异等）。
- 提示优化：若你已有一批常见标签，可先写入 tags_set.json 作为初始集合，帮助模型更快收敛，同时保留自由新增能力。
- 速率与成本：并发值建议 2–8；如事件含很长补丁，可调低 MAX_MSG_CHARS 以控成本。
- 容错：LLM 返回非 JSON 会自动重试；仍失败则把文件移到 bad_dir，便于二次处理。

如果你希望改为“按事件级别输出主题/争议点/阻塞项”等更高层次标签，也可以在同一 JSON 顶层增加 event_topics、blocking_points 等字段，我可以再帮你扩展提示词与解析逻辑。
'''

import os
import json
import asyncio
import shutil
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Set

from tqdm import tqdm
from openai import AsyncOpenAI, OpenAIError

# ----------------------------
# Tunables
# ----------------------------

MAX_MSG_CHARS = 4000
TEMPERATURE = 0.0

# ----------------------------
# Helpers
# ----------------------------

def is_message_empty(msg: Dict[str, Any]) -> bool:
    content = msg.get("content")
    if content is None:
        return True
    if isinstance(content, str) and content.strip() == "":
        return True
    return False

def truncate(s: Optional[str], max_len: int) -> str:
    if not s:
        return ""
    if len(s) <= max_len:
        return s
    return s[:max_len] + "\n...[truncated]..."

def validate_event_json(event: Dict[str, Any]) -> bool:
    if not isinstance(event, dict):
        return False
    if "event_id" not in event or "messages" not in event:
        return False
    if not isinstance(event["messages"], list):
        return False
    for m in event["messages"]:
        if not isinstance(m, dict):
            return False
        if "content" not in m:
            return False
    return True

def normalize_tag_soft(tag: str) -> str:
    # 轻度规范化：去首尾空白 + 小写 + 合并连续空白为单空格
    # 保留自由命名，不做下划线/去标点等强侵入性操作
    t = (tag or "").strip().lower()
    # 合并多空格
    t = " ".join(t.split())
    return t

def build_prompt(event_id: str,
                 root_url: Optional[str],
                 messages: List[Dict[str, Any]],
                 existing_tags: List[str]) -> str:
    """
    Build prompt that suggests existing tags but allows new tags freely.
    """
    lines = []
    lines.append("任务：为下列邮件逐条打能反映“发送原因/动机”的多标签（可多选）。")
    lines.append("你可以从当前标签集合中选择，也可以自由创建新标签。")
    lines.append("请保持标签简洁（建议4-6个词），标签仅为英文，可使用下划线或短语，避免过长句子。")
    lines.append("如可复用已有标签，请优先复用；确有必要时再新增。")
    lines.append("")
    lines.append(f"事件ID: {event_id}")
    if root_url:
        lines.append(f"事件根URL: {root_url}")
    lines.append("")
    if existing_tags:
        # 给模型看当前已有标签集合（仅建议，不限制）
        sample = existing_tags[:500]  # 防止太长
        lines.append("当前已存在的标签集合（建议复用）：")
        lines.append(", ".join(sample))
        if len(existing_tags) > len(sample):
            lines.append(f"...（共 {len(existing_tags)} 个，已截断展示）")
        lines.append("")
    lines.append("注意：只对提供的非空邮件打标签。以下索引 index 从 0 开始，对应这里的列表顺序。")
    lines.append("邮件列表：")
    for i, m in enumerate(messages):
        subj = m.get("subject") or ""
        url = m.get("url") or ""
        content = truncate(m.get("content") or "", MAX_MSG_CHARS)
        lines.append(f"\n[#{i}] subject: {subj}\nurl: {url}\ncontent:\n{content}\n")
    lines.append(
        "请输出严格 JSON，且仅输出 JSON："
        "{"
        "\"event_id\": str, "
        "\"messages\": [ "
        "{ \"index\": int, \"tags\": [str,...], \"notes\": str } ... "
        "], "
        "\"event_summary\": str "
        "}"
    )
    lines.append("notes 用一句话解释标签选择理由。不要输出除 JSON 之外的任何文本。")
    return "\n".join(lines)

def merge_tags_into_event(event: Dict[str, Any], tag_result: Dict[str, Any]) -> Dict[str, Any]:
    non_empty_positions = [i for i, m in enumerate(event["messages"]) if not is_message_empty(m)]
    idx_map = {analysis_idx: orig_idx for analysis_idx, orig_idx in enumerate(non_empty_positions)}

    analyzed = tag_result.get("messages", [])
    for entry in analyzed:
        aidx = entry.get("index")
        if aidx is None:
            continue
        oidx = idx_map.get(aidx)
        if oidx is None:
            continue
        tags = entry.get("tags", [])
        notes = entry.get("notes", "")
        if not isinstance(tags, list):
            tags = []
        event["messages"][oidx]["tags"] = tags
        if notes:
            event["messages"][oidx]["tag_notes"] = notes

    if "event_summary" in tag_result and isinstance(tag_result["event_summary"], str):
        event["event_summary"] = tag_result["event_summary"]
    event["tags_version"] = "v1.0"
    event["tags_source"] = "LLM"
    return event

async def call_llm(client: AsyncOpenAI, model: str, prompt: str, retries: int = 3) -> Dict[str, Any]:
    last_err = None
    for attempt in range(retries):
        try:
            # Try with response_format for JSON-only models; fallback if not supported
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    temperature=TEMPERATURE,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": "你是资深内核邮件分析助手。只输出严格 JSON。"},
                        {"role": "user", "content": prompt},
                    ]
                )
            except Exception:
                resp = await client.chat.completions.create(
                    model=model,
                    temperature=TEMPERATURE,
                    messages=[
                        {"role": "system", "content": "你是资深内核邮件分析助手。只输出严格 JSON。"},
                        {"role": "user", "content": prompt},
                    ]
                )
            text = resp.choices[0].message.content or ""
            text = text.strip()
            if text.startswith("```"):
                # strip code fences if present
                parts = text.split("```")
                if len(parts) >= 3:
                    text = parts[1].strip()
            return json.loads(text)
        except (json.JSONDecodeError, OpenAIError, Exception) as e:
            last_err = e
            await asyncio.sleep(1.0 + attempt * 1.5)
            continue
    raise RuntimeError(f"LLM failed to return valid JSON after {retries} attempts: {last_err}")

async def process_file(file_path: Path,
                      out_dir: Path,
                      bad_dir: Path,
                      client: AsyncOpenAI,
                      model: str,
                      tag_set: Set[str],
                      tag_set_lock: asyncio.Lock,
                      normalize_tags: bool) -> None:
    try:
        with file_path.open("r", encoding="utf-8") as f:
            event = json.load(f)
    except Exception:
        dest = bad_dir / file_path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(file_path), str(dest))
        return

    if not validate_event_json(event):
        dest = bad_dir / file_path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(file_path), str(dest))
        return

    non_empty_msgs = []
    for m in event["messages"]:
        if is_message_empty(m):
            continue
        non_empty_msgs.append({
            "subject": m.get("subject", ""),
            "url": m.get("url", ""),
            "content": m.get("content", "")
        })

    # If nothing to analyze, just copy and mark
    if len(non_empty_msgs) == 0:
        out_path = out_dir / file_path.name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        event["tags_version"] = "v1.0"
        event["tags_source"] = "LLM"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(event, f, ensure_ascii=False, indent=2)
        return

    # Snapshot existing tags for prompt
    async with tag_set_lock:
        existing_tags_list = sorted(list(tag_set))[:2000]  # cap to avoid huge prompts

    prompt = build_prompt(
        event_id=event.get("event_id", ""),
        root_url=event.get("root_url", ""),
        messages=non_empty_msgs,
        existing_tags=existing_tags_list
    )

    try:
        tag_result = await call_llm(client, model, prompt)
    except Exception:
        dest = bad_dir / file_path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(file_path), str(dest))
        return

    # Merge into event
    updated_event = merge_tags_into_event(event, tag_result)

    # Collect tags used in this event
    used_tags_in_event: Set[str] = set()
    for msg in updated_event.get("messages", []):
        tags = msg.get("tags")
        if isinstance(tags, list):
            for t in tags:
                if not isinstance(t, str):
                    continue
                used_tags_in_event.add(t)

    # Compute new tags vs global set and update global set
    async with tag_set_lock:
        before = set(tag_set)
        # Optional normalization applied before merging to global set and writing back to messages if enabled
        if normalize_tags:
            # Prepare mapping old->normalized for writing back to messages
            norm_map = {}
            for t in list(used_tags_in_event):
                nt = normalize_tag_soft(t)
                norm_map[t] = nt
            # Rewrite tags in event messages to normalized form
            for msg in updated_event.get("messages", []):
                if isinstance(msg.get("tags"), list):
                    new_list = []
                    for t in msg["tags"]:
                        if isinstance(t, str):
                            new_list.append(norm_map.get(t, normalize_tag_soft(t)))
                    msg["tags"] = list(dict.fromkeys(new_list))  # de-dup preserve order
            # Recompute used_tags_in_event after normalization
            used_tags_in_event = set()
            for msg in updated_event.get("messages", []):
                for t in (msg.get("tags") or []):
                    if isinstance(t, str):
                        used_tags_in_event.add(t)

        new_tags = sorted(list(used_tags_in_event - before))
        # Update global set
        tag_set.update(used_tags_in_event)

    # Record new tags for this event
    if new_tags:
        updated_event["new_tags_in_event"] = new_tags

    # Write output file
    out_path = out_dir / file_path.name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(updated_event, f, ensure_ascii=False, indent=2)

async def run(input_dir: Path,
             output_dir: Path,
             bad_dir: Path,
             model: str,
             concurrency: int,
             tag_set_path: Path,
             normalize_tags: bool,
             base_url: Optional[str] = None,
             api_key: Optional[str] = None,
             timeout: float = 60.0) -> None:
    # Configure AsyncOpenAI client to support custom base_url and api_key (for local providers like Ollama)
    client_kwargs = {}
    # The openai.AsyncOpenAI client accepts 'base_url' and 'api_key' in some local shims; set via environment as fallback
    if base_url:
        os.environ.setdefault("OPENAI_BASE_URL", base_url)
        client_kwargs["base_url"] = base_url
    if api_key:
        os.environ.setdefault("OPENAI_API_KEY", api_key)
        client_kwargs["api_key"] = api_key
    # timeout isn't universally supported in the thin client constructor; pass via client kwargs if available
    client_kwargs["timeout"] = timeout
    try:
        client = AsyncOpenAI(**client_kwargs)
    except Exception:
        # Fallback to default client but ensure env vars are set
        client = AsyncOpenAI()

    files = sorted([p for p in input_dir.glob("*.json") if p.is_file()])
    if not files:
        print("No JSON files found in input_dir.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    bad_dir.mkdir(parents=True, exist_ok=True)

    # Load global tag set
    tag_set: Set[str] = set()
    if tag_set_path.exists():
        try:
            with tag_set_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    tag_set = set([t for t in data if isinstance(t, str)])
                elif isinstance(data, dict) and "tags" in data and isinstance(data["tags"], list):
                    tag_set = set([t for t in data["tags"] if isinstance(t, str)])
        except Exception:
            # If corrupted, start fresh but keep the broken file as .bak
            bak = tag_set_path.with_suffix(".bak")
            try:
                shutil.move(str(tag_set_path), str(bak))
            except Exception:
                pass

    tag_set_lock = asyncio.Lock()

    sem = asyncio.Semaphore(concurrency)
    pbar = tqdm(total=len(files), desc="Processing events", unit="file")

    async def wrapped(fp: Path):
        async with sem:
            await process_file(fp, output_dir, bad_dir, client, model, tag_set, tag_set_lock, normalize_tags)
            pbar.update(1)

    try:
        await asyncio.gather(*[wrapped(fp) for fp in files])
    finally:
        pbar.close()
        # Persist global tag set
        async with tag_set_lock:
            tag_list = sorted(list(tag_set))
        tag_set_path.parent.mkdir(parents=True, exist_ok=True)
        with tag_set_path.open("w", encoding="utf-8") as f:
            json.dump({"tags": tag_list, "count": len(tag_list)}, f, ensure_ascii=False, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Tag LKML event JSONs with an open tag set using LLM")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--bad_dir", type=str, required=True, help="Malformed or failed files moved here")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--tag_set_path", type=str, default=None, help="Path to persist the global tag set JSON")
    parser.add_argument("--normalize_tags", action="store_true", help="Soft-normalize tags (lowercase, trim, collapse spaces)")
    parser.add_argument("--base_url", type=str, default=None, help="Custom OpenAI-compatible base URL (e.g., http://localhost:11434/v1 for Ollama)")
    parser.add_argument("--api_key", type=str, default=None, help="API key; for local Ollama you can use a placeholder like 'ollama'")
    parser.add_argument("--timeout", type=float, default=60.0, help="Request timeout in seconds")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    bad_dir = Path(args.bad_dir)
    tag_set_path = Path(args.tag_set_path) if args.tag_set_path else (output_dir / "tags_set.json")

    asyncio.run(run(input_dir, output_dir, bad_dir,
                    args.model, args.concurrency,
                    tag_set_path, args.normalize_tags,
                    args.base_url, args.api_key, args.timeout))

if __name__ == "__main__":
    main()
