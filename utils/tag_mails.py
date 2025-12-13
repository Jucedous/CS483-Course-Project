import os
import sys
import json
import time
import random
import requests
import re
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============== 顶部配置（在此填写/修改） ==============
OPENAI_API_KEY = "sk-wZ3Zpfep5N2wa2KXDcBbB29075554b0988196b645cE44615"  # 必填：你的 OpenAI API key
OPENAI_BASE_URL = "https://api.bianxie.ai/v1"  # 可选：兼容端点
OPENAI_MODEL = "gpt-4o-mini"  # 可选：模型名称

# 并发与速率控制
FILE_WORKERS = 20                 # 同时处理的文件数量
MIN_DELAY_BETWEEN_CALLS = 0.25   # 单文件内每次 API 调用之间的最小延迟（秒）
MAX_RETRY = 5                    # OpenAI 调用最大重试次数

# 处理策略
ALWAYS_WRITE_FIELDS = True       # 每个 mail 都写入 tags/llm_tag_meta（空也写）
SKIP_ALREADY_TAGGED = True       # 若邮件已有非空 tags 且带 llm_tag_meta，则跳过
FORCE_RETAG = False              # 设为 True 则无视已有标签，强制重打

# 文本截断
MAX_CHARS_PER_EMAIL = 12000
FRONT_CHARS = 8000
TAIL_CHARS = 4000

# ============== OpenAI 兼容客户端 ==============
def openai_chat(messages: List[Dict[str, str]], temperature: float = 0.1, as_json=True) -> Dict[str, Any]:
    if not OPENAI_API_KEY or OPENAI_API_KEY == "PUT_YOUR_KEY_HERE":
        raise RuntimeError("请在脚本顶部配置 OPENAI_API_KEY")
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": OPENAI_MODEL, "messages": messages, "temperature": temperature}
    if as_json:
        payload["response_format"] = {"type": "json_object"}
    url = f"{OPENAI_BASE_URL}/chat/completions"
    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
    if resp.status_code >= 400:
        raise RuntimeError(f"OpenAI API error: {resp.status_code} {resp.text}")
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    return json.loads(content) if as_json else content

# ============== 提示与角度引导（角度仅作参考，不作为标签输出） ==============
ANGLES_GUIDE = """
请从这些角度观察内容，但不要把它们当作标签输出：
- 社会–技术：维护者重写规律、审阅者偏好、新人主要问题在提交说明、子系统文化差异、组织因素主导延迟；
  研究视角：政策/工具自然实验、审阅网络中心性与把门效应、历史回放若有 LLM、静态工具解释有限。
- 技术：并发与生命周期管理（锁/引用计数/错误路径）、API 迁移、错误路径标准化、提交信息语义影响审阅效率。
"""

SYSTEM_PROMPT = f"""你是严格的技术邮件标注器。任务：基于邮件的主题与正文（可能含 diff），生成具体、内容相关、不泛化的短标签。
注意：
- 这些“角度”只用于启发你关注点，绝不能作为标签输出：{ANGLES_GUIDE}
- 标签必须具体，且可从文本中直接或高度确定地提取：
  - 子系统/模块/组件（如 acpi、acpica、ec、gpe）
  - 具体 api/函数/结构/宏/常量名（如 acpi_set_gpe、ACPI_GPE_DISPATCH_RAW_HANDLER）
  - 关键改动类型或意图（如 race fix、locking order、api addition、polling mode、edge-triggered）
  - 重要行为/结果（如 acked-by、queued for 3.20、patch series v4）
- 避免泛化或抽象。优先选文中出现的专业术语、符号名、文件路径、概念短语。
- 输出 5~12 个标签；每个标签不超过 4 个词；小写；下划线/连字符/点号可用；保留代码符号原样。标签必须为英文。

输出为 JSON：
{{
  "tags": ["...", "..."],
  "note": "（可选）"
}}
"""

def build_user_prompt(email_text: str) -> str:
    return f"""请基于以下邮件文本生成具体短标签（非泛化），并严格按 JSON 返回：
{email_text}
"""

# ============== 文本处理与标签生成 ==============
def truncate_text(s: str, max_chars: int = MAX_CHARS_PER_EMAIL, head: int = FRONT_CHARS, tail: int = TAIL_CHARS) -> str:
    if not s:
        return ""
    if len(s) <= max_chars:
        return s
    return s[:head] + "\n...\n" + s[-tail:]

def build_email_text(msg: Dict[str, Any]) -> str:
    subj = (msg.get("subject") or "").strip()
    cont = (msg.get("content") or "").strip()
    s = ""
    if subj:
        s += f"[subject] {subj}\n\n"
    if cont:
        s += cont
    return truncate_text(s)

def tag_one_email(text: str) -> Dict[str, Any]:
    if not text.strip():
        return {"tags": [], "note": "empty"}
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(text)},
    ]
    for attempt in range(1, MAX_RETRY + 1):
        try:
            out = openai_chat(messages, temperature=0.1, as_json=True)
            tags = out.get("tags") or []
            tags = [normalize_tag(t) for t in tags if isinstance(t, str)]
            tags = dedup_tags(tags)
            note = out.get("note") or ""
            if len(tags) < 3:
                tags = expand_from_text(text, tags)
            return {"tags": tags, "note": note}
        except Exception as e:
            if attempt >= MAX_RETRY:
                return {"tags": [], "note": f"error: {e}"}
            # 指数退避
            backoff = min(2 ** (attempt - 1) + random.random(), 20)
            time.sleep(backoff)

def normalize_tag(tag: str) -> str:
    t = " ".join(tag.strip().split())
    t = t.lower().replace(" ", "_")
    return t.strip(",.;:()[]{}")

def dedup_tags(tags: List[str]) -> List[str]:
    seen, out = set(), []
    for t in tags:
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out

SYMBOL_RE = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]{3,})\b")
SUBSYSTEM_HINTS = ["acpi", "acpica", "ec", "gpe", "irq", "lock", "spinlock", "poll", "edge-triggered", "level-triggered", "storm", "linux", "kernel"]

def expand_from_text(text: str, existing: List[str]) -> List[str]:
    base = set(existing)
    low = text.lower()
    for h in SUBSYSTEM_HINTS:
        if h in low:
            base.add(h.replace(" ", "_"))
    symbols = SYMBOL_RE.findall(text)
    preferred = [s for s in symbols if any(k in s.lower() for k in ["gpe", "acpi", "ec", "handler", "lock"])]
    for s in preferred[:6]:
        base.add(normalize_tag(s))
    out = dedup_tags(list(base))
    return out[:8] if len(out) > 8 else out

# ============== 文件/目录处理 ==============
def gather_json_files(path: str) -> List[str]:
    if os.path.isfile(path):
        return [path] if path.lower().endswith(".json") else []
    found = []
    for root, _, files in os.walk(path):
        for fn in files:
            if fn.lower().endswith(".json"):
                found.append(os.path.join(root, fn))
    return sorted(found)

def should_skip_message(msg: Dict[str, Any]) -> bool:
    if FORCE_RETAG:
        return False
    if not SKIP_ALREADY_TAGGED:
        return False
    # 已有非空 tags 且有 llm_tag_meta 则跳过
    tags = msg.get("tags")
    meta = msg.get("llm_tag_meta")
    if isinstance(tags, list) and len(tags) > 0 and isinstance(meta, dict):
        return True
    return False

def process_json_file(json_path: str) -> Tuple[str, int, int, str]:
    """返回 (文件路径, 更新的邮件数, 调用API次数, 错误信息)"""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return (json_path, 0, 0, f"read_error: {e}")

    msgs = data.get("messages", [])
    now_iso = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    updated = 0
    api_calls = 0
    changed = False

    for m in msgs:
        # 先保证目标字段存在
        if ALWAYS_WRITE_FIELDS and not should_skip_message(m):
            # nothing here; actual write happens after tag generation
            pass

        if should_skip_message(m):
            continue

        text = build_email_text(m)
        if not text.strip():
            res = {"tags": [], "note": "empty"}
        else:
            res = tag_one_email(text)
            api_calls += 1
            # 简单速率限制，避免打爆 QPS
            if MIN_DELAY_BETWEEN_CALLS > 0:
                time.sleep(MIN_DELAY_BETWEEN_CALLS)

        m["tags"] = res.get("tags", [])
        m["llm_tag_meta"] = {
            "model": OPENAI_MODEL,
            "time": now_iso,
            "note": res.get("note", ""),
            "angles_as_guidance": True
        }
        updated += 1
        changed = True

    # 备份（仅首次）
    if changed:
        bak = json_path + ".bak"
        try:
            with open(bak, "x", encoding="utf-8") as bf:
                json.dump(data, bf, ensure_ascii=False, indent=2)
        except FileExistsError:
            pass
        except Exception:
            # 备份失败不致命，继续写主文件
            pass

        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            return (json_path, updated, api_calls, f"write_error: {e}")

    return (json_path, updated, api_calls, "")

def process_path(path: str):
    files = gather_json_files(path)
    if not files:
        print(f"未找到可处理的 JSON：{path}")
        return

    total_files = len(files)
    print(f"共 {total_files} 个 JSON 待处理，文件并发={FILE_WORKERS}")

    results = []
    errors = 0
    total_updated = 0
    total_calls = 0

    if FILE_WORKERS <= 1:
        for idx, fp in enumerate(files, 1):
            res = process_json_file(fp)
            results.append(res)
            total_updated += res[1]
            total_calls += res[2]
            if res[3]:
                errors += 1
                print(f"[{idx}/{total_files}] {fp} -> err: {res[3]}")
            else:
                print(f"[{idx}/{total_files}] {fp} -> 更新{res[1]}封, 调用{res[2]}次")
    else:
        with ThreadPoolExecutor(max_workers=FILE_WORKERS) as ex:
            fut2fp = {ex.submit(process_json_file, fp): fp for fp in files}
            done = 0
            for fut in as_completed(fut2fp):
                fp = fut2fp[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    errors += 1
                    print(f"[{done+1}/{total_files}] {fp} -> 异常: {e}")
                    done += 1
                    continue
                results.append(res)
                total_updated += res[1]
                total_calls += res[2]
                if res[3]:
                    errors += 1
                    print(f"[{done+1}/{total_files}] {fp} -> err: {res[3]}")
                else:
                    print(f"[{done+1}/{total_files}] {fp} -> 更新{res[1]}封, 调用{res[2]}次")
                done += 1

    print(f"完成：{total_files} 个文件，更新邮件 {total_updated} 封，API 调用 {total_calls} 次，错误 {errors} 个。")

# ============== 入口 ==============
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法：python tag_emails.py <path_to_json_or_dir>")
        sys.exit(1)
    process_path(sys.argv[1])

