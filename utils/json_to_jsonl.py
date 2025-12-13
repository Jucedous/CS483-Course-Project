#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re
import json
import argparse
import unicodedata
from typing import List, Dict, Any, Optional
try:
    from ftfy import fix_text
except Exception:
    fix_text = None

FOOTER_PATTERNS = [
    r"http://vger\.kernel\.org/majordomo-info\.html",
    r"http://www\.tux\.org/lkml/",
]
ON_WROTE_RE = re.compile(r"^On .+ wrote:$", flags=re.IGNORECASE)

def normalize_text(s: str, use_ftfy: bool = True) -> str:
    if not s:
        return ""
    if use_ftfy and fix_text is not None:
        s = fix_text(s)
    s = unicodedata.normalize("NFC", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return s

def strip_quotes_and_footers(s: str, strip_quotes: bool = True, strip_footers: bool = True) -> str:
    lines = s.split("\n")
    out = []
    for line in lines:
        l = line.strip()
        if strip_quotes and (l.startswith(">") or ON_WROTE_RE.match(l)):
            continue
        if strip_footers and any(re.search(pat, l) for pat in FOOTER_PATTERNS):
            continue
        out.append(line)
    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def clean_text(s: str, use_ftfy: bool, strip_quotes: bool, strip_footers: bool) -> str:
    s = normalize_text(s, use_ftfy=use_ftfy)
    s = strip_quotes_and_footers(s, strip_quotes=strip_quotes, strip_footers=strip_footers)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()

def build_samples_from_thread(
    thread: Dict[str, Any],
    history_size: int,
    min_chars: int,
    max_chars: Optional[int],
    use_ftfy: bool,
    strip_quotes: bool,
    strip_footers: bool,
) -> List[Dict[str, str]]:
    msgs = thread.get("messages") or []
    if len(msgs) < 2:
        return []
    cleaned, subjects = [], []
    for m in msgs:
        content = clean_text(m.get("content") or "", use_ftfy, strip_quotes, strip_footers)
        subj = (m.get("subject") or "").strip()
        cleaned.append(content)
        subjects.append(subj)
    root_subject = subjects[0] if subjects else ""
    samples: List[Dict[str, str]] = []
    for j in range(1, len(cleaned)):
        target = cleaned[j]
        if not target:
            continue
        if len(target) < min_chars:
            continue
        if max_chars and len(target) > max_chars:
            continue
        start = max(0, j - history_size)
        history_msgs = cleaned[start:j]
        parts = []
        for idx, txt in enumerate(history_msgs, 1):
            parts.append(f"info{idx}：\n{txt}")
        history_str = "\n\n".join(parts) if parts else "(null)"
        curr_subject = subjects[j] if j < len(subjects) else root_subject
        instruction = (
            "You are a technical assistant. The following is a history of an email discussion. Based on the context, please compose the next reasonable reply.\n"
            f"suject：{root_subject or curr_subject}\n"
            "history：\n"
            f"{history_str}\n"
            "next response："
        )
        samples.append({"instruction": instruction, "input": "", "output": target})
    return samples

def iter_json_files_in_dir(input_dir: str, recursive: bool, ext: str = ".json"):
    if not os.path.isdir(input_dir):
        raise ValueError(f"--input-dir 必须是目录：{input_dir}")
    if recursive:
        for root, _, files in os.walk(input_dir):
            for fn in files:
                if fn.lower().endswith(ext):
                    yield os.path.join(root, fn)
    else:
        for fn in os.listdir(input_dir):
            fp = os.path.join(input_dir, fn)
            if os.path.isfile(fp) and fn.lower().endswith(ext):
                yield fp

def main():
    parser = argparse.ArgumentParser(description="将目录中的 LKML JSON 线程转为 SFT JSONL（instruction/input/output）。")
    parser.add_argument("--input-dir", required=True, help="输入目录（只接受目录）")
    parser.add_argument("--output", required=True, help="输出 JSONL 文件路径")
    parser.add_argument("--recursive", action="store_true", help="递归遍历子目录")
    parser.add_argument("--ext", default=".json", help="仅处理该扩展名的文件，默认 .json")
    parser.add_argument("--history-size", type=int, default=3, help="指令中包含的历史消息条数")
    parser.add_argument("--min-chars", type=int, default=30, help="输出最少字符数过滤")
    parser.add_argument("--max-chars", type=int, default=0, help="输出最多字符数；0表示不限制")
    parser.add_argument("--no-ftfy", action="store_true", help="不使用 ftfy 修复文本")
    parser.add_argument("--keep-quotes", action="store_true", help="保留以 '>' 开头的引用行与 'On ... wrote:'")
    parser.add_argument("--keep-footers", action="store_true", help="保留常见页脚/链接行")
    parser.add_argument("--limit", type=int, default=0, help="最多导出样本条数；0表示不限制")
    parser.add_argument("--log-every", type=int, default=200, help="每处理多少个文件打印一次进度")
    args = parser.parse_args()

    use_ftfy = not args.no_ftfy
    max_chars = args.max_chars if args.max_chars and args.max_chars > 0 else None

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    total_threads = 0
    total_msgs = 0
    total_samples = 0
    written_limit = args.limit if args.limit and args.limit > 0 else None

    with open(args.output, "w", encoding="utf-8") as out_f:
        for i, fp in enumerate(iter_json_files_in_dir(args.input_dir, args.recursive, args.ext)):
            if args.log_every and i % args.log_every == 0:
                print(f"[INFO] 进度：文件 {i} -> {fp}")
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"[WARN] 读取失败: {fp}: {e}")
                continue
            total_threads += 1
            msgs = data.get("messages") or []
            total_msgs += len(msgs)

            samples = build_samples_from_thread(
                data,
                history_size=args.history_size,
                min_chars=args.min_chars,
                max_chars=max_chars,
                use_ftfy=use_ftfy,
                strip_quotes=not args.keep_quotes,
                strip_footers=not args.keep_footers,
            )
            for row in samples:
                if written_limit is not None and total_samples >= written_limit:
                    break
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                total_samples += 1
            if written_limit is not None and total_samples >= written_limit:
                print(f"[INFO] 达到样本上限 {written_limit}，停止。")
                break

    print(f"完成：处理文件={total_threads}，原始消息数={total_msgs}，导出样本数={total_samples}")
    if total_samples == 0:
        print("提示：请检查输入目录、扩展名过滤，或调低 min-chars / 调整 history-size。")

if __name__ == "__main__":
    main()