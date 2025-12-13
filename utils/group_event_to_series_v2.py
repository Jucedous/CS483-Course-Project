#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LKML Event Series Grouper (Parallel Accelerated Version)
======================================================

功能:
    并行读取大量 Event JSON 文件，按主题聚合为 Series。
    Series ID 自动取该系列中最早的 Event ID。

优化点:
    1. 使用 multiprocessing 并行读取和解析 JSON (大幅减少 I/O 和正则匹配时间)。
    2. 主进程仅负责简单的字典聚合，效率极大提升。

用法:
    python group_event_to_series_v2 \
        --input ./data/events \
        --output ./data/series \
        --workers 8
"""

import os
import re
import json
import argparse
import multiprocessing
from collections import defaultdict
from tqdm import tqdm # 建议安装: pip install tqdm

# ==========================================
# 1. 纯函数 worker (必须在全局作用域)
# ==========================================

def parse_subject_general(subject: str):
    """解析主题结构 (CPU 密集型)"""
    if not subject:
        return {
            "prefix": "GENERAL",
            "version": "v1",
            "topic": "unknown_topic",
            "normalized_subject": "unknown"
        }

    subject = subject.strip()
    subject = re.sub(r'^\s*(?:Re|Fwd|Aw|Sv):\s*', '', subject, flags=re.IGNORECASE)
    
    pattern = re.compile(
        r'\[(?P<prefix>[^\]]*(?:PATCH|RFC|GIT PULL|RESEND|TEST)[^\]]*)\]'
        r'[\s:-]*(?P<core>.*)',
        re.IGNORECASE
    )
    m = pattern.match(subject)
    
    if not m:
        return {
            "prefix": "GENERAL",
            "version": "v1",
            "topic": subject.strip(),
            "normalized_subject": subject.strip()
        }

    raw_prefix_content = m.group('prefix').strip().upper()
    core = m.group('core').strip()

    version_match = re.search(r'\bv(\d+)', raw_prefix_content, re.IGNORECASE)
    version = f"v{version_match.group(1)}" if version_match else "v1"
    
    type_match = re.search(r'[A-Z]+', raw_prefix_content)
    clean_prefix = type_match.group(0) if type_match else "PATCH"

    core = re.sub(r'^\d+/\d+\s*[:\-]?\s*', '', core)
    normalized_subject = f"{clean_prefix} {core}".strip()

    return {
        "prefix": clean_prefix,
        "version": version,
        "topic": core,
        "normalized_subject": normalized_subject
    }

def extract_subject_from_messages(data):
    """从 JSON 数据中提取主题"""
    messages = data.get("messages", [])
    if not isinstance(messages, list):
        return ""
    for msg in messages:
        candidates = [msg.get("subject"), msg.get("title")]
        for c in candidates:
            if c and isinstance(c, str) and c.strip():
                return c.strip()
    return ""

def worker_process_file(file_path):
    """
    [Worker 进程] 读取单个文件并提取聚合所需的关键信息。
    不返回完整的大 JSON，只返回轻量级元数据，减少进程间通信开销。
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        raw_subject = extract_subject_from_messages(data)
        if not raw_subject:
            return None

        subj_info = parse_subject_general(raw_subject)
        
        # 返回聚合所需的数据包
        return {
            "key": subj_info["normalized_subject"],
            "topic": subj_info["topic"],
            "prefix": subj_info["prefix"],
            "version": subj_info["version"],
            "variant_data": {
                "event_id": data.get("event_id"),
                "root_url": data.get("root_url"),
                "source_file": os.path.basename(file_path),
                "message_count": data.get("message_count", 0)
            }
        }
    except Exception:
        return None

def parse_event_id_to_tuple(event_id):
    """排序辅助函数"""
    if not event_id: return (9999, 99, 99, 9999)
    try:
        clean_id = event_id.replace("lkml_", "")
        parts = [int(p) for p in clean_id.split("_") if p.isdigit()]
        if len(parts) >= 4: return tuple(parts[:4])
        return tuple(parts)
    except: return (9999, 99, 99, 9999)


# ==========================================
# 2. 主流程
# ==========================================

def group_event_series_parallel(input_dir, output_dir, workers=None):
    if not workers:
        workers = multiprocessing.cpu_count()

    # 1. 扫描所有文件路径
    print(f"Scanning files in {input_dir}...")
    all_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".json")]
    total_files = len(all_files)
    print(f"Found {total_files} files. Starting parallel processing with {workers} workers.")

    # 2. 准备聚合容器
    grouped = defaultdict(lambda: {
        "series_id": "",
        "subject": "",
        "topic_type": "",
        "variants": {},
        "connections": []
    })

    # 3. 并行 Map-Reduce
    # 使用 imap_unordered 边处理边返回，节省内存
    with multiprocessing.Pool(processes=workers) as pool:
        # chunksize 设大一点 (比如 50) 可以减少进程通信次数
        iterator = pool.imap_unordered(worker_process_file, all_files, chunksize=50)
        
        for result in tqdm(iterator, total=total_files, unit="file"):
            if not result:
                continue
            
            # --- Reduce Phase (Main Process) ---
            key = result["key"]
            group = grouped[key]
            
            # 填充组信息 (第一次遇到该主题时)
            if not group["subject"]:
                group["subject"] = result["topic"]
                group["topic_type"] = result["prefix"]
            
            # 存入 Variant
            version = result["version"]
            group["variants"][version] = result["variant_data"]

    print(f"Aggregation complete. Found {len(grouped)} unique series.")
    print(f"Writing output to {output_dir}...")

    os.makedirs(output_dir, exist_ok=True)
    output_count = 0

    # 4. 串行写入 (由于文件数已聚合减少，写入通常很快，无需并行)
    for k, v in tqdm(grouped.items(), desc="Writing Series"):
        if not v["variants"]:
            continue

        # 寻找最早的 Event ID 作为 Series ID
        all_event_ids = [var.get("event_id") for var in v["variants"].values() if var.get("event_id")]
        if not all_event_ids:
            continue

        earliest_event_id = sorted(all_event_ids, key=parse_event_id_to_tuple)[0]
        v["series_id"] = earliest_event_id

        # 构建版本连接
        def version_sort_key(v_str):
            nums = re.findall(r'\d+', v_str)
            return int(nums[0]) if nums else 0
        
        sorted_versions = sorted(v["variants"].keys(), key=version_sort_key)
        v["connections"] = [
            {"from": sorted_versions[i], "to": sorted_versions[i + 1]} 
            for i in range(len(sorted_versions) - 1)
        ]

        # 写入文件
        safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', earliest_event_id)
        out_path = os.path.join(output_dir, f"series_{safe_name}.json")
        
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(v, f, ensure_ascii=False, indent=2)
            output_count += 1
        except Exception as e:
            pass

    print(f"Done! Created {output_count} series files.")

if __name__ == "__main__":
    multiprocessing.freeze_support() # Windows 必须
    
    parser = argparse.ArgumentParser(description="LKML Event Series Grouper (Parallel)")
    parser.add_argument("--input", required=True, help="Input directory with event JSONs")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--workers", type=int, default=None, help="Number of CPU cores to use")
    
    args = parser.parse_args()
    
    group_event_series_parallel(args.input, args.output, args.workers)
