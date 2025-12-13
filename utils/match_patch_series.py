#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
match_patch_series.py
----------------------------------------
功能说明：
    本脚本用于将一批 Linux patch 的事件文件（event JSON）
    与对应的历史版本记录文件（series JSON）进行匹配。
    每当匹配成功，会在输出目录下为该 patch 创建一个独立文件夹，
    并复制：
        1. patch 文件（来自 patch_dir）
        2. 对应的 series 文件（来自 record_dir）
        3. 自动生成的 summary.json（记录版本匹配信息）

使用方法：
    python match_patch_series.py \
        --patch_dir ./patch_jsons \
        --record_dir ./record_jsons \
        --output_dir ./output_series \
        --concurrency 20

参数说明：
    --patch_dir     存放单个 patch JSON 文件的目录
    --record_dir    存放 series（历史版本记录）JSON 文件的目录
    --output_dir    输出文件夹的根目录（自动创建）
    --concurrency   并发线程数量（默认 20）

输出目录结构示例：
    output_series/
    ├── series_lkml_2018_1_10_435/
    │   ├── lkml_2018_1_10_435.json
    │   ├── series_lkml_2018_6_20_1196.json
    │   └── summary.json
    ├── series_lkml_2018_6_22_111/
    │   ├── lkml_2018_6_22_111.json
    │   ├── series_lkml_2018_6_24_183.json
    │   └── summary.json

依赖：
    pip install tqdm
"""

import os
import json
import shutil
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ========== 工具函数 ==========

def load_json_safe(path):
    """安全加载 JSON 文件"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")
        return None

def ensure_dir(path):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)

def copy_safe(src, dst):
    """带错误处理的文件复制"""
    try:
        shutil.copy2(src, dst)
    except Exception as e:
        print(f"[WARN] Failed to copy {src} -> {dst}: {e}")

# ========== 单个 Patch 的处理函数 ==========

def process_patch(patch_file, patch_dir, record_data, record_dir, output_root):
    patch_path = os.path.join(patch_dir, patch_file)
    patch_data = load_json_safe(patch_path)
    if not patch_data:
        return None

    patch_id = patch_data.get("event_id")
    patch_url = patch_data.get("root_url", "")
    matched_record_file = None
    found_versions = {}
    matched_version = None

    # 遍历 record_data 查找匹配项
    for record_file, record in record_data.items():
        if not isinstance(record, dict) or "variants" not in record:
            continue
        for version, info in record["variants"].items():
            if info.get("event_id") == patch_id or info.get("url") == patch_url:
                matched_record_file = record_file
                found_versions = {v: i.get("url") for v, i in record["variants"].items()}
                matched_version = version
                break
        if matched_record_file:
            break

    if matched_record_file:
        series_dir = os.path.join(output_root, patch_id)
        ensure_dir(series_dir)

        # 拷贝 patch 文件
        copy_safe(patch_path, os.path.join(series_dir, patch_file))

        # 拷贝匹配到的 record 文件
        src_record_path = os.path.join(record_dir, matched_record_file)
        if os.path.exists(src_record_path):
            copy_safe(src_record_path, os.path.join(series_dir, matched_record_file))

        # 写 summary.json
        summary = {
            "event_id": patch_id,
            "root_url": patch_url,
            "matched_version": matched_version,
            "found_versions": found_versions,
            "record_variants_count": len(found_versions),
            "matched_record": matched_record_file
        }
        summary_path = os.path.join(series_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        return patch_id
    return None

# ========== 主函数 ==========

def match_and_export_series_parallel(patch_dir, record_dir, output_root, concurrency=20):
    ensure_dir(output_root)
    patch_files = [f for f in os.listdir(patch_dir) if f.endswith(".json")]
    record_files = [f for f in os.listdir(record_dir) if f.endswith(".json")]

    # 预加载 record 数据
    print("[INFO] Loading record files...")
    record_data = {}
    for record_file in tqdm(record_files):
        path = os.path.join(record_dir, record_file)
        data = load_json_safe(path)
        if data:
            record_data[record_file] = data

    print(f"[INFO] Starting parallel matching with concurrency={concurrency} ...")
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(process_patch, pf, patch_dir, record_data, record_dir, output_root)
            for pf in patch_files
        ]
        for _ in tqdm(as_completed(futures), total=len(futures)):
            pass

    print(f"[DONE] All matched series exported to {output_root}")

# ========== 命令行入口 ==========

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Match patch event JSONs with their historical series records and export grouped results."
    )
    parser.add_argument("--patch_dir", required=True, help="Directory containing patch JSON files")
    parser.add_argument("--record_dir", required=True, help="Directory containing record (series) JSON files")
    parser.add_argument("--output_dir", required=True, help="Output root directory for matched series")
    parser.add_argument("--concurrency", type=int, default=20, help="Number of concurrent threads (default=20)")

    args = parser.parse_args()

    match_and_export_series_parallel(
        args.patch_dir,
        args.record_dir,
        args.output_dir,
        args.concurrency
    )
