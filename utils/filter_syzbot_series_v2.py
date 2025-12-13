#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
filter_syzbot_series.py
-----------------------
功能：
    1. 扫描 Series 文件夹，建立 "Event ID <-> Series File" 的映射关系。
    2. 扫描 Events 文件夹，找出内容包含 Syzbot 关键词的 Events。
    3. 找到这些 Syzbot Events 所属的 Series。
    4. 将这些 Series 文件，以及这些 Series 所引用的 *所有* Events 文件（不仅仅是提到 Syzbot 的那个）复制到输出目录。

输入结构：
    --series_dir: 存放 series_*.json 的文件夹
    --events_dir: 存放 lkml_*.json 的文件夹
    --output_dir: 结果输出文件夹
    
python utils/filter_syzbot_series.py \
  --series_dir ./data/series_raw \
  --events_dir ./data/events_raw \
  --output_dir ./data/syzbot_dataset
"""

import os
import json
import shutil
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from tqdm import tqdm

# 定义关键词 (全小写)
KEYWORDS = {"syzbot", "syzkaller", "kasan", "kmsan", "kcsan"}

def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def analyze_series_file(file_path):
    """
    解析单个 Series 文件
    返回: (series_filename, list_of_event_ids_in_this_series)
    """
    filename = os.path.basename(file_path)
    data = load_json(file_path)
    
    if not data:
        return filename, []
    
    event_ids = []
    # 根据提供的 JSON 结构，event_id 在 variants -> vX -> event_id 中
    variants = data.get("variants", {})
    for version, info in variants.items():
        eid = info.get("event_id")
        if eid:
            event_ids.append(eid)
            
    return filename, event_ids

def check_event_for_syzbot(file_path):
    """
    检查单个 Event 文件是否包含 syzbot 关键词
    返回: (filename, is_related, event_id_from_content)
    """
    filename = os.path.basename(file_path)
    data = load_json(file_path)
    
    if not data:
        return filename, False, None

    # 尝试获取 event_id，通常文件名就是 event_id.json，但也可能在 json 内部
    event_id = data.get("event_id")
    if not event_id:
        # 如果 json 里没有，假设文件名去掉 .json 就是 id
        event_id = os.path.splitext(filename)[0]

    messages = data.get("messages", [])
    is_related = False
    
    if messages:
        for msg in messages:
            subject = str(msg.get("subject") or msg.get("title") or "")
            content = str(msg.get("content") or "")
            text_corpus = (subject + " " + content).lower()
            
            for kw in KEYWORDS:
                if kw in text_corpus:
                    is_related = True
                    break
            if is_related:
                break
                
    return filename, is_related, event_id

def copy_file(src_path, dest_path):
    if not os.path.exists(dest_path):
        try:
            shutil.copy2(src_path, dest_path)
            return 1
        except Exception:
            return 0
    return 0

def main():
    parser = argparse.ArgumentParser(description="Filter Syzbot related Series and Events.")
    parser.add_argument("--series_dir", required=True, help="Directory containing Series JSONs")
    parser.add_argument("--events_dir", required=True, help="Directory containing Event JSONs")
    parser.add_argument("--output_dir", required=True, help="Output directory for all related files")
    parser.add_argument("--workers", type=int, default=8, help="Number of worker processes")
    args = parser.parse_args()

    # 0. 准备路径
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    series_files = [os.path.join(args.series_dir, f) for f in os.listdir(args.series_dir) if f.endswith(".json")]
    event_files = [os.path.join(args.events_dir, f) for f in os.listdir(args.events_dir) if f.endswith(".json")]

    print(f"[INFO] Found {len(series_files)} Series files and {len(event_files)} Event files.")

    # ---------------------------------------------------------
    # Step 1: 建立索引 (Map Series to Events)
    # ---------------------------------------------------------
    print("[Step 1/4] Indexing Series files...")
    
    # 存储: Event_ID -> Series_Filename (用于反查)
    event_to_series_map = {}
    # 存储: Series_Filename -> List[Event_IDs] (用于最后提取所有相关事件)
    series_content_map = {}

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(analyze_series_file, f) for f in series_files]
        for future in tqdm(as_completed(futures), total=len(series_files), desc="Indexing Series"):
            s_filename, e_ids = future.result()
            if e_ids:
                series_content_map[s_filename] = e_ids
                for eid in e_ids:
                    # 注意：一个 event 理论上可能属于多个 series（极少见），这里简单覆盖或记录均可
                    event_to_series_map[eid] = s_filename

    # ---------------------------------------------------------
    # Step 2: 筛选 Syzbot Events
    # ---------------------------------------------------------
    print("[Step 2/4] Scanning Events for keywords...")
    
    syzbot_related_event_ids = set()
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(check_event_for_syzbot, f) for f in event_files]
        for future in tqdm(as_completed(futures), total=len(event_files), desc="Scanning Events"):
            fname, is_related, eid = future.result()
            if is_related and eid:
                syzbot_related_event_ids.add(eid)

    print(f"[INFO] Found {len(syzbot_related_event_ids)} events directly mentioning Syzbot.")

    # ---------------------------------------------------------
    # Step 3: 确定需要复制的文件列表
    # ---------------------------------------------------------
    print("[Step 3/4] Resolving dependencies...")

    series_to_copy = set()
    events_to_copy = set()

    # 3.1 找到相关的 Series
    for eid in syzbot_related_event_ids:
        # 查找该 Event 是否属于某个 Series
        if eid in event_to_series_map:
            series_filename = event_to_series_map[eid]
            series_to_copy.add(series_filename)
        else:
            # 如果这是一个 orphan event (不属于任何 series)，是否要复制？
            # 这里的逻辑是：用户想要 "Series"，所以如果它不属于 Series，可能不需要放入结果，
            # 或者你可以选择把这个孤立 Event 也放进去。
            # 这里我选择：如果用户只想要 Series 相关的，就只存 Series 相关的。
            # 如果你也想保存孤立的 Syzbot Event，请取消下面这行的注释：
            # events_to_copy.add(eid) 
            pass

    # 3.2 找到这些 Series 包含的所有 Events (上下文扩展)
    for s_filename in series_to_copy:
        related_eids = series_content_map.get(s_filename, [])
        for related_eid in related_eids:
            events_to_copy.add(related_eid)

    print(f"[INFO] Identified {len(series_to_copy)} Series files and {len(events_to_copy)} Event files to copy.")

    # ---------------------------------------------------------
    # Step 4: 执行复制
    # ---------------------------------------------------------
    print("[Step 4/4] Copying files...")

    copy_count = 0
    
    # 复制 Series
    for s_file in tqdm(series_to_copy, desc="Copying Series"):
        src = os.path.join(args.series_dir, s_file)
        dst = os.path.join(args.output_dir, s_file)
        copy_count += copy_file(src, dst)

    # 复制 Events
    # 注意：event_id 需要转换回文件名。通常是 ID + ".json"
    # 如果你的文件名不规则，可能需要在 Step 2 记录 ID 到 文件名 的映射
    for eid in tqdm(events_to_copy, desc="Copying Events"):
        # 假设文件名是 id.json
        filename = f"{eid}.json"
        src = os.path.join(args.events_dir, filename)
        dst = os.path.join(args.output_dir, filename)
        
        # 检查文件是否存在（因为 ID 是从 Series 里解析的，对应的 Event 文件可能还没下载或已丢失）
        if os.path.exists(src):
            copy_count += copy_file(src, dst)
        else:
            # 尝试另一种命名可能 (e.g. 有些 ID 带有 lkml_ 前缀有些没有)
            pass 

    print("-" * 50)
    print(f"[DONE] Extraction complete.")
    print(f"Output Directory: {args.output_dir}")
    print(f"Total Files Copied: {copy_count}")

if __name__ == "__main__":
    main()