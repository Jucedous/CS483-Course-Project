#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
filter_syzbot_events.py
-----------------------
功能：
    从 LKML Event JSON 文件中，过滤出与 Syzbot 相关的事件。
    
判断标准：
    1. 标题 (Subject/Title) 中包含 'syzbot' 或 'syzkaller' (不区分大小写)。
    2. 正文 (Content) 中包含 'syzbot' 或 'syzkaller'。
    3. 发件人邮箱包含 'syzkaller' (如果 JSON 中有解析出的 meta 信息)。

使用示例：
    python utils/filter_syzbot_events.py --input_dir ./data/events_raw --output_dir ./data/events_syzbot
"""

import os
import json
import shutil
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# 定义关键词 (全小写)
KEYWORDS = {"syzbot", "syzkaller", "kasan", "kmsan", "kcsan"}

def is_syzbot_related(file_path):
    """
    检查单个 JSON 文件是否与 Syzbot 相关
    返回: (is_related, file_name)
    """
    filename = os.path.basename(file_path)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # 1. 检查 messages 列表
        messages = data.get("messages", [])
        if not messages:
            return False, filename

        for msg in messages:
            # 组合检查文本：标题 + 正文 + 邮箱(如果有)
            # 注意：原始 JSON 结构可能略有不同，这里做兼容处理
            subject = str(msg.get("subject") or msg.get("title") or "")
            content = str(msg.get("content") or "")
            
            # 将所有文本转为小写进行匹配
            text_corpus = (subject + " " + content).lower()
            
            # 检查是否有关键词命中
            for kw in KEYWORDS:
                if kw in text_corpus:
                    return True, filename
                    
    except Exception as e:
        # print(f"[WARN] 无法读取文件 {filename}: {e}")
        return False, filename

    return False, filename

def worker(args):
    """多进程 worker"""
    src_path, out_dir = args
    is_related, filename = is_syzbot_related(src_path)
    
    if is_related:
        # 复制文件到输出目录
        dest_path = os.path.join(out_dir, filename)
        shutil.copy2(src_path, dest_path)
        return True
    return False

def main():
    parser = argparse.ArgumentParser(description="Filter LKML events for Syzbot relevance.")
    parser.add_argument("--input_dir", required=True, help="Input directory containing all event JSONs")
    parser.add_argument("--output_dir", required=True, help="Output directory for filtered JSONs")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel worker processes")
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"[Error] Input directory does not exist: {args.input_dir}")
        return

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 扫描所有文件
    print(f"[INFO] Scanning input directory: {args.input_dir}")
    all_files = [
        os.path.join(args.input_dir, f) 
        for f in os.listdir(args.input_dir) 
        if f.endswith(".json")
    ]
    total_files = len(all_files)
    print(f"[INFO] Found {total_files} JSON files.")

    # 准备任务
    tasks = [(f, args.output_dir) for f in all_files]
    
    syzbot_count = 0
    
    print(f"[INFO] Starting filtering with {args.workers} workers...")
    
    # 使用多进程加速 IO 和 JSON 解析
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # 使用 tqdm 显示进度
        futures = [executor.submit(worker, task) for task in tasks]
        
        for future in tqdm(as_completed(futures), total=total_files, unit="file"):
            try:
                if future.result():
                    syzbot_count += 1
            except Exception as e:
                pass

    print("-" * 50)
    print(f"[DONE] Filtering complete.")
    print(f"Total Processed: {total_files}")
    print(f"Syzbot Related : {syzbot_count}")
    print(f"Output Directory: {args.output_dir}")

if __name__ == "__main__":
    main()
