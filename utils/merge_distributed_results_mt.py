#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_distributed_results_mt.py
--------------------------------------------------------
多线程并行合并版
功能：
1. 扫描多个机器的输出目录。
2. 解决冲突（安全优先原则）。
3. 使用线程池并发复制文件，极大提升合并速度。

使用方法：
python utils/merge_distributed_results_mt.py \
    --inputs ./data/m1 ./data/m2 ./data/m3 \
    --output ./data/final \
    --workers 16
"""

import os
import shutil
import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def copy_worker(task):
    """线程工作函数：执行单个文件复制"""
    src, dest = task
    try:
        shutil.copy2(src, dest)
        return True
    except Exception as e:
        return f"Error copying {src}: {e}"

def main():
    parser = argparse.ArgumentParser(description="Merge Distributed Results (Multi-threaded)")
    parser.add_argument("--inputs", nargs='+', required=True, help="Input directories")
    parser.add_argument("--output", required=True, help="Final output directory")
    parser.add_argument("--workers", type=int, default=8, help="Number of copy threads (default: 8)")
    args = parser.parse_args()

    # 1. 准备目录
    final_out = os.path.abspath(args.output)
    final_rel = os.path.join(final_out, "security_relevant")
    final_irr = os.path.join(final_out, "security_irrelevant")
    ensure_dir(final_rel)
    ensure_dir(final_irr)

    # 2. 扫描阶段 (单线程，确保逻辑正确)
    series_map = defaultdict(lambda: {'relevant': [], 'irrelevant': []})
    print(f"[INIT] Scanning {len(args.inputs)} inputs...")

    for input_dir in args.inputs:
        if not os.path.exists(input_dir):
            continue
        # 扫描 relevant
        rel_dir = os.path.join(input_dir, "security_relevant")
        if os.path.exists(rel_dir):
            for f in os.listdir(rel_dir):
                if f.endswith(".json"):
                    series_map[f]['relevant'].append(os.path.join(rel_dir, f))
        # 扫描 irrelevant
        irr_dir = os.path.join(input_dir, "security_irrelevant")
        if os.path.exists(irr_dir):
            for f in os.listdir(irr_dir):
                if f.endswith(".json"):
                    series_map[f]['irrelevant'].append(os.path.join(irr_dir, f))

    print(f"[PLAN] Found {len(series_map)} unique series. Resolving conflicts...")

    # 3. 决策阶段：生成复制任务列表
    copy_tasks = [] # List of (src_path, dest_path)
    
    cnt_conflict = 0
    cnt_duplicate = 0
    
    # 遍历字典，决定每个文件的最终去向
    for filename, paths in series_map.items():
        has_relevant = len(paths['relevant']) > 0
        has_irrelevant = len(paths['irrelevant']) > 0
        
        src_path = None
        dest_folder = None
        
        if has_relevant and has_irrelevant:
            # 冲突 -> 强制选 Relevant
            src_path = paths['relevant'][0]
            dest_folder = final_rel
            cnt_conflict += 1
        elif has_relevant:
            src_path = paths['relevant'][0]
            dest_folder = final_rel
            if len(paths['relevant']) > 1: cnt_duplicate += 1
        elif has_irrelevant:
            src_path = paths['irrelevant'][0]
            dest_folder = final_irr
            if len(paths['irrelevant']) > 1: cnt_duplicate += 1
            
        if src_path and dest_folder:
            dest_path = os.path.join(dest_folder, filename)
            # 只有当目标不存在时才添加任务 (支持断点续传)
            if not os.path.exists(dest_path):
                copy_tasks.append((src_path, dest_path))

    print(f"[PLAN] {cnt_conflict} conflicts resolved (promoted to relevant).")
    print(f"[PLAN] {cnt_duplicate} duplicates ignored.")
    print(f"[PLAN] {len(copy_tasks)} files need to be copied.")
    
    if not copy_tasks:
        print("[DONE] Nothing to copy.")
        return

    # 4. 执行阶段 (多线程)
    print(f"[EXEC] Starting copy with {args.workers} threads...")
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # 提交所有任务
        futures = [executor.submit(copy_worker, task) for task in copy_tasks]
        
        # 使用 tqdm 显示进度
        for f in tqdm(as_completed(futures), total=len(futures), unit="file"):
            result = f.result()
            if result is not True:
                print(f"[ERR] {result}")

    # 5. 合并日志 (单线程)
    print("[LOG] Merging log files...")
    final_log = os.path.join(final_out, "merged_analysis.log")
    with open(final_log, "w", encoding="utf-8") as outfile:
        for input_dir in args.inputs:
            possible_logs = ["full_analysis.log", "llm_security_judgment.log"]
            for log_name in possible_logs:
                log_path = os.path.join(input_dir, log_name)
                if os.path.exists(log_path):
                    outfile.write(f"\n=== Merging Log from {input_dir} ===\n")
                    try:
                        with open(log_path, "r", encoding="utf-8") as infile:
                            shutil.copyfileobj(infile, outfile)
                    except: pass

    print(f"[DONE] All operations completed. Results in {final_out}")

if __name__ == "__main__":
    main()
