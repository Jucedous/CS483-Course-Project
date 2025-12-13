#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
security_filter_series_v2.py
--------------------------------------------------------
功能：
1. 读取 series_dir 下的 series_*.json 文件。
2. 读取对应的 event json (只读，用于提取内容)。
3. 使用 Ollama 判断是否与 "System Security" 相关。
4. 根据判定结果，将包含分析数据的 series.json 保存到：
   - output_dir/security_relevant/
   - output_dir/security_irrelevant/

--series_dir: 包含 series_*.json 文件的目录。
--events_dir: 包含 event json 文件的目录。
--output_dir: 输出分类结果的目录。
--ollama_model: 使用的 Ollama 模型名称 (默认: gamma3:12b)。
--force: 强制重新分析已存在的文件。
--retries: LLM 请求失败时的重试次数 (默认: 2)。


使用示例：
python security_filter_series_v2.py \
    --series_dir ./data/all/1_series_json_data \
    --events_dir ./data/all/0_event_json_data \
    --output_dir ./data/all/3_classified_series \
    --ollama_model gemma3:12b \
    --start_index 6600 \
    --end_index 10000



新增功能：
支持通过 --start_index 和 --end_index 指定处理范围。
例如：总共有 10000 个文件，你有 2 台机器。
机器 A 运行： --start_index 0 --end_index 5000
机器 B 运行： --start_index 5000 --end_index 10000

注意：脚本内部会对文件列表强制排序 (sort)，确保不同机器看到的文件顺序一致。
"""

import os
import json
import shutil
import hashlib
import argparse
import subprocess
import time
import datetime
import sys
import traceback
import re


def get_timestamp():
    return datetime.datetime.now().strftime("%H:%M:%S")

def log(sid, variant, msg, level="INFO"):
    """Console log printer"""
    timestamp = get_timestamp()
    variant_tag = f"[{variant}]" if variant else "[-]"
    # Color codes
    color = "\033[0m"
    if level == "SECURE": color = "\033[31m" # Red
    elif level == "SAFE": color = "\033[32m" # Green
    elif level == "LLM": color = "\033[36m"  # Cyan
    
    print(f"{color}[{timestamp}] [{level:<6}] [{sid}] {variant_tag} {msg}\033[0m")

def write_judgment_log(log_path, sid, variant_id, is_security, reason, title):
    """Write detailed judgment log to file"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status = "SECURITY_RELEVANT" if is_security else "NOT_SECURITY"
    content = (
        f"[{timestamp}] Series: {sid} | Variant: {variant_id}\n"
        f"Title: {title}\n"
        f"Judgment: {status}\n"
        f"Reason: {reason}\n"
        f"{'-'*80}\n"
    )
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        print(f"[ERR] Failed to write judgment log: {e}")

# ---------- Basic Utils ----------

def load_json_safe(path):
    try:
        with open(path, "r", encoding="utf-8") as f: return json.load(f)
    except: return None

def save_json(data, path):
    try:
        with open(path, "w", encoding="utf-8") as f: json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e: print(f"[ERR] Failed to save JSON: {e}")

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:16]

# ---------- Health Checks ----------

def check_ollama_health(model_name):
    print("-" * 50)
    print(f"[INIT] Checking Ollama Environment...")
    if shutil.which("ollama") is None:
        print(f"[FATAL] 'ollama' command not found. Please install Ollama first.")
        return False
    try:
        subprocess.run(["ollama", "list"], capture_output=True, check=True)
        res = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if model_name not in res.stdout:
            print(f"[WARN] Model '{model_name}' not found locally. Attempting pull...")
            subprocess.run(["ollama", "pull", model_name], check=True)
        else:
            print(f"[INIT] Model '{model_name}' is ready.")
    except Exception as e:
        print(f"[FATAL] Ollama check failed: {e}")
        return False
    print(f"[INIT] Ollama system is healthy.")
    print("-" * 50)
    return True

# ---------- LLM Core ----------

class OllamaCache:
    def __init__(self, path):
        self.path = path
        self.data = {}
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f: self.data = json.load(f)
            except: pass
    
    def get(self, content_hash): 
        return self.data.get(content_hash)
    
    def set(self, content_hash, is_sec, reason): 
        self.data[content_hash] = {"is_security": is_sec, "reason": reason}
    
    def save(self):
        with open(self.path, "w", encoding="utf-8") as f: json.dump(self.data, f, indent=2, ensure_ascii=False)

def analyze_security_relevance(model, title, content, cache, max_retries=2):
    input_text = f"{title}\n{content}"
    input_hash = short_hash(input_text)
    
    cached = cache.get(input_hash)
    if cached:
        return cached["is_security"], cached["reason"]

    prompt = f"""
You are a Linux Kernel Security Expert.
Analyze the following patch/email title and content.
Determine if this patch is related to **System Security**.
(e.g., fixing a vulnerability, CVE, memory safety issue, buffer overflow, race condition, permission check, or hardening).

Title: {title}
Content Snippet:
{content[:8000]} 

Instructions:
1. Answer 'YES' or 'NO' in the very first line.
2. In the second line, provide a concise 1-sentence reason describing the security impact (or lack thereof).
"""

    for attempt in range(max_retries + 1):
        try:
            res = subprocess.run(
                ["ollama", "run", model], 
                input=prompt.encode("utf-8"), 
                capture_output=True, 
                timeout=120
            )
            if res.returncode != 0: raise Exception(res.stderr.decode())
            
            out = res.stdout.decode().strip()
            lines = out.split("\n")
            first_line = lines[0].strip().upper()
            
            is_security = "YES" in first_line
            reason = " ".join(lines[1:]).strip()
            if not reason: reason = out 
            reason = re.sub(r'\s+', ' ', reason)
            
            cache.set(input_hash, is_security, reason)
            return is_security, reason

        except subprocess.TimeoutExpired:
            if attempt < max_retries: time.sleep(2)
        except Exception:
            if attempt < max_retries: time.sleep(2)
            
    return False, "LLM analysis timed out or failed."

# ---------- Main Logic ----------

def process_single_series(series_path, events_dir, dir_relevant, dir_irrelevant, log_file, model, force, global_cache):
    sdata = load_json_safe(series_path)
    if not sdata: return

    sid = sdata.get("series_id", os.path.splitext(os.path.basename(series_path))[0])
    series_filename = os.path.basename(series_path)
    
    # Target Paths
    path_relevant = os.path.join(dir_relevant, series_filename)
    path_irrelevant = os.path.join(dir_irrelevant, series_filename)

    # Check Skip (如果任一文件夹中已存在，则跳过)
    if not force:
        if os.path.exists(path_relevant):
            log(sid, None, "Already classified as RELEVANT. Skipping.", "SKIP")
            return
        if os.path.exists(path_irrelevant):
            log(sid, None, "Already classified as IRRELEVANT. Skipping.", "SKIP")
            return

    variants = sdata.get("variants", {})
    analyzed_variants = {}
    
    security_flag_count = 0
    total_variants = len(variants)
    
    log(sid, None, f"Start processing {total_variants} variants...", "START")

    for ver, info in variants.items():
        src_file = info.get("source_file")
        if not src_file:
            analyzed_variants[ver] = info
            continue

        src_full = os.path.join(events_dir, src_file)
        if not os.path.exists(src_full):
            for root, _, files in os.walk(events_dir):
                if src_file in files:
                    src_full = os.path.join(root, src_file)
                    break
        
        if not os.path.exists(src_full):
            log(sid, ver, f"Event file missing: {src_file}", "ERR")
            analyzed_variants[ver] = info
            continue

        event_data = load_json_safe(src_full)
        if not event_data: continue

        msg = event_data.get("messages", [{}])[0]
        title = msg.get("subject", "") or msg.get("title", "")
        content = msg.get("content", "")

        if not content:
            log(sid, ver, "Content empty.", "WARN")
            analyzed_variants[ver] = info
            continue

        log(sid, ver, f"Analyzing title: {title[:40]}...", "LLM")
        is_sec, reason = analyze_security_relevance(model, title, content, global_cache)
        
        log_level = "SECURE" if is_sec else "SAFE"
        log(sid, ver, f"Judgment: {log_level} | Reason: {reason[:60]}...", log_level)
        
        write_judgment_log(log_file, sid, ver, is_sec, reason, title)

        if is_sec: security_flag_count += 1

        updated_info = info.copy()
        updated_info["security_analysis"] = {
            "is_security_relevant": is_sec,
            "reason": reason,
            "analyzed_at": datetime.datetime.now().isoformat()
        }
        analyzed_variants[ver] = updated_info

    is_series_secure = security_flag_count > 0
    
    final_data = sdata.copy()
    final_data["variants"] = analyzed_variants
    final_data["security_summary"] = {
        "total_variants": total_variants,
        "security_relevant_count": security_flag_count,
        "is_security_series": is_series_secure
    }

    target_path = path_relevant if is_series_secure else path_irrelevant
    folder_name = "security_relevant" if is_series_secure else "security_irrelevant"
    
    save_json(final_data, target_path)
    log(sid, None, f"Finished. Saved to [{folder_name}]", "DONE")


def main():
    parser = argparse.ArgumentParser(description="Distributed Security Filter")
    parser.add_argument("--series_dir", required=True)
    parser.add_argument("--events_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--ollama_model", default="gamma3:12b")
    parser.add_argument("--force", action="store_true")
    # 新增 Range 参数
    parser.add_argument("--start_index", type=int, default=0, help="Start index of tasks to process")
    parser.add_argument("--end_index", type=int, default=-1, help="End index of tasks to process (-1 for all)")
    
    args = parser.parse_args()

    if not check_ollama_health(args.ollama_model): sys.exit(1)

    output_dir = os.path.abspath(args.output_dir)
    ensure_dir(output_dir)
    
    dir_relevant = os.path.join(output_dir, "security_relevant")
    dir_irrelevant = os.path.join(output_dir, "security_irrelevant")
    ensure_dir(dir_relevant)
    ensure_dir(dir_irrelevant)

    cache_file = os.path.join(output_dir, "llm_analysis_cache.json")
    global_cache = OllamaCache(cache_file)
    
    full_log_file = os.path.join(output_dir, "full_analysis.log")

    # 1. 扫描所有任务
    all_tasks = []
    print(f"\n[MAIN] Scanning tasks in {args.series_dir}...")
    for root, _, files in os.walk(args.series_dir):
        for f in files:
            if f.startswith("series_") and f.endswith(".json"):
                all_tasks.append(os.path.join(root, f))
    
    # 2. 关键：强制排序，确保多机顺序一致
    all_tasks.sort()
    total_found = len(all_tasks)
    
    # 3. 计算切片
    start = args.start_index
    end = args.end_index
    if end == -1 or end > total_found:
        end = total_found
    
    # 4. 获取当前机器的任务子集
    my_tasks = all_tasks[start:end]
    
    print(f"[MAIN] Total Tasks Found: {total_found}")
    print(f"[MAIN] Machine Range:     {start} to {end}")
    print(f"[MAIN] My Task Count:     {len(my_tasks)}")
    print(f"[MAIN] Starting distributed processing...\n")

    if len(my_tasks) == 0:
        print("[WARN] No tasks in this range. Exiting.")
        return

    for i, task_file in enumerate(my_tasks):
        # 计算全局进度
        global_idx = start + i + 1
        print(f"--- Global Progress [{global_idx}/{total_found}] | Local [{i+1}/{len(my_tasks)}] ---")
        try:
            process_single_series(
                task_file,
                args.events_dir,
                dir_relevant,
                dir_irrelevant,
                full_log_file,
                args.ollama_model,
                args.force,
                global_cache
            )
        except KeyboardInterrupt:
            print("\n[STOP] Interrupted by user.")
            global_cache.save()
            sys.exit(0)
        except Exception as e:
            print(f"[CRASH] Failed processing {task_file}: {e}")
            traceback.print_exc()
        
        if i % 10 == 0: global_cache.save()
        print("")

    global_cache.save()
    print(f"[DONE] Tasks {start} to {end} completed.")

if __name__ == "__main__":
    main()
