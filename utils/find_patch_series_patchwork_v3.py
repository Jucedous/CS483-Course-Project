#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
find_patch_series_patchwork_v3
--------------------------------------------------------
使用示例：

python find_patch_series_patchwork_v3 \
    --series_dir ./data/20250101-20250601/1_series_json_data \
    --events_dir ./data/20250101-20250601/0_event_json_data \
    --output_dir ./data/20250101-20250601/2_series_archived \
    --ollama_model gamma3:12b \
    --token <YOUR_PATCHWORK_TOKEN> \
    --concurrency 2
    
改进点 (v20 - Model Health Check Fix):
    1. [严格模型检查] 启动前不仅检查 Ollama 服务，还会检查指定的 Model 是否存在。
       - 如果不存在，自动尝试 `ollama pull`。
       - 如果 pull 失败（例如模型名错误），直接退出程序。
       这解决了 "pull model manifest: file does not exist" 在运行时反复出现的问题。
    2. [日志优化] 优化了多行错误日志的显示格式。

改进点 (v21 - Cache & Archive Separation):
    1. [逻辑分离] 
       - 搜索到的所有候选数据先下载到全局 `cache/` 目录（避免污染结果目录）。
       - 只有当 LLM 判定为 **MATCH** 时，才将对应的文件/文件夹从 `cache/` **复制** 到当前 Series 的 `patchwork_remote/` 目录。
    2. [路径管理] 
       - 报告中的文件路径在匹配时指向 `patchwork_remote/` (归档副本)。
       - 不匹配时指向 `cache/` (缓存源)，方便排查但保持结果目录整洁。
    3. [继承 v20] 保留了模型健康检查、Token 验证、详细日志等所有健壮性功能。
    
改进点 (v23 - Strict Archive Logic & Path Consistency):
    1. [归档原则] 严格执行 "Download to Cache -> Compare -> Copy if Match" 流程。
       - 不匹配的数据只留在 ../cache/ 中，绝不污染 series 文件夹。
       - 只有匹配的数据才会被复制到 ./patchwork_remote/ 中。
    2. [路径一致性] 报告中的所有文件路径统一为相对于 analysis_report.json 的相对路径。
       - 匹配： "patchwork_remote/series_123/..."
       - 不匹配： "../cache/series_123/..." (明确指向外部缓存)
    3. [日志明确] 控制台增加 [CACHE] 和 [COPY] 状态日志，行为一目了然。
    
改进点 (v26 - Fix Missing Function):
    1. [紧急修复] 恢复了 v25 中误删的 `copy_from_cache` 函数。
       该函数用于将匹配成功的补丁从全局缓存复制到 Series 归档目录，缺失会导致运行报错。
    2. [完整性] 包含 v25 的所有调试日志改进（跳过原因可见、API 错误显形）。
    
改进点 (v28 - Log Immediate Visibility):
    1. [即时日志] 任务一开始就创建 llm_judgment.log 并写入 Header，
       并在下载阶段写入状态日志，消除“卡住”的错觉。
    2. [日志增强] 即使是不匹配 (Mismatch) 的情况，也会在控制台打印简短原因，
       不再隐藏在 DEBUG 级别下。
    3. [状态明确] 明确记录 "Downloading..." 和 "Comparing..." 到日志文件中。
    
Updates (v31 - English Logging):
    1. [English Output] Translated all console logs and file logs from Chinese to English.
    2. [Inherited Features] Keeps strictly single-threaded execution, global cache isolation, 
       download-first strategy, and detailed step-by-step logging from v30.
Updates (v36 - Iterate ALL Versions):
    1. [Fix Logic] removed the `if matched: break` statement completely.
       Now the script will check every single candidate found in Patchwork.
       If search returns v2, v3, and v4, it will try to match and download ALL of them.
    2. [Inherited Features] Includes all previous fixes (logs, single-thread, strict caching, force retry).
"""

import os
import re
import json
import shutil
import hashlib
import argparse
import requests
import subprocess
import time
import datetime
import sys
import traceback
from tqdm import tqdm

PATCHWORK_API = "https://patchwork.kernel.org/api/"

# ---------- Log Utils ----------

def get_timestamp():
    return datetime.datetime.now().strftime("%H:%M:%S")

def log(sid, variant, msg, level="INFO"):
    """Console log printer"""
    timestamp = get_timestamp()
    variant_tag = f"[{variant}]" if variant else "[System]"
    print(f"[{timestamp}] [{level}] [{sid}] {variant_tag} {msg}")

def write_judgment_log(log_path, patch_id, result_status, reason, url):
    """Write detailed judgment log to file"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    content = (
        f"[{timestamp}] Patch ID: {patch_id}\n"
        f"URL: {url}\n"
        f"Result: {result_status}\n"
        f"Reason: {reason}\n"
        f"{'-'*60}\n"
    )
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        print(f"[ERR] Failed to write judgment log: {e}")

# ---------- Health Checks ----------

def check_ollama_health(model_name):
    print("-" * 50)
    print(f"[INIT] Checking Ollama Environment...")
    if shutil.which("ollama") is None:
        print(f"[FATAL] 'ollama' command not found. Please install Ollama first.")
        return False
    try:
        # Check service
        subprocess.run(["ollama", "list"], capture_output=True, check=True)
        
        # Check model
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

def check_patchwork_token(token):
    if not token:
        print("[INIT] No Token provided. API limits will apply.")
        return True
    print(f"[INIT] Verifying Token...")
    try:
        r = requests.get(f"{PATCHWORK_API}projects/", headers={"Authorization": f"Token {token}"}, timeout=10)
        if r.status_code in [401, 403]:
            print(f"[FATAL] Invalid Token (HTTP {r.status_code}).")
            return False
    except Exception:
        print(f"[WARN] Network error, cannot verify token.")
    return True

# ---------- Basic Utils ----------

def clean_title(title: str) -> str:
    if not title: return ""
    # Remove Re: Fwd: etc.
    title = re.sub(r'^\s*(?:Re|Fwd|Aw|Sv|Resend):\s*', '', title, flags=re.IGNORECASE)
    # Remove [PATCH v2] etc.
    title = re.sub(r"^\s*(\[[^\]]*\]\s*)+", "", title)
    return " ".join(title.split())

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

# ---------- Download Utils (Strict Separation) ----------

def make_request(url, params=None, token=None):
    headers = {"Accept": "application/json"}
    if token: headers["Authorization"] = f"Token {token}"
    try:
        r = requests.get(url, params=params, headers=headers, timeout=30)
        if r.status_code == 429:
            time.sleep(5)
            return make_request(url, params, token)
        if r.status_code == 404: return None
        r.raise_for_status()
        return r.json()
    except Exception:
        # print(f"[WARN] API Request failed {url}: {e}") 
        return None

def download_file_stream(url, save_path, token=None):
    if os.path.exists(save_path) and os.path.getsize(save_path) > 0: return True
    headers = {}
    if token: headers["Authorization"] = f"Token {token}"
    try:
        with requests.get(url, headers=headers, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(save_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
        return True
    except Exception: return False

def download_entire_series_to_cache(series_id, cache_root, token=None):
    """Download entire series to global cache"""
    if not series_id: return None
    
    series_dir = os.path.join(cache_root, f"series_{series_id}")
    
    # Cache hit
    if os.path.exists(os.path.join(series_dir, "series.json")):
        return series_dir

    ensure_dir(series_dir)
    ensure_dir(os.path.join(series_dir, "patches_meta"))
    ensure_dir(os.path.join(series_dir, "diffs"))
    ensure_dir(os.path.join(series_dir, "mboxes"))

    # 1. Series Info
    series_url = f"{PATCHWORK_API}series/{series_id}/"
    series_data = make_request(series_url, None, token)
    if not series_data: return None

    with open(os.path.join(series_dir, "series.json"), "w", encoding="utf-8") as f:
        json.dump(series_data, f, indent=2, ensure_ascii=False)

    # 2. Patches
    patches = series_data.get("patches", [])
    for p_summary in patches:
        pid = p_summary["id"]
        p_detail = make_request(f"{PATCHWORK_API}patches/{pid}/", None, token)
        if not p_detail: continue

        # Meta
        with open(os.path.join(series_dir, "patches_meta", f"{pid}.json"), "w", encoding="utf-8") as f:
            json.dump(p_detail, f, indent=2, ensure_ascii=False)
        # Diff
        if p_detail.get("diff"):
            with open(os.path.join(series_dir, "diffs", f"{pid}.diff"), "w", encoding="utf-8") as f:
                f.write(p_detail["diff"])
        # Mbox
        if p_detail.get("mbox"):
            download_file_stream(p_detail["mbox"], os.path.join(series_dir, "mboxes", f"{pid}.mbox"), token)
    
    return series_dir

def download_isolated_patch_to_cache(patch_data, cache_root, token=None):
    """Download isolated patch to global cache"""
    pid = patch_data['id']
    isolated_dir = os.path.join(cache_root, "isolated")
    ensure_dir(isolated_dir)
    
    diff_path = os.path.join(isolated_dir, f"patch_{pid}.diff")
    mbox_path = os.path.join(isolated_dir, f"patch_{pid}.mbox")
    
    # Diff
    diff_content = patch_data.get("diff")
    if diff_content:
        with open(diff_path, "w", encoding="utf-8") as f: f.write(diff_content)
    
    # Mbox
    mbox_url = patch_data.get("mbox")
    if mbox_url:
        download_file_stream(mbox_url, mbox_path, token)
        
    return diff_path, mbox_path

# ---------- LLM Core ----------

class OllamaCache:
    def __init__(self, path):
        self.path = path
        self.data = {}
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f: self.data = json.load(f)
            except: pass
    def get(self, lhash, rhash): return self.data.get(f"{lhash}::{rhash}")
    def set(self, lhash, rhash, match, reason): self.data[f"{lhash}::{rhash}"] = {"match": match, "reason": reason}
    def save(self):
        with open(self.path, "w", encoding="utf-8") as f: json.dump(self.data, f, indent=2, ensure_ascii=False)

def ollama_compare(model, lt, lc, rt, rc, cache, max_retries=2):
    lhash = short_hash(lt + lc)
    rhash = short_hash(rt + rc)
    
    cached = cache.get(lhash, rhash)
    if cached: return cached["match"], cached["reason"]

    prompt = f"""
You are a Linux kernel patch comparison expert.
Determine if the following two patches represent different versions of the same change (i.e., v1 vs v2).
Ignore minor differences in context lines, timestamps, or comments. Focus on the code logic and file paths.

PATCH A (Local Variant):
Title: {lt}
Content:
{lc}

PATCH B (Remote Candidate):
Title: {rt}
Content:
{rc}

Instructions:
1. Analyze the semantic similarity of the code changes.
2. Answer 'yes' or 'no' in the first line.
3. Provide a brief 1-sentence reason in the second line.
"""
    for attempt in range(max_retries + 1):
        try:
            res = subprocess.run(["ollama", "run", model], input=prompt.encode("utf-8"), capture_output=True, timeout=180)
            if res.returncode != 0: raise Exception(res.stderr.decode())
            
            out = res.stdout.decode().strip()
            match = "yes" in out.lower().split("\n")[0]
            reason = out.split("\n", 1)[-1].strip() if "\n" in out else out
            
            cache.set(lhash, rhash, match, reason)
            return match, reason
        except subprocess.TimeoutExpired:
            if attempt < max_retries: time.sleep(2)
        except Exception:
            if attempt < max_retries: time.sleep(2)
            
    return "TIMEOUT", "LLM response timed out."

# ---------- Archive Utils ----------

def copy_from_cache(src, dest_dir, is_dir=False):
    if not src or not os.path.exists(src): return None
    basename = os.path.basename(src)
    dest = os.path.join(dest_dir, basename)
    try:
        if is_dir:
            if os.path.exists(dest): shutil.rmtree(dest)
            shutil.copytree(src, dest)
        else:
            shutil.copy2(src, dest)
        return dest
    except: return None

# ---------- API Helpers ----------

def safe_extract_results(api_response):
    """[v33 Fix] Handle both list and dict responses from Patchwork API"""
    if not api_response:
        return []
    if isinstance(api_response, list):
        return api_response
    if isinstance(api_response, dict):
        return api_response.get("results", [])
    return []

def search_patchwork(title, token):
    return make_request(f"{PATCHWORK_API}patches/", {"q": title}, token)

def get_author_series(email, token):
    return make_request(f"{PATCHWORK_API}series/", {"submitter": email}, token)

def get_series_patches(sid, token):
    data = make_request(f"{PATCHWORK_API}series/{sid}/", None, token)
    if data and isinstance(data, dict):
        return data.get("patches", [])
    return []

def get_patch_detail(pid, token):
    return make_request(f"{PATCHWORK_API}patches/{pid}/", None, token)

# ---------- Single Task Flow ----------

def process_single_task(series_path, events_dir, output_root, cache_root, model, token, force, max_retries):
    sdata = load_json_safe(series_path)
    if not sdata: return

    sid = sdata.get("series_id", os.path.splitext(os.path.basename(series_path))[0])
    folder_name = sid if sid.startswith("series_") else f"series_{sid}"
    
    # Directories
    series_out_dir = os.path.join(output_root, folder_name)
    events_out_dir = os.path.join(series_out_dir, "events")
    remote_out_dir = os.path.join(series_out_dir, "patchwork_remote")
    
    judgment_log_file = os.path.join(series_out_dir, "llm_judgment.log") # [v34] fixed log name
    report_file = os.path.join(series_out_dir, "analysis_report.json")

    # Check Skip
    if os.path.exists(report_file) and not force:
        log(folder_name, None, "Report exists, skipping.", "SKIP")
        return

    ensure_dir(series_out_dir)
    ensure_dir(events_out_dir)
    # remote_out_dir created only on match

    # Archive Series.json
    shutil.copy2(series_path, os.path.join(series_out_dir, "series.json"))
    
    # Init Log
    with open(judgment_log_file, "w", encoding="utf-8") as f:
        f.write(f"=== LLM Judgment Log for {sid} ===\n\n")

    variants = sdata.get("variants", {})
    results_data = []
    
    # Cache for LLM (local to this run)
    llm_cache_file = os.path.join(output_root, "llm_cache.json")
    llm_cache = OllamaCache(llm_cache_file)

    for ver, info in variants.items():
        # 1. Archive Event
        src_file = info.get("source_file")
        src_full = os.path.join(events_dir, src_file)
        
        if not os.path.exists(src_full):
            for root, _, files in os.walk(events_dir):
                if src_file in files:
                    src_full = os.path.join(root, src_file)
                    break
        
        if not os.path.exists(src_full):
            log(folder_name, ver, f"Source file missing: {src_file}", "ERR")
            continue
            
        dest_full = os.path.join(events_out_dir, src_file)
        shutil.copy2(src_full, dest_full)
        
        local_event = load_json_safe(dest_full)
        if not local_event: continue
        
        msg = local_event["messages"][0]
        raw_title = msg.get("subject", "") or msg.get("title", "")
        clean_t = clean_title(raw_title)
        content = msg.get("content", "")
        
        if not content:
            log(folder_name, ver, "Email content empty, skipping.", "WARN")
            continue

        log(folder_name, ver, f"Searching Patchwork: '{clean_t}'...", "SEARCH")
        
        # 2. Search
        pw_res = search_patchwork(clean_t, token)
        pw_list = safe_extract_results(pw_res)
        
        # Fallback
        if not pw_list and ":" in clean_t:
            fallback = clean_t.split(":", 1)[1].strip()
            if len(fallback) > 10:
                log(folder_name, ver, f"Exact search failed, trying fallback: '{fallback}'", "SEARCH")
                pw_res = search_patchwork(fallback, token)
                pw_list = safe_extract_results(pw_res)

        log(folder_name, ver, f"Found {len(pw_list)} related patches.", "INFO")
        
        if not pw_list:
            write_judgment_log(judgment_log_file, "N/A", "NO_SEARCH_RESULT", f"Title: {clean_t}", "N/A")
            results_data.append({"variant": ver, "match": False, "status": "NO_MATCH"})
            continue

        # 3. Build Candidates
        candidates = [] # (source_type, patch_summary, series_id)
        seen_sids = set()
        
        # Strategy A: Direct Series
        for item in pw_list[:3]:
            series_list = item.get("series", [])
            if series_list:
                for s in series_list:
                    sid_remote = s.get("id")
                    if sid_remote and sid_remote not in seen_sids:
                        seen_sids.add(sid_remote)
                        s_patches = get_series_patches(sid_remote, token)
                        log(folder_name, ver, f"Found associated Series #{sid_remote} ({len(s_patches)} patches)", "INFO")
                        for sp in s_patches:
                            candidates.append(("direct_series", sp, sid_remote))
            else:
                # Isolated
                candidates.append(("search_result", item, None))

        # Strategy B: Author Search
        if len(candidates) < 5:
            author_email = pw_list[0].get("submitter", {}).get("email")
            if author_email:
                log(folder_name, ver, f"Searching Author: {author_email}", "INFO")
                auth_series = get_author_series(author_email, token)
                as_list = safe_extract_results(auth_series)
                log(folder_name, ver, f"Author has {len(as_list)} Series.", "INFO")
                
                for aser in as_list[:5]: # Limit to 5
                    if aser["id"] not in seen_sids:
                        s_patches = get_series_patches(aser["id"], token)
                        for sp in s_patches:
                            candidates.append(("author_series", sp, aser["id"]))

        log(folder_name, ver, f"Total {len(candidates)} candidates to check.", "INFO")

        # 4. Loop & Compare (Check ALL candidates - v36 update)
        
        any_match_found = False # To track if ANY match was found for this variant
        copied_series_to_local = set() # Deduplicate local series copy operations

        for idx, (src_type, p_sum, s_id_remote) in enumerate(candidates):
            # NO BREAK HERE! We iterate through all.
            
            pid = p_sum["id"]
            p_detail = get_patch_detail(pid, token)
            if not p_detail: continue
            
            rt = p_detail.get("name", "")
            rc = p_detail.get("diff", "") or p_detail.get("content", "")
            
            if not rc:
                log(folder_name, ver, f"Skipping candidate {pid}: No Diff content.", "WARN")
                continue

            # A. Download to Global Cache
            cache_d = None; cache_m = None; cache_s = None
            
            if s_id_remote:
                log(folder_name, ver, f"[{idx+1}/{len(candidates)}] Caching Series {s_id_remote}...", "DOWNLOAD")
                cache_s = download_entire_series_to_cache(s_id_remote, cache_root, token)
                if cache_s:
                    cache_d = os.path.join(cache_s, "diffs", f"{pid}.diff")
                    cache_m = os.path.join(cache_s, "mboxes", f"{pid}.mbox")
            else:
                log(folder_name, ver, f"[{idx+1}/{len(candidates)}] Caching isolated patch {pid}...", "DOWNLOAD")
                cache_d, cache_m = download_isolated_patch_to_cache(p_detail, cache_root, token)

            # B. LLM Compare
            log(folder_name, ver, f"Comparing Candidate ID:{pid}...", "LLM")
            is_match, reason = ollama_compare(model, clean_t, content, rt, rc, llm_cache, max_retries)
            
            res_str = "MATCH" if is_match is True else ("TIMEOUT" if is_match == "TIMEOUT" else "MISMATCH")
            log(folder_name, ver, f"Result: {res_str}", "RESULT")
            if is_match == "TIMEOUT":
                log(folder_name, ver, "WARN: LLM timed out. Check model status.", "WARN")

            write_judgment_log(judgment_log_file, pid, res_str, reason, p_detail.get("web_url"))

            # C. Archive (Copy if Match)
            final_files = {"diff": None, "mbox": None, "full_series": None}
            
            if is_match is True:
                ensure_dir(remote_out_dir)
                log(folder_name, ver, f"MATCH CONFIRMED! Archiving files to {remote_out_dir}...", "COPY")
                
                any_match_found = True

                if s_id_remote and cache_s:
                    # Copy Folder
                    # Deduplication logic: if we already copied this series for this variant task, skip re-copying IO
                    # But we still need to populate final_files paths
                    local_s_basename = os.path.basename(cache_s)
                    local_s_dest = os.path.join(remote_out_dir, local_s_basename)
                    
                    if s_id_remote not in copied_series_to_local:
                        copy_from_cache(cache_s, remote_out_dir, is_dir=True)
                        copied_series_to_local.add(s_id_remote)
                    
                    if os.path.exists(local_s_dest):
                        final_files["full_series"] = os.path.relpath(local_s_dest, series_out_dir)
                        final_files["diff"] = os.path.join(final_files["full_series"], "diffs", f"{pid}.diff")
                        final_files["mbox"] = os.path.join(final_files["full_series"], "mboxes", f"{pid}.mbox")

                else:
                    # Copy Single Files
                    local_d = copy_from_cache(cache_d, remote_out_dir)
                    local_m = copy_from_cache(cache_m, remote_out_dir)
                    if local_d: final_files["diff"] = os.path.relpath(local_d, series_out_dir)
                    if local_m: final_files["mbox"] = os.path.relpath(local_m, series_out_dir)
                
            else:
                # Reference Cache
                if cache_d: final_files["diff"] = os.path.relpath(cache_d, series_out_dir)
                if cache_m: final_files["mbox"] = os.path.relpath(cache_m, series_out_dir)
                if cache_s: final_files["full_series"] = os.path.relpath(cache_s, series_out_dir)

            results_data.append({
                "variant": ver,
                "match": is_match,
                "reason": reason,
                "remote_info": {
                    "id": pid,
                    "series_id": s_id_remote,
                    "title": rt,
                    "url": p_detail.get("web_url"),
                    "files": final_files
                }
            })
        
        if not any_match_found:
             log(folder_name, ver, "Finished checking all candidates. No matches found.", "INFO")

    save_json(results_data, report_file)
    llm_cache.save()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--series_dir", required=True)
    parser.add_argument("--events_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--ollama_model", default="qwen2.5-coder")
    parser.add_argument("--token")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--retries", type=int, default=2, help="Number of retries for LLM timeouts")
    args = parser.parse_args()

    if not check_ollama_health(args.ollama_model): sys.exit(1)
    if not check_patchwork_token(args.token): sys.exit(1)

    output_dir = os.path.abspath(args.output_dir)
    ensure_dir(output_dir)
    cache_root = os.path.join(output_dir, "cache")
    ensure_dir(cache_root)

    tasks = []
    for root, _, files in os.walk(args.series_dir):
        for f in files:
            if f.startswith("series_") and f.endswith(".json"):
                tasks.append(os.path.join(root, f))
    
    print(f"\n[MAIN] Scanned {len(tasks)} Series tasks.")
    print(f"[MAIN] Global Cache Dir: {cache_root}")
    print(f"[MAIN] Starting sequential processing...\n")

    for i, task_file in enumerate(tasks):
        print(f"=== Progress [{i+1}/{len(tasks)}] ===")
        try:
            process_single_task(
                task_file,
                args.events_dir,
                output_dir,
                cache_root,
                args.ollama_model,
                args.token,
                args.force,
                args.retries
            )
        except Exception as e:
            print(f"[CRASH] Failed processing {task_file}: {e}")
            traceback.print_exc()
        print("\n")

    print("[DONE] All tasks completed.")

if __name__ == "__main__":
    main()
