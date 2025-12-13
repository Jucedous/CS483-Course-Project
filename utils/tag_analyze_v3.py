

import os, json, re, time, requests, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ========== 参数配置 ==========

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze LKML JSON files and generate event series with security filtering and resume support.")
    parser.add_argument("--input", type=str, required=True, help="Input directory containing LKML JSON files.")
    parser.add_argument("--output", type=str, required=True, help="Output directory for event series.")
    parser.add_argument("--concurrency", type=int, default=20, help="Number of concurrent threads.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for verbose output.")
    return parser.parse_args()

# ========== STEP 1: LLM 安全分析 ==========

def query_local_llm(prompt, model="llama3"):
    """本地 LLM 接口，可根据实际修改为 Ollama 或其他模型"""
    import subprocess
    try:
        cmd = ["ollama", "run", model, prompt]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result.stdout.strip()
    except Exception as e:
        print(f"[ERROR] LLM query failed: {e}")
        return "false"

def analyze_security_with_llm(content, model="llama3"):
    prompt = f"Is the following Linux kernel patch related to system security? Answer only true or false.\n\n{content[:5000]}"
    result = query_local_llm(prompt, model)
    return "true" in result.lower()

# ========== STEP 2: 工具函数 ==========

def extract_patch_version(subject):
    match = re.search(r'\[PATCH(?:\s+v(\d+))?', subject, re.IGNORECASE)
    return f"v{match.group(1)}" if match and match.group(1) else "v1"

def find_related_versions(subject, author):
    """从 patchwork 查找同主题不同版本（简化伪代码）"""
    query = subject.split(":")[0]
    url = f"https://patchwork.kernel.org/api/patches/?q={query}&submitter={author}"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            return resp.json().get("results", [])
    except Exception as e:
        print(f"[WARN] patchwork query failed: {e}")
    return []

def build_event_series(base_patch, related_patches):
    event_series = {
        "subject": base_patch["subject"],
        "topic_type": "PATCH",
        "variants": {},
        "connections": []
    }

    all_patches = [base_patch] + related_patches
    all_patches = sorted(all_patches, key=lambda x: extract_patch_version(x["subject"]))

    for p in all_patches:
        v = extract_patch_version(p["subject"])
        event_series["variants"][v] = {
            "event_id": p.get("event_id", ""),
            "url": p.get("url", ""),
            "source_file": p.get("source_file", "")
        }

    versions = list(event_series["variants"].keys())
    for i in range(len(versions) - 1):
        event_series["connections"].append({"from": versions[i], "to": versions[i+1]})
    return event_series

# ========== STEP 3: 断点续跑机制 ==========

def load_progress(path):
    if os.path.exists(path):
        with open(path) as f:
            return set(json.load(f))
    return set()

def save_progress(path, done_files):
    with open(path, "w") as f:
        json.dump(list(done_files), f, indent=2)

def load_cache(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

def save_cache(path, cache):
    with open(path, "w") as f:
        json.dump(cache, f, indent=2)

# ========== STEP 4: 主处理函数 ==========

def process_file(file_path, output_dir, done_files, cache, args, progress_path, cache_path):
    filename = os.path.basename(file_path)
    if filename in done_files:
        return f"[SKIP] {filename} already processed."

    output_path = os.path.join(output_dir, filename)
    tmp_path = output_path + ".tmp"

    if os.path.exists(output_path):
        done_files.add(filename)
        return f"[SKIP] {filename} output exists."

    with open(file_path) as f:
        data = json.load(f)

    content = data.get("content", "")
    subject = data.get("subject", "")
    author = data.get("from", "")

    # 安全缓存检查
    if filename in cache:
        security_related = cache[filename]
    else:
        security_related = analyze_security_with_llm(content)
        cache[filename] = security_related
        save_cache(cache_path, cache)

    if not security_related:
        done_files.add(filename)
        save_progress(progress_path, done_files)
        return f"[NON-SECURITY] {filename}"

    # 搜索相关版本
    related = find_related_versions(subject, author)
    event_series = build_event_series(data, related)

    # 安全写入
    with open(tmp_path, "w") as f:
        json.dump(event_series, f, indent=2)
    os.rename(tmp_path, output_path)

    done_files.add(filename)
    save_progress(progress_path, done_files)

    if args.debug:
        print(f"[DEBUG] Finished {filename}")

    return f"[DONE] {filename}"

# ========== STEP 5: 主程序入口 ==========

def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    progress_path = os.path.join(args.output, "progress.json")
    cache_path = os.path.join(args.output, "security_cache.json")

    done_files = load_progress(progress_path)
    cache = load_cache(cache_path)

    files = [str(p) for p in Path(args.input).glob("*.json")]
    print(f"[INFO] Total files: {len(files)}, Already done: {len(done_files)}")

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {executor.submit(process_file, f, args.output, done_files, cache, args, progress_path, cache_path): f for f in files}
        for future in as_completed(futures):
            print(future.result())

if __name__ == "__main__":
    main()