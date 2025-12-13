"""
Concurrent resumable LLM-based classifier for security-related LKML events.
Adds `"security_related": true/false` to each JSON and saves results into a new folder.
"""

import os
import json
import subprocess
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# ========== STEP 1. LLM 调用 ==========
def query_local_llm(prompt, model="llama3", debug=False):
    """调用本地 LLM（如 Ollama），带重试和超时保护"""
    for attempt in range(3):
        try:
            result = subprocess.run(
                ["ollama", "run", model, prompt],
                capture_output=True, text=True, timeout=180
            )
            output = result.stdout.strip()
            error_output = result.stderr.strip()

            if error_output:
                print(f"[LLM STDERR] {error_output}")

            if not output:
                raise RuntimeError("LLM returned empty output.")

            if debug:
                print(f"[LLM OUTPUT] {output}")

            return output
        except subprocess.TimeoutExpired:
            print(f"[TIMEOUT] LLM query exceeded 180s (attempt {attempt+1}/3)")
        except Exception as e:
            print(f"[RETRY {attempt+1}/3] LLM query failed: {e}")
        time.sleep(3)
    raise RuntimeError("LLM query failed after 3 retries.")

def analyze_security_with_llm(content, model="llama3", debug=False):
    """分析 patch 是否安全相关"""
    if not content:
        raise ValueError("Empty content provided to LLM.")
    prompt = (
        f"Is the following Linux kernel patch related to system security? "
        f"Answer only true or false.\n\n{content[:10000]}"
    )
    result = query_local_llm(prompt, model, debug)

    result_lower = result.lower().strip()
    if "true" in result_lower:
        return True
    elif "false" in result_lower:
        return False
    else:
        raise ValueError(f"Unexpected LLM response (not true/false): {result}")

# ========== STEP 2. 文件处理函数 ==========

def process_file(file_path, output_dir, done_files, progress_path, model="llama3", debug=False):
    filename = os.path.basename(file_path)
    if filename in done_files:
        return f"[SKIP] {filename} (already done)"

    output_path = os.path.join(output_dir, filename)
    tmp_path = output_path + ".tmp"

    # 如果已存在结果文件，直接跳过并写入进度
    if os.path.exists(output_path):
        done_files.add(filename)
        save_progress(progress_path, done_files)
        return f"[SKIP] {filename} (output exists)"

    try:
        with open(file_path) as f:
            data = json.load(f)

        # 获取内容
        content = (
            data.get("content") or
            data.get("body") or
            data.get("diff") or
            json.dumps(data)
        )

        # 调用 LLM
        result = analyze_security_with_llm(content, model, debug)

        # 输出结果状态
        print(f"[RESULT] {filename}: {result}")

        # 如果不是安全相关，跳过输出但仍记录为已完成
        if not result:
            done_files.add(filename)
            save_progress(progress_path, done_files)
            return f"[NON-SECURITY] {filename}"

        # 仅写入 security_related == true 的文件
        data["security_related"] = True

        # 写入结果（临时文件方式）
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
        os.rename(tmp_path, output_path)

        # 更新进度文件
        done_files.add(filename)
        save_progress(progress_path, done_files)

        return f"[DONE] {filename} (security)"

    except Exception as e:
        print(f"[ERROR] {filename}: LLM query failed: {e}")
        return f"[ERROR] {filename}: {e}"

# ========== STEP 3. 进度保存与加载 ==========

def load_progress(progress_path):
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            try:
                return set(json.load(f))
            except Exception:
                return set()
    return set()

def save_progress(progress_path, done_files):
    with open(progress_path, "w") as f:
        json.dump(list(done_files), f, indent=2)

# ========== STEP 4. 主逻辑 ==========

def main():
    parser = argparse.ArgumentParser(description="Concurrent resumable LLM-based security classification for LKML JSON events.")
    parser.add_argument("--input", type=str, required=True, help="Input directory containing LKML event JSON files.")
    parser.add_argument("--output", type=str, required=True, help="Output directory for processed JSON files.")
    parser.add_argument("--concurrency", type=int, default=10, help="Number of concurrent LLM workers.")
    parser.add_argument("--model", type=str, default="llama3", help="Local LLM model name.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    progress_path = os.path.join(args.output, "progress.json")

    # 加载已完成文件记录
    done_files = load_progress(progress_path)

    files = list(Path(args.input).glob("*.json"))
    print(f"[INFO] Total files: {len(files)}, Completed: {len(done_files)}")

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(process_file, str(file), args.output, done_files, progress_path, args.model, args.debug): file
            for file in files
        }

        for future in as_completed(futures):
            print(future.result())

if __name__ == "__main__":
    main()
