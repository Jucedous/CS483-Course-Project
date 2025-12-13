"""
Concurrent resumable LLM-based classifier for security-related LKML events.
Adds:
    "security_related": true/false
    "security_reason": "<LLM Explanation>"   <-- NEW
"""

import os
import json
import subprocess
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# ========== STEP 1. LLM 调用 ==========

def query_local_llm(prompt, model="gemma:4B", debug=False):
    """调用本地 LLM（Ollama），带重试和超时保护"""
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


# NEW —— 提取 reasoning
def extract_reason(text):
    """
    尝试从 LLM 输出中提取一句英文理由。
    如果 LLM 只输出 true/false，则自动给默认理由。
    """
    lower = text.lower().strip()

    # 只输出 true 或 false
    if lower in ("true", "false"):
        return "Reason not provided by model."

    # 常见格式： "true - because ..." 或 "true: This patch ..."
    if lower.startswith("true"):
        parts = text.split(" ", 1)
        if len(parts) == 2:
            return parts[1].strip()
        return "Security-relevant (no detailed reason)."

    # 其它文本格式，直接返回完整文本
    return text.strip()


def analyze_security_with_llm(content, model="llama3", debug=False):
    """分析 patch 是否安全相关 + 提取理由"""

    if not content:
        raise ValueError("Empty content provided to LLM.")

    prompt = (
        "Analyze the following Linux kernel patch. "
        "First output either 'true' or 'false' on the first line to indicate whether the patch is related to system security. "
        "On the second line, provide a short English explanation (one sentence).\n\n"
        f"{content[:10000]}"
    )

    result = query_local_llm(prompt, model, debug)

    # 拆分两行输出
    lines = [l.strip() for l in result.splitlines() if l.strip()]
    verdict = lines[0].lower()

    if "true" in verdict:
        reason = extract_reason(result)
        return True, reason

    if "false" in verdict:
        return False, None

    raise ValueError(f"Unexpected LLM response (not true/false): {result}")


# ========== STEP 2. 文件处理函数 ==========

def process_file(file_path, output_dir, done_files, progress_path, model="llama3", debug=False):
    filename = os.path.basename(file_path)
    if filename in done_files:
        return f"[SKIP] {filename} (already done)"

    output_path = os.path.join(output_dir, filename)
    tmp_path = output_path + ".tmp"

    # 如果已经存在结果文件，则跳过
    if os.path.exists(output_path):
        done_files.add(filename)
        save_progress(progress_path, done_files)
        return f"[SKIP] {filename} (output exists)"

    try:
        with open(file_path) as f:
            data = json.load(f)

        # 获取 LLM 输入内容
        content = (
            data.get("content")
            or data.get("body")
            or data.get("diff")
            or json.dumps(data)
        )

        # LLM 判断
        is_security, reason = analyze_security_with_llm(content, model, debug)

        print(f"[RESULT] {filename}: {is_security}")

        # false → 不保存文件，但标记完成
        if not is_security:
            done_files.add(filename)
            save_progress(progress_path, done_files)
            return f"[NON-SECURITY] {filename}"

        # true → 写入文件
        data["security_related"] = True
        data["security_reason"] = reason or "Security-relevant change."

        # 写临时文件
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
        os.rename(tmp_path, output_path)

        done_files.add(filename)
        save_progress(progress_path, done_files)

        return f"[DONE] {filename} (security)"

    except Exception as e:
        print(f"[ERROR] {filename}: LLM query failed: {e}")
        return f"[ERROR] {filename}: {e}"


# ========== STEP 3. 进度文件 ==========

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


# ========== STEP 4. 主程序入口 ==========

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