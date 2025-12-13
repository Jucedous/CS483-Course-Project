"""
Security relevance tagging for LKML event series (simplified).

This script reads LKML event_series JSON files, extracts their subject line,
and uses the GPT-5-Nano model to determine if the patch is related to security issues.

Usage:
    python security_filter_series.py --input ./event_series --output_true ./security_related --output_false ./non_security

Arguments:
    --input          Path to the folder containing aggregated event_series JSON files.
    --output_true    Folder for JSON files identified as security-related.
    --output_false   Folder for JSON files identified as non-security-related.

The output JSON will include one additional field:
    "security_analysis": {"security_related": true or false}

Example:
    {
        "subject": "/proc/kcore: Fix SMAP violation when dumping vsyscall user page",
        "security_analysis": {"security_related": true}
    }
"""

import os
import json
import argparse
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

STATS = {"total_files": 0, "security_related": 0}
LOCK = threading.Lock()

# Configuration constants
CONCURRENCY = 50          # Number of concurrent threads
MAX_RETRY = 3             # Maximum retry attempts per API call
RETRY_INTERVAL = 3        # Seconds between retries

OPENAI_API_KEY = "sk-wZ3Zpfep5N2wa2KXDcBbB29075554b0988196b645cE44615"
OPENAI_BASE_URL = "https://api.bianxie.ai/v1"
OPENAI_MODEL = "gpt-5-nano-2025-08-07"

client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

PROMPT_TEMPLATE = """
You are an expert in Linux kernel and software security.
Determine whether the following LKML patch subject is related to security issues such as vulnerabilities, privilege escalation, memory corruption, race conditions, kernel hardening, or mitigations.
Respond strictly in JSON with a single key:
{"security_related": true} or {"security_related": false}

Subject: {subject}
"""

def call_openai(subject, debug):
    for _ in range(MAX_RETRY):
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a Linux kernel security expert."},
                    {"role": "user", "content": PROMPT_TEMPLATE.format(subject=subject)},
                ],
                temperature=0,
                max_tokens=50,
            )
            content = response.choices[0].message.content.strip()
            if debug:
                print(f"\n[DEBUG] Subject: {subject}\n[DEBUG] Raw output:\n{content}\n")

            try:
                data = json.loads(content)
                return data
            except Exception as e:
                if debug:
                    print(f"[DEBUG] JSON parse error for subject '{subject}': {e}")
                text = content.lower()
                if "true" in text:
                    return {"security_related": True}
                elif "false" in text:
                    return {"security_related": False}
                else:
                    return {"security_related": False}
        except Exception as e:
            if debug:
                print(f"[DEBUG] API error for subject '{subject}': {e}")
            time.sleep(RETRY_INTERVAL)
    return {"security_related": False}

def process_one(input_path, output_true_dir, output_false_dir, debug):
    with open(input_path, "r") as f:
        data = json.load(f)
    subject = data.get("subject", "")
    result = call_openai(subject, debug)
    is_security = result.get("security_related", False)
    data["security_analysis"] = {"security_related": is_security}

    output_dir = output_true_dir if is_security else output_false_dir
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(input_path))
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    with LOCK:
        STATS["total_files"] += 1
        if is_security:
            STATS["security_related"] += 1
    print(f"{os.path.basename(input_path)} => {is_security}")

def main():
    parser = argparse.ArgumentParser(description="Detect security-related LKML event series.")
    parser.add_argument("--input", required=True, help="Input folder with event_series JSONs")
    parser.add_argument("--output_true", required=True, help="Output folder for security-related JSONs")
    parser.add_argument("--output_false", required=True, help="Output folder for non-security JSONs")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode to print model outputs")
    args = parser.parse_args()

    files = [f for f in os.listdir(args.input) if f.endswith(".json")]
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futures = [
            ex.submit(
                process_one,
                os.path.join(args.input, f),
                args.output_true,
                args.output_false,
                args.debug,
            )
            for f in files
        ]
        for f in tqdm(as_completed(futures), total=len(futures)):
            f.result()

    print("\nSummary:")
    print(f"Total processed: {STATS['total_files']}")
    print(f"Security related: {STATS['security_related']}")

if __name__ == "__main__":
    main()