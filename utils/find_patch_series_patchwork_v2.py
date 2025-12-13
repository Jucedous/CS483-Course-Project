#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
find_patch_series_author_batch_v3.py
--------------------------------------------------------
功能说明：
    从 match_patch_series.py 的 output_dir 中读取每个事件系列 (event_series[...] 文件)，
    对其中的每个 patch variant，使用 Patchwork + Ollama 对比判断是否为历史版本。
    结果输出到指定 output_dir 下每个 series 文件夹中。

使用示例：
    python find_patch_series_patchwork_v2.py \
        --series_dir ./match_patch_series_output \
        --output_dir ./patchwork_compare_output \
        --ollama_model qwen2 \
        --token <YOUR_PATCHWORK_TOKEN> \
        --skip_download \
        --debug
"""

import os
import re
import json
import hashlib
import argparse
import requests
import subprocess
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

PATCHWORK_API = "https://patchwork.kernel.org/api/"


# ---------- 基础工具 ----------

def clean_title(title: str) -> str:
    return re.sub(r'\[.*?\]\s*', '', title).strip()


def load_json_safe(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_json(data, path):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[WARN] Failed to save {path}: {e}")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:16]


# ---------- Ollama 缓存 ----------

class OllamaCache:
    def __init__(self, path):
        self.path = path
        self.data = {}
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception:
                print(f"[WARN] Failed to load cache: {path}")

    def get_key(self, lhash, rhash):
        return f"{lhash}::{rhash}"

    def get(self, lhash, rhash):
        return self.data.get(self.get_key(lhash, rhash))

    def set(self, lhash, rhash, match, reason):
        self.data[self.get_key(lhash, rhash)] = {"match": match, "reason": reason}

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)


def ollama_compare(model, lt, lc, rt, rc, cache: OllamaCache):
    lhash = short_hash(lt + lc)
    rhash = short_hash(rt + rc)
    cached = cache.get(lhash, rhash)
    if cached:
        return cached["match"], cached["reason"]

    prompt = f"""
You are a Linux kernel patch comparison expert.
Determine if the following two patches represent different versions of the same change.

PATCH A (local):
Title: {lt}
Content:
{lc[:2000]}

PATCH B (Patchwork):
Title: {rt}
Content:
{rc[:2000]}

Answer 'yes' or 'no' first, then briefly explain in one English sentence.
"""
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode("utf-8"),
            capture_output=True,
            timeout=90
        )
        out = result.stdout.decode("utf-8", errors="ignore").strip()
        match = "yes" in out.lower()
        reason = out.split("\n", 1)[-1].strip() if "\n" in out else out
        cache.set(lhash, rhash, match, reason)
        return match, reason
    except Exception as e:
        cache.set(lhash, rhash, False, f"Ollama error: {e}")
        return False, str(e)


# ---------- Patchwork API ----------

def make_request(url, params=None, token=None, debug=False):
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Token {token}"
    if debug:
        qstr = "&".join([f"{k}={v}" for k, v in (params or {}).items()])
        print(f"[DEBUG] GET {url}?{qstr}")
    try:
        r = requests.get(url, params=params, headers=headers, timeout=30)
        if debug:
            print(f"[DEBUG] Status {r.status_code}")
        r.raise_for_status()
        data = r.json()
        if debug and "count" in data:
            print(f"[DEBUG] Found {data['count']} entries")
        return data
    except Exception as e:
        print(f"[WARN] Request failed: {e}")
        return None


def search_patchwork(title, token=None, debug=False):
    p = {"q": title, "format": "json"}
    data = make_request(PATCHWORK_API + "patches/", p, token, debug)
    return data.get("results", []) if data else []


def get_series_by_author(email, token=None, debug=False):
    p = {"submitter": email, "format": "json"}
    data = make_request(PATCHWORK_API + "series/", p, token, debug)
    return data.get("results", []) if data else []


def fetch_patch_details(pid, token=None, debug=False):
    return make_request(f"{PATCHWORK_API}patches/{pid}/", None, token, debug)


def fetch_series_details(sid, token=None, debug=False):
    return make_request(f"{PATCHWORK_API}series/{sid}/", None, token, debug)


# ---------- 主流程 ----------

def process_series(series_path, output_root, ollama_model, cache, token=None,
                   skip_download=False, debug=False):
    sdata = load_json_safe(series_path)
    if not sdata:
        return

    base_dir = os.path.dirname(series_path)
    sid = os.path.splitext(os.path.basename(series_path))[0]
    out_dir = os.path.join(output_root, sid)
    ensure_dir(out_dir)

    reasoning = []

    # 遍历 variants
    for ver, info in sdata.get("variants", {}).items():
        local_file = os.path.join(base_dir, info["source_file"])
        local_event = load_json_safe(local_file)
        if not local_event:
            continue
        msg = local_event["messages"][0]
        lt = clean_title(msg["subject"])
        lc = msg.get("content", "")

        print(f"[INFO] Series {sid} | {ver} | {lt[:50]}...")

        patches = search_patchwork(lt, token, debug)
        if not patches:
            print(f"[WARN] No Patchwork result for {lt}")
            continue

        author_email = patches[0]["submitter"]["email"]
        series_list = get_series_by_author(author_email, token, debug)
        for s in series_list:
            sdet = fetch_series_details(s["id"], token, debug)
            if not sdet:
                continue
            for p in sdet.get("patches", []):
                pdata = fetch_patch_details(p["id"], token, debug)
                if not pdata:
                    continue
                rt = pdata["name"]
                rc = pdata.get("diff", "") or pdata.get("commit_ref", "")
                match, reason = ollama_compare(ollama_model, lt, lc, rt, rc, cache)
                reasoning.append({
                    "variant": ver,
                    "remote_patch_id": p["id"],
                    "remote_title": rt,
                    "match": match,
                    "reason": reason
                })

    save_json(reasoning, os.path.join(out_dir, "summary.json"))
    print(f"[INFO] Done series {sid}")


def main(series_dir, output_dir, ollama_model, concurrency=5,
         token=None, skip_download=False, cache_path="./ollama_cache_v3.json", debug=False):

    ensure_dir(output_dir)
    series_files = []
    for root, _, files in os.walk(series_dir):
        for f in files:
            if f.startswith("event_series[") and f.endswith(".json"):
                series_files.append(os.path.join(root, f))

    print(f"[INFO] Found {len(series_files)} series to process.")
    cache = OllamaCache(cache_path)

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [
            ex.submit(process_series, f, output_dir, ollama_model, cache, token, skip_download, debug)
            for f in series_files
        ]
        for _ in tqdm(as_completed(futures), total=len(futures)):
            pass

    cache.save()
    print("[DONE] All series processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch compare series patches with Patchwork via Ollama.")
    parser.add_argument("--series_dir", required=True, help="Directory from match_patch_series.py output")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--ollama_model", default="qwen2", help="Ollama model (default=qwen2)")
    parser.add_argument("--token", help="Patchwork API token")
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--skip_download", action="store_true")
    parser.add_argument("--cache_path", default="./ollama_cache_v3.json")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    main(args.series_dir, args.output_dir, args.ollama_model,
         args.concurrency, args.token, args.skip_download, args.cache_path, args.debug)
