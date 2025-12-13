#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
get_series_by_subject_and_download.py
--------------------------------------------------------
批量模式：
    在 security_series_data/ 目录下遍历每个事件文件夹，
    自动读取其中的 event_series*.json 文件，
    基于其中的 subject 搜索 Patchwork，
    自动下载该 subject 对应的 series 以及所有 patch。

功能：
    1. 自动从 subject 中提取 title（去除所有 [xxx] 前缀）
    2. 从 Patchwork 搜索 patch → series
    3. 下载 series 的所有 patch：metadata + diff + mbox
    4. 保存到当前事件目录下

使用示例：
    python get_series_by_subject_and_download.py \
        --input_dir ./security_series_data \
        --token TOKEN \
        --debug
"""

import os
import json
import re
import requests
import argparse


PATCHWORK_API = "https://patchwork.kernel.org/api/"


# -------------------- Utils --------------------

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def make_request(url, params=None, token=None, debug=False):
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Token {token}"

    if debug:
        qs = "&".join([f"{k}={v}" for k, v in (params or {}).items()])
        print(f"[DEBUG] GET {url}?{qs}")

    r = requests.get(url, params=params, headers=headers, timeout=30)
    if debug:
        print(f"[DEBUG] Status {r.status_code}")

    r.raise_for_status()
    return r.json()


def extract_patch_results(data):
    if isinstance(data, dict) and "results" in data:
        return data["results"]
    if isinstance(data, list):
        return data
    return []


def normalize_series_entry(series_entry):
    if isinstance(series_entry, str):
        return series_entry
    if isinstance(series_entry, dict):
        if "url" in series_entry:
            return series_entry["url"]
        if "id" in series_entry:
            return f"https://patchwork.kernel.org/api/series/{series_entry['id']}/"
    return None


# -------------------- Title from subject --------------------

def title_from_subject(subject):
    """
    去掉所有 [xxx] 前缀，例如：
    '[REG][V2] something something' → 'something something'
    """
    cleaned = re.sub(r"^\s*(\[[^\]]*\]\s*)+", "", subject).strip()
    return cleaned


# -------------------- Patchwork find series --------------------

def fetch_series_by_title(title, token=None, debug=False):
    params = {"q": title, "format": "json"}
    data = make_request(PATCHWORK_API + "patches/", params, token, debug)

    results = extract_patch_results(data)
    if not results:
        print(f"[WARN] No Patchwork result for title: {title}")
        return None

    patch = results[0]
    print(f"[INFO] Found patch: {patch['name']} (id={patch['id']})")

    series_entries = patch.get("series", [])
    if not series_entries:
        print("[WARN] Patch has no series.")
        return None

    series_url = normalize_series_entry(series_entries[0])
    if not series_url:
        print("[WARN] Cannot parse series entry.")
        return None

    series_url = series_url.split("?")[0]
    return make_request(series_url, None, token, debug)


# -------------------- Download one patch --------------------

def download_patch(patch_id, patch_dir, mbox_dir, token=None, debug=False):
    url = f"{PATCHWORK_API}patches/{patch_id}/"
    pdata = make_request(url, None, token, debug)

    # JSON metadata
    meta_file = os.path.join(patch_dir, f"patch_{patch_id}.json")
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(pdata, f, indent=2, ensure_ascii=False)

    # mbox
    mbox_url = pdata.get("mbox")
    if mbox_url:
        if debug:
            print(f"[DEBUG] Download mbox: {mbox_url}")
        r = requests.get(mbox_url, timeout=30)
        r.raise_for_status()
        with open(os.path.join(mbox_dir, f"patch_{patch_id}.mbox"), "wb") as f:
            f.write(r.content)

    return pdata


# -------------------- Download Series --------------------

def download_series(series, save_root, token=None, debug=False):
    sid = series["id"]
    series_dir = os.path.join(save_root, f"series_{sid}")

    ensure_dir(series_dir)
    ensure_dir(os.path.join(series_dir, "patches"))
    ensure_dir(os.path.join(series_dir, "mbox"))

    # save series metadata
    with open(os.path.join(series_dir, "series.json"), "w", encoding="utf-8") as f:
        json.dump(series, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Downloading series {sid} with {len(series['patches'])} patches...")

    for p in series["patches"]:
        pid = p["id"]
        print(f"[INFO]   -> patch {pid}: {p['name']}")

        download_patch(
            pid,
            patch_dir=os.path.join(series_dir, "patches"),
            mbox_dir=os.path.join(series_dir, "mbox"),
            token=token,
            debug=debug
        )


# -------------------- Main Loop --------------------

def main():
    parser = argparse.ArgumentParser(description="Batch: find series by subject in event folders and download all patches.")
    parser.add_argument("--input_dir", required=True, help="Directory that contains many event folders.")
    parser.add_argument("--token", help="Patchwork API token")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # 遍历事件目录
    for event_dir in sorted(os.listdir(args.input_dir)):
        full_event_path = os.path.join(args.input_dir, event_dir)
        if not os.path.isdir(full_event_path):
            continue

        print(f"\n===== Processing event folder: {event_dir} =====")

        # 寻找 event_series*.json
        json_files = [f for f in os.listdir(full_event_path) if f.startswith("event_series") and f.endswith(".json")]
        if not json_files:
            print("[WARN] No event_series JSON found, skip.")
            continue

        json_path = os.path.join(full_event_path, json_files[0])

        # 读取 subject
        with open(json_path) as f:
            data = json.load(f)

        subject = data.get("subject")
        if not subject:
            print("[WARN] No subject in event JSON, skip.")
            continue

        title = title_from_subject(subject)
        print(f"[INFO] Extracted search title: {title}")

        # 查 Patchwork series
        series = fetch_series_by_title(title, args.token, args.debug)
        if not series:
            print("[WARN] No series found for this title.")
            continue

        # 下载数据到事件文件夹
        download_series(series, full_event_path, args.token, args.debug)

    print("\n[DONE] All events processed.")


if __name__ == "__main__":
    main()