'''
python utils/group_event_to_series.py \
  --input /Users/luyaobai/Workspace/lkml-mining/data/events_tagged \
  --output /Users/luyaobai/Workspace/lkml-mining/data/event_series
'''

import os
import re
import json
import argparse
from collections import defaultdict

def parse_subject_general(subject: str):
    """统一解析 LKML 标题，提取前缀、版本号、主题"""
    if not subject:
        return {
            "prefix": "GENERAL",
            "version": "v1",
            "topic": "",
            "normalized_subject": ""
        }

    subject = subject.strip()
    subject = re.sub(r'^(Re:\s*)+', '', subject, flags=re.IGNORECASE)

    pattern = re.compile(
        r'\[(?P<prefix>[A-Z\- ]*PATCH|RFC|GIT PULL|RESEND|TEST)[^\]]*\]'
        r'[\s:-]*(?P<core>.*)',
        re.IGNORECASE
    )
    m = pattern.match(subject)
    if not m:
        return {
            "prefix": "GENERAL",
            "version": "v1",
            "topic": subject.strip(),
            "normalized_subject": subject.strip()
        }

    raw_prefix = m.group('prefix').strip().upper()
    core = m.group('core').strip()

    # 提取版本号
    version_match = re.search(r'v(\d+)', subject, re.IGNORECASE)
    version = f"v{version_match.group(1)}" if version_match else "v1"

    # 提取索引 (02/39)
    sub_match = re.search(r'\d+/\d+', subject)
    subindex = sub_match.group(0) if sub_match else None

    # 清理多余符号
    core = re.sub(r'^\d+/\d+\s*', '', core)
    core = re.sub(r'^[\s:\-]+', '', core)

    return {
        "prefix": raw_prefix,
        "version": version,
        "subindex": subindex,
        "topic": core.strip(),
        "normalized_subject": f"{raw_prefix} {core.strip()}"
    }


def group_event_series(input_dir, output_dir):
    """将相同主题（不同版本或回复）的事件聚合为一个系列"""
    grouped = defaultdict(lambda: {
        "subject": "",
        "topic_type": "",
        "variants": {},
        "connections": []
    })

    # 遍历所有 JSON 事件文件
    for file in os.listdir(input_dir):
        if not file.endswith(".json"):
            continue
        path = os.path.join(input_dir, file)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"⚠️ 无法读取 {file}: {e}")
            continue

        subject = (
            data.get("merged_units", [{}])[0].get("core_subject")
            or data.get("core_subject")
            or ""
        )
        subj_info = parse_subject_general(subject)
        key = subj_info["normalized_subject"]

        group = grouped[key]
        group["subject"] = subj_info["topic"]
        group["topic_type"] = subj_info["prefix"]

        version = subj_info["version"]
        group["variants"][version] = {
            "event_id": data.get("event_id"),
            "url": data.get("root_url"),
            "source_file": file
        }

    # 构建版本连接
    for g in grouped.values():
        versions = sorted(g["variants"].keys(), key=lambda v: int(re.sub(r'\D', '', v)))
        g["connections"] = [
            {"from": versions[i], "to": versions[i + 1]} for i in range(len(versions) - 1)
        ]

    # 输出每个系列为独立 JSON 文件
    os.makedirs(output_dir, exist_ok=True)
    for k, v in grouped.items():
        if v["variants"]:
            first_variant = list(v["variants"].values())[0]
            first_event_id = first_variant.get("event_id", "unknown")
        else:
            first_event_id = "unknown"
        out_path = os.path.join(output_dir, f"event_series[{first_event_id}].json")
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(v, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ 写入 {out_path} 失败: {e}")

    print(f"✅ 生成 {len(grouped)} 个事件系列，存储于 {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LKML Event Series Grouper")
    parser.add_argument("--input", required=True, help="输入事件文件目录（包含 .json）")
    parser.add_argument("--output", required=True, help="输出目录（将生成多个 event_series_*.json）")
    args = parser.parse_args()

    group_event_series(args.input, args.output)
