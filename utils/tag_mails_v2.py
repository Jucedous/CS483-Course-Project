#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hierarchical tagging for structured LKML event JSONs.

Usage:
    python tag_mails_v2.py --input ./events_merged --output ./events_tagged

This script tags:
  - code nodes (core_code_content)
  - message contents (message_content)
  - connections (relation between cores)
  - entire event (overall topic and merge outcome classification)

It uses the same model interface as tag_mails.py and keeps the OpenAI chat call structure.
"""

import os
import json
import argparse
from tqdm import tqdm
from openai import OpenAI
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Global stats dictionary to track progress and usage
STATS = {
    "total_files": 0,
    "total_nodes": 0,
    "api_calls": 0,
    "tokens": 0,
}
STATS_LOCK = threading.Lock()

# Configuration block
OPENAI_API_KEY = "sk-wZ3Zpfep5N2wa2KXDcBbB29075554b0988196b645cE44615"  # 必填：你的 OpenAI API key
OPENAI_BASE_URL = "https://api.bianxie.ai/v1"  # 可选：兼容端点
OPENAI_MODEL = "gpt-4o-mini"  # 可选：模型名称

CONCURRENCY = 20
MAX_RETRY = 3
RETRY_INTERVAL = 3

TAGGING_ENABLED = True
TAGGING_LIMIT = 10000
TAGGING_BATCH_SIZE = 10

TRUNCATE_TEXT = True
TRUNCATE_LIMIT = 8000


# Initialize OpenAI client (reuse existing API structure)
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

# ANGLES_GUIDE and SYSTEM_PROMPT (English)
ANGLES_GUIDE = """
You are an expert annotator for Linux kernel mailing list events. Your task is to assign concise, accurate, and relevant tags to code snippets, message contents, connections between nodes, and entire events. Use the following instructions integrating socio-technical and technical insights:

1. Tags should be short (1-4 words), informative, and use common technical terminology.
2. Tags describe the main topic, functionality, change, or socio-technical aspect (e.g., "memory leak fix", "device tree update", "refactoring", "error handling", "code ownership", "review discussion", "dependency introduced").
3. Avoid generic tags like "code", "patch", "discussion".
4. For connections, describe the nature of the relationship by combining content from both endpoints (e.g., "fixes bug", "responds to review", "dependency introduced", "addresses feedback").
5. For events, summarize the overall topic or main change, and include a merge outcome classification: "merge_success" if the patch series was merged successfully, "merge_failed" if it was rejected, or "still_fixing" if it is still under development or review.
6. Use lowercase, no punctuation unless necessary.
7. If unsure, use your best judgment based on context.
"""

SYSTEM_PROMPT = (
    "You are a helpful assistant skilled at tagging software engineering artifacts. "
    "Follow the provided guidelines and output only the requested tags."
)

# Tagging prompts for different node types with socio-technical and technical insights
CODE_PROMPT = (
    "Given the following Linux kernel code snippet from a mailing list patch, "
    "assign 1-3 concise tags describing its main purpose, functionality, change, or technical aspect. "
    "Consider socio-technical context if relevant. "
    "Example tags: memory allocation, error handling, device driver update, concurrency fix. "
    "Code snippet:\n{content}\nTags:"
)

MESSAGE_PROMPT = (
    "Given the following message content from a Linux kernel mailing list thread, "
    "assign 1-3 concise tags describing the main topic, question, discussion point, or socio-technical aspect. "
    "Examples: review comments, bug report, performance concern, code ownership discussion. "
    "Message content:\n{content}\nTags:"
)

CONNECTION_PROMPT = (
    "Given the following relationship between two code/message nodes in a Linux kernel mailing list event, "
    "combine the full content from both endpoints (subject, message content, code content) and assign 1-2 concise tags describing the nature of this connection. "
    "Examples: fixes bug, responds to review, introduces dependency, addresses feedback. "
    "Combined content:\n{content}\nTags:"
)

EVENT_PROMPT = (
    "Given the following Linux kernel mailing list event (patch series or discussion thread), "
    "assign 1-3 concise tags summarizing the overall topic, main technical changes, socio-technical context, and classify the merge outcome as one of: merge_success, merge_failed, still_fixing. "
    "Examples: memory management, driver update, refactoring, merge_success. "
    "Event summary:\n{content}\nTags:"
)


# Helper: truncate text if too long
def truncate_text(text, limit=TRUNCATE_LIMIT):
    if text is None:
        return ""
    if not TRUNCATE_TEXT or len(text) <= limit:
        return text
    # Try to truncate at a line boundary
    lines = text.splitlines()
    result = []
    total = 0
    for line in lines:
        if total + len(line) + 1 > limit:
            break
        result.append(line)
        total += len(line) + 1
    return "\n".join(result)


# OpenAI call with retry and note
def call_openai(prompt, system_prompt=SYSTEM_PROMPT, model=OPENAI_MODEL, max_retry=MAX_RETRY):
    import time
    for attempt in range(max_retry):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=64,
            )
            tags = response.choices[0].message.content.strip()
            with STATS_LOCK:
                STATS["api_calls"] += 1
            note = {
                "model": model,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
            return tags, note
        except Exception as e:
            if attempt < max_retry - 1:
                time.sleep(RETRY_INTERVAL)
            else:
                raise e


# Helper to build combined content for connection tagging
def build_connection_content(conn):
    parts = []
    # Include description if present
    desc = conn.get("description", "")
    if desc:
        parts.append(desc)
    # Include content from both endpoints if available
    endpoints = conn.get("endpoints", [])
    for ep in endpoints:
        # Each endpoint may have subject, message_content, core_code_content
        subject = ep.get("subject", "")
        if subject:
            parts.append(subject)
        message_content = ep.get("message_content", "")
        if message_content:
            parts.append(message_content)
        code_content = ep.get("core_code_content", "")
        if code_content:
            parts.append(code_content)
    combined = "\n".join(parts)
    return truncate_text(combined)


# Helper to convert tag string to list if needed
def parse_tags(tags):
    if isinstance(tags, list):
        return tags
    if isinstance(tags, str):
        # Split by comma and strip whitespace
        return [t.strip() for t in tags.split(",") if t.strip()]
    return []


# Tag a single event node (code, message, connection, or event)
def tag_event(event, angles_guide=ANGLES_GUIDE):
    nodes_tagged = 0
    # Tag code and message nodes
    for node in event.get("nodes", []):
        node_type = node.get("type")
        if node_type == "core_code_content":
            content_raw = node.get("content", None)
            if content_raw is None:
                continue
            content = truncate_text(content_raw)
            prompt = angles_guide + "\n" + CODE_PROMPT.format(content=content)
            tags, note = call_openai(prompt)
            tags = parse_tags(tags)
            node["tags"] = tags
            node["tag_note"] = note
            nodes_tagged += 1
        elif node_type == "message_content":
            content_raw = node.get("content", None)
            if content_raw is None:
                continue
            content = truncate_text(content_raw)
            prompt = angles_guide + "\n" + MESSAGE_PROMPT.format(content=content)
            tags, note = call_openai(prompt)
            tags = parse_tags(tags)
            node["tags"] = tags
            node["tag_note"] = note
            nodes_tagged += 1
        else:
            # For any other node types, still tag with message prompt as fallback
            content_raw = node.get("content", None)
            if content_raw is None:
                continue
            content = truncate_text(content_raw)
            if content:
                prompt = angles_guide + "\n" + MESSAGE_PROMPT.format(content=content)
                tags, note = call_openai(prompt)
                tags = parse_tags(tags)
                node["tags"] = tags
                node["tag_note"] = note
                nodes_tagged += 1

    # Tag messages inside merged_units, and also tag core_code_content inside merged_units
    for merged_unit in event.get("merged_units", []):
        # Tag core_code_content inside merged_unit
        core_code_content = merged_unit.get("core_code_content", None)
        if core_code_content is not None:
            content = truncate_text(core_code_content)
            if content:
                prompt = angles_guide + "\n" + CODE_PROMPT.format(content=content)
                tags, note = call_openai(prompt)
                tags = parse_tags(tags)
                merged_unit["core_tags"] = tags
                merged_unit["core_tag_note"] = note
                nodes_tagged += 1
        for message in merged_unit.get("messages_context", []):
            content_raw = message.get("message_content", None)
            if content_raw is None:
                continue
            content = truncate_text(content_raw)
            if content:
                prompt = angles_guide + "\n" + MESSAGE_PROMPT.format(content=content)
                tags, note = call_openai(prompt)
                tags = parse_tags(tags)
                message["tags"] = tags
                message["tag_note"] = note
                nodes_tagged += 1

    # Tag connections: for each connection, combine both endpoints' core_subject, core_code_content,
    # and all messages_context contents from corresponding merged_units, then tag.
    merged_units = event.get("merged_units", [])
    core_url_to_unit = {}
    for mu in merged_units:
        url = mu.get("core_url")
        if url:
            core_url_to_unit[url] = mu
    for conn in event.get("connections", []):
        if not conn:
            continue
        from_url = conn.get("from")
        to_url = conn.get("to")
        from_unit = core_url_to_unit.get(from_url)
        to_unit = core_url_to_unit.get(to_url)
        # If can't find both endpoints, skip
        if not from_unit or not to_unit:
            continue
        # Gather content from both endpoints
        parts = []
        # from endpoint
        subj = from_unit.get("core_subject", "")
        if subj:
            parts.append(subj)
        code = from_unit.get("core_code_content", "")
        if code:
            parts.append(code)
        for msg in from_unit.get("messages_context", []):
            msg_content = msg.get("message_content", "")
            if msg_content:
                parts.append(msg_content)
        # to endpoint
        subj = to_unit.get("core_subject", "")
        if subj:
            parts.append(subj)
        code = to_unit.get("core_code_content", "")
        if code:
            parts.append(code)
        for msg in to_unit.get("messages_context", []):
            msg_content = msg.get("message_content", "")
            if msg_content:
                parts.append(msg_content)
        combined_text = "\n".join(parts)
        if not combined_text.strip():
            continue
        combined_text = truncate_text(combined_text)
        prompt = angles_guide + "\n" + CONNECTION_PROMPT.format(content=combined_text)
        tags, note = call_openai(prompt)
        tags = parse_tags(tags)
        conn["connection_tags"] = tags
        conn["connection_tag_note"] = note
        nodes_tagged += 1

    # Tag event as a whole including merge outcome classification
    summary = truncate_text(event.get("summary", ""))
    # If summary is missing or empty, build from merged_units core_code_content and message_content
    if not summary.strip():
        parts = []
        for merged_unit in event.get("merged_units", []):
            core_code = merged_unit.get("core_code_content", "")
            if core_code:
                parts.append(core_code)
            for message in merged_unit.get("messages_context", []):
                msg_content = message.get("message_content", "")
                if msg_content:
                    parts.append(msg_content)
        summary = truncate_text("\n".join(parts))
    # Append merge outcome info if available
    merge_outcome = event.get("merge_outcome", "")
    if merge_outcome:
        summary += f"\nMerge outcome: {merge_outcome}"
    prompt = angles_guide + "\n" + EVENT_PROMPT.format(content=summary)
    tags, note = call_openai(prompt)
    tags = parse_tags(tags)
    event["tags"] = tags
    event["tag_note"] = note
    nodes_tagged += 1

    return event, nodes_tagged


# Process a single file for tagging
def process_one_file(input_file, output_file):
    with open(input_file, "r") as fin:
        event = json.load(fin)
    tagged, nodes_tagged = tag_event(event)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as fout:
        json.dump(tagged, fout, indent=2)
    with STATS_LOCK:
        STATS["total_files"] += 1
        STATS["total_nodes"] += nodes_tagged
        total_files = STATS["total_files"]
        total_nodes = STATS["total_nodes"]
        api_calls = STATS["api_calls"]
    print(f"Processed file: {os.path.basename(input_file)} | Tagged nodes: {total_nodes} | API calls: {api_calls}")
    if total_files % 1000 == 0:
        print(f"Summary after {total_files} files:")
        print(f"  Total files processed: {total_files}")
        print(f"  Total nodes tagged: {total_nodes}")
        print(f"  Total API calls: {api_calls}")
        print(f"  Total tokens used: {STATS['tokens']}")


# Process all files in a directory or a single file with concurrency
def process_path(input_path, output_path, resume=True):
    if os.path.isdir(input_path):
        os.makedirs(output_path, exist_ok=True)
        files = [f for f in os.listdir(input_path) if f.endswith(".json")]
        input_files = [os.path.join(input_path, f) for f in files]
        output_files = [os.path.join(output_path, f) for f in files]
        to_process = []
        if resume:
            # Gather already processed files in output_path (non-empty .json files)
            processed = set()
            for fname in os.listdir(output_path):
                if fname.endswith(".json"):
                    fpath = os.path.join(output_path, fname)
                    try:
                        if os.path.getsize(fpath) > 50:
                            # Try loading to check if it's valid JSON
                            with open(fpath, "r") as fin:
                                json.load(fin)
                            processed.add(fname)
                    except Exception:
                        pass
            for inf, outf in zip(input_files, output_files):
                filename = os.path.basename(outf)
                if filename in processed:
                    print(f"Skipping already tagged file: {filename}")
                    continue
                to_process.append((inf, outf))
        else:
            to_process = list(zip(input_files, output_files))
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
            futures = {executor.submit(process_one_file, inf, outf): inf for inf, outf in to_process}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Tagging events"):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing file {futures[future]}: {e}")
    else:
        process_one_file(input_path, output_path)


def main():
    parser = argparse.ArgumentParser(description="Hierarchical tagging for LKML event JSONs.")
    parser.add_argument("--input", required=True, help="Input event JSON file or directory")
    parser.add_argument("--output", required=True, help="Output file or directory for tagged events")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from previously processed files")
    args = parser.parse_args()
    process_path(args.input, args.output, resume=args.resume)
    # Print final stats summary
    print("\nTagging Summary:")
    print(f"Total files processed: {STATS['total_files']}")
    print(f"Total nodes tagged: {STATS['total_nodes']}")
    print(f"Total API calls: {STATS['api_calls']}")
    print(f"Total tokens used: {STATS['tokens']}")


if __name__ == "__main__":
    main()
