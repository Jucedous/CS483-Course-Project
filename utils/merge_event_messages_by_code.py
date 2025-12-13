#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge LKML event JSONs by code content core nodes.

Usage:
    python merge_event_messages_by_code.py --input ./events_processed --output ./events_merged

Description:
    - Iterates through all LKML event JSON files in the input directory.
    - Merges messages based on `code_content`:
        * Nodes with non-null `code_content` become cores.
        * Following messages with null `code_content` merge into the previous core.
    - Builds merged units, each including a list of message contexts.
    - Updates the event's connections to reflect only valid core-to-core edges.
"""

import os
import json
import argparse
import re
from datetime import datetime
from tqdm import tqdm


def parse_url_timestamp(url):
    """Extract (datetime, id) from LKML URL of form .../YYYY/MM/DD/HHmmss_id."""
    # Example URL: https://lkml.org/lkml/2021/05/10/123456_1234
    m = re.search(r'/(\d{4})/(\d{2})/(\d{2})/(\d{6})_(\d+)', url)
    if m:
        year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
        time_str = m.group(4)  # HHmmss
        hour, minute, second = int(time_str[0:2]), int(time_str[2:4]), int(time_str[4:6])
        msg_id = m.group(5)
        try:
            dt = datetime(year, month, day, hour, minute, second)
            return dt, msg_id
        except Exception:
            return None, None
    return None, None


def orient_edge_by_time(u, v, time_cache):
    """Orient edge from earlier to later node by timestamp; fallback to lex order."""
    if u not in time_cache:
        time_cache[u] = parse_url_timestamp(u)
    if v not in time_cache:
        time_cache[v] = parse_url_timestamp(v)
    u_time, u_id = time_cache[u]
    v_time, v_id = time_cache[v]

    if u_time and v_time:
        if u_time < v_time:
            return u, v
        elif v_time < u_time:
            return v, u
        else:
            # same time, fallback to lex order of id if both present
            if u_id and v_id:
                if u_id <= v_id:
                    return u, v
                else:
                    return v, u
            else:
                # fallback lex order of URLs
                if u <= v:
                    return u, v
                else:
                    return v, u
    else:
        # missing timestamps, fallback lex order of URLs
        if u <= v:
            return u, v
        else:
            return v, u


def merge_messages(event):
    """Merge messages in an event by code_content."""
    messages = event.get("messages", [])
    connections = event.get("connections", [])

    merged_units = []
    core_urls = set()
    url_to_core = {}

    current_unit = None

    # Pass 1: merge messages by code_content
    for msg in messages:
        url = msg.get("url")
        subject = msg.get("subject")
        message_content = msg.get("message_content")
        code_content = msg.get("code_content")

        if code_content:  # start new core unit
            current_unit = {
                "core_url": url,
                "core_subject": subject,
                "core_code_content": code_content,
                "messages_context": [
                    {
                        "url": url,
                        "subject": subject,
                        "message_content": message_content,
                    }
                ],
            }
            merged_units.append(current_unit)
            core_urls.add(url)
            url_to_core[url] = url
        else:
            # attach to last core unit
            if current_unit:
                current_unit["messages_context"].append(
                    {
                        "url": url,
                        "subject": subject,
                        "message_content": message_content,
                    }
                )
                if url not in core_urls:
                    url_to_core[url] = current_unit["core_url"]

    # Pass 2: rebuild connections enforcing core-to-core edges, no indirect chaining, orient by time
    new_connections = []
    seen_edges = set()
    time_cache = {}

    for conn in connections:
        from_url = conn.get("from")
        to_url = conn.get("to")
        if not from_url or not to_url:
            continue

        # map to core nodes only; do not propagate through content nodes
        from_core = url_to_core.get(from_url) if from_url in url_to_core else (from_url if from_url in core_urls else None)
        to_core = url_to_core.get(to_url) if to_url in url_to_core else (to_url if to_url in core_urls else None)

        if from_core and to_core and from_core != to_core and from_core in core_urls and to_core in core_urls:
            # Prevent future -> past edges
            from_time = parse_url_timestamp(from_core)[0]
            to_time = parse_url_timestamp(to_core)[0]
            if from_time and to_time:
                if from_time > to_time:
                    continue
            u, v = orient_edge_by_time(from_core, to_core, time_cache)
            if u != v:
                edge = (u, v)
                if edge not in seen_edges:
                    new_connections.append({"from": u, "to": v, "connection_tags": None})
                    seen_edges.add(edge)

    # Fallback: If no merged_units but messages exist, create a pseudo-core merged unit
    if not merged_units and messages:
        first_msg = messages[0]
        merged_units = [{
            "core_url": first_msg.get("url"),
            "core_subject": first_msg.get("subject"),
            "core_code_content": None,
            "messages_context": [
                {
                    "url": msg.get("url"),
                    "subject": msg.get("subject"),
                    "message_content": msg.get("message_content"),
                } for msg in messages
            ],
        }]
        new_connections = []

    result = {
        "event_id": event.get("event_id"),
        "root_url": event.get("root_url"),
        "merged_units": merged_units,
        "connections": new_connections,
    }

    return result


def process_file(file_path, output_dir):
    """Process and merge a single JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        event = json.load(f)

    merged_event = merge_messages(event)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, os.path.basename(file_path))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged_event, f, ensure_ascii=False, indent=2)

    return out_path


def main():
    parser = argparse.ArgumentParser(description="Merge LKML events by code core nodes.")
    parser.add_argument("--input", required=True, help="Input directory containing event JSONs")
    parser.add_argument("--output", required=True, help="Output directory for merged events")
    args = parser.parse_args()

    files = [
        os.path.join(args.input, fn)
        for fn in os.listdir(args.input)
        if fn.endswith(".json") and fn.startswith("lkml_")
    ]

    for fpath in tqdm(files, desc="Merging events", unit="file"):
        process_file(fpath, args.output)

    print(f"Merged {len(files)} event JSON files.")


if __name__ == "__main__":
    main()
