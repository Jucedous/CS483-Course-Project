

import os
import json
import argparse
from collections import Counter, defaultdict
from tqdm import tqdm
import pandas as pd

def extract_tags_from_event(json_obj):
    """
    Extract tags from the event JSON structure:
    - Event-level: tags
    - Connection-level: connections[].connection_tags
    - Code/message-level: merged_units[].core_tags and messages_context[].tags
    """
    event_tags, connection_tags, code_tags = [], [], []

    # Event-level
    if isinstance(json_obj.get("tags"), list):
        event_tags.extend(str(t) for t in json_obj["tags"])

    # Connection-level
    for conn in json_obj.get("connections", []):
        if isinstance(conn.get("connection_tags"), list):
            connection_tags.extend(str(t) for t in conn["connection_tags"])

    # Code/message-level
    for mu in json_obj.get("merged_units", []):
        if isinstance(mu.get("core_tags"), list):
            code_tags.extend(str(t) for t in mu["core_tags"])
        for msg in mu.get("messages_context", []):
            if isinstance(msg.get("tags"), list):
                code_tags.extend(str(t) for t in msg["tags"])

    # Log summary counts for debugging
    if any([event_tags, connection_tags, code_tags]):
        print(f"Extracted: {len(event_tags)} event tags, {len(connection_tags)} connection tags, {len(code_tags)} code tags")
    else:
        print("[Warning] No tags found in event JSON structure")

    return event_tags, connection_tags, code_tags

def count_tags(tag_lists):
    """
    Given a list of lists of tags, return a Counter of tag frequencies.
    """
    flat_tags = [tag for sublist in tag_lists for tag in sublist]
    return Counter(flat_tags)

def build_cooccurrence_matrix(tag_lists):
    """
    Given a list of lists of tags, build a co-occurrence Counter for tag pairs.
    Returns: Counter mapping (tag1, tag2) -> count, always tag1 <= tag2
    """
    pair_counter = Counter()
    for tags in tag_lists:
        unique_tags = sorted(set(tags))
        for i in range(len(unique_tags)):
            for j in range(i+1, len(unique_tags)):
                pair = (unique_tags[i], unique_tags[j])
                pair_counter[pair] += 1
    return pair_counter

def process_json_files(input_path):
    """
    Walk through a directory or process a single file, yielding JSON objects.
    """
    if os.path.isdir(input_path):
        json_files = []
        for root, dirs, files in os.walk(input_path):
            for file in files:
                if file.endswith(".json"):
                    json_files.append(os.path.join(root, file))
    else:
        json_files = [input_path]

    for file_path in tqdm(json_files, desc="Processing JSON files"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # If data is a list of events
                if isinstance(data, list):
                    for event in data:
                        yield event
                # If data is a single event
                elif isinstance(data, dict):
                    yield data
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

def save_tag_frequency_summary(tag_counter, output_csv):
    if not tag_counter:
        print(f"[Warning] No tags found for {output_csv}")
        pd.DataFrame(columns=["tag", "count"]).to_csv(output_csv, index=False, encoding="utf-8")
        return
    df = pd.DataFrame(tag_counter.items(), columns=["tag", "count"])
    df = df.sort_values(by="count", ascending=False)
    df.to_csv(output_csv, index=False, encoding="utf-8")

def save_cooccurrence_matrix(pair_counter, output_csv):
    if not pair_counter:
        print(f"[Warning] No tag pairs found for {output_csv}")
        pd.DataFrame(columns=["tag1", "tag2", "count"]).to_csv(output_csv, index=False, encoding="utf-8")
        return
    rows = []
    for (t1, t2), count in pair_counter.items():
        rows.append({"tag1": t1, "tag2": t2, "count": count})
    df = pd.DataFrame(rows)
    df = df.sort_values(by="count", ascending=False)
    df.to_csv(output_csv, index=False, encoding="utf-8")

def main():
    parser = argparse.ArgumentParser(description="Tag analysis for event, connection, and code/message levels.")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file or directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory for CSVs")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    event_tag_lists = []
    connection_tag_lists = []
    code_tag_lists = []

    for event in process_json_files(args.input):
        event_tags, connection_tags, code_tags = extract_tags_from_event(event)
        if event_tags:
            event_tag_lists.append(event_tags)
        if connection_tags:
            connection_tag_lists.append(connection_tags)
        if code_tags:
            code_tag_lists.append(code_tags)

    # Event-level analysis
    event_tag_counter = count_tags(event_tag_lists)
    event_pair_counter = build_cooccurrence_matrix(event_tag_lists)
    save_tag_frequency_summary(event_tag_counter, os.path.join(args.output, "event_tag_frequency_summary.csv"))
    save_cooccurrence_matrix(event_pair_counter, os.path.join(args.output, "event_association_tag_pairs.csv"))

    # Connection-level analysis
    connection_tag_counter = count_tags(connection_tag_lists)
    connection_pair_counter = build_cooccurrence_matrix(connection_tag_lists)
    save_tag_frequency_summary(connection_tag_counter, os.path.join(args.output, "connection_tag_frequency_summary.csv"))
    save_cooccurrence_matrix(connection_pair_counter, os.path.join(args.output, "connection_association_tag_pairs.csv"))

    # Code/message-level analysis
    code_tag_counter = count_tags(code_tag_lists)
    code_pair_counter = build_cooccurrence_matrix(code_tag_lists)
    save_tag_frequency_summary(code_tag_counter, os.path.join(args.output, "code_tag_frequency_summary.csv"))
    save_cooccurrence_matrix(code_pair_counter, os.path.join(args.output, "code_association_tag_pairs.csv"))

    print("Analysis complete. Results saved to", args.output)

if __name__ == "__main__":
    main()