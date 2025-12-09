#!/usr/bin/env python3
"""
Quick structural inspection of LKML JSON data.

- Looks inside:
    0_event_json_data/
    1_series_json_data/
- Samples a few JSON files from each folder.
- Prints:
    - number of files and total size
    - top-level keys and their frequencies
    - a short, nested summary of sample JSON contents

Run from the directory that contains those folders, e.g.:

    cd ~/all
    python inspect_lkml_data.py > lkml_data_report.txt
"""

import json
import gzip
import os
from pathlib import Path
from collections import Counter
import random

# CONFIG --------------------------------------------------------------------

DATA_FOLDERS = ["0_event_json_data", "1_series_json_data"]
SAMPLE_PER_FOLDER = 5            # how many files to inspect per folder
MAX_DEPTH = 2                    # recursion depth when summarizing nested structures
MAX_LIST_ITEMS = 2               # how many items per list to show
MAX_STR_LEN = 160                # max chars to show from any string

# ---------------------------------------------------------------------------

def open_json_file(path: Path):
    """Open a .json or .json.gz file and return the loaded object."""
    try:
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
                return json.load(f)
        else:
            with path.open("rt", encoding="utf-8", errors="replace") as f:
                return json.load(f)
    except Exception as e:
        print(f"  !! ERROR reading {path.name}: {e}")
        return None


def print_header(text, char="="):
    print()
    print(text)
    print(char * len(text))


def summarize(obj, name="", indent="", depth=0):
    """
    Recursively summarize a Python object (dict/list/primitive) without
    dumping everything. Meant for quickly understanding schema.
    """
    tname = type(obj).__name__

    if isinstance(obj, (int, float, bool)) or obj is None:
        print(f"{indent}{name} ({tname}): {obj}")
        return

    if isinstance(obj, str):
        snippet = obj.replace("\n", " ")[:MAX_STR_LEN]
        if len(obj) > MAX_STR_LEN:
            snippet += " ..."
        print(f"{indent}{name} (str, len={len(obj)}): {snippet!r}")
        return

    if isinstance(obj, list):
        print(f"{indent}{name} (list, len={len(obj)})")
        if depth >= MAX_DEPTH or not obj:
            return
        for i, item in enumerate(obj[:MAX_LIST_ITEMS]):
            summarize(item, name=f"{name}[{i}]", indent=indent + "  ", depth=depth + 1)
        if len(obj) > MAX_LIST_ITEMS:
            print(f"{indent}  ... ({len(obj) - MAX_LIST_ITEMS} more items not shown)")
        return

    if isinstance(obj, dict):
        keys = list(obj.keys())
        print(f"{indent}{name} (dict, {len(keys)} keys): {keys}")
        if depth >= MAX_DEPTH:
            return

        # Show up to MAX_LIST_ITEMS keys in more detail
        for k in keys[:MAX_LIST_ITEMS]:
            summarize(obj[k], name=f"{name}.{k}" if name else k,
                      indent=indent + "  ", depth=depth + 1)
        if len(keys) > MAX_LIST_ITEMS:
            print(f"{indent}  ... ({len(keys) - MAX_LIST_ITEMS} more keys not shown)")
        return

    # Fallback
    print(f"{indent}{name} ({tname}): {repr(obj)[:MAX_STR_LEN]}")


def inspect_folder(root: Path, folder_name: str):
    dir_path = root / folder_name
    print_header(f"FOLDER: {dir_path}")

    if not dir_path.is_dir():
        print("  !! Not a directory or does not exist.")
        return

    # Collect files (json + json.gz)
    all_files = [p for p in dir_path.rglob("*") if p.is_file()
                 and (p.suffix == ".json" or p.suffix == ".gz")]

    if not all_files:
        print("  !! No .json or .json.gz files found.")
        return

    total_size = sum(p.stat().st_size for p in all_files)
    print(f"  Total files found: {len(all_files)}")
    print(f"  Total size (approx): {total_size / (1024 * 1024):.2f} MB")

    # Sample some files deterministically
    random.seed(0)
    sample_files = all_files[:SAMPLE_PER_FOLDER] \
        if len(all_files) <= SAMPLE_PER_FOLDER \
        else random.sample(all_files, SAMPLE_PER_FOLDER)

    print(f"  Inspecting {len(sample_files)} sample files:")

    key_counter = Counter()

    for path in sample_files:
        print_header(f"Sample file: {path.name}", char="-")
        data = open_json_file(path)
        if data is None:
            continue

        print(f"  Top-level type: {type(data).__name__}")
        if isinstance(data, dict):
            keys = list(data.keys())
            key_counter.update(keys)
            print(f"  Top-level keys: {keys}")
            # Show a structured summary
            summarize(data)
        elif isinstance(data, list):
            print(f"  List length: {len(data)}")
            if data:
                print(f"  First element type: {type(data[0]).__name__}")
            summarize(data, name="root_list")
        else:
            summarize(data, name="root_value")

    print_header("Aggregated top-level key frequencies (sample)", char="-")
    for key, count in key_counter.most_common():
        print(f"  {key}: {count}")


def main():
    root = Path(os.getcwd())
    print_header(f"LKML data inspection in root: {root}")

    for folder in DATA_FOLDERS:
        inspect_folder(root, folder)


if __name__ == "__main__":
    main()
