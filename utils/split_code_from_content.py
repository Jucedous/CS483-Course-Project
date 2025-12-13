

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split code segments from 'content' fields in LKML event JSON files.

Usage:
    python split_code_from_content.py --input ./events --output ./events_processed

This script:
  - Reads all JSON files in the input directory that start with 'lkml_'.
  - For each message, splits its 'content' field into:
      'message_content' (text)
      'code_content' (code/diff/patch lines)
  - If content is null, sets both to null.
  - Writes updated JSON files to the output directory.
"""

import os
import re
import json
import argparse
from tqdm import tqdm

# Regular expressions to identify code or diff content
PATCH_START_PAT = re.compile(
    r"^(diff --git .+|Index:|---\s+[ab]/|\+\+\+\s+[ab]/|@@\s.*@@)",
    re.IGNORECASE,
)
CODE_LINE_PAT = re.compile(
    r"^\s{4,}|\t|#include|#define|;|[{|}]|->|=\s*\w"
)


def split_content(content: str):
    """Split plain text into message_content and code_content."""
    if not content:
        return None, None

    lines = content.splitlines()
    mail_lines, code_lines = [], []
    in_patch = False

    for ln in lines:
        if PATCH_START_PAT.match(ln):
            in_patch = True
        if in_patch or CODE_LINE_PAT.search(ln):
            code_lines.append(ln)
        else:
            mail_lines.append(ln)

    message_content = "\n".join(mail_lines).strip() or None
    code_content = "\n".join(code_lines).strip() or None

    return message_content, code_content


def process_file(file_path, output_dir):
    """Process a single JSON file, splitting code from content, streaming output."""
    with open(file_path, "r", encoding="utf-8") as f:
        event = json.load(f)

    # Extract outer metadata except "messages"
    event_out = {}
    for k, v in event.items():
        if k != "messages":
            event_out[k] = v

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, os.path.basename(file_path))
    updated = False

    # Stream output: write metadata, open messages array, write each message, close array/object
    with open(out_path, "w", encoding="utf-8") as fout:
        fout.write('{\n')
        # Write outer fields except messages
        keys = list(event_out.keys())
        for idx, k in enumerate(keys):
            json.dump(k, fout, ensure_ascii=False)
            fout.write(': ')
            json.dump(event_out[k], fout, ensure_ascii=False, indent=2)
            if idx < len(keys) - 1 or "messages" in event:
                fout.write(',\n')
            else:
                fout.write('\n')

        fout.write('"messages": [\n')
        messages = event.get("messages", [])
        for i, msg in enumerate(messages):
            content = msg.get("content")
            message_content, code_content = split_content(content)
            msg_out = dict(msg)
            msg_out["message_content"] = message_content
            msg_out["code_content"] = code_content
            if "content" in msg_out:
                del msg_out["content"]
            updated = True
            json.dump(msg_out, fout, ensure_ascii=False, indent=2)
            if i < len(messages) - 1:
                fout.write(",\n")
            else:
                fout.write("\n")
        fout.write(']\n')
        fout.write('}\n')

    return out_path, updated


def main():
    parser = argparse.ArgumentParser(description="Split code from LKML JSON content.")
    parser.add_argument("--input", required=True, help="Input directory of JSON files")
    parser.add_argument("--output", required=True, help="Output directory for processed files")
    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output

    files = [
        os.path.join(input_dir, fn)
        for fn in os.listdir(input_dir)
        if fn.endswith(".json") and fn.startswith("lkml_")
    ]

    for fpath in tqdm(files, desc="Processing JSON files", unit="file"):
        process_file(fpath, output_dir)

    print(f"Processed {len(files)} JSON files. Output saved to {output_dir}")


if __name__ == "__main__":
    main()