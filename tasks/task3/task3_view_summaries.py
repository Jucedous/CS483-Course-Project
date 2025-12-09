#!/usr/bin/env python3
"""
Task 3: View / Inspect Summaries

Reads:
    - tasks/task3/summaries_task3_with_roles.csv

Prints:
    - basic stats
    - a few example summaries (both long and short threads)
"""

from pathlib import Path

import pandas as pd


THIS_FILE = Path(__file__).resolve()
TASK3_DIR = THIS_FILE.parent
ROOT = TASK3_DIR.parents[1]
SUMMARIES_CSV = TASK3_DIR / "summaries_task3_with_roles.csv"


def main():
    print(f"[Task3-View] ROOT = {ROOT}")
    print(f"[Task3-View] TASK3_DIR = {TASK3_DIR}")

    if not SUMMARIES_CSV.is_file():
        raise FileNotFoundError(
            f"Cannot find {SUMMARIES_CSV}. Run task3_build_summaries.py first."
        )

    df = pd.read_csv(SUMMARIES_CSV)
    print(f"[Task3-View] Loaded {len(df)} summaries.")

    print("\n[Task3-View] long_thread distribution:")
    print(df["long_thread"].value_counts())

    print("\n[Task3-View] === Example LONG threads ===")
    long_examples = df[df["long_thread"] == 1].head(3)
    for _, row in long_examples.iterrows():
        print("\n---")
        print(f"event_id: {row['event_id']}")
        print(f"root_subject: {row['root_subject']}")
        print(row["summary_text"])

    print("\n[Task3-View] === Example SHORT threads ===")
    short_examples = df[df["long_thread"] == 0].head(3)
    for _, row in short_examples.iterrows():
        print("\n---")
        print(f"event_id: {row['event_id']}")
        print(f"root_subject: {row['root_subject']}")
        print(row["summary_text"])

    print("\n[Task3-View] Done.")


if __name__ == "__main__":
    main()
