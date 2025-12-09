#!/usr/bin/env python3
"""
Task 3: Print one long / highly branched thread summary example.

- Reads:
    - tasks/task1/events_with_clusters_task1.csv
      (for structural stats: graph_depth, graph_max_branching, etc.)
    - tasks/task3/summaries_task3_with_roles.csv
      (summaries built by task3_build_summaries.py)

- Chooses:
    - a thread that is long_thread = 1
    - and has high graph_max_branching (i.e., lots of branching)

- Prints:
    - event_id, subject, structural stats (no participant count)
    - the generated summary_text
"""

from pathlib import Path
import textwrap

import pandas as pd


THIS_FILE = Path(__file__).resolve()
TASK3_DIR = THIS_FILE.parent
ROOT = TASK3_DIR.parents[1]

EVENTS_CSV = ROOT / "tasks" / "task1" / "events_with_clusters_task1.csv"
SUMMARIES_CSV = TASK3_DIR / "summaries_task3_with_roles.csv"


def load_data():
    if not EVENTS_CSV.is_file():
        raise FileNotFoundError(
            f"Cannot find {EVENTS_CSV}. "
            f"Run Task 1 to create events_with_clusters_task1.csv."
        )

    if not SUMMARIES_CSV.is_file():
        raise FileNotFoundError(
            f"Cannot find {SUMMARIES_CSV}. "
            f"Run task3_build_summaries.py first."
        )

    events = pd.read_csv(EVENTS_CSV, low_memory=False)
    summaries = pd.read_csv(SUMMARIES_CSV, low_memory=False)

    events["event_id"] = events["event_id"].astype(str)
    summaries["event_id"] = summaries["event_id"].astype(str)

    df = summaries.merge(
        events,
        on="event_id",
        how="inner",
        suffixes=("", "_event"),
    )

    return df


def pick_long_branchy_example(df):
    """
    Pick one thread that is long and has a lot of branching.
    Fallback to 'longest' thread if branching columns are missing.
    """
    candidates = df.copy()

    if "long_thread" in candidates.columns:
        candidates = candidates[candidates["long_thread"] == 1].copy()

    has_branch_cols = all(
        col in candidates.columns for col in ["graph_max_branching", "graph_depth", "graph_num_nodes"]
    )

    if has_branch_cols:
        candidates["graph_max_branching"] = candidates["graph_max_branching"].fillna(0)
        branchy = candidates[candidates["graph_max_branching"] >= 3].copy()
        if branchy.empty:
            branchy = candidates

        branchy = branchy.sort_values(
            ["graph_max_branching", "graph_depth", "graph_num_nodes"],
            ascending=[False, False, False],
        )
        example = branchy.iloc[0]
    else:
        candidates = candidates.sort_values(
            ["message_count", "num_participants"],
            ascending=[False, False],
        )
        example = candidates.iloc[0]

    return example


def print_example(row):
    event_id = row.get("event_id", "")
    subject = str(row.get("root_subject", row.get("root_subject_event", "")) or "")
    message_count = int(row.get("message_count", 0))
    long_thread = int(row.get("long_thread", 0))

    graph_num_nodes = row.get("graph_num_nodes", None)
    graph_depth = row.get("graph_depth", None)
    graph_max_branching = row.get("graph_max_branching", None)
    graph_avg_branching = row.get("graph_avg_branching", None)
    topology_cluster = row.get("topology_cluster", None)

    summary_text = str(row.get("summary_text", ""))

    print("=" * 80)
    print("Task 3: Example summary for a long / highly branched thread")
    print("=" * 80)
    print(f"event_id        : {event_id}")
    print(f"subject         : {subject}")
    print(f"long_thread     : {long_thread}")
    print(f"message_count   : {message_count}")

    if graph_num_nodes is not None:
        print(f"graph_num_nodes : {int(graph_num_nodes)}")
    if graph_depth is not None:
        print(f"graph_depth     : {int(graph_depth)}")
    if graph_max_branching is not None:
        print(f"graph_max_branch: {int(graph_max_branching)}")
    if graph_avg_branching is not None:
        print(f"graph_avg_branch: {graph_avg_branching:.2f}")
    if topology_cluster is not None:
        print(f"topology_cluster: {int(topology_cluster)}")

    print("-" * 80)
    print("Summary:")
    print("-" * 80)

    wrapper = textwrap.TextWrapper(width=78)
    for line in str(summary_text).splitlines():
        if not line.strip():
            print()
            continue
        wrapped = wrapper.fill(line)
        print(wrapped)

    print("=" * 80)


def main():
    print("[Task3-Long-Example] Loading data...")
    df = load_data()
    print(f"[Task3-Long-Example] Merged events + summaries: {len(df)} rows")

    example = pick_long_branchy_example(df)
    print_example(example)


if __name__ == "__main__":
    main()
