"""
Task 1: Topology cluster summary

Reads:
    events_with_clusters_task1.csv  (from Task 1 analysis)

Computes, per topology cluster:
    - number of threads
    - fraction of all threads
    - mean / median graph_num_nodes
    - mean / median graph_depth
    - mean graph_max_branching
    - mean / median message_count
    - fraction of long threads (long_thread == 1)

Outputs:
    - prints a nice table to stdout
    - saves CSV to figures/task1/topology_cluster_summary.csv
"""

from pathlib import Path

import pandas as pd


THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parents[1]

EVENTS_WITH_CLUSTERS = ROOT / "events_with_clusters_task1.csv"
FIG_DIR = ROOT / "figures" / "task1"
SUMMARY_CSV = FIG_DIR / "topology_cluster_summary.csv"


def main():
    print(f"[Task1] Project root: {ROOT}")
    print(f"[Task1] Reading clustered events from: {EVENTS_WITH_CLUSTERS}")

    if not EVENTS_WITH_CLUSTERS.is_file():
        raise FileNotFoundError(
            f"Missing {EVENTS_WITH_CLUSTERS}. "
            f"Run tasks/task1_analysis.py first to generate it."
        )

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(EVENTS_WITH_CLUSTERS, low_memory=False)

    if "topology_cluster" not in df.columns:
        raise KeyError("Column 'topology_cluster' not found in events_with_clusters_task1.csv")

    total = len(df)
    print(f"[Task1] Total events: {total:,}")

    group = df.groupby("topology_cluster")

    summary = group.agg(
        n_threads=("event_id", "count"),
        mean_nodes=("graph_num_nodes", "mean"),
        median_nodes=("graph_num_nodes", "median"),
        mean_depth=("graph_depth", "mean"),
        median_depth=("graph_depth", "median"),
        mean_max_branch=("graph_max_branching", "mean"),
        mean_msg_count=("message_count", "mean"),
        median_msg_count=("message_count", "median"),
        long_frac=("long_thread", "mean"),
    )

    summary["cluster_frac"] = summary["n_threads"] / total

    summary = summary[
        [
            "n_threads",
            "cluster_frac",
            "mean_nodes",
            "median_nodes",
            "mean_depth",
            "median_depth",
            "mean_max_branch",
            "mean_msg_count",
            "median_msg_count",
            "long_frac",
        ]
    ]

    summary = summary.sort_index()

    print("\n[Task1] Topology cluster summary:\n")

    display = summary.copy()
    display["cluster_frac"] = (display["cluster_frac"] * 100).round(2)
    display["long_frac"] = (display["long_frac"] * 100).round(2)
    for col in [
        "mean_nodes",
        "median_nodes",
        "mean_depth",
        "median_depth",
        "mean_max_branch",
        "mean_msg_count",
        "median_msg_count",
    ]:
        display[col] = display[col].round(2)

    print(display)

    summary.to_csv(SUMMARY_CSV, index=True)
    print(f"\n[Task1] Saved raw summary to {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
