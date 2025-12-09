"""
Task 1: More intuitive cluster visualizations.

Reads:
    events_with_clusters_task1.csv  (from task1_analysis.py)

Produces in figures/task1/:
    - length_box_by_cluster.png
    - depth_box_by_cluster.png
    - branching_box_by_cluster.png
    - long_fraction_by_cluster.png
    - nodes_vs_depth_clusters_jittered.png  (scatter, but with jitter)

These plots are meant to make it easy to describe clusters in words
(e.g., "Cluster 0 = tiny shallow threads", etc.).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parents[1]

EVENTS_WITH_CLUSTERS = ROOT / "events_with_clusters_task1.csv"
FIG_DIR = ROOT / "figures" / "task1"


def load_events_with_clusters():
    print(f"[Task1-Plots] Project root: {ROOT}")
    print(f"[Task1-Plots] Reading clustered events from: {EVENTS_WITH_CLUSTERS}")
    if not EVENTS_WITH_CLUSTERS.is_file():
        raise FileNotFoundError(
            f"Missing {EVENTS_WITH_CLUSTERS}. "
            f"Run tasks/task1_analysis.py first."
        )

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(EVENTS_WITH_CLUSTERS, low_memory=False)
    if "topology_cluster" not in df.columns:
        raise KeyError("Column 'topology_cluster' not found in events table.")
    return df


def plot_box_by_cluster(
    df: pd.DataFrame, column: str, ylabel: str, filename: str, logy: bool = False
):
    """
    Make a boxplot of a numeric column broken down by topology_cluster.
    """
    plt.figure()
    clusters = sorted(df["topology_cluster"].unique())

    data = [df.loc[df["topology_cluster"] == c, column].dropna() for c in clusters]

    plt.boxplot(data, labels=[str(c) for c in clusters], showfliers=False)
    plt.xlabel("Topology cluster")
    plt.ylabel(ylabel)
    title = f"Task 1: {ylabel} by cluster"
    plt.title(title)
    if logy:
        plt.yscale("log")
    plt.tight_layout()
    out_path = FIG_DIR / filename
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[Task1-Plots] Saved {title} -> {out_path}")


def plot_long_fraction(df: pd.DataFrame):
    """
    Bar chart: fraction of long threads (long_thread == 1) per cluster.
    """
    grouped = df.groupby("topology_cluster")["long_thread"].mean()
    clusters = grouped.index.tolist()
    fractions = grouped.values * 100.0

    plt.figure()
    plt.bar([str(c) for c in clusters], fractions)
    plt.xlabel("Topology cluster")
    plt.ylabel("Long threads (%)")
    plt.title("Task 1: Fraction of long threads by cluster")
    plt.tight_layout()
    out_path = FIG_DIR / "long_fraction_by_cluster.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[Task1-Plots] Saved fraction-of-long-threads plot -> {out_path}")


def plot_nodes_vs_depth_jittered(df: pd.DataFrame, max_points: int = 50_000):
    """
    Scatter plot of graph_num_nodes vs graph_depth, colored by cluster,
    but with a tiny vertical jitter on depth so the stripes are easier to see.
    """

    df = df[["graph_num_nodes", "graph_depth", "topology_cluster"]].dropna()

    if len(df) > max_points:
        df = df.sample(n=max_points, random_state=42)

    nodes = df["graph_num_nodes"].values
    depth = df["graph_depth"].values.astype(float)
    clusters = df["topology_cluster"].values

    jitter = np.random.normal(loc=0.0, scale=0.1, size=len(depth))
    depth_jittered = depth + jitter

    plt.figure()
    scatter = plt.scatter(nodes, depth_jittered, c=clusters, s=4, alpha=0.6)
    plt.xlabel("Number of nodes (graph_num_nodes)")
    plt.ylabel("Depth (graph_depth, jittered)")
    plt.title("Task 1: Nodes vs depth by cluster (with jitter)")

    handles, _ = scatter.legend_elements(prop="colors", num=None)
    labels = [f"Cluster {int(c)}" for c in sorted(np.unique(clusters))]
    plt.legend(handles, labels, title="Topology cluster", loc="best")

    plt.tight_layout()
    out_path = FIG_DIR / "nodes_vs_depth_clusters_jittered.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[Task1-Plots] Saved jittered nodes-vs-depth scatter -> {out_path}")


def main():
    df = load_events_with_clusters()

    plot_box_by_cluster(
        df,
        column="message_count",
        ylabel="Thread length (message_count)",
        filename="length_box_by_cluster.png",
        logy=True,
    )

    plot_box_by_cluster(
        df,
        column="graph_depth",
        ylabel="Graph depth",
        filename="depth_box_by_cluster.png",
        logy=False,
    )

    plot_box_by_cluster(
        df,
        column="graph_max_branching",
        ylabel="Max branching factor",
        filename="branching_box_by_cluster.png",
        logy=False,
    )

    plot_long_fraction(df)
    plot_nodes_vs_depth_jittered(df)

    print("[Task1-Plots] Done.")


if __name__ == "__main__":
    main()
