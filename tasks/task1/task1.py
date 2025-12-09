"""
Task 1: Discussion Dynamics and Graph Topology

- Loads event-level compact table: events_compact_all.csv
- Computes and plots:
    * Thread length distribution (message_count)
    * Thread duration distribution (if available)
    * k-means clusters over graph topology features
    * Scatter plots of size vs depth by cluster

Outputs:
    figures/task1/thread_length_hist.png
    figures/task1/thread_duration_hist.png  (only if durations exist)
    figures/task1/topology_clusters_nodes_depth.png
    figures/task1/topology_clusters_nodes_branching.png

    events_with_clusters_task1.csv (in project root)
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parents[1]

EVENTS_CSV = ROOT / "events_compact_all.csv"
FIG_DIR = ROOT / "figures" / "task1"
OUTPUT_EVENTS_WITH_CLUSTERS = ROOT / "events_with_clusters_task1.csv"

N_CLUSTERS = 4
SCATTER_MAX_POINTS = 50_000


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_events():
    print(f"[Task1] Loading events from: {EVENTS_CSV}")
    if not EVENTS_CSV.is_file():
        raise FileNotFoundError(f"Cannot find events CSV at {EVENTS_CSV}")

    events = pd.read_csv(EVENTS_CSV, low_memory=False)
    print(f"[Task1] Loaded {len(events):,} events")
    return events


def plot_thread_length(events: pd.DataFrame):
    """Plot histogram of thread lengths (message_count)."""
    fig_path = FIG_DIR / "thread_length_hist.png"
    lengths = events["message_count"].dropna()

    plt.figure()

    max_len = int(lengths.max())
    bins = np.unique(
        np.concatenate(
            [
                np.arange(0, min(50, max_len) + 1),
                np.linspace(50, max_len, num=50, dtype=int),
            ]
        )
    )
    plt.hist(lengths, bins=bins, edgecolor="black")
    plt.xlabel("Thread length (message_count)")
    plt.ylabel("Number of threads")
    plt.title("Task 1: Distribution of thread lengths")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"[Task1] Saved thread length histogram to {fig_path}")


def plot_thread_duration(events: pd.DataFrame):
    """Plot histogram of thread durations (hours), if we actually have them."""
    fig_path = FIG_DIR / "thread_duration_hist.png"

    if "duration_hours" not in events.columns:
        print("[Task1] duration_hours column not found; skipping duration plot")
        return

    durations = events["duration_hours"].dropna()
    if durations.empty:
        print("[Task1] duration_hours is empty; skipping duration plot")
        return

    plt.figure()

    upper = durations.quantile(0.99)
    clipped = durations[durations <= upper]

    bins = np.linspace(0, upper, num=50)
    plt.hist(clipped, bins=bins, edgecolor="black")
    plt.xlabel("Thread duration (hours)")
    plt.ylabel("Number of threads")
    plt.title("Task 1: Distribution of thread durations (clipped at 99th percentile)")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"[Task1] Saved thread duration histogram to {fig_path}")


def cluster_topology(events: pd.DataFrame):
    """
    Run k-means over topology features and save cluster labels.

    Uses features:
        - graph_num_nodes      (size)
        - graph_depth          (max depth)
        - graph_max_branching  (max out-degree)

    Returns:
        events_with_clusters (DataFrame)
    """

    cols = ["graph_num_nodes", "graph_depth", "graph_max_branching"]
    for c in cols:
        if c not in events.columns:
            raise KeyError(f"Column {c} not found in events table")

    topo = events[cols].fillna(0)

    print("[Task1] Topology feature stats before scaling:")
    print(topo.describe())

    scaler = StandardScaler()
    topo_scaled = scaler.fit_transform(topo.values)

    print(f"[Task1] Running KMeans with k={N_CLUSTERS} on {len(topo_scaled):,} events...")
    km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init="auto")
    cluster_labels = km.fit_predict(topo_scaled)

    events_with_clusters = events.copy()
    events_with_clusters["topology_cluster"] = cluster_labels

    print("[Task1] Cluster sizes (counts):")
    print(events_with_clusters["topology_cluster"].value_counts().sort_index())

    events_with_clusters.to_csv(OUTPUT_EVENTS_WITH_CLUSTERS, index=False)
    print(f"[Task1] Saved events with topology clusters to {OUTPUT_EVENTS_WITH_CLUSTERS}")

    return events_with_clusters, cluster_labels


def _sample_for_scatter(df: pd.DataFrame, max_points: int):
    """Subsample rows for plotting scatter to keep figure manageable."""
    if len(df) <= max_points:
        return df
    return df.sample(n=max_points, random_state=42)


def plot_topology_scatter(events_with_clusters: pd.DataFrame):
    """
    Produce scatter plots like:
        - num_nodes vs graph_depth (colored by cluster)
        - num_nodes vs graph_max_branching (colored by cluster)
    """

    mask = (
        (events_with_clusters["graph_num_nodes"] >= 0)
        & (events_with_clusters["graph_depth"] >= 0)
        & (events_with_clusters["graph_max_branching"] >= 0)
    )
    df = events_with_clusters[mask].copy()

    df["nodes_clipped"] = df["graph_num_nodes"].clip(
        upper=df["graph_num_nodes"].quantile(0.99)
    )
    df["depth_clipped"] = df["graph_depth"].clip(upper=df["graph_depth"].quantile(0.99))
    df["branch_clipped"] = df["graph_max_branching"].clip(
        upper=df["graph_max_branching"].quantile(0.99)
    )

    df_plot = _sample_for_scatter(df, SCATTER_MAX_POINTS)

    fig_path1 = FIG_DIR / "topology_clusters_nodes_depth.png"
    plt.figure()
    scatter = plt.scatter(
        df_plot["nodes_clipped"],
        df_plot["depth_clipped"],
        c=df_plot["topology_cluster"],
        s=5,
        alpha=0.6,
    )
    plt.xlabel("Number of nodes (graph_num_nodes, clipped)")
    plt.ylabel("Depth (graph_depth, clipped)")
    plt.title("Task 1: Topology clusters (nodes vs depth)")

    handles, _ = scatter.legend_elements(prop="colors", num=None)
    labels = [f"Cluster {i}" for i in range(events_with_clusters["topology_cluster"].nunique())]
    plt.legend(handles, labels, title="Topology cluster", loc="best")
    plt.tight_layout()
    plt.savefig(fig_path1, dpi=200)
    plt.close()
    print(f"[Task1] Saved topology scatter (nodes vs depth) to {fig_path1}")

    fig_path2 = FIG_DIR / "topology_clusters_nodes_branching.png"
    plt.figure()
    scatter2 = plt.scatter(
        df_plot["nodes_clipped"],
        df_plot["branch_clipped"],
        c=df_plot["topology_cluster"],
        s=5,
        alpha=0.6,
    )
    plt.xlabel("Number of nodes (graph_num_nodes, clipped)")
    plt.ylabel("Max branching factor (graph_max_branching, clipped)")
    plt.title("Task 1: Topology clusters (nodes vs max branching)")
    handles2, _ = scatter2.legend_elements(prop="colors", num=None)
    labels2 = [f"Cluster {i}" for i in range(events_with_clusters['topology_cluster'].nunique())]
    plt.legend(handles2, labels2, title="Topology cluster", loc="best")
    plt.tight_layout()
    plt.savefig(fig_path2, dpi=200)
    plt.close()
    print(f"[Task1] Saved topology scatter (nodes vs branching) to {fig_path2}")


def main():
    print(f"[Task1] Project root: {ROOT}")
    ensure_dirs()

    events = load_events()

    plot_thread_length(events)
    plot_thread_duration(events)

    events_with_clusters, _ = cluster_topology(events)
    plot_topology_scatter(events_with_clusters)

    print("[Task1] Done.")


if __name__ == "__main__":
    main()
