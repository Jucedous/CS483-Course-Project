"""
Task 2b: Debate Trajectory Prediction (text-only)

- Label: long_thread ∈ {0, 1} from tasks/task1/events_with_clusters_task1.csv
- Input: root_text = subject + body of the root message (from messages_compact_all.csv)
- Model: Logistic regression on TF–IDF features.

Outputs (under project_root/figures/task2/):
    - debate_text_only_roc.png
    - debate_text_only_pr.png
    - debate_text_only_metrics.txt

Splits are stored under tasks/task2/splits/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split


THIS_FILE = Path(__file__).resolve()
TASK2_DIR = THIS_FILE.parent
ROOT = TASK2_DIR.parents[1]

EVENTS_WITH_CLUSTERS = ROOT / "tasks" / "task1" / "events_with_clusters_task1.csv"
MESSAGES_CSV = ROOT / "messages_compact_all.csv"

SPLIT_DIR = TASK2_DIR / "splits"
SPLIT_TRAIN = SPLIT_DIR / "split_train_event_ids.txt"
SPLIT_VAL = SPLIT_DIR / "split_val_event_ids.txt"
SPLIT_TEST = SPLIT_DIR / "split_test_event_ids.txt"

FIG_DIR = ROOT / "figures" / "task2"
OUT_METRICS = FIG_DIR / "debate_text_only_metrics.txt"


MAX_TRAIN_EVENTS = 200_000
RANDOM_STATE = 42


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)


def create_or_load_splits():
    """
    Create event-level train/val/test splits if they don't exist yet.
    Returns three sets of event_ids.
    """
    if SPLIT_TRAIN.is_file() and SPLIT_VAL.is_file() and SPLIT_TEST.is_file():
        print("[Task2-Debate] Using existing splits.")
    else:
        print("[Task2-Debate] Creating new event-level splits...")
        if not EVENTS_WITH_CLUSTERS.is_file():
            raise FileNotFoundError(
                f"Cannot find {EVENTS_WITH_CLUSTERS}. Run Task 1 first."
            )
        events_df = pd.read_csv(EVENTS_WITH_CLUSTERS, usecols=["event_id"], low_memory=False)
        event_ids = events_df["event_id"].unique()
        train_ids, temp_ids = train_test_split(
            event_ids, test_size=0.30, random_state=RANDOM_STATE
        )
        val_ids, test_ids = train_test_split(
            temp_ids, test_size=0.50, random_state=RANDOM_STATE
        )

        np.savetxt(SPLIT_TRAIN, train_ids, fmt="%s")
        np.savetxt(SPLIT_VAL, val_ids, fmt="%s")
        np.savetxt(SPLIT_TEST, test_ids, fmt="%s")
        print(
            f"[Task2-Debate] Saved splits: "
            f"{len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test events."
        )

    train_ids = set(np.loadtxt(SPLIT_TRAIN, dtype=str))
    val_ids = set(np.loadtxt(SPLIT_VAL, dtype=str))
    test_ids = set(np.loadtxt(SPLIT_TEST, dtype=str))
    return train_ids, val_ids, test_ids


def load_events_and_root_text():
    """
    Load event-level labels and root message text for each event_id.
    """
    print(f"[Task2-Debate] Reading events from {EVENTS_WITH_CLUSTERS}")
    events = pd.read_csv(EVENTS_WITH_CLUSTERS, low_memory=False)

    if "long_thread" not in events.columns:
        raise KeyError("Column 'long_thread' not found in events_with_clusters_task1.csv")

    print(f"[Task2-Debate] Reading messages from {MESSAGES_CSV}")
    msgs = pd.read_csv(
        MESSAGES_CSV,
        usecols=["event_id", "msg_index", "is_root", "subject", "text_trunc"],
        low_memory=False,
    )

    msgs["is_root_flag"] = msgs["is_root"].fillna(0).astype(int)
    root_msgs = msgs[msgs["is_root_flag"] == 1].copy()
    if root_msgs.empty:
        root_msgs = msgs[msgs["msg_index"] == 0].copy()

    root_msgs["subject"] = root_msgs["subject"].fillna("")
    root_msgs["text_trunc"] = root_msgs["text_trunc"].fillna("")
    root_msgs["root_text"] = (root_msgs["subject"] + " " + root_msgs["text_trunc"]).str.strip()

    root_msgs = root_msgs[["event_id", "root_text"]].drop_duplicates(subset=["event_id"])
    print(f"[Task2-Debate] Found root text for {len(root_msgs):,} events")

    events = events.merge(root_msgs, on="event_id", how="left")
    events["root_text"] = events["root_text"].fillna("")
    events = events[events["root_text"] != ""].copy()
    print(f"[Task2-Debate] Events with non-empty root_text: {len(events):,}")

    return events


def split_events(events, train_ids, val_ids, test_ids):
    def split_for_event(eid: str) -> str:
        if eid in train_ids:
            return "train"
        if eid in val_ids:
            return "val"
        if eid in test_ids:
            return "test"
        return "ignore"

    events["split"] = events["event_id"].astype(str).map(split_for_event)
    events = events[events["split"] != "ignore"].copy()
    print("[Task2-Debate] Events per split:\n", events["split"].value_counts())
    return events


def train_logreg_binary(X_train, y_train):
    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=200,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf


def eval_binary_model(
    name,
    clf,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    metrics_file,
):
    results = []

    for split_name, X, y_true in [
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        prob = clf.predict_proba(X)[:, 1]
        auc = roc_auc_score(y_true, prob)
        ap = average_precision_score(y_true, prob)

        thresholds = np.linspace(0.1, 0.9, 17)
        best_f1 = -1.0
        best_thr = 0.5
        for thr in thresholds:
            y_pred = (prob >= thr).astype(int)
            report = classification_report(
                y_true, y_pred, labels=[0, 1], target_names=["short", "long"], digits=3
            )
            lines = report.splitlines()
            try:
                long_line = [ln for ln in lines if ln.strip().startswith("long")][0]
                f1_long = float(long_line.split()[-2])
            except Exception:
                f1_long = 0.0
            if f1_long > best_f1:
                best_f1 = f1_long
                best_thr = thr

        results.append(
            f"[{name}] {split_name}: AUC={auc:.3f}, AP={ap:.3f}, "
            f"best_F1_long={best_f1:.3f} at thr={best_thr:.2f}"
        )

        if split_name == "test":
            prob_test = clf.predict_proba(X_test)[:, 1]
            y_pred_best = (prob_test >= best_thr).astype(int)
            report_test = classification_report(
                y_test, y_pred_best, labels=[0, 1], target_names=["short", "long"], digits=3
            )
            results.append(f"\n[{name}] Test classification report (thr={best_thr:.2f}):\n")
            results.append(report_test)

    prob_test = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, prob_test)
    prec, rec, _ = precision_recall_curve(y_test, prob_test)

    roc_path = FIG_DIR / f"debate_{name}_roc.png"
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Task 2b: ROC curve ({name})")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=200)
    plt.close()

    pr_path = FIG_DIR / f"debate_{name}_pr.png"
    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Task 2b: Precision-Recall curve ({name})")
    plt.tight_layout()
    plt.savefig(pr_path, dpi=200)
    plt.close()

    results.append(f"[{name}] ROC curve saved to {roc_path}")
    results.append(f"[{name}] PR curve saved to {pr_path}")

    with metrics_file.open("a", encoding="utf-8") as f:
        f.write("\n".join(results))
        f.write("\n\n")


def main():
    print(f"[Task2-Debate] ROOT = {ROOT}")
    print(f"[Task2-Debate] TASK2_DIR = {TASK2_DIR}")
    ensure_dirs()
    train_ids, val_ids, test_ids = create_or_load_splits()

    events = load_events_and_root_text()
    events = split_events(events, train_ids, val_ids, test_ids)

    y_all = events["long_thread"].astype(int).values
    splits = events["split"].values
    root_text_all = events["root_text"].tolist()

    idx_train = np.where(splits == "train")[0]
    idx_val = np.where(splits == "val")[0]
    idx_test = np.where(splits == "test")[0]

    train_indices = idx_train.copy()
    if MAX_TRAIN_EVENTS is not None and len(train_indices) > MAX_TRAIN_EVENTS:
        rng = np.random.RandomState(RANDOM_STATE)
        train_indices = rng.choice(train_indices, size=MAX_TRAIN_EVENTS, replace=False)
        print(
            f"[Task2-Debate] Subsampled train events for text-only model "
            f"to {len(train_indices)}"
        )

    X_train_text_raw = np.array(root_text_all, dtype=object)[train_indices]
    y_train = y_all[train_indices]

    X_val_text_raw = np.array(root_text_all, dtype=object)[idx_val]
    y_val = y_all[idx_val]

    X_test_text_raw = np.array(root_text_all, dtype=object)[idx_test]
    y_test = y_all[idx_test]

    vect = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=50_000,
        min_df=5,
    )
    X_train = vect.fit_transform(X_train_text_raw)
    X_val = vect.transform(X_val_text_raw)
    X_test = vect.transform(X_test_text_raw)

    clf = train_logreg_binary(X_train, y_train)

    OUT_METRICS.write_text("", encoding="utf-8")
    eval_binary_model(
        name="text_only",
        clf=clf,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        metrics_file=OUT_METRICS,
    )

    print(f"[Task2-Debate] Metrics written to {OUT_METRICS}")
    print("[Task2-Debate] Done.")


if __name__ == "__main__":
    main()
