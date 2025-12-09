#!/usr/bin/env python3
"""
Task 2a: Message Role Classification - TRAINING

- Reads:
    - messages_compact_all.csv (project root)
    - events_with_clusters_task1.csv (for event_ids)
- Creates:
    - tasks/task2/splits/*.txt  (event-level train/val/test split)
- Trains:
    - TF–IDF vectorizer
    - Multinomial logistic regression classifier
- Saves:
    - tasks/task2/models/task2_roles_tfidf.joblib
    - tasks/task2/models/task2_roles_logreg.joblib
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


THIS_FILE = Path(__file__).resolve()
TASK2_DIR = THIS_FILE.parent
ROOT = TASK2_DIR.parents[1]

MESSAGES_CSV = ROOT / "messages_compact_all.csv"
EVENTS_WITH_CLUSTERS = ROOT / "tasks" / "task1" / "events_with_clusters_task1.csv"

SPLIT_DIR = TASK2_DIR / "splits"
SPLIT_TRAIN = SPLIT_DIR / "split_train_event_ids.txt"
SPLIT_VAL = SPLIT_DIR / "split_val_event_ids.txt"
SPLIT_TEST = SPLIT_DIR / "split_test_event_ids.txt"

MODEL_DIR = TASK2_DIR / "models"
MODEL_VECT = MODEL_DIR / "task2_roles_tfidf.joblib"
MODEL_CLF = MODEL_DIR / "task2_roles_logreg.joblib"

MAX_TRAIN_PER_CLASS = {
    "PATCH": 300_000,
    "OTHER": 300_000,
    "REVIEW": None,
    "ACK": None,
}

RANDOM_STATE = 42


def ensure_dirs() -> None:
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def create_or_load_splits():
    """
    Create event-level train/val/test splits if they don't exist yet.
    Returns three sets of event_ids.
    """
    if SPLIT_TRAIN.is_file() and SPLIT_VAL.is_file() and SPLIT_TEST.is_file():
        print("[Task2-Roles-Train] Using existing splits.")
    else:
        print("[Task2-Roles-Train] Creating new event-level splits...")
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
            f"[Task2-Roles-Train] Saved splits: "
            f"{len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test events."
        )

    train_ids = set(np.loadtxt(SPLIT_TRAIN, dtype=str))
    val_ids = set(np.loadtxt(SPLIT_VAL, dtype=str))
    test_ids = set(np.loadtxt(SPLIT_TEST, dtype=str))
    return train_ids, val_ids, test_ids


def load_and_prepare_messages(train_ids, val_ids, test_ids):
    print(f"[Task2-Roles-Train] Loading messages from: {MESSAGES_CSV}")
    if not MESSAGES_CSV.is_file():
        raise FileNotFoundError(f"Cannot find {MESSAGES_CSV}")

    msgs = pd.read_csv(MESSAGES_CSV, low_memory=False)

    msgs["subject"] = msgs["subject"].fillna("")
    msgs["text_trunc"] = msgs["text_trunc"].fillna("")
    msgs["raw_text"] = (msgs["subject"] + " " + msgs["text_trunc"]).str.strip()
    msgs = msgs[msgs["raw_text"] != ""].copy()

    labels = ["PATCH", "REVIEW", "ACK", "OTHER"]
    msgs = msgs[msgs["weak_role"].isin(labels)].copy()

    def split_for_event(eid):
        if eid in train_ids:
            return "train"
        if eid in val_ids:
            return "val"
        if eid in test_ids:
            return "test"
        return "ignore"

    msgs["split"] = msgs["event_id"].astype(str).map(split_for_event)
    msgs = msgs[msgs["split"] != "ignore"].copy()

    print(
        "[Task2-Roles-Train] Messages per split:\n",
        msgs["split"].value_counts(),
    )

    return msgs


def sample_train_messages(msgs_train: pd.DataFrame) -> pd.DataFrame:
    """Downsample large classes in the training set to MAX_TRAIN_PER_CLASS."""
    groups = []
    for label, group_df in msgs_train.groupby("weak_role"):
        max_n = MAX_TRAIN_PER_CLASS.get(label)
        if max_n is not None and len(group_df) > max_n:
            group_df = group_df.sample(n=max_n, random_state=RANDOM_STATE)
        groups.append(group_df)
    sampled = pd.concat(groups, axis=0).sample(frac=1.0, random_state=RANDOM_STATE)
    print("[Task2-Roles-Train] After per-class sampling, train size:", len(sampled))
    print("[Task2-Roles-Train] Train label distribution:\n", sampled["weak_role"].value_counts())
    return sampled


def train_model(msgs: pd.DataFrame):
    msgs_train = msgs[msgs["split"] == "train"].copy()

    msgs_train = sample_train_messages(msgs_train)

    X_train_text = msgs_train["raw_text"].tolist()
    y_train = msgs_train["weak_role"].tolist()

    vect = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=50_000,
        min_df=5,
    )
    X_train = vect.fit_transform(X_train_text)

    clf = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        class_weight="balanced",
        max_iter=200,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    joblib.dump(vect, MODEL_VECT)
    joblib.dump(clf, MODEL_CLF)
    print(f"[Task2-Roles-Train] Saved vectorizer to {MODEL_VECT}")
    print(f"[Task2-Roles-Train] Saved classifier to {MODEL_CLF}")


def main():
    print(f"[Task2-Roles-Train] ROOT = {ROOT}")
    print(f"[Task2-Roles-Train] TASK2_DIR = {TASK2_DIR}")
    ensure_dirs()
    train_ids, val_ids, test_ids = create_or_load_splits()
    msgs = load_and_prepare_messages(train_ids, val_ids, test_ids)
    train_model(msgs)
    print("[Task2-Roles-Train] Done.")


if __name__ == "__main__":
    main()
