#!/usr/bin/env python3
"""
Task 2a: Message Role Classification - EVALUATION & PLOTTING

- Loads:
    - tasks/task2/splits/*.txt  (event-level splits)
    - tasks/task2/models/task2_roles_tfidf.joblib
    - tasks/task2/models/task2_roles_logreg.joblib
    - messages_compact_all.csv
- Computes:
    - classification report (val + test)
    - confusion matrix on test

- Saves:
    - figures/task2/roles_classification_report.txt
    - figures/task2/roles_confusion_matrix.png  (counts annotated;
      colors = row-normalized fractions of the true class)
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


THIS_FILE = Path(__file__).resolve()
TASK2_DIR = THIS_FILE.parent
ROOT = TASK2_DIR.parents[1]

MESSAGES_CSV = ROOT / "messages_compact_all.csv"

SPLIT_DIR = TASK2_DIR / "splits"
SPLIT_TRAIN = SPLIT_DIR / "split_train_event_ids.txt"
SPLIT_VAL = SPLIT_DIR / "split_val_event_ids.txt"
SPLIT_TEST = SPLIT_DIR / "split_test_event_ids.txt"

MODEL_DIR = TASK2_DIR / "models"
MODEL_VECT = MODEL_DIR / "task2_roles_tfidf.joblib"
MODEL_CLF = MODEL_DIR / "task2_roles_logreg.joblib"

FIG_DIR = ROOT / "figures" / "task2"
OUT_REPORT = FIG_DIR / "roles_classification_report.txt"

LABELS = ["PATCH", "OTHER", "REVIEW", "ACK"]


def ensure_dirs():
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_splits():
    if not (SPLIT_TRAIN.is_file() and SPLIT_VAL.is_file() and SPLIT_TEST.is_file()):
        raise FileNotFoundError(
            f"Split files not found in {SPLIT_DIR}. "
            f"Run the training script to create them."
        )

    train_ids = set(np.loadtxt(SPLIT_TRAIN, dtype=str))
    val_ids = set(np.loadtxt(SPLIT_VAL, dtype=str))
    test_ids = set(np.loadtxt(SPLIT_TEST, dtype=str))

    print("[Task2-Roles-Eval] Loaded splits:")
    print(f"  train events: {len(train_ids)}")
    print(f"  val events:   {len(val_ids)}")
    print(f"  test events:  {len(test_ids)}")

    return train_ids, val_ids, test_ids


def load_and_prepare_messages(train_ids, val_ids, test_ids):
    print(f"[Task2-Roles-Eval] Loading messages from: {MESSAGES_CSV}")
    if not MESSAGES_CSV.is_file():
        raise FileNotFoundError(f"Cannot find {MESSAGES_CSV}")

    msgs = pd.read_csv(MESSAGES_CSV, low_memory=False)

    msgs["subject"] = msgs["subject"].fillna("")
    msgs["text_trunc"] = msgs["text_trunc"].fillna("")
    msgs["raw_text"] = (msgs["subject"] + " " + msgs["text_trunc"]).str.strip()
    msgs = msgs[msgs["raw_text"] != ""].copy()

    msgs = msgs[msgs["weak_role"].isin(LABELS)].copy()

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
        "[Task2-Roles-Eval] Messages per split:\n",
        msgs["split"].value_counts(),
    )

    return msgs


def load_model():
    if not MODEL_VECT.is_file() or not MODEL_CLF.is_file():
        raise FileNotFoundError(
            f"Model files not found in {MODEL_DIR}. Run the training script first."
        )
    vect = joblib.load(MODEL_VECT)
    clf = joblib.load(MODEL_CLF)
    print("[Task2-Roles-Eval] Loaded vectorizer and classifier.")
    return vect, clf


def evaluate_and_report(msgs, vect, clf):
    msgs_val = msgs[msgs["split"] == "val"].copy()
    msgs_test = msgs[msgs["split"] == "test"].copy()

    X_val = vect.transform(msgs_val["raw_text"].tolist())
    y_val = msgs_val["weak_role"].tolist()

    X_test = vect.transform(msgs_test["raw_text"].tolist())
    y_test = msgs_test["weak_role"].tolist()

    y_val_pred = clf.predict(X_val)
    val_report = classification_report(
        y_val, y_val_pred, labels=LABELS, digits=3
    )
    print(f"\n[Task2-Roles-Eval] Classification report (val):\n{val_report}\n")

    y_test_pred = clf.predict(X_test)
    test_report = classification_report(
        y_test, y_test_pred, labels=LABELS, digits=3
    )
    print(f"\n[Task2-Roles-Eval] Classification report (test):\n{test_report}\n")

    with OUT_REPORT.open("w", encoding="utf-8") as f:
        f.write("=== Validation ===\n")
        f.write(val_report)
        f.write("\n\n=== Test ===\n")
        f.write(test_report)
    print(f"[Task2-Roles-Eval] Saved classification reports to {OUT_REPORT}")

    return y_test, y_test_pred


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig_path = FIG_DIR / "roles_confusion_matrix.png"
    plt.figure(figsize=(6, 5))

    im = plt.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    tick_labels = LABELS
    plt.xticks(range(len(tick_labels)), tick_labels, rotation=45, ha="right")
    plt.yticks(range(len(tick_labels)), tick_labels)

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(
        "Task 2a: Role classification confusion matrix\n"
        "Colors: row-normalized fractions, numbers: counts"
    )

    thresh = cm_norm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            value = cm_norm[i, j]
            color = "white" if value > thresh else "black"
            plt.text(
                j,
                i,
                f"{count}",
                ha="center",
                va="center",
                color=color,
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"[Task2-Roles-Eval] Saved confusion matrix to {fig_path}")


def main():
    print(f"[Task2-Roles-Eval] ROOT = {ROOT}")
    print(f"[Task2-Roles-Eval] TASK2_DIR = {TASK2_DIR}")

    ensure_dirs()
    train_ids, val_ids, test_ids = load_splits()
    msgs = load_and_prepare_messages(train_ids, val_ids, test_ids)
    vect, clf = load_model()
    y_test, y_test_pred = evaluate_and_report(msgs, vect, clf)
    plot_confusion_matrix(y_test, y_test_pred)
    print("[Task2-Roles-Eval] Done.")


if __name__ == "__main__":
    main()
