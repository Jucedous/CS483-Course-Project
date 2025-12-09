#!/usr/bin/env python3
"""
Task 3: Thread Summarization (Extractive, Using Task 2 Roles)

Reads:
    - tasks/task1/events_with_clusters_task1.csv
    - messages_compact_all.csv
    - tasks/task2/models/task2_roles_tfidf.joblib
    - tasks/task2/models/task2_roles_logreg.joblib

For a subset of threads (prioritizing long debates), it:
    - predicts message roles with the Task 2a model,
    - builds simple extractive summaries:

        * Proposal    (from root email)
        * Review      (from messages predicted as REVIEW / review-like)
        * Outcome     (from messages predicted as ACK / outcome keywords)
        * Discussion evolution (extra points showing how the thread progressed)

Outputs:
    - tasks/task3/summaries_task3_with_roles.csv
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


THIS_FILE = Path(__file__).resolve()
TASK3_DIR = THIS_FILE.parent
ROOT = TASK3_DIR.parents[1]

EVENTS_CSV = ROOT / "tasks" / "task1" / "events_with_clusters_task1.csv"
MESSAGES_CSV = ROOT / "messages_compact_all.csv"


TASK2_MODELS_DIR = ROOT / "tasks" / "task2" / "models"
MODEL_VECT = TASK2_MODELS_DIR / "task2_roles_tfidf.joblib"
MODEL_CLF = TASK2_MODELS_DIR / "task2_roles_logreg.joblib"

OUT_CSV = TASK3_DIR / "summaries_task3_with_roles.csv"
FIG_DIR = ROOT / "figures" / "task3"

MAX_THREADS = 5000
MIN_MESSAGES = 3

RANDOM_STATE = 42

POSITIVE_OUTCOME_KEYWORDS = [
    "acked-by",
    "reviewed-by",
    "tested-by",
    "applied",
    "queued",
    "merged",
    "pulled",
]

NEGATIVE_OUTCOME_KEYWORDS = [
    "nak",
    "nacked-by",
    "rejected",
    "reject",
    "revert",
]

REVIEW_HINT_KEYWORDS = [
    "reviewed-by",
    "comment",
    "suggest",
    "issue",
    "fix",
    "problem",
    "concern",
    "nit",
    "typo",
    "improve",
    "cleanup",
]

PROGRESSION_HINT_KEYWORDS = [
    "v2",
    "v3",
    "v4",
    "resend",
    "re-spin",
    "respinned",
    "updated patch",
    "new version",
    "addressed",
    "fixed",
    "fixes",
    "after discussion",
    "follow-up",
    "follow up",
]


def ensure_dirs():
    TASK3_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_events():
    print(f"[Task3-Build] Reading events from {EVENTS_CSV}")
    if not EVENTS_CSV.is_file():
        raise FileNotFoundError(f"Cannot find {EVENTS_CSV}")
    events = pd.read_csv(EVENTS_CSV, low_memory=False)

    required_cols = ["event_id", "root_subject", "message_count", "num_participants"]
    for c in required_cols:
        if c not in events.columns:
            raise KeyError(f"Missing column in events file: {c}")

    events["event_id"] = events["event_id"].astype(str)

    if "long_thread" not in events.columns:
        events["long_thread"] = (events["message_count"] >= 8).astype(int)

    events = events[events["message_count"] >= MIN_MESSAGES].copy()

    events = events.sort_values(
        ["long_thread", "message_count"], ascending=[False, False]
    )

    if len(events) > MAX_THREADS:
        print(
            f"[Task3-Build] Restricting to top {MAX_THREADS} threads "
            f"by long_thread + size (from {len(events)})"
        )
        events = events.head(MAX_THREADS)

    print(f"[Task3-Build] Will summarize {len(events)} threads.")
    return events


def load_messages_for_events(event_ids):
    print(f"[Task3-Build] Reading messages from {MESSAGES_CSV}")
    if not MESSAGES_CSV.is_file():
        raise FileNotFoundError(f"Cannot find {MESSAGES_CSV}")

    msgs = pd.read_csv(
        MESSAGES_CSV,
        usecols=[
            "event_id",
            "msg_index",
            "is_root",
            "subject",
            "text_trunc",
            "weak_role",
        ],
        low_memory=False,
    )
    msgs["event_id"] = msgs["event_id"].astype(str)
    msgs = msgs[msgs["event_id"].isin(event_ids)].copy()

    msgs["msg_index"] = msgs["msg_index"].fillna(0).astype(int)
    msgs["is_root"] = msgs["is_root"].fillna(0).astype(int)
    msgs["subject"] = msgs["subject"].fillna("")
    msgs["text_trunc"] = msgs["text_trunc"].fillna("")
    msgs["weak_role"] = msgs["weak_role"].fillna("OTHER")

    msgs = msgs.sort_values(["event_id", "msg_index"])
    print(f"[Task3-Build] Loaded {len(msgs)} messages for {len(event_ids)} events.")
    return msgs


def load_role_model():
    if not MODEL_VECT.is_file() or not MODEL_CLF.is_file():
        raise FileNotFoundError(
            f"Task 2a model files not found in {TASK2_MODELS_DIR}. "
            f"Run task2_roles_train.py first."
        )
    vect = joblib.load(MODEL_VECT)
    clf = joblib.load(MODEL_CLF)
    print("[Task3-Build] Loaded Task 2a vectorizer and classifier.")
    return vect, clf


def predict_roles_for_messages(msgs, vect, clf):
    """
    Use Task 2a model to predict roles for each message.
    Adds a 'pred_role' column to msgs.
    """
    print("[Task3-Build] Predicting roles for messages using Task 2a model...")
    raw_text = (
        msgs["subject"].fillna("") + " " + msgs["text_trunc"].fillna("")
    ).str.strip()

    X = vect.transform(raw_text.tolist())
    preds = clf.predict(X)

    msgs = msgs.copy()
    msgs["pred_role"] = preds
    print("[Task3-Build] Role prediction done. Distribution:")
    print(msgs["pred_role"].value_counts())
    return msgs


def clean_lines(text):
    """
    Split text into lines, strip whitespace, and drop quoted or empty lines.
    """
    if not isinstance(text, str):
        return []
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if s.startswith(">"):
            continue
        cleaned.append(s)
    return cleaned


def summarize_proposal(root_subject, root_text):
    """
    Build a short 'Proposal' snippet from the root email.
    """
    lines = clean_lines(root_text)
    if not lines:
        proposal = root_subject or ""
    else:
        proposal = " ".join(lines[:2])
    proposal = proposal.strip()
    if len(proposal) > 300:
        proposal = proposal[:297] + "..."
    return proposal


def pick_review_sentences(thread_msgs, max_reviews=3):
    """
    Pick up to max_reviews short review snippets from messages
    predicted as REVIEW or containing review-like hints.
    """
    review_snippets = []
    seen = set()

    for _, row in thread_msgs.iterrows():
        role = str(row.get("pred_role", "OTHER"))
        text = str(row.get("text_trunc", ""))
        text_lower = text.lower()

        is_reviewish = role == "REVIEW" or any(k in text_lower for k in REVIEW_HINT_KEYWORDS)
        if not is_reviewish:
            continue

        lines = clean_lines(text)
        if not lines:
            continue

        first_line = lines[0]
        if first_line in seen:
            continue

        review_snippets.append(first_line)
        seen.add(first_line)
        if len(review_snippets) >= max_reviews:
            break

    return review_snippets


def pick_progress_sentences(thread_msgs, max_points=2):
    """
    Pick up to max_points sentences that indicate how the discussion evolved
    (e.g., new versions, fixes, follow-ups).
    """
    progression_snippets = []
    seen = set()

    for _, row in thread_msgs.iterrows():
        text = str(row.get("text_trunc", ""))
        lines = clean_lines(text)
        if not lines:
            continue

        for ln in lines:
            ln_lower = ln.lower()
            if any(k in ln_lower for k in PROGRESSION_HINT_KEYWORDS):
                if ln in seen:
                    continue
                progression_snippets.append(ln)
                seen.add(ln)
                if len(progression_snippets) >= max_points:
                    return progression_snippets
                break

    return progression_snippets


def infer_outcome(thread_msgs):
    """
    Scan messages from last to first to infer an outcome from predicted roles
    and keywords. Returns (outcome_label, outcome_text).
    """
    for _, row in thread_msgs.sort_values("msg_index", ascending=False).iterrows():
        text = str(row.get("text_trunc", ""))
        lines = clean_lines(text)
        if not lines:
            continue

        text_lower = text.lower()
        role = str(row.get("pred_role", "OTHER"))

        has_pos = any(k in text_lower for k in POSITIVE_OUTCOME_KEYWORDS)
        has_neg = any(k in text_lower for k in NEGATIVE_OUTCOME_KEYWORDS)

        outcome_line = None
        for ln in lines:
            ln_lower = ln.lower()
            if any(k in ln_lower for k in POSITIVE_OUTCOME_KEYWORDS + NEGATIVE_OUTCOME_KEYWORDS):
                outcome_line = ln
                break
        if outcome_line is None:
            outcome_line = lines[0]

        if has_pos or role == "ACK":
            return "likely accepted or queued", outcome_line
        if has_neg:
            return "likely rejected or contentious", outcome_line

    return "outcome unclear", "Outcome unclear from thread (no explicit ACK/NAK found)."


def summarize_thread(event_row, thread_msgs):
    """
    Build a summary string for a single event_id using predicted roles.
    """
    root_subject = str(event_row.get("root_subject", "")) or ""

    roots = thread_msgs[thread_msgs["is_root"] == 1]
    if roots.empty:
        roots = thread_msgs[thread_msgs["msg_index"] == 0]
    if roots.empty:
        return ""

    root_msg = roots.iloc[0]
    root_text = str(root_msg.get("text_trunc", ""))

    proposal = summarize_proposal(root_subject, root_text)
    review_snippets = pick_review_sentences(thread_msgs, max_reviews=3)
    progression_snippets = pick_progress_sentences(thread_msgs, max_points=2)
    outcome_label, outcome_line = infer_outcome(thread_msgs)

    parts = []
    parts.append(f"Thread: {root_subject}")
    parts.append(f"Length: {int(event_row.get('message_count', 0))} messages.")
    parts.append(f"Proposal: {proposal}")

    if review_snippets:
        parts.append("Key review points:")
        for s in review_snippets:
            parts.append(f"- {s}")
    else:
        parts.append("Key review points: (no explicit review comments extracted)")

    if progression_snippets:
        parts.append("How the discussion evolved:")
        for s in progression_snippets:
            parts.append(f"- {s}")

    parts.append(f"Outcome: {outcome_label} — {outcome_line}")

    return "\n".join(parts)


def main():
    print(f"[Task3-Build] ROOT = {ROOT}")
    print(f"[Task3-Build] TASK3_DIR = {TASK3_DIR}")
    ensure_dirs()

    events = load_events()
    event_ids = events["event_id"].tolist()

    msgs = load_messages_for_events(event_ids)

    vect, clf = load_role_model()
    msgs = predict_roles_for_messages(msgs, vect, clf)

    grouped = msgs.groupby("event_id")

    summaries = []
    missing_threads = 0

    for _, ev in events.iterrows():
        eid = ev["event_id"]
        if eid not in grouped.groups:
            missing_threads += 1
            continue
        thread_msgs = grouped.get_group(eid)
        summary_text = summarize_thread(ev, thread_msgs)
        summaries.append(
            {
                "event_id": eid,
                "root_subject": ev.get("root_subject", ""),
                "message_count": int(ev.get("message_count", 0)),
                "num_participants": int(ev.get("num_participants", 0)),
                "long_thread": int(ev.get("long_thread", 0)),
                "summary_text": summary_text,
            }
        )

    if missing_threads > 0:
        print(f"[Task3-Build] Warning: {missing_threads} events had no messages.")

    summaries_df = pd.DataFrame(summaries)
    summaries_df.to_csv(OUT_CSV, index=False)
    print(f"[Task3-Build] Wrote {len(summaries_df)} summaries to {OUT_CSV}")


if __name__ == "__main__":
    main()
