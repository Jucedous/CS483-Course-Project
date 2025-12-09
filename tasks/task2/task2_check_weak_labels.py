#!/usr/bin/env python3
"""
Quick sanity check for weak_role labels in messages_compact_all.csv.

For each role (PATCH / REVIEW / ACK), we measure how many messages actually
contain the expected keyword(s) in subject or text_trunc.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MESSAGES_CSV = ROOT / "messages_compact_all.csv"


def main():
    print(f"[Check-Weak] ROOT = {ROOT}")
    print(f"[Check-Weak] Reading {MESSAGES_CSV} ...")

    df = pd.read_csv(MESSAGES_CSV, low_memory=False)

    df["subject"] = df["subject"].fillna("").astype(str)
    df["text_trunc"] = df["text_trunc"].fillna("").astype(str)

    df["subject_lower"] = df["subject"].str.lower()
    df["text_lower"] = df["text_trunc"].str.lower()

    def frac_with(cond_series):
        return cond_series.sum() / max(len(cond_series), 1)

    patch = df[df["weak_role"] == "PATCH"].copy()
    patch_kw = (
        patch["subject_lower"].str.contains("patch", na=False)
        | patch["text_lower"].str.contains("patch", na=False)
    )
    print(f"\nPATCH messages: {len(patch)}")
    print(f"  fraction with 'patch' in subject/text: {frac_with(patch_kw):.3f}")

    ack = df[df["weak_role"] == "ACK"].copy()
    ack_kw = ack["text_lower"].str.contains("acked-by", na=False)
    print(f"\nACK messages: {len(ack)}")
    print(f"  fraction with 'acked-by' in text_trunc: {frac_with(ack_kw):.3f}")

    review = df[df["weak_role"] == "REVIEW"].copy()
    review_kw = review["text_lower"].str.contains("reviewed-by", na=False)
    print(f"\nREVIEW messages: {len(review)}")
    print(f"  fraction with 'reviewed-by' in text_trunc: {frac_with(review_kw):.3f}")

    print("\n[Check-Weak] Done.")


if __name__ == "__main__":
    main()
