# LKML Thread Analysis: Dynamics, Role Modeling, and Summarization

This repository implements an end-to-end pipeline for analyzing Linux Kernel Mailing List (LKML) discussion threads. It is organized as three tasks built on top of a preprocessing stage that converts raw LKML JSON dumps into compact, analysis-friendly tables.

## What this project does

Given raw LKML thread data:

1. **Preprocess / Compact the dataset**
   - Convert nested JSON threads into two flat tables:
     - **Event-level** (one row per thread)
     - **Message-level** (one row per email)

2. **Task 1 — Discussion dynamics + topology clustering**
   - Characterize thread length and duration distributions
   - Build reply-graph features (depth, branching, etc.)
   - Cluster threads by topology (K-means) and analyze cluster properties

3. **Task 2 — Text-based prediction**
   - **Task 2a:** Predict a message “role” (PATCH / REVIEW / ACK / OTHER) from text using supervision
   - **Task 2b:** Predict whether a thread becomes a *long debate* from the root email text only

4. **Task 3 — Role-informed extractive summarization**
   - Generate structured extractive summaries using predicted roles + lightweight keyword heuristics

---

## Expected data layout

The code assumes the raw LKML data is available locally in two directories:

- `0_event_json_data/` — thread/event JSON files (each corresponds to one discussion thread)
- `1_series_json_data/` — series JSON files (optional for Tasks 1–3; used for inspection / potential extensions)

Each event JSON is expected to contain (at minimum) thread metadata, a list of messages, and reply `connections`.

> If your raw data lives elsewhere, you may need to adjust the path constants in the scripts (see **Troubleshooting**).

---

## Environment / dependencies

Python 3 is required. Typical dependencies used across the pipeline include:

- `numpy`, `pandas`
- `scikit-learn`
- `matplotlib`
- `joblib`
- `tqdm` (if enabled in your environment)

If you run into import errors, install missing packages via:
```bash
pip install numpy pandas scikit-learn matplotlib joblib tqdm
