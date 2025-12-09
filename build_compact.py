"""
Parallel compaction of LKML event JSONs for Tasks 1–3.

- Input:
    0_event_json_data/*.json
- Output:
    events_compact_all.csv    (one row per thread/event, with graph features)
    messages_compact_all.csv  (one row per message/email, truncated text)

Run from the directory that contains 0_event_json_data:

    cd ~/all
    python3 build_compact.py
"""

import csv
import json
import os
from pathlib import Path
import re
from email.utils import parsedate_to_datetime
import multiprocessing as mp
from datetime import datetime

# ========================= CONFIG ========================= #

ROOT = Path(os.getcwd())
EVENT_DIR = ROOT / "0_event_json_data"

OUTPUT_EVENTS   = ROOT / "events_compact_all.csv"
OUTPUT_MESSAGES = ROOT / "messages_compact_all.csv"

# Process ALL events (no year filter, no max)
# Set TARGET_YEARS = {2018, 2019} if you ever want to restrict.
TARGET_YEARS = None

# For testing, you can set this to a small number, e.g. 1000.
# Set to None for no limit.
MAX_EVENTS = None

# Truncate message text so messages_compact_all.csv doesn't explode in size.
TRUNCATE_TEXT_CHARS = 1000

# Number of worker processes. Feel free to tune.
N_WORKERS = max(1, (mp.cpu_count() or 2) - 1)

# ========================================================== #


def parse_year_from_event_id(event_id: str):
    """Extract year from event_id like 'lkml_2018_9_12_1234'."""
    if not event_id:
        return None
    m = re.match(r"lkml_(\d{4})_", event_id)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


def get_first_nonempty(d, keys, default=""):
    """Return d[k] for the first k in keys that exists and is non-empty."""
    for k in keys:
        if k in d and d[k]:
            return d[k]
    return default


def weak_role_label(subject: str, body: str) -> str:
    """
    Very simple heuristic role labeling.
    Replace/extend this with your more careful rules from the midterm.
    """
    subj = (subject or "").lower()
    text = (body or "").lower()

    if "[patch" in subj or subj.startswith("patch "):
        return "PATCH"
    if "acked-by:" in text:
        return "ACK"
    if "reviewed-by:" in text or "comments below" in text:
        return "REVIEW"
    return "OTHER"


def parse_timestamp_generic(ts_raw):
    """
    Try to parse a timestamp that may be:
      - email-style date string
      - numeric Unix timestamp (sec)
      - numeric Unix timestamp string
    Returns datetime or None.
    """
    if ts_raw is None or ts_raw == "":
        return None

    # Already numeric
    if isinstance(ts_raw, (int, float)):
        try:
            return datetime.fromtimestamp(float(ts_raw))
        except Exception:
            return None

    # String
    if isinstance(ts_raw, str):
        s = ts_raw.strip()
        # numeric string -> epoch
        if s.isdigit():
            try:
                return datetime.fromtimestamp(float(s))
            except Exception:
                pass
        # fallback: email-style
        try:
            return parsedate_to_datetime(s)
        except Exception:
            return None

    return None


def extract_edge_uv(conn):
    """
    Try to extract (u, v) from a connection item.

    Supports:
      - list/tuple: [u, v]
      - dict with common key pairs: (src, dst), (source, target),
        (from, to), (parent, child), (u, v)
    """
    if isinstance(conn, (list, tuple)) and len(conn) >= 2:
        return conn[0], conn[1]

    if isinstance(conn, dict):
        candidate_pairs = [
            ("src", "dst"),
            ("source", "target"),
            ("from", "to"),
            ("parent", "child"),
            ("u", "v"),
        ]
        for fk, tk in candidate_pairs:
            if fk in conn and tk in conn:
                return conn[fk], conn[tk]

    return None, None


def compute_graph_features(num_messages, connections):
    """
    Compute simple graph features from connections.

    Accepts ANY hashable node ID (strings, ints, etc.).
    We map them internally to integer indices.

    Returns:
      num_nodes, num_edges, depth, max_branching, avg_branching
    """
    if not connections:
        # no connections; treat as edgeless graph
        num_nodes = num_messages
        return num_nodes, 0, 0, 0, 0.0

    node_index = {}
    next_idx = 0
    edges = []

    for conn in connections:
        u_raw, v_raw = extract_edge_uv(conn)
        if u_raw is None or v_raw is None:
            continue

        for node in (u_raw, v_raw):
            if node not in node_index:
                node_index[node] = next_idx
                next_idx += 1
        u = node_index[u_raw]
        v = node_index[v_raw]
        edges.append((u, v))

    if not edges:
        # connections exist but we couldn't parse them
        num_nodes = max(num_messages, len(node_index))
        return num_nodes, 0, 0, 0, 0.0

    # use nodes seen in edges; also consider num_messages
    num_nodes = max(num_messages, len(node_index))

    adj = {i: [] for i in range(len(node_index))}
    indegree = [0] * len(node_index)
    for u, v in edges:
        adj[u].append(v)
        indegree[v] += 1

    edge_count = len(edges)

    # choose root as any node with indegree 0 (fallback to 0)
    root = 0
    for i in range(len(node_index)):
        if indegree[i] == 0:
            root = i
            break

    # BFS to compute depths
    depth = [-1] * len(node_index)
    queue = [root]
    depth[root] = 0
    for u in queue:
        for v in adj[u]:
            if depth[v] == -1:
                depth[v] = depth[u] + 1
                queue.append(v)

    max_depth = max(d for d in depth if d >= 0)
    max_branch = max(len(adj[i]) for i in range(len(node_index)))
    avg_branch = edge_count / float(len(node_index)) if len(node_index) > 0 else 0.0

    return num_nodes, edge_count, max_depth, max_branch, avg_branch


def iter_event_files():
    """Yield .json files in EVENT_DIR recursively, sorted by name."""
    if not EVENT_DIR.is_dir():
        raise SystemExit(f"Directory not found: {EVENT_DIR}")
    for path in sorted(EVENT_DIR.rglob("*.json")):
        if path.is_file():
            yield path


# --------------------- Worker: process one file --------------------- #

def process_file(path_str):
    """
    Worker function.

    Returns (event_rows, msg_rows) where:
      - event_rows: list[dict] for events_compact_all.csv
      - msg_rows:   list[dict] for messages_compact_all.csv
    """
    path = Path(path_str)
    event_rows = []
    msg_rows = []

    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
    except Exception:
        return event_rows, msg_rows

    # Some files are dict, some may be list; normalize to list of dicts.
    if isinstance(data, dict):
        events = [data]
    elif isinstance(data, list):
        events = [d for d in data if isinstance(d, dict)]
        if not events:
            return event_rows, msg_rows
    else:
        return event_rows, msg_rows

    for ev in events:
        event_id = ev.get("event_id") or path.stem
        year = parse_year_from_event_id(event_id)

        if TARGET_YEARS is not None and year not in TARGET_YEARS:
            continue

        root_url = ev.get("root_url", "")
        msgs = ev.get("messages", []) or []
        conns = ev.get("connections", []) or []

        message_count = ev.get("message_count", len(msgs))
        num_messages = len(msgs)
        num_connections = len(conns)

        participants = set()
        timestamps = []

        for msg in msgs:
            if not isinstance(msg, dict):
                continue
            sender = get_first_nonempty(msg, ["from", "sender", "author"])
            if sender:
                participants.add(str(sender))
            ts_raw = get_first_nonempty(
                msg,
                ["timestamp", "date", "Date", "time", "Time", "sent_time", "sent", "date_time"],
            )
            dt = parse_timestamp_generic(ts_raw)
            if dt is not None:
                timestamps.append(dt)

        num_participants = len(participants)

        root_msg = msgs[0] if msgs else {}
        root_subject = ""
        if isinstance(root_msg, dict):
            root_subject = get_first_nonempty(root_msg, ["subject", "Subject", "title"])

        long_thread = 1 if message_count is not None and message_count >= 8 else 0

        graph_num_nodes, graph_num_edges, graph_depth, \
            graph_max_branching, graph_avg_branching = compute_graph_features(
                num_messages, conns
            )

        duration_hours = ""
        if timestamps:
            timestamps.sort()
            dt_min = timestamps[0]
            dt_max = timestamps[-1]
            delta = dt_max - dt_min
            duration_hours = delta.total_seconds() / 3600.0

        event_rows.append(
            {
                "event_id": event_id,
                "year": year,
                "root_url": root_url,
                "message_count": message_count,
                "num_messages": num_messages,
                "num_connections": num_connections,
                "num_participants": num_participants,
                "root_subject": root_subject,
                "long_thread": long_thread,
                "graph_num_nodes": graph_num_nodes,
                "graph_num_edges": graph_num_edges,
                "graph_depth": graph_depth,
                "graph_max_branching": graph_max_branching,
                "graph_avg_branching": graph_avg_branching,
                "duration_hours": duration_hours,
            }
        )

        for i, msg in enumerate(msgs):
            if not isinstance(msg, dict):
                continue

            subj = get_first_nonempty(msg, ["subject", "Subject", "title"])
            sender = get_first_nonempty(msg, ["from", "sender", "author"])
            timestamp_raw = get_first_nonempty(
                msg,
                ["timestamp", "date", "Date", "time", "Time", "sent_time", "sent", "date_time"],
            )

            text = get_first_nonempty(
                msg, ["clean_body", "body", "text", "content", "payload"]
            )
            if text and len(text) > TRUNCATE_TEXT_CHARS:
                text_trunc = text[:TRUNCATE_TEXT_CHARS] + "...[TRUNCATED]"
            else:
                text_trunc = text

            weak_role = weak_role_label(subj, text_trunc or "")

            msg_rows.append(
                {
                    "event_id": event_id,
                    "msg_index": i,
                    "is_root": 1 if i == 0 else 0,
                    "sender": str(sender) if sender is not None else "",
                    "timestamp_raw": str(timestamp_raw) if timestamp_raw is not None else "",
                    "subject": subj,
                    "text_trunc": text_trunc,
                    "weak_role": weak_role,
                }
            )

    return event_rows, msg_rows


# --------------------------- Main routine --------------------------- #

def main():
    print(f"Root: {ROOT}")
    print(f"Reading events from: {EVENT_DIR}")
    print(f"Using {N_WORKERS} worker processes")

    events_out_f = OUTPUT_EVENTS.open("w", newline="", encoding="utf-8")
    msgs_out_f   = OUTPUT_MESSAGES.open("w", newline="", encoding="utf-8")

    events_writer = csv.DictWriter(
        events_out_f,
        fieldnames=[
            "event_id",
            "year",
            "root_url",
            "message_count",
            "num_messages",
            "num_connections",
            "num_participants",
            "root_subject",
            "long_thread",
            "graph_num_nodes",
            "graph_num_edges",
            "graph_depth",
            "graph_max_branching",
            "graph_avg_branching",
            "duration_hours",
        ],
    )
    msgs_writer = csv.DictWriter(
        msgs_out_f,
        fieldnames=[
            "event_id",
            "msg_index",
            "is_root",
            "sender",
            "timestamp_raw",
            "subject",
            "text_trunc",
            "weak_role",
        ],
    )

    events_writer.writeheader()
    msgs_writer.writeheader()

    events_count = 0
    files_count = 0

    files_iter = iter_event_files()

    with mp.Pool(processes=N_WORKERS) as pool:
        for event_rows, msg_rows in pool.imap_unordered(
            process_file, (str(p) for p in files_iter)
        ):
            files_count += 1

            for row in event_rows:
                if MAX_EVENTS is not None and events_count >= MAX_EVENTS:
                    break
                events_writer.writerow(row)
                events_count += 1

            for mrow in msg_rows:
                if MAX_EVENTS is not None and events_count >= MAX_EVENTS:
                    break
                msgs_writer.writerow(mrow)

            if MAX_EVENTS is not None and events_count >= MAX_EVENTS:
                break

            if events_count and events_count % 1000 == 0:
                print(f"Processed {events_count} events (from {files_count} files)...")

    events_out_f.close()
    msgs_out_f.close()

    print(f"Done. Events written:   {events_count}")
    print(f"Files processed:        {files_count}")
    print(f"  -> {OUTPUT_EVENTS}")
    print(f"  -> {OUTPUT_MESSAGES}")


if __name__ == "__main__":
    main()
