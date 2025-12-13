#!/usr/bin/env python3
"""
Build LKML event JSONs from an SQLite mails table.

- Reads: mails(id, title, url, html_content, saved_time)
- Parses each html_content to extract the left-side "Messages in this thread" list
- Uses the "First message in thread" link as the thread root, aggregates messages by root
- Writes one JSON per event, including all URLs in the thread (even if not in DB; those have empty title/html_content)
"""

import argparse
import json
import os
import re
import sqlite3
from urllib.parse import urlparse, urljoin, unquote

from bs4 import BeautifulSoup


def normalize_path(u: str) -> str:
    """
    Normalize an lkml message URL to a canonical path (e.g., '/lkml/2015/1/1/62').
    Works for absolute and relative URLs. Removes trailing slashes (except root),
    and unquotes percent-encoded characters.
    """
    if not u:
        return ""
    try:
        p = urlparse(u)
        path = p.path
        if not path:
            if u.startswith("/"):
                path = u
            else:
                path = "/" + u
        if len(path) > 1 and path.endswith("/"):
            path = path[:-1]
        path = unquote(path)
        return path
    except Exception:
        if u.startswith("/"):
            return u.rstrip("/")
        return "/" + u.rstrip("/")


def safe_event_id_from_path(root_path: str) -> str:
    """
    Make a filesystem-safe event id from a root path like '/lkml/2015/1/1/1'
    """
    s = root_path.lstrip("/")
    s = s.replace("/", "_")
    s = re.sub(r"[^A-Za-z0-9_\-\.]", "_", s)
    if not s:
        s = "event"
    return s


def extract_thread_section_ordered_paths(html: str):
    """
    Parse the HTML and extract:
      - ordered_paths: list of normalized hrefs as they appear in the thread list
      - all_paths: set of normalized hrefs
      - root_path: normalized path of the "First message in thread" (if identifiable)
      - has_explicit_root: whether we found an explicit 'li.root' or 'First message in thread'
    Returns (ordered_paths, all_paths, root_path, has_explicit_root)
    """
    ordered_paths = []
    all_paths = set()
    root_path = None
    has_explicit_root = False

    if not html:
        return ordered_paths, all_paths, root_path, has_explicit_root

    soup = BeautifulSoup(html, "html.parser")

    # Find the specific thread list for "Messages in this thread"
    target_ul = None
    for div in soup.find_all("div", class_="threadlist"):
        label = (div.get_text() or "").strip().lower()
        if "messages in this thread" in label:
            sib = div.find_next_sibling()
            while sib is not None:
                if getattr(sib, "name", None) == "ul" and "threadlist" in (sib.get("class") or []):
                    target_ul = sib
                    break
                sib = sib.find_next_sibling()
            if target_ul is not None:
                break

    if target_ul is None:
        # Fallback: try any ul.threadlist that contains anchor text "First message in thread"
        for ul in soup.find_all("ul", class_="threadlist"):
            if ul.find("a", string=lambda x: isinstance(x, str) and "first message in thread" in x.lower()):
                target_ul = ul
                break

    if target_ul is None:
        return ordered_paths, all_paths, root_path, has_explicit_root

    # Collect anchors under this UL (including nested lists)
    for a in target_ul.find_all("a", href=True):
        href = a["href"]
        path = normalize_path(href)
        if not path:
            continue
        ordered_paths.append(path)
        all_paths.add(path)

    # Identify the root by li.root if present
    li_root = target_ul.find("li", class_="root")
    if li_root:
        a_root = li_root.find("a", href=True)
        if a_root:
            root_path = normalize_path(a_root["href"])
            if root_path:
                has_explicit_root = True

    # Or link text "First message in thread"
    if not has_explicit_root:
        a_first = target_ul.find("a", string=lambda x: isinstance(x, str) and "first message in thread" in x.lower())
        if a_first and a_first.has_attr("href"):
            rp = normalize_path(a_first["href"])
            if rp:
                root_path = rp
                has_explicit_root = True

    # Fallback heuristic for root: pick path with smallest last numeric segment
    if not root_path and all_paths:
        def last_seg_num(p):
            try:
                seg = p.rstrip("/").split("/")[-1]
                return int(seg)
            except Exception:
                return 10**12
        root_path = sorted(list(all_paths), key=lambda p: (last_seg_num(p), p))[0]

    return ordered_paths, all_paths, root_path, has_explicit_root


def main():
    ap = argparse.ArgumentParser(description="Build LKML event JSONs from SQLite mails table.")
    ap.add_argument("--db", required=True, help="Path to SQLite database file")
    ap.add_argument("--out", required=True, help="Output directory for event JSON files")
    ap.add_argument("--base-url", default="https://lkml.org", help="Base URL to prefix paths (default: https://lkml.org)")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of rows processed (for debugging)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    sql = "SELECT id, title, url, html_content, saved_time FROM mails"
    if args.limit and args.limit > 0:
        sql += f" LIMIT {int(args.limit)}"
    cur.execute(sql)

    # Index of message records by normalized path
    msg_by_path = {}
    # Event aggregation by root path
    threads = {}

    row_count = 0
    for row in cur:
        row_count += 1
        url = row["url"] or ""
        path = normalize_path(url)
        title = row["title"] or ""
        html = row["html_content"] or ""
        saved_time = row["saved_time"]

        # Index this message content by its path
        prior = msg_by_path.get(path)
        if prior is None or (not prior["title"] and title) or (not prior["html_content"] and html):
            msg_by_path[path] = {
                "url": url,
                "title": title,
                "html_content": html,
                "saved_time": saved_time,
            }

        # Parse thread info from this page
        ordered_paths, all_paths, root_path, has_explicit_root = extract_thread_section_ordered_paths(html)

        # If thread list is empty, treat the single message as its own event
        if not all_paths:
            root_path = root_path or path
            ordered_paths = [path] if path else []
            all_paths = {path} if path else set()
            has_explicit_root = False

        # Ensure the current page's path is included
        if path:
            if path not in all_paths:
                all_paths.add(path)
                ordered_paths = ordered_paths + [path] if ordered_paths else [path]

        if not root_path:
            root_path = path

        # Initialize/update thread aggregation
        t = threads.get(root_path)
        if t is None:
            t = {"urls": set(), "best_order": None, "order_quality": 0}
            threads[root_path] = t

        t["urls"].update(all_paths)

        # Decide whether to set/replace the best order
        this_quality = 2 if has_explicit_root else (1 if ordered_paths else 0)
        if ordered_paths:
            if t["best_order"] is None or this_quality > t["order_quality"]:
                t["best_order"] = list(dict.fromkeys(ordered_paths))  # dedupe preserving order
                t["order_quality"] = this_quality
            else:
                existing = set(t["best_order"])
                for p in ordered_paths:
                    if p not in existing:
                        t["best_order"].append(p)
                        existing.add(p)

    # Produce event JSONs
    for root_path, t in threads.items():
        ordered = t["best_order"][:] if t["best_order"] else []
        seen = set(ordered)
        for p in t["urls"]:
            if p not in seen:
                ordered.append(p)
                seen.add(p)

        if not ordered and root_path:
            ordered = [root_path]

        messages = []
        for p in ordered:
            rec = msg_by_path.get(p)
            abs_url = urljoin(args.base_url, p)
            if rec is None:
                messages.append({
                    "url": abs_url,
                    "title": "",
                    "html_content": "",
                    "saved_time": None
                })
            else:
                messages.append({
                    "url": abs_url,
                    "title": rec.get("title") or "",
                    "html_content": rec.get("html_content") or "",
                    "saved_time": rec.get("saved_time")
                })

        event = {
            "event_id": safe_event_id_from_path(root_path),
            "root_url": urljoin(args.base_url, root_path),
            "message_count": len(messages),
            "messages": messages
        }

        out_name = safe_event_id_from_path(root_path) + ".json"
        out_path = os.path.join(args.out, out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(event, f, ensure_ascii=False, indent=2)

    print(f"Processed rows: {row_count}")
    print(f"Events written: {len(threads)}")
    print(f"Output dir: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
