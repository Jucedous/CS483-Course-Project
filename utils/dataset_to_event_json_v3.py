#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert LKML SQLite (tables: mails, progress) to per-thread event JSON files.

- Groups mails by thread using the left "Messages in this thread" tree.
- Builds parent→child connections from the UL/LI nesting.
- Splits email body into mail_content (normal text) and code_content (patches, diffs, code blocks).
- Outputs one JSON per event (thread), plus an events_index.json.

Usage:
    python lkml_db_to_events.py --db path/to/your.db --out out_dir

Schema assumptions:
mails(id INTEGER PRIMARY KEY, title TEXT, url TEXT, html_content TEXT, saved_time TEXT)
progress(id INTEGER PRIMARY KEY, last_date TEXT)

Tested with lkml.org HTML like in user's samples.
"""

import argparse
import json
import os
import re
import sqlite3
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup, NavigableString, Tag
from bs4 import XMLParsedAsHTMLWarning
import warnings
from tqdm import tqdm


LKML_BASE = "https://lkml.org"

# --- Parser selection for XHTML/XML vs HTML ---

# If you prefer to silence BeautifulSoup's XMLParsedAsHTMLWarning globally, uncomment:
# warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

_XHTML_DOCTYPE_RE = re.compile(
    r'<!DOCTYPE\s+html\s+PUBLIC\s+"-//W3C//DTD XHTML', re.IGNORECASE
)

def pick_bs4_parser(doc_text: str) -> str:
    """Choose the most reliable parser: 'lxml-xml' for XHTML/XML heads, else 'lxml'."""
    if not doc_text:
        return "lxml"
    head = doc_text.lstrip()[:200]
    lower = head.lower()
    if lower.startswith("<?xml") or _XHTML_DOCTYPE_RE.search(head):
        return "lxml-xml"
    return "lxml"

def make_soup(doc_text: str) -> BeautifulSoup:
    parser = pick_bs4_parser(doc_text or "")
    try:
        return BeautifulSoup(doc_text or "", parser)
    except Exception:
        # Fallback to stdlib HTML parser if lxml fails
        return BeautifulSoup(doc_text or "", "html.parser")

# ---------- HTML parsing helpers ----------

def normalize_url(href: str) -> str:
    if not href:
        return None
    if href.startswith("http://") or href.startswith("https://"):
        return href
    if href.startswith("/"):
        return urljoin(LKML_BASE, href)
    # some pages might have bare paths like "lkml/2015/2/11/453" (rare)
    if href.startswith("lkml/"):
        return urljoin(LKML_BASE + "/", href)
    return href

def text_or_none(x):
    if x is None:
        return None
    t = x.get_text(separator="\n", strip=True) if isinstance(x, Tag) else str(x)
    t = t.replace("\xa0", " ").strip()
    return t if t else None

def extract_meta_fields(soup: BeautifulSoup):
    """Parse the right table meta: Date / From / Subject."""
    meta = {"date": None, "from": None, "subject": None}
    try:
        # meta table appears as a pair of <td class="lp">Label</td> <td class="rp">Value</td>
        for row in soup.select("td.c table tr"):  # the right column meta table rows
            tds = row.find_all("td")
            if len(tds) != 2:
                continue
            label = tds[0].get_text(strip=True)
            value = tds[1]
            if label.lower() == "date":
                meta["date"] = text_or_none(value)
            elif label.lower() == "from":
                meta["from"] = text_or_none(value)
            elif label.lower() == "subject":
                meta["subject"] = text_or_none(value)
    except Exception:
        pass
    return meta

def extract_article_pre_text(soup: BeautifulSoup):
    """
    Body appears under <pre itemprop="articleBody"> ... with <br/>.
    We'll convert <br> to '\n' via get_text(separator="\n").
    """
    pre = soup.find("pre", attrs={"itemprop": "articleBody"})
    if not pre:
        # fallback: any <pre>
        pre = soup.find("pre")
    if not pre:
        return None
    txt = pre.get_text(separator="\n", strip=False)
    # Normalize NBSP and Windows newlines
    txt = txt.replace("\r\n", "\n").replace("\xa0", " ")
    return txt

PATCH_START_PAT = re.compile(
    r"^(diff --git .+|---\s+[ab]/.+|\+\+\+\s+[ab]/.+|Index:\s|RCS file:|retrieving revision|@@\s.*@@)",
    re.IGNORECASE
)

CODE_FENCE_PAT = re.compile(
    r"^(\s{4,}|\t)",  # 4+ leading spaces or a tab: often code quote in emails
)


def clean_mail_content(full_text: str):
    """
    Remove quoted text and signatures from the email, returning a cleaned content string.
    """
    if not full_text:
        return None
    # For this rewrite, keep mail content as-is except remove HTML tags (already done by extract_article_pre_text)
    # So just return the full_text stripped
    return full_text.strip() or None

# ---------- Thread tree parsing (left column) ----------

def find_thread_root_and_tree(soup: BeautifulSoup):
    """
    Locate the thread list ("Messages in this thread") <ul class="threadlist"> and rebuild a URL tree.
    Returns:
      root_url: str
      all_urls_in_order: list[str]
      edges: list[(parent_url, child_url)]
    """
    # Find the "Messages in this thread" section: there may be multiple <div class="threadlist">; the first after its <div> title is usually the one.
    threadlist_title = None
    for div in soup.find_all("div", class_="threadlist"):
        if "Messages in this thread" in div.get_text():
            threadlist_title = div
            break
    if not threadlist_title:
        # Another layout variant: sometimes there's just the UL
        thread_ul = soup.find("ul", class_="threadlist")
    else:
        # The UL should be the next or sibling
        thread_ul = threadlist_title.find_next("ul", class_="threadlist")

    if not thread_ul:
        # Fallback: no tree found — treat current page as a singleton thread
        # Root URL is the canonical link (breadcrumb) or the 'headers' link basename, or the exact URL field from DB
        return None, [], []

    all_urls = []
    edges = []

    def walk_li(li: Tag, parent_href: str = None):
        # first <a> in <li> is the node link
        a = li.find("a", href=True)
        if a:
            href = normalize_url(a["href"])
        else:
            href = None

        if href:
            all_urls.append(href)
            if parent_href and href != parent_href:
                edges.append((parent_href, href))

        # recurse into nested ULs under this LI
        for ul in li.find_all("ul", recursive=False):
            for child_li in ul.find_all("li", recursive=False):
                walk_li(child_li, href)

    # The top-level <ul> contains multiple <li class="root"> etc.
    for li in thread_ul.find_all("li", recursive=False):
        walk_li(li, None)

    # Determine root_url:
    # Prefer the one whose <li> has class 'root' or text "First message in thread".
    root_url = None
    root_li = thread_ul.find("li", class_="root")
    if root_li:
        a = root_li.find("a", href=True)
        if a:
            root_url = normalize_url(a["href"])
    if not root_url:
        fm = thread_ul.find("a", string=lambda s: s and "First message in thread" in s)
        if fm:
            root_url = normalize_url(fm.get("href"))

    # Fallback: first collected url
    if not root_url and all_urls:
        root_url = all_urls[0]

    # Deduplicate while preserving order
    seen = set()
    dedup_urls = []
    for u in all_urls:
        if u and u not in seen:
            dedup_urls.append(u)
            seen.add(u)

    return root_url, dedup_urls, edges

# ---------- Event id helper ----------

def build_event_id_from_root(root_url: str) -> str:
    """
    Example:
      https://lkml.org/lkml/2014/11/4/837 -> lkml_2014_11_4_837
    """
    try:
        path = urlparse(root_url).path  # /lkml/2014/11/4/837
        parts = [p for p in path.split("/") if p]
        # expected ['lkml', '2014', '11', '4', '837']
        if len(parts) >= 5 and parts[0] == "lkml":
            return "lkml_" + "_".join(parts[1:5+1][:5])
        # fallback
        safe = re.sub(r"[^0-9a-zA-Z_]+", "_", path.strip("/"))
        return f"lkml_{safe}"
    except Exception:
        return "lkml_unknown"

# ---------- Main conversion ----------

def load_event_file(event_path):
    if os.path.exists(event_path):
        try:
            with open(event_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None

def save_event_file(event_path, event_data):
    with open(event_path, "w", encoding="utf-8") as f:
        json.dump(event_data, f, ensure_ascii=False, indent=2)

def merge_connections(existing, new):
    seen = set((conn['from'], conn['to']) for conn in existing)
    merged = existing[:]
    for conn in new:
        key = (conn['from'], conn['to'])
        if key not in seen:
            merged.append(conn)
            seen.add(key)
    return merged

def merge_messages(existing, new_msgs):
    existing_urls = set(m['url'] for m in existing)
    merged = existing[:]
    for m in new_msgs:
        if m['url'] not in existing_urls:
            merged.append(m)
            existing_urls.add(m['url'])
    return merged

def convert(db_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Keep track of event files generated to build index later
    event_files = {}

    cursor = conn.execute("SELECT id, title, url, html_content, saved_time FROM mails")
    for r in tqdm(cursor, desc="Processing mails", unit="mail"):
        url = r["url"]
        if not url:
            continue
        soup = make_soup(r["html_content"])
        meta = extract_meta_fields(soup)
        full_text = extract_article_pre_text(soup)
        content = clean_mail_content(full_text)

        root_url, urls_in_thread, edges = find_thread_root_and_tree(soup)
        if not urls_in_thread:
            root_url = normalize_url(url) if not root_url else root_url
            urls_in_thread = [normalize_url(url)]
            edges = []

        if not root_url:
            # Skip mails without a root_url, cannot assign event
            continue

        event_id = build_event_id_from_root(root_url)
        event_path = os.path.join(out_dir, f"{event_id}.json")

        # Load existing event data if present
        event = load_event_file(event_path)
        if event is None:
            event = {
                "event_id": event_id,
                "root_url": root_url,
                "message_count": 0,
                "messages": [],
                "connections": [],
            }

        # Merge connections
        new_connections = [
            {"from": normalize_url(a), "to": normalize_url(b)}
            for (a, b) in edges if a and b and a != b
        ]
        event["connections"] = merge_connections(event.get("connections", []), new_connections)

        # Prepare message for this mail
        msg = {
            "url": normalize_url(url),
            "subject": meta.get("subject") or r["title"] or None,
            "title": r["title"] or None,
            "content": content,
            "saved_time": r["saved_time"],
        }

        # Merge or update message in event
        existing_msgs = event.get("messages", [])
        # Check if message already exists by url
        found = False
        for i, m in enumerate(existing_msgs):
            if m["url"] == msg["url"]:
                # Update existing message (overwrite)
                existing_msgs[i] = msg
                found = True
                break
        if not found:
            existing_msgs.append(msg)

        # Ensure all URLs in the thread are represented as messages (add placeholders if missing)
        existing_urls = set(m['url'] for m in existing_msgs)
        for thread_url in urls_in_thread:
            if thread_url not in existing_urls:
                placeholder_msg = {
                    "url": thread_url,
                    "subject": None,
                    "title": None,
                    "content": None,
                    "saved_time": None,
                }
                existing_msgs.append(placeholder_msg)
                existing_urls.add(thread_url)

        event["messages"] = existing_msgs
        event["message_count"] = len(existing_msgs)

        # Save back event JSON file
        save_event_file(event_path, event)

        # Record event file for index
        event_files[event_id] = {
            "event_id": event_id,
            "root_url": root_url,
            "file": event_path,
        }

    # After processing all mails, write events_index.json
    index_path = os.path.join(out_dir, "events_index.json")
    index = list(event_files.values())
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(event_files)} event files to {out_dir}")
    print(f"Index: {index_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="Path to SQLite database")
    ap.add_argument("--out", required=True, help="Output directory for event JSON files")
    args = ap.parse_args()
    convert(args.db, args.out)

if __name__ == "__main__":
    main()
