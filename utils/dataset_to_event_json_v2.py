#!/usr/bin/env python3
"""
Build LKML event JSONs from an SQLite mails table using BeautifulSoup to parse local HTML.

- 仅解析数据库中的本地 html_content，不会访问网络
- 读取: mails(id, title, url, html_content, saved_time)
- 解析页面左侧“Messages in this thread”列表，按线程聚合
- 以 “First message in thread” 或 li.root 为线程根
- 增强点：
  * 动态进度条（tqdm 优先；未安装则简易文本进度）
  * 支持增量写入（--incremental-write）：每次线程聚合更新后就覆盖写事件 JSON（原子替换）
  * 线程合并去重：不同 root 的相同线程自动合并，保留一个事件文件
  * 每封邮件仅输出“当前邮件自己的新内容”（纯文本），不再输出 html
"""

import argparse
import json
import os
import re
import sqlite3
from urllib.parse import urlparse, urljoin, unquote

from bs4 import BeautifulSoup

try:
    from tqdm import tqdm  # pip install tqdm
except Exception:
    tqdm = None


# ---------------------- 进度条 ----------------------

class Progress:
    def __init__(self, total: int, desc: str, unit: str = "it"):
        self.total = total
        self.count = 0
        self.desc = desc
        self.unit = unit
        self._bar = None
        if tqdm is not None and total > 0:
            self._bar = tqdm(total=total, desc=desc, unit=unit)
        else:
            self._next_print = 0

    def update(self, n: int = 1):
        self.count += n
        if self._bar is not None:
            self._bar.update(n)
        else:
            if self.total > 0:
                pct = int(self.count * 100 / self.total)
                if pct >= self._next_print:
                    print(f"{self.desc}: {self.count}/{self.total} ({pct}%)")
                    self._next_print = min(100, pct + 5)
            else:
                if self.count % 1000 == 0:
                    print(f"{self.desc}: {self.count}")

    def close(self):
        if self._bar is not None:
            self._bar.close()


# ---------------------- URL/ID 工具 ----------------------

def normalize_path(u: str) -> str:
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
    s = root_path.lstrip("/")
    s = s.replace("/", "_")
    s = re.sub(r"[^A-Za-z0-9_\-\.]", "_", s)
    return s or "event"


def last_seg_num(p: str) -> int:
    try:
        seg = p.rstrip("/").split("/")[-1]
        return int(seg)
    except Exception:
        return 10**12


# ---------------------- HTML -> 仅“当前邮件新内容”的提取 ----------------------

QUOTE_INTRO_RE = re.compile(
    r'^\s*(On .+ wrote:|Le .+ a écrit:|Am .+ schrieb .+:|.*wrote:)\s*$',
    re.IGNORECASE
)

def _has_quote_class(c):
    if not c:
        return False
    s = c if isinstance(c, str) else " ".join(c)
    return re.search(r'\b(quote|quoted|gmail_quote|moz-cite-prefix)\b', s, re.IGNORECASE) is not None

def extract_new_content_from_html(html: str) -> str:
    """
    提取单封邮件“新增内容”的纯文本：
      - 去掉导航和线程列表
      - 去掉 <blockquote> 与带 quote 类的元素（旧内容）
      - 行级别去掉以 '>' 或 '| ' 开头的引用行
      - 去掉常见引用引导行（“On ... wrote:” 等）
      - 去掉签名（遇到以 '-- ' 的签名分隔线即截断）
      - 尽量保留换行，合并多余空行
    """
    if not html:
        return ""

    soup = BeautifulSoup(html, "html.parser")

    # 去掉明显与正文无关的块
    for tag in soup.find_all(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    for div in soup.find_all("div", class_="threadlist"):
        div.decompose()

    # 选正文容器（多重兜底）
    candidates = [
        "div#body", "div.message", "div#msg", "div#content", "article",
        "div#contentbody", "div.text", "pre"
    ]
    body = None
    for sel in candidates:
        body = soup.select_one(sel)
        if body:
            break
    if body is None:
        body = soup

    # 删除引用块（旧内容）
    for bq in body.find_all(["blockquote", "q"]):
        bq.decompose()
    for el in body.find_all(class_=_has_quote_class):
        el.decompose()

    # 抽文本（尽量保留行）
    text = body.get_text("\n")
    lines = [ln.rstrip() for ln in text.splitlines()]

    # 去掉可能的头部 meta 区（多行 "Key: value" 直到空行）
    i = 0
    header_count = 0
    while i < len(lines) and re.match(r"^[A-Za-z0-9-]+:\s", lines[i]):
        header_count += 1
        i += 1
    if header_count >= 2:
        if i < len(lines) and lines[i].strip() == "":
            i += 1
        lines = lines[i:]

    # 逐行过滤引用与签名
    filtered = []
    for ln in lines:
        s = ln.lstrip()
        # 签名分隔线，截断
        if re.match(r"^--\s?$", s):
            break
        # 常见引用引导语
        if QUOTE_INTRO_RE.match(s):
            continue
        # 行级引用
        if s.startswith(">") or s.startswith("|>") or s.startswith("| "):
            continue
        filtered.append(ln)

    # 收敛空行
    final_lines = []
    prev_blank = False
    for ln in filtered:
        blank = (ln.strip() == "")
        if blank and prev_blank:
            continue
        final_lines.append(ln)
        prev_blank = blank

    content = "\n".join(final_lines).strip()

    # 万一过度清理，兜底从 <pre> 中再尝试一次仅去掉以 '>' 开头的行
    if not content:
        for pre in soup.find_all("pre"):
            txt = pre.get_text("\n")
            lns = [l.rstrip() for l in txt.splitlines() if not l.lstrip().startswith(">")]
            content = "\n".join(lns).strip()
            if content:
                break

    return content


# ---------------------- 线程解析 ----------------------

def extract_thread_section_ordered_paths(html: str):
    ordered_paths = []
    all_paths = set()
    root_path = None
    has_explicit_root = False

    if not html:
        return ordered_paths, all_paths, root_path, has_explicit_root

    soup = BeautifulSoup(html, "html.parser")

    # 寻找“Messages in this thread”对应的 ul.threadlist
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
        for ul in soup.find_all("ul", class_="threadlist"):
            if ul.find("a", string=lambda x: isinstance(x, str) and "first message in thread" in x.lower()):
                target_ul = ul
                break

    if target_ul is None:
        return ordered_paths, all_paths, root_path, has_explicit_root

    for a in target_ul.find_all("a", href=True):
        href = a["href"]
        path = normalize_path(href)
        if not path:
            continue
        ordered_paths.append(path)
        all_paths.add(path)

    li_root = target_ul.find("li", class_="root")
    if li_root:
        a_root = li_root.find("a", href=True)
        if a_root:
            rp = normalize_path(a_root["href"])
            if rp:
                root_path = rp
                has_explicit_root = True

    if not has_explicit_root:
        a_first = target_ul.find("a", string=lambda x: isinstance(x, str) and "first message in thread" in x.lower())
        if a_first and a_first.has_attr("href"):
            rp = normalize_path(a_first["href"])
            if rp:
                root_path = rp
                has_explicit_root = True

    if not root_path and all_paths:
        root_path = sorted(list(all_paths), key=lambda p: (last_seg_num(p), p))[0]

    return ordered_paths, all_paths, root_path, has_explicit_root


# ---------------------- 线程合并去重 ----------------------

def choose_canonical_root(candidate_root, candidate_explicit, existing_roots, threads):
    roots = set(existing_roots)
    if candidate_root:
        roots.add(candidate_root)
    if not roots:
        return candidate_root

    def sort_key(r):
        explicit = candidate_explicit if r == candidate_root else (threads.get(r, {}).get("has_explicit_root") or False)
        return (0 if explicit else 1, last_seg_num(r), r)

    final_root = sorted(roots, key=sort_key)[0]
    return final_root


def merge_threads(into_root, from_root, threads, path_owner, out_dir, clean_disk=False):
    if into_root == from_root or from_root not in threads:
        return
    t_into = threads.setdefault(into_root, {"urls": set(), "best_order": None, "order_quality": 0, "has_explicit_root": False})
    t_from = threads[from_root]

    t_into["urls"].update(t_from["urls"])

    if t_from.get("best_order"):
        q_into = t_into.get("order_quality", 0)
        q_from = t_from.get("order_quality", 0)
        if t_into.get("best_order") is None or q_from > q_into:
            t_into["best_order"] = list(dict.fromkeys(t_from["best_order"]))
            t_into["order_quality"] = q_from
        else:
            existing = set(t_into["best_order"])
            for p in t_from["best_order"]:
                if p not in existing:
                    t_into["best_order"].append(p)
                    existing.add(p)

    t_into["has_explicit_root"] = bool(t_into.get("has_explicit_root") or t_from.get("has_explicit_root"))

    for p in t_from["urls"]:
        path_owner[p] = into_root

    del threads[from_root]

    if clean_disk:
        try:
            stale_path = os.path.join(out_dir, f"{safe_event_id_from_path(from_root)}.json")
            if os.path.exists(stale_path):
                os.remove(stale_path)
        except Exception:
            pass


# ---------------------- 写文件（原子替换） ----------------------

def build_ordered_paths_for_thread(t):
    ordered = list(t["best_order"]) if t.get("best_order") else []
    seen = set(ordered)
    for p in t["urls"]:
        if p not in seen:
            ordered.append(p)
            seen.add(p)
    return ordered


def write_event_file(root_path, t, out_dir, base_url, msg_by_path):
    ordered = build_ordered_paths_for_thread(t)
    if not ordered and root_path:
        ordered = [root_path]

    messages = []
    for p in ordered:
        rec = msg_by_path.get(p)
        abs_url = urljoin(base_url, p)
        if rec is None:
            messages.append({
                "url": abs_url,
                "subject": "",
                "content": "",
                "saved_time": None
            })
        else:
            messages.append({
                "url": abs_url,
                "subject": rec.get("title") or "",
                "content": rec.get("content_text") or "",
                "saved_time": rec.get("saved_time")
            })

    event = {
        "event_id": safe_event_id_from_path(root_path),
        "root_url": urljoin(base_url, root_path),
        "message_count": len(messages),
        "messages": messages
    }

    os.makedirs(out_dir, exist_ok=True)
    out_name = safe_event_id_from_path(root_path) + ".json"
    out_path = os.path.join(out_dir, out_name)
    tmp_path = out_path + ".tmp"

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(event, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, out_path)


# ---------------------- 主流程 ----------------------

def main():
    ap = argparse.ArgumentParser(description="Build LKML event JSONs from SQLite mails table (local HTML with BeautifulSoup).")
    ap.add_argument("--db", required=True, help="Path to SQLite database file")
    ap.add_argument("--out", required=True, help="Output directory for event JSON files")
    ap.add_argument("--base-url", default="https://lkml.org", help="Base URL to prefix paths (default: https://lkml.org)")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of rows processed (for debugging)")
    ap.add_argument("--incremental-write", action="store_true",
                    help="Write/overwrite event JSON immediately whenever a thread is updated (may rewrite files multiple times).")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    total_rows = cur.execute("SELECT COUNT(*) FROM mails").fetchone()[0]
    if args.limit and args.limit > 0:
        total_rows = min(total_rows, int(args.limit))

    sql = "SELECT id, title, url, html_content, saved_time FROM mails"
    if args.limit and args.limit > 0:
        sql += f" LIMIT {int(args.limit)}"
    cur.execute(sql)

    msg_by_path = {}
    threads = {}
    path_owner = {}

    p_rows = Progress(total_rows, "Reading rows", unit="row")
    row_count = 0

    for row in cur:
        row_count += 1
        p_rows.update(1)

        url = row["url"] or ""
        path = normalize_path(url)
        title = row["title"] or ""
        html = row["html_content"] or ""
        saved_time = row["saved_time"]

        # 预先抽取“当前邮件的新内容”（纯文本），仅计算一次
        content_text = extract_new_content_from_html(html)

        prior = msg_by_path.get(path)
        if prior is None or (not prior.get("title") and title) or (not prior.get("content_text") and content_text):
            msg_by_path[path] = {
                "url": url,
                "title": title,
                "content_text": content_text,
                "saved_time": saved_time,
            }

        ordered_paths, all_paths, root_path, has_explicit_root = extract_thread_section_ordered_paths(html)

        if not all_paths:
            root_path = root_path or path
            ordered_paths = [path] if path else []
            all_paths = {path} if path else set()
            has_explicit_root = False

        if path and path not in all_paths:
            all_paths.add(path)
            ordered_paths = ordered_paths + [path] if ordered_paths else [path]

        if not root_path:
            root_path = path

        existing_roots = set()
        for p in all_paths:
            r = path_owner.get(p)
            if r:
                existing_roots.add(r)

        final_root = choose_canonical_root(root_path, has_explicit_root, existing_roots, threads)

        t = threads.get(final_root)
        if t is None:
            t = {"urls": set(), "best_order": None, "order_quality": 0, "has_explicit_root": False}
            threads[final_root] = t

        to_merge = set(existing_roots)
        if root_path and root_path in threads and root_path != final_root:
            to_merge.add(root_path)
        for r in list(to_merge):
            if r != final_root and r in threads:
                merge_threads(final_root, r, threads, path_owner, args.out, clean_disk=args.incremental_write)

        t = threads[final_root]
        t["urls"].update(all_paths)

        this_quality = 2 if has_explicit_root else (1 if ordered_paths else 0)
        if ordered_paths:
            if t.get("best_order") is None or this_quality > t.get("order_quality", 0):
                t["best_order"] = list(dict.fromkeys(ordered_paths))
                t["order_quality"] = this_quality
            else:
                existing = set(t["best_order"])
                for p in ordered_paths:
                    if p not in existing:
                        t["best_order"].append(p)
                        existing.add(p)

        if has_explicit_root:
            t["has_explicit_root"] = True

        for p in all_paths:
            path_owner[p] = final_root

        if args.incremental_write:
            write_event_file(final_root, t, args.out, args.base_url, msg_by_path)

    p_rows.close()

    p_write = Progress(len(threads), "Writing events", unit="event")
    for root_path, t in threads.items():
        write_event_file(root_path, t, args.out, args.base_url, msg_by_path)
        p_write.update(1)
    p_write.close()

    print(f"Processed rows: {row_count}")
    print(f"Events written: {len(threads)}")
    print(f"Output dir: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
