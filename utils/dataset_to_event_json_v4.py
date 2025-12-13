#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LKML SQLite to JSON Events Converter
==============================================================

功能:
    将存储 LKML (Linux Kernel Mailing List) 邮件的 SQLite 数据库转换为
    以 "线程 (Thread)" 为单位的 JSON 事件文件。

核心特性:
    1. 并行加速: 使用 multiprocessing 利用多核 CPU 解析 HTML。
    2. 内存安全: 使用 Batch (分批) 机制，允许处理数 GB 级别的数据库而不爆内存。
    3. 增量合并: 支持将分散在数据库不同位置的邮件合并到同一个线程文件中。

用法示例:
    python lkml_convert_final.py --db lkml.db --out ./output_events --batch-size 20000 --workers 16

"""

import argparse
import json
import os
import re
import sqlite3
import multiprocessing
from collections import defaultdict
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup, Tag
from tqdm import tqdm

# --- 全局配置 ---
LKML_BASE = "https://lkml.org"

# 用于检测 XHTML 文档类型的正则，帮助选择正确的解析器
_XHTML_DOCTYPE_RE = re.compile(r'<!DOCTYPE\s+html\s+PUBLIC\s+"-//W3C//DTD XHTML', re.IGNORECASE)


# ==========================================
# 1. HTML 解析与数据提取辅助函数 (Helpers)
# ==========================================

def pick_bs4_parser(doc_text: str) -> str:
    """
    根据文档头部内容选择最合适的 BeautifulSoup 解析器。
    优先使用 'lxml' (速度快)，如果是 XHTML 则使用 'lxml-xml'。
    """
    if not doc_text: 
        return "lxml"
    # 只检查前200个字符来判断头部
    head = doc_text.lstrip()[:200].lower()
    if head.startswith("<?xml") or _XHTML_DOCTYPE_RE.search(doc_text[:200]):
        return "lxml-xml"
    return "lxml"

def make_soup(doc_text: str) -> BeautifulSoup:
    """创建 BeautifulSoup 对象，包含错误回退机制。"""
    try:
        return BeautifulSoup(doc_text or "", pick_bs4_parser(doc_text or ""))
    except Exception:
        # 如果 lxml 失败，回退到 Python 内置的 html.parser
        return BeautifulSoup(doc_text or "", "html.parser")

def normalize_url(href: str) -> str:
    """将相对路径转换为完整的 URL (https://lkml.org/...)"""
    if not href: return None
    if href.startswith("http"): return href
    if href.startswith("/"): return urljoin(LKML_BASE, href)
    if href.startswith("lkml/"): return urljoin(LKML_BASE + "/", href)
    return href

def text_or_none(x):
    """清洗提取的文本：去除 HTML 实体空格，去除首尾空白。"""
    if x is None: return None
    t = x.get_text(separator="\n", strip=True) if isinstance(x, Tag) else str(x)
    return t.replace("\xa0", " ").strip() or None

def extract_meta_fields(soup: BeautifulSoup):
    """
    从邮件详情页右侧的表格中提取元数据 (Date, From, Subject)。
    """
    meta = {"date": None, "from": None, "subject": None}
    try:
        # 选择器定位：class="c" 的 td 下的 table tr
        for row in soup.select("td.c table tr"):
            tds = row.find_all("td")
            if len(tds) == 2:
                label = tds[0].get_text(strip=True).lower()
                if label in meta: 
                    meta[label] = text_or_none(tds[1])
    except Exception:
        pass
    return meta

def extract_article_pre_text(soup: BeautifulSoup):
    """提取邮件正文，通常在 <pre itemprop="articleBody"> 中。"""
    pre = soup.find("pre", attrs={"itemprop": "articleBody"}) or soup.find("pre")
    if pre:
        # 保留换行格式，将 <br> 转换为 \n
        return pre.get_text(separator="\n", strip=False).replace("\r\n", "\n").replace("\xa0", " ")
    return None

def find_thread_root_and_tree(soup: BeautifulSoup):
    """
    解析左侧 "Messages in this thread" 树状结构。
    
    Returns:
        root_url (str): 整个线程的根节点 URL。
        all_urls (list): 线程中所有出现的 URL。
        edges (list of tuple): 父子引用关系 [(parent_url, child_url), ...]。
    """
    thread_ul = None
    # 寻找包含 "Messages in this thread" 文本的 div 附近的 ul
    for div in soup.find_all("div", class_="threadlist"):
        if "Messages in this thread" in div.get_text():
            thread_ul = div.find_next("ul", class_="threadlist")
            break
    # 如果没找到特定的 div，尝试直接找 ul
    if not thread_ul: 
        thread_ul = soup.find("ul", class_="threadlist")
    
    # 如果页面没有线程树（孤立邮件），返回空
    if not thread_ul: 
        return None, [], []

    all_urls = []
    edges = []
    
    # 递归遍历 li > ul > li 结构建立引用树
    def walk(li, parent_url):
        a = li.find("a", href=True)
        curr_url = normalize_url(a["href"]) if a else None
        
        if curr_url:
            all_urls.append(curr_url)
            if parent_url and curr_url != parent_url: 
                edges.append((parent_url, curr_url))
        
        # 递归处理子节点
        for ul in li.find_all("ul", recursive=False):
            for child in ul.find_all("li", recursive=False):
                walk(child, curr_url)

    for li in thread_ul.find_all("li", recursive=False):
        walk(li, None)

    # 确定 Root URL (通常标记为 class='root' 或列表第一个)
    root_url = None
    root_li = thread_ul.find("li", class_="root")
    if root_li and root_li.find("a", href=True):
        root_url = normalize_url(root_li.find("a")["href"])
    
    # 回退策略：如果没找到 root class，取第一个 URL
    if not root_url and all_urls:
        root_url = all_urls[0]

    # 去重并保持顺序
    unique_urls = list(dict.fromkeys(all_urls))
    return root_url, unique_urls, edges

def build_event_id_from_root(root_url: str) -> str:
    """将 Root URL 转换为合法的文件名 ID，例如: lkml_2014_11_4_837"""
    try:
        path = urlparse(root_url).path
        parts = [p for p in path.split("/") if p]
        # 预期路径结构: /lkml/2014/11/4/837
        if len(parts) >= 5 and parts[0] == "lkml":
            return "lkml_" + "_".join(parts[1:6])
        # 异常路径处理
        return "lkml_" + re.sub(r"[^0-9a-zA-Z_]+", "_", path.strip("/"))
    except: 
        return "lkml_unknown"


# ==========================================
# 2. 工作进程逻辑 (Worker Process)
# ==========================================

def process_one_mail(row_data):
    """
    [纯函数] 单个邮件的处理逻辑，将被 Pool 分发到不同 CPU 核心执行。
    
    Args:
        row_data (tuple): (id, title, url, html_content, saved_time)
    
    Returns:
        dict: 包含解析后的结构化数据，若解析失败或无效则返回 None。
    """
    _id, title, url, html_content, saved_time = row_data
    
    if not url: 
        return None
    
    try:
        # CPU 密集操作开始
        soup = make_soup(html_content)
        meta = extract_meta_fields(soup)
        content = extract_article_pre_text(soup)
        if content: 
            content = content.strip()
        
        root_url, urls_in_thread, edges = find_thread_root_and_tree(soup)
        norm_url = normalize_url(url)
        
        # 如果没有线程树信息，将自己作为单节点处理
        if not urls_in_thread:
            root_url = norm_url if not root_url else root_url
            urls_in_thread = [norm_url]
        
        if not root_url: 
            return None
        
        event_id = build_event_id_from_root(root_url)
        
        # 返回数据结构 (注意：这里不进行任何文件写入)
        return {
            "event_id": event_id,
            "root_url": root_url,
            "edges": [(normalize_url(a), normalize_url(b)) for a, b in edges if a!=b],
            "msg": {
                "url": norm_url,
                "subject": meta.get("subject") or title,
                "title": title,
                "content": content,
                "saved_time": saved_time,
            },
            "thread_urls": urls_in_thread
        }
    except Exception:
        # 忽略解析错误的邮件，避免中断整个批次
        return None


# ==========================================
# 3. 批处理与 I/O 管理 (Batch Manager)
# ==========================================

def flush_batch(batch_map, out_dir, index_tracker):
    """
    将内存中的批次数据写入磁盘。
    逻辑：读取旧文件 -> 合并新数据 -> 覆盖写入。
    
    Args:
        batch_map (dict): 当前批次聚合的数据 {event_id: data}
        out_dir (str): 输出目录
        index_tracker (dict): 用于更新总索引的字典
    """
    for eid, data in batch_map.items():
        path = os.path.join(out_dir, f"{eid}.json")
        
        # --- Step A: 读取现有文件 (如果存在) ---
        event = None
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    event = json.load(f)
            except Exception:
                # 文件损坏或格式错误，视为新文件处理
                event = None
        
        if event is None:
            event = {
                "event_id": eid,
                "root_url": data['root_url'],
                "messages": [],
                "connections": []
            }
        
        # --- Step B: 合并 Connections (使用 Set 去重) ---
        # 将现有的 connections 转换为 tuple set
        existing_conns = set((c['from'], c['to']) for c in event.get("connections", []))
        # 加入新的
        existing_conns.update(data['connections'])
        # 转回 list of dict
        event["connections"] = [{"from": f, "to": t} for f, t in existing_conns]
        
        # --- Step C: 合并 Messages (使用 URL Dict 去重/更新) ---
        msg_map = {m['url']: m for m in event.get("messages", [])}
        
        # 更新当前批次解析到的完整邮件内容
        for m in data['messages']:
            msg_map[m['url']] = m
            
        # 为线程树中出现但尚未抓取到的 URL 创建占位符 (Placeholders)
        existing_urls = set(msg_map.keys())
        for t_url in data['known_urls']:
            if t_url not in existing_urls:
                msg_map[t_url] = {
                    "url": t_url, 
                    "subject": None, 
                    "title": None, 
                    "content": None, 
                    "saved_time": None
                }
                existing_urls.add(t_url)
        
        # --- Step D: 写回文件 ---
        event["messages"] = list(msg_map.values())
        event["message_count"] = len(event["messages"])
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(event, f, ensure_ascii=False, indent=2)
            
        # 更新全局索引记录
        index_tracker[eid] = {"event_id": eid, "root_url": data['root_url'], "file": path}


def run_conversion(db_path, out_dir, batch_size, workers=None):
    """主控制流程"""
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Connecting to database: {db_path}")
    conn = sqlite3.connect(db_path)
    
    # 获取总数用于进度条
    total_count = conn.execute("SELECT COUNT(*) FROM mails").fetchone()[0]
    
    # ORDER BY id 优化：相近 ID 的邮件通常属于同一线程，按顺序处理可减少反复打开同一文件的次数
    cursor = conn.execute("SELECT id, title, url, html_content, saved_time FROM mails ORDER BY id") 
    
    if not workers:
        workers = multiprocessing.cpu_count()
        
    print(f"Starting processing:")
    print(f"  - Total Mails: {total_count}")
    print(f"  - Workers:     {workers} (CPU Cores)")
    print(f"  - Batch Size:  {batch_size} (Mails per flush)")
    print("-" * 60)
    
    # 启动进程池
    pool = multiprocessing.Pool(processes=workers)
    
    index_tracker = {} # 仅存储轻量级的索引信息 {eid: info}
    
    # 初始化进度条
    pbar = tqdm(total=total_count, unit="mail", desc="Processing")
    
    try:
        while True:
            # 1. 从数据库获取一批数据 (Fetch)
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            
            # 2. 并行解析 (Map)
            # pool.map 会阻塞直到这批数据全部处理完
            results = pool.map(process_one_mail, rows)
            
            # 3. 内存聚合 (Aggregate)
            # 使用 batch_map 暂存当前 5000 条数据产生的结果
            batch_map = defaultdict(lambda: {
                "root_url": None, 
                "connections": set(), 
                "messages": [], 
                "known_urls": set()
            })
            
            valid_results_count = 0
            for res in results:
                if not res: continue
                valid_results_count += 1
                
                ev = batch_map[res['event_id']]
                if not ev['root_url']: 
                    ev['root_url'] = res['root_url']
                
                ev['connections'].update(res['edges'])
                ev['messages'].append(res['msg'])
                ev['known_urls'].update(res['thread_urls'])
                
            # 4. 写入磁盘 (Flush)
            flush_batch(batch_map, out_dir, index_tracker)
            
            # 更新进度条
            pbar.update(len(rows))
            
            # 显式清理内存引用
            del rows
            del results
            del batch_map
            
    except KeyboardInterrupt:
        print("\nUser interrupted. Saving progress...")
    finally:
        pool.close()
        pool.join()
        conn.close()
        pbar.close()
    
    # 5. 生成总索引文件
    print(f"\nWriting main index file for {len(index_tracker)} events...")
    index_path = os.path.join(out_dir, "events_index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(list(index_tracker.values()), f, ensure_ascii=False, indent=2)
        
    print(f"Success! Output directory: {out_dir}")


# ==========================================
# 4. 入口函数
# ==========================================

def main():
    ap = argparse.ArgumentParser(description="Convert LKML SQLite DB to Thread-based JSON Events.")
    
    ap.add_argument("--db", required=True, help="Path to the input SQLite database file.")
    ap.add_argument("--out", required=True, help="Directory to save the output JSON files.")
    
    ap.add_argument(
        "--batch-size", 
        type=int, 
        default=5000, 
        dest="batch_size",
        help="Number of mails to process in memory before writing to disk. "
             "Default: 5000. Reduce to 2000 if running on low RAM (8GB)."
    )
    
    ap.add_argument(
        "--workers", 
        type=int, 
        default=None, 
        help="Number of parallel CPU processes. Default: All available cores."
    )
    
    args = ap.parse_args()
    
    # Windows 平台下使用 multiprocessing 必须调用 freeze_support
    multiprocessing.freeze_support()
    
    run_conversion(
        db_path=args.db, 
        out_dir=args.out, 
        batch_size=args.batch_size, 
        workers=args.workers
    )

if __name__ == "__main__":
    main()
