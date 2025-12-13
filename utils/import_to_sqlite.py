#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
import_to_sqlite_v2.py
--------------------------------------------------------
功能：
1. 多线程并行读取 Event JSON, Series JSON (Relevant/Irrelevant) 和 Log。
2. 使用单线程高速批量写入 SQLite (WAL模式)。
3. 自动适配 lkml_*.json 的数据结构。

使用方法：
python utils/import_to_sqlite.py \
    --db_path ./data/lkml-sqlite-dataset-2015-2025.db \
    --event_dir /mnt/d/Haku/Downloads/2015-2025-LKML/0_event_json_data/ \
    --relevant_dir /mnt/d/Haku/Downloads/2015-2025-LKML/2_security_related_classed/security_relevant/ \
    --irrelevant_dir /mnt/d/Haku/Downloads/2015-2025-LKML/2_security_related_classed/security_irrelevant/ \
    --log_file /mnt/d/Haku/Downloads/2015-2025-LKML/2_security_related_classed/merged_analysis.log \
    --workers 16
"""

import sqlite3
import json
import os
import argparse
import re
import time
import threading
from queue import Queue, Empty
from tqdm import tqdm

# ---------- 数据库初始化 ----------

def init_db(db_path):
    """初始化数据库表结构"""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # 开启 WAL 模式 (Write-Ahead Logging)，极大提升并发读写性能
    c.execute('PRAGMA journal_mode=WAL;')
    c.execute('PRAGMA synchronous=NORMAL;')
    
    # 1. Events 表 (邮件原文数据)
    # 根据提供的 lkml_*.json 样本调整字段
    c.execute('''
    CREATE TABLE IF NOT EXISTS events (
        event_id TEXT PRIMARY KEY,
        root_url TEXT,
        subject TEXT,
        date TEXT,
        message_count INTEGER,
        content TEXT,
        raw_json TEXT
    );
    ''')
    
    # 2. Series 表 (包含安全分析概览)
    c.execute('''
    CREATE TABLE IF NOT EXISTS series (
        series_id TEXT PRIMARY KEY,
        category TEXT, -- 'relevant' or 'irrelevant'
        is_security_series BOOLEAN,
        total_variants INTEGER,
        security_relevant_count INTEGER,
        raw_json TEXT
    );
    ''')

    # 3. Variants Analysis 表 (把 Series 里的每个 variant 拆出来存，方便查询)
    c.execute('''
    CREATE TABLE IF NOT EXISTS analysis_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        series_id TEXT,
        variant_id TEXT,
        is_security_relevant BOOLEAN,
        reason TEXT,
        analyzed_at TEXT,
        FOREIGN KEY(series_id) REFERENCES series(series_id)
    );
    ''')
    
    # 创建索引加速查询
    c.execute('CREATE INDEX IF NOT EXISTS idx_analysis_series ON analysis_results(series_id);')
    c.execute('CREATE INDEX IF NOT EXISTS idx_analysis_secure ON analysis_results(is_security_relevant);')

    # 4. Logs 表
    c.execute('''
    CREATE TABLE IF NOT EXISTS logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        level TEXT,
        series_id TEXT,
        variant_id TEXT,
        message TEXT
    );
    ''')

    conn.commit()
    conn.close()

# ---------- 消费者 (DB Writer) ----------

class DBWriter(threading.Thread):
    def __init__(self, db_path, queue):
        super().__init__()
        self.db_path = db_path
        self.queue = queue
        self.daemon = True # 设置为守护线程
        self.running = True
        self.batch_size = 5000 # 每 5000 条提交一次事务

    def run(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # 不同的数据类型对应不同的 SQL
        sql_map = {
            "event": "INSERT OR REPLACE INTO events (event_id, root_url, subject, date, message_count, content, raw_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
            "series": "INSERT OR REPLACE INTO series (series_id, category, is_security_series, total_variants, security_relevant_count, raw_json) VALUES (?, ?, ?, ?, ?, ?)",
            "analysis": "INSERT INTO analysis_results (series_id, variant_id, is_security_relevant, reason, analyzed_at) VALUES (?, ?, ?, ?, ?)",
            "log": "INSERT INTO logs (timestamp, level, series_id, variant_id, message) VALUES (?, ?, ?, ?, ?)"
        }
        
        batch_buffer = {k: [] for k in sql_map.keys()}
        
        while self.running or not self.queue.empty():
            try:
                # 阻塞获取，超时 1 秒以便检查 running 状态
                item_type, item_data = self.queue.get(timeout=1)
                
                batch_buffer[item_type].append(item_data)
                
                # 批量提交
                if len(batch_buffer[item_type]) >= self.batch_size:
                    c.executemany(sql_map[item_type], batch_buffer[item_type])
                    conn.commit()
                    batch_buffer[item_type] = [] # 清空
                
                self.queue.task_done()
                
            except Empty:
                continue
            except Exception as e:
                print(f"[DB ERR] {e}")

        # 最后一波提交
        for dtype, data in batch_buffer.items():
            if data:
                c.executemany(sql_map[dtype], data)
                conn.commit()
        
        conn.close()

# ---------- 生产者 (File Readers) ----------

def parse_event_file(filepath):
    """解析 Event JSON (根据 lkml_*.json 样本)"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            data = json.load(f)
            
            # 1. 提取基础信息
            event_id = data.get("event_id", os.path.basename(filepath).replace(".json", ""))
            root_url = data.get("root_url", "")
            msg_count = data.get("message_count", 0)
            
            # 2. 提取第一条消息的内容作为索引
            subject = ""
            date = ""
            content = ""
            
            messages = data.get("messages", [])
            if isinstance(messages, list) and len(messages) > 0:
                first_msg = messages[0]
                subject = first_msg.get("subject", "")
                # 样本中使用 "saved_time"
                date = first_msg.get("saved_time", "") 
                content = first_msg.get("content", "")

            return (
                event_id,
                root_url,
                subject,
                date,
                msg_count,
                content,
                json.dumps(data, ensure_ascii=False) # 存原始 JSON 以防万一
            )
    except Exception as e:
        # print(f"Error parsing event {filepath}: {e}")
        return None

def parse_series_file(filepath, category):
    """解析 Series JSON (包含 Security Analysis)"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            data = json.load(f)
            
            sid = data.get("series_id", os.path.basename(filepath).replace(".json", ""))
            
            # 1. Series 主表数据
            summary = data.get("security_summary", {})
            series_row = (
                sid,
                category,
                summary.get("is_security_series", False),
                summary.get("total_variants", 0),
                summary.get("security_relevant_count", 0),
                json.dumps(data, ensure_ascii=False)
            )
            
            # 2. Analysis 子表数据列表
            analysis_rows = []
            variants = data.get("variants", {})
            for vid, vdata in variants.items():
                sec_info = vdata.get("security_analysis", {})
                
                analysis_rows.append((
                    sid,
                    vid,
                    sec_info.get("is_security_relevant", False),
                    sec_info.get("reason", "No reason provided"),
                    sec_info.get("analyzed_at", "")
                ))
                
            return series_row, analysis_rows
    except Exception as e:
        return None, []

def parse_log_line(line):
    """解析 Log 行"""
    # 格式示例: [10:00:05] [SECURE] [series_123] [v1] Msg...
    try:
        # 简单正则，提取方括号内容
        # 匹配前4个方括号
        parts = re.findall(r'\[(.*?)\]', line)
        if len(parts) >= 4:
            timestamp = parts[0]
            level = parts[1].strip()
            sid = parts[2].strip()
            vid = parts[3].strip()
            # 消息内容是最后一个 ] 之后的部分
            message = line.split(']', 4)[-1].strip()
            return (timestamp, level, sid, vid, message)
    except:
        pass
    return None

def file_worker(file_list, task_type, queue, pbar):
    """通用文件处理线程"""
    for filepath in file_list:
        if task_type == "event":
            res = parse_event_file(filepath)
            if res: queue.put(("event", res))
        
        elif task_type == "series":
            # 区分 relevant / irrelevant
            category = "relevant" if "security_relevant" in filepath else "irrelevant"
            s_row, a_rows = parse_series_file(filepath, category)
            if s_row:
                queue.put(("series", s_row))
                for a_row in a_rows:
                    queue.put(("analysis", a_row))
        
        pbar.update(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_path", required=True, help="Path to output sqlite db")
    parser.add_argument("--event_dir", required=True, help="Directory containing event jsons")
    parser.add_argument("--relevant_dir", required=True, help="Directory containing security relevant series")
    parser.add_argument("--irrelevant_dir", required=True, help="Directory containing security irrelevant series")
    parser.add_argument("--log_file", help="Path to merged log file")
    parser.add_argument("--workers", type=int, default=8, help="Number of parser threads")
    args = parser.parse_args()

    # 1. 初始化 DB
    print("[INIT] Initializing Database...")
    init_db(args.db_path)
    
    # 2. 扫描文件
    print("[SCAN] Scanning files...")
    event_files = []
    series_files = []
    
    # 扫描 Events
    for root, _, files in os.walk(args.event_dir):
        for f in files:
            if f.endswith(".json"): event_files.append(os.path.join(root, f))
            
    # 扫描 Series (Relevant + Irrelevant)
    for root, _, files in os.walk(args.relevant_dir):
        for f in files:
            if f.endswith(".json"): series_files.append(os.path.join(root, f))
    for root, _, files in os.walk(args.irrelevant_dir):
        for f in files:
            if f.endswith(".json"): series_files.append(os.path.join(root, f))

    print(f"  - Events found: {len(event_files)}")
    print(f"  - Series found: {len(series_files)}")

    # 3. 启动 DB Writer (消费者)
    data_queue = Queue(maxsize=50000) 
    writer = DBWriter(args.db_path, data_queue)
    writer.start()

    # 4. 启动 Log 处理 (单线程读 Log 即可)
    if args.log_file and os.path.exists(args.log_file):
        print(f"[LOG] Processing Log file: {args.log_file}")
        def log_reader():
            with open(args.log_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    res = parse_log_line(line)
                    if res: data_queue.put(("log", res))
        threading.Thread(target=log_reader, daemon=True).start()

    # 5. 启动多线程处理 JSON (生产者)
    print(f"[EXEC] Starting workers ({args.workers} threads)...")
    
    threads = []
    
    # 处理 Events
    if event_files:
        chunk_size = len(event_files) // args.workers + 1
        pbar_events = tqdm(total=len(event_files), desc="Events", unit="file")
        for i in range(args.workers):
            chunk = event_files[i*chunk_size : (i+1)*chunk_size]
            if not chunk: continue
            t = threading.Thread(target=file_worker, args=(chunk, "event", data_queue, pbar_events))
            t.start()
            threads.append(t)
    
    # 处理 Series
    if series_files:
        chunk_size_s = len(series_files) // args.workers + 1
        pbar_series = tqdm(total=len(series_files), desc="Series", unit="file")
        for i in range(args.workers):
            chunk = series_files[i*chunk_size_s : (i+1)*chunk_size_s]
            if not chunk: continue
            t = threading.Thread(target=file_worker, args=(chunk, "series", data_queue, pbar_series))
            t.start()
            threads.append(t)

    # 6. 等待所有解析线程完成
    for t in threads:
        t.join()
    
    try: pbar_events.close()
    except: pass
    try: pbar_series.close()
    except: pass

    # 7. 等待队列写空
    print("[WAIT] Waiting for DB writer to finish writing...")
    data_queue.join()
    
    # 停止 Writer
    writer.running = False
    writer.join()
    
    print(f"[DONE] Data successfully imported to {args.db_path}")

if __name__ == "__main__":
    main()