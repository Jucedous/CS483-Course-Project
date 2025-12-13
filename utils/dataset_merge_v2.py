import sqlite3, glob, os, time
import re
from datetime import datetime

def extract_date_from_filename(filename):
    match = re.search(r'(\d{4}[-]?\d{2}[-]?\d{2})', filename)
    if match:
        date_str = match.group(1)
        try:
            if '-' in date_str:
                return datetime.strptime(date_str, '%Y-%m-%d')
            else:
                return datetime.strptime(date_str, '%Y%m%d')
        except ValueError:
            return datetime.min
    return datetime.min

# 输出文件
merged_path = "lkml-2015-2025.db"
if os.path.exists(merged_path):
    os.remove(merged_path)

merged_conn = sqlite3.connect(merged_path, uri=True)
merged_cur = merged_conn.cursor()

# 创建目标表
merged_cur.executescript("""
CREATE TABLE IF NOT EXISTS mails (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  title TEXT,
  url TEXT UNIQUE,
  html_content TEXT,
  saved_time TEXT
);
""")

db_files = sorted(glob.glob("data-*.db"), key=extract_date_from_filename)

# 逐个导入
for db_path in db_files:
    try:
        file_size_kb = os.path.getsize(db_path) / 1024
        if file_size_kb < 5:
            print(f"⚠️ Skipping {db_path} due to small file size: {file_size_kb:.2f} KB")
            continue

        print(f"→ Merging {db_path} (Size: {file_size_kb:.2f} KB)")
        merged_cur.execute("BEGIN IMMEDIATE;")
        merged_cur.execute(f"ATTACH DATABASE 'file:{db_path}?mode=ro' AS src")

        # 检查是否存在 mails 表
        merged_cur.execute("SELECT name FROM src.sqlite_master WHERE type='table'")
        tables = [row[0] for row in merged_cur.fetchall()]
        print(f"✓ Tables in {db_path}: {tables}")
        if "mails" not in tables:
            print(f"⚠️ Warning: 'mails' table not found in {db_path}, skipping this database.")
            merged_cur.execute("DETACH DATABASE src")
            merged_conn.rollback()
            continue

        # 忽略重复 url
        merged_cur.execute("""
        INSERT OR IGNORE INTO mails (title, url, html_content, saved_time)
        SELECT title, url, html_content, saved_time FROM src.mails
        """)
        merged_conn.commit()
        merged_cur.execute("DETACH DATABASE src")
        merged_cur.execute("SELECT COUNT(*) FROM mails")
        count = merged_cur.fetchone()[0]
        print(f"✓ Current total mails: {count}")
        time.sleep(0.1)
    except sqlite3.OperationalError as e:
        print(f"❌ OperationalError during merging {db_path}: {e}")
        merged_conn.rollback()
        try:
            merged_cur.execute("DETACH DATABASE src")
        except Exception:
            pass
        time.sleep(0.1)
    except Exception as e:
        print(f"❌ Unexpected error during merging {db_path}: {e}")
        merged_conn.rollback()
        try:
            merged_cur.execute("DETACH DATABASE src")
        except Exception:
            pass
        time.sleep(0.1)

merged_conn.commit()
merged_conn.close()

print(f"✓ All databases merged into: {merged_path}")
