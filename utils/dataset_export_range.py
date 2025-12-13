"""
File Purpose:
    Export data from the 'mails' table in the source SQLite database within a specified time range, and save it into a new SQLite database file.

Usage:
    python dataset_export_range.py --source source.db --start 2023-01-01 --end 2023-01-31 --output output.db

Arguments Description:
    --source    Path to the source database file, must contain the 'mails' table
    --start     Start date, format: YYYY-MM-DD
    --end       End date, format: YYYY-MM-DD
    --output    Path to the exported database file

Output Explanation:
    The generated database file contains a table named 'mails' with the following structure:
        id INTEGER PRIMARY KEY,
        saved_time TEXT,
        sender TEXT,
        subject TEXT,
        body TEXT
"""

import argparse
import sqlite3
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='Export mails from source DB within a time range to a new DB.')
    parser.add_argument('--source', required=True, help='Source database file')
    parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', required=True, help='Output database file')
    return parser.parse_args()

def create_output_db(conn):
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS mails (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            url TEXT UNIQUE,
            html_content TEXT,
            saved_time TEXT
        );
    ''')
    conn.commit()

import re

def extract_date_from_url(url):
    # Use regex to extract /lkml/YYYY/M/D
    match = re.search(r'/lkml/(\d{4})/(\d{1,2})/(\d{1,2})', url)
    if match:
        try:
            year = int(match.group(1))
            month = int(match.group(2))
            day = int(match.group(3))
            return datetime(year, month, day)
        except ValueError:
            return None
    else:
        if "lkml" in url:
            print(f"Warning: URL with 'lkml' but no valid date pattern: {url}")
        return None

def export_mails(source_db, start_date, end_date, output_db):
    src_conn = sqlite3.connect(source_db)
    out_conn = sqlite3.connect(output_db)
    create_output_db(out_conn)

    src_cursor = src_conn.cursor()
    out_cursor = out_conn.cursor()

    src_cursor.execute('SELECT title, url, html_content, saved_time FROM mails')
    all_rows = src_cursor.fetchall()

    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')

    filtered_rows = []
    for row in all_rows:
        url = row[1]
        url_date = extract_date_from_url(url)
        if url_date and start_dt <= url_date <= end_dt:
            filtered_rows.append(row)

    print(f'Selected {len(filtered_rows)} mails based on URL date range.')

    out_cursor.executemany('''
        INSERT OR IGNORE INTO mails (title, url, html_content, saved_time)
        VALUES (?, ?, ?, ?)
    ''', filtered_rows)
    out_conn.commit()

    src_conn.close()
    out_conn.close()

    return len(filtered_rows)

def main():
    args = parse_args()
    count = export_mails(args.source, args.start, args.end, args.output)
    print(f'Exported {count} mails from {args.start} to {args.end} into {args.output}')

if __name__ == '__main__':
    main()
