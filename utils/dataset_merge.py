import sqlite3
import os

def merge_mail_tables(source_db_folder, target_db_filename):
    """
    Merge mail tables from all SQLite databases in the specified folder and properly update sqlite_sequence.
    
    Parameters:
        source_db_folder (str): Path to the folder containing source SQLite databases.
        target_db_filename (str): Path to the target database file.
    """
    # Check if folder exists
    if not os.path.exists(source_db_folder):
        print(f"[ERROR] The folder '{source_db_folder}' does not exist.")
        return
    
    # Get all SQLite database files in the folder (assuming .db extension)
    source_db_filenames = [
        os.path.join(source_db_folder, f) for f in os.listdir(source_db_folder) if f.endswith('.db')
    ]
    
    if not source_db_filenames:
        print(f"[ERROR] No SQLite database files found in folder: {source_db_folder}")
        return
    
    print(f"[INFO] Found {len(source_db_filenames)} database file(s) in folder: {source_db_folder}")
    
    # Create target database if it doesn't exist
    if not os.path.exists(target_db_filename):
        print(f"[INFO] Target database '{target_db_filename}' not found. Creating a new one...")
        conn = sqlite3.connect(target_db_filename)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS mails (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                url TEXT NOT NULL UNIQUE,
                html_content TEXT NOT NULL,
                saved_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
    
    # Open target database
    target_conn = sqlite3.connect(target_db_filename)
    target_cursor = target_conn.cursor()
    
    for source_db_filename in source_db_filenames:
        print(f"[INFO] Merging from '{source_db_filename}'...")
        
        # Open source database
        source_conn = sqlite3.connect(source_db_filename)
        source_cursor = source_conn.cursor()
        
        # Read data from source mails table (ignore id)
        try:
            source_cursor.execute("SELECT title, url, html_content FROM mails")
            mails = source_cursor.fetchall()
            
            for title, url, html_content in mails:
                try:
                    # Attempt to insert new record (let target database generate new id)
                    target_cursor.execute("""
                        INSERT OR IGNORE INTO mails (title, url, html_content) VALUES (?, ?, ?)
                    """, (title, url, html_content))
                    if target_cursor.rowcount > 0:
                        print(f"[INFO] Inserted email: {title} with URL: {url}")
                    else:
                        print(f"[INFO] Skipping duplicate email with URL: {url}")
                except sqlite3.Error as e:
                    print(f"[ERROR] Failed to insert email with URL: {url}. Error: {e}")
            
            source_conn.close()
        except sqlite3.Error as e:
            print(f"[ERROR] Failed to read mails table from '{source_db_filename}'. Error: {e}")
            source_conn.close()
    
    # Update sqlite_sequence table to match the max id
    target_cursor.execute("SELECT MAX(id) FROM mails")
    max_id = target_cursor.fetchone()[0]
    if max_id is not None:  # If mails table has data
        target_cursor.execute("UPDATE sqlite_sequence SET seq = ? WHERE name = 'mails'", (max_id,))
        print(f"[INFO] sqlite_sequence updated to: {max_id}")
    else:  # If mails table is empty, reset sequence
        target_cursor.execute("DELETE FROM sqlite_sequence WHERE name = 'mails'")
        print(f"[INFO] sqlite_sequence reset for mails table.")

    # Commit changes and close target database
    target_conn.commit()
    target_conn.close()
    print("[SUCCESS] All mails have been merged into the target database.")

# Example usage: pass source folder and target database path
source_folder = "/mnt/c/Users/Haku/OneDrive/Temp/20250606 - lkml-dataset"  # Replace with source folder path
target_db = "/mnt/c/Users/Haku/OneDrive/Temp/lkml-data-2014-2024.db"               # Target database filename
merge_mail_tables(source_folder, target_db)
