import sqlite3
import requests
import time
from tqdm import tqdm

def query_and_filter_emails(filtered_db_filename, new_db_filename):
    """
    Process ~9000 emails with rate limiting to avoid overloading the Ollama server.
    
    Args:
        filtered_db_filename (str): Path to source SQLite database
        new_db_filename (str): Path for destination SQLite database
    """
    # Database setup
    new_conn = sqlite3.connect(new_db_filename)
    new_cursor = new_conn.cursor()
    new_cursor.execute("""
        CREATE TABLE IF NOT EXISTS relevant_mails (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            url TEXT NOT NULL UNIQUE,
            html_content TEXT NOT NULL,
            saved_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Connect to source database
    filtered_conn = sqlite3.connect(filtered_db_filename)
    filtered_cursor = filtered_conn.cursor()
    filtered_cursor.execute("SELECT title, url, html_content FROM mails")
    emails = filtered_cursor.fetchall()

    # Ollama configuration
    OLLAMA_URL = "http://192.168.15.102:11434/api/chat"
    MODEL = "deepseek-v2:16B"
    REQUEST_DELAY = 0.5  # 500ms between requests
    MAX_RETRIES = 3
    TIMEOUT = 30  # seconds

    # Processing loop with progress bar
    processed_count = 0
    with tqdm(emails, desc="Processing Emails") as pbar:
        for title, url, html_content in pbar:
            payload = {
                "model": MODEL,
                "messages": [{
                    "role": "user",
                    "content": f"Email content: {html_content[:10000]}...\n\nIs this specifically about Linux kernel security (not peripheral components)? Reply ONLY with 'True' or 'False'."
                }],
                "stream": False,
                "options": {"temperature": 0.1}
            }

            # Retry mechanism
            for attempt in range(MAX_RETRIES):
                try:
                    response = requests.post(
                        OLLAMA_URL,
                        json=payload,
                        timeout=TIMEOUT
                    )
                    
                    if response.status_code == 200:
                        answer = response.json()['message']['content'].strip().lower()
                        if "true" in answer:
                            new_cursor.execute(
                                "INSERT OR IGNORE INTO relevant_mails (title, url, html_content) VALUES (?, ?, ?)",
                                (title, url, html_content)
                            )
                            processed_count += 1
                            pbar.set_postfix({'Matched': processed_count})
                        break  # Success - exit retry loop
                    else:
                        print(f"\n[WARN] Attempt {attempt+1}: Status {response.status_code} for {url[:50]}...")
                        if attempt == MAX_RETRIES - 1:
                            print(f"[ERROR] Failed after {MAX_RETRIES} attempts: {title}")
                        time.sleep(2 ** attempt)  # Exponential backoff

                except Exception as e:
                    print(f"\n[ERROR] Attempt {attempt+1}: {str(e)}")
                    if attempt == MAX_RETRIES - 1:
                        print(f"[CRITICAL] Failed processing: {title}")
                    time.sleep(5 * (attempt + 1))  # Longer delay for exceptions

            # Rate limiting
            time.sleep(REQUEST_DELAY)

    # Finalization
    new_conn.commit()
    filtered_conn.close()
    new_conn.close()
    print(f"\n[SUCCESS] Processed {len(emails)} emails. Saved {processed_count} relevant emails to {new_db_filename}")

if __name__ == "__main__":
    query_and_filter_emails(
        filtered_db_filename="/home/haku/Workspace-WSL/lkml-mining/database/filtered_mails_security.db",
        new_db_filename="/home/haku/Workspace-WSL/lkml-mining/database/filtered_mails_gpt.db"
    )
