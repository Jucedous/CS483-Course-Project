import sqlite3
from bs4 import BeautifulSoup
from typing import List, Dict, Optional, Set, Tuple
import openai
import time
import re
import os
from datetime import datetime
import logging
import json
from dotenv import load_dotenv

# Configuration
with open('config/config.json') as config_file:
    config = json.load(config_file)

SQLITE_DB_PATH = config['sqlite_db_path']
PATCH_ANALYSUS_LOG_PATH = config['patch_analysis_log_path']

load_dotenv("config/openai.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")

openai.api_key = OPENAI_API_KEY

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Output to console
        logging.FileHandler(PATCH_ANALYSUS_LOG_PATH)  # Output to file
    ]
)
logger = logging.getLogger(__name__)

class ThreadAnalyzer:
    def __init__(self, db_path: str):
        """Initialize the analyzer with database connection"""
        logger.info(f"Initializing SQLite database connection: {db_path}")
        try:
            self.conn = sqlite3.connect(db_path)
            self.cursor = self.conn.cursor()
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Database connection failed: {str(e)}")
            raise

    def __del__(self):
        """Clean up resources"""
        if hasattr(self, 'conn'):
            self.conn.close()
            logger.info("Database connection closed")

    def normalize_lkml_url(self, url: str) -> str:
        """
        Normalize LKML email URL format
        Converts various URL formats to standard: https://lkml.org/lkml/YYYY/M/D/ID
        """
        if not url or not url.strip():
            logger.warning("Received empty URL")
            return ""
        
        url = url.strip().rstrip('/')
        
        # Handle double slashes
        url = url.replace("//lkml/", "/lkml/")
        
        # Extract path
        path = re.sub(r'^https?://[^/]+', '', url)
        path = path.strip('/')
        
        # Extract date and message ID
        match = re.search(r'lkml/(\d{4})[/-](\d{1,2})[/-](\d{1,2})[/-]?(\d+)/?$', path)
        if match:
            year, month, day, msg_id = match.groups()
            return f"https://lkml.org/lkml/{year}/{month}/{day}/{msg_id}"  # No trailing slash
        
        logger.warning(f"Unrecognized URL format: {url}")
        return url

    def get_email_by_url(self, url: str) -> Optional[Dict]:
        """Get single email by URL"""
        normalized_url = self.normalize_lkml_url(url)
        logger.debug(f"Querying email: {url}")
        try:
            self.cursor.execute(
                "SELECT title, url, html_content FROM mails WHERE url = ? OR url = ?", 
                (normalized_url, normalized_url + "/")
            )
            result = self.cursor.fetchone()
            if result:
                logger.debug(f"Found email: {result[0][:50]}...")
                return {
                    "title": result[0], 
                    "url": result[1], 
                    "html_content": result[2],
                    "date": self.extract_date_from_url(result[1])
                }
            logger.warning(f"Email not found: {url}")
            return None
        except Exception as e:
            logger.error(f"Email query failed: {str(e)}")
            return None

    def parse_relationships_from_html(self, html_content: str) -> List[Tuple[str, str]]:
        """
        Parse email relationships from HTML content
        Returns:
            - List of email relationships [(parent_url, child_url)]
        """
        logger.debug("Parsing email relationships...")
        soup = BeautifulSoup(html_content, "html.parser")
        relationships = []

        def traverse_threadlist(threadlist, parent_url=None):
            """Recursively parse email relationships"""
            for li in threadlist.find_all("li", recursive=False):
                link = li.find("a")
                if not link:
                    continue

                child_url = self.normalize_lkml_url(link.get("href", "").strip())
                if not child_url:
                    continue

                if not child_url.startswith('http'):
                    child_url = f"https://lore.kernel.org/lkml/{child_url}"
                    logger.debug(f"Converted relative URL to absolute: {child_url}")

                if parent_url:
                    relationships.append((parent_url, child_url))
                    logger.debug(f"Found relationship: {parent_url} → {child_url}")

                child_ul = li.find("ul")
                if child_ul:
                    traverse_threadlist(child_ul, parent_url=child_url)

        threadlist = soup.find("ul", class_="threadlist")
        if threadlist:
            logger.debug("Found threadlist element, parsing relationships")
            traverse_threadlist(threadlist)
        else:
            logger.debug("Threadlist element not found")

        logger.info(f"Found {len(relationships)} email relationships")
        return relationships

    def build_thread_graph(self, start_url: str) -> Dict[str, Dict]:
        """
        Build complete email thread graph
        Returns: {url: {email_data, parents: [], children: []}}
        """
        logger.info(f"Building email thread graph, starting URL: {start_url}")
        graph = {}
        visited = set()
        
        def traverse(url: str):
            if url in visited:
                logger.debug(f"Skipping already visited URL: {url}")
                return
            visited.add(url)
            
            logger.info(f"Processing email: {url}")
            email = self.get_email_by_url(url)
            if not email:
                logger.warning(f"Email not found, skipping: {url}")
                return
                
            if url not in graph:
                graph[url] = {
                    "email": email,
                    "parents": set(),
                    "children": set()
                }
                logger.debug(f"Added to graph: {email['title'][:50]}...")
            
            logger.debug("Parsing email relationships...")
            relationships = self.parse_relationships_from_html(email["html_content"])
            
            for parent_url, child_url in relationships:
                logger.debug(f"Processing relationship: {parent_url} → {child_url}")
                
                for rel_url in [parent_url, child_url]:
                    if rel_url not in graph:
                        logger.debug(f"Checking if related email exists: {rel_url}")
                        rel_email = self.get_email_by_url(rel_url)
                        if rel_email:
                            graph[rel_url] = {
                                "email": rel_email,
                                "parents": set(),
                                "children": set()
                            }
                            logger.debug(f"Added related email to graph: {rel_email['title'][:50]}...")
                
                if parent_url in graph and child_url in graph:
                    graph[parent_url]["children"].add(child_url)
                    graph[child_url]["parents"].add(parent_url)
                    logger.debug(f"Established relationship: {parent_url} → {child_url}")
                else:
                    logger.warning(f"Related email not found, skipping: {parent_url} → {child_url}")
            
            for child_url in graph[url]["children"]:
                logger.debug(f"Recursively processing child email: {child_url}")
                traverse(child_url)
            
            for parent_url in graph[url]["parents"]:
                logger.debug(f"Recursively processing parent email: {parent_url}")
                traverse(parent_url)
        
        traverse(start_url)
        logger.info(f"Graph construction complete, {len(graph)} emails in total")
        return graph

    def get_chronological_thread(self, graph: Dict[str, Dict]) -> List[Dict]:
        """Sort thread emails chronologically"""
        logger.info("Sorting emails chronologically...")
        emails = [node["email"] for node in graph.values()]
        sorted_emails = sorted(emails, key=lambda x: x["date"])
        logger.info(f"Sorting complete, earliest: {sorted_emails[0]['date']}, latest: {sorted_emails[-1]['date']}")
        return sorted_emails

    @staticmethod
    def extract_text_from_html(html: str) -> str:
        """Extract plain text from HTML"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            for elem in soup(['script', 'style', 'header', 'footer', 'blockquote']):
                elem.decompose()
            text = soup.get_text(separator='\n', strip=True)
            logger.debug(f"Extracted text, length: {len(text)} chars")
            return text
        except Exception as e:
            logger.error(f"HTML parsing failed: {str(e)}")
            return ""

    @staticmethod
    def extract_date_from_url(url: str) -> datetime:
        """Extract date from URL"""
        try:
            match = re.search(r'lkml/(\d{4})/(\d{1,2})/(\d{1,2})/', url)
            if match:
                year, month, day = map(int, match.groups())
                date = datetime(year, month, day)
                logger.debug(f"Extracted date from URL: {date}")
                return date
            logger.warning(f"Could not extract date from URL: {url}")
            return datetime.min
        except Exception as e:
            logger.error(f"Date extraction failed: {str(e)}")
            return datetime.min

    def analyze_email(self, email: Dict) -> str:
        """
        Analyze single email using GPT
        Returns analysis result as string
        """
        logger.info(f"Analyzing email: {email['title'][:50]}...")
        
        prompt = f"""
        [Role]
        You are a Linux kernel maintainer reviewing patch discussions on mailing lists.

        [Email Info]
        Title: {email.get('title', 'Untitled')}
        Date: {email.get('date', 'Unknown date')}
        URL: {email.get('url', 'No URL')}

        [Email Content]
        {self.extract_text_from_html(email.get('html_content', ''))}

        [Analysis Requirements]
        1. Core Contribution: What technical change is proposed?
        2. Technical Details: Which kernel subsystems/mechanisms are involved?
        3. Motivation: Why is this change needed? What problem does it solve?
        4. Controversy: What objections or issues were raised?
        5. Solution: What was the final adopted solution?
        6. Impact Assessment: How will this change affect the kernel?

        [Output Format]
        Use this Markdown format:
        ### Email Summary
        - **Core Change**: 
        - **Affected Subsystems**: 
        - **Change Motivation**: 
        - **Technical Controversy**: 
        - **Final Solution**: 
        - **Potential Impact**: 
        """
        logger.debug(f"Prepared analysis prompt, length: {len(prompt)}")

        try:
            logger.info("Calling GPT API for analysis...")
            start_time = time.time()
            
            response = openai.ChatCompletion.create(
                api_base=OPENAI_API_BASE,
                model="deepseek-v3-250324",
                messages=[
                    {"role": "system", "content": "You are skilled at analyzing Linux kernel technical discussions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            
            analysis = response.choices[0].message.content
            elapsed = time.time() - start_time
            logger.info(f"Analysis complete, took: {elapsed:.2f}s, result length: {len(analysis)}")
            return analysis
        except Exception as e:
            logger.error(f"GPT analysis failed: {str(e)}")
            return f"Analysis failed: {str(e)}"

    def summarize_thread(self, individual_analyses: List[str], graph: Dict[str, Dict]) -> str:
        """
        Generate summary of the entire email thread
        """
        logger.info("Generating thread summary...")
        
        timeline = "\n".join(
            f"- {analysis.split('### Email Summary')[0].strip()}" 
            for analysis in individual_analyses
        )
        
        prompt = f"""
        [Context]
        You're reviewing a complete Linux kernel patch discussion. Here's the timeline:

        {timeline}

        [Thread Structure]
        {self.graph_to_text(graph)}

        [Task]
        Generate a professional technical report containing:

        1. Patch Evolution
           - Initial technical proposal
           - Key modification points
           - Final adopted solution

        2. Technical Controversy Analysis
           - Main points of contention
           - Positions of different parties
           - Solution evolution

        3. Merge Process Evaluation
           - Discussion duration and intensity
           - Maintainer involvement
           - Key decision points

        [Requirements]
        - Use professional terminology
        - Reference specific arguments from emails
        - Maintain objective neutrality
        - Output in Markdown format
        """
        logger.debug(f"Prepared summary prompt, length: {len(prompt)}")

        try:
            logger.info("Calling GPT API for summary...")
            start_time = time.time()
            
            response = openai.ChatCompletion.create(
                api_base=OPENAI_API_BASE,
                model="deepseek-v3-250324",
                messages=[
                    {"role": "system", "content": "You're a senior kernel maintainer skilled in technical decision analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            summary = response.choices[0].message.content
            elapsed = time.time() - start_time
            logger.info(f"Summary complete, took: {elapsed:.2f}s, result length: {len(summary)}")
            return summary
        except Exception as e:
            logger.error(f"GPT summary failed: {str(e)}")
            return f"Summary failed: {str(e)}"

    def graph_to_text(self, graph: Dict[str, Dict]) -> str:
        """Convert thread graph to readable text"""
        logger.debug("Converting graph to text...")
        lines = []
        for url, node in graph.items():
            lines.append(f"- {node['email']['title']} ({node['email']['date']})")
            if node['parents']:
                lines.append(f"  ← Replies to: {len(node['parents'])} parent emails")
            if node['children']:
                lines.append(f"  → Generated: {len(node['children'])} replies")
        return "\n".join(lines)

    def analyze_patch_merge(self, start_url: str):
        """
        Analyze complete patch merge process
        """
        logger.info(f"=== Starting patch thread analysis ===")
        logger.info(f"Starting URL: {start_url}")
        
        try:
            # 1. Build thread graph
            logger.info("Phase 1: Building email thread graph")
            graph = self.build_thread_graph(start_url)
            if not graph:
                logger.error("Failed to build graph, terminating analysis")
                return
            
            logger.info(f"Found {len(graph)} related emails")
            
            # 2. Get emails in chronological order
            logger.info("Phase 2: Sorting emails chronologically")
            thread_emails = self.get_chronological_thread(graph)
            
            # 3. Analyze individual emails
            logger.info("Phase 3: Analyzing individual emails")
            individual_analyses = []
            for i, email in enumerate(thread_emails, 1):
                logger.info(f"Progress: {i}/{len(thread_emails)} - {email['title'][:50]}...")
                analysis = self.analyze_email(email)
                individual_analyses.append(analysis)
                
                # Avoid API rate limiting
                time.sleep(1)
            
            # 4. Generate summary
            logger.info("Phase 4: Generating comprehensive report")
            summary = self.summarize_thread(individual_analyses, graph)
            
            # 5. Output results
            # TODO: Uncomment to enable detailed logging
            # logger.info("\n=== Patch Merge Analysis Report ===")
            # logger.info(summary[:500] + "...")  # Only print first 500 chars
            
            # Save results
            self.save_analysis_report(start_url, individual_analyses, summary, graph)
            logger.info("\n=== Analysis completed successfully ===")
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}", exc_info=True)
            raise

    def save_analysis_report(self, start_url: str, analyses: List[str], summary: str, graph: Dict):
        """Save analysis results to file"""
        filename = f"patch_analysis_{start_url.split('/')[-4]}-{start_url.split('/')[-3]}-{start_url.split('/')[-2]}-{start_url.split('/')[-1]}.md"
        logger.info(f"Saving report to file: {filename}")
        
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"# Linux Kernel Patch Merge Analysis Report\n\n")
                f.write(f"- **Root Email**: [{start_url}]({start_url})\n")
                f.write(f"- **Emails Analyzed**: {len(graph)}\n")
                f.write(f"- **Time Span**: {self.get_time_span(graph)}\n\n")
                
                f.write("## Email Thread Structure\n```\n")
                f.write(self.graph_to_text(graph))
                f.write("\n```\n\n")
                
                f.write("## Detailed Email Analysis\n")
                for i, analysis in enumerate(analyses, 1):
                    f.write(f"\n### Email {i}\n{analysis}\n")
                
                f.write("\n## Comprehensive Analysis\n")
                f.write(summary)
            
            logger.info("Report saved successfully")
        except Exception as e:
            logger.error(f"Failed to save report: {str(e)}")

    def get_time_span(self, graph: Dict) -> str:
        """Get thread time span"""
        dates = [node["email"]["date"] for node in graph.values()]
        min_date = min(dates)
        max_date = max(dates)
        return f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"

if __name__ == "__main__":
    try:
        analyzer = ThreadAnalyzer(SQLITE_DB_PATH)
        
        # Example: Analyze specific patch email
        target_url = "https://lkml.org/lkml/2024/6/2/360"  # Replace with
        analyzer.analyze_patch_merge(target_url)
    except Exception as e:
        logging.error(f"Failed: {str(e)}", exc_info=True)
