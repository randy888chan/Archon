import os
import sys
import asyncio
import threading
import subprocess
import requests
import json
import time
from typing import List, Dict, Any, Optional, Callable
from xml.etree import ElementTree
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse
from dotenv import load_dotenv
from openai import AsyncOpenAI
import re
import html2text
import BeautifulSoup

# Add the parent directory to sys.path to allow importing from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import get_env_var, get_clients

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

load_dotenv()

# Initialize embedding and Supabase clients
embedding_client, supabase = get_clients()

# Define the embedding model for embedding the documentation for RAG
embedding_model = get_env_var('EMBEDDING_MODEL') or 'text-embedding-3-small'

# LLM client setup
llm_client = None
base_url = get_env_var('BASE_URL') or 'https://api.openai.com/v1'
api_key = get_env_var('LLM_API_KEY') or 'no-api-key-provided'
provider = get_env_var('LLM_PROVIDER') or 'OpenAI'

# Setup OpenAI client for LLM
if provider == "Ollama":
    if api_key == "NOT_REQUIRED":
        api_key = "ollama"  # Use a dummy key for Ollama
    llm_client = AsyncOpenAI(base_url=base_url, api_key=api_key)
else:
    llm_client = AsyncOpenAI(base_url=base_url, api_key=api_key)

@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]

class CrawlProgressTracker:
    """Class to track progress of the crawling process."""
    
    def __init__(self, 
                 progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None):
        """Initialize the progress tracker.
        
        Args:
            progress_callback: Function to call with progress updates
        """
        self.progress_callback = progress_callback
        self.urls_found = 0
        self.urls_processed = 0
        self.urls_succeeded = 0
        self.urls_failed = 0
        self.chunks_stored = 0
        self.logs = []
        self.is_running = False
        self.start_time = None
        self.end_time = None
    
    def log(self, message: str):
        """Add a log message and update progress."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.logs.append(log_entry)
        print(message)  # Also print to console
        
        # Call the progress callback if provided
        if self.progress_callback:
            self.progress_callback(self.get_status())
    
    def start(self):
        """Mark the crawling process as started."""
        self.is_running = True
        self.start_time = datetime.now()
        self.log("Crawling process started")
        
        # Call the progress callback if provided
        if self.progress_callback:
            self.progress_callback(self.get_status())
    
    def complete(self):
        """Mark the crawling process as completed."""
        self.is_running = False
        self.end_time = datetime.now()
        duration = self.end_time - self.start_time if self.start_time else None
        duration_str = str(duration).split('.')[0] if duration else "unknown"
        self.log(f"Crawling process completed in {duration_str}")
        
        # Call the progress callback if provided
        if self.progress_callback:
            self.progress_callback(self.get_status())
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the crawling process."""
        return {
            "is_running": self.is_running,
            "urls_found": self.urls_found,
            "urls_processed": self.urls_processed,
            "urls_succeeded": self.urls_succeeded,
            "urls_failed": self.urls_failed,
            "chunks_stored": self.chunks_stored,
            "progress_percentage": (self.urls_processed / self.urls_found * 100) if self.urls_found > 0 else 0,
            "logs": self.logs,
            "start_time": self.start_time,
            "end_time": self.end_time
        }
    
    @property
    def is_completed(self) -> bool:
        """Return True if the crawling process is completed."""
        return not self.is_running and self.end_time is not None
    
    @property
    def is_successful(self) -> bool:
        """Return True if the crawling process completed successfully."""
        return self.is_completed and self.urls_failed == 0 and self.urls_succeeded > 0

def chunk_text(text: str, chunk_size: int = 5000) -> List[str]:
    """Split text into chunks, respecting code blocks and paragraphs."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # Calculate end position
        end = start + chunk_size

        # If we're at the end of the text, just take what's left
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        # Try to find a code block boundary first (```)
        chunk = text[start:end]
        code_block = chunk.rfind('```')
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block

        # If no code block, try to break at a paragraph
        elif '\n\n' in chunk:
            # Find the last paragraph break
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_break

        # If no paragraph break, try to break at a sentence
        elif '. ' in chunk:
            # Find the last sentence break
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_period + 1

        # Extract chunk and clean it up
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position for next chunk
        start = max(start + 1, end)

    return chunks

async def get_title_and_summary(chunk: str, url: str) -> Dict[str, str]:
    """Extract title and summary using GPT-4."""
    system_prompt = """You are an AI that extracts titles and summaries from documentation chunks.
    Return a JSON object with 'title' and 'summary' keys.
    For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.
    For the summary: Create a concise summary of the main points in this chunk.
    Keep both title and summary concise but informative."""
    
    try:
        response = await llm_client.chat.completions.create(
            model=get_env_var("PRIMARY_MODEL") or "gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"URL: {url}\n\nContent:\n{chunk[:1000]}..."}  # Send first 1000 chars for context
            ],
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error getting title and summary: {e}")
        return {"title": "Error processing title", "summary": "Error processing summary"}

async def get_embedding(text: str) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await embedding_client.embeddings.create(
            model=embedding_model,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error

async def process_chunk(chunk: str, chunk_number: int, url: str) -> ProcessedChunk:
    """Process a single chunk of text."""
    # Get title and summary
    extracted = await get_title_and_summary(chunk, url)
    
    # Get embedding
    embedding = await get_embedding(chunk)
    
    # Create metadata
    metadata = {
        "source": "pydantic_ai_docs",
        "chunk_size": len(chunk),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "url_path": urlparse(url).path
    }
    
    return ProcessedChunk(
        url=url,
        chunk_number=chunk_number,
        title=extracted['title'],
        summary=extracted['summary'],
        content=chunk,  # Store the original chunk content
        metadata=metadata,
        embedding=embedding
    )

async def insert_chunk(chunk: ProcessedChunk):
    """Insert a processed chunk into Supabase."""
    try:
        data = {
            "url": chunk.url,
            "chunk_number": chunk.chunk_number,
            "title": chunk.title,
            "summary": chunk.summary,
            "content": chunk.content,
            "metadata": chunk.metadata,
            "embedding": chunk.embedding
        }
        
        result = supabase.table("site_pages").insert(data).execute()
        print(f"Inserted chunk {chunk.chunk_number} for {chunk.url}")
        return result
    except Exception as e:
        print(f"Error inserting chunk: {e}")
        return None

async def process_and_store_document(url: str, markdown: str, tracker: Optional[CrawlProgressTracker] = None):
    """Process a document and store its chunks in parallel."""
    # Split into chunks
    chunks = chunk_text(markdown)
    
    if tracker:
        tracker.log(f"Split document into {len(chunks)} chunks for {url}")
        # Ensure UI gets updated
        if tracker.progress_callback:
            tracker.progress_callback(tracker.get_status())
    else:
        print(f"Split document into {len(chunks)} chunks for {url}")
    
    # Process chunks in parallel
    tasks = [
        process_chunk(chunk, i, url) 
        for i, chunk in enumerate(chunks)
    ]
    processed_chunks = await asyncio.gather(*tasks)
    
    if tracker:
        tracker.log(f"Processed {len(processed_chunks)} chunks for {url}")
        # Ensure UI gets updated
        if tracker.progress_callback:
            tracker.progress_callback(tracker.get_status())
    else:
        print(f"Processed {len(processed_chunks)} chunks for {url}")
    
    # Store chunks in parallel
    insert_tasks = [
        insert_chunk(chunk) 
        for chunk in processed_chunks
    ]
    await asyncio.gather(*insert_tasks)
    
    if tracker:
        tracker.chunks_stored += len(processed_chunks)
        tracker.log(f"Stored {len(processed_chunks)} chunks for {url}")
        # Ensure UI gets updated
        if tracker.progress_callback:
            tracker.progress_callback(tracker.get_status())
    else:
        print(f"Stored {len(processed_chunks)} chunks for {url}")

class SmartContentExtractor:
    def __init__(self):
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = False
        self.html_converter.ignore_images = False
        self.html_converter.ignore_tables = False
        self.html_converter.body_width = 0
        
        # Define content selectors for each framework
        self.content_selectors = {
            "pydantic_ai": [
                "article",
                "main .markdown-body",
                ".content-area",
                ".documentation-content"
            ],
            "langchain": [
                "article",
                ".doc-content",
                "main article",
                ".markdown-body",
                ".content"
            ],
            "crewai": [
                ".markdown-content"
                "article",
                ".docs-content",
                "main .content",
                ".documentation"
            ]
        }
        
        # Elements to remove (navigation, sidebars, etc.)
        self.elements_to_remove = [
            'nav', 'header', 'footer', 
            '.sidebar', '.navigation', '.nav',
            '.breadcrumb', '.toc', '.table-of-contents',
            '.prev-next', '.pagination',
            '.header', '.top-bar', '.menu',
            'script', 'style', 'noscript'
        ]

    def extract_main_content(self, html: str, framework: str, url: str) -> str:
        """Extract only the main documentation content, filtering out navigation and sidebars."""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove unwanted elements
        for selector in self.elements_to_remove:
            for element in soup.select(selector):
                element.decompose()
        
        # Try framework-specific selectors first
        content_element = None
        if framework in self.content_selectors:
            for selector in self.content_selectors[framework]:
                content_element = soup.select_one(selector)
                if content_element:
                    print(f"Found content using selector: {selector} for {url}")
                    break
        
        # Fallback to common content selectors
        if not content_element:
            fallback_selectors = [
                'main', 'article', '.content', '.main-content',
                '.documentation', '.docs', '.markdown-body',
                '#content', '#main-content'
            ]
            for selector in fallback_selectors:
                content_element = soup.select_one(selector)
                if content_element:
                    print(f"Found content using fallback selector: {selector} for {url}")
                    break
        
        # If still no content found, use body but remove common navigation elements
        if not content_element:
            content_element = soup.find('body')
            if content_element:
                # Remove additional navigation elements from body
                for nav_elem in content_element.select('.nav, .navbar, .menu, .sidebar, .header, .footer'):
                    nav_elem.decompose()
        
        if content_element:
            # Convert to markdown
            clean_html = str(content_element)
            markdown = self.html_converter.handle(clean_html)
            
            # Clean up markdown
            markdown = self._clean_markdown(markdown)
            return markdown
        
        print(f"Warning: No content found for {url}, falling back to full page")
        return self.html_converter.handle(html)

    def _clean_markdown(self, markdown: str) -> str:
        """Clean up the extracted markdown."""
        import re
        
        # Remove excessive newlines
        markdown = re.sub(r'\n{3,}', '\n\n', markdown)
        
        # Remove navigation links patterns
        markdown = re.sub(r'^\* \[.*?\]\(.*?\)\s*$', '', markdown, flags=re.MULTILINE)
        
        # Remove breadcrumb patterns
        markdown = re.sub(r'^.*?».*?».*?$', '', markdown, flags=re.MULTILINE)
        
        # Remove "Edit this page" type links
        markdown = re.sub(r'^\[Edit.*?\].*?$', '', markdown, flags=re.MULTILINE)
        
        # Clean up remaining whitespace
        markdown = re.sub(r'^\s*$', '', markdown, flags=re.MULTILINE)
        markdown = re.sub(r'\n{3,}', '\n\n', markdown)
        
        return markdown.strip()

#When crawling certain dropdowns from pydantic ai docs, it crawls the line numbers before the actual code, 
#which is unnecessary. Thus, it would be better for us to remove them
def remove_line_numbers(text):
    """Remove line numbers from crawled content"""
    # Pattern to match line numbers (sequences of digits on their own lines)
    patterns = [
        r'^\s*\d+\s*$',  # Just digits on a line
        r'^\s*\d+\s*\n',  # Digits followed by newline
        r'\n\s*\d+\s*\n',  # Digits between newlines
        r'\n\s*\d+\s*$',  # Digits at end after newline
    ]
    
    cleaned_text = text
    for pattern in patterns:
        cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.MULTILINE)
    
    # Remove sequences of consecutive numbers
    cleaned_text = re.sub(r'(\n\s*\d+\s*){3,}', '\n', cleaned_text, flags=re.MULTILINE)
    
    # Clean up extra whitespace
    cleaned_text = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_text)
    cleaned_text = cleaned_text.strip()
    
    return cleaned_text   
    
# Updated url crawling content to exclude headers, navigators, margins and other unnecessary information
def fetch_url_content_improved(url: str, framework: str = "unknown") -> str:
    """Fetch and extract main content from URL."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    extractor = SmartContentExtractor()
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Extract main content instead of converting entire page
        markdown = extractor.extract_main_content(response.text, framework, url)
        
        cleaned_markdown = remove_line_numbers(markdown)

        with open("html_crawl.md", 'w') as file:
            file.write(cleaned_markdown)
        return cleaned_markdown
    except Exception as e:
        raise Exception(f"Error fetching {url}: {str(e)}")

async def crawl_parallel_with_requests(urls: List[str], tracker: Optional[CrawlProgressTracker] = None, max_concurrent: int = 5):
    """Crawl multiple URLs in parallel with a concurrency limit using direct HTTP requests."""
    # Create a semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_url(url: str):
        async with semaphore:
            if tracker:
                tracker.log(f"Crawling: {url}")
                # Ensure UI gets updated
                if tracker.progress_callback:
                    tracker.progress_callback(tracker.get_status())
            else:
                print(f"Crawling: {url}")
            
            try:
                # Use a thread pool to run the blocking HTTP request
                loop = asyncio.get_running_loop()
                if tracker:
                    tracker.log(f"Fetching content from: {url}")
                else:
                    print(f"Fetching content from: {url}")
                markdown = await loop.run_in_executor(None, fetch_url_content, url)
                
                if markdown:
                    if tracker:
                        tracker.urls_succeeded += 1
                        tracker.log(f"Successfully crawled: {url}")
                        # Ensure UI gets updated
                        if tracker.progress_callback:
                            tracker.progress_callback(tracker.get_status())
                    else:
                        print(f"Successfully crawled: {url}")
                    
                    await process_and_store_document(url, markdown, tracker)
                else:
                    if tracker:
                        tracker.urls_failed += 1
                        tracker.log(f"Failed: {url} - No content retrieved")
                        # Ensure UI gets updated
                        if tracker.progress_callback:
                            tracker.progress_callback(tracker.get_status())
                    else:
                        print(f"Failed: {url} - No content retrieved")
            except Exception as e:
                if tracker:
                    tracker.urls_failed += 1
                    tracker.log(f"Error processing {url}: {str(e)}")
                    # Ensure UI gets updated
                    if tracker.progress_callback:
                        tracker.progress_callback(tracker.get_status())
                else:
                    print(f"Error processing {url}: {str(e)}")
            finally:
                if tracker:
                    tracker.urls_processed += 1
                    # Ensure UI gets updated
                    if tracker.progress_callback:
                        tracker.progress_callback(tracker.get_status())

        time.sleep(2)
    
    # Process all URLs in parallel with limited concurrency
    if tracker:
        tracker.log(f"Processing {len(urls)} URLs with concurrency {max_concurrent}")
        # Ensure UI gets updated
        if tracker.progress_callback:
            tracker.progress_callback(tracker.get_status())
    else:
        print(f"Processing {len(urls)} URLs with concurrency {max_concurrent}")
    await asyncio.gather(*[process_url(url) for url in urls])

def get_pydantic_ai_docs_urls() -> List[str]:
    """Get URLs from Pydantic AI docs sitemap."""
    sitemap_url = "https://ai.pydantic.dev/sitemap.xml"
    try:
        response = requests.get(sitemap_url)
        response.raise_for_status()
        
        # Parse the XML
        root = ElementTree.fromstring(response.content)
        
        # Extract all URLs from the sitemap
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        urls = [loc.text for loc in root.findall('.//ns:loc', namespace)]
        
        return urls
    except Exception as e:
        print(f"Error fetching sitemap: {e}")
        return []

def clear_existing_records():
    """Clear all existing records with source='pydantic_ai_docs' from the site_pages table."""
    try:
        result = supabase.table("site_pages").delete().eq("metadata->>source", "pydantic_ai_docs").execute()
        print("Cleared existing pydantic_ai_docs records from site_pages")
        return result
    except Exception as e:
        print(f"Error clearing existing records: {e}")
        return None

async def main_with_requests(tracker: Optional[CrawlProgressTracker] = None):
    """Main function using direct HTTP requests instead of browser automation."""
    try:
        # Start tracking if tracker is provided
        if tracker:
            tracker.start()
        else:
            print("Starting crawling process...")
        
        # Clear existing records first
        if tracker:
            tracker.log("Clearing existing Pydantic AI docs records...")
        else:
            print("Clearing existing Pydantic AI docs records...")
        clear_existing_records()
        if tracker:
            tracker.log("Existing records cleared")
        else:
            print("Existing records cleared")
        
        # Get URLs from Pydantic AI docs
        if tracker:
            tracker.log("Fetching URLs from Pydantic AI sitemap...")
        else:
            print("Fetching URLs from Pydantic AI sitemap...")
        urls = get_pydantic_ai_docs_urls()
        
        if not urls:
            if tracker:
                tracker.log("No URLs found to crawl")
                tracker.complete()
            else:
                print("No URLs found to crawl")
            return
        
        if tracker:
            tracker.urls_found = len(urls)
            tracker.log(f"Found {len(urls)} URLs to crawl")
        else:
            print(f"Found {len(urls)} URLs to crawl")
        
        # Crawl the URLs using direct HTTP requests
        await crawl_parallel_with_requests(urls, tracker)
        
        # Mark as complete if tracker is provided
        if tracker:
            tracker.complete()
        else:
            print("Crawling process completed")
            
    except Exception as e:
        if tracker:
            tracker.log(f"Error in crawling process: {str(e)}")
            tracker.complete()
        else:
            print(f"Error in crawling process: {str(e)}")

def start_crawl_with_requests(progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> CrawlProgressTracker:
    """Start the crawling process using direct HTTP requests in a separate thread and return the tracker."""
    tracker = CrawlProgressTracker(progress_callback)
    
    def run_crawl():
        try:
            asyncio.run(main_with_requests(tracker))
        except Exception as e:
            print(f"Error in crawl thread: {e}")
            tracker.log(f"Thread error: {str(e)}")
            tracker.complete()
    
    # Start the crawling process in a separate thread
    thread = threading.Thread(target=run_crawl)
    thread.daemon = True
    thread.start()
    
    return tracker

if __name__ == "__main__":    
    # Run the main function directly
    print("Starting crawler...")
    asyncio.run(main_with_requests())
    print("Crawler finished.")
