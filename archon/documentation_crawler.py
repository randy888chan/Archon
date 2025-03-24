import os
import sys
import asyncio
import threading
import requests
import json
import time
from typing import List, Dict, Any, Optional, Callable, Type, Set, Union
from xml.etree import ElementTree
from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.parse import urlparse
from dotenv import load_dotenv
from openai import AsyncOpenAI
import re
import concurrent.futures
import logging
import uuid

# Try to import Crawl4ai but provide a fallback if it's not available
try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
    crawl4ai_available = True
except ImportError:
    crawl4ai_available = False
    print("Crawl4ai not available, falling back to requests-based crawling")

# Try to import html2text but provide a fallback if it's not available
try:
    import html2text
    html2text_available = True
except ImportError:
    html2text_available = False

# Try to import BeautifulSoup but provide a fallback if it's not available
try:
    from bs4 import BeautifulSoup
    bs4_available = True
except ImportError:
    bs4_available = False

# Try to import lxml but provide a fallback if it's not available
try:
    from lxml import etree
    lxml_available = True
except ImportError:
    lxml_available = False

# Add the project root to the Python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from utils.utils import get_env_var, get_clients

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

# Initialize HTML to Markdown converter if available
if html2text_available:
    html_converter = html2text.HTML2Text()
    html_converter.ignore_links = False
    html_converter.ignore_images = False
    html_converter.ignore_tables = False
    html_converter.body_width = 0  # No wrapping

def convert_html_to_text(html_content):
    """
    Convert HTML to plain text or markdown, with fallback methods if dependencies are not available.
    
    This function will try multiple approaches:
    1. Use html2text if available (preferred for markdown)
    2. Use BeautifulSoup if available
    3. Use a simple regex-based approach as a last resort
    
    Args:
        html_content: HTML content to convert
    
    Returns:
        str: Plain text or markdown version of the HTML
    """
    # Method 1: Use html2text (best for markdown)
    if html2text_available:
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = False
        h.ignore_tables = False
        h.body_width = 0
        return h.handle(html_content)
    
    # Method 2: Use BeautifulSoup
    if bs4_available:
        soup = BeautifulSoup(html_content, 'html.parser')
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        # Get text
        text = soup.get_text(separator='\n')
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        return text
    
    # Method 3: Simple regex-based fallback (least effective)
    print("Warning: Using simple regex for HTML parsing. Install html2text or beautifulsoup4 for better results.")
    # Remove HTML tags, but keep the content
    text = re.sub(r'<[^>]+>', ' ', html_content)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Replace multiple newlines with a single newline
    text = re.sub(r'\n+', '\n', text)
    return text.strip()

@dataclass
class DocumentationSource:
    """Configuration for a documentation source."""
    
    # Unique identifier for the source
    id: str
    
    # Display name shown in UI
    name: str
    
    # Description shown in the UI
    description: str
    
    # Function that returns list of URLs to crawl
    url_fetcher: Callable[[], List[str]]
    
    # Sitemap URL (if applicable)
    sitemap_url: Optional[str] = None
    
    # Custom parameters for this source
    parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProcessedChunk:
    """A processed chunk of documentation ready for storage."""
    
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]

class CrawlProgressTracker:
    """
    Tracks the progress of a crawl operation.
    """
    
    def __init__(self, progress_callback=None):
        self.total_urls = 0
        self.processed_count = 0
        self.success_count = 0
        self.error_count = 0
        self.logs = []
        self.is_running = True
        self.progress_callback = progress_callback
    
    def log(self, message):
        """Add a log message with timestamp."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.logs.append(log_entry)
        print(log_entry)
        
        # Call progress callback if provided
        if self.progress_callback:
            self.progress_callback(self.get_status())
    
    def add_log(self, message):
        """Add a log message (alias for log method)."""
        self.log(message)
    
    def increment_processed(self):
        """Increment the processed count."""
        self.processed_count += 1
        if self.progress_callback:
            self.progress_callback(self.get_status())
    
    def increment_success(self):
        """Increment the success count."""
        self.success_count += 1
        if self.progress_callback:
            self.progress_callback(self.get_status())
    
    def increment_error(self):
        """Increment the error count."""
        self.error_count += 1
        if self.progress_callback:
            self.progress_callback(self.get_status())
    
    def complete(self):
        """Mark the crawl as completed."""
        self.is_running = False
        self.log("Crawl completed!")
        if self.progress_callback:
            self.progress_callback(self.get_status())
    
    def get_status(self):
        """Get the status as a dictionary for UI display."""
        return {
            "urls_found": self.total_urls,
            "urls_processed": self.processed_count,
            "urls_succeeded": self.success_count,
            "urls_failed": self.error_count,
            "logs": self.logs,
            "is_running": self.is_running
        }

class DocumentationCrawler:
    """Base class for crawling documentation sources."""
    
    def __init__(self, source: DocumentationSource):
        """Initialize the crawler with a documentation source.
        
        Args:
            source: The documentation source to crawl
        """
        self.source = source
    
    def chunk_text(self, text: str, chunk_size: int = 5000) -> List[str]:
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
    
    async def get_title_and_summary(self, chunk: str, url: str) -> Dict[str, str]:
        """Extract title and summary using LLM."""
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
    
    async def get_embedding(self, text: str) -> List[float]:
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
    
    async def process_chunk(self, chunk: str, chunk_number: int, url: str) -> ProcessedChunk:
        """Process a single chunk of text."""
        # Get title and summary
        extracted = await self.get_title_and_summary(chunk, url)
        
        # Get embedding
        embedding = await self.get_embedding(chunk)
        
        # Create metadata
        metadata = {
            "source": self.source.id,
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
    
    async def insert_chunk(self, chunk: ProcessedChunk):
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
    
    async def process_and_store_document(self, url: str, markdown: str, tracker: Optional[CrawlProgressTracker] = None):
        """Process a document and store its chunks in parallel."""
        # Split into chunks
        chunks = self.chunk_text(markdown)
        
        if tracker:
            tracker.log(f"Split document into {len(chunks)} chunks for {url}")
            # Ensure UI gets updated
            if tracker.progress_callback:
                tracker.progress_callback(tracker.get_status())
        else:
            print(f"Split document into {len(chunks)} chunks for {url}")
        
        # Process chunks in parallel
        tasks = [
            self.process_chunk(chunk, i, url) 
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
            self.insert_chunk(chunk) 
            for chunk in processed_chunks
        ]
        await asyncio.gather(*insert_tasks)
        
        if tracker:
            tracker.success_count += len(processed_chunks)
            tracker.log(f"Stored {len(processed_chunks)} chunks for {url}")
            # Ensure UI gets updated
            if tracker.progress_callback:
                tracker.progress_callback(tracker.get_status())
        else:
            print(f"Stored {len(processed_chunks)} chunks for {url}")
    
    def fetch_url_content(self, url: str) -> str:
        """Fetch content from a URL using requests and convert to markdown."""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Convert HTML to Markdown
            markdown = convert_html_to_text(response.text)
            
            # Clean up the markdown
            markdown = re.sub(r'\n{3,}', '\n\n', markdown)  # Remove excessive newlines
            
            return markdown
        except Exception as e:
            raise Exception(f"Error fetching {url}: {str(e)}")
    
    async def crawl_parallel_with_requests(self, tracker: Optional[CrawlProgressTracker] = None, max_concurrent: int = 5):
        """Crawl multiple URLs in parallel with a concurrency limit using direct HTTP requests."""
        # Get URLs from the source
        urls = self.source.url_fetcher()
        
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
                    markdown = await loop.run_in_executor(None, self.fetch_url_content, url)
                    
                    if markdown:
                        if tracker:
                            tracker.success_count += 1
                            tracker.log(f"Successfully crawled: {url}")
                            # Ensure UI gets updated
                            if tracker.progress_callback:
                                tracker.progress_callback(tracker.get_status())
                        else:
                            print(f"Successfully crawled: {url}")
                        
                        await self.process_and_store_document(url, markdown, tracker)
                    else:
                        if tracker:
                            tracker.error_count += 1
                            tracker.log(f"Failed: {url} - No content retrieved")
                            # Ensure UI gets updated
                            if tracker.progress_callback:
                                tracker.progress_callback(tracker.get_status())
                        else:
                            print(f"Failed: {url} - No content retrieved")
                except Exception as e:
                    if tracker:
                        tracker.error_count += 1
                        tracker.log(f"Error processing {url}: {str(e)}")
                        # Ensure UI gets updated
                        if tracker.progress_callback:
                            tracker.progress_callback(tracker.get_status())
                    else:
                        print(f"Error processing {url}: {str(e)}")
                finally:
                    if tracker:
                        tracker.processed_count += 1
                        # Ensure UI gets updated
                        if tracker.progress_callback:
                            tracker.progress_callback(tracker.get_status())

            time.sleep(2)  # Small delay between URLs to avoid overwhelming servers
        
        # Process all URLs in parallel with limited concurrency
        if tracker:
            tracker.total_urls = len(urls)
            tracker.log(f"Processing {len(urls)} URLs with concurrency {max_concurrent}")
            # Ensure UI gets updated
            if tracker.progress_callback:
                tracker.progress_callback(tracker.get_status())
        else:
            print(f"Processing {len(urls)} URLs with concurrency {max_concurrent}")
        
        await asyncio.gather(*[process_url(url) for url in urls])
    
    def clear_existing_records(self):
        """Clear existing records for this source from the database."""
        source_id = self.source.id
        print(f"Clearing existing {self.source.name} records...")
        try:
            # Delete records with matching source ID in metadata
            response = supabase.table("site_pages").delete().eq("metadata->>source", source_id).execute()
            print(f"Cleared existing {source_id} records from site_pages")
            print("Existing records cleared")
        except Exception as e:
            print(f"Error clearing records: {str(e)}")
            raise
    
    async def run_crawler(self, tracker: Optional[CrawlProgressTracker] = None):
        """Main function to run the crawler."""
        try:
            # Start tracking if tracker is provided
            if tracker:
                tracker.log(f"Clearing existing {self.source.name} records...")
            else:
                print(f"Starting crawling process for {self.source.name}...")
            
            # Clear existing records first
            if tracker:
                tracker.log(f"Clearing existing {self.source.name} records...")
            else:
                print(f"Clearing existing {self.source.name} records...")
            
            self.clear_existing_records()
            
            if tracker:
                tracker.log("Existing records cleared")
            else:
                print("Existing records cleared")
            
            # Crawl URLs using direct HTTP requests
            await self.crawl_parallel_with_requests(tracker)
            
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
    
    def start_crawler(self, progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> CrawlProgressTracker:
        """Start the crawler and return a tracker compatible with the UI's expectations.
        Returns a tracker with get_status method that provides progress information."""
        # Create tracker
        tracker = CrawlProgressTracker(progress_callback)
        
        # Start the crawl in a separate thread
        def run_crawler():
            try:
                # If Crawl4ai is available, use it; otherwise use HTTP requests
                if crawl4ai_available and not get_env_var("FORCE_REQUESTS_CRAWLER"):
                    print(f"Using Crawl4ai for {self.source.name} documentation")
                    asyncio.run(self.crawl_with_crawl4ai(tracker))
                else:
                    print(f"Using requests-based crawler for {self.source.name} documentation")
                    asyncio.run(self.crawl_parallel_with_requests(tracker))
                # Mark as complete when done
                tracker.complete()
            except Exception as e:
                # Log error and mark as complete in case of failure
                tracker.log(f"Error in crawler: {str(e)}")
                tracker.complete()
        
        thread = threading.Thread(
            target=run_crawler
        )
        thread.daemon = True
        thread.start()
        
        return tracker

    def _process_url_with_semaphore(self, url, semaphore, tracker=None):
        """
        Process a single URL with a semaphore to limit concurrency.
        
        Args:
            url: The URL to process
            semaphore: Semaphore to limit concurrency
            tracker: Optional tracker to track progress
        """
        try:
            # Acquire semaphore
            semaphore.acquire()
            
            # Log
            if tracker:
                tracker.log(f"Crawling: {url}")
            else:
                print(f"Crawling: {url}")
                
            try:
                # Fetch content from URL
                print(f"Fetching content from: {url}")
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                html_content = response.text
                
                # Convert HTML to markdown or plain text
                markdown = convert_html_to_text(html_content)
                
                if markdown:
                    # Process content
                    chunks = self.chunk_text(markdown, url)
                    processed_chunks = []
                    
                    for i, chunk in enumerate(chunks):
                        # Process each chunk
                        try:
                            processed_chunk = self.process_chunk(chunk, url, i, len(chunks))
                            if processed_chunk:
                                processed_chunks.append(processed_chunk)
                                # Insert into database
                                self.insert_chunk(processed_chunk)
                        except Exception as chunk_error:
                            if tracker:
                                tracker.log(f"Error processing chunk {i} from {url}: {str(chunk_error)}")
                            else:
                                print(f"Error processing chunk {i} from {url}: {str(chunk_error)}")
                    
                    if processed_chunks:
                        if tracker:
                            tracker.increment_success()
                            tracker.log(f"Successfully processed {len(processed_chunks)} chunks from {url}")
                        else:
                            print(f"Successfully processed {len(processed_chunks)} chunks from {url}")
                    else:
                        if tracker:
                            tracker.increment_error()
                            tracker.log(f"No chunks processed from {url}")
                        else:
                            print(f"No chunks processed from {url}")
                else:
                    if tracker:
                        tracker.increment_error()
                        tracker.log(f"Failed to extract content from {url}")
                    else:
                        print(f"Failed to extract content from {url}")
            except Exception as e:
                if tracker:
                    tracker.increment_error()
                    tracker.log(f"Error processing {url}: {str(e)}")
                else:
                    print(f"Error processing {url}: {str(e)}")
            finally:
                if tracker:
                    tracker.increment_processed()
                
                # Release semaphore
                semaphore.release()
        except Exception as outer_e:
            # Handle exceptions in the outer try block
            if tracker:
                tracker.increment_error()
                tracker.log(f"Unhandled error for {url}: {str(outer_e)}")
            else:
                print(f"Unhandled error for {url}: {str(outer_e)}")
            
            # Make sure semaphore is released
            try:
                semaphore.release()
            except:
                pass

    async def crawl_with_crawl4ai(self, tracker: Optional[CrawlProgressTracker] = None):
        """
        Crawl documentation using Crawl4ai.
        
        Args:
            tracker: Optional tracker to monitor progress.
        """
        try:
            # Start tracking if tracker is provided
            if tracker:
                tracker.log(f"Starting Crawl4ai crawling process for {self.source.name}...")
            else:
                print(f"Starting Crawl4ai crawling process for {self.source.name}...")
            
            # Clear existing records first
            if tracker:
                tracker.log(f"Clearing existing {self.source.name} records...")
            else:
                print(f"Clearing existing {self.source.name} records...")
            
            self.clear_existing_records()
            
            if tracker:
                tracker.log("Existing records cleared")
            else:
                print("Existing records cleared")
            
            # Get URLs from the source
            if tracker:
                tracker.log(f"Fetching URLs from {self.source.name} sitemap...")
            else:
                print(f"Fetching URLs from {self.source.name} sitemap...")
            
            urls = self.source.url_fetcher()
            
            if not urls:
                if tracker:
                    tracker.log("No URLs found to crawl")
                    # Mark as complete if tracker is provided
                    tracker.complete()
                else:
                    print("No URLs found to crawl")
                return
            
            if tracker:
                tracker.total_urls = len(urls)
                tracker.log(f"Found {len(urls)} URLs to crawl")
            else:
                print(f"Found {len(urls)} URLs to crawl")
            
            # Set up Crawl4ai crawler
            browser_config = BrowserConfig(
                headless=True,
                ignore_https_errors=True,
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            )
            
            crawler_config = CrawlerRunConfig(
                cache_mode=CacheMode.MEMORY,
                ignore_robots=True,
                max_depth=1,  # Just crawl the specified URLs, don't follow links
                max_reqs_per_min=60,
                timeout_secs=60
            )
            
            crawler = AsyncWebCrawler(browser_config)
            
            # Process URLs in batches to avoid memory issues
            batch_size = 10
            url_batches = [urls[i:i + batch_size] for i in range(0, len(urls), batch_size)]
            
            for batch_idx, url_batch in enumerate(url_batches):
                if tracker:
                    tracker.log(f"Processing batch {batch_idx+1}/{len(url_batches)} ({len(url_batch)} URLs)")
                else:
                    print(f"Processing batch {batch_idx+1}/{len(url_batches)} ({len(url_batch)} URLs)")
                
                # Crawl the batch
                results = await crawler.crawl(url_batch, crawler_config)
                
                # Process the results
                for url, result in results.items():
                    try:
                        if result and result.content:
                            # Convert HTML to markdown
                            markdown = convert_html_to_text(result.content)
                            
                            if markdown:
                                if tracker:
                                    tracker.success_count += 1
                                    tracker.log(f"Successfully crawled: {url}")
                                else:
                                    print(f"Successfully crawled: {url}")
                                
                                # Process and store the document
                                await self.process_and_store_document(url, markdown, tracker)
                            else:
                                if tracker:
                                    tracker.error_count += 1
                                    tracker.log(f"Failed to convert content for {url}")
                                else:
                                    print(f"Failed to convert content for {url}")
                        else:
                            if tracker:
                                tracker.error_count += 1
                                tracker.log(f"Failed to crawl {url} - No content retrieved")
                            else:
                                print(f"Failed to crawl {url} - No content retrieved")
                    except Exception as e:
                        if tracker:
                            tracker.error_count += 1
                            tracker.log(f"Error processing {url}: {str(e)}")
                        else:
                            print(f"Error processing {url}: {str(e)}")
                    finally:
                        if tracker:
                            tracker.processed_count += 1
                            # Ensure UI gets updated
                            if tracker.progress_callback:
                                tracker.progress_callback(tracker.get_status())
                
                # Close the crawler for this batch to free resources
                await crawler.close()
                
                # Create a new crawler for the next batch
                crawler = AsyncWebCrawler(browser_config)
                
                # Small delay between batches
                await asyncio.sleep(1)
            
            # Mark as complete if tracker is provided
            if tracker:
                tracker.complete()
            else:
                print("Crawling process completed")
                
        except Exception as e:
            if tracker:
                tracker.log(f"Error in Crawl4ai crawling process: {str(e)}")
                tracker.complete()
            else:
                print(f"Error in Crawl4ai crawling process: {str(e)}")
            
            # Try to close the crawler if it exists
            try:
                if 'crawler' in locals():
                    await crawler.close()
            except:
                pass


class DocumentationSourceRegistry:
    """Registry for managing documentation sources."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DocumentationSourceRegistry, cls).__new__(cls)
            cls._instance.sources = {}
        return cls._instance
    
    def register(self, source: DocumentationSource):
        """Register a documentation source."""
        self.sources[source.id] = source
        return source
    
    def get_source(self, source_id: str) -> Optional[DocumentationSource]:
        """Get a documentation source by ID."""
        return self.sources.get(source_id)
    
    def get_all_sources(self) -> List[DocumentationSource]:
        """Get all registered documentation sources."""
        return list(self.sources.values())
    
    def get_crawler(self, source_id: str) -> Optional[DocumentationCrawler]:
        """Get a crawler for a specific source."""
        source = self.get_source(source_id)
        if source:
            return DocumentationCrawler(source)
        return None


# Initialize the registry
registry = DocumentationSourceRegistry()

# Register the Pydantic AI source
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

# Register the Pydantic AI source
pydantic_ai_source = registry.register(DocumentationSource(
    id="pydantic_ai_docs",
    name="Pydantic AI Docs",
    description="""
    Pydantic AI Documentation.
    This crawler fetches URLs from the Pydantic AI sitemap, crawls each page,
    extracts content, splits it into chunks, generates embeddings, and stores in the database.
    """,
    url_fetcher=get_pydantic_ai_docs_urls,
    sitemap_url="https://ai.pydantic.dev/sitemap.xml"
))

# Register LangChain source as an example of another source
def get_langchain_docs_urls() -> List[str]:
    """Get URLs from LangChain docs sitemap."""
    sitemap_url = "https://python.langchain.com/sitemap.xml"
    try:
        response = requests.get(sitemap_url)
        response.raise_for_status()
        
        # Parse the XML
        root = ElementTree.fromstring(response.content)
        
        # Extract all URLs from the sitemap
        # Note: LangChain's sitemap might have a different structure than Pydantic's
        urls = []
        
        # Try to find direct loc elements (standard sitemap)
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        direct_urls = [loc.text for loc in root.findall('.//ns:loc', namespace)]
        
        if direct_urls:
            urls.extend(direct_urls)
        else:
            # Check if this is a sitemap index (contains references to other sitemaps)
            sitemap_refs = [loc.text for loc in root.findall('.//ns:sitemap/ns:loc', namespace)]
            
            # If we found sitemap references, fetch each one
            for sitemap_url in sitemap_refs:
                try:
                    sitemap_response = requests.get(sitemap_url)
                    sitemap_response.raise_for_status()
                    sitemap_root = ElementTree.fromstring(sitemap_response.content)
                    sitemap_urls = [loc.text for loc in sitemap_root.findall('.//ns:loc', namespace)]
                    urls.extend(sitemap_urls)
                except Exception as e:
                    print(f"Error fetching sub-sitemap {sitemap_url}: {e}")
        
        return urls
    except Exception as e:
        print(f"Error fetching LangChain sitemap: {e}")
        return []

# Register the LangChain source
langchain_source = registry.register(DocumentationSource(
    id="langchain_docs",
    name="LangChain Docs",
    description="""
    LangChain Documentation.
    This crawler fetches URLs from the LangChain sitemap, crawls each page,
    extracts content, splits it into chunks, generates embeddings, and stores in the database.
    """,
    url_fetcher=get_langchain_docs_urls,
    sitemap_url="https://python.langchain.com/sitemap.xml"
))

# Register LangGraph source
def get_langgraph_docs_urls() -> List[str]:
    """Get URLs from LangGraph docs sitemap."""
    sitemap_url = "https://langchain-ai.github.io/langgraph/sitemap.xml"
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
        print(f"Error fetching LangGraph sitemap: {e}")
        return []

# Register the LangGraph source
langgraph_source = registry.register(DocumentationSource(
    id="langgraph_docs",
    name="LangGraph Docs",
    description="""
    LangGraph Documentation.
    This crawler fetches URLs from the LangGraph sitemap, crawls each page,
    extracts content, splits it into chunks, generates embeddings, and stores in the database.
    
    LangGraph is a library for building stateful, multi-actor applications with LLMs.
    """,
    url_fetcher=get_langgraph_docs_urls,
    sitemap_url="https://langchain-ai.github.io/langgraph/sitemap.xml"
))

# Backward compatibility functions for pydantic_ai_docs.py
def clear_existing_records():
    """Clear all existing records with source='pydantic_ai_docs' from the site_pages table."""
    source = registry.get_source("pydantic_ai_docs")
    if source:
        crawler = DocumentationCrawler(source)
        return crawler.clear_existing_records()
    return None

async def main_with_requests(tracker: Optional[CrawlProgressTracker] = None):
    """Main function using direct HTTP requests instead of browser automation."""
    source = registry.get_source("pydantic_ai_docs")
    if source:
        crawler = DocumentationCrawler(source)
        # If Crawl4ai is available, use it; otherwise use HTTP requests
        if crawl4ai_available and not get_env_var("FORCE_REQUESTS_CRAWLER"):
            await crawler.crawl_with_crawl4ai(tracker)
        else:
            await crawler.crawl_parallel_with_requests(tracker)
    else:
        print("Pydantic AI source not registered")

def start_crawl_with_requests(progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> CrawlProgressTracker:
    """Start the crawling process using direct HTTP requests in a separate thread and return the tracker."""
    source = registry.get_source("pydantic_ai_docs")
    if source:
        crawler = DocumentationCrawler(source)
        return crawler.start_crawler(progress_callback)
    else:
        print("Pydantic AI source not registered")
        # Return a dummy tracker
        tracker = CrawlProgressTracker(progress_callback)
        tracker.log("Error: Pydantic AI source not registered")
        tracker.complete()
        return tracker

def get_sitemap_urls(sitemap_url: str) -> List[str]:
    """Extract URLs from a sitemap XML.
    
    Uses fallback methods if dependencies are not available:
    1. Try lxml's etree first if available
    2. Use ElementTree from the standard library as a fallback
    """
    try:
        response = requests.get(sitemap_url, timeout=30)
        response.raise_for_status()
        
        urls = []
        
        # Approach 1: Try with lxml's etree if available (more robust)
        if lxml_available:
            try:
                root = etree.fromstring(response.content)
                
                # Extract URLs using XML namespaces
                namespaces = {'sm': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
                urls = [url.text for url in root.xpath('//sm:url/sm:loc', namespaces=namespaces)]
                
                # If we didn't find any URLs, check if this is a sitemap index
                if not urls:
                    urls = [url.text for url in root.xpath('//sm:sitemap/sm:loc', namespaces=namespaces)]
                    
                    # If we found sitemap references, fetch those sitemaps
                    if urls:
                        all_urls = []
                        for sitemap_ref in urls:
                            sub_urls = get_sitemap_urls(sitemap_ref)
                            all_urls.extend(sub_urls)
                        urls = all_urls
            except Exception as lxml_error:
                print(f"Error using lxml parser: {lxml_error}")
                urls = []  # Reset for the next approach
        
        # Approach 2: Fallback to ElementTree from standard library
        if not urls:
            try:
                # Try to parse using ElementTree
                root = ElementTree.fromstring(response.content)
                
                # Extract URLs - first try standard sitemap format
                namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
                urls = [loc.text for loc in root.findall('.//ns:loc', namespace)]
                
                # If we didn't find any URLs, check if this is a sitemap index
                if not urls:
                    sitemap_refs = [loc.text for loc in root.findall('.//ns:sitemap/ns:loc', namespace)]
                    
                    # If we found sitemap references, fetch those sitemaps
                    all_urls = []
                    for sitemap_ref in sitemap_refs:
                        sub_urls = get_sitemap_urls(sitemap_ref)
                        all_urls.extend(sub_urls)
                    urls = all_urls
            except Exception as et_error:
                print(f"Error using ElementTree parser: {et_error}")
                
                # Last resort: try regex-based extraction
                try:
                    content = response.text
                    # Simple regex to extract URLs from loc tags
                    url_matches = re.findall(r'<loc>(.*?)</loc>', content)
                    urls = url_matches
                except Exception as regex_error:
                    print(f"Error using regex-based extraction: {regex_error}")
        
        return urls
    except Exception as e:
        print(f"Error fetching sitemap from {sitemap_url}: {e}")
        return [] 