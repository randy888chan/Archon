from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from utils.utils import get_env_var, get_clients
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
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import openai  # Import base openai module for exceptions
from bs4 import BeautifulSoup
import redis.asyncio as redis  # Added import

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


load_dotenv()

embedding_client, supabase = get_clients()

embedding_model = get_env_var('EMBEDDING_MODEL') or 'text-embedding-3-small'

# Define Redis keys matching backend/socket.py
REDIS_KEY_STATUS = "crawl:status"
REDIS_KEY_LOGS = "crawl:logs"
REDIS_KEY_ERRORS = "crawl:errors"
REDIS_KEY_RUNNING_FLAG = "crawl:running"

llm_client = None
base_url = get_env_var('BASE_URL') or 'https://api.openai.com/v1'
api_key = get_env_var('LLM_API_KEY') or 'no-api-key-provided'
provider = get_env_var('LLM_PROVIDER') or 'OpenAI'


if provider == "Ollama":
    if api_key == "NOT_REQUIRED":
        api_key = "ollama"
    llm_client = AsyncOpenAI(base_url=base_url, api_key=api_key)
else:
    llm_client = AsyncOpenAI(base_url=base_url, api_key=api_key)

html_converter = html2text.HTML2Text()
html_converter.ignore_links = False
html_converter.ignore_images = False
html_converter.ignore_tables = False
html_converter.body_width = 0


@dataclass
class ProcessedChunk:
    url: str
    source: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]


class CrawlProgressTracker:
    """Class to track progress of the crawling process. Stores state directly."""

    def __init__(self, redis_client=None):
        """Initialize the progress tracker."""
        self.urls_found = 0
        self.urls_processed = 0
        self.urls_succeeded = 0
        self.urls_failed = 0
        self.chunks_stored = 0
        self.logs = []
        self.is_running = False
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.current_url: Optional[str] = None
        self.errors: List[str] = []
        self.urls_skipped: int = 0
        # Store Redis client for status updates
        self.redis_client = redis_client

    async def _update_redis_status(self, status_updates=None):
        """Update Redis with the current status."""
        if not self.redis_client:
            return

        try:
            # Default status updates based on tracker state
            if status_updates is None:
                status_updates = {
                    "is_running": "1" if self.is_running else "0",
                    "processed_count": str(self.urls_processed),
                    "total_urls": str(self.urls_found),
                    "urls_succeeded": str(self.urls_succeeded),
                    "urls_failed": str(self.urls_failed),
                    "urls_skipped": str(self.urls_skipped),
                    "chunks_stored": str(self.chunks_stored),
                    "current_url": self.current_url or ""
                }

                # Add time fields if available
                if self.start_time:
                    status_updates["start_time"] = self.start_time.isoformat()
                if self.end_time:
                    status_updates["end_time"] = self.end_time.isoformat()

                # Calculate duration if possible
                if self.start_time:
                    if self.end_time:
                        duration = (self.end_time -
                                    self.start_time).total_seconds()
                    elif self.is_running:
                        duration = (datetime.now(timezone.utc) -
                                    self.start_time).total_seconds()
                    else:
                        duration = 0
                    status_updates["duration_seconds"] = str(duration)

            # Update Redis with the status
            await self.redis_client.hset(REDIS_KEY_STATUS, mapping=status_updates)
        except Exception as e:
            print(f"Failed to update Redis status: {e}")

    async def log(self, message: str):
        """Add a log message and update Redis."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.logs.append(log_entry)
        print(log_entry)

        # Update Redis
        if self.redis_client:
            try:
                async with self.redis_client.pipeline() as pipe:
                    await pipe.rpush(REDIS_KEY_LOGS, log_entry)
                    await pipe.hset(REDIS_KEY_STATUS, "message", message)
                    await pipe.execute()
                await self._update_redis_status()
            except Exception as e:
                print(f"Redis log update failed: {e}")

    async def log_error(self, error_message: str, url: Optional[str] = None):
        """Log an error message and update Redis."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] ERROR: {error_message}" + \
            (f" (URL: {url})" if url else "")
        self.logs.append(log_entry)
        self.errors.append(log_entry)
        print(log_entry)

        # Update Redis
        if self.redis_client:
            try:
                async with self.redis_client.pipeline() as pipe:
                    await pipe.rpush(REDIS_KEY_LOGS, log_entry)
                    await pipe.rpush(REDIS_KEY_ERRORS, log_entry)
                    await pipe.hset(REDIS_KEY_STATUS, "message", f"Error: {error_message}")
                    await pipe.execute()
                await self._update_redis_status()
            except Exception as e:
                print(f"Redis error log update failed: {e}")

    async def start(self):
        """Mark the crawling process as started and update Redis."""
        self.is_running = True
        self.start_time = datetime.now(timezone.utc)
        await self.log("Crawling process started")

        # Additional Redis updates specific to starting
        if self.redis_client:
            try:
                start_time_iso = self.start_time.isoformat()
                await self.redis_client.hset(REDIS_KEY_STATUS, mapping={
                    "is_running": "1",
                    "start_time": start_time_iso,
                    "message": "Crawl process started"
                })
                # Set running flag with TTL
                await self.redis_client.set(REDIS_KEY_RUNNING_FLAG, "1", ex=7200)
            except Exception as e:
                print(f"Redis start update failed: {e}")

    async def complete(self):
        """Mark the crawling process as completed and update Redis."""
        self.is_running = False
        self.end_time = datetime.now(timezone.utc)
        duration = self.end_time - self.start_time if self.start_time else None
        duration_str = str(duration).split('.')[0] if duration else "unknown"
        await self.log(f"Crawling process completed in {duration_str}")

        # Final Redis updates
        if self.redis_client:
            try:
                end_time_iso = self.end_time.isoformat()
                final_message = "Crawl completed."
                if self.urls_failed > 0:
                    final_message = f"Crawl completed with {self.urls_failed} errors."

                await self.redis_client.hset(REDIS_KEY_STATUS, mapping={
                    "is_running": "0",
                    "end_time": end_time_iso,
                    "message": final_message,
                    "processed_count": str(self.urls_processed),
                    "urls_succeeded": str(self.urls_succeeded),
                    "urls_failed": str(self.urls_failed),
                    "urls_skipped": str(self.urls_skipped),
                    "chunks_stored": str(self.chunks_stored)
                })
                # Remove running flag
                await self.redis_client.delete(REDIS_KEY_RUNNING_FLAG)
            except Exception as e:
                print(f"Redis completion update failed: {e}")

    async def update_progress(self, **kwargs):
        """Update progress fields and Redis in one call."""
        # Update local tracker fields
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # Update Redis
        if self.redis_client:
            try:
                # Convert all values to strings for Redis hset
                redis_updates = {k: str(v) for k, v in kwargs.items()}
                await self._update_redis_status(redis_updates)
            except Exception as e:
                print(f"Redis progress update failed: {e}")

    def get_status_dict(self) -> Dict[str, Any]:
        """Get the current status as a serializable dictionary."""
        duration = None
        if self.start_time:
            if self.end_time:
                duration = (self.end_time - self.start_time).total_seconds()
            elif self.is_running:
                duration = (datetime.now() - self.start_time).total_seconds()

        return {
            "is_running": self.is_running,
            "processed_count": self.urls_processed,
            "total_urls": self.urls_found,
            "urls_succeeded": self.urls_succeeded,
            "urls_failed": self.urls_failed,
            "urls_skipped": self.urls_skipped,
            "chunks_stored": self.chunks_stored,
            "progress_percentage": (self.urls_processed / self.urls_found * 100) if self.urls_found > 0 else 0,
            "logs": self.logs[-50:],
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": duration,
            "current_url": self.current_url,
            "errors": self.errors[-50:],
        }


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


@retry(
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(
        (openai.RateLimitError, openai.APIError, openai.APITimeoutError, openai.InternalServerError)),
    reraise=True
)
async def get_title_and_summary(chunk: str, url: str) -> Dict[str, str]:
    """Extract title and summary using the configured LLM, with retries."""
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
                {"role": "user",
                    "content": f"URL: {url}\n\nContent:\n{chunk[:1000]}..."}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error getting title and summary after retries for {url}: {e}")
        return {"title": f"Error processing title ({type(e).__name__})", "summary": f"Error processing summary ({type(e).__name__})"}


@retry(
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(
        (openai.RateLimitError, openai.APIError, openai.APITimeoutError, openai.InternalServerError)),
    reraise=True
)
async def get_embedding(text: str, url: Optional[str] = None) -> List[float]:
    """Get embedding vector from embedding provider, with retries."""
    try:
        if not text or text.isspace():
            print(
                f"Warning: Attempting to get embedding for empty text (URL: {url}). Returning zero vector.")
            return [0.0] * 1536

        response = await embedding_client.embeddings.create(
            model=embedding_model,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding after retries for URL {url}: {e}")
        raise


async def process_chunk(chunk: str, chunk_number: int, url: str, source: str) -> Optional[ProcessedChunk]:
    """Process a single chunk of text. Returns None if processing fails."""
    try:
        extracted = await get_title_and_summary(chunk, url)
        if "Error processing" in extracted["title"]:
            print(
                f"Skipping chunk {chunk_number} for {url} due to title/summary error.")
            return None

        embedding = await get_embedding(chunk, url=url)

        metadata = {
            "source": source,
            "chunk_size": len(chunk),
            "crawled_at": datetime.now(timezone.utc).isoformat(),
            "url_path": urlparse(url).path
        }

        return ProcessedChunk(
            url=url,
            source=source,
            chunk_number=chunk_number,
            title=extracted['title'],
            summary=extracted['summary'],
            content=chunk,
            metadata=metadata,
            embedding=embedding
        )
    except Exception as e:
        print(f"Failed to process chunk {chunk_number} for {url}: {e}")
        return None


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
        print(
            f"Inserted chunk {chunk.chunk_number} for {chunk.url} (Source: {chunk.source})")
        return result
    except Exception as e:
        print(f"Error inserting chunk: {e}")
        return None


async def process_and_store_document(url: str, markdown: str, source: str, tracker: CrawlProgressTracker):
    """Process a document and store its chunks in parallel."""
    await tracker.update_progress(current_url=url)
    chunks = chunk_text(markdown)

    tasks = [
        process_chunk(chunk, i + 1, url, source)
        for i, chunk in enumerate(chunks)
    ]
    processed_chunks = await asyncio.gather(*tasks)

    local_chunks_stored = 0
    processing_failed = False

    for i, chunk_result in enumerate(processed_chunks):
        if chunk_result:
            try:
                await insert_chunk(chunk_result)
                local_chunks_stored += 1
                await tracker.update_progress(chunks_stored=tracker.chunks_stored + 1)
            except Exception as insert_e:
                await tracker.log_error(
                    f"Failed to insert chunk {i+1}: {insert_e}", url=url)
                processing_failed = True
        else:
            processing_failed = True

    if processing_failed is True:
        await tracker.update_progress(urls_failed=tracker.urls_failed + 1)
        await tracker.log_error(
            message=f"Overall processing/storing failed for URL", url=url)
    else:
        await tracker.update_progress(urls_succeeded=tracker.urls_succeeded + 1)
        await tracker.log(
            message=f"Successfully processed and stored {local_chunks_stored} chunks for URL: {url}")


def fetch_url_content(url: str) -> str:
    """Fetch content from a URL using requests and convert to markdown."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        # Parse the HTML content using BeautifulSoup
        # Using 'html.parser' initially, could switch to 'lxml' if installed & needed
        soup = BeautifulSoup(response.text, 'html.parser')

        # Optional: Select only the main content area if identifiable
        # to avoid converting headers, footers, navbars, etc.
        # Example (needs inspection of Pydantic docs HTML structure):
        # main_content = soup.find('main') # Or appropriate tag/class
        # html_to_convert = str(main_content) if main_content else str(soup)

        # For now, convert the whole parsed soup back to string for html2text
        html_to_convert = str(soup)

        # Convert cleaned HTML string to Markdown
        markdown = html_converter.handle(html_to_convert)

        # Clean up the markdown
        # Remove excessive newlines
        markdown = re.sub(r'\n{3,}', '\n\n', markdown)

        return markdown
    except Exception as e:
        # Add more detailed logging
        error_type = type(e).__name__
        print(
            f"Detailed Error Processing {url}: Type={error_type}, Error={str(e)}")
        # Re-raise the exception (or a new one)
        raise Exception(f"Error processing {url}: {error_type} - {str(e)}")


async def crawl_parallel_with_requests(urls_with_source: Dict[str, str], tracker: CrawlProgressTracker, max_concurrent: int = 5):
    """Fetches content using requests and processes in parallel using asyncio."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_url(url: str, source: str):
        async with semaphore:
            try:
                loop = asyncio.get_running_loop()
                await tracker.log(f"Fetching content from: {url}")

                # Update current URL in Redis
                await tracker.update_progress(current_url=url)

                markdown = await loop.run_in_executor(None, fetch_url_content, url)

                if markdown:
                    await tracker.log(f"Successfully fetched: {url}")

                    await process_and_store_document(url, markdown, source, tracker)
                else:
                    await tracker.update_progress(urls_failed=tracker.urls_failed + 1)
                    await tracker.log(f"Failed: {url} - No content retrieved")
            except Exception as e:
                await tracker.update_progress(urls_failed=tracker.urls_failed + 1)
                await tracker.log_error(f"Error processing {url}: {str(e)}", url=url)
            finally:
                await tracker.update_progress(urls_processed=tracker.urls_processed + 1)

        await asyncio.sleep(1)

    await tracker.log(
        f"Processing {len(urls_with_source)} URLs from {len(set(urls_with_source.values()))} sources with concurrency {max_concurrent}")

    tasks = [process_url(url, source)
             for url, source in urls_with_source.items()]
    await asyncio.gather(*tasks)


def _fetch_sitemap_urls(sitemap_url: str, source_name: str) -> List[str]:
    """Helper function to fetch URLs from a sitemap."""
    print(f"Fetching URLs from {source_name} sitemap ({sitemap_url})...")
    try:
        response = requests.get(sitemap_url, timeout=30)
        response.raise_for_status()
        root = ElementTree.fromstring(response.content)
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        urls = [loc.text for loc in root.findall('.//ns:loc', namespace)]
        print(f"Found {len(urls)} URLs for {source_name}")
        return urls
    except requests.exceptions.RequestException as e:
        print(
            f"HTTP Error fetching {source_name} sitemap ({sitemap_url}): {e}")
        return []
    except ElementTree.ParseError as e:
        print(
            f"XML Parsing Error for {source_name} sitemap ({sitemap_url}): {e}")
        return []
    except Exception as e:
        print(
            f"Unexpected Error fetching {source_name} sitemap ({sitemap_url}): {e}")
        return []


def get_pydantic_ai_docs_urls() -> List[str]:
    """Get URLs from Pydantic AI docs sitemap."""
    return _fetch_sitemap_urls("https://ai.pydantic.dev/sitemap.xml", "pydantic_ai_docs")


def get_langgraph_docs_urls() -> List[str]:
    """Get URLs from LangGraph docs sitemap."""
    return _fetch_sitemap_urls("https://langchain-ai.github.io/langgraph/sitemap.xml", "langgraph_docs")


def get_langgraphjs_docs_urls() -> List[str]:
    """Get URLs from LangGraph JS docs sitemap."""
    return _fetch_sitemap_urls("https://langchain-ai.github.io/langgraphjs/sitemap.xml", "langgraphjs_docs")


def get_langsmith_docs_urls() -> List[str]:
    """Get URLs from LangSmith docs sitemap."""
    return _fetch_sitemap_urls("https://docs.smith.langchain.com/sitemap.xml", "langsmith_docs")


def get_langchain_python_docs_urls() -> List[str]:
    """Get URLs from LangChain Python docs sitemap."""
    return _fetch_sitemap_urls("https://python.langchain.com/sitemap.xml", "langchain_python_docs")


def get_langchain_js_docs_urls() -> List[str]:
    """Get URLs from LangChain JS docs sitemap."""
    return _fetch_sitemap_urls("https://js.langchain.com/sitemap.xml", "langchain_js_docs")


def clear_existing_records(source: str):
    """Clear all existing records for a specific source from the site_pages table."""
    try:
        print(f"Clearing existing records for source: {source}...")

        delete_result = supabase.table("site_pages").delete().eq(
            "metadata->>source", source).execute()

        print(f"Cleared existing records for source: {source}")
        return delete_result
    except Exception as e:
        print(f"Error clearing existing records for {source}: {e}")
        return None


async def main_with_requests(tracker: CrawlProgressTracker, process_only_new: bool = False, redis_client: redis.Redis = None):
    """Main function to crawl documentation using requests and process in parallel."""
    start_time_main = time.time()
    # Use pipeline for efficiency where appropriate
    redis_pipe = redis_client.pipeline()
    try:
        # Start tracker is now handled in run_crawl_task

        sources = {
            "pydantic_ai_docs": get_pydantic_ai_docs_urls,
            "langgraph_docs": get_langgraph_docs_urls,
            "langgraphjs_docs": get_langgraphjs_docs_urls,
            "langsmith_docs": get_langsmith_docs_urls,
            "langchain_python_docs": get_langchain_python_docs_urls,
            "langchain_js_docs": get_langchain_js_docs_urls,
        }

        # # --- Conditionally clear existing records ---
        # if not process_only_new:
        #     clear_start_time = time.time()
        #     tracker.log(
        #         "Clearing existing records for all sources as process_only_new is False...")

        #     clear_tasks = [asyncio.to_thread(
        #         clear_existing_records, source_name) for source_name in sources.keys()]
        #     results = await asyncio.gather(*clear_tasks, return_exceptions=True)
        #     for source_name, result in zip(sources.keys(), results):
        #         if isinstance(result, Exception):
        #             log_msg = f"Failed to clear records for {source_name}: {result}"
        #             # Log as error if clearing fails
        #             tracker.log_error(log_msg)

        #     clear_duration = time.time() - clear_start_time
        #     log_msg_clear = f"Finished clearing records for {len(sources)} sources in {clear_duration:.2f} seconds."
        #     tracker.log(log_msg_clear)
        # else:
        #     tracker.log(
        #         "Skipping record clearing as process_only_new is True.")

        fetch_start_time = time.time()
        all_urls_with_source = {}
        total_urls_initially_found = 0  # Renamed to avoid confusion

        fetch_tasks = [asyncio.to_thread(fetch_func)
                       for fetch_func in sources.values()]
        fetch_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

        for source_name, result in zip(sources.keys(), fetch_results):
            if isinstance(result, Exception):
                log_msg = f"Failed to fetch URLs for {source_name}: {result}"
                await tracker.log_error(log_msg)  # Now using async method
                continue
            elif isinstance(result, list):
                urls = result
                for url in urls:
                    all_urls_with_source[url] = source_name
                # Count all URLs found from sitemaps
                total_urls_initially_found += len(urls)

        fetch_duration = time.time() - fetch_start_time

        if not all_urls_with_source:
            log_msg = f"No URLs found to crawl from any source after {fetch_duration:.2f} seconds."
            await tracker.log(log_msg)  # Now using async method
            # Complete tracker is now handled in run_crawl_task
            return

        log_msg_fetch = f"Found a total of {total_urls_initially_found} URLs from sitemaps for {len(sources)} sources in {fetch_duration:.2f} seconds."
        await tracker.log(log_msg_fetch)  # Now using async method
        # Update tracker with Redis update
        await tracker.update_progress(urls_found=total_urls_initially_found)

        urls_to_process = all_urls_with_source.copy()  # Start with all URLs

        # --- Filter out existing URLs if requested ---
        if process_only_new:
            # Now using async method
            await tracker.log("Checking for previously processed URLs in the database...")
            try:
                # Query Supabase to get existing URLs using RPC function
                # This is more efficient for large datasets
                def fetch_existing_urls_sync():
                    return supabase.rpc('get_distinct_processed_urls').execute()

                response = await asyncio.to_thread(fetch_existing_urls_sync)

                if not hasattr(response, 'data'):
                    raise Exception(
                        f"RPC response did not contain 'data': {response}")

                existing_urls_data = response.data
                existing_urls = {item['url'] for item in existing_urls_data}
                # Now using async method
                await tracker.log(f"Found {len(existing_urls)} unique URLs already in the database.")

                # Filter the dictionary
                urls_to_process_filtered = {
                    url: src for url, src in all_urls_with_source.items() if url not in existing_urls}
                skipped_count = total_urls_initially_found - \
                    len(urls_to_process_filtered)

                # Update tracker with Redis update
                await tracker.update_progress(
                    urls_skipped=skipped_count,
                    urls_found=len(urls_to_process_filtered)
                )

                # Now using async method
                await tracker.log(f"Skipping {skipped_count} URLs. Will process {tracker.urls_found} new URLs.")
                urls_to_process = urls_to_process_filtered  # Replace with filtered list

            except Exception as e:
                # Now using async method
                await tracker.log_error(f"Error fetching or filtering existing URLs: {e}. Proceeding with all initially found URLs.")
                # Reset skipped count and total count if filtering failed
                await tracker.update_progress(
                    urls_skipped=0,
                    urls_found=total_urls_initially_found
                )
                urls_to_process = all_urls_with_source  # Revert to all URLs

        # --- Proceed with Crawling ---
        if not urls_to_process:
            # Use tracker's async log method
            no_crawl_msg = "No new URLs found to crawl."
            await tracker.log(no_crawl_msg)
            # Complete tracker is now handled in run_crawl_task
            return

        crawl_start_time = time.time()
        # Pass tracker as second parameter, not redis_client
        await crawl_parallel_with_requests(urls_to_process, tracker, max_concurrent=5)
        crawl_duration = time.time() - crawl_start_time

        # Use tracker's async log method
        log_msg_crawl_done = f"Crawling and processing finished in {crawl_duration:.2f} seconds."
        await tracker.log(log_msg_crawl_done)

        # Final counts/status are set in the calling run_crawl_task function
    except Exception as e:
        error_msg = f"Error in main crawling process: {str(e)}"
        await tracker.log_error(error_msg)  # Now using async method
    finally:
        main_duration = time.time() - start_time_main
        print(
            f"Total execution time for main_with_requests: {main_duration:.2f} seconds.")


async def run_crawl_task(tracker: CrawlProgressTracker, process_only_new: bool = False, redis_client: redis.Redis = None):
    """Runs the main crawl logic within an async context, updating Redis.

    Args:
        tracker: The CrawlProgressTracker instance (mostly for logging within task).
        process_only_new: Flag to indicate if only new URLs should be processed.
        redis_client: The Redis client instance for updating shared state.
    """
    if not redis_client:
        # This case should ideally not happen if called from the API correctly,
        # but handle it defensively.
        print("ERROR: Redis client was not provided to run_crawl_task.")
        # Optionally update tracker status if possible, though it won't reflect in API
        if tracker:
            # Can't use async methods when Redis is missing
            print("Internal Error: Redis client missing.")
        return  # Cannot proceed without Redis

    # Set the Redis client on the tracker
    if tracker:
        tracker.redis_client = redis_client
    else:
        # Create a new tracker with the Redis client
        tracker = CrawlProgressTracker(redis_client=redis_client)

    if not supabase or not embedding_client or not llm_client:
        # Log error to Redis if possible
        error_msg = "Initialization failed: Clients (Supabase/Embedding/LLM) not available."
        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] ERROR: {error_msg}"
        try:
            await redis_client.rpush(REDIS_KEY_ERRORS, log_entry)
            await redis_client.hset(REDIS_KEY_STATUS, "message", error_msg)
            # Ensure running flag is cleared
            await redis_client.delete(REDIS_KEY_RUNNING_FLAG)
            await redis_client.hset(REDIS_KEY_STATUS, "is_running", "0")
        except Exception as redis_e:
            print(f"Redis Error logging initialization failure: {redis_e}")
        # Also log locally via tracker
        if tracker:
            await tracker.log_error(error_msg)
            if tracker.is_running:
                await tracker.complete()
        return

    try:
        # Start the tracker with Redis updates
        await tracker.start()

        # Run the main crawl logic
        await main_with_requests(tracker, process_only_new=process_only_new, redis_client=redis_client)

        # Mark completion using the tracker (which now updates Redis)
        await tracker.complete()

        # Additional success updates happen in tracker.complete()

    except Exception as e:
        error_msg = f"Unhandled exception during crawl: {e}"
        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        print(f"ERROR: {error_msg}")  # Print locally too

        # Update local tracker & Redis
        if tracker:
            await tracker.log_error(error_msg)
            # Approximate failed if stopped early
            tracker.urls_failed += (tracker.urls_found -
                                    tracker.urls_processed)
            tracker.urls_processed = tracker.urls_found
            if tracker.is_running:
                await tracker.complete()  # Mark tracker complete with Redis updates

    finally:
        # Final cleanup: ensure running flag is cleared in Redis, just in case
        try:
            await redis_client.delete(REDIS_KEY_RUNNING_FLAG)
        except Exception as redis_e:
            print(f"Redis Error during final cleanup: {redis_e}")

        # Local tracker completion (might be redundant if done in try/except)
        if tracker and tracker.is_running:
            await tracker.complete()


if __name__ == "__main__":
    print("Starting crawler...")
    # This direct execution won't have Redis unless manually set up
    # Consider adding dummy Redis or conditional logic if direct runs are needed
    # For now, it will likely fail if run directly due to missing redis_client

    async def dummy_main():
        print(
            "Running crawl_ai_docs.py directly is intended for testing internal functions.")
        print("To run the full crawl integrated with the API and Redis, start the backend.")
        # Example: Test a specific function like fetching sitemap
        # try:
        #     urls = get_pydantic_ai_docs_urls()
        #     print(f"Fetched {len(urls)} pydantic URLs.")
        # except Exception as e:
        #     print(f"Error testing sitemap fetch: {e}")

    asyncio.run(dummy_main())
    # asyncio.run(run_crawl_task()) # Original call - will fail without Redis client
    print("Crawler finished.")
