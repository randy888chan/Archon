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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


load_dotenv()

embedding_client, supabase = get_clients()

embedding_model = get_env_var('EMBEDDING_MODEL') or 'text-embedding-3-small'


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
                {"role": "user",
                    "content": f"URL: {url}\n\nContent:\n{chunk[:1000]}..."}
            ],
            response_format={"type": "json_object"}
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


async def process_chunk(chunk: str, chunk_number: int, url: str, source: str) -> ProcessedChunk:
    """Process a single chunk of text."""
    # Get title and summary
    extracted = await get_title_and_summary(chunk, url)

    # Get embedding
    embedding = await get_embedding(chunk)

    # Create metadata
    metadata = {
        "source": source,
        "chunk_size": len(chunk),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "url_path": urlparse(url).path
    }

    return ProcessedChunk(
        url=url,
        source=source,  # Store the source
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
        print(
            f"Inserted chunk {chunk.chunk_number} for {chunk.url} (Source: {chunk.source})")
        return result
    except Exception as e:
        print(f"Error inserting chunk: {e}")
        return None


async def process_and_store_document(url: str, markdown: str, source: str, tracker: Optional[CrawlProgressTracker] = None):
    """Process a document and store its chunks in parallel."""

    chunks = chunk_text(markdown)

    if tracker:
        tracker.log(
            f"Split document into {len(chunks)} chunks for {url} (Source: {source})")

        if tracker.progress_callback:
            tracker.progress_callback(tracker.get_status())
    else:
        print(
            f"Split document into {len(chunks)} chunks for {url} (Source: {source})")

    tasks = [
        process_chunk(chunk, i, url, source)
        for i, chunk in enumerate(chunks)
    ]
    processed_chunks = await asyncio.gather(*tasks)

    if tracker:
        tracker.log(
            f"Processed {len(processed_chunks)} chunks for {url} (Source: {source})")

        if tracker.progress_callback:
            tracker.progress_callback(tracker.get_status())
    else:
        print(
            f"Processed {len(processed_chunks)} chunks for {url} (Source: {source})")

    insert_tasks = [
        insert_chunk(chunk)
        for chunk in processed_chunks
    ]
    await asyncio.gather(*insert_tasks)

    if tracker:
        tracker.chunks_stored += len(processed_chunks)
        tracker.log(
            f"Stored {len(processed_chunks)} chunks for {url} (Source: {source})")

        if tracker.progress_callback:
            tracker.progress_callback(tracker.get_status())
    else:
        print(
            f"Stored {len(processed_chunks)} chunks for {url} (Source: {source})")


def fetch_url_content(url: str) -> str:
    """Fetch content from a URL using requests and convert to markdown."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        # Convert HTML to Markdown
        markdown = html_converter.handle(response.text)

        # Clean up the markdown
        # Remove excessive newlines
        markdown = re.sub(r'\n{3,}', '\n\n', markdown)

        return markdown
    except Exception as e:
        raise Exception(f"Error fetching {url}: {str(e)}")


async def crawl_parallel_with_requests(urls_with_source: Dict[str, str], tracker: Optional[CrawlProgressTracker] = None, max_concurrent: int = 5):
    """Crawl multiple URLs in parallel with a concurrency limit using direct HTTP requests."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_url(url: str, source: str):
        async with semaphore:
            if tracker:
                tracker.log(f"Crawling: {url} (Source: {source})")
                if tracker.progress_callback:
                    tracker.progress_callback(tracker.get_status())
            else:
                print(f"Crawling: {url} (Source: {source})")

            try:
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
                        if tracker.progress_callback:
                            tracker.progress_callback(tracker.get_status())
                    else:
                        print(f"Successfully crawled: {url}")

                    await process_and_store_document(url, markdown, source, tracker)
                else:
                    if tracker:
                        tracker.urls_failed += 1
                        tracker.log(f"Failed: {url} - No content retrieved")
                        if tracker.progress_callback:
                            tracker.progress_callback(tracker.get_status())
                    else:
                        print(f"Failed: {url} - No content retrieved")
            except Exception as e:
                if tracker:
                    tracker.urls_failed += 1
                    tracker.log(f"Error processing {url}: {str(e)}")
                    if tracker.progress_callback:
                        tracker.progress_callback(tracker.get_status())
                else:
                    print(f"Error processing {url}: {str(e)}")
            finally:
                if tracker:
                    tracker.urls_processed += 1
                    if tracker.progress_callback:
                        tracker.progress_callback(tracker.get_status())

        await asyncio.sleep(1)

    if tracker:
        tracker.log(
            f"Processing {len(urls_with_source)} URLs from {len(set(urls_with_source.values()))} sources with concurrency {max_concurrent}")
        if tracker.progress_callback:
            tracker.progress_callback(tracker.get_status())
    else:
        print(
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


async def main_with_requests(tracker: Optional[CrawlProgressTracker] = None):
    """Main function using direct HTTP requests instead of browser automation."""
    start_time_main = time.time()
    try:
        if tracker:
            tracker.start()
        else:
            print("Starting crawling process...")

        sources = {
            "pydantic_ai_docs": get_pydantic_ai_docs_urls,
            "langgraph_docs": get_langgraph_docs_urls,
            "langgraphjs_docs": get_langgraphjs_docs_urls,
            "langsmith_docs": get_langsmith_docs_urls,
            "langchain_python_docs": get_langchain_python_docs_urls,
            "langchain_js_docs": get_langchain_js_docs_urls,
        }

        clear_start_time = time.time()

        clear_tasks = [asyncio.to_thread(
            clear_existing_records, source_name) for source_name in sources.keys()]
        results = await asyncio.gather(*clear_tasks, return_exceptions=True)
        for source_name, result in zip(sources.keys(), results):
            if isinstance(result, Exception):
                log_msg = f"Failed to clear records for {source_name}: {result}"
                if tracker:
                    tracker.log(log_msg)
                else:
                    print(log_msg)

        clear_duration = time.time() - clear_start_time
        log_msg_clear = f"Finished clearing records for {len(sources)} sources in {clear_duration:.2f} seconds."
        if tracker:
            tracker.log(log_msg_clear)
        else:
            print(log_msg_clear)

        fetch_start_time = time.time()
        all_urls_with_source = {}
        total_urls = 0

        fetch_tasks = [asyncio.to_thread(fetch_func)
                       for fetch_func in sources.values()]
        fetch_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

        for source_name, result in zip(sources.keys(), fetch_results):
            if isinstance(result, Exception):
                log_msg = f"Failed to fetch URLs for {source_name}: {result}"
                if tracker:
                    tracker.log(log_msg)
                else:
                    print(log_msg)
                continue
            elif isinstance(result, list):
                urls = result
                for url in urls:

                    all_urls_with_source[url] = source_name
                total_urls += len(urls)

        fetch_duration = time.time() - fetch_start_time

        if not all_urls_with_source:
            log_msg = f"No URLs found to crawl from any source after {fetch_duration:.2f} seconds."
            if tracker:
                tracker.log(log_msg)
                tracker.complete()
            else:
                print(log_msg)
            return

        log_msg_fetch = f"Found a total of {total_urls} URLs to crawl from {len(sources)} sources in {fetch_duration:.2f} seconds."
        if tracker:
            tracker.urls_found = total_urls
            tracker.log(log_msg_fetch)
        else:
            print(log_msg_fetch)

        crawl_start_time = time.time()
        await crawl_parallel_with_requests(all_urls_with_source, tracker)
        crawl_duration = time.time() - crawl_start_time
        log_msg_crawl = f"Crawling and processing finished in {crawl_duration:.2f} seconds."
        if tracker:
            tracker.log(log_msg_crawl)
        else:
            print(log_msg_crawl)

        if tracker:
            tracker.complete()
        else:
            print("Crawling process completed")

    except Exception as e:
        error_msg = f"Error in main crawling process: {str(e)}"
        if tracker:
            tracker.log(error_msg)
            if tracker.is_running:
                tracker.complete()
        else:
            print(error_msg)
    finally:
        main_duration = time.time() - start_time_main
        print(
            f"Total execution time for main_with_requests: {main_duration:.2f} seconds.")


def start_crawl_with_requests(progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> CrawlProgressTracker:
    """Start the crawling process using direct HTTP requests in a separate thread and return the tracker."""
    tracker = CrawlProgressTracker(progress_callback)

    def run_crawl():
        try:

            asyncio.run(main_with_requests(tracker))
        except Exception as e:
            error_msg = f"Error in crawl thread: {e}"
            print(error_msg)
            if tracker:
                tracker.log(f"Thread error: {str(e)}")
                if tracker.is_running:
                    tracker.complete()

    thread = threading.Thread(target=run_crawl)
    thread.daemon = True
    thread.start()

    return tracker


if __name__ == "__main__":
    print("Starting crawler...")
    asyncio.run(main_with_requests())
    print("Crawler finished.")
