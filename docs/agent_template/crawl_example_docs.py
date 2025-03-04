from __future__ import annotations

import os
import sys
import asyncio
import json
import re
import time
import traceback
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urljoin, urlparse
import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from supabase import Client, create_client
from openai import AsyncOpenAI

# Add the parent directory to sys.path to allow importing from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import get_env_var

# Load environment variables
load_dotenv()

# Initialize OpenAI client
openai_api_key = get_env_var("OPENAI_API_KEY")
openai_client = AsyncOpenAI(api_key=openai_api_key)
embedding_model = get_env_var("EMBEDDING_MODEL") or "text-embedding-3-small"

# Initialize Supabase client
supabase_url = get_env_var("SUPABASE_URL")
supabase_key = get_env_var("SUPABASE_SERVICE_KEY")
supabase = create_client(supabase_url, supabase_key)

# Configuration settings (adjust as needed)
BASE_URL = "https://example-technology.org/docs"  # Replace with the actual documentation site
CHUNK_SIZE = 1500  # Target size for content chunks
CHUNK_OVERLAP = 150  # Overlap between chunks to maintain context
MAX_PAGES = 500  # Maximum number of pages to process
RATE_LIMIT_DELAY = 0.5  # Delay between requests to avoid rate limiting

# Crawling state
processed_urls = set()
docs_progress_callback = None  # Will hold the progress update function

async def get_embedding(text: str) -> List[float]:
    """
    Generate an embedding for the given text using OpenAI's API.
    
    Args:
        text: The text to generate an embedding for
    
    Returns:
        List[float]: The embedding vector
    """
    # Truncate text if too long (OpenAI's embedding model has a token limit)
    if len(text) > 8000:
        text = text[:8000]
    
    # Call the OpenAI API to generate the embedding
    response = await openai_client.embeddings.create(
        model=embedding_model,
        input=text
    )
    
    # Return the embedding as a list of floats
    return response.data[0].embedding

def clean_content(html_content: str) -> str:
    """
    Clean HTML content by removing scripts, styles, and converting to plain text.
    
    Args:
        html_content: Raw HTML content
    
    Returns:
        str: Cleaned plain text
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.extract()
    
    # Extract text
    text = soup.get_text()
    
    # Clean up whitespace
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)
    
    return text

def split_into_chunks(text: str, title: str) -> List[Tuple[str, str]]:
    """
    Split content into overlapping chunks for better embedding and retrieval.
    
    Args:
        text: The text content to split
        title: The title of the page
    
    Returns:
        List[Tuple[str, str]]: List of (chunk_content, summary) tuples
    """
    chunks = []
    
    # Simple splitting by paragraphs first
    paragraphs = text.split('\n\n')
    current_chunk = ""
    
    for paragraph in paragraphs:
        # If adding this paragraph would exceed chunk size, save current chunk and start a new one
        if len(current_chunk) + len(paragraph) > CHUNK_SIZE:
            # Generate a simple summary for the chunk
            summary = f"Content from {title}: " + current_chunk[:100] + "..."
            chunks.append((current_chunk, summary))
            
            # Start new chunk with overlap from previous chunk for context
            if len(current_chunk) > CHUNK_OVERLAP:
                current_chunk = current_chunk[-CHUNK_OVERLAP:] + "\n\n" + paragraph
            else:
                current_chunk = paragraph
        else:
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
    
    # Add the last chunk if not empty
    if current_chunk:
        summary = f"Content from {title}: " + current_chunk[:100] + "..."
        chunks.append((current_chunk, summary))
    
    return chunks

async def process_page(url: str, client: httpx.AsyncClient) -> Optional[Dict[str, Any]]:
    """
    Process a documentation page: fetch content, clean, extract info, and split into chunks.
    
    Args:
        url: The URL of the page to process
        client: The httpx client to use for requests
    
    Returns:
        Optional[Dict[str, Any]]: Page data if successful, None otherwise
    """
    try:
        # Skip already processed URLs
        if url in processed_urls:
            return None
        
        processed_urls.add(url)
        
        # Fetch the page
        response = await client.get(url)
        if response.status_code != 200:
            print(f"Failed to fetch {url}, status code: {response.status_code}")
            return None
        
        html_content = response.text
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract title
        title_tag = soup.find('title')
        title = title_tag.text.strip() if title_tag else url.split("/")[-1]
        
        # Extract main content (adjust selectors for your specific documentation site)
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
        if main_content:
            content = clean_content(str(main_content))
        else:
            content = clean_content(html_content)
        
        return {
            "url": url,
            "title": title,
            "content": content,
        }
        
    except Exception as e:
        print(f"Error processing {url}: {e}")
        traceback.print_exc()
        return None

async def save_page_chunks(page_data: Dict[str, Any], chunk_number: int = 0) -> int:
    """
    Save page chunks to Supabase with embeddings.
    
    Args:
        page_data: Page data including content
        chunk_number: Starting chunk number
    
    Returns:
        int: Number of chunks saved
    """
    if not page_data or not page_data.get("content"):
        return chunk_number
    
    # Split content into chunks
    chunks = split_into_chunks(page_data["content"], page_data["title"])
    
    for chunk_content, chunk_summary in chunks:
        # Generate embedding for the chunk
        embedding = await get_embedding(chunk_content)
        
        # Prepare the record
        record = {
            "url": page_data["url"],
            "title": page_data["title"],
            "content": chunk_content,
            "summary": chunk_summary,
            "chunk_number": chunk_number,
            "embedding": embedding,
            "metadata": {
                "source": "example_docs",
                "processed_at": time.time()
            }
        }
        
        # Save to Supabase
        result = supabase.table("site_pages").insert(record).execute()
        
        if hasattr(result, 'error') and result.error is not None:
            print(f"Error saving chunk {chunk_number} of {page_data['url']}: {result.error}")
        
        chunk_number += 1
    
    return chunk_number

async def crawl_documentation(start_url: str, status_callback=None):
    """
    Crawl documentation site starting from the specified URL.
    
    Args:
        start_url: The URL to start crawling from
        status_callback: Optional callback function to report progress
    """
    global docs_progress_callback
    docs_progress_callback = status_callback
    
    to_crawl = [start_url]
    processed_urls.clear()
    chunk_number = 0
    
    # Create a list to store all found documentation URLs
    all_found_urls = []
    
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        while to_crawl and len(processed_urls) < MAX_PAGES:
            # Update progress
            if docs_progress_callback:
                docs_progress_callback({
                    "processed": len(processed_urls),
                    "queued": len(to_crawl),
                    "total_found": len(all_found_urls)
                })
            
            # Get the next URL to process
            current_url = to_crawl.pop(0)
            
            # Process the page
            page_data = await process_page(current_url, client)
            if page_data:
                # Save page chunks
                chunk_number = await save_page_chunks(page_data, chunk_number)
                
                # Find links to other documentation pages
                try:
                    response = await client.get(current_url)
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Find all links in the page
                    links = soup.find_all('a', href=True)
                    for link in links:
                        href = link['href']
                        
                        # Convert relative URLs to absolute
                        if not href.startswith(('http://', 'https://')):
                            href = urljoin(current_url, href)
                        
                        # Check if the link is within the documentation site
                        if href.startswith(BASE_URL) and href not in processed_urls and href not in to_crawl:
                            # Skip URLs that are likely not documentation pages
                            if any(skip in href for skip in ['twitter.com', 'github.com', 'facebook.com']):
                                continue
                                
                            to_crawl.append(href)
                            all_found_urls.append(href)
                    
                except Exception as e:
                    print(f"Error finding links in {current_url}: {e}")
            
            # Delay to avoid rate limiting
            await asyncio.sleep(RATE_LIMIT_DELAY)
    
    # Final progress update
    if docs_progress_callback:
        docs_progress_callback({
            "processed": len(processed_urls),
            "queued": 0,
            "total_found": len(all_found_urls),
            "completed": True
        })
    
    print(f"Crawling completed. Processed {len(processed_urls)} pages, saved {chunk_number} chunks.")
    return all_found_urls

def get_example_docs_urls() -> List[str]:
    """
    Get a list of known Example documentation URLs.
    
    Returns:
        List[str]: List of documentation URLs
    """
    # Replace this with actual code to get URLs or a specific list of URLs
    return [
        "https://example-technology.org/docs/getting-started",
        "https://example-technology.org/docs/concepts",
        "https://example-technology.org/docs/api-reference",
        # Add more URLs as needed
    ]

async def clear_existing_records():
    """Clear all existing records with source='example_docs' from the site_pages table."""
    try:
        result = supabase.table("site_pages").delete().eq("metadata->>source", "example_docs").execute()
        print("Cleared existing example_docs records from site_pages")
        return True
    except Exception as e:
        print(f"Error clearing records: {e}")
        return False

async def start_crawl_with_requests(status_callback=None):
    """
    Start the documentation crawling process.
    
    Args:
        status_callback: Optional callback to report progress
    """
    # Clear existing records first
    await clear_existing_records()
    
    # Get the URLs to crawl
    urls = get_example_docs_urls()
    
    # Start the crawling process from each URL
    for url in urls:
        await crawl_documentation(url, status_callback)
    
    return True

# CLI execution
if __name__ == "__main__":
    print("Starting Example documentation crawler...")
    asyncio.run(start_crawl_with_requests())
    print("Crawling complete!") 