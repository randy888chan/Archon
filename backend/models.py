from pydantic import BaseModel
from typing import List, Dict, Optional, Any


class CrawlStatusResponse(BaseModel):
    message: str = "No crawl initiated yet."
    is_running: bool = False
    processed_count: int = 0
    total_urls: int = 0
    urls_succeeded: int = 0
    urls_failed: int = 0
    urls_skipped: int = 0
    current_url: Optional[str] = None
    errors: List[str] = []
    logs: List[str] = []
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_seconds: Optional[float] = None
    # state_dict: Optional[Dict[str, Any]] = None # Removed

# You might add other models here as needed, e.g., for request bodies if not inline
