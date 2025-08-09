# Fix for Stuck Crawl Issue

## Problem Summary
The crawl process gets stuck during browser automation, specifically when Crawl4AI is executing JavaScript (`update_image_dimensions_js`). The frontend detects this as stuck after 2 minutes of no updates and displays "Crawl appears to be stuck."

## Root Cause
The issue occurs when:
1. Crawl4AI's browser automation hangs while executing JavaScript on certain pages
2. The page timeout (30-45 seconds) isn't triggering properly
3. The crawler remains stuck indefinitely, blocking the async task

## Immediate Workarounds

### 1. Restart Docker Services
```bash
# Stop and restart the services to clear stuck processes
docker-compose restart archon-main
```

### 2. Clear Browser Cache
The crawler might be stuck due to browser state issues:
```bash
# Remove crawler cache and restart
docker-compose down
docker volume prune -f  # Clear any cached browser data
docker-compose up -d
```

### 3. Try Simpler URLs First
Test with simple, static pages to verify the crawler works:
- `https://example.com`
- `https://httpbin.org/html`

## Code Fixes

### Fix 1: Add Timeout to Crawler Operations
Update `/python/src/server/services/rag/crawling_service.py`:

```python
# Line 243 - Reduce page timeout and add overall timeout
page_timeout=15000,  # Reduced from 30000 (15 seconds instead of 30)

# Add asyncio timeout wrapper around crawl operations
import asyncio

async def crawl_with_timeout(self, url, config, timeout=60):
    """Crawl with hard timeout to prevent hanging."""
    try:
        return await asyncio.wait_for(
            self.crawler.crawl(url, config=config),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        logger.error(f"Crawl timed out after {timeout}s for {url}")
        raise Exception(f"Crawl timed out after {timeout} seconds")
```

### Fix 2: Disable Problematic JavaScript Execution
Update `/python/src/server/services/crawler_manager.py`:

Add these browser arguments to line 70:
```python
extra_args=[
    # ... existing args ...
    '--disable-javascript',  # Temporarily disable JS if pages are hanging
    '--disable-web-security=false',  # Re-enable security
    '--block-new-web-contents',  # Prevent popups that might hang
]
```

### Fix 3: Add Heartbeat During Long Operations
Update `/python/src/server/services/knowledge/crawl_orchestration_service.py`:

The heartbeat mechanism is already in place but might not be frequent enough. Reduce the interval:
```python
# Line 115
heartbeat_interval = 10.0  # Reduced from 30 seconds to 10 seconds
```

### Fix 4: Improve Stuck Detection and Recovery
Update `/python/src/server/services/rag/crawling_service.py`:

Add stuck detection within the crawler:
```python
async def crawl_single_page(self, url: str, **kwargs) -> Dict[str, Any]:
    """Crawl a single page with stuck detection."""
    max_attempts = 3
    
    for attempt in range(max_attempts):
        try:
            # Use asyncio timeout for each attempt
            result = await asyncio.wait_for(
                self._crawl_page_internal(url, **kwargs),
                timeout=45  # 45 second hard limit per page
            )
            if result and result.get('markdown'):
                return result
        except asyncio.TimeoutError:
            logger.warning(f"Attempt {attempt + 1} timed out for {url}")
            if attempt == max_attempts - 1:
                raise Exception(f"Page crawl timed out after {max_attempts} attempts")
            await asyncio.sleep(2)  # Brief pause before retry
        except Exception as e:
            logger.error(f"Crawl error on attempt {attempt + 1}: {e}")
            if attempt == max_attempts - 1:
                raise
    
    return None
```

## Configuration Adjustments

### 1. Reduce Concurrent Crawls
In `/python/src/server/services/knowledge/crawl_orchestration_service.py` line 495:
```python
# Reduce concurrent crawls to prevent resource exhaustion
max_concurrent = 5 if self._is_documentation_site(url) else 3  # Reduced from 20/10
```

### 2. Simplify Browser Configuration
For problematic sites, use a simpler configuration:
```python
# In crawler_manager.py, add a "safe mode" option
if os.getenv("CRAWLER_SAFE_MODE", "false") == "true":
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        browser_type="chromium",
        extra_args=[
            '--no-sandbox',
            '--disable-setuid-sandbox',
            '--disable-javascript',  # No JS execution
            '--disable-images',      # No images
            '--disable-plugins',     # No plugins
        ]
    )
```

## Environment Variables to Add

Add to your `.env` file:
```bash
# Crawler timeout settings
CRAWLER_PAGE_TIMEOUT=15000      # 15 seconds per page
CRAWLER_MAX_RETRIES=2           # Retry failed pages twice
CRAWLER_SAFE_MODE=false         # Enable for problematic sites
CRAWLER_HEARTBEAT_INTERVAL=10   # Heartbeat every 10 seconds
```

## Monitoring and Debugging

### Check Crawler Status
```bash
# Watch crawler logs in real-time
docker-compose logs -f archon-main | grep -E "CRAWLER|crawl|Crawl4AI"

# Check for stuck browser processes
docker exec archon-main ps aux | grep chromium
```

### Force Kill Stuck Crawls
If a crawl is truly stuck:
```bash
# Restart just the main service
docker-compose restart archon-main

# Or kill chromium processes inside container
docker exec archon-main pkill -f chromium
```

## Prevention Strategies

1. **Pre-validate URLs**: Check if URLs are accessible before crawling
2. **Implement circuit breaker**: Track failing URLs and skip them temporarily
3. **Use simpler crawl modes**: For problematic sites, disable JavaScript and complex rendering
4. **Monitor resource usage**: Check CPU/memory during crawls to detect issues early

## Testing the Fix

1. Apply the timeout changes above
2. Restart services: `docker-compose restart`
3. Test with a known problematic URL
4. Monitor logs: `docker-compose logs -f archon-main`
5. Verify the crawl completes or fails cleanly within the timeout period

## Long-term Solution

Consider implementing a crawler worker pool with process isolation:
- Each crawl runs in a separate process
- Processes are killed if they exceed time limits
- Failed crawls are automatically retried with simpler settings
- Resource limits prevent system exhaustion

This ensures that one stuck crawl cannot block the entire system.