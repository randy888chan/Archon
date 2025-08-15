"""
Crawler Manager Service

Handles initialization and management of the Crawl4AI crawler instance.
This avoids circular imports by providing a service-level access to the crawler.
"""

import os
from typing import Optional

try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig
except ImportError:
    AsyncWebCrawler = None
    BrowserConfig = None

from ..config.logfire_config import get_logger, safe_logfire_error, safe_logfire_info

logger = get_logger(__name__)


class CrawlerManager:
    """Manages the global crawler instance."""

    _instance: Optional["CrawlerManager"] = None
    _crawler: AsyncWebCrawler | None = None
    _initialized: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    async def get_crawler(self) -> AsyncWebCrawler:
        """Get or create the crawler instance."""
        if not self._initialized:
            await self.initialize()
        return self._crawler

    async def initialize(self):
        """Initialize the crawler if not already initialized."""
        if self._initialized and self._crawler is not None:
            safe_logfire_info("Crawler already initialized, skipping")
            return

        try:
            safe_logfire_info("Initializing Crawl4AI crawler...")
            logger.info("=== CRAWLER INITIALIZATION START ===")

            # Clean up any existing crawler first
            if self._crawler is not None:
                logger.info("Cleaning up existing crawler before reinitializing...")
                try:
                    await self._crawler.__aexit__(None, None, None)
                except Exception as cleanup_e:
                    logger.warning(f"Error during crawler cleanup: {cleanup_e}")
                finally:
                    self._crawler = None
                    self._initialized = False

            # Check if crawl4ai is available
            if not AsyncWebCrawler or not BrowserConfig:
                logger.error("ERROR: crawl4ai not available")
                logger.error(f"AsyncWebCrawler: {AsyncWebCrawler}")
                logger.error(f"BrowserConfig: {BrowserConfig}")
                raise ImportError("crawl4ai is not installed or available")

            # Check for Docker environment
            in_docker = os.path.exists("/.dockerenv") or os.getenv("DOCKER_CONTAINER", False)

            # Initialize browser config - enhanced for stability
            browser_config = BrowserConfig(
                headless=True,
                verbose=False,
                # Set viewport for proper rendering
                viewport_width=1920,
                viewport_height=1080,
                # Add user agent to appear as a real browser
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                # Set browser type
                browser_type="chromium",
                # Extra args for Chromium - optimized for speed and stability
                extra_args=[
                    "--disable-blink-features=AutomationControlled",
                    "--disable-dev-shm-usage",
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                    "--disable-web-security",
                    "--disable-features=IsolateOrigins,site-per-process",
                    # Performance optimizations
                    "--disable-images",  # Skip image loading for faster page loads
                    "--disable-gpu",
                    "--disable-extensions",
                    "--disable-plugins",
                    "--disable-background-timer-throttling",
                    "--disable-backgrounding-occluded-windows",
                    "--disable-renderer-backgrounding",
                    "--disable-features=TranslateUI",
                    "--disable-ipc-flooding-protection",
                    # Additional speed optimizations
                    "--aggressive-cache-discard",
                    "--disable-background-networking",
                    "--disable-default-apps",
                    "--disable-sync",
                    "--metrics-recording-only",
                    "--no-first-run",
                    "--disable-popup-blocking",
                    "--disable-prompt-on-repost",
                    "--disable-domain-reliability",
                    "--disable-component-update",
                    # Stability improvements for browser context
                    "--disable-session-crashed-bubbles",
                    "--disable-infobars",
                    "--disable-crash-reporter",
                    "--disable-logging",
                    "--no-crash-upload",
                    "--disable-breakpad",
                ],
            )

            safe_logfire_info(f"Creating AsyncWebCrawler with config | in_docker={in_docker}")

            # Initialize crawler with the correct parameter name
            self._crawler = AsyncWebCrawler(config=browser_config)
            safe_logfire_info("AsyncWebCrawler instance created, entering context...")
            
            # Add timeout for context initialization
            import asyncio
            try:
                await asyncio.wait_for(self._crawler.__aenter__(), timeout=30.0)
            except asyncio.TimeoutError:
                logger.error("Crawler context initialization timed out")
                self._crawler = None
                self._initialized = False
                raise Exception("Crawler context initialization timed out after 30 seconds")
            
            self._initialized = True
            safe_logfire_info(f"Crawler entered context successfully | crawler={self._crawler}")

            safe_logfire_info("✅ Crawler initialized successfully")
            logger.info("=== CRAWLER INITIALIZATION SUCCESS ===")
            logger.info(f"Crawler instance: {self._crawler}")
            logger.info(f"Initialized: {self._initialized}")

        except Exception as e:
            safe_logfire_error(f"Failed to initialize crawler: {e}")
            import traceback

            tb = traceback.format_exc()
            safe_logfire_error(f"Crawler initialization traceback: {tb}")
            # Log error details
            logger.error("=== CRAWLER INITIALIZATION ERROR ===")
            logger.error(f"Error: {e}")
            logger.error(f"Traceback:\n{tb}")
            logger.error("=== END CRAWLER ERROR ===")
            # Don't mark as initialized if the crawler is None
            # This allows retries and proper error propagation
            self._crawler = None
            self._initialized = False
            raise Exception(f"Failed to initialize Crawl4AI crawler: {e}")

    async def cleanup(self):
        """Clean up the crawler resources."""
        if self._crawler and self._initialized:
            try:
                await self._crawler.__aexit__(None, None, None)
                safe_logfire_info("Crawler cleaned up successfully")
            except Exception as e:
                safe_logfire_error(f"Error cleaning up crawler: {e}")
            finally:
                self._crawler = None
                self._initialized = False

    async def force_reinitialize(self):
        """Force reinitialize the crawler - useful when context issues occur."""
        safe_logfire_info("Force reinitializing crawler due to context issues")
        try:
            await self.cleanup()
        except Exception as cleanup_e:
            safe_logfire_error(f"Error during force cleanup: {cleanup_e}")
        
        # Wait a moment before reinitializing
        import asyncio
        await asyncio.sleep(1.0)
        
        try:
            await self.initialize()
            safe_logfire_info("Force reinitialization completed successfully")
        except Exception as init_e:
            safe_logfire_error(f"Force reinitialization failed: {init_e}")
            raise


# Global instance
_crawler_manager = CrawlerManager()


async def get_crawler() -> AsyncWebCrawler | None:
    """Get the global crawler instance with enhanced error recovery."""
    global _crawler_manager
    
    # Always try to reinitialize if we don't have a valid crawler
    # This handles cases where the browser context was closed
    if not _crawler_manager._initialized or _crawler_manager._crawler is None:
        logger.info("Crawler not initialized or None, attempting to initialize...")
        try:
            await _crawler_manager.initialize()
        except Exception as e:
            logger.error(f"Failed to initialize crawler: {e}")
            return None
    
    crawler = await _crawler_manager.get_crawler()
    
    # Additional validation - try to verify the crawler is still working
    if crawler is not None:
        try:
            # Quick check to see if the crawler context is still valid
            # We can do this by checking if we can access browser info
            if hasattr(crawler, '_browser_manager') and crawler._browser_manager is not None:
                # The crawler appears to be in a good state
                return crawler
            else:
                logger.warning("Crawler browser_manager is None, reinitializing...")
                await _crawler_manager.cleanup()
                await _crawler_manager.initialize()
                return _crawler_manager._crawler
        except Exception as e:
            logger.warning(f"Crawler validation failed, reinitializing: {e}")
            try:
                await _crawler_manager.cleanup()
                await _crawler_manager.initialize()
                return _crawler_manager._crawler
            except Exception as init_e:
                logger.error(f"Failed to reinitialize crawler: {init_e}")
                return None
    
    if crawler is None:
        logger.warning("get_crawler() returning None")
        logger.warning(f"_crawler_manager: {_crawler_manager}")
        logger.warning(
            f"_crawler_manager._crawler: {_crawler_manager._crawler if _crawler_manager else 'N/A'}"
        )
        logger.warning(
            f"_crawler_manager._initialized: {_crawler_manager._initialized if _crawler_manager else 'N/A'}"
        )
    return crawler


async def initialize_crawler():
    """Initialize the global crawler."""
    await _crawler_manager.initialize()


async def cleanup_crawler():
    """Clean up the global crawler."""
    await _crawler_manager.cleanup()
