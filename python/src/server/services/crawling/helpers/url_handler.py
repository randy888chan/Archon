"""
URL Handler Helper

Handles URL transformations and validations.
"""
import re
from urllib.parse import urlparse

from ....config.logfire_config import get_logger

logger = get_logger(__name__)


class URLHandler:
    """Helper class for URL operations."""
    
    @staticmethod
    def is_sitemap(url: str) -> bool:
        """
        Check if a URL is a sitemap with error handling.
        
        Args:
            url: URL to check
            
        Returns:
            True if URL is a sitemap, False otherwise
        """
        try:
            return url.endswith('sitemap.xml') or 'sitemap' in urlparse(url).path
        except Exception as e:
            logger.warning(f"Error checking if URL is sitemap: {e}")
            return False
    
    @staticmethod
    def is_txt(url: str) -> bool:
        """
        Check if a URL is a text file with error handling.
        
        Args:
            url: URL to check
            
        Returns:
            True if URL is a text file, False otherwise
        """
        try:
            return url.endswith('.txt')
        except Exception as e:
            logger.warning(f"Error checking if URL is text file: {e}")
            return False
    
    @staticmethod
    def transform_github_url(url: str) -> str:
        """
        Transform GitHub URLs to raw content URLs for better content extraction.
        
        Args:
            url: URL to transform
            
        Returns:
            Transformed URL (or original if not a GitHub file URL)
        """
        # Pattern for GitHub file URLs
        github_file_pattern = r'https://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.+)'
        match = re.match(github_file_pattern, url)
        if match:
            owner, repo, branch, path = match.groups()
            raw_url = f'https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}'
            logger.info(f"Transformed GitHub file URL to raw: {url} -> {raw_url}")
            return raw_url
        
        # Pattern for GitHub directory URLs
        github_dir_pattern = r'https://github\.com/([^/]+)/([^/]+)/tree/([^/]+)/(.+)'
        match = re.match(github_dir_pattern, url)
        if match:
            # For directories, we can't directly get raw content
            # Return original URL but log a warning
            logger.warning(f"GitHub directory URL detected: {url} - consider using specific file URLs or GitHub API")
        
        return url