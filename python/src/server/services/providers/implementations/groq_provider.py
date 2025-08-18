"""
Groq provider implementation.

Provides access to Groq's high-speed inference for open models.
"""

from typing import Any, Dict, List, Optional
import openai
from ..base_provider import BaseProvider
from ..provider_config import ProviderModels, RateLimitConfig
import logging

logger = logging.getLogger(__name__)


class GroqProvider(BaseProvider):
    """
    Groq provider implementation.
    
    Supports Llama, Mixtral, and other open models with fast inference.
    Note: Groq has lower rate limits but very fast response times.
    """
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs):
        """
        Initialize Groq provider.
        
        TODO: Add Groq-specific optimizations
        
        Args:
            api_key: Groq API key
            base_url: Optional base URL
            **kwargs: Additional configuration
        """
        super().__init__(api_key, base_url)
        self._client: Optional[openai.AsyncOpenAI] = None
    
    async def get_client(self) -> openai.AsyncOpenAI:
        """
        Get Groq client using OpenAI-compatible interface.
        
        TODO: Add connection pooling for high throughput
        TODO: Optimize for Groq's fast response times
        
        Returns:
            Configured OpenAI client pointing to Groq
        """
        if not self._client:
            self._client = openai.AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url or "https://api.groq.com/openai/v1"
            )
        return self._client
    
    def get_chat_models(self) -> List[str]:
        """
        Get available Groq models.
        
        TODO: Fetch dynamically from API
        TODO: Add model performance metrics
        
        Returns:
            List of available models on Groq
        """
        models = ProviderModels.for_groq()
        return models.chat_models
    
    def get_embedding_models(self) -> List[str]:
        """
        Get embedding models.
        
        Note: Groq doesn't currently provide embedding models.
        
        Returns:
            Empty list
        """
        return []
    
    async def validate_credentials(self, api_key: str) -> bool:
        """
        Validate Groq API key.
        
        TODO: Implement actual validation
        TODO: Check rate limit status
        
        Args:
            api_key: API key to validate
            
        Returns:
            True if valid
        """
        try:
            # Create temporary client for validation
            temp_client = openai.AsyncOpenAI(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1"
            )
            
            # Try to list models
            await temp_client.models.list()
            return True
        except Exception as e:
            logger.error(f"Error validating Groq credentials: {e}")
            return False
    
    def get_model_mapping(self) -> Dict[str, str]:
        """
        Get PydanticAI model mappings.
        
        TODO: Verify Groq model support in PydanticAI
        
        Returns:
            Model name to PydanticAI format mapping
        """
        models = self.get_chat_models()
        return {model: f"groq:{model}" for model in models}
    
    def supports_streaming(self) -> bool:
        """
        Check if streaming is supported.
        
        Returns:
            True (Groq supports streaming)
        """
        return True
    
    def get_rate_limits(self) -> Dict[str, Any]:
        """
        Get Groq rate limits.
        
        Note: Groq has lower rate limits but much faster response times.
        
        TODO: Fetch actual limits from account
        TODO: Implement burst handling
        
        Returns:
            Rate limit configuration
        """
        # Groq has more restrictive rate limits
        config = RateLimitConfig(
            requests_per_minute=30,
            tokens_per_minute=30000,
            requests_per_day=14400,  # Approximate
            concurrent_requests=5
        )
        
        return {
            "requests_per_minute": config.requests_per_minute,
            "tokens_per_minute": config.tokens_per_minute,
            "requests_per_day": config.requests_per_day,
            "concurrent_requests": config.concurrent_requests,
            "delay_between_requests": config.get_delay_between_requests(),
            "note": "Groq has lower limits but much faster response times"
        }