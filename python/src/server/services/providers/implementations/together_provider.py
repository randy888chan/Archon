"""
Together AI provider implementation.

Provides access to Together AI's model hosting platform.
"""

from typing import Any, Dict, List, Optional
import openai
from ..base_provider import BaseProvider
from ..provider_config import ProviderModels, RateLimitConfig
import logging

logger = logging.getLogger(__name__)


class TogetherProvider(BaseProvider):
    """
    Together AI provider implementation.
    
    Supports various open-source models including Llama, Mixtral, and others.
    Also provides embedding models.
    """
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs):
        """
        Initialize Together provider.
        
        TODO: Add model routing optimization
        
        Args:
            api_key: Together API key
            base_url: Optional base URL
            **kwargs: Additional configuration
        """
        super().__init__(api_key, base_url)
        self._client: Optional[openai.AsyncOpenAI] = None
    
    async def get_client(self) -> openai.AsyncOpenAI:
        """
        Get Together client using OpenAI-compatible interface.
        
        TODO: Add Together-specific features
        TODO: Implement model availability checking
        
        Returns:
            Configured OpenAI client pointing to Together
        """
        if not self._client:
            self._client = openai.AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url or "https://api.together.xyz/v1"
            )
        return self._client
    
    def get_chat_models(self) -> List[str]:
        """
        Get available Together AI models.
        
        TODO: Fetch dynamically from API
        TODO: Add model categorization (size, speed, capability)
        TODO: Filter by availability
        
        Returns:
            List of available models
        """
        models = ProviderModels.for_together()
        return models.chat_models
    
    def get_embedding_models(self) -> List[str]:
        """
        Get available embedding models.
        
        TODO: Add more embedding models
        
        Returns:
            List of embedding models
        """
        models = ProviderModels.for_together()
        return models.embedding_models
    
    async def validate_credentials(self, api_key: str) -> bool:
        """
        Validate Together API key.
        
        TODO: Implement actual validation
        TODO: Check account balance/credits
        
        Args:
            api_key: API key to validate
            
        Returns:
            True if valid
        """
        try:
            # Create temporary client for validation
            temp_client = openai.AsyncOpenAI(
                api_key=api_key,
                base_url="https://api.together.xyz/v1"
            )
            
            # Try to list models
            await temp_client.models.list()
            return True
        except Exception as e:
            logger.error(f"Error validating Together credentials: {e}")
            return False
    
    def get_model_mapping(self) -> Dict[str, str]:
        """
        Get PydanticAI model mappings.
        
        TODO: Handle Together's model naming format
        
        Returns:
            Model name to PydanticAI format mapping
        """
        models = self.get_chat_models()
        return {model: f"together:{model}" for model in models}
    
    def supports_streaming(self) -> bool:
        """
        Check if streaming is supported.
        
        Returns:
            True (Together supports streaming)
        """
        return True
    
    def get_rate_limits(self) -> Dict[str, Any]:
        """
        Get Together rate limits.
        
        TODO: Fetch actual limits from account
        TODO: Handle different model tiers
        
        Returns:
            Rate limit configuration
        """
        config = RateLimitConfig(
            requests_per_minute=60,
            tokens_per_minute=100000,
            requests_per_day=None,
            concurrent_requests=20
        )
        
        return {
            "requests_per_minute": config.requests_per_minute,
            "tokens_per_minute": config.tokens_per_minute,
            "requests_per_day": config.requests_per_day,
            "concurrent_requests": config.concurrent_requests,
            "delay_between_requests": config.get_delay_between_requests()
        }