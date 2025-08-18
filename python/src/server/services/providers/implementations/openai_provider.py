"""
OpenAI provider implementation.

Provides access to OpenAI's GPT models and embeddings.
"""

from typing import Any, Dict, List, Optional
import openai
from ..base_provider import BaseProvider
from ..provider_config import ProviderModels, RateLimitConfig
import logging

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseProvider):
    """
    OpenAI provider implementation.
    
    Supports GPT-4, GPT-3.5, and embedding models.
    """
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs):
        """
        Initialize OpenAI provider.
        
        Args:
            api_key: OpenAI API key
            base_url: Optional base URL for API
            **kwargs: Additional configuration
        """
        super().__init__(api_key, base_url)
        self.organization_id = kwargs.get("organization_id")
        self._client: Optional[openai.AsyncOpenAI] = None
    
    async def get_client(self) -> openai.AsyncOpenAI:
        """
        Get OpenAI async client.
        
        TODO: Add connection pooling
        TODO: Add retry configuration
        
        Returns:
            Configured OpenAI client
        """
        if not self._client:
            self._client = openai.AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                organization=self.organization_id
            )
        return self._client
    
    def get_chat_models(self) -> List[str]:
        """
        Get available OpenAI chat models.
        
        TODO: Fetch dynamically from API
        
        Returns:
            List of model names
        """
        models = ProviderModels.for_openai()
        return models.chat_models
    
    def get_embedding_models(self) -> List[str]:
        """
        Get available OpenAI embedding models.
        
        TODO: Fetch dynamically from API
        
        Returns:
            List of embedding model names
        """
        models = ProviderModels.for_openai()
        return models.embedding_models
    
    async def validate_credentials(self, api_key: str) -> bool:
        """
        Validate OpenAI API key.
        
        TODO: Implement actual validation
        TODO: Cache validation results
        
        Args:
            api_key: API key to validate
            
        Returns:
            True if valid
        """
        try:
            # Create temporary client for validation
            temp_client = openai.AsyncOpenAI(api_key=api_key)
            
            # Try to list models as a validation check
            await temp_client.models.list()
            return True
        except openai.AuthenticationError:
            logger.warning("Invalid OpenAI API key")
            return False
        except Exception as e:
            logger.error(f"Error validating OpenAI credentials: {e}")
            return False
    
    def get_model_mapping(self) -> Dict[str, str]:
        """
        Get PydanticAI model mappings.
        
        Returns:
            Model name to PydanticAI format mapping
        """
        models = self.get_chat_models()
        return {model: f"openai:{model}" for model in models}
    
    def supports_streaming(self) -> bool:
        """
        Check if streaming is supported.
        
        Returns:
            True (OpenAI supports streaming)
        """
        return True
    
    def get_rate_limits(self) -> Dict[str, Any]:
        """
        Get OpenAI rate limits.
        
        TODO: Fetch actual limits from account
        TODO: Implement tier detection
        
        Returns:
            Rate limit configuration
        """
        # Default tier 1 limits
        config = RateLimitConfig(
            requests_per_minute=500,
            tokens_per_minute=60000,
            requests_per_day=10000,
            concurrent_requests=50
        )
        
        return {
            "requests_per_minute": config.requests_per_minute,
            "tokens_per_minute": config.tokens_per_minute,
            "requests_per_day": config.requests_per_day,
            "concurrent_requests": config.concurrent_requests,
            "delay_between_requests": config.get_delay_between_requests()
        }
    
    async def create_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-4o-mini",
        **kwargs
    ) -> Any:
        """
        Create a chat completion.
        
        TODO: Implement with proper error handling
        TODO: Add streaming support
        
        Args:
            messages: Chat messages
            model: Model to use
            **kwargs: Additional parameters
            
        Returns:
            Completion response
        """
        client = await self.get_client()
        
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )
            return response
        except openai.RateLimitError as e:
            logger.error(f"Rate limit error: {e}")
            raise
        except Exception as e:
            logger.error(f"Chat completion error: {e}")
            raise
    
    async def create_embedding(
        self,
        text: str,
        model: str = "text-embedding-3-small",
        **kwargs
    ) -> List[float]:
        """
        Create an embedding.
        
        TODO: Implement batch embedding
        TODO: Add dimension configuration
        
        Args:
            text: Text to embed
            model: Embedding model
            **kwargs: Additional parameters
            
        Returns:
            Embedding vector
        """
        client = await self.get_client()
        
        try:
            response = await client.embeddings.create(
                model=model,
                input=text,
                **kwargs
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Embedding creation error: {e}")
            raise