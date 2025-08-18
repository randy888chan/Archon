"""
Ollama provider implementation.

Provides access to locally-hosted models via Ollama.
"""

from typing import Any, Dict, List, Optional
import openai
import httpx
from ..base_provider import BaseProvider
from ..provider_config import RateLimitConfig
import logging

logger = logging.getLogger(__name__)


class OllamaProvider(BaseProvider):
    """
    Ollama provider implementation.
    
    Supports local model hosting with Ollama.
    No API key required, connects to local Ollama instance.
    """
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs):
        """
        Initialize Ollama provider.
        
        TODO: Add health checking for local instance
        TODO: Support remote Ollama instances
        
        Args:
            api_key: Not used for Ollama (kept for interface compatibility)
            base_url: Ollama server URL (defaults to localhost)
            **kwargs: Additional configuration
        """
        # Ollama doesn't need an API key
        super().__init__(api_key="ollama", base_url=base_url)
        self._client: Optional[openai.AsyncOpenAI] = None
        self._available_models: Optional[List[str]] = None
    
    async def get_client(self) -> openai.AsyncOpenAI:
        """
        Get Ollama client using OpenAI-compatible interface.
        
        TODO: Add connection retry logic
        TODO: Implement connection pooling
        
        Returns:
            Configured OpenAI client pointing to Ollama
        """
        if not self._client:
            self._client = openai.AsyncOpenAI(
                api_key="ollama",  # Required but unused by Ollama
                base_url=self.base_url or "http://localhost:11434/v1"
            )
        return self._client
    
    async def _fetch_available_models(self) -> List[str]:
        """
        Fetch available models from Ollama instance.
        
        TODO: Cache results with TTL
        TODO: Handle connection errors gracefully
        
        Returns:
            List of available model names
        """
        try:
            # Use the non-OpenAI endpoint to list models
            base = (self.base_url or "http://localhost:11434").replace("/v1", "")
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{base}/api/tags")
                response.raise_for_status()
                data = response.json()
                
                models = []
                for model in data.get("models", []):
                    models.append(model["name"])
                
                self._available_models = models
                return models
        except Exception as e:
            logger.error(f"Failed to fetch Ollama models: {e}")
            return []
    
    def get_chat_models(self) -> List[str]:
        """
        Get available Ollama models.
        
        TODO: Implement async model fetching
        TODO: Categorize models by capability
        
        Returns:
            List of model names (returns common models as fallback)
        """
        # Return cached models or common defaults
        if self._available_models:
            return self._available_models
        
        # Common Ollama models as fallback
        return [
            "llama3.3:latest",
            "llama3.1:latest",
            "mistral:latest",
            "mixtral:latest",
            "gemma2:latest",
            "qwen2.5:latest",
            "phi3:latest",
            "deepseek-coder-v2:latest",
        ]
    
    def get_embedding_models(self) -> List[str]:
        """
        Get available embedding models.
        
        TODO: Fetch actual embedding models from Ollama
        
        Returns:
            List of embedding models
        """
        return [
            "nomic-embed-text:latest",
            "all-minilm:latest",
            "mxbai-embed-large:latest",
        ]
    
    async def validate_credentials(self, api_key: str) -> bool:
        """
        Validate Ollama connection.
        
        Note: Ollama doesn't use API keys, this checks connectivity.
        
        TODO: Implement actual connectivity check
        TODO: Check Ollama version compatibility
        
        Args:
            api_key: Ignored for Ollama
            
        Returns:
            True if Ollama is accessible
        """
        try:
            # Check if Ollama is running
            base = (self.base_url or "http://localhost:11434").replace("/v1", "")
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{base}/api/tags", timeout=5.0)
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama connection check failed: {e}")
            return False
    
    def get_model_mapping(self) -> Dict[str, str]:
        """
        Get PydanticAI model mappings.
        
        TODO: Verify Ollama support in PydanticAI
        
        Returns:
            Model name to PydanticAI format mapping
        """
        models = self.get_chat_models()
        # Ollama models might need special handling in PydanticAI
        return {model: f"openai:{model}" for model in models}
    
    def supports_streaming(self) -> bool:
        """
        Check if streaming is supported.
        
        Returns:
            True (Ollama supports streaming)
        """
        return True
    
    def get_rate_limits(self) -> Dict[str, Any]:
        """
        Get Ollama rate limits.
        
        Note: Local Ollama has no rate limits, only hardware constraints.
        
        Returns:
            Rate limit configuration (very high limits for local)
        """
        # Local instance has no API rate limits
        config = RateLimitConfig(
            requests_per_minute=1000,
            tokens_per_minute=1000000,
            requests_per_day=None,
            concurrent_requests=10  # Limited by local resources
        )
        
        return {
            "requests_per_minute": config.requests_per_minute,
            "tokens_per_minute": config.tokens_per_minute,
            "requests_per_day": config.requests_per_day,
            "concurrent_requests": config.concurrent_requests,
            "delay_between_requests": 0,  # No delay needed for local
            "note": "Local Ollama has no API limits, only hardware constraints"
        }
    
    async def pull_model(self, model_name: str) -> bool:
        """
        Pull a model to the Ollama instance.
        
        TODO: Implement model pulling
        TODO: Add progress tracking
        
        Args:
            model_name: Name of model to pull
            
        Returns:
            True if successful
        """
        try:
            base = (self.base_url or "http://localhost:11434").replace("/v1", "")
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base}/api/pull",
                    json={"name": model_name},
                    timeout=None  # Model downloads can take a while
                )
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to pull model {model_name}: {e}")
            return False