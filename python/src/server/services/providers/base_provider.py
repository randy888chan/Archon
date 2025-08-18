"""
Base provider interface for all LLM providers.

Defines the contract that all provider implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ProviderCapabilities:
    """Capabilities of a provider."""
    supports_streaming: bool = True
    supports_functions: bool = True
    supports_vision: bool = False
    max_context_length: int = 4096
    max_output_tokens: int = 4096


class BaseProvider(ABC):
    """
    Abstract base class for all LLM providers.
    
    Each provider implementation must inherit from this class and implement
    all abstract methods to ensure compatibility with the system.
    """
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize the provider.
        
        Args:
            api_key: API key for authentication
            base_url: Base URL for the provider's API (optional)
        """
        self.api_key = api_key
        self.base_url = base_url
        self.name = self.__class__.__name__.replace("Provider", "").lower()
    
    @abstractmethod
    async def get_client(self) -> Any:
        """
        Get the provider-specific client for making API calls.
        
        TODO: Implement for each provider
        
        Returns:
            Provider-specific client instance
        """
        pass
    
    @abstractmethod
    def get_chat_models(self) -> List[str]:
        """
        Get list of available chat models for this provider.
        
        TODO: Return provider-specific model list
        
        Returns:
            List of model identifiers
        """
        pass
    
    @abstractmethod
    def get_embedding_models(self) -> List[str]:
        """
        Get list of available embedding models for this provider.
        
        TODO: Return provider-specific embedding model list
        
        Returns:
            List of embedding model identifiers
        """
        pass
    
    @abstractmethod
    async def validate_credentials(self, api_key: str) -> bool:
        """
        Validate API credentials for this provider.
        
        TODO: Implement credential validation logic
        
        Args:
            api_key: API key to validate
            
        Returns:
            True if credentials are valid, False otherwise
        """
        pass
    
    @abstractmethod
    def get_model_mapping(self) -> Dict[str, str]:
        """
        Get mapping of model names to PydanticAI format.
        
        TODO: Return mapping like {"gpt-4": "openai:gpt-4"}
        
        Returns:
            Dictionary mapping model names to PydanticAI format
        """
        pass
    
    @abstractmethod
    def supports_streaming(self) -> bool:
        """
        Check if this provider supports streaming responses.
        
        TODO: Return provider streaming capability
        
        Returns:
            True if streaming is supported
        """
        pass
    
    @abstractmethod
    def get_rate_limits(self) -> Dict[str, Any]:
        """
        Get rate limiting configuration for this provider.
        
        TODO: Return provider-specific rate limits
        
        Returns:
            Dictionary with rate limit information
        """
        pass
    
    def get_capabilities(self) -> ProviderCapabilities:
        """
        Get provider capabilities.
        
        TODO: Override in specific providers if needed
        
        Returns:
            ProviderCapabilities instance
        """
        return ProviderCapabilities(
            supports_streaming=self.supports_streaming()
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the provider.
        
        TODO: Implement provider-specific health check
        
        Returns:
            Dictionary with health status information
        """
        try:
            is_valid = await self.validate_credentials(self.api_key or "")
            return {
                "status": "healthy" if is_valid else "unhealthy",
                "provider": self.name,
                "message": "Credentials valid" if is_valid else "Invalid credentials"
            }
        except Exception as e:
            return {
                "status": "error",
                "provider": self.name,
                "message": str(e)
            }
    
    def get_default_model(self, model_type: str = "chat") -> str:
        """
        Get the default model for a given type.
        
        TODO: Override in specific providers
        
        Args:
            model_type: Type of model ("chat" or "embedding")
            
        Returns:
            Default model identifier
        """
        if model_type == "chat":
            models = self.get_chat_models()
            return models[0] if models else ""
        else:
            models = self.get_embedding_models()
            return models[0] if models else ""