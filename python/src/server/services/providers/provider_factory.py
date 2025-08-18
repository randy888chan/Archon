"""
Factory for creating provider instances with configuration.

Handles provider instantiation with proper configuration loading.
"""

from typing import Optional, Dict, Any
from .base_provider import BaseProvider
from .provider_registry import ProviderRegistry
import logging

logger = logging.getLogger(__name__)


class ProviderFactory:
    """
    Factory for creating configured provider instances.
    
    Handles loading configuration from database and environment variables.
    """
    
    def __init__(self, registry: Optional[ProviderRegistry] = None):
        """
        Initialize the factory.
        
        Args:
            registry: Provider registry to use (defaults to singleton)
        """
        self.registry = registry or ProviderRegistry()
    
    async def create_provider(
        self,
        provider_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ) -> BaseProvider:
        """
        Create a configured provider instance.
        
        TODO: Implement configuration loading from database
        TODO: Add environment variable fallback
        TODO: Validate configuration
        
        Args:
            provider_name: Name of the provider
            api_key: API key (optional, will load from config if not provided)
            base_url: Base URL (optional)
            **kwargs: Additional provider-specific configuration
            
        Returns:
            Configured provider instance
        """
        # Load configuration from database if not provided
        if not api_key:
            api_key = await self._load_api_key(provider_name)
        
        if not base_url:
            base_url = await self._load_base_url(provider_name)
        
        # Merge configurations
        config = {
            "api_key": api_key,
            "base_url": base_url,
            **kwargs
        }
        
        # Update registry configuration
        self.registry.update_provider_config(provider_name, config)
        
        # Get configured instance
        return self.registry.get_provider(provider_name)
    
    async def _load_api_key(self, provider_name: str) -> Optional[str]:
        """
        Load API key from configuration.
        
        TODO: Implement database loading
        TODO: Add environment variable fallback
        
        Args:
            provider_name: Provider name
            
        Returns:
            API key or None
        """
        # This will be connected to credential_service
        from ..credential_service import credential_service
        
        key_mapping = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
            "groq": "GROQ_API_KEY",
            "together": "TOGETHER_API_KEY",
            "mistral": "MISTRAL_API_KEY",
            "cohere": "COHERE_API_KEY",
        }
        
        key_name = key_mapping.get(provider_name)
        if key_name:
            return await credential_service.get_credential(key_name)
        
        return None
    
    async def _load_base_url(self, provider_name: str) -> Optional[str]:
        """
        Load base URL from configuration.
        
        TODO: Implement database loading
        TODO: Add provider-specific defaults
        
        Args:
            provider_name: Provider name
            
        Returns:
            Base URL or None
        """
        # This will be connected to credential_service
        from ..credential_service import credential_service
        
        # Load from settings
        url_key = f"{provider_name.upper()}_BASE_URL"
        base_url = await credential_service.get_credential(url_key)
        
        # Use defaults if not configured
        if not base_url:
            defaults = {
                "ollama": "http://localhost:11434/v1",
                "google": "https://generativelanguage.googleapis.com/v1beta/openai/",
            }
            base_url = defaults.get(provider_name)
        
        return base_url
    
    async def create_from_settings(self, service_type: str = "llm") -> BaseProvider:
        """
        Create a provider from database settings.
        
        TODO: Connect to existing credential service
        TODO: Handle embedding vs chat provider selection
        
        Args:
            service_type: Type of service ("llm" or "embedding")
            
        Returns:
            Configured provider instance
        """
        from ..credential_service import credential_service
        
        # Get active provider configuration
        provider_config = await credential_service.get_active_provider(service_type)
        
        provider_name = provider_config["provider"]
        api_key = provider_config["api_key"]
        base_url = provider_config["base_url"]
        
        return await self.create_provider(
            provider_name=provider_name,
            api_key=api_key,
            base_url=base_url
        )
    
    def get_available_providers(self) -> list[str]:
        """
        Get list of available providers.
        
        Returns:
            List of provider names
        """
        return self.registry.list_providers()
    
    async def validate_provider(
        self,
        provider_name: str,
        api_key: str
    ) -> Dict[str, Any]:
        """
        Validate provider credentials.
        
        TODO: Implement validation logic
        
        Args:
            provider_name: Provider name
            api_key: API key to validate
            
        Returns:
            Validation result dictionary
        """
        try:
            provider = await self.create_provider(provider_name, api_key=api_key)
            is_valid = await provider.validate_credentials(api_key)
            
            return {
                "valid": is_valid,
                "provider": provider_name,
                "message": "Credentials valid" if is_valid else "Invalid credentials"
            }
        except Exception as e:
            return {
                "valid": False,
                "provider": provider_name,
                "message": str(e)
            }