"""
Provider registry for managing multiple LLM providers.

Singleton registry that handles provider registration, retrieval, and fallback chains.
"""

from typing import Dict, List, Type, Optional, Any
from .base_provider import BaseProvider
import logging

logger = logging.getLogger(__name__)


class ProviderRegistry:
    """
    Singleton registry for managing LLM providers.
    
    Handles provider registration, retrieval, configuration, and fallback mechanisms.
    """
    
    _instance = None
    
    def __new__(cls):
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the registry."""
        if self._initialized:
            return
            
        self._providers: Dict[str, Type[BaseProvider]] = {}
        self._instances: Dict[str, BaseProvider] = {}
        self._aliases: Dict[str, str] = {}
        self._fallback_chain: List[str] = []
        self._default_provider: Optional[str] = None
        self._configs: Dict[str, Dict[str, Any]] = {}
        self._initialized = True
    
    def register(self, name: str, provider_class: Type[BaseProvider]) -> None:
        """
        Register a provider class.
        
        TODO: Add validation and error handling
        
        Args:
            name: Name to register the provider under
            provider_class: Provider class to register
            
        Raises:
            ValueError: If provider is already registered
        """
        if name in self._providers:
            raise ValueError(f"Provider '{name}' is already registered")
        
        self._providers[name] = provider_class
        logger.info(f"Registered provider: {name}")
    
    def unregister(self, name: str) -> None:
        """
        Unregister a provider.
        
        TODO: Clean up instances and aliases
        
        Args:
            name: Name of provider to unregister
        """
        if name in self._providers:
            del self._providers[name]
            
        if name in self._instances:
            del self._instances[name]
            
        # Remove from aliases
        self._aliases = {k: v for k, v in self._aliases.items() if v != name}
        
        # Remove from fallback chain
        self._fallback_chain = [p for p in self._fallback_chain if p != name]
        
        logger.info(f"Unregistered provider: {name}")
    
    def get_provider(self, name: str) -> BaseProvider:
        """
        Get a provider instance by name.
        
        TODO: Implement lazy instantiation and configuration
        
        Args:
            name: Name of the provider
            
        Returns:
            Provider instance
            
        Raises:
            KeyError: If provider not found
        """
        # Check aliases
        if name in self._aliases:
            name = self._aliases[name]
        
        if name not in self._providers:
            raise KeyError(f"Provider '{name}' not found")
        
        # Lazy instantiation
        if name not in self._instances:
            config = self._configs.get(name, {})
            self._instances[name] = self._providers[name](**config)
        
        return self._instances[name]
    
    def list_providers(self) -> List[str]:
        """
        List all registered provider names.
        
        TODO: Add filtering options
        
        Returns:
            List of provider names
        """
        return list(self._providers.keys())
    
    def get_provider_capabilities(self, name: str) -> Dict[str, Any]:
        """
        Get capabilities of a provider.
        
        TODO: Implement capability detection
        
        Args:
            name: Provider name
            
        Returns:
            Dictionary of capabilities
        """
        provider = self.get_provider(name)
        capabilities = provider.get_capabilities()
        
        return {
            "name": name,
            "supports_streaming": capabilities.supports_streaming,
            "supports_functions": capabilities.supports_functions,
            "supports_vision": capabilities.supports_vision,
            "max_context_length": capabilities.max_context_length,
            "max_output_tokens": capabilities.max_output_tokens,
            "chat_models": provider.get_chat_models(),
            "embedding_models": provider.get_embedding_models(),
        }
    
    def set_fallback_chain(self, providers: List[str]) -> None:
        """
        Set the fallback chain for providers.
        
        TODO: Validate provider names
        
        Args:
            providers: Ordered list of provider names
        """
        # Validate all providers exist
        for provider in providers:
            if provider not in self._providers and provider not in self._aliases:
                raise ValueError(f"Provider '{provider}' not registered")
        
        self._fallback_chain = providers
        logger.info(f"Set fallback chain: {providers}")
    
    def get_fallback_chain(self) -> List[str]:
        """
        Get the current fallback chain.
        
        Returns:
            List of provider names in fallback order
        """
        return self._fallback_chain.copy()
    
    async def get_available_provider(self) -> Optional[BaseProvider]:
        """
        Get the first available provider from the fallback chain.
        
        TODO: Implement availability checking
        
        Returns:
            First available provider or None
        """
        for provider_name in self._fallback_chain:
            try:
                provider = self.get_provider(provider_name)
                # Check if provider is available
                health = await provider.health_check()
                if health.get("status") == "healthy":
                    return provider
            except Exception as e:
                logger.warning(f"Provider {provider_name} not available: {e}")
                continue
        
        return None
    
    def add_alias(self, alias: str, provider_name: str) -> None:
        """
        Add an alias for a provider.
        
        TODO: Validate provider exists
        
        Args:
            alias: Alias name
            provider_name: Actual provider name
        """
        if provider_name not in self._providers:
            raise ValueError(f"Provider '{provider_name}' not registered")
        
        self._aliases[alias] = provider_name
        logger.info(f"Added alias '{alias}' for provider '{provider_name}'")
    
    def set_default_provider(self, name: str) -> None:
        """
        Set the default provider.
        
        TODO: Validate provider exists
        
        Args:
            name: Provider name
        """
        if name not in self._providers and name not in self._aliases:
            raise ValueError(f"Provider '{name}' not registered")
        
        self._default_provider = name
        logger.info(f"Set default provider: {name}")
    
    def get_default_provider(self) -> Optional[BaseProvider]:
        """
        Get the default provider.
        
        Returns:
            Default provider instance or None
        """
        if self._default_provider:
            return self.get_provider(self._default_provider)
        return None
    
    def update_provider_config(self, name: str, config: Dict[str, Any]) -> None:
        """
        Update configuration for a provider.
        
        TODO: Apply configuration to existing instances
        
        Args:
            name: Provider name
            config: Configuration dictionary
        """
        self._configs[name] = config
        
        # If instance exists, recreate with new config
        if name in self._instances:
            del self._instances[name]
            logger.info(f"Updated configuration for provider: {name}")
    
    async def check_provider_health(self, name: str) -> Dict[str, Any]:
        """
        Check health of a specific provider.
        
        TODO: Implement comprehensive health check
        
        Args:
            name: Provider name
            
        Returns:
            Health status dictionary
        """
        try:
            provider = self.get_provider(name)
            return await provider.health_check()
        except Exception as e:
            return {
                "status": "error",
                "provider": name,
                "message": str(e)
            }
    
    def clear(self) -> None:
        """Clear all registered providers (mainly for testing)."""
        self._providers.clear()
        self._instances.clear()
        self._aliases.clear()
        self._fallback_chain.clear()
        self._default_provider = None
        self._configs.clear()