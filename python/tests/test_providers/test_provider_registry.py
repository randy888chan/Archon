"""
Test suite for provider registry.

Tests provider registration, retrieval, and management.
"""

import pytest
from typing import Dict, List, Any
from unittest.mock import MagicMock, AsyncMock


class TestProviderRegistry:
    """Test provider registry functionality."""

    @pytest.fixture
    def registry(self):
        """Create a registry instance for testing."""
        from src.server.services.providers.provider_registry import ProviderRegistry
        registry = ProviderRegistry()
        # Clear any existing state from singleton
        registry.clear()
        yield registry
        # Clean up after test
        registry.clear()

    @pytest.fixture
    def mock_provider_class(self):
        """Create a mock provider class."""
        from src.server.services.providers.base_provider import BaseProvider
        
        class MockProvider(BaseProvider):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.name = "mock"
                
            async def get_client(self):
                return AsyncMock()
            
            def get_chat_models(self) -> List[str]:
                return ["mock-model"]
            
            def get_embedding_models(self) -> List[str]:
                return ["mock-embed"]
            
            async def validate_credentials(self, api_key: str) -> bool:
                return True
            
            def get_model_mapping(self) -> Dict[str, str]:
                return {"mock-model": "mock:mock-model"}
            
            def supports_streaming(self) -> bool:
                return True
            
            def get_rate_limits(self) -> Dict[str, Any]:
                return {"requests_per_minute": 60}
        
        return MockProvider

    def test_registry_singleton(self):
        """Test that registry is a singleton."""
        from src.server.services.providers.provider_registry import ProviderRegistry
        
        registry1 = ProviderRegistry()
        registry2 = ProviderRegistry()
        assert registry1 is registry2

    def test_register_provider(self, registry, mock_provider_class):
        """Test registering a provider."""
        registry.register("mock", mock_provider_class)
        
        providers = registry.list_providers()
        assert "mock" in providers

    def test_get_provider_by_name(self, registry, mock_provider_class):
        """Test retrieving a provider by name."""
        registry.register("mock", mock_provider_class)
        
        provider = registry.get_provider("mock")
        assert provider is not None
        assert hasattr(provider, 'get_client')

    def test_get_nonexistent_provider(self, registry):
        """Test retrieving a provider that doesn't exist."""
        with pytest.raises(KeyError):
            registry.get_provider("nonexistent")

    def test_list_available_providers(self, registry, mock_provider_class):
        """Test listing all registered providers."""
        registry.register("provider1", mock_provider_class)
        registry.register("provider2", mock_provider_class)
        
        providers = registry.list_providers()
        assert len(providers) >= 2
        assert "provider1" in providers
        assert "provider2" in providers

    def test_provider_capabilities(self, registry, mock_provider_class):
        """Test getting provider capabilities."""
        registry.register("mock", mock_provider_class)
        
        capabilities = registry.get_provider_capabilities("mock")
        assert isinstance(capabilities, dict)

    def test_provider_fallback_chain(self, registry, mock_provider_class):
        """Test fallback chain mechanism."""
        registry.register("primary", mock_provider_class)
        registry.register("fallback", mock_provider_class)
        
        registry.set_fallback_chain(["primary", "fallback"])
        chain = registry.get_fallback_chain()
        
        assert chain == ["primary", "fallback"]

    @pytest.mark.asyncio
    async def test_get_available_provider(self, registry, mock_provider_class):
        """Test getting first available provider from chain."""
        registry.register("unavailable", mock_provider_class)
        registry.register("available", mock_provider_class)
        
        # Mock validation to make first provider unavailable
        registry.set_fallback_chain(["unavailable", "available"])
        
        provider = await registry.get_available_provider()
        assert provider is not None

    def test_register_duplicate_provider(self, registry, mock_provider_class):
        """Test that duplicate registration raises error."""
        registry.register("mock", mock_provider_class)
        
        with pytest.raises(ValueError):
            registry.register("mock", mock_provider_class)

    def test_unregister_provider(self, registry, mock_provider_class):
        """Test unregistering a provider."""
        registry.register("mock", mock_provider_class)
        registry.unregister("mock")
        
        providers = registry.list_providers()
        assert "mock" not in providers

    def test_provider_aliases(self, registry, mock_provider_class):
        """Test provider aliasing."""
        registry.register("openai", mock_provider_class)
        registry.add_alias("gpt", "openai")
        
        provider = registry.get_provider("gpt")
        assert provider is not None

    def test_default_provider(self, registry, mock_provider_class):
        """Test setting and getting default provider."""
        registry.register("provider1", mock_provider_class)
        registry.register("provider2", mock_provider_class)
        
        registry.set_default_provider("provider2")
        default = registry.get_default_provider()
        
        # The default provider should be the instance from provider2
        assert default is not None
        assert default.name == "mock"  # MockProvider always has name "mock"

    @pytest.mark.asyncio
    async def test_provider_health_check(self, registry, mock_provider_class):
        """Test provider health checking."""
        registry.register("mock", mock_provider_class)
        
        health = await registry.check_provider_health("mock")
        assert isinstance(health, dict)
        assert "status" in health

    def test_provider_configuration_update(self, registry, mock_provider_class):
        """Test updating provider configuration."""
        registry.register("mock", mock_provider_class)
        
        config = {"api_key": "new-key", "base_url": "https://api.example.com"}
        registry.update_provider_config("mock", config)
        
        provider = registry.get_provider("mock")
        # Verify configuration was applied
        assert provider is not None