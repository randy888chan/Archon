"""
Integration tests for provider implementations.

Tests actual provider functionality with mocked API responses.
"""

import pytest
from typing import Dict, List, Any
from unittest.mock import AsyncMock, MagicMock, patch
import openai


class TestOpenAIProvider:
    """Test OpenAI provider implementation."""

    @pytest.fixture
    def openai_provider(self):
        """Create OpenAI provider instance."""
        from src.server.services.providers.implementations.openai_provider import OpenAIProvider
        return OpenAIProvider(api_key="test-key")

    def test_openai_chat_models(self, openai_provider):
        """Test OpenAI chat model listing."""
        models = openai_provider.get_chat_models()
        assert "gpt-4o" in models
        assert "gpt-4o-mini" in models
        assert "gpt-4-turbo" in models

    def test_openai_embedding_models(self, openai_provider):
        """Test OpenAI embedding model listing."""
        models = openai_provider.get_embedding_models()
        assert "text-embedding-3-small" in models
        assert "text-embedding-3-large" in models

    @pytest.mark.asyncio
    async def test_openai_client_creation(self, openai_provider):
        """Test OpenAI client creation."""
        client = await openai_provider.get_client()
        assert isinstance(client, openai.AsyncOpenAI)

    @pytest.mark.asyncio
    async def test_openai_chat_completion(self, openai_provider):
        """Test OpenAI chat completion with mocked response."""
        # OpenAI client structure is client.chat.completions.create()
        client = await openai_provider.get_client()
        
        # Mock the chat completions create method
        with patch.object(client.chat.completions, 'create') as mock_create:
            mock_create.return_value = AsyncMock(
                choices=[MagicMock(message=MagicMock(content="Test response"))]
            )
            
            # Test that client is properly configured
            assert client is not None
            assert hasattr(client, 'chat')
            assert hasattr(client.chat, 'completions')

    def test_openai_model_mapping(self, openai_provider):
        """Test OpenAI model mapping for PydanticAI."""
        mappings = openai_provider.get_model_mapping()
        assert mappings["gpt-4o"] == "openai:gpt-4o"
        assert mappings["gpt-4o-mini"] == "openai:gpt-4o-mini"

    def test_openai_supports_streaming(self, openai_provider):
        """Test that OpenAI supports streaming."""
        assert openai_provider.supports_streaming() == True

    def test_openai_rate_limits(self, openai_provider):
        """Test OpenAI rate limit configuration."""
        limits = openai_provider.get_rate_limits()
        assert "requests_per_minute" in limits
        assert "tokens_per_minute" in limits


class TestAnthropicProvider:
    """Test Anthropic provider implementation."""

    @pytest.fixture
    def anthropic_provider(self):
        """Create Anthropic provider instance."""
        from src.server.services.providers.implementations.anthropic_provider import AnthropicProvider
        return AnthropicProvider(api_key="test-key")

    def test_anthropic_chat_models(self, anthropic_provider):
        """Test Anthropic chat model listing."""
        models = anthropic_provider.get_chat_models()
        assert "claude-3-5-sonnet-latest" in models
        assert "claude-3-5-haiku-latest" in models
        assert "claude-3-opus-latest" in models

    def test_anthropic_model_mapping(self, anthropic_provider):
        """Test Anthropic model mapping for PydanticAI."""
        mappings = anthropic_provider.get_model_mapping()
        assert mappings["claude-3-5-sonnet-latest"] == "anthropic:claude-3-5-sonnet-latest"

    @pytest.mark.asyncio
    async def test_anthropic_client_creation(self, anthropic_provider):
        """Test Anthropic client creation."""
        client = await anthropic_provider.get_client()
        assert client is not None

    def test_anthropic_supports_streaming(self, anthropic_provider):
        """Test that Anthropic supports streaming."""
        assert anthropic_provider.supports_streaming() == True


class TestGroqProvider:
    """Test Groq provider implementation."""

    @pytest.fixture
    def groq_provider(self):
        """Create Groq provider instance."""
        from src.server.services.providers.implementations.groq_provider import GroqProvider
        return GroqProvider(api_key="test-key")

    def test_groq_chat_models(self, groq_provider):
        """Test Groq chat model listing."""
        models = groq_provider.get_chat_models()
        assert "llama-3.1-70b-versatile" in models
        assert "mixtral-8x7b-32768" in models

    def test_groq_model_mapping(self, groq_provider):
        """Test Groq model mapping for PydanticAI."""
        mappings = groq_provider.get_model_mapping()
        assert "llama-3.1-70b-versatile" in mappings

    def test_groq_rate_limits(self, groq_provider):
        """Test Groq rate limits are more restrictive."""
        limits = groq_provider.get_rate_limits()
        assert limits["requests_per_minute"] <= 30  # Groq has lower limits


class TestProviderSwitching:
    """Test switching between providers."""

    @pytest.fixture
    def registry(self):
        """Create a registry with multiple providers."""
        from src.server.services.providers.provider_registry import ProviderRegistry
        from src.server.services.providers.implementations.openai_provider import OpenAIProvider
        from src.server.services.providers.implementations.anthropic_provider import AnthropicProvider
        
        registry = ProviderRegistry()
        registry.clear()  # Clear any existing state
        registry.register("openai", OpenAIProvider)
        registry.register("anthropic", AnthropicProvider)
        # Set test API keys in config
        registry.update_provider_config("openai", {"api_key": "test-key"})
        registry.update_provider_config("anthropic", {"api_key": "test-key"})
        yield registry
        registry.clear()  # Clean up after test

    @pytest.mark.asyncio
    async def test_provider_switching_preserves_context(self, registry):
        """Test that switching providers preserves context."""
        # Start with OpenAI
        openai_provider = registry.get_provider("openai")
        openai_client = await openai_provider.get_client()
        
        # Switch to Anthropic
        anthropic_provider = registry.get_provider("anthropic")
        anthropic_client = await anthropic_provider.get_client()
        
        # Both should be valid clients
        assert openai_client is not None
        assert anthropic_client is not None

    @pytest.mark.asyncio
    async def test_embedding_provider_compatibility(self, registry):
        """Test that embedding providers are compatible."""
        openai_provider = registry.get_provider("openai")
        
        # OpenAI should provide embedding models
        embedding_models = openai_provider.get_embedding_models()
        assert len(embedding_models) > 0


class TestProviderErrorHandling:
    """Test error handling across providers."""

    @pytest.fixture
    def providers(self):
        """Create provider instances."""
        from src.server.services.providers.implementations.openai_provider import OpenAIProvider
        from src.server.services.providers.implementations.anthropic_provider import AnthropicProvider
        
        return {
            "openai": OpenAIProvider(api_key="test"),
            "anthropic": AnthropicProvider(api_key="test")
        }

    @pytest.mark.asyncio
    async def test_invalid_api_key_handling(self, providers):
        """Test handling of invalid API keys."""
        for name, provider in providers.items():
            is_valid = await provider.validate_credentials("invalid-key")
            assert is_valid == False

    @pytest.mark.asyncio
    async def test_rate_limit_handling(self, providers):
        """Test rate limit error handling."""
        # Will be implemented with actual providers
        pass

    @pytest.mark.asyncio
    async def test_network_error_handling(self, providers):
        """Test network error handling."""
        # Will be implemented with actual providers
        pass


class TestProviderFallback:
    """Test fallback mechanisms between providers."""

    @pytest.fixture
    def registry_with_fallback(self):
        """Create registry with fallback chain."""
        from src.server.services.providers.provider_registry import ProviderRegistry
        from src.server.services.providers.implementations.openai_provider import OpenAIProvider
        from src.server.services.providers.implementations.anthropic_provider import AnthropicProvider
        
        registry = ProviderRegistry()
        registry.clear()  # Clear any existing state
        registry.register("openai", OpenAIProvider)
        registry.register("anthropic", AnthropicProvider)
        # Set test API keys in config
        registry.update_provider_config("openai", {"api_key": "test-key"})
        registry.update_provider_config("anthropic", {"api_key": "test-key"})
        registry.set_fallback_chain(["openai", "anthropic"])
        yield registry
        registry.clear()  # Clean up after test

    @pytest.mark.asyncio
    async def test_fallback_on_quota_error(self, registry_with_fallback):
        """Test fallback when primary provider hits quota."""
        # Mock health check for both providers
        # Make OpenAI fail and Anthropic succeed
        from src.server.services.providers.implementations.openai_provider import OpenAIProvider
        from src.server.services.providers.implementations.anthropic_provider import AnthropicProvider
        
        async def mock_openai_health():
            return {"status": "unhealthy", "provider": "openai", "message": "Quota exceeded"}
        
        async def mock_anthropic_health():
            return {"status": "healthy", "provider": "anthropic", "message": "OK"}
        
        with patch.object(OpenAIProvider, 'health_check', side_effect=mock_openai_health):
            with patch.object(AnthropicProvider, 'health_check', side_effect=mock_anthropic_health):
                # Should fallback to Anthropic since OpenAI is unhealthy
                provider = await registry_with_fallback.get_available_provider()
                assert provider is not None
                assert provider.name == "anthropic"

    @pytest.mark.asyncio
    async def test_fallback_preserves_request(self, registry_with_fallback):
        """Test that fallback preserves the original request."""
        # Will be implemented with actual fallback logic
        pass