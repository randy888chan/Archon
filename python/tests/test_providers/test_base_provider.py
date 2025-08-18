"""
Test suite for base provider interface.

Tests the contract that all providers must implement.
"""

import pytest
from abc import ABC
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock


class TestBaseProvider:
    """Test base provider interface contract."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider for testing."""
        from src.server.services.providers.base_provider import BaseProvider
        
        class MockProvider(BaseProvider):
            async def get_client(self):
                return AsyncMock()
            
            def get_chat_models(self) -> List[str]:
                return ["model-1", "model-2"]
            
            def get_embedding_models(self) -> List[str]:
                return ["embed-1", "embed-2"]
            
            async def validate_credentials(self, api_key: str) -> bool:
                return api_key == "valid-key"
            
            def get_model_mapping(self) -> Dict[str, str]:
                return {"model-1": "provider:model-1"}
            
            def supports_streaming(self) -> bool:
                return True
            
            def get_rate_limits(self) -> Dict[str, Any]:
                return {"requests_per_minute": 60}
        
        return MockProvider()

    def test_provider_is_abstract(self):
        """Test that BaseProvider cannot be instantiated directly."""
        from src.server.services.providers.base_provider import BaseProvider
        
        with pytest.raises(TypeError):
            BaseProvider()

    def test_provider_has_required_methods(self, mock_provider):
        """Test that provider has all required methods."""
        required_methods = [
            'get_client',
            'get_chat_models',
            'get_embedding_models',
            'validate_credentials',
            'get_model_mapping',
            'supports_streaming',
            'get_rate_limits'
        ]
        
        for method in required_methods:
            assert hasattr(mock_provider, method), f"Provider missing method: {method}"

    @pytest.mark.asyncio
    async def test_provider_client_creation(self, mock_provider):
        """Test that provider can create a client."""
        client = await mock_provider.get_client()
        assert client is not None

    def test_provider_model_listing(self, mock_provider):
        """Test that provider returns model lists."""
        chat_models = mock_provider.get_chat_models()
        assert isinstance(chat_models, list)
        assert len(chat_models) > 0
        
        embedding_models = mock_provider.get_embedding_models()
        assert isinstance(embedding_models, list)
        assert len(embedding_models) > 0

    @pytest.mark.asyncio
    async def test_provider_credential_validation(self, mock_provider):
        """Test that provider can validate credentials."""
        assert await mock_provider.validate_credentials("valid-key") == True
        assert await mock_provider.validate_credentials("invalid-key") == False

    def test_provider_model_mapping(self, mock_provider):
        """Test that provider returns model mappings."""
        mappings = mock_provider.get_model_mapping()
        assert isinstance(mappings, dict)
        assert len(mappings) > 0

    def test_provider_capability_detection(self, mock_provider):
        """Test that provider reports capabilities."""
        assert isinstance(mock_provider.supports_streaming(), bool)
        
        rate_limits = mock_provider.get_rate_limits()
        assert isinstance(rate_limits, dict)

    def test_provider_configuration(self):
        """Test that provider can be configured."""
        from src.server.services.providers.base_provider import BaseProvider
        
        # This will be implemented when we create the actual base class
        pass

    @pytest.mark.asyncio
    async def test_provider_error_handling(self, mock_provider):
        """Test that provider handles errors gracefully."""
        # Will be implemented with actual provider
        pass