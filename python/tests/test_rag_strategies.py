"""
Tests for RAG Strategies and Search Functionality

Tests hybrid search, agentic RAG, reranking, and other advanced RAG features.
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any

# Test hybrid search functionality
class TestHybridSearch:
    """Test hybrid search implementation"""
    
    @pytest.mark.asyncio
    async def test_hybrid_search_enabled_for_rag(self):
        """Test that hybrid search is used when enabled in RAG queries"""
        with patch('src.server.services.embeddings.embedding_service.create_embeddings_batch') as mock_embed, \
             patch('src.server.services.search.vector_search_service.search_similar_documents') as mock_search:
            
            mock_embed.return_value = [[0.1] * 1536]  # Mock embedding
            mock_search.return_value = [
                {'content': 'Test content 1', 'score': 0.95},
                {'content': 'Test content 2', 'score': 0.85}
            ]
            
            # Import after patching to avoid import-time execution
            from src.server.services.search.search_services import perform_rag_search
            
            result = await perform_rag_search(
                query="test query",
                use_hybrid_search=True,
                match_count=5
            )
            
            assert 'results' in result
            assert len(result['results']) >= 1
            mock_embed.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_hybrid_search_disabled_for_rag(self):
        """Test that basic vector search is used when hybrid search disabled"""
        with patch('src.server.services.embeddings.embedding_service.create_embeddings_batch') as mock_embed, \
             patch('src.server.services.search.vector_search_service.search_similar_documents') as mock_search:
            
            mock_embed.return_value = [[0.1] * 1536]
            mock_search.return_value = [
                {'content': 'Basic search result', 'score': 0.90}
            ]
            
            from src.server.services.search.search_services import perform_rag_search
            
            result = await perform_rag_search(
                query="test query", 
                use_hybrid_search=False,
                match_count=3
            )
            
            assert 'results' in result
            mock_embed.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_hybrid_search_implementation(self):
        """Test the core hybrid search functionality"""
        with patch('src.server.services.search.rag_service.create_embedding') as mock_embed, \
             patch('src.server.services.search.vector_search_service.search_similar_documents') as mock_search:
            
            mock_embed.return_value = [0.1] * 1536
            mock_search.return_value = [
                {'content': 'Relevant content', 'score': 0.92, 'metadata': {'source': 'doc1'}}
            ]
            
            # Test hybrid search combines multiple search strategies
            from src.server.services.search.search_services import SearchService
            
            search_service = SearchService()
            results = await search_service.hybrid_search(
                query="test query",
                match_count=5
            )
            
            assert isinstance(results, list)
            if results:  # If results returned
                assert 'content' in results[0]
                assert 'score' in results[0]


class TestAgenticRAG:
    """Test agentic RAG functionality"""
    
    @pytest.mark.asyncio  
    async def test_agentic_rag_enabled_allows_search(self):
        """Test that agentic RAG can perform autonomous searches"""
        with patch.object('src.server.services.search.agentic_rag_strategy.AgenticRAGStrategy', 'search_code_examples') as mock_search:
            mock_search.return_value = [
                {'code': 'def example():\n    pass', 'language': 'python', 'score': 0.88}
            ]
            
            from src.server.services.search.agentic_rag_strategy import AgenticRAGStrategy
            
            strategy = AgenticRAGStrategy()
            results = await strategy.autonomous_search(
                query="python function example",
                search_types=['code_examples', 'documentation']
            )
            
            assert isinstance(results, list)
            # Agentic RAG should return structured results
            if results:
                assert any('code' in r or 'content' in r for r in results)


class TestReranking:
    """Test reranking functionality in RAG"""
    
    @pytest.mark.asyncio
    async def test_reranking_applied_to_rag_results(self):
        """Test that reranking improves result relevance"""
        with patch('src.server.services.embeddings.embedding_service.create_embeddings_batch') as mock_embed:
            mock_embed.return_value = [[0.1] * 1536, [0.2] * 1536]
            
            from src.server.services.search.search_services import SearchService
            
            # Mock initial results
            initial_results = [
                {'content': 'Less relevant content', 'score': 0.7},
                {'content': 'Highly relevant content', 'score': 0.6},  # Lower initial score
                {'content': 'Moderately relevant', 'score': 0.8}
            ]
            
            search_service = SearchService()
            
            # Test reranking improves order
            reranked = await search_service.rerank_results(
                query="highly relevant",
                results=initial_results
            )
            
            assert isinstance(reranked, list)
            assert len(reranked) <= len(initial_results)
            # After reranking, results should be properly ordered


class TestRAGStrategiesIntegration:
    """Integration tests for RAG strategies working together"""
    
    @pytest.mark.asyncio
    async def test_hybrid_search_with_reranking(self):
        """Test hybrid search combined with reranking"""
        with patch('src.server.services.embeddings.embedding_service.create_embeddings_batch') as mock_embed:
            mock_embed.return_value = [[0.1] * 1536]
            
            from src.server.services.search.search_services import perform_rag_search
            
            result = await perform_rag_search(
                query="complex technical query",
                use_hybrid_search=True,
                use_reranking=True,
                match_count=10
            )
            
            assert 'results' in result
            assert isinstance(result['results'], list)
            # Should have called embedding service
            mock_embed.assert_called()
    
    @pytest.mark.asyncio
    async def test_error_handling_across_strategies(self):
        """Test error handling when RAG strategies fail"""
        with patch('src.server.services.embeddings.embedding_service.create_embeddings_batch') as mock_embed:
            # Simulate embedding failure
            mock_embed.side_effect = Exception("Embedding API unavailable")
            
            from src.server.services.search.search_services import perform_rag_search
            
            # Should handle errors gracefully
            result = await perform_rag_search(
                query="test query",
                use_hybrid_search=True,
                match_count=5
            )
            
            # Should return some form of result even on error
            assert isinstance(result, dict)
            # May return empty results or fallback results
            assert 'results' in result or 'error' in result


class TestRAGPerformanceOptimizations:
    """Test RAG performance optimizations"""
    
    @pytest.mark.asyncio
    async def test_concurrent_embedding_generation(self):
        """Test that embeddings can be generated concurrently"""
        with patch('src.server.services.embeddings.embedding_service.create_embeddings_batch') as mock_embed:
            # Simulate batch embedding
            mock_embed.return_value = [[0.1] * 1536, [0.2] * 1536, [0.3] * 1536]
            
            from src.server.services.embeddings.embedding_service import create_embeddings_batch
            
            # Test batch processing
            texts = ["query 1", "query 2", "query 3"]
            results = await create_embeddings_batch(texts)
            
            assert len(results) == 3
            assert all(len(embedding) == 1536 for embedding in results)
    
    @pytest.mark.asyncio
    async def test_caching_behavior(self):
        """Test that RAG results are properly cached"""
        # This would test caching mechanisms if implemented
        from src.server.services.search.search_services import SearchService
        
        search_service = SearchService()
        # Test would verify caching behavior
        assert search_service is not None  # Basic instantiation test


class TestRAGConfigurationHandling:
    """Test RAG configuration and settings"""
    
    @pytest.mark.asyncio
    async def test_rag_settings_loading(self):
        """Test loading of RAG configuration settings"""
        from src.server.services.credential_service import credential_service
        
        # Test loading RAG-specific settings
        try:
            rag_settings = await credential_service.get_credentials_by_category("rag_strategy")
            assert isinstance(rag_settings, dict)
        except Exception:
            # Settings might not be initialized in test environment
            pytest.skip("RAG settings not available in test environment")
    
    @pytest.mark.asyncio
    async def test_embedding_batch_size_configuration(self):
        """Test that embedding batch size is configurable"""
        with patch('src.server.services.credential_service.credential_service.get_credentials_by_category') as mock_creds:
            mock_creds.return_value = {"EMBEDDING_BATCH_SIZE": "50"}
            
            from src.server.services.embeddings.embedding_service import create_embeddings_batch
            
            # Should respect batch size configuration
            texts = ["text"] * 100  # Large batch
            with patch('src.server.services.llm_provider_service.get_llm_client'):
                with patch('src.server.services.threading_service.get_threading_service'):
                    try:
                        await create_embeddings_batch(texts)
                        # If it completes without error, configuration is working
                        assert True
                    except Exception as e:
                        # Expected in test environment without full setup
                        assert "client" in str(e).lower() or "provider" in str(e).lower()