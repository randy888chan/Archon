"""
RAG Service - Thin Coordinator

This service acts as a coordinator that delegates to specific strategy implementations.
It combines multiple RAG strategies in a pipeline fashion:

1. Base vector search
2. + Hybrid search (if enabled) - combines vector + keyword
3. + Reranking (if enabled) - reorders results using CrossEncoder
4. + Agentic RAG (if enabled) - enhanced code example search

Multiple strategies can be enabled simultaneously and work together.
"""

import os
from typing import List, Dict, Any, Optional, Tuple

# Import CrossEncoder for test compatibility
try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

from ...utils import get_supabase_client
from ..embeddings.embedding_service import create_embedding
from ...config.logfire_config import safe_span, get_logger

# Import all strategies
from .hybrid_search_strategy import HybridSearchStrategy
from .reranking_strategy import RerankingStrategy, create_reranking_strategy
from .agentic_rag_strategy import AgenticRAGStrategy

logger = get_logger(__name__)

# Fixed similarity threshold for RAG queries
SIMILARITY_THRESHOLD = 0.15


class RAGService:
    """
    Coordinator service that orchestrates multiple RAG strategies.
    
    This service delegates to strategy implementations and combines them
    based on configuration settings.
    """

    def __init__(self, supabase_client=None, reranking_model=None):
        """Initialize with optional supabase client and reranking model"""
        self.supabase_client = supabase_client or get_supabase_client()
        
        # Initialize strategies
        self.hybrid_strategy = HybridSearchStrategy(self.supabase_client)
        
        # Initialize reranking strategy with proper settings integration
        if reranking_model is not None:
            # Direct model passed (for tests)
            if hasattr(reranking_model, 'rerank_results'):
                self.reranking_strategy = reranking_model
                self.reranking_model = getattr(reranking_model, 'model', reranking_model)
            else:
                # It's a raw model, create strategy wrapper
                from .reranking_strategy import RerankingStrategy
                self.reranking_strategy = RerankingStrategy(model_instance=reranking_model)
                self.reranking_model = reranking_model
        else:
            # Load based on settings
            use_reranking = self.get_bool_setting("USE_RERANKING", False)
            if use_reranking:
                from .reranking_strategy import RerankingStrategy
                try:
                    self.reranking_strategy = RerankingStrategy()
                    self.reranking_model = getattr(self.reranking_strategy, 'model', None)
                except Exception as e:
                    logger.warning(f"Failed to load reranking strategy: {e}")
                    self.reranking_strategy = None
                    self.reranking_model = None
            else:
                self.reranking_strategy = None
                self.reranking_model = None
        
        self.agentic_strategy = AgenticRAGStrategy(self.supabase_client)

    def get_setting(self, key: str, default: str = "false") -> str:
        """Get a setting from the credential service or fall back to environment variable."""
        try:
            from ..credential_service import credential_service
            if hasattr(credential_service, '_cache') and credential_service._cache_initialized:
                cached_value = credential_service._cache.get(key)
                if isinstance(cached_value, dict) and cached_value.get("is_encrypted"):
                    encrypted_value = cached_value.get("encrypted_value")
                    if encrypted_value:
                        try:
                            return credential_service._decrypt_value(encrypted_value)
                        except Exception:
                            pass
                elif cached_value:
                    return str(cached_value)
            # Fallback to environment variable
            return os.getenv(key, default)
        except Exception:
            return os.getenv(key, default)

    def get_bool_setting(self, key: str, default: bool = False) -> bool:
        """Get a boolean setting from credential service."""
        value = self.get_setting(key, "false" if not default else "true")
        return value.lower() in ("true", "1", "yes", "on")

    async def rerank_results(self, query: str, results: List[Dict[str, Any]], content_key: str = "content") -> List[Dict[str, Any]]:
        """
        Backward compatibility method - delegates to reranking strategy.
        
        Args:
            query: The search query
            results: List of search results to rerank
            content_key: The key in each result dict containing text content
            
        Returns:
            Reranked list of results
        """
        if self.reranking_strategy:
            # Check if it's actually a RerankingStrategy instance
            from .reranking_strategy import RerankingStrategy
            if isinstance(self.reranking_strategy, RerankingStrategy):
                # It's a proper strategy object
                return await self.reranking_strategy.rerank_results(query, results, content_key)
            elif hasattr(self.reranking_strategy, 'predict'):
                # It's a raw model for tests - implement reranking inline
                if not results:
                    return results
                
                try:
                    # Extract texts from results
                    texts = [result.get(content_key, "") for result in results]
                    
                    # Create query-document pairs
                    pairs = [[query, text] for text in texts]
                    
                    # Get reranking scores
                    scores = self.reranking_strategy.predict(pairs)
                    
                    # Add scores to results and sort
                    for i, result in enumerate(results):
                        result["rerank_score"] = float(scores[i])
                    
                    # Sort by rerank score descending
                    reranked = sorted(results, key=lambda x: x.get("rerank_score", 0), reverse=True)
                    
                    return reranked
                except Exception as e:
                    logger.error(f"Error during raw model reranking: {e}")
                    return results
        
        return results

    async def search_documents_async(
        self,
        query: str,
        match_count: int = 5,
        filter_metadata: Optional[dict] = None,
        use_hybrid_search: bool = False,
        cached_api_key: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Async document search with hybrid search capability.
        
        Args:
            query: Search query string
            match_count: Number of results to return
            filter_metadata: Optional metadata filter dict
            use_hybrid_search: Whether to use hybrid search
            cached_api_key: Deprecated parameter for compatibility
            
        Returns:
            List of matching documents
        """
        with safe_span("rag_search_documents_async", 
                       query_length=len(query),
                       match_count=match_count,
                       hybrid_enabled=use_hybrid_search) as span:
            try:
                # Create embedding for the query
                query_embedding = await create_embedding(query)
                
                if not query_embedding:
                    logger.error("Failed to create embedding for query")
                    return []
                
                if use_hybrid_search:
                    # Use hybrid strategy
                    results = self.hybrid_strategy.search_documents_hybrid(
                        query=query,
                        query_embedding=query_embedding,
                        match_count=match_count,
                        filter_metadata=filter_metadata
                    )
                    span.set_attribute("search_mode", "hybrid")
                else:
                    # Use basic vector search
                    results = await self._basic_vector_search(
                        query_embedding=query_embedding,
                        match_count=match_count,
                        filter_metadata=filter_metadata
                    )
                    span.set_attribute("search_mode", "vector")
                
                span.set_attribute("results_found", len(results))
                return results
                
            except Exception as e:
                logger.error(f"Document search failed: {e}")
                span.set_attribute("error", str(e))
                return []

    async def _basic_vector_search(
        self,
        query_embedding: List[float],
        match_count: int,
        filter_metadata: Optional[dict] = None
    ) -> List[Dict[str, Any]]:
        """Perform basic vector search using RPC function."""
        try:
            # Build RPC parameters
            rpc_params = {
                "query_embedding": query_embedding,
                "match_count": match_count
            }
            
            # Add filter parameters
            if filter_metadata:
                if "source" in filter_metadata:
                    rpc_params["source_filter"] = filter_metadata["source"]
                    rpc_params["filter"] = {}
                else:
                    rpc_params["filter"] = filter_metadata
            else:
                rpc_params["filter"] = {}
            
            # Execute search
            response = self.supabase_client.rpc("match_crawled_pages", rpc_params).execute()
            
            # Filter by similarity threshold
            filtered_results = []
            if response.data:
                for result in response.data:
                    similarity = float(result.get("similarity", 0.0))
                    if similarity >= SIMILARITY_THRESHOLD:
                        filtered_results.append(result)
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Basic vector search failed: {e}")
            return []

    def search_documents(
        self,
        query: str,
        match_count: int = 5,
        filter_metadata: Optional[dict] = None,
        use_hybrid_search: bool = False,
        cached_api_key: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Sync version of document search - delegates to async implementation.
        
        This method exists for backward compatibility with existing sync code.
        """
        import asyncio
        
        # Check if we're already in an async context
        try:
            loop = asyncio.get_running_loop()
            # If we get here, we're in an async context - need to handle differently
            logger.warning("search_documents called from async context - results may be limited")
            # Return empty results rather than causing issues
            return []
        except RuntimeError:
            # Not in async context, safe to run
            return asyncio.run(self.search_documents_async(
                query, match_count, filter_metadata, use_hybrid_search, cached_api_key
            ))

    async def search_code_examples(
        self,
        query: str, 
        match_count: int = 10, 
        filter_metadata: Optional[Dict[str, Any]] = None,
        source_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for code examples - delegates to agentic strategy.
        
        Args:
            query: Query text
            match_count: Maximum number of results to return
            filter_metadata: Optional metadata filter
            source_id: Optional source ID to filter results
            
        Returns:
            List of matching code examples
        """
        return await self.agentic_strategy.search_code_examples(
            query=query,
            match_count=match_count,
            filter_metadata=filter_metadata,
            source_id=source_id,
            use_enhancement=True
        )

    async def perform_rag_query(self, query: str, source: str = None, match_count: int = 5) -> Tuple[bool, Dict[str, Any]]:
        """
        Perform a comprehensive RAG query that combines all enabled strategies.
        
        Pipeline:
        1. Start with vector search
        2. Apply hybrid search if enabled
        3. Apply reranking if enabled
        
        Args:
            query: The search query
            source: Optional source domain to filter results
            match_count: Maximum number of results to return
            
        Returns:
            Tuple of (success, result_dict)
        """
        with safe_span("rag_query_pipeline",
                       query_length=len(query),
                       source=source,
                       match_count=match_count) as span:
            try:
                logger.info(f"RAG query started: {query[:100]}{'...' if len(query) > 100 else ''}")
                
                # Build filter metadata
                filter_metadata = {"source": source} if source else None
                
                # Check which strategies are enabled
                use_hybrid_search = self.get_bool_setting("USE_HYBRID_SEARCH", False)
                use_reranking = self.get_bool_setting("USE_RERANKING", False)
                
                # Step 1 & 2: Get results (with hybrid search if enabled)
                results = await self.search_documents_async(
                    query=query,
                    match_count=match_count,
                    filter_metadata=filter_metadata,
                    use_hybrid_search=use_hybrid_search
                )
                
                span.set_attribute("raw_results_count", len(results))
                span.set_attribute("hybrid_search_enabled", use_hybrid_search)
                
                # Format results for processing
                formatted_results = []
                for i, result in enumerate(results):
                    try:
                        formatted_result = {
                            "id": result.get("id", f"result_{i}"),
                            "content": result.get("content", "")[:1000],  # Limit content
                            "metadata": result.get("metadata", {}),
                            "similarity_score": result.get("similarity", 0.0)
                        }
                        formatted_results.append(formatted_result)
                    except Exception as format_error:
                        logger.warning(f"Failed to format result {i}: {format_error}")
                        continue
                
                # Step 3: Apply reranking if we have a strategy or if enabled
                reranking_applied = False
                if self.reranking_strategy and formatted_results:
                    try:
                        formatted_results = await self.rerank_results(query, formatted_results, content_key="content")
                        reranking_applied = True
                        logger.debug(f"Reranking applied to {len(formatted_results)} results")
                    except Exception as e:
                        logger.warning(f"Reranking failed: {e}")
                        reranking_applied = False
                
                # Build response
                response_data = {
                    "results": formatted_results,
                    "query": query,
                    "source": source,
                    "match_count": match_count,
                    "total_found": len(formatted_results),
                    "execution_path": "rag_service_pipeline",
                    "search_mode": "hybrid" if use_hybrid_search else "vector",
                    "reranking_applied": reranking_applied
                }
                
                span.set_attribute("final_results_count", len(formatted_results))
                span.set_attribute("reranking_applied", reranking_applied)
                span.set_attribute("success", True)
                
                logger.info(f"RAG query completed - {len(formatted_results)} results found")
                return True, response_data
                
            except Exception as e:
                logger.error(f"RAG query failed: {e}")
                span.set_attribute("error", str(e))
                span.set_attribute("success", False)
                
                return False, {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "query": query,
                    "source": source,
                    "execution_path": "rag_service_pipeline"
                }

    async def search_code_examples_service(self, query: str, source_id: Optional[str] = None, match_count: int = 5) -> Tuple[bool, Dict[str, Any]]:
        """
        Search for code examples using agentic strategy with hybrid search and reranking.
        
        Pipeline for code examples:
        1. Check if agentic RAG is enabled
        2. Use agentic strategy for enhanced code search
        3. Apply hybrid search if enabled
        4. Apply reranking if enabled
        
        Args:
            query: The search query
            source_id: Optional source ID to filter results
            match_count: Maximum number of results to return
            
        Returns:
            Tuple of (success, result_dict)
        """
        with safe_span("code_examples_pipeline", 
                       query_length=len(query),
                       source_id=source_id,
                       match_count=match_count) as span:
            try:
                # Check if agentic RAG is enabled
                if not self.agentic_strategy.is_enabled():
                    return False, {
                        "error": "Code example extraction is disabled. Enable USE_AGENTIC_RAG setting to use this feature.",
                        "query": query
                    }
                
                # Check which strategies are enabled
                use_hybrid_search = self.get_bool_setting("USE_HYBRID_SEARCH", False)
                use_reranking = self.get_bool_setting("USE_RERANKING", False)
                
                # Prepare filter
                filter_metadata = {"source": source_id} if source_id and source_id.strip() else None
                
                if use_hybrid_search:
                    # Use hybrid search for code examples
                    results = self.hybrid_strategy.search_code_examples_hybrid(
                        query=query,
                        match_count=match_count,
                        filter_metadata=filter_metadata,
                        source_id=source_id
                    )
                else:
                    # Use standard agentic search
                    results = await self.agentic_strategy.search_code_examples_async(
                        query=query,
                        match_count=match_count,
                        filter_metadata=filter_metadata,
                        source_id=source_id,
                        use_enhancement=True
                    )
                
                # Apply reranking if we have a strategy
                if self.reranking_strategy and results:
                    try:
                        results = await self.rerank_results(query, results, content_key="content")
                    except Exception as e:
                        logger.warning(f"Code reranking failed: {e}")
                
                # Format results
                formatted_results = []
                for result in results:
                    formatted_result = {
                        "url": result.get("url"),
                        "code": result.get("content"),
                        "summary": result.get("summary"),
                        "metadata": result.get("metadata"),
                        "source_id": result.get("source_id"),
                        "similarity": result.get("similarity")
                    }
                    # Include rerank score if available
                    if "rerank_score" in result:
                        formatted_result["rerank_score"] = result["rerank_score"]
                    formatted_results.append(formatted_result)
                
                response_data = {
                    "query": query,
                    "source_filter": source_id,
                    "search_mode": "hybrid" if use_hybrid_search else "vector",
                    "reranking_applied": self.reranking_strategy is not None,
                    "results": formatted_results,
                    "count": len(formatted_results)
                }
                
                span.set_attribute("results_found", len(formatted_results))
                span.set_attribute("hybrid_used", use_hybrid_search)
                span.set_attribute("reranking_used", use_reranking)
                
                return True, response_data
                
            except Exception as e:
                logger.error(f"Code example search failed: {e}")
                span.set_attribute("error", str(e))
                return False, {
                    "query": query,
                    "error": str(e)
                }


# Legacy function wrappers for backward compatibility
def search_documents(
    client, query: str, match_count: int = 5, filter_metadata: Optional[dict] = None,
    use_hybrid_search: bool = False, cached_api_key: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Legacy wrapper for RAGService.search_documents"""
    service = RAGService(supabase_client=client)
    return service.search_documents(query, match_count, filter_metadata, use_hybrid_search, cached_api_key)


async def search_documents_async(
    client, query: str, match_count: int = 5, filter_metadata: Optional[dict] = None,
    use_hybrid_search: bool = False, cached_api_key: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Legacy wrapper for RAGService.search_documents_async"""
    service = RAGService(supabase_client=client)
    return await service.search_documents_async(query, match_count, filter_metadata, use_hybrid_search, cached_api_key)


async def search_code_examples(
    client, query: str, match_count: int = 10, filter_metadata: Optional[Dict[str, Any]] = None,
    source_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Legacy wrapper for RAGService.search_code_examples"""
    service = RAGService(supabase_client=client)
    return await service.search_code_examples(query, match_count, filter_metadata, source_id)