"""
Vector Search Service

Handles vector similarity search for documents and code examples.
"""
from typing import List, Dict, Any, Optional
from supabase import Client

from ...config.logfire_config import safe_span, get_logger

logger = get_logger(__name__)
from ..embeddings.embedding_service import create_embedding, create_embedding_async, get_dimension_column_name
from ..embeddings.dimension_validator import validate_rpc_parameters, log_dimension_operation, validate_embedding_dimensions
from ..embeddings.exceptions import VectorSearchError, DimensionValidationError

# Fixed similarity threshold for RAG queries
# Could make this configurable in the future, but that is unnecessary for now
SIMILARITY_THRESHOLD = 0.15


def build_rpc_params(query_embedding, match_count, filter_metadata=None, source_filter=None):
    """Build RPC parameters with dimension-specific embedding parameter.
    
    Args:
        query_embedding: The query embedding vector
        match_count: Number of results to return
        filter_metadata: Optional metadata filter dict
        source_filter: Optional source filter string
        
    Returns:
        Dictionary with appropriate RPC parameters
        
    Raises:
        DimensionValidationError: If embedding validation fails and no fallback allowed
    """
    try:
        # Validate embedding dimensions
        is_valid, dims, error_msg = validate_embedding_dimensions(
            query_embedding, 
            allow_fallback=True
        )
        
        if not is_valid:
            logger.warning(f"RPC parameter validation: {error_msg}")
            log_dimension_operation("vector_search_rpc", dims, False, error_msg)
            # Use fallback parameter with validated dimensions
            param_name = f"query_embedding_{dims}"
        else:
            log_dimension_operation("vector_search_rpc", dims, True)
            param_name = f"query_embedding_{dims}"
        
    except Exception as e:
        logger.error(f"Failed to validate embedding dimensions: {e}")
        # Fallback to default 1536-dimensional parameter
        param_name = "query_embedding_1536"
        dims = 1536
        log_dimension_operation("vector_search_rpc", dims, False, f"Validation exception: {e}")
    
    # Build parameters dictionary
    params = {
        param_name: query_embedding if query_embedding else [0.0] * dims,
        "match_count": match_count,
        "filter": filter_metadata or {}
    }
    
    if source_filter:
        params["source_filter"] = source_filter
    
    # Validate final RPC parameters for security
    try:
        validated_params = validate_rpc_parameters(params)
        return validated_params
    except DimensionValidationError as e:
        logger.error(f"RPC parameter security validation failed: {e}")
        # Return original params if validation fails but log the issue
        return params




def search_documents(
    client: Client,
    query: str,
    match_count: int = 5,
    filter_metadata: Optional[dict] = None,
    use_hybrid_search: bool = False,
    cached_api_key: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Search for documents in the database using semantic search.
    
    Args:
        client: Supabase client
        query: Search query string
        match_count: Number of results to return
        filter_metadata: Optional metadata filter dict
        use_hybrid_search: Whether to use hybrid keyword + semantic search
        cached_api_key: Cached OpenAI API key for embeddings (deprecated)
    
    Returns:
        List of matching documents
    """
    with safe_span("vector_search", 
                           query_length=len(query),
                           match_count=match_count,
                           has_filter=filter_metadata is not None) as span:
        try:
            logger.info(f"Document search started - query: {query[:100]}{'...' if len(query) > 100 else ''}, match_count: {match_count}, filter: {filter_metadata}")
            
            # Create embedding for the query
            with safe_span("create_embedding"):
                query_embedding = create_embedding(query)
                
                if not query_embedding:
                    logger.error("Failed to create embedding for query")
                    return []
                
                span.set_attribute("embedding_dimensions", len(query_embedding))
            
            # Build the filter for the RPC call
            with safe_span("prepare_rpc_params"):
                # Handle source filter extraction
                source_filter = None
                final_filter_metadata = filter_metadata
                
                if filter_metadata and "source" in filter_metadata:
                    source_filter = filter_metadata["source"]
                    # Use empty filter for the general filter parameter
                    final_filter_metadata = {}
                
                # Build RPC params with dimension-specific embedding parameter
                rpc_params = build_rpc_params(
                    query_embedding,
                    match_count,
                    final_filter_metadata,
                    source_filter
                )
                
                if filter_metadata:
                    span.set_attribute("filter_applied", True)
                    span.set_attribute("filter_keys", list(filter_metadata.keys()) if filter_metadata else [])
            
            # Call the RPC function
            with safe_span("supabase_rpc_call"):
                logger.debug(f"Calling Supabase RPC function: match_archon_crawled_pages, params: {list(rpc_params.keys())}")
                
                response = client.rpc("match_archon_crawled_pages", rpc_params).execute()
                
                # Apply threshold filtering to results
                filtered_results = []
                if response.data:
                    for result in response.data:
                        similarity = float(result.get("similarity", 0.0))
                        if similarity >= SIMILARITY_THRESHOLD:
                            filtered_results.append(result)
                
                span.set_attribute("rpc_success", True)
                span.set_attribute("raw_results_count", len(response.data) if response.data else 0)
                span.set_attribute("filtered_results_count", len(filtered_results))
                span.set_attribute("threshold_used", SIMILARITY_THRESHOLD)
            
            results_count = len(filtered_results)
            
            span.set_attribute("success", True)
            span.set_attribute("final_results_count", results_count)
            
            # Enhanced logging for debugging
            if results_count == 0:
                logger.warning(f"Document search returned 0 results - query: {query[:100]}{'...' if len(query) > 100 else ''}, raw_count: {len(response.data) if response.data else 0}, filter: {filter_metadata}")
            else:
                logger.info(f"Document search completed - query: {query[:100]}{'...' if len(query) > 100 else ''}, results: {results_count}, raw_count: {len(response.data) if response.data else 0}")
            
            return filtered_results
        
        except Exception as e:
            span.set_attribute("success", False)
            span.set_attribute("error", str(e))
            
            logger.error(f"Document search failed - query: {query[:100]}{'...' if len(query) > 100 else ''}, error: {e} ({type(e).__name__})")
            
            # Return empty list on error
            return []


async def search_documents_async(
    client: Client,
    query: str,
    match_count: int = 5,
    filter_metadata: Optional[dict] = None,
    use_hybrid_search: bool = False,
    cached_api_key: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Async version of search_documents that properly awaits embedding creation.
    
    Args:
        client: Supabase client
        query: Search query string
        match_count: Number of results to return
        filter_metadata: Optional metadata filter dict
        use_hybrid_search: Whether to use hybrid keyword + semantic search
        cached_api_key: Cached OpenAI API key for embeddings (deprecated)
    
    Returns:
        List of matching documents
    """
    with safe_span("vector_search_async", 
                           query_length=len(query),
                           match_count=match_count,
                           has_filter=filter_metadata is not None) as span:
        try:
            logger.info(f"Document search started (async) - query: {query[:100]}{'...' if len(query) > 100 else ''}, match_count: {match_count}, filter: {filter_metadata}")
            
            # Create embedding for the query - using async version
            with safe_span("create_embedding_async"):
                query_embedding = await create_embedding_async(query)
                
                if not query_embedding:
                    logger.error("Failed to create embedding for query")
                    return []
                
                span.set_attribute("embedding_dimensions", len(query_embedding))
            
            # Build the filter for the RPC call
            with safe_span("prepare_rpc_params"):
                # Handle source filter extraction
                source_filter = None
                final_filter_metadata = filter_metadata
                
                if filter_metadata and "source" in filter_metadata:
                    source_filter = filter_metadata["source"]
                    # Use empty filter for the general filter parameter
                    final_filter_metadata = {}
                
                # Build RPC params with dimension-specific embedding parameter
                rpc_params = build_rpc_params(
                    query_embedding,
                    match_count,
                    final_filter_metadata,
                    source_filter
                )
                
                if filter_metadata:
                    span.set_attribute("filter_applied", True)
                    span.set_attribute("filter_keys", list(filter_metadata.keys()) if filter_metadata else [])
            
            # Call the RPC function
            with safe_span("supabase_rpc_call"):
                logger.debug(f"Calling Supabase RPC function: match_archon_crawled_pages, params: {list(rpc_params.keys())}")
                
                response = client.rpc("match_archon_crawled_pages", rpc_params).execute()
                
                # Apply threshold filtering to results
                filtered_results = []
                if response.data:
                    for result in response.data:
                        similarity = float(result.get("similarity", 0.0))
                        if similarity >= SIMILARITY_THRESHOLD:
                            filtered_results.append(result)
                
                span.set_attribute("rpc_success", True)
                span.set_attribute("raw_results_count", len(response.data) if response.data else 0)
                span.set_attribute("filtered_results_count", len(filtered_results))
           
            results_count = len(filtered_results)
            
            span.set_attribute("success", True)
            span.set_attribute("final_results_count", results_count)
            
            logger.info(f"Document search completed (async) - query: {query[:100]}{'...' if len(query) > 100 else ''}, results: {results_count}")
            
            return filtered_results
        
        except Exception as e:
            span.set_attribute("success", False)
            span.set_attribute("error", str(e))
            
            logger.error(f"Document search failed (async) - query: {query[:100]}{'...' if len(query) > 100 else ''}, error: {e} ({type(e).__name__})")
            
            # Return empty list on error
            return []


def search_code_examples(
    client: Client, 
    query: str, 
    match_count: int = 10, 
    filter_metadata: Optional[Dict[str, Any]] = None,
    source_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Search for code examples in Supabase using vector similarity.
    
    Args:
        client: Supabase client
        query: Query text
        match_count: Maximum number of results to return
        filter_metadata: Optional metadata filter
        source_id: Optional source ID to filter results
        
    Returns:
        List of matching code examples
    """
    # Create a more descriptive query for better embedding match
    # Since code examples are embedded with their summaries, we should make the query more descriptive
    enhanced_query = f"Code example for {query}\n\nSummary: Example code showing {query}"
    
    # Create embedding for the enhanced query
    query_embedding = create_embedding(enhanced_query)
    
    # Execute the search using the match_archon_code_examples function
    try:
        # Build RPC params with dimension-specific embedding parameter
        params = build_rpc_params(
            query_embedding,
            match_count,
            filter_metadata,
            source_id
        )
        
        result = client.rpc('match_archon_code_examples', params).execute()
        
        return result.data
    except Exception as e:
        logger.error(f"Error searching code examples: {e}")
        return []