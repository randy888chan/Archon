"""
Search Services

Consolidated search and RAG functionality with strategy pattern support.
"""
# Main RAG service
from .rag_service import (
    RAGService,
    search_documents,
    search_documents_async,
    search_code_examples
)

# Strategy implementations
from .hybrid_search_strategy import HybridSearchStrategy
from .reranking_strategy import RerankingStrategy, create_reranking_strategy
from .agentic_rag_strategy import AgenticRAGStrategy

# Legacy compatibility
from .rag_service import RAGService as SearchService

__all__ = [
    # Main service classes
    'RAGService',
    'SearchService',  # Legacy compatibility
    
    # Strategy classes
    'HybridSearchStrategy',
    'RerankingStrategy', 
    'AgenticRAGStrategy',
    
    # Utility functions
    'search_documents',
    'search_documents_async',
    'search_code_examples',
    'create_reranking_strategy'
]