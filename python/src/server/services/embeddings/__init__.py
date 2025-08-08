"""
Embedding Services

Handles all embedding-related operations.
"""
from .embedding_service import (
    create_embedding,
    create_embeddings_batch,
    get_openai_client,
    get_openai_api_key
)

from .contextual_embedding_service import (
    generate_contextual_embedding,
    generate_contextual_embeddings_batch,
    process_chunk_with_context
)

__all__ = [
    # Embedding functions (now async-only)
    'create_embedding',
    'create_embeddings_batch',
    'get_openai_client',
    
    # Deprecated function (kept for compatibility)
    'get_openai_api_key',
    
    # Contextual embedding functions (now async-only)
    'generate_contextual_embedding',
    'generate_contextual_embeddings_batch',
    'process_chunk_with_context'
]