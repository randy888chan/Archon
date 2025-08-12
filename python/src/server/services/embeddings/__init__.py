"""
Embedding Services

Handles all embedding-related operations.
"""

from .contextual_embedding_service import (
    generate_contextual_embedding,
    generate_contextual_embeddings_batch,
    process_chunk_with_context,
)
from .embedding_service import create_embedding, create_embeddings_batch, get_openai_client

from .embedding_dimension_service import (
    embedding_dimension_service,
    RECOMMENDED_MODELS
)

__all__ = [
    # Embedding functions
    "create_embedding",
    "create_embeddings_batch",
    "get_openai_client",
    # Contextual embedding functions
    "generate_contextual_embedding",
    "generate_contextual_embeddings_batch",
    "process_chunk_with_context",
    
    # Dimension service
    "embedding_dimension_service",
    "RECOMMENDED_MODELS"
]
