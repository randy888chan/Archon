"""
Embedding Service

Handles all OpenAI embedding operations with proper rate limiting and error handling.
"""
import os
import asyncio
from typing import List, Optional, Any
import openai

from ...config.logfire_config import search_logger, safe_span
from ..threading_service import get_threading_service
from ..llm_provider_service import get_llm_client, get_embedding_model
from ..credential_service import credential_service
from .dimension_validator import (
    validate_embedding_dimensions, log_dimension_operation, 
    ensure_valid_embedding, validate_batch_consistency
)
from .exceptions import (
    EmbeddingCreationError, UnsupportedDimensionError, 
    QuotaExhaustedError, RateLimitError, handle_dimension_error
)


def get_embedding_dimensions(model_name: str) -> int:
    """
    Get the number of dimensions for a given embedding model.
    
    Args:
        model_name: Name of the embedding model
        
    Returns:
        Number of dimensions for the model
    """
    # OpenAI models
    if model_name in ['text-embedding-3-large']:
        return 3072
    elif model_name in ['text-embedding-3-small', 'text-embedding-ada-002']:
        return 1536
    # Sentence-transformers models (common dimensions)
    elif 'all-MiniLM-L6-v2' in model_name:
        return 384
    elif 'all-mpnet-base-v2' in model_name:
        return 768
    elif 'all-MiniLM-L12-v2' in model_name:
        return 384
    # Default fallback
    else:
        search_logger.warning(f"Unknown model dimensions for {model_name}, defaulting to 1536")
        return 1536


def get_dimension_column_name(dimensions: int) -> str:
    """
    Get the appropriate database column name for given dimensions.
    
    Args:
        dimensions: Number of embedding dimensions
        
    Returns:
        Column name to use for storage
    """
    if dimensions == 768:
        return "embedding_768"
    elif dimensions == 1024:
        return "embedding_1024"
    elif dimensions == 1536:
        return "embedding_1536"
    elif dimensions == 3072:
        return "embedding_3072"
    else:
        search_logger.warning(f"Unsupported dimensions {dimensions}, defaulting to embedding_1536")
        return "embedding_1536"


# Provider-aware client factory
get_openai_client = get_llm_client


async def create_embedding(text: str, provider: Optional[str] = None) -> List[float]:
    """
    Create an embedding for a single text using the configured provider.
    
    Args:
        text: Text to create an embedding for
        provider: Optional provider override
        
    Returns:
        List of floats representing the embedding
    """
    try:
        embeddings = await create_embeddings_batch([text], provider=provider)
        return embeddings[0] if embeddings else [0.0] * 1536
    except Exception as e:
        # Enhanced logging for zero embedding fallback
        search_logger.warning(f"Async embedding creation failed, using zero fallback: {str(e)}")
        search_logger.warning(f"Failed text preview: {text[:100]}...")
        
        # Track failure metrics
        if "insufficient_quota" in str(e):
            search_logger.error("OpenAI quota exhausted - zero embeddings returned")
        elif "rate_limit" in str(e).lower():
            search_logger.warning("Rate limit hit - zero embeddings returned")
        else:
            search_logger.error(f"Unexpected embedding error: {type(e).__name__}")
        
        return [0.0] * 1536



async def create_embeddings_batch(
    texts: List[str], 
    websocket: Optional[Any] = None,
    progress_callback: Optional[Any] = None,
    provider: Optional[str] = None
) -> List[List[float]]:
    """
    Create embeddings for multiple texts with threading optimizations.
    
    Args:
        texts: List of texts to create embeddings for
        websocket: Optional WebSocket for progress updates
        progress_callback: Optional callback for progress reporting
        provider: Optional provider override
        
    Returns:
        List of embeddings (each embedding is a list of floats)
    """
    if not texts:
        return []
    
    # Validate that all items in texts are strings
    validated_texts = []
    for i, text in enumerate(texts):
        if not isinstance(text, str):
            search_logger.error(f"Invalid text type at index {i}: {type(text)}, value: {text}")
            # Try to convert to string
            try:
                validated_texts.append(str(text))
            except Exception as e:
                search_logger.error(f"Failed to convert text at index {i} to string: {e}")
                validated_texts.append("")  # Use empty string as fallback
        else:
            validated_texts.append(text)
    
    texts = validated_texts
    
    threading_service = get_threading_service()
    
    with safe_span("create_embeddings_batch_async", 
                           text_count=len(texts),
                           total_chars=sum(len(t) for t in texts)) as span:
        
        try:
            async with get_llm_client(provider=provider, use_embedding_provider=True) as client:
                # Load batch size from settings
                try:
                    rag_settings = await credential_service.get_credentials_by_category("rag_strategy")
                    batch_size = int(rag_settings.get("EMBEDDING_BATCH_SIZE", "100"))
                except Exception as e:
                    search_logger.warning(f"Failed to load embedding batch size: {e}, using default")
                    batch_size = 100
                
                all_embeddings = []
                total_tokens_used = 0
                
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    
                    # Estimate tokens for this specific batch
                    batch_tokens = sum(len(text.split()) for text in batch) * 1.3  # Rough estimate
                    total_tokens_used += batch_tokens
                    
                    # Rate limit each batch individually
                    async with threading_service.rate_limited_operation(batch_tokens):
                        retry_count = 0
                        max_retries = 3
                        
                        while retry_count < max_retries:
                            try:
                                # Create embeddings for this batch
                                embedding_model = await get_embedding_model(provider=provider)
                                response = await client.embeddings.create(
                                    model=embedding_model,
                                    input=batch,
                                    dimensions=get_embedding_dimensions(embedding_model)
                                )
                                
                                batch_embeddings = [item.embedding for item in response.data]
                                
                                # Validate embedding dimensions
                                embedding_model_dims = get_embedding_dimensions(embedding_model)
                                is_consistent, consistency_msg = validate_batch_consistency(batch_embeddings)
                                
                                if not is_consistent:
                                    search_logger.warning(f"Batch consistency validation failed: {consistency_msg}")
                                    log_dimension_operation("embedding_creation", embedding_model_dims, False, consistency_msg)
                                    # Use fallback embeddings for consistency
                                    batch_embeddings = [[0.0] * embedding_model_dims for _ in batch]
                                else:
                                    log_dimension_operation("embedding_creation", embedding_model_dims, True)
                                
                                all_embeddings.extend(batch_embeddings)
                                break  # Success, exit retry loop
                                
                            except openai.RateLimitError as e:
                                error_message = str(e)
                                if "insufficient_quota" in error_message:
                                    # Calculate approximate cost (using OpenAI pricing as estimate)
                                    tokens_so_far = total_tokens_used - batch_tokens
                                    cost_so_far = (tokens_so_far / 1_000_000) * 0.02  # Estimated cost
                                    
                                    search_logger.error(
                                        f"⚠️ OpenAI BILLING QUOTA EXHAUSTED! You need to add more credits to your OpenAI account.\n"
                                        f"Tokens used so far: {tokens_so_far:,} (≈${cost_so_far:.4f})\n"
                                        f"Error: {error_message}"
                                    )
                                    span.set_attribute("quota_exhausted", True)
                                    span.set_attribute("tokens_used_before_quota", tokens_so_far)
                                    
                                    # Return zero embeddings for remaining texts
                                    remaining = len(texts) - len(all_embeddings)
                                    all_embeddings.extend([[0.0] * 1536 for _ in range(remaining)])
                                    
                                    # Notify via progress callback
                                    if progress_callback:
                                        await progress_callback(
                                            f"❌ QUOTA EXHAUSTED - Add credits to OpenAI account! (used {tokens_so_far:,} tokens ≈${cost_so_far:.4f})",
                                            100
                                        )
                                    
                                    return all_embeddings
                                else:
                                    retry_count += 1
                                    if retry_count < max_retries:
                                        wait_time = 2 ** retry_count  # Exponential backoff
                                        search_logger.warning(f"Rate limit hit (not quota), waiting {wait_time}s before retry {retry_count}/{max_retries}")
                                        await asyncio.sleep(wait_time)
                                    else:
                                        search_logger.error(f"Max retries exceeded for batch {i//batch_size + 1}")
                                        # Add zero embeddings for this batch
                                        all_embeddings.extend([[0.0] * 1536 for _ in batch])
                        
                        # Progress reporting with cost estimation
                        progress = ((i + len(batch)) / len(texts)) * 100
                        if progress_callback:
                            cost_estimate = (total_tokens_used / 1_000_000) * 0.02  # Estimated cost
                            await progress_callback(
                                f"Created embeddings for {i + len(batch)}/{len(texts)} texts (tokens: {total_tokens_used:,} ≈${cost_estimate:.4f})",
                                progress
                            )
                        
                        # WebSocket progress update
                        if websocket:
                            await websocket.send_json({
                                "type": "embedding_progress",
                                "processed": i + len(batch),
                                "total": len(texts),
                                "percentage": progress,
                                "tokens_used": total_tokens_used
                            })
                        
                        # Yield control for WebSocket health
                        await asyncio.sleep(0.1)
                
                span.set_attribute("embeddings_created", len(all_embeddings))
                span.set_attribute("success", True)
                span.set_attribute("total_tokens_used", total_tokens_used)
                
                return all_embeddings
                    
        except Exception as e:
            span.set_attribute("success", False)
            span.set_attribute("error", str(e))
            search_logger.error(f"Failed to create embeddings batch: {e}")
            
            # Return zero embeddings as fallback
            return [[0.0] * 1536 for _ in texts]
