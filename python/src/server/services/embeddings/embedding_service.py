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
from .embedding_exceptions import (
    EmbeddingError,
    EmbeddingQuotaExhaustedError,
    EmbeddingRateLimitError,
    EmbeddingAsyncContextError,
    EmbeddingAPIError,
    EmbeddingValidationError
)


# Use the new provider-aware client factory
get_openai_client = get_llm_client


def create_embedding(text: str, provider: Optional[str] = None) -> List[float]:
    """
    Create an embedding for a single text using the configured provider.
    
    This is a synchronous wrapper around the async version for backward compatibility.
    
    Args:
        text: Text to create an embedding for
        provider: Optional provider override
        
    Returns:
        List of floats representing the embedding
    """
    try:
        # Check if we're in an async context
        try:
            loop = asyncio.get_running_loop()
            # If we're already in an async context, we can't run sync - must use async version
            raise EmbeddingAsyncContextError(
                "create_embedding (sync) called from async context - use create_embedding_async instead",
                text_preview=text
            )
        except RuntimeError:
            # No running loop, safe to use asyncio.run
            return asyncio.run(create_embedding_async(text, provider=provider))
    except EmbeddingError:
        # Re-raise our custom exceptions as-is
        raise
    except Exception as e:
        # Convert other exceptions to EmbeddingAPIError
        error_msg = str(e)
        search_logger.error(f"Embedding creation failed: {error_msg}")
        search_logger.error(f"Failed text preview: {text[:100]}...")
        
        # Check for specific error types
        if "insufficient_quota" in error_msg:
            raise EmbeddingQuotaExhaustedError(
                f"OpenAI quota exhausted: {error_msg}",
                text_preview=text
            )
        elif "rate_limit" in error_msg.lower():
            raise EmbeddingRateLimitError(
                f"Rate limit hit: {error_msg}",
                text_preview=text
            )
        else:
            raise EmbeddingAPIError(
                f"Unexpected embedding error: {error_msg}",
                text_preview=text,
                original_error=e
            )


async def create_embedding_async(text: str, provider: Optional[str] = None) -> List[float]:
    """
    Create an embedding for a single text using the configured provider.
    
    Args:
        text: Text to create an embedding for
        provider: Optional provider override
        
    Returns:
        List of floats representing the embedding
    """
    try:
        embeddings = await create_embeddings_batch_async([text], provider=provider)
        if not embeddings:
            raise EmbeddingAPIError(
                "No embeddings returned from batch creation",
                text_preview=text
            )
        return embeddings[0]
    except EmbeddingError:
        # Re-raise our custom exceptions
        raise
    except Exception as e:
        # Convert to appropriate exception type
        error_msg = str(e)
        search_logger.error(f"Async embedding creation failed: {error_msg}")
        search_logger.error(f"Failed text preview: {text[:100]}...")
        
        if "insufficient_quota" in error_msg:
            raise EmbeddingQuotaExhaustedError(
                f"OpenAI quota exhausted: {error_msg}",
                text_preview=text
            )
        elif "rate_limit" in error_msg.lower():
            raise EmbeddingRateLimitError(
                f"Rate limit hit: {error_msg}",
                text_preview=text
            )
        else:
            raise EmbeddingAPIError(
                f"Async embedding error: {error_msg}",
                text_preview=text,
                original_error=e
            )


def create_embeddings_batch(texts: List[str], provider: Optional[str] = None) -> List[List[float]]:
    """
    Create embeddings for multiple texts in a single API call.
    
    This is a synchronous wrapper around the async version for backward compatibility.
    
    Args:
        texts: List of texts to create embeddings for
        provider: Optional provider override
        
    Returns:
        List of embeddings (each embedding is a list of floats)
    """
    if not texts:
        return []
    
    try:
        # Check if we're in an async context
        try:
            loop = asyncio.get_running_loop()
            # If we're already in an async context, we can't run sync - must use async version
            raise EmbeddingAsyncContextError(
                f"create_embeddings_batch (sync) called from async context - use create_embeddings_batch_async instead. Batch size: {len(texts)}",
                text_preview=texts[0] if texts else None
            )
        except RuntimeError:
            # No running loop, safe to use asyncio.run
            return asyncio.run(create_embeddings_batch_async(texts, provider=provider))
    except EmbeddingError:
        # Re-raise our custom exceptions
        raise
    except Exception as e:
        # Convert to appropriate exception type
        error_msg = str(e)
        search_logger.error(f"Batch embedding creation failed: {error_msg}")
        search_logger.error(f"Batch size: {len(texts)}, first text preview: {texts[0][:100] if texts else 'empty'}...")
        
        if "insufficient_quota" in error_msg:
            raise EmbeddingQuotaExhaustedError(
                f"OpenAI quota exhausted during batch processing: {error_msg}",
                text_preview=texts[0] if texts else None
            )
        elif "rate_limit" in error_msg.lower():
            raise EmbeddingRateLimitError(
                f"Rate limit hit during batch processing: {error_msg}",
                text_preview=texts[0] if texts else None
            )
        else:
            raise EmbeddingAPIError(
                f"Batch embedding error: {error_msg}",
                text_preview=texts[0] if texts else None,
                original_error=e,
                batch_size=len(texts)
            )


async def create_embeddings_batch_async(
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
                                    dimensions=1536
                                )
                                
                                batch_embeddings = [item.embedding for item in response.data]
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
                                    
                                    # Notify via progress callback before raising
                                    if progress_callback:
                                        await progress_callback(
                                            f"❌ QUOTA EXHAUSTED - Add credits to OpenAI account! (used {tokens_so_far:,} tokens ≈${cost_so_far:.4f})",
                                            100
                                        )
                                    
                                    # Raise exception to stop process - this is critical
                                    raise EmbeddingQuotaExhaustedError(
                                        f"OpenAI billing quota exhausted after {tokens_so_far:,} tokens (≈${cost_so_far:.4f}). Add credits to continue.",
                                        tokens_used=tokens_so_far,
                                        text_preview=batch[0] if batch else None,
                                        batch_index=i // batch_size
                                    )
                                else:
                                    retry_count += 1
                                    if retry_count < max_retries:
                                        wait_time = 2 ** retry_count  # Exponential backoff
                                        search_logger.warning(f"Rate limit hit (not quota), waiting {wait_time}s before retry {retry_count}/{max_retries}")
                                        await asyncio.sleep(wait_time)
                                    else:
                                        # Max retries exceeded - raise error for this batch
                                        raise EmbeddingRateLimitError(
                                            f"Max retries ({max_retries}) exceeded for batch {i//batch_size + 1} after rate limiting",
                                            retry_count=retry_count,
                                            text_preview=batch[0] if batch else None,
                                            batch_index=i // batch_size
                                        )
                        
                        # Progress reporting with cost estimation
                        if progress_callback:
                            progress = ((i + len(batch)) / len(texts)) * 100
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
                    
        except EmbeddingError:
            # Re-raise our custom exceptions with span info
            span.set_attribute("success", False)
            raise
        except Exception as e:
            span.set_attribute("success", False)
            span.set_attribute("error", str(e))
            search_logger.error(f"Failed to create embeddings batch: {e}")
            
            # Raise as API error
            raise EmbeddingAPIError(
                f"Failed to create embeddings batch: {str(e)}",
                original_error=e,
                batch_size=len(texts)
            )


# Deprecated functions - kept for backward compatibility
async def get_openai_api_key() -> Optional[str]:
    """
    DEPRECATED: Use os.getenv("OPENAI_API_KEY") directly.
    API key is loaded into environment at startup.
    """
    return os.getenv("OPENAI_API_KEY")


def get_openai_api_key_sync() -> Optional[str]:
    """
    DEPRECATED: Use os.getenv("OPENAI_API_KEY") directly.
    API key is loaded into environment at startup.
    """
    return os.getenv("OPENAI_API_KEY")