"""
Embedding Service

Handles all OpenAI embedding operations with proper rate limiting and error handling.
"""
import os
import asyncio
from typing import List, Optional, Any, Dict
from dataclasses import dataclass, field
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


@dataclass
class EmbeddingBatchResult:
    """Result of batch embedding creation with success/failure tracking."""
    embeddings: List[List[float]] = field(default_factory=list)
    failed_items: List[Dict[str, Any]] = field(default_factory=list)
    success_count: int = 0
    failure_count: int = 0
    texts_processed: List[str] = field(default_factory=list)  # Successfully processed texts
    
    def add_success(self, embedding: List[float], text: str):
        """Add a successful embedding."""
        self.embeddings.append(embedding)
        self.texts_processed.append(text)
        self.success_count += 1
    
    def add_failure(self, text: str, error: Exception, batch_index: Optional[int] = None):
        """Add a failed item with error details."""
        error_dict = {
            "text": text[:200] if text else None,
            "error": str(error),
            "error_type": type(error).__name__,
            "batch_index": batch_index
        }
        
        # Add extra context from EmbeddingError if available
        if isinstance(error, EmbeddingError):
            error_dict.update(error.to_dict())
        
        self.failed_items.append(error_dict)
        self.failure_count += 1
    
    @property
    def has_failures(self) -> bool:
        return self.failure_count > 0
    
    @property
    def total_requested(self) -> int:
        return self.success_count + self.failure_count


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
        result = await create_embeddings_batch_async([text], provider=provider)
        if not result.embeddings:
            # Check if there were failures
            if result.has_failures and result.failed_items:
                # Re-raise the original error for single embeddings
                error_info = result.failed_items[0]
                error_msg = error_info.get('error', 'Unknown error')
                if 'quota' in error_msg.lower():
                    raise EmbeddingQuotaExhaustedError(
                        f"OpenAI quota exhausted: {error_msg}",
                        text_preview=text
                    )
                elif 'rate' in error_msg.lower():
                    raise EmbeddingRateLimitError(
                        f"Rate limit hit: {error_msg}",
                        text_preview=text
                    )
                else:
                    raise EmbeddingAPIError(
                        f"Failed to create embedding: {error_msg}",
                        text_preview=text
                    )
            else:
                raise EmbeddingAPIError(
                    "No embeddings returned from batch creation",
                    text_preview=text
                )
        return result.embeddings[0]
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


def create_embeddings_batch(texts: List[str], provider: Optional[str] = None) -> EmbeddingBatchResult:
    """
    Create embeddings for multiple texts with graceful failure handling.
    
    This is a synchronous wrapper around the async version for backward compatibility.
    
    Args:
        texts: List of texts to create embeddings for
        provider: Optional provider override
        
    Returns:
        EmbeddingBatchResult with successful embeddings and failure details
    """
    if not texts:
        return EmbeddingBatchResult()
    
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
) -> EmbeddingBatchResult:
    """
    Create embeddings for multiple texts with graceful failure handling.
    
    This function processes texts in batches and returns a structured result
    containing both successful embeddings and failed items. It follows the
    "skip, don't corrupt" principle - failed items are tracked but not stored
    with zero embeddings.
    
    Args:
        texts: List of texts to create embeddings for
        websocket: Optional WebSocket for progress updates
        progress_callback: Optional callback for progress reporting
        provider: Optional provider override
        
    Returns:
        EmbeddingBatchResult with successful embeddings and failure details
    """
    if not texts:
        return EmbeddingBatchResult()
    
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
    
    result = EmbeddingBatchResult()
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
                
                total_tokens_used = 0
                
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    batch_index = i // batch_size
                    
                    try:
                        # Estimate tokens for this batch
                        batch_tokens = sum(len(text.split()) for text in batch) * 1.3
                        total_tokens_used += batch_tokens
                        
                        # Rate limit each batch
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
                                    
                                    # Add successful embeddings
                                    for text, item in zip(batch, response.data):
                                        result.add_success(item.embedding, text)
                                    
                                    break  # Success, exit retry loop
                                    
                                except openai.RateLimitError as e:
                                    error_message = str(e)
                                    if "insufficient_quota" in error_message:
                                        # Quota exhausted is critical - stop everything
                                        tokens_so_far = total_tokens_used - batch_tokens
                                        cost_so_far = (tokens_so_far / 1_000_000) * 0.02
                                        
                                        search_logger.error(
                                            f"⚠️ QUOTA EXHAUSTED at batch {batch_index}! "
                                            f"Processed {result.success_count} texts successfully."
                                        )
                                        
                                        # Add remaining texts as failures
                                        for text in texts[i:]:
                                            result.add_failure(
                                                text,
                                                EmbeddingQuotaExhaustedError(
                                                    "OpenAI quota exhausted",
                                                    tokens_used=tokens_so_far
                                                ),
                                                batch_index
                                            )
                                        
                                        # Return what we have so far
                                        span.set_attribute("quota_exhausted", True)
                                        span.set_attribute("partial_success", True)
                                        return result
                                        
                                    else:
                                        # Regular rate limit - retry
                                        retry_count += 1
                                        if retry_count < max_retries:
                                            wait_time = 2 ** retry_count
                                            search_logger.warning(
                                                f"Rate limit hit for batch {batch_index}, "
                                                f"waiting {wait_time}s before retry {retry_count}/{max_retries}"
                                            )
                                            await asyncio.sleep(wait_time)
                                        else:
                                            raise  # Will be caught by outer try
                        
                                            
                    except Exception as e:
                        # This batch failed - track failures but continue with next batch
                        search_logger.error(f"Batch {batch_index} failed: {e}")
                        
                        for text in batch:
                            if isinstance(e, EmbeddingError):
                                result.add_failure(text, e, batch_index)
                            else:
                                result.add_failure(
                                    text,
                                    EmbeddingAPIError(
                                        f"Failed to create embedding: {str(e)}",
                                        original_error=e
                                    ),
                                    batch_index
                                )
                    
                    # Progress reporting
                    if progress_callback:
                        processed = result.success_count + result.failure_count
                        progress = (processed / len(texts)) * 100
                        
                        message = f"Processed {processed}/{len(texts)} texts"
                        if result.has_failures:
                            message += f" ({result.failure_count} failed)"
                        
                        await progress_callback(message, progress)
                    
                    # WebSocket update
                    if websocket:
                        await websocket.send_json({
                            "type": "embedding_progress",
                            "processed": result.success_count + result.failure_count,
                            "successful": result.success_count,
                            "failed": result.failure_count,
                            "total": len(texts),
                            "percentage": progress
                        })
                    
                    # Yield control
                    await asyncio.sleep(0.01)
                
                span.set_attribute("embeddings_created", result.success_count)
                span.set_attribute("embeddings_failed", result.failure_count)
                span.set_attribute("success", not result.has_failures)
                span.set_attribute("total_tokens_used", total_tokens_used)
                
                return result
                    
        except Exception as e:
            # Catastrophic failure - return what we have
            span.set_attribute("catastrophic_failure", True)
            search_logger.error(f"Catastrophic failure in batch embedding: {e}")
            
            # Mark remaining texts as failed
            processed_count = result.success_count + result.failure_count
            for text in texts[processed_count:]:
                result.add_failure(
                    text,
                    EmbeddingAPIError(f"Catastrophic failure: {str(e)}", original_error=e)
                )
            
            return result



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