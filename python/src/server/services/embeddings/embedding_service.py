"""
Embedding Service

Handles all OpenAI embedding operations with proper rate limiting and error handling.
"""

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any

import openai

from ...config.logfire_config import safe_span, search_logger
from ..credential_service import credential_service
from ..llm_provider_service import get_embedding_model, get_llm_client
from ..threading_service import get_threading_service
from .embedding_exceptions import (
    EmbeddingAPIError,
    EmbeddingError,
    EmbeddingQuotaExhaustedError,
    EmbeddingRateLimitError,
)
from .dimension_validator import (
    validate_embedding_dimensions, log_dimension_operation, 
    ensure_valid_embedding, validate_batch_consistency
)
from .exceptions import (
    EmbeddingCreationError, UnsupportedDimensionError, 
    QuotaExhaustedError, RateLimitError, handle_dimension_error
)


@dataclass
class EmbeddingBatchResult:
    """Result of batch embedding creation with success/failure tracking."""

    embeddings: list[list[float]] = field(default_factory=list)
    failed_items: list[dict[str, Any]] = field(default_factory=list)
    success_count: int = 0
    failure_count: int = 0
    texts_processed: list[str] = field(default_factory=list)  # Successfully processed texts
    embedding_model: str | None = None  # The embedding model used
    embedding_dimensions: int | None = None  # The dimension size of the embeddings

    def add_success(self, embedding: list[float], text: str):
        """Add a successful embedding."""
        self.embeddings.append(embedding)
        self.texts_processed.append(text)
        self.success_count += 1

    def add_failure(self, text: str, error: Exception, batch_index: int | None = None):
        """Add a failed item with error details."""
        error_dict = {
            "text": text[:200] if text else None,
            "error": str(error),
            "error_type": type(error).__name__,
            "batch_index": batch_index,
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


def get_embedding_dimensions(model_name: str) -> int:
    """
    Get the number of dimensions for a given embedding model.
    
    Args:
        model_name: Name of the embedding model
        
    Returns:
        Number of dimensions for the model
    """
    # Import the new dimension service 
    from .embedding_dimension_service import embedding_dimension_service
    
    # Try to get dimensions from the new service first
    try:
        # Use the new dimension service which has comprehensive model support
        dimensions = embedding_dimension_service._get_known_model_dimension(model_name)
        if dimensions:
            return dimensions
    except Exception as e:
        search_logger.warning(f"Failed to get dimensions from dimension service: {e}")
    
    # Fallback to legacy hardcoded dimensions for backward compatibility
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
    # Ollama models - now handled by dimension service, but keep one for fallback
    elif 'snowflake-arctic-embed2' in model_name:
        return 1024  # Updated to correct dimension
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


async def create_embedding(text: str, provider: str | None = None) -> list[float]:
    """
    Create an embedding for a single text using the configured provider.

    Args:
        text: Text to create an embedding for
        provider: Optional provider override

    Returns:
        List of floats representing the embedding

    Raises:
        EmbeddingQuotaExhaustedError: When OpenAI quota is exhausted
        EmbeddingRateLimitError: When rate limited
        EmbeddingAPIError: For other API errors
    """
    try:
        result = await create_embeddings_batch([text], provider=provider)
        if not result.embeddings:
            # Check if there were failures
            if result.has_failures and result.failed_items:
                # Re-raise the original error for single embeddings
                error_info = result.failed_items[0]
                error_msg = error_info.get("error", "Unknown error")
                if "quota" in error_msg.lower():
                    raise EmbeddingQuotaExhaustedError(
                        f"OpenAI quota exhausted: {error_msg}", text_preview=text
                    )
                elif "rate" in error_msg.lower():
                    raise EmbeddingRateLimitError(f"Rate limit hit: {error_msg}", text_preview=text)
                else:
                    raise EmbeddingAPIError(
                        f"Failed to create embedding: {error_msg}", text_preview=text
                    )
            else:
                raise EmbeddingAPIError(
                    "No embeddings returned from batch creation", text_preview=text
                )
        return result.embeddings[0]
    except EmbeddingError:
        # Re-raise our custom exceptions
        raise
    except Exception as e:
        # Convert to appropriate exception type
        error_msg = str(e)
        search_logger.error(f"Embedding creation failed: {error_msg}", exc_info=True)
        search_logger.error(f"Failed text preview: {text[:100]}...")

        if "insufficient_quota" in error_msg:
            raise EmbeddingQuotaExhaustedError(
                f"OpenAI quota exhausted: {error_msg}", text_preview=text
            )
        elif "rate_limit" in error_msg.lower():
            raise EmbeddingRateLimitError(f"Rate limit hit: {error_msg}", text_preview=text)
        else:
            raise EmbeddingAPIError(
                f"Embedding error: {error_msg}", text_preview=text, original_error=e
            )

# Alias for backward compatibility with tests and other modules
create_embedding_async = create_embedding



async def create_embeddings_batch(
    texts: list[str],
    websocket: Any | None = None,
    progress_callback: Any | None = None,
    provider: str | None = None,
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
            search_logger.error(
                f"Invalid text type at index {i}: {type(text)}, value: {text}", exc_info=True
            )
            # Try to convert to string
            try:
                validated_texts.append(str(text))
            except Exception as e:
                search_logger.error(
                    f"Failed to convert text at index {i} to string: {e}", exc_info=True
                )
                validated_texts.append("")  # Use empty string as fallback
        else:
            validated_texts.append(text)

    texts = validated_texts

    result = EmbeddingBatchResult()
    threading_service = get_threading_service()

    with safe_span(
        "create_embeddings_batch", text_count=len(texts), total_chars=sum(len(t) for t in texts)
    ) as span:
        try:
            async with get_llm_client(provider=provider, use_embedding_provider=True) as client:
                # Load batch size and dimensions from settings
                try:
                    rag_settings = await credential_service.get_credentials_by_category(
                        "rag_strategy"
                    )
                    batch_size = int(rag_settings.get("EMBEDDING_BATCH_SIZE", "100"))
                    embedding_dimensions = int(rag_settings.get("EMBEDDING_DIMENSIONS", "1536"))
                except Exception as e:
                    search_logger.warning(f"Failed to load embedding settings: {e}, using defaults")
                    batch_size = 100
                    embedding_dimensions = 1536

                total_tokens_used = 0

                for i in range(0, len(texts), batch_size):
                    batch = texts[i : i + batch_size]
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
                                        dimensions=get_embedding_dimensions(embedding_model)
                                    )
                                    
                                    # Extract embeddings and validate consistency
                                    batch_embeddings = [item.embedding for item in response.data]
                                    
                                    # Validate embedding dimensions
                                    embedding_model_dims = get_embedding_dimensions(embedding_model)
                                    is_consistent, consistency_msg = validate_batch_consistency(batch_embeddings)
                                    
                                    if not is_consistent:
                                        search_logger.warning(f"Batch consistency validation failed: {consistency_msg}")
                                        log_dimension_operation("embedding_creation", embedding_model_dims, False, consistency_msg)
                                    else:
                                        log_dimension_operation("embedding_creation", embedding_model_dims, True)
                                    
                                    # Set model info on first successful batch (all batches use same model)
                                    if result.embedding_model is None:
                                        result.embedding_model = embedding_model
                                        result.embedding_dimensions = embedding_model_dims
                                    
                                    # Add successful embeddings
                                    for text, embedding in zip(batch, batch_embeddings):
                                        result.add_success(embedding, text)
                                    break  # Success, exit retry loop

                                except openai.RateLimitError as e:
                                    error_message = str(e)
                                    if "insufficient_quota" in error_message:
                                        # Quota exhausted is critical - stop everything
                                        tokens_so_far = total_tokens_used - batch_tokens
                                        cost_so_far = (tokens_so_far / 1_000_000) * 0.02

                                        search_logger.error(
                                            f"⚠️ QUOTA EXHAUSTED at batch {batch_index}! "
                                            f"Processed {result.success_count} texts successfully.",
                                            exc_info=True,
                                        )

                                        # Add remaining texts as failures
                                        for text in texts[i:]:
                                            result.add_failure(
                                                text,
                                                EmbeddingQuotaExhaustedError(
                                                    "OpenAI quota exhausted",
                                                    tokens_used=tokens_so_far,
                                                ),
                                                batch_index,
                                            )

                                        # Return what we have so far
                                        span.set_attribute("quota_exhausted", True)
                                        span.set_attribute("partial_success", True)
                                        return result

                                    else:
                                        # Regular rate limit - retry
                                        retry_count += 1
                                        if retry_count < max_retries:
                                            wait_time = 2**retry_count
                                            search_logger.warning(
                                                f"Rate limit hit for batch {batch_index}, "
                                                f"waiting {wait_time}s before retry {retry_count}/{max_retries}"
                                            )
                                            await asyncio.sleep(wait_time)
                                        else:
                                            raise  # Will be caught by outer try

                    except Exception as e:
                        # This batch failed - track failures but continue with next batch
                        search_logger.error(f"Batch {batch_index} failed: {e}", exc_info=True)

                        for text in batch:
                            if isinstance(e, EmbeddingError):
                                result.add_failure(text, e, batch_index)
                            else:
                                result.add_failure(
                                    text,
                                    EmbeddingAPIError(
                                        f"Failed to create embedding: {str(e)}", original_error=e
                                    ),
                                    batch_index,
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
                        processed = result.success_count + result.failure_count
                        ws_progress = (processed / len(texts)) * 100
                        await websocket.send_json({
                            "type": "embedding_progress",
                            "processed": processed,
                            "successful": result.success_count,
                            "failed": result.failure_count,
                            "total": len(texts),
                            "percentage": ws_progress,
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
            search_logger.error(f"Catastrophic failure in batch embedding: {e}", exc_info=True)

            # Mark remaining texts as failed
            processed_count = result.success_count + result.failure_count
            for text in texts[processed_count:]:
                result.add_failure(
                    text, EmbeddingAPIError(f"Catastrophic failure: {str(e)}", original_error=e)
                )

            return result


# Deprecated functions - kept for backward compatibility
async def get_openai_api_key() -> str | None:
    """
    DEPRECATED: Use os.getenv("OPENAI_API_KEY") directly.
    API key is loaded into environment at startup.
    """
    return os.getenv("OPENAI_API_KEY")
