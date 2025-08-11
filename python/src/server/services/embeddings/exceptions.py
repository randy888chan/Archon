"""
Multi-dimensional Vector System Exception Classes

Provides comprehensive exception handling for all vector operations.
"""

class VectorSystemError(Exception):
    """Base exception for all vector system errors."""
    
    def __init__(self, message: str, dimensions: int = None, operation: str = None, context: dict = None):
        self.dimensions = dimensions
        self.operation = operation
        self.context = context or {}
        
        super().__init__(message)
    
    def __str__(self):
        parts = [super().__str__()]
        
        if self.dimensions:
            parts.append(f"Dimensions: {self.dimensions}")
        
        if self.operation:
            parts.append(f"Operation: {self.operation}")
            
        if self.context:
            parts.append(f"Context: {self.context}")
        
        return " | ".join(parts)


class DimensionValidationError(VectorSystemError):
    """Raised when dimension validation fails."""
    pass


class UnsupportedDimensionError(DimensionValidationError):
    """Raised when an unsupported dimension is encountered."""
    
    def __init__(self, dimensions: int, supported_dims: set = None, **kwargs):
        supported_dims = supported_dims or {768, 1024, 1536, 3072}
        message = f"Dimension {dimensions} is not supported. Supported dimensions: {sorted(supported_dims)}"
        super().__init__(message, dimensions=dimensions, **kwargs)


class DimensionMismatchError(DimensionValidationError):
    """Raised when embedding dimensions don't match expected dimensions."""
    
    def __init__(self, expected: int, actual: int, **kwargs):
        message = f"Dimension mismatch: expected {expected}, got {actual}"
        super().__init__(message, dimensions=actual, context={'expected': expected, 'actual': actual}, **kwargs)


class InvalidColumnNameError(DimensionValidationError):
    """Raised when column name generation fails validation."""
    
    def __init__(self, column_name: str, reason: str = None, **kwargs):
        message = f"Invalid column name: {column_name}"
        if reason:
            message += f" - {reason}"
        super().__init__(message, context={'column_name': column_name, 'reason': reason}, **kwargs)


class EmbeddingCreationError(VectorSystemError):
    """Raised when embedding creation fails."""
    
    def __init__(self, model: str = None, provider: str = None, **kwargs):
        message = "Failed to create embedding"
        if model:
            message += f" using model {model}"
        if provider:
            message += f" with provider {provider}"
        super().__init__(message, context={'model': model, 'provider': provider}, **kwargs)


class VectorStorageError(VectorSystemError):
    """Raised when vector storage operations fail."""
    
    def __init__(self, table: str = None, batch_size: int = None, **kwargs):
        message = "Vector storage operation failed"
        if table:
            message += f" for table {table}"
        super().__init__(message, context={'table': table, 'batch_size': batch_size}, **kwargs)


class VectorSearchError(VectorSystemError):
    """Raised when vector search operations fail."""
    
    def __init__(self, rpc_function: str = None, match_count: int = None, **kwargs):
        message = "Vector search operation failed"
        if rpc_function:
            message += f" calling {rpc_function}"
        super().__init__(message, context={'rpc_function': rpc_function, 'match_count': match_count}, **kwargs)


class ModelCompatibilityError(VectorSystemError):
    """Raised when model and dimension compatibility issues occur."""
    
    def __init__(self, model: str, dimensions: int, **kwargs):
        message = f"Model {model} is not compatible with {dimensions} dimensions"
        super().__init__(message, dimensions=dimensions, context={'model': model}, **kwargs)


class QuotaExhaustedError(VectorSystemError):
    """Raised when API quota is exhausted."""
    
    def __init__(self, provider: str, tokens_used: int = None, **kwargs):
        message = f"Quota exhausted for provider {provider}"
        if tokens_used:
            message += f" after {tokens_used:,} tokens"
        super().__init__(message, context={'provider': provider, 'tokens_used': tokens_used}, **kwargs)


class RateLimitError(VectorSystemError):
    """Raised when rate limits are hit."""
    
    def __init__(self, provider: str, retry_after: int = None, **kwargs):
        message = f"Rate limit hit for provider {provider}"
        if retry_after:
            message += f", retry after {retry_after} seconds"
        super().__init__(message, context={'provider': provider, 'retry_after': retry_after}, **kwargs)


class BatchProcessingError(VectorSystemError):
    """Raised when batch processing fails."""
    
    def __init__(self, batch_num: int = None, total_batches: int = None, successful_items: int = None, **kwargs):
        message = "Batch processing failed"
        if batch_num and total_batches:
            message += f" for batch {batch_num}/{total_batches}"
        super().__init__(
            message, 
            context={
                'batch_num': batch_num, 
                'total_batches': total_batches,
                'successful_items': successful_items
            }, 
            **kwargs
        )


class DatabaseConnectionError(VectorSystemError):
    """Raised when database connection issues occur."""
    
    def __init__(self, database: str = None, **kwargs):
        message = "Database connection failed"
        if database:
            message += f" for {database}"
        super().__init__(message, context={'database': database}, **kwargs)


class SecurityValidationError(VectorSystemError):
    """Raised when security validation fails."""
    
    def __init__(self, validation_type: str, **kwargs):
        message = f"Security validation failed: {validation_type}"
        super().__init__(message, context={'validation_type': validation_type}, **kwargs)


# Utility functions for error handling
def handle_dimension_error(error: Exception, operation: str, fallback_dims: int = 1536) -> tuple:
    """
    Handle dimension-related errors and return appropriate fallback.
    
    Args:
        error: The original exception
        operation: The operation that failed
        fallback_dims: Fallback dimensions to use
        
    Returns:
        Tuple of (success: bool, result: any, error_message: str)
    """
    if isinstance(error, UnsupportedDimensionError):
        return False, [0.0] * fallback_dims, f"Unsupported dimension in {operation}, using fallback"
    
    elif isinstance(error, DimensionMismatchError):
        return False, [0.0] * fallback_dims, f"Dimension mismatch in {operation}, using fallback"
    
    elif isinstance(error, EmbeddingCreationError):
        return False, [0.0] * fallback_dims, f"Embedding creation failed in {operation}, using fallback"
    
    elif isinstance(error, QuotaExhaustedError):
        return False, [0.0] * fallback_dims, f"Quota exhausted in {operation}, using fallback"
    
    else:
        return False, [0.0] * fallback_dims, f"Unknown error in {operation}: {str(error)}"


def create_detailed_error_context(operation: str, 
                                dimensions: int = None,
                                model: str = None,
                                batch_size: int = None,
                                provider: str = None) -> dict:
    """Create detailed error context for exception reporting."""
    context = {
        'operation': operation,
        'timestamp': None,  # Will be filled by logging system
        'supported_dimensions': [768, 1024, 1536, 3072]
    }
    
    if dimensions:
        context['dimensions'] = dimensions
        context['dimension_supported'] = dimensions in {768, 1024, 1536, 3072}
    
    if model:
        context['model'] = model
        
    if batch_size:
        context['batch_size'] = batch_size
        
    if provider:
        context['provider'] = provider
    
    return context


def is_recoverable_error(error: Exception) -> bool:
    """
    Determine if an error is recoverable with retry or fallback.
    
    Args:
        error: Exception to check
        
    Returns:
        True if error is recoverable
    """
    recoverable_errors = (
        RateLimitError,
        DatabaseConnectionError,
        BatchProcessingError,
        DimensionMismatchError,
        UnsupportedDimensionError
    )
    
    # QuotaExhaustedError is not recoverable without user action
    non_recoverable_errors = (
        QuotaExhaustedError,
        SecurityValidationError
    )
    
    if isinstance(error, non_recoverable_errors):
        return False
        
    if isinstance(error, recoverable_errors):
        return True
    
    # For unknown errors, assume they might be recoverable
    return True