"""
Dimension Validator Module

Provides comprehensive validation and error handling for multi-dimensional vector operations.
Ensures data integrity, security, and graceful error handling across the vector system.
"""
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from functools import wraps
import re

from ...config.logfire_config import search_logger

# Supported dimensions for the multi-dimensional vector system
SUPPORTED_DIMENSIONS = {768, 1024, 1536, 3072}
DEFAULT_DIMENSION = 1536
MAX_DIMENSION = 3072
MIN_DIMENSION = 384


class DimensionValidationError(Exception):
    """Raised when dimension validation fails."""
    pass


class UnsupportedDimensionError(DimensionValidationError):
    """Raised when an unsupported dimension is encountered."""
    pass


class DimensionMismatchError(DimensionValidationError):
    """Raised when embedding dimensions don't match expected dimensions."""
    pass


class InvalidColumnNameError(DimensionValidationError):
    """Raised when column name generation fails validation."""
    pass


def validate_embedding_dimensions(embedding: Optional[List[float]], 
                                expected_dims: Optional[int] = None,
                                allow_fallback: bool = True) -> Tuple[bool, int, str]:
    """
    Validate embedding vector dimensions and return validation result.
    
    Args:
        embedding: The embedding vector to validate
        expected_dims: Expected number of dimensions (optional)
        allow_fallback: Whether to allow fallback to default dimension
        
    Returns:
        Tuple of (is_valid, actual_dims, error_message)
    """
    if not embedding:
        if allow_fallback:
            return False, DEFAULT_DIMENSION, "Empty embedding, using fallback dimension"
        return False, 0, "Empty or None embedding provided"
    
    if not isinstance(embedding, list):
        if allow_fallback:
            return False, DEFAULT_DIMENSION, f"Invalid embedding type: {type(embedding)}, using fallback"
        return False, 0, f"Invalid embedding type: {type(embedding)}"
    
    actual_dims = len(embedding)
    
    # Check if dimensions are within reasonable bounds
    if actual_dims < MIN_DIMENSION or actual_dims > MAX_DIMENSION:
        if allow_fallback:
            return False, DEFAULT_DIMENSION, f"Dimension {actual_dims} outside bounds [{MIN_DIMENSION}, {MAX_DIMENSION}], using fallback"
        return False, actual_dims, f"Dimension {actual_dims} outside supported bounds"
    
    # Check if dimensions are supported
    if actual_dims not in SUPPORTED_DIMENSIONS:
        if allow_fallback:
            search_logger.warning(f"Unsupported dimension {actual_dims}, using fallback {DEFAULT_DIMENSION}")
            return False, DEFAULT_DIMENSION, f"Unsupported dimension {actual_dims}, using fallback"
        return False, actual_dims, f"Unsupported dimension: {actual_dims}"
    
    # Check expected dimensions match if provided
    if expected_dims and actual_dims != expected_dims:
        if allow_fallback:
            return False, DEFAULT_DIMENSION, f"Dimension mismatch: expected {expected_dims}, got {actual_dims}, using fallback"
        return False, actual_dims, f"Dimension mismatch: expected {expected_dims}, got {actual_dims}"
    
    return True, actual_dims, "Valid dimensions"


def validate_column_name(column_name: str) -> bool:
    """
    Validate that a column name is safe and follows expected patterns.
    Prevents SQL injection through dimension routing.
    
    Args:
        column_name: Column name to validate
        
    Returns:
        True if column name is valid and safe
        
    Raises:
        InvalidColumnNameError: If column name is invalid or unsafe
    """
    if not column_name:
        raise InvalidColumnNameError("Column name cannot be empty")
    
    if not isinstance(column_name, str):
        raise InvalidColumnNameError(f"Column name must be string, got {type(column_name)}")
    
    # Check for valid embedding column pattern
    valid_pattern = re.compile(r'^embedding_(768|1024|1536|3072)$')
    if not valid_pattern.match(column_name):
        raise InvalidColumnNameError(f"Invalid column name pattern: {column_name}")
    
    # Additional security checks - prevent SQL injection
    dangerous_chars = [';', '--', '/*', '*/', 'union', 'select', 'drop', 'delete', 'update', 'insert']
    column_lower = column_name.lower()
    
    for dangerous in dangerous_chars:
        if dangerous in column_lower:
            raise InvalidColumnNameError(f"Potentially dangerous column name: {column_name}")
    
    return True


def get_safe_dimension_column(dimensions: int, fallback: bool = True) -> str:
    """
    Get a safe database column name for the given dimensions.
    
    Args:
        dimensions: Number of embedding dimensions
        fallback: Whether to use fallback on validation failure
        
    Returns:
        Safe column name for database operations
        
    Raises:
        UnsupportedDimensionError: If dimensions are unsupported and fallback=False
        InvalidColumnNameError: If generated column name fails validation
    """
    from .embedding_service import get_dimension_column_name
    
    # Validate dimensions first
    is_valid, validated_dims, error_msg = validate_embedding_dimensions(
        [0.0] * dimensions if dimensions > 0 else None,
        allow_fallback=fallback
    )
    
    if not is_valid and not fallback:
        raise UnsupportedDimensionError(error_msg)
    
    # Get column name using validated dimensions
    column_name = get_dimension_column_name(validated_dims)
    
    # Validate column name for security
    try:
        validate_column_name(column_name)
        return column_name
    except InvalidColumnNameError as e:
        search_logger.error(f"Column name validation failed: {e}")
        if fallback:
            return "embedding_1536"  # Safe fallback
        raise


def validate_rpc_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate RPC parameters for vector search operations.
    
    Args:
        params: RPC parameters dictionary
        
    Returns:
        Validated and sanitized parameters
        
    Raises:
        DimensionValidationError: If parameters are invalid
    """
    if not isinstance(params, dict):
        raise DimensionValidationError(f"RPC parameters must be dict, got {type(params)}")
    
    validated_params = {}
    
    # Find and validate embedding parameter
    embedding_param = None
    embedding_key = None
    
    for key, value in params.items():
        if key.startswith('query_embedding_'):
            if embedding_param is not None:
                raise DimensionValidationError("Multiple embedding parameters found")
            embedding_param = value
            embedding_key = key
        else:
            validated_params[key] = value
    
    if embedding_param is None:
        raise DimensionValidationError("No embedding parameter found")
    
    # Validate embedding parameter
    is_valid, dims, error_msg = validate_embedding_dimensions(embedding_param)
    
    if not is_valid:
        search_logger.warning(f"RPC parameter validation: {error_msg}")
        # Use fallback embedding parameter
        validated_params[f"query_embedding_{DEFAULT_DIMENSION}"] = [0.0] * DEFAULT_DIMENSION
    else:
        validated_params[embedding_key] = embedding_param
    
    return validated_params


def dimension_validation_decorator(validate_input: bool = True, 
                                 validate_output: bool = False,
                                 allow_fallback: bool = True):
    """
    Decorator for embedding operations that adds dimension validation.
    
    Args:
        validate_input: Whether to validate input embeddings
        validate_output: Whether to validate output embeddings  
        allow_fallback: Whether to allow fallback on validation failure
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                # Input validation
                if validate_input:
                    # Look for embedding parameters in args/kwargs
                    for i, arg in enumerate(args):
                        if isinstance(arg, list) and len(arg) > 0 and isinstance(arg[0], (int, float)):
                            is_valid, dims, error_msg = validate_embedding_dimensions(arg, allow_fallback=allow_fallback)
                            if not is_valid and not allow_fallback:
                                raise DimensionValidationError(f"Input validation failed: {error_msg}")
                            if not is_valid:
                                search_logger.warning(f"Input validation warning in {func.__name__}: {error_msg}")
                
                # Execute function
                result = await func(*args, **kwargs)
                
                # Output validation
                if validate_output and isinstance(result, list):
                    if result and isinstance(result[0], list):
                        # List of embeddings
                        for i, embedding in enumerate(result):
                            is_valid, dims, error_msg = validate_embedding_dimensions(embedding, allow_fallback=allow_fallback)
                            if not is_valid:
                                search_logger.warning(f"Output validation warning in {func.__name__} for embedding {i}: {error_msg}")
                    else:
                        # Single embedding
                        is_valid, dims, error_msg = validate_embedding_dimensions(result, allow_fallback=allow_fallback)
                        if not is_valid:
                            search_logger.warning(f"Output validation warning in {func.__name__}: {error_msg}")
                
                return result
                
            except Exception as e:
                search_logger.error(f"Dimension validation error in {func.__name__}: {e}")
                if allow_fallback:
                    # Return fallback result based on expected return type
                    if validate_output:
                        if 'batch' in func.__name__.lower():
                            return [[0.0] * DEFAULT_DIMENSION]
                        else:
                            return [0.0] * DEFAULT_DIMENSION
                raise
                
        @wraps(func) 
        def sync_wrapper(*args, **kwargs):
            # Similar logic for sync functions
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                search_logger.error(f"Dimension validation error in {func.__name__}: {e}")
                if allow_fallback:
                    if validate_output:
                        if 'batch' in func.__name__.lower():
                            return [[0.0] * DEFAULT_DIMENSION]
                        else:
                            return [0.0] * DEFAULT_DIMENSION
                raise
        
        # Return appropriate wrapper based on function type
        if hasattr(func, '__code__') and 'await' in func.__code__.co_names:
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def log_dimension_operation(operation: str, 
                          dimensions: int, 
                          success: bool,
                          error_msg: Optional[str] = None,
                          metadata: Optional[Dict[str, Any]] = None):
    """
    Log dimension-specific operations for monitoring and debugging.
    
    Args:
        operation: Type of operation (e.g., 'storage', 'search', 'embedding')
        dimensions: Number of dimensions involved
        success: Whether operation was successful
        error_msg: Error message if operation failed
        metadata: Additional metadata for logging
    """
    log_data = {
        'operation': operation,
        'dimensions': dimensions,
        'success': success,
        'supported_dimension': dimensions in SUPPORTED_DIMENSIONS
    }
    
    if metadata:
        log_data.update(metadata)
    
    if success:
        search_logger.info(f"Dimension operation successful: {operation} with {dimensions} dimensions", extra=log_data)
    else:
        search_logger.error(f"Dimension operation failed: {operation} with {dimensions} dimensions - {error_msg}", extra=log_data)


def get_validation_summary() -> Dict[str, Any]:
    """
    Get summary of current validation configuration.
    
    Returns:
        Dictionary with validation configuration details
    """
    return {
        'supported_dimensions': list(SUPPORTED_DIMENSIONS),
        'default_dimension': DEFAULT_DIMENSION,
        'dimension_bounds': {
            'min': MIN_DIMENSION,
            'max': MAX_DIMENSION
        },
        'column_patterns': [f'embedding_{dim}' for dim in sorted(SUPPORTED_DIMENSIONS)],
        'validation_active': True
    }


# Utility functions for common validation scenarios
def ensure_valid_embedding(embedding: Optional[List[float]]) -> List[float]:
    """Ensure embedding is valid, return fallback if not."""
    is_valid, dims, _ = validate_embedding_dimensions(embedding, allow_fallback=True)
    if is_valid:
        return embedding
    return [0.0] * DEFAULT_DIMENSION


def ensure_valid_column(dimensions: int) -> str:
    """Ensure column name is valid, return fallback if not."""
    try:
        return get_safe_dimension_column(dimensions, fallback=True)
    except Exception:
        return f"embedding_{DEFAULT_DIMENSION}"


def validate_batch_consistency(embeddings: List[List[float]]) -> Tuple[bool, str]:
    """
    Validate that all embeddings in a batch have consistent dimensions.
    
    Args:
        embeddings: List of embedding vectors
        
    Returns:
        Tuple of (is_consistent, error_message)
    """
    if not embeddings:
        return True, "Empty batch is consistent"
    
    if not all(isinstance(emb, list) for emb in embeddings):
        return False, "Not all items in batch are lists"
    
    dimensions = [len(emb) for emb in embeddings]
    unique_dims = set(dimensions)
    
    if len(unique_dims) > 1:
        return False, f"Inconsistent dimensions in batch: {unique_dims}"
    
    batch_dim = dimensions[0] if dimensions else 0
    if batch_dim not in SUPPORTED_DIMENSIONS:
        return False, f"Batch uses unsupported dimension: {batch_dim}"
    
    return True, f"Batch is consistent with {batch_dim} dimensions"