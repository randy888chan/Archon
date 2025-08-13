# LLM Provider Integration Guide

## Overview

Archon uses a centralized LLM provider service that supports multiple LLM providers through a unified OpenAI-compatible interface. This document explains the proper integration patterns and helps prevent common anti-patterns that bypass the provider service.

## Architecture

### Provider Service Location
- **File**: `python/src/server/services/llm_provider_service.py`
- **Main Function**: `get_llm_client(provider: str | None = None, use_embedding_provider: bool = False)`
- **Pattern**: Async context manager that yields OpenAI-compatible clients

### Supported Providers

1. **OpenAI** - Standard OpenAI API with API key authentication
2. **Ollama** - Local LLM server with custom base URL
3. **Google Gemini** - Google's API through OpenAI-compatible interface

## Integration Patterns

### ✅ Correct Pattern: Provider Service Integration

```python
from .llm_provider_service import get_llm_client

async def extract_source_summary(
    source_id: str, content: str, max_length: int = 500, provider: str = None
) -> str:
    """Generate source summary using provider service."""
    # Get model choice from credential service
    model_choice = _get_model_choice()
    
    # Create prompt
    prompt = f"Summarize this content from {source_id}:\n{content[:25000]}"
    
    try:
        # Use provider service context manager
        async with get_llm_client(provider=provider) as client:
            response = await client.chat.completions.create(
                model=model_choice,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides concise summaries."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Validate response structure
            if not response or not response.choices or len(response.choices) == 0:
                logger.error(f"Empty or invalid response from LLM for {source_id}")
                return f"Content from {source_id}"
                
            message_content = response.choices[0].message.content
            if message_content is None:
                logger.error(f"LLM returned None content for {source_id}")
                return f"Content from {source_id}"
                
            summary = message_content.strip()
            
            # Ensure summary is not too long
            if len(summary) > max_length:
                summary = summary[:max_length] + "..."
                
            return summary
            
    except Exception as e:
        logger.error(f"Error generating summary with LLM for {source_id}: {e}")
        return f"Content from {source_id}"
```

### ❌ Anti-Pattern: Direct Provider Creation

```python
import openai  # NEVER DO THIS IN BUSINESS LOGIC

def extract_source_summary(source_id: str, content: str) -> str:
    """WRONG: Creates hardcoded OpenAI client"""
    
    # ANTI-PATTERN: Bypasses provider service
    client = openai.OpenAI(api_key="sk-...")  
    
    # PROBLEMS:
    # 1. Only works with OpenAI, breaks multi-provider support
    # 2. Hardcodes API key instead of using credential service
    # 3. Not async compatible
    # 4. No provider configuration support
    # 5. Bypasses caching and error handling
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": content}]
    )
    return response.choices[0].message.content
```

## Function Signature Requirements

### Required Elements

1. **Async Function**: All functions using LLM provider service MUST be async
2. **Provider Parameter**: Include `provider: str = None` for override capability
3. **Error Handling**: Implement proper fallback behavior
4. **Response Validation**: Check response structure before using

```python
# ✅ CORRECT: Complete function signature
async def my_llm_function(
    content: str,
    context: str = "",
    max_length: int = 500,
    provider: str = None  # Always include provider override
) -> str:
    """Function that uses LLM provider service correctly."""
    default_result = "Fallback value"
    
    try:
        async with get_llm_client(provider=provider) as client:
            # LLM processing logic
            response = await client.chat.completions.create(...)
            
            # Always validate response
            if not response or not response.choices:
                return default_result
                
            content = response.choices[0].message.content
            return content.strip() if content else default_result
            
    except Exception as e:
        logger.error(f"LLM operation failed: {e}")
        return default_result

# ❌ WRONG: Missing required elements
def my_llm_function(content: str) -> str:  # Not async, no provider param
    # Cannot use async provider service properly
    pass
```

## Error Handling Best Practices

### Provider Configuration Errors

```python
async def robust_llm_function(content: str, provider: str = None) -> str:
    """Example of comprehensive error handling."""
    default_result = "Default fallback"
    
    try:
        async with get_llm_client(provider=provider) as client:
            logger.info(f"Using LLM provider: {provider or 'default configured'}")
            
            response = await client.chat.completions.create(
                model=model_choice,
                messages=[{"role": "user", "content": content}]
            )
            
            # Validate response structure
            if not response:
                logger.error("Received None response from LLM")
                return default_result
                
            if not response.choices or len(response.choices) == 0:
                logger.error("LLM response has no choices")
                return default_result
                
            message_content = response.choices[0].message.content
            if message_content is None:
                logger.error("LLM message content is None")
                return default_result
                
            return message_content.strip()
            
    except ValueError as e:
        # Provider configuration errors (missing API keys, invalid base URLs)
        logger.error(f"Provider configuration error: {e}")
        return default_result
        
    except Exception as e:
        # Network errors, API errors, timeout errors, etc.
        logger.error(f"LLM request failed: {e}")
        return default_result
```

### Response Validation Patterns

```python
# ✅ CORRECT: Complete response validation
def validate_llm_response(response) -> bool:
    """Validate LLM response structure."""
    if not response:
        logger.error("Response is None or falsy")
        return False
        
    if not hasattr(response, 'choices'):
        logger.error("Response missing choices attribute")
        return False
        
    if not response.choices or len(response.choices) == 0:
        logger.error("Response choices is empty")
        return False
        
    if not hasattr(response.choices[0], 'message'):
        logger.error("First choice missing message attribute")
        return False
        
    if response.choices[0].message.content is None:
        logger.error("Message content is None")
        return False
        
    return True

async def process_with_validation(content: str, provider: str = None) -> str:
    """Example using response validation."""
    try:
        async with get_llm_client(provider=provider) as client:
            response = await client.chat.completions.create(
                model=model_choice,
                messages=[{"role": "user", "content": content}]
            )
            
            if validate_llm_response(response):
                return response.choices[0].message.content.strip()
            else:
                return "Validation failed - using default"
                
    except Exception as e:
        logger.error(f"LLM processing failed: {e}")
        return "Error occurred - using default"
```

## Code Review Checklist

When reviewing code that integrates with LLM providers, verify:

### Function Structure
- [ ] Function is declared `async`
- [ ] Includes `provider: str = None` parameter
- [ ] Uses `async with get_llm_client(provider=provider) as client:` pattern
- [ ] Has proper exception handling with try/except blocks

### Provider Service Usage
- [ ] No direct `import openai` in business logic files
- [ ] No `openai.OpenAI()` or `openai.AsyncOpenAI()` instantiation
- [ ] Uses provider service context manager correctly
- [ ] Passes provider parameter through the call chain

### Error Handling
- [ ] Has meaningful fallback values for all error conditions
- [ ] Validates LLM response structure before using
- [ ] Logs errors with sufficient context for debugging
- [ ] Handles both configuration and runtime errors

### Response Processing
- [ ] Checks `response.choices` exists and is not empty
- [ ] Validates `message.content` is not None before using
- [ ] Strips whitespace from response content
- [ ] Applies length limits or other post-processing as needed

## Common Anti-Patterns to Avoid

### 1. Direct Client Creation
```python
# ❌ NEVER DO THIS
client = openai.OpenAI(api_key=api_key)
client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)

# ✅ ALWAYS USE THIS
async with get_llm_client(provider=provider) as client:
    # Use client here
```

### 2. Hardcoded Provider Logic
```python
# ❌ NEVER DO THIS
if provider == "openai":
    client = openai.AsyncOpenAI(api_key=openai_key)
elif provider == "ollama":
    client = openai.AsyncOpenAI(api_key="ollama", base_url=ollama_url)

# ✅ ALWAYS USE THIS
async with get_llm_client(provider=provider) as client:
    # Provider selection handled by service
```

### 3. Sync Functions with LLM Calls
```python
# ❌ NEVER DO THIS
def sync_function_with_llm(content):
    # Cannot properly use async provider service
    pass

# ✅ ALWAYS USE THIS
async def async_function_with_llm(content, provider=None):
    async with get_llm_client(provider=provider) as client:
        # Proper async usage
```

### 4. Missing Error Handling
```python
# ❌ NEVER DO THIS
async def fragile_llm_function(content, provider=None):
    async with get_llm_client(provider=provider) as client:
        response = await client.chat.completions.create(...)
        return response.choices[0].message.content  # Can crash

# ✅ ALWAYS USE THIS
async def robust_llm_function(content, provider=None):
    try:
        async with get_llm_client(provider=provider) as client:
            response = await client.chat.completions.create(...)
            if response and response.choices:
                content = response.choices[0].message.content
                return content.strip() if content else default_value
            return default_value
    except Exception as e:
        logger.error(f"LLM failed: {e}")
        return default_value
```

## Testing Provider Integration

### Unit Test Pattern
```python
import pytest
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_extract_source_summary():
    """Test source summary extraction with mocked provider."""
    
    # Mock the LLM client response
    mock_response = AsyncMock()
    mock_response.choices = [AsyncMock()]
    mock_response.choices[0].message.content = "Test summary content"
    
    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = mock_response
    
    # Mock the provider service
    with patch('your_module.get_llm_client') as mock_get_client:
        mock_get_client.return_value.__aenter__.return_value = mock_client
        
        # Test the function
        result = await extract_source_summary(
            "test-source", 
            "Test content to summarize"
        )
        
        assert result == "Test summary content"
        mock_client.chat.completions.create.assert_called_once()
```

### Integration Test Pattern
```python
@pytest.mark.asyncio
async def test_provider_integration():
    """Test actual provider integration (requires running providers)."""
    
    # Test with different providers
    providers_to_test = ["openai", "ollama"]  # Add available providers
    
    for provider in providers_to_test:
        try:
            result = await extract_source_summary(
                "test-source",
                "Simple test content",
                provider=provider
            )
            assert isinstance(result, str)
            assert len(result) > 0
            print(f"✅ Provider {provider} working correctly")
            
        except Exception as e:
            print(f"❌ Provider {provider} failed: {e}")
```

## Troubleshooting

### Common Issues

**Issue**: Function only works with OpenAI, fails with other providers
- **Cause**: Direct OpenAI client creation instead of provider service
- **Solution**: Replace with `async with get_llm_client(provider=provider)` pattern

**Issue**: `TypeError: object async context manager cannot be used in sync function`
- **Cause**: Trying to use async provider service in sync function
- **Solution**: Convert function to async: `async def function_name(...)`

**Issue**: `ValueError: Unsupported LLM provider: {name}`
- **Cause**: Provider not configured or invalid provider name
- **Solution**: Check provider configuration in database/environment

**Issue**: Functions fail silently with no output
- **Cause**: Missing error handling and response validation
- **Solution**: Add try/except blocks and validate response structure

### Debug Logging

Enable debug logging to trace provider selection:

```python
import logging
logging.getLogger('your_module.llm_provider_service').setLevel(logging.DEBUG)
```

Look for these log entries:
- "Creating LLM client for provider: {name}"
- "Using cached {service_type} provider config"
- "Fetched and cached {service_type} provider config"

### Configuration Validation

Verify provider configuration:

```python
async def validate_provider_config():
    """Validate all provider configurations."""
    providers = ["openai", "ollama", "google"]
    
    for provider in providers:
        try:
            async with get_llm_client(provider=provider) as client:
                print(f"✅ {provider} configuration valid")
        except Exception as e:
            print(f"❌ {provider} configuration error: {e}")
```

## Migration Guide

### Migrating Existing Code

1. **Identify Direct Client Usage**:
   ```bash
   grep -r "openai\." python/src/
   grep -r "OpenAI(" python/src/
   ```

2. **Convert Function to Async**:
   ```python
   # Before
   def my_function(content):
   
   # After  
   async def my_function(content, provider=None):
   ```

3. **Replace Client Creation**:
   ```python
   # Before
   client = openai.OpenAI(api_key=api_key)
   
   # After
   async with get_llm_client(provider=provider) as client:
   ```

4. **Add Error Handling**:
   ```python
   # Before
   response = client.chat.completions.create(...)
   return response.choices[0].message.content
   
   # After
   try:
       response = await client.chat.completions.create(...)
       if response and response.choices:
           content = response.choices[0].message.content
           return content.strip() if content else default_value
       return default_value
   except Exception as e:
       logger.error(f"LLM failed: {e}")
       return default_value
   ```

5. **Update Function Calls**:
   ```python
   # Before
   result = my_function(content)
   
   # After
   result = await my_function(content, provider=provider)
   ```

This guide ensures consistent, maintainable LLM provider integration across the Archon codebase while supporting multiple providers and providing robust error handling.