# NUM_CTX Implementation Documentation

## Overview

This document describes the implementation of custom NUM_CTX settings for Ollama provider configuration in Archon V2 Alpha. NUM_CTX controls the context window size (token count) that Ollama models can process in a single request.

## What is NUM_CTX?

NUM_CTX is an Ollama-specific parameter that:
- Controls the maximum number of tokens the model can process at once
- Affects memory usage (higher values require more RAM)
- Impacts model performance and response quality
- Valid range: 512-32768 tokens
- Default value: 4096 tokens

## Implementation Details

### 1. Frontend Implementation

#### UI Component (`RAGSettings.tsx`)
- Added `OLLAMA_NUM_CTX` field to the `RAGSettingsProps` interface
- Created NUM_CTX input field that appears only when Ollama provider is selected
- Includes validation (min: 512, max: 32768)
- Provides helpful tooltip explaining the parameter
- Styled with amber accent to indicate Ollama-specific configuration

#### TypeScript Interface (`credentialsService.ts`)
- Extended `RagSettings` interface with `OLLAMA_NUM_CTX?: number`
- Ensures type safety across the frontend

### 2. Backend Implementation

#### Credential Service (`credential_service.py`)
- Added `_get_provider_num_ctx()` method to extract NUM_CTX from RAG settings
- Enhanced `get_active_provider()` to include NUM_CTX in provider configuration
- Implements validation and fallback logic (default: 4096)

#### LLM Provider Service (`llm_provider_service.py`)
- Added `get_ollama_extra_params()` to format NUM_CTX for Ollama API
- Created `get_client_extra_params()` to extract extra params from configured clients  
- Enhanced context manager to attach provider config to client instances
- Enables services to access provider-specific parameters dynamically

#### Service Integration
Updated the following services to use NUM_CTX:
- `source_management_service.py`: Chat completions for summaries and titles
- `code_storage_service.py`: Code analysis API calls

### 3. Data Flow

1. **Configuration Storage**: NUM_CTX stored in `archon_settings` table as `OLLAMA_NUM_CTX`
2. **Provider Resolution**: Credential service resolves NUM_CTX when getting active provider
3. **Client Creation**: LLM provider service attaches config to OpenAI client instances
4. **API Calls**: Services call `get_client_extra_params()` to get NUM_CTX for Ollama requests

### 4. Usage Example

```python
# In any service making LLM API calls
from ..llm_provider_service import get_llm_client, get_client_extra_params

async with get_llm_client() as client:
    extra_params = get_client_extra_params(client)
    
    response = client.chat.completions.create(
        model="llama2",
        messages=[...],
        **extra_params  # Includes {"options": {"num_ctx": 8192}} for Ollama
    )
```

### 5. Configuration UI

The NUM_CTX setting appears in the RAG Settings page:
- Only visible when Ollama is selected as LLM provider
- Number input with validation
- Helper text explaining memory implications
- Saves to database with other RAG settings

## Default Values

- **Default NUM_CTX**: 4096 tokens
- **Minimum**: 512 tokens  
- **Maximum**: 32768 tokens
- **Validation**: Automatically clamped to valid range

## Benefits

1. **Performance Optimization**: Users can tune context window for their use case
2. **Memory Management**: Lower values reduce RAM usage on resource-constrained systems
3. **Quality Control**: Higher values allow processing of longer documents
4. **Provider Isolation**: NUM_CTX only affects Ollama, not OpenAI or Google
5. **Seamless Integration**: Works with existing provider separation architecture

## Compatibility

- **Frontend**: React TypeScript component with validation
- **Backend**: Python FastAPI with async context managers  
- **Database**: Uses existing `archon_settings` flexible configuration system
- **Providers**: Ollama-specific, ignored by OpenAI and Google Gemini
- **Migration**: No database schema changes required

## Testing

- Frontend builds successfully with TypeScript validation
- Backend services import without errors
- Provider service creates clients with attached configuration
- Configuration properly passed to Ollama API calls as `options.num_ctx`

## Future Enhancements

Potential improvements for future releases:
1. **Dynamic Validation**: Check actual model capabilities
2. **Memory Monitoring**: Display current memory usage with NUM_CTX settings
3. **Model-Specific Defaults**: Different defaults per Ollama model
4. **Auto-Tuning**: Automatically adjust based on document length
5. **Performance Metrics**: Track response times vs NUM_CTX values

## Troubleshooting

### Common Issues

1. **High Memory Usage**: Reduce NUM_CTX value
2. **Context Too Short**: Increase NUM_CTX value
3. **Model Errors**: Ensure NUM_CTX is within model's supported range
4. **UI Not Showing**: Verify Ollama is selected as LLM provider

### Debug Steps

1. Check RAG settings in database: `SELECT * FROM archon_settings WHERE key = 'OLLAMA_NUM_CTX'`
2. Verify provider configuration: Look for NUM_CTX in active provider config
3. Check API calls: Ensure `options.num_ctx` is passed to Ollama
4. Monitor Ollama logs: Check for NUM_CTX-related errors

This implementation provides a complete, production-ready NUM_CTX configuration system integrated with Archon's existing provider separation architecture.