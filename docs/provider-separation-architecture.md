# LLM/Embedding Provider Separation Architecture

## Executive Summary

The Archon V2 Alpha system has implemented a comprehensive provider separation architecture that allows independent configuration of LLM and embedding providers. This document analyzes the architectural changes, configuration mechanisms, validation workflows, and integration impacts of the provider separation system.

## Architecture Overview

### Key Architectural Changes

The provider separation system introduces several critical architectural changes:

1. **Independent Provider Configuration**: LLM and embedding providers can now be configured separately
2. **Service Type Differentiation**: The system distinguishes between "llm" and "embedding" service types
3. **Dynamic Provider Resolution**: Provider selection is resolved at runtime based on service type
4. **Unified Client Interface**: All providers use OpenAI-compatible client interfaces

### Core Components

#### 1. LLM Provider Service (`llm_provider_service.py`)

The core service manages provider client creation with the following key features:

```python
@asynccontextmanager
async def get_llm_client(provider: Optional[str] = None, use_embedding_provider: bool = False):
    """
    Create an async OpenAI-compatible client based on the configured provider.
    
    Args:
        provider: Override provider selection
        use_embedding_provider: Use the embedding-specific provider if different
    """
```

**Key Implementation Details:**
- **Caching Mechanism**: Implements 5-minute TTL cache for provider configurations to reduce database queries
- **Service Type Routing**: `use_embedding_provider` flag routes to embedding-specific provider configuration
- **Provider Support**: OpenAI, Ollama, and Google Gemini with unified OpenAI-compatible interface
- **Error Handling**: Comprehensive error handling with detailed logging

#### 2. Credential Service Provider Configuration

The credential service provides provider-specific configuration through:

```python
async def get_active_provider(service_type: str) -> Dict[str, Any]:
    """
    Get the active provider configuration for a specific service type.
    
    Args:
        service_type: Either "llm" or "embedding"
        
    Returns:
        Dict containing provider, api_key, base_url, and embedding_model
    """
```

## Provider Configuration Schema

### Database Structure

The provider separation uses a flexible configuration schema stored in the `archon_settings` table:

| Field | Description | Example |
|-------|-------------|---------|
| `key` | Configuration identifier | `LLM_PROVIDER`, `EMBEDDING_PROVIDER` |
| `value` | Plain text configuration | `openai`, `ollama` |
| `encrypted_value` | Encrypted sensitive data | API keys |
| `is_encrypted` | Encryption flag | `true` for API keys |
| `category` | Grouping mechanism | `rag_strategy`, `api_keys` |

### Provider Selection Mechanisms

#### 1. Service-Type Based Selection

```python
# Get LLM provider for chat/generation
service_type = "llm"
provider_config = await credential_service.get_active_provider(service_type)

# Get embedding provider for vector operations
service_type = "embedding" 
provider_config = await credential_service.get_active_provider(service_type)
```

#### 2. Explicit Provider Override

```python
# Override provider selection
async with get_llm_client(provider="ollama") as client:
    # Use Ollama specifically regardless of configuration
```

#### 3. Configuration Hierarchy

1. **Explicit Override**: Direct provider parameter
2. **Service-Type Configuration**: Database stored per-service provider
3. **Fallback**: Default provider (OpenAI)

### Credential Management Updates

#### API Key Management

```python
async def _get_provider_api_key(self, provider: str) -> Optional[str]:
    """Get API key for specific provider with encryption support."""
    key_mapping = {
        "openai": "OPENAI_API_KEY",
        "google": "GOOGLE_API_KEY", 
        "ollama": None  # Ollama doesn't require API key
    }
```

#### Base URL Configuration

```python
def _get_provider_base_url(self, provider: str, rag_settings: Dict) -> Optional[str]:
    """Get base URL for provider with configuration fallbacks."""
    if provider == "ollama":
        return rag_settings.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    elif provider == "google":
        return "https://generativelanguage.googleapis.com/v1beta/openai/"
```

## Validation Workflows

### Embedding Model Validation API

The system provides comprehensive validation for embedding model changes through `/api/embedding-models/validate`:

#### Validation Request Schema

```python
class ModelValidationRequest(BaseModel):
    provider: str  # "openai", "ollama", "google"
    model_name: str  # "text-embedding-3-small", "nomic-embed-text"
```

#### Validation Response Schema

```python
class ModelValidationResponse(BaseModel):
    is_valid: bool
    is_change: bool
    dimensions_change: bool
    requires_migration: bool
    data_loss_warning: bool
    current: Dict[str, Any]
    new: Dict[str, Any]
    error: str = None
```

### Dimension Awareness Integration

#### Model Dimension Detection

```python
async def detect_model_dimensions(self, model_name: str, provider: str = None) -> int:
    """
    Detect the dimensions of an embedding model by creating a test embedding.
    Handles provider-specific model configurations and caches results.
    """
```

#### Recommended Models Configuration

```python
RECOMMENDED_MODELS = {
    "openai": {
        "text-embedding-3-small": {
            "dimensions": 1536,
            "description": "OpenAI's general-purpose model - high quality, balanced performance",
            "use_case": "General purpose, good for most applications",
            "provider": "openai"
        }
    },
    "ollama": {
        "nomic-embed-text": {
            "dimensions": 768,
            "description": "Fast, lightweight model good for quick processing",
            "use_case": "Fast processing, resource-constrained environments",
            "provider": "ollama"
        }
    }
}
```

### Provider Compatibility Checks

#### Validation Workflow

1. **Current Model Detection**: Detect current provider and model
2. **New Model Validation**: Validate requested provider/model combination
3. **Dimension Comparison**: Check for dimension changes requiring migration
4. **Data Impact Assessment**: Evaluate potential data loss scenarios
5. **Migration Requirements**: Determine if re-embedding is needed

#### Data Loss Scenarios

The system identifies three critical data loss scenarios:

1. **Dimension Changes**: When vector dimensions change (e.g., 1536 → 768)
2. **Existing Embeddings**: When embeddings already exist in the database
3. **Schema Migration**: When database schema updates are required

## Integration Impact

### API Endpoint Changes

#### New Embedding Model Management Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/embedding-models/recommendations` | GET | Get all recommended models |
| `/api/embedding-models/current` | GET | Get current model information |
| `/api/embedding-models/validate` | POST | Validate model change |
| `/api/embedding-models/change` | POST | Execute model change |

### UI Configuration Changes

#### Frontend Provider Selection

The React UI now supports independent provider selection:

```tsx
// RAGSettings component shows separate provider dropdowns
<Select
  label="LLM Provider"
  value={ragSettings.LLM_PROVIDER || 'openai'}
  onChange={e => setRagSettings({
    ...ragSettings,
    LLM_PROVIDER: e.target.value
  })}
/>

<Select
  label="Embedding Provider"  
  value={ragSettings.EMBEDDING_PROVIDER || 'openai'}
  onChange={e => setRagSettings({
    ...ragSettings,
    EMBEDDING_PROVIDER: e.target.value
  })}
/>
```

#### EmbeddingModelChanger Component

New dedicated component for embedding model management:

```tsx
interface ModelRecommendation {
  model_name: string;
  provider: string;
  dimensions: number;
  description: string;
  use_case: string;
}
```

### Backward Compatibility Measures

#### Configuration Migration

1. **Legacy Setting Support**: Existing `LLM_PROVIDER` settings work for both LLM and embedding
2. **Graceful Fallbacks**: Missing provider configuration falls back to OpenAI
3. **Default Behavior**: System maintains previous behavior when no explicit embedding provider is set

#### API Compatibility

1. **Optional Parameters**: All new provider parameters are optional
2. **Existing Workflows**: Current crawling and embedding workflows unchanged
3. **Error Handling**: Enhanced error messages guide users through provider issues

## Performance Considerations

### Caching Implementation

The provider service implements intelligent caching to reduce database queries:

```python
# Settings cache with TTL
_settings_cache: Dict[str, Tuple[Any, float]] = {}
_CACHE_TTL_SECONDS = 300  # 5 minutes

def _get_cached_settings(key: str) -> Optional[Any]:
    """Get cached settings if not expired."""
    if key in _settings_cache:
        value, timestamp = _settings_cache[key]
        if time.time() - timestamp < _CACHE_TTL_SECONDS:
            return value
```

### Provider Resolution Optimization

1. **Lazy Loading**: Providers are resolved only when needed
2. **Connection Pooling**: OpenAI clients reuse connections where possible
3. **Context Management**: Proper cleanup of provider clients

## Security Considerations

### API Key Management

1. **Encryption at Rest**: All API keys stored encrypted in database
2. **Memory Safety**: API keys not logged or cached in plain text
3. **Provider Isolation**: Each provider's credentials stored separately

### Provider Validation

1. **Input Validation**: All provider configurations validated before storage
2. **Model Verification**: Embedding models tested before activation
3. **Error Sanitization**: Error messages don't expose sensitive configuration

## Migration Path

### From Single to Separated Providers

1. **Automatic Detection**: System detects existing single provider configuration
2. **Configuration Replication**: LLM_PROVIDER setting applied to both services initially
3. **Independent Configuration**: Users can then configure providers separately
4. **Validation Workflow**: Changes validated before application

### Database Schema Evolution

The provider separation doesn't require schema changes but utilizes the existing flexible configuration system:

```sql
-- Example provider configuration storage
INSERT INTO archon_settings (key, value, category, description) VALUES
('LLM_PROVIDER', 'openai', 'rag_strategy', 'Provider for LLM operations'),
('EMBEDDING_PROVIDER', 'ollama', 'rag_strategy', 'Provider for embedding operations');
```

## Conclusion

The provider separation architecture provides significant flexibility while maintaining backward compatibility. The system allows independent optimization of LLM and embedding providers, enabling users to:

1. **Optimize Costs**: Use different providers based on cost/performance needs
2. **Leverage Strengths**: Combine best-in-class providers for different operations
3. **Ensure Availability**: Configure fallbacks and alternatives
4. **Maintain Security**: Keep provider credentials isolated and encrypted

The implementation demonstrates careful consideration of existing workflows while providing powerful new capabilities for provider management and optimization.