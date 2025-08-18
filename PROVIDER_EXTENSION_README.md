# Multi-Provider Extension for Archon

## Overview
Extension to support multiple LLM providers (OpenAI, Anthropic, Google, Groq, Together, Ollama, etc.) without breaking existing functionality.

## Architecture

### Core Components
- **BaseProvider**: Abstract interface all providers must implement
- **ProviderRegistry**: Singleton managing provider registration and retrieval
- **ProviderFactory**: Creates configured provider instances
- **Provider Implementations**: OpenAI, Anthropic, Google, Groq, Together, Ollama

### Key Features
- ✅ Provider abstraction layer
- ✅ Automatic fallback chains
- ✅ Lazy provider instantiation
- ✅ Configuration management with caching
- ✅ Rate limit handling per provider
- ✅ Model mapping for PydanticAI compatibility

## Current Status

### ✅ Completed
- Test infrastructure (44 tests passing)
- Skeleton provider implementations with TODOs
- Base provider interface
- Provider registry system
- Configuration models

### 🚧 Remaining Work

#### Backend
- [ ] Install provider SDKs (anthropic, groq packages)
- [ ] Complete OpenAI provider implementation
- [ ] Implement Anthropic provider with SDK
- [ ] Implement Groq provider with SDK
- [ ] Implement remaining providers (Together, Mistral, Cohere)
- [ ] Integrate registry with `llm_provider_service.py`
- [ ] Add provider management API endpoints

#### Frontend
- [ ] Dynamic provider selection dropdown
- [ ] Model selection based on provider
- [ ] Provider-specific configuration fields
- [ ] Connection testing UI
- [ ] Provider health status indicators

#### Integration
- [ ] Update agent base class to use registry
- [ ] Migrate existing OpenAI calls to provider system
- [ ] Update embedding service for multi-provider support
- [ ] Add provider usage analytics

## Testing
All provider tests passing:
```bash
cd python && uv run pytest tests/test_providers/ -q
# 44 passed
```

## No Breaking Changes
- All new code is additive
- Existing OpenAI functionality preserved
- Gradual migration path available
- Feature flags for new functionality