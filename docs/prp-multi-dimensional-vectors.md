# 📋 PRP: Multi-Dimensional Vector Support Implementation

**Document Type**: Product Requirement Prompt (PRP)  
**Title**: Multi-Dimensional Vector Support Implementation  
**Version**: 1.0  
**Author**: AI IDE Agent  
**Date**: 2025-08-10  
**Status**: APPROVED  
**Project ID**: 82665932-7a8f-459b-a8d6-afb8b509cf8a

## 🎯 Goal

Complete the multi-dimensional vector migration by fixing application code compatibility issues identified during database schema review.

## 🤔 Why

1. **Database schema migration to multi-dimensional vectors is complete but application code still references legacy single 'embedding' column**
2. **Current code cannot utilize the new dimensional-specific columns (embedding_768, embedding_1024, embedding_1536, embedding_3072) properly**
3. **Vector search functions expect dimension-specific parameters but current code uses generic parameters**
4. **System needs full compatibility with multiple embedding models (text-embedding-3-small, text-embedding-3-large, custom models)**
5. **Hard-coded 1536 dimensions prevent utilizing optimal embedding sizes for different models**

## 📝 What

### Description
Fix all application code compatibility issues to complete multi-dimensional vector migration and enable full utilization of dimensional-specific embedding storage and search.

### Success Criteria
- Document storage services use dynamic dimension-specific column names instead of hardcoded 'embedding'
- Code storage services properly route embeddings to correct dimensional columns
- Vector search services call RPC functions with dimension-specific parameters
- Embedding service creates embeddings with model-appropriate dimensions
- All storage operations work correctly with 768, 1024, 1536, and 3072 dimensional embeddings
- Vector similarity search performs optimally using proper indexes for each dimension
- System gracefully handles unsupported dimensions with clear error messages

### User Stories
- **As a developer**, I want to store embeddings from text-embedding-3-large (3072 dims) without compatibility issues
- **As a user**, I want vector search to work seamlessly regardless of which embedding model was used
- **As a system administrator**, I want the system to automatically detect and use the correct dimensional columns
- **As a data scientist**, I want to experiment with different embedding models without system limitations

## 🗄️ Context

### Documentation
- **Source**: `/root/Archon-V2-Alpha/migration/complete_setup.sql`  
  **Why**: Contains updated multi-dimensional vector schema and RPC function definitions
- **Source**: `/root/Archon-V2-Alpha/migration/upgrade_multi_dimensional_vectors.sql`  
  **Why**: Migration script showing proper multi-dimensional implementation
- **Source**: `python/src/server/services/embeddings/embedding_service.py`  
  **Why**: Contains dimension detection utilities already implemented

### Existing Code
- **File**: `python/src/server/services/storage/document_storage_service.py`  
  **Purpose**: Document storage with vector embeddings - needs dimension-specific column usage
- **File**: `python/src/server/services/storage/code_storage_service.py`  
  **Purpose**: Code example storage with embeddings - needs dimension-specific column usage
- **File**: `python/src/server/services/search/vector_search_service.py`  
  **Purpose**: Vector similarity search operations - needs RPC parameter updates

### Gotchas
- Document storage service line 253 still uses hardcoded 'embedding' key
- Code storage service line 756 still uses hardcoded 'embedding' key
- Vector search RPC calls use 'query_embedding' instead of dimension-specific parameters like 'query_embedding_1536'
- Embedding service hardcodes 1536 dimensions instead of using model-appropriate sizes
- get_dimension_column_name() utility exists but isn't used by storage services
- RPC functions expect parameters like query_embedding_768, query_embedding_1024, etc.
- 3072-dimension vectors don't have ivfflat indexes due to pgvector limitations

### Current State
Database schema supports multi-dimensional vectors with proper tables, columns, indexes, and RPC functions. Application code has compatibility gaps preventing full utilization.

### Dependencies
- pgvector PostgreSQL extension
- Supabase client library
- OpenAI embeddings API
- Existing embedding service utilities

### Environment Variables
- SUPABASE_URL
- SUPABASE_SERVICE_KEY
- OPENAI_API_KEY

## 🏗️ Implementation Blueprint

### Phase 1: Storage Service Fixes
**Description**: Update storage services to use dynamic dimension-specific columns

**Tasks**:
1. **Fix document storage service embedding column references**
   - **Files**: [`python/src/server/services/storage/document_storage_service.py`]
   - **Details**: Replace hardcoded 'embedding' key with dynamic column selection using get_dimension_column_name() at lines 253, 268, 306

2. **Fix code storage service embedding column references**
   - **Files**: [`python/src/server/services/storage/code_storage_service.py`]
   - **Details**: Replace hardcoded 'embedding' key with dynamic column selection using get_dimension_column_name() at lines 756, 765, 783

3. **Add dimension detection to storage operations**
   - **Files**: [`python/src/server/services/storage/document_storage_service.py`, `python/src/server/services/storage/code_storage_service.py`]
   - **Details**: Implement embedding dimension detection and appropriate error handling for unsupported dimensions

### Phase 2: Vector Search Fixes
**Description**: Update vector search to use dimension-specific RPC parameters

**Tasks**:
1. **Fix vector search RPC parameter names**
   - **Files**: [`python/src/server/services/search/vector_search_service.py`]
   - **Details**: Update RPC calls at lines 91, 199, 275 to use dimension-specific parameters (query_embedding_768, query_embedding_1536, etc.)

2. **Create RPC parameter builder utility**
   - **Files**: [`python/src/server/services/search/vector_search_service.py`]
   - **Details**: Implement helper function to build correct RPC parameters based on embedding dimensions

3. **Add dimension-aware search routing**
   - **Files**: [`python/src/server/services/search/vector_search_service.py`]
   - **Details**: Ensure search operations route to correct dimensional columns and handle edge cases

### Phase 3: Embedding Service Enhancements
**Description**: Make embedding service fully dimension-aware

**Tasks**:
1. **Remove hardcoded 1536 dimensions in embedding creation**
   - **Files**: [`python/src/server/services/embeddings/embedding_service.py`]
   - **Details**: Update line 262 to use model-appropriate dimensions from get_embedding_dimensions()

2. **Add comprehensive dimension validation**
   - **Files**: [`python/src/server/services/embeddings/embedding_service.py`]
   - **Details**: Enhance error handling for unsupported dimensions and provide clear feedback

3. **Create embedding model configuration management**
   - **Files**: [`python/src/server/services/embeddings/embedding_service.py`]
   - **Details**: Add support for configurable embedding models and their dimensional requirements

### Phase 4: Testing and Validation
**Description**: Comprehensive testing of multi-dimensional vector support

**Tasks**:
1. **Create multi-dimensional vector integration tests**
   - **Files**: [`python/tests/test_multi_dimensional_vectors.py`]
   - **Details**: Test storage, search, and retrieval across all supported dimensions (768, 1024, 1536, 3072)

2. **Validate performance with dimensional indexes**
   - **Files**: [`python/tests/test_vector_performance.py`]
   - **Details**: Benchmark vector search performance across different dimensions and verify index utilization

3. **Test embedding model compatibility**
   - **Files**: [`python/tests/test_embedding_models.py`]
   - **Details**: Verify system works with text-embedding-3-small (768, 1536), text-embedding-3-large (3072), and custom models

## ✅ Validation

### Level 1: Syntax
- `python -m pytest python/tests/test_multi_dimensional_vectors.py -v`
- `python -m ruff check python/src/server/services/storage/ python/src/server/services/search/ python/src/server/services/embeddings/`
- `python -m mypy python/src/server/services/storage/ python/src/server/services/search/ python/src/server/services/embeddings/`

### Level 2: Unit Tests
- `pytest python/tests/test_document_storage_service.py -v -k multi_dimensional`
- `pytest python/tests/test_code_storage_service.py -v -k multi_dimensional`
- `pytest python/tests/test_vector_search_service.py -v -k multi_dimensional`
- `pytest python/tests/test_embedding_service.py -v -k dimensions`

### Level 3: Integration
- `pytest python/tests/test_integration_multi_dimensional.py -v`
- `curl -X POST http://localhost:8181/api/knowledge/search -d '{"query": "test", "embedding_model": "text-embedding-3-large"}'`
- `curl -X POST http://localhost:8181/api/knowledge/crawl -d '{"url": "https://example.com", "embedding_model": "text-embedding-3-small"}'`
- `python -m src.scripts.test_all_dimensions`

### Level 4: End-to-End
- Start development server: `uvicorn src.server.main:app --reload --port 8181`
- Test document storage with text-embedding-3-large (3072 dims) via API
- Test code storage with text-embedding-3-small reduced (768 dims) via API
- Verify vector search works across all dimensional embeddings
- Test performance with large datasets using different embedding models
- Validate database storage shows correct dimensional column usage

## 📚 Additional Context

### Security Considerations
- Ensure dimension validation prevents potential SQL injection through column name manipulation
- Validate embedding vector lengths match expected dimensions to prevent data corruption
- Implement proper error handling to avoid exposing internal database structure
- Ensure proper access control for multi-dimensional vector operations

### Testing Strategies
- Mock embedding API responses with different dimensional outputs
- Test edge cases: empty embeddings, oversized vectors, unsupported dimensions
- Performance test search operations across all supported dimensions
- Test migration scenarios from single to multi-dimensional storage
- Validate backward compatibility with existing embedded documents

### Monitoring and Logging
- Log embedding dimension detection and routing decisions
- Monitor storage operations to ensure correct dimensional column usage
- Track vector search performance across different dimensions
- Alert on unsupported dimension usage attempts
- Monitor index usage for optimal query performance

---

This PRP provides comprehensive guidance for completing the multi-dimensional vector migration, ensuring all application code compatibility issues are addressed while maintaining system robustness and performance.