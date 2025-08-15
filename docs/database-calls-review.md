# 📊 Complete Database Calls Review - Archon V2 Alpha

**Date**: 2025-08-10  
**Project**: Multi-Dimensional Vector Migration  
**Branch**: feature/multi-dimensional-vectors

## 🏗️ Database Architecture Overview

**Primary Database**: PostgreSQL via Supabase  
**Vector Extension**: pgvector with multi-dimensional support  
**Connection Pattern**: Supabase client via `get_supabase_client()` in `/root/Archon-V2-Alpha/python/src/server/services/client_manager.py`

## 📋 Complete Database Calls Inventory

### 1. **Document Storage Operations**
**File**: `python/src/server/services/storage/document_storage_service.py`

| Line | Operation | Table | Vector Column | Status |
|------|-----------|-------|---------------|---------|
| 84 | DELETE | archon_crawled_pages | N/A | ✅ Compatible |
| 101 | DELETE | archon_crawled_pages | N/A | ✅ Compatible |
| 253 | INSERT | archon_crawled_pages | `"embedding"` | ❌ **NEEDS FIX** |
| 268 | INSERT | archon_crawled_pages | `"embedding"` | ❌ **NEEDS FIX** |
| 306 | INSERT | archon_crawled_pages | `"embedding"` | ❌ **NEEDS FIX** |

### 2. **Code Storage Operations**
**File**: `python/src/server/services/storage/code_storage_service.py`

| Line | Operation | Table | Vector Column | Status |
|------|-----------|-------|---------------|---------|
| 652 | DELETE | archon_code_examples | N/A | ✅ Compatible |
| 756 | INSERT | archon_code_examples | `"embedding"` | ❌ **NEEDS FIX** |
| 765 | INSERT | archon_code_examples | `"embedding"` | ❌ **NEEDS FIX** |
| 783 | INSERT | archon_code_examples | `"embedding"` | ❌ **NEEDS FIX** |

### 3. **Vector Search Operations**
**File**: `python/src/server/services/search/vector_search_service.py`

| Line | Operation | RPC Function | Parameters | Status |
|------|-----------|--------------|------------|---------|
| 91 | RPC | match_archon_crawled_pages | `query_embedding` | ❌ **NEEDS FIX** |
| 199 | RPC | match_archon_crawled_pages | `query_embedding` | ❌ **NEEDS FIX** |
| 275 | RPC | match_archon_code_examples | `query_embedding` | ❌ **NEEDS FIX** |

### 4. **Project Management Operations**
**File**: `python/src/server/services/projects/project_service.py`

| Line | Operation | Table | Purpose | Status |
|------|-----------|-------|---------|---------|
| 52 | INSERT | archon_projects | Create project | ✅ Compatible |
| 83 | SELECT | archon_projects | List projects | ✅ Compatible |
| 117 | SELECT | archon_projects | Get project by ID | ✅ Compatible |
| 128 | SELECT | archon_project_sources | Get project sources | ✅ Compatible |
| 173 | SELECT | archon_projects | Check project exists | ✅ Compatible |
| 178 | SELECT | archon_tasks | Get project tasks | ✅ Compatible |
| 182 | DELETE | archon_projects | Delete project | ✅ Compatible |
| 255 | UPDATE | archon_projects | Unpin projects | ✅ Compatible |
| 258 | UPDATE | archon_projects | Update project | ✅ Compatible |

### 5. **Task Management Operations**
**File**: `python/src/server/services/projects/task_service.py`

| Line | Operation | Table | Purpose | Status |
|------|-----------|-------|---------|---------|
| 89 | SELECT | archon_tasks | Get tasks for reordering | ✅ Compatible |
| 102 | UPDATE | archon_tasks | Update task orders | ✅ Compatible |
| 124 | INSERT | archon_tasks | Create new task | ✅ Compatible |
| 170 | SELECT | archon_tasks | List tasks with filtering | ✅ Compatible |
| 273 | SELECT | archon_tasks | Get task by ID | ✅ Compatible |
| 324 | UPDATE | archon_tasks | Update task | ✅ Compatible |
| 367 | SELECT | archon_tasks | Get task before archiving | ✅ Compatible |
| 385 | UPDATE | archon_tasks | Archive task | ✅ Compatible |

### 6. **Source Management Operations**
**File**: `python/src/server/services/source_management_service.py`

| Line | Operation | Table | Purpose | Status |
|------|-----------|-------|---------|---------|
| 208 | SELECT | archon_sources | Check existing source | ✅ Compatible |
| 226 | UPDATE | archon_sources | Update existing source | ✅ Compatible |
| 246 | INSERT | archon_sources | Create new source | ✅ Compatible |
| 277 | SELECT | archon_sources | List all sources | ✅ Compatible |
| 314 | DELETE | archon_crawled_pages | Delete pages by source | ✅ Compatible |
| 324 | DELETE | archon_code_examples | Delete code by source | ✅ Compatible |
| 334 | DELETE | archon_sources | Delete source record | ✅ Compatible |
| 383 | SELECT | archon_sources | Get source metadata | ✅ Compatible |
| 397 | UPDATE | archon_sources | Update source metadata | ✅ Compatible |
| 468 | SELECT | archon_sources | Get source for item listing | ✅ Compatible |
| 476 | SELECT | archon_crawled_pages | Count pages by source | ✅ Compatible |
| 480 | SELECT | archon_code_examples | Count code by source | ✅ Compatible |
| 504 | SELECT | archon_sources | Search sources | ✅ Compatible |

### 7. **Settings & Credentials Operations**
**File**: `python/src/server/services/credential_service.py`

| Line | Operation | Table | Purpose | Status |
|------|-----------|-------|---------|---------|
| 125 | SELECT | archon_settings | Load all settings | ✅ Compatible |
| 219 | UPSERT | archon_settings | Save/update settings | ✅ Compatible |
| 242 | DELETE | archon_settings | Delete setting | ✅ Compatible |
| 281 | SELECT | archon_settings | Get settings by category | ✅ Compatible |
| 311 | SELECT | archon_settings | Get all settings | ✅ Compatible |

### 8. **Database Metrics Operations**
**File**: `python/src/server/services/knowledge/database_metrics_service.py`

| Line | Operation | Table | Purpose | Status |
|------|-----------|-------|---------|---------|
| 40 | SELECT | archon_sources | Count total sources | ✅ Compatible |
| 44 | SELECT | archon_crawled_pages | Count total documents | ✅ Compatible |
| 49 | SELECT | archon_code_examples | Count total code examples | ✅ Compatible |
| 82 | SELECT | archon_sources | Get knowledge types | ✅ Compatible |
| 94 | SELECT | archon_sources | Get recent sources | ✅ Compatible |

### 9. **Embedding Service Operations**
**File**: `python/src/server/services/embeddings/embedding_service.py`

| Line | Operation | API/Service | Purpose | Status |
|------|-----------|-------------|---------|---------|
| 17-65 | Function | Dimension Mapping | Map models to dimensions | ✅ Compatible |
| 262 | API | OpenAI Embeddings | Create embeddings | ⚠️ **HARD-CODED 1536** |

## 🚨 Critical Issues Found

### **Issue 1: Legacy "embedding" Column References**
**Affected Files**: 
- `document_storage_service.py` (line 253, 268, 306)
- `code_storage_service.py` (line 756, 765, 783)

**Problem**: Code still references the removed `"embedding"` column instead of using dimension-specific columns (`embedding_768`, `embedding_1024`, `embedding_1536`, `embedding_3072`).

### **Issue 2: RPC Function Parameter Mismatch**
**Affected File**: `vector_search_service.py` (lines 91, 199, 275)

**Problem**: Vector search calls use generic `query_embedding` parameter, but the new multi-dimensional RPC functions expect dimension-specific parameters like `query_embedding_1536`.

### **Issue 3: Hard-coded Embedding Dimensions**
**Affected File**: `embedding_service.py` (line 262)

**Problem**: Embedding creation always requests 1536 dimensions instead of model-appropriate dimensions.

## ✅ Database Connection Patterns

### **Primary Connection**
```python
# File: client_manager.py
def get_supabase_client() -> Client:
    url = os.getenv("SUPABASE_URL")
    service_key = os.getenv("SUPABASE_SERVICE_KEY")
    return create_client(url, service_key)
```

### **Usage Pattern**
All services use: `supabase_client = get_supabase_client()`

## 📊 Database Schema Overview

### **Current Multi-Dimensional Vector Schema**
```sql
-- archon_crawled_pages & archon_code_examples tables
embedding_768   VECTOR(768)   -- text-embedding-3-small (reduced)
embedding_1024  VECTOR(1024)  -- Custom models  
embedding_1536  VECTOR(1536)  -- text-embedding-3-small, ada-002 (default)
embedding_3072  VECTOR(3072)  -- text-embedding-3-large
```

### **Updated RPC Functions**
```sql
-- Multi-dimensional vector search functions
match_archon_crawled_pages(
    query_embedding_768 VECTOR(768) DEFAULT NULL,
    query_embedding_1024 VECTOR(1024) DEFAULT NULL, 
    query_embedding_1536 VECTOR(1536) DEFAULT NULL,
    query_embedding_3072 VECTOR(3072) DEFAULT NULL,
    match_count INT DEFAULT 10,
    filter JSONB DEFAULT '{}'::jsonb,
    source_filter TEXT DEFAULT NULL
)

match_archon_code_examples(
    -- Same parameters as above
)
```

## 🔧 Required Fixes Summary

1. **Update Storage Services**: Replace hardcoded `"embedding"` keys with dynamic column selection
2. **Fix RPC Calls**: Update vector search to use dimension-specific parameter names  
3. **Dynamic Embedding Creation**: Remove 1536-dimension hardcoding
4. **Add Error Handling**: Handle unsupported dimensions gracefully

## ⚠️ Migration Status

**Schema Migration**: ✅ **COMPLETE** - Database properly supports multi-dimensional vectors  
**Application Code**: ❌ **INCOMPLETE** - Still has legacy column references  
**Vector Search**: ❌ **INCOMPLETE** - RPC parameter names need updating  
**Embedding Service**: ⚠️ **PARTIAL** - Utility functions exist but not fully utilized

---

The database review shows that while the schema migration was successful, several critical application code updates are needed to fully support the new multi-dimensional vector architecture. The specific fixes identified above will complete the migration and ensure compatibility with multiple embedding models.