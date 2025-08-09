# Testing Plan for fix/zero-embedding-fallback Branch

## Branch Purpose
Remove zero-embedding fallbacks and implement proper error handling for embedding failures to maintain data integrity.

## Changes Made
1. **embedding_exceptions.py**: New custom exception classes for different embedding failure scenarios
2. **embedding_service.py**: Removed zero-embedding fallbacks, now raises proper exceptions
3. **document_storage_service.py**: Added try-catch to handle embedding failures gracefully
4. **document_storage_sync.py**: Updated to skip batches with embedding failures
5. **test_embedding_service_no_zeros.py**: Tests to verify no zero embeddings are returned

## Testing Checklist

### 1. Basic Functionality Tests
- [ ] **Normal crawl with valid API key**: Should complete successfully
  ```bash
  # Test with a simple URL
  curl -X POST http://localhost:8181/api/crawl \
    -H "Content-Type: application/json" \
    -d '{"url": "https://example.com", "max_depth": 1}'
  ```

- [ ] **Crawl without OpenAI API key**: Should complete but skip embedding generation
  ```bash
  # Remove API key temporarily
  unset OPENAI_API_KEY
  # Try crawling - should skip batches but not crash
  ```

- [ ] **Crawl with invalid API key**: Should handle authentication errors gracefully
  ```bash
  # Set invalid key
  export OPENAI_API_KEY="sk-invalid-key-12345"
  # Try crawling - should skip batches but continue
  ```

### 2. Error Handling Tests

- [ ] **Rate limit simulation**: 
  - Crawl a large site to trigger rate limits
  - Verify crawl continues with skipped batches
  - Check logs for "Skipping batch X due to embedding failure"

- [ ] **Quota exhaustion handling**:
  - Use an API key with exhausted quota
  - Verify proper error messages in logs
  - Ensure crawl doesn't store corrupted data

- [ ] **Network failure resilience**:
  - Disconnect network during embedding generation
  - Verify crawl handles timeout gracefully
  - Check that partial data is still saved

### 3. Data Integrity Tests

- [ ] **No zero embeddings in database**:
  ```sql
  -- Check for zero embeddings (should return 0 rows)
  SELECT COUNT(*) FROM archon_crawled_pages 
  WHERE embedding[1] = 0 AND embedding[2] = 0 AND embedding[3] = 0;
  
  SELECT COUNT(*) FROM archon_code_examples
  WHERE embedding[1] = 0 AND embedding[2] = 0 AND embedding[3] = 0;
  ```

- [ ] **Verify skipped batches aren't stored**:
  ```sql
  -- Compare crawled URLs vs stored documents
  SELECT COUNT(DISTINCT url) as crawled_count FROM archon_crawled_pages;
  -- Should be less than total pages if some batches were skipped
  ```

### 4. Service Integration Tests

- [ ] **Vector search functionality**:
  ```bash
  # Test RAG search still works with stored embeddings
  curl -X POST http://localhost:8181/api/search \
    -H "Content-Type: application/json" \
    -d '{"query": "test query", "match_count": 5}'
  ```

- [ ] **Code example extraction**:
  - Verify code examples are still extracted and stored
  - Check that code example embeddings work properly

### 5. Regression Tests

- [ ] **Existing crawls still accessible**: Previously crawled content should remain searchable
- [ ] **Re-crawling works**: Can refresh existing knowledge items
- [ ] **Document upload works**: PDF/DOCX upload should handle embedding failures

### 6. Performance Tests

- [ ] **Large batch processing**: Crawl site with 100+ pages
- [ ] **Concurrent crawls**: Start multiple crawls simultaneously
- [ ] **Memory usage**: Monitor memory during large crawls

## Test Scenarios by Component

### Document Storage Service
1. **Happy path**: All embeddings succeed → All documents stored
2. **Partial failure**: Some batches fail → Only successful batches stored
3. **Complete failure**: No API key → No documents stored, crawl completes with warning

### Embedding Service
1. **Sync from async context**: Should raise `EmbeddingAsyncContextError`
2. **API failures**: Should raise appropriate exception types
3. **No fallback to zeros**: Never return zero vectors

### Vector Search
1. **Search with embeddings**: Should work normally
2. **Search after partial crawl**: Should return available results
3. **No zero-vector matches**: Shouldn't match against failed embeddings

## Monitoring During Tests

Watch for these log patterns:

### Success Indicators
```
✅ "Successfully created embeddings for batch"
✅ "Inserted batch X of Y documents"
✅ "Crawl completed successfully"
```

### Expected Warnings (Not Failures)
```
⚠️ "Failed to create embeddings for batch X"
⚠️ "Skipping batch X due to embedding failure"
⚠️ "OpenAI API key not configured"
```

### Error Indicators (Should Not Appear)
```
❌ "Crawl failed"
❌ "Fatal error"
❌ "Unhandled exception"
❌ Zero embeddings stored in database
```

## Rollback Plan

If issues are found:
1. Document the specific failure scenario
2. Check if it's a regression from main branch
3. Determine if fix is needed in:
   - Error handling logic
   - Service integration points
   - Test coverage

## Success Criteria

The branch is ready for merge when:
1. ✅ All normal crawls complete successfully
2. ✅ Embedding failures don't crash the crawl
3. ✅ No zero embeddings in the database
4. ✅ Partial data is better than no data
5. ✅ Clear error messages in logs
6. ✅ Vector search still works
7. ✅ No regressions from main branch

## Commands for Quick Testing

```bash
# 1. Run unit tests
cd python && uv run pytest tests/test_embedding_service_no_zeros.py -v

# 2. Test crawl with small site
curl -X POST http://localhost:8181/api/crawl \
  -H "Content-Type: application/json" \
  -d '{"url": "https://httpbin.org", "max_depth": 1}'

# 3. Check for zero embeddings
docker exec archon-db psql -U postgres -d postgres -c \
  "SELECT COUNT(*) FROM archon_crawled_pages WHERE embedding[1] = 0;"

# 4. Monitor logs during crawl
docker-compose logs -f archon-server | grep -E "embedding|batch|skip"
```