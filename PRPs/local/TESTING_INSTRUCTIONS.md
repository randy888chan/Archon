# Testing Instructions for Zero Embedding Fix

## 1. Database Verification

### Check for Existing Zero Embeddings

1. **Go to your Supabase Dashboard**
   - Open https://supabase.com/dashboard
   - Select your project

2. **Open SQL Editor** (left sidebar)

3. **Run this query to count zero embeddings:**
```sql
SELECT COUNT(*) as zero_embedding_count 
FROM archon_crawled_pages 
WHERE embedding[1] = 0.0 
  AND embedding[2] = 0.0 
  AND embedding[3] = 0.0;
```

4. **If count > 0, see which sources are affected:**
```sql
SELECT DISTINCT source_id, COUNT(*) as affected_docs
FROM archon_crawled_pages 
WHERE embedding[1] = 0.0 
  AND embedding[2] = 0.0 
  AND embedding[3] = 0.0
GROUP BY source_id;
```

5. **Check code examples table as well:**
```sql
SELECT COUNT(*) as zero_code_examples
FROM archon_code_examples 
WHERE embedding[1] = 0.0 
  AND embedding[2] = 0.0 
  AND embedding[3] = 0.0;
```

6. **Optional - Clean up zero embeddings:**
```sql
-- First, see what you're about to delete
SELECT id, source_id, url, chunk_number 
FROM archon_crawled_pages 
WHERE embedding[1] = 0.0 
  AND embedding[2] = 0.0 
  AND embedding[3] = 0.0
LIMIT 10;

-- If you want to delete them from crawled pages
DELETE FROM archon_crawled_pages 
WHERE embedding[1] = 0.0 
  AND embedding[2] = 0.0 
  AND embedding[3] = 0.0;

-- Also clean code examples if needed
DELETE FROM archon_code_examples 
WHERE embedding[1] = 0.0 
  AND embedding[2] = 0.0 
  AND embedding[3] = 0.0;
```

## 2. Functional Testing

### Test 1: Verify No New Zero Embeddings Created

1. **Set an invalid API key in the UI Settings**
   - Go to http://localhost:3737
   - Navigate to Settings
   - Set OpenAI API Key to "invalid-test-key"

2. **Attempt to crawl a website**
   - Go to Knowledge Base page
   - Try to crawl any URL

3. **Expected Result:**
   - Crawl should fail with clear error about invalid API key
   - No documents should be stored

4. **Verify in database:**
```sql
-- Check no new zero embeddings were created
SELECT COUNT(*) as new_zeros
FROM archon_crawled_pages 
WHERE embedding[1] = 0.0 
  AND embedding[2] = 0.0 
  AND embedding[3] = 0.0
  AND created_at > NOW() - INTERVAL '5 minutes';
-- Should return 0
```

### Test 2: Verify Successful Crawl Works

1. **Set a valid API key**
   - Update Settings with your actual OpenAI API key

2. **Crawl a small website**
   - Try https://example.com or any small site

3. **Expected Result:**
   - Crawl completes successfully
   - Documents are stored with valid embeddings

4. **Verify embeddings are not zero:**
```sql
-- Check recent documents have valid embeddings
SELECT id, url, 
       embedding[1] as first_val,
       embedding[2] as second_val,
       embedding[3] as third_val
FROM archon_crawled_pages 
WHERE created_at > NOW() - INTERVAL '5 minutes'
LIMIT 5;
-- Should show non-zero values
```

### Test 3: Search Functionality

1. **After successful crawl, test search**
   - Search for content from the crawled site

2. **Expected Result:**
   - Search returns relevant results
   - No interference from zero vectors

## 3. Unit Test Verification

Run the unit tests to ensure the fix works correctly:

```bash
# From the python directory
cd python
uv run pytest tests/test_embedding_service_no_zeros.py -v

# Expected output:
# ✅ 10 tests should pass
# - 6 core functionality tests
# - 4 EmbeddingBatchResult tests
```

## 4. Docker Service Health Check

Verify all services are running properly:

```bash
# Check service status
docker-compose ps

# All services should show as "Up" and healthy

# Check server logs for errors
docker-compose logs archon-server | tail -50

# Should see:
# "✅ Credentials initialized"
# "🎉 Archon backend started successfully!"
```

## 5. Manual API Test (Optional)

Test the embedding service directly:

```bash
# Test with invalid key (should fail clearly)
curl -X POST http://localhost:8181/api/test/embedding \
  -H "Content-Type: application/json" \
  -d '{"text": "test", "api_key": "invalid"}'

# Expected: Error response, not success with zeros
```

## Success Criteria

✅ **No new zero embeddings** can be created  
✅ **Existing zero embeddings** identified and can be cleaned  
✅ **Clear error messages** when embeddings fail  
✅ **Normal operations** continue to work  
✅ **Unit tests** all pass  

## Troubleshooting

If you encounter issues:

1. **Services not starting:** 
   ```bash
   docker-compose down
   docker-compose up --build -d
   ```

2. **Database connection issues:**
   - Check your `.env` file has correct Supabase credentials
   - Verify Supabase project is active

3. **API key issues:**
   - Ensure OpenAI API key is valid
   - Check quota/billing on OpenAI dashboard

## Notes

- The fix prevents NEW zero embeddings from being created
- Existing zero embeddings need manual cleanup via SQL
- The system now follows "fail fast and loud" principle for data integrity