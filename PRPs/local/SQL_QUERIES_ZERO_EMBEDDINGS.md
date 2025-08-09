# SQL Queries for Detecting Zero Embeddings

Since the `vector` type in PostgreSQL doesn't support array subscripting, here are working queries to detect zero embeddings in your Supabase database.

## 1. Simple Pattern Check (Recommended - Start Here)

### Check for zero pattern at beginning of vector
```sql
-- Check if embeddings start with zeros (good indicator of zero vector)
SELECT COUNT(*) as potential_zero_embeddings
FROM archon_crawled_pages
WHERE LEFT(embedding::text, 50) = '[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,';
```

### Check code examples table
```sql
SELECT COUNT(*) as potential_zero_code_examples
FROM archon_code_examples
WHERE LEFT(embedding::text, 50) = '[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,';
```

## 2. View Sample Embeddings (Visual Inspection)

### Look at actual embedding values
```sql
-- View first 100 characters of embeddings to see if they're zeros
SELECT id, url, source_id, 
       LEFT(embedding::text, 100) as embedding_preview
FROM archon_crawled_pages
LIMIT 10;
```

### Check recently added documents
```sql
-- Check embeddings from recent crawls
SELECT id, url, 
       LEFT(embedding::text, 100) as embedding_preview,
       created_at
FROM archon_crawled_pages
WHERE created_at > NOW() - INTERVAL '1 day'
ORDER BY created_at DESC
LIMIT 10;
```

## 3. Check for NULL Embeddings

### Find documents without embeddings
```sql
-- NULL embeddings are also problematic
SELECT COUNT(*) as null_embeddings
FROM archon_crawled_pages
WHERE embedding IS NULL;
```

### Find which sources have NULL embeddings
```sql
SELECT source_id, COUNT(*) as null_count
FROM archon_crawled_pages
WHERE embedding IS NULL
GROUP BY source_id;
```

## 4. Vector Distance Check (Advanced)

### Find vectors very close to zero vector
```sql
-- This creates a zero vector and finds embeddings very close to it
-- Distance < 0.01 means essentially a zero vector
WITH zero_vector AS (
  SELECT ('['|| string_agg('0', ',') ||']')::vector as vec
  FROM generate_series(1, 1536)
)
SELECT COUNT(*) as zero_embeddings
FROM archon_crawled_pages, zero_vector
WHERE embedding <-> zero_vector.vec < 0.01;
```

## 5. Identify Affected Sources

### Find which sources have potential zero embeddings
```sql
SELECT source_id, COUNT(*) as affected_docs
FROM archon_crawled_pages
WHERE LEFT(embedding::text, 50) = '[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,'
GROUP BY source_id
ORDER BY affected_docs DESC;
```

### Get details of affected documents
```sql
SELECT id, url, source_id, chunk_number, created_at
FROM archon_crawled_pages
WHERE LEFT(embedding::text, 50) = '[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,'
LIMIT 20;
```

## 6. Cleanup Queries (After Verification)

### Delete zero embeddings from crawled pages
```sql
-- First, verify what you're deleting
SELECT COUNT(*) 
FROM archon_crawled_pages
WHERE LEFT(embedding::text, 50) = '[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,';

-- Then delete if confirmed
DELETE FROM archon_crawled_pages
WHERE LEFT(embedding::text, 50) = '[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,';
```

### Delete zero embeddings from code examples
```sql
-- First, verify what you're deleting
SELECT COUNT(*) 
FROM archon_code_examples
WHERE LEFT(embedding::text, 50) = '[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,';

-- Then delete if confirmed
DELETE FROM archon_code_examples
WHERE LEFT(embedding::text, 50) = '[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,';
```

### Delete NULL embeddings
```sql
-- Delete documents with NULL embeddings
DELETE FROM archon_crawled_pages
WHERE embedding IS NULL;

DELETE FROM archon_code_examples
WHERE embedding IS NULL;
```

## 7. Verification After Fix

### Check no new zero embeddings after deployment
```sql
-- Run this after deploying the fix and trying a crawl
SELECT COUNT(*) as new_zero_embeddings
FROM archon_crawled_pages
WHERE LEFT(embedding::text, 50) = '[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,'
  AND created_at > NOW() - INTERVAL '10 minutes';
-- Should return 0
```

### Verify recent crawls have valid embeddings
```sql
-- Check that new documents have non-zero embeddings
SELECT id, url,
       LEFT(embedding::text, 100) as embedding_preview,
       created_at
FROM archon_crawled_pages
WHERE created_at > NOW() - INTERVAL '10 minutes'
ORDER BY created_at DESC
LIMIT 5;
-- Should show embeddings with varied non-zero values
```

## Usage Instructions

1. **Start with Query #1** (Simple Pattern Check) - This will quickly tell you if you have zero embeddings
2. **Use Query #2** (View Samples) - Visually confirm what the embeddings look like
3. **Run Query #3** (NULL check) - Check for missing embeddings
4. **If zeros found, use Query #5** - Identify which sources are affected
5. **After confirming, use Query #6** - Clean up the bad data
6. **After fix deployment, use Query #7** - Verify the fix is working

## Notes

- The `vector` type stores embeddings as a special PostgreSQL type
- We check for zeros by converting to text and pattern matching
- A zero embedding will show as `[0,0,0,0,0,0,...]` for all 1536 dimensions
- NULL embeddings indicate the embedding was never created
- After the fix, no new zero or NULL embeddings should appear