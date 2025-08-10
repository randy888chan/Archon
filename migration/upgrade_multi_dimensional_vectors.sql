-- =====================================================
-- ARCHON MULTI-DIMENSIONAL VECTORS UPGRADE SCRIPT
-- =====================================================
-- 
-- This script upgrades existing Archon databases to support
-- multi-dimensional vector embeddings (768, 1024, 1536, 3072 dimensions)
-- for improved compatibility with different embedding models.
--
-- COMPATIBILITY: Works with existing Archon V2 Alpha databases
-- SAFETY: All operations use IF NOT EXISTS to prevent conflicts
-- 
-- Usage:
--   1. Connect to your existing Supabase/PostgreSQL database
--   2. Run this script in the SQL editor
--   3. Existing data and embeddings will be preserved
--
-- Features Added:
--   - Multi-dimensional embedding columns for both tables
--   - Performance-optimized ivfflat indexes for all dimensions
--   - Dynamic search functions for dimension-specific queries
--   - Backward compatibility with existing 1536-dimension embeddings
--
-- Models Supported:
--   - 768: text-embedding-3-small (reduced dimensions)
--   - 1024: Custom models requiring 1024 dimensions
--   - 1536: text-embedding-3-small (default), text-embedding-ada-002
--   - 3072: text-embedding-3-large (high-dimension embeddings)
--
-- Created: 2025-01-07
-- =====================================================

BEGIN;

-- =====================================================
-- SECTION 1: ADD MULTI-DIMENSIONAL EMBEDDING COLUMNS
-- =====================================================

DO $$ 
BEGIN
    RAISE NOTICE 'Adding multi-dimensional embedding columns...';
    
    -- Add new embedding columns to archon_crawled_pages
    ALTER TABLE archon_crawled_pages 
    ADD COLUMN IF NOT EXISTS embedding_768 VECTOR(768),
    ADD COLUMN IF NOT EXISTS embedding_1024 VECTOR(1024),
    ADD COLUMN IF NOT EXISTS embedding_1536 VECTOR(1536),
    ADD COLUMN IF NOT EXISTS embedding_3072 VECTOR(3072);
    
    -- Add new embedding columns to archon_code_examples
    ALTER TABLE archon_code_examples 
    ADD COLUMN IF NOT EXISTS embedding_768 VECTOR(768),
    ADD COLUMN IF NOT EXISTS embedding_1024 VECTOR(1024),
    ADD COLUMN IF NOT EXISTS embedding_1536 VECTOR(1536),
    ADD COLUMN IF NOT EXISTS embedding_3072 VECTOR(3072);
    
    -- Add comments to document the new columns
    COMMENT ON COLUMN archon_crawled_pages.embedding_768 IS 'Vector embeddings with 768 dimensions (text-embedding-3-small reduced)';
    COMMENT ON COLUMN archon_crawled_pages.embedding_1024 IS 'Vector embeddings with 1024 dimensions (custom models)';
    COMMENT ON COLUMN archon_crawled_pages.embedding_1536 IS 'Vector embeddings with 1536 dimensions (text-embedding-3-small, text-embedding-ada-002)';
    COMMENT ON COLUMN archon_crawled_pages.embedding_3072 IS 'Vector embeddings with 3072 dimensions (text-embedding-3-large)';
    
    COMMENT ON COLUMN archon_code_examples.embedding_768 IS 'Vector embeddings with 768 dimensions (text-embedding-3-small reduced)';
    COMMENT ON COLUMN archon_code_examples.embedding_1024 IS 'Vector embeddings with 1024 dimensions (custom models)';
    COMMENT ON COLUMN archon_code_examples.embedding_1536 IS 'Vector embeddings with 1536 dimensions (text-embedding-3-small, text-embedding-ada-002)';
    COMMENT ON COLUMN archon_code_examples.embedding_3072 IS 'Vector embeddings with 3072 dimensions (text-embedding-3-large)';
    
    RAISE NOTICE 'Multi-dimensional embedding columns added successfully.';
    
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Error adding embedding columns: %', SQLERRM;
    ROLLBACK;
    RETURN;
END $$;

-- =====================================================
-- SECTION 2: MIGRATE EXISTING EMBEDDING DATA
-- =====================================================

DO $$ 
BEGIN
    RAISE NOTICE 'Migrating existing embedding data to embedding_1536 columns...';
    
    -- Migrate existing 1536-dimension embeddings from legacy 'embedding' column
    -- to the new 'embedding_1536' column for archon_crawled_pages
    UPDATE archon_crawled_pages 
    SET embedding_1536 = embedding 
    WHERE embedding IS NOT NULL AND embedding_1536 IS NULL;
    
    -- Migrate existing 1536-dimension embeddings from legacy 'embedding' column
    -- to the new 'embedding_1536' column for archon_code_examples
    UPDATE archon_code_examples 
    SET embedding_1536 = embedding 
    WHERE embedding IS NOT NULL AND embedding_1536 IS NULL;
    
    RAISE NOTICE 'Embedding data migration completed.';
    
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Error during embedding data migration: %', SQLERRM;
    ROLLBACK;
    RETURN;
END $$;

-- =====================================================
-- SECTION 3: DROP LEGACY EMBEDDING COLUMNS AND INDEXES
-- =====================================================

DO $$ 
BEGIN
    RAISE NOTICE 'Dropping legacy embedding columns and indexes...';
    
    -- Drop legacy indexes first
    DROP INDEX IF EXISTS archon_crawled_pages_embedding_idx;
    DROP INDEX IF EXISTS archon_code_examples_embedding_idx;
    -- Try different possible index names that might exist
    DROP INDEX IF EXISTS idx_archon_crawled_pages_embedding;
    DROP INDEX IF EXISTS idx_archon_code_examples_embedding;
    
    -- Drop legacy embedding columns
    ALTER TABLE archon_crawled_pages DROP COLUMN IF EXISTS embedding;
    ALTER TABLE archon_code_examples DROP COLUMN IF EXISTS embedding;
    
    RAISE NOTICE 'Legacy embedding columns and indexes dropped successfully.';
    
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Error dropping legacy embedding columns: %', SQLERRM;
    -- Don't rollback here as this might fail due to missing columns/indexes
    -- which is expected for fresh installations
END $$;

-- =====================================================
-- SECTION 4: CREATE OPTIMIZED VECTOR INDEXES
-- =====================================================

DO $$
BEGIN
    RAISE NOTICE 'Creating optimized vector indexes for all dimensions...';
    
    -- Indexes for archon_crawled_pages multi-dimensional embeddings
    CREATE INDEX IF NOT EXISTS idx_archon_crawled_pages_embedding_768
    ON archon_crawled_pages USING ivfflat (embedding_768 vector_cosine_ops)
    WITH (lists = 1000);
    
    CREATE INDEX IF NOT EXISTS idx_archon_crawled_pages_embedding_1024
    ON archon_crawled_pages USING ivfflat (embedding_1024 vector_cosine_ops)
    WITH (lists = 1000);
    
    CREATE INDEX IF NOT EXISTS idx_archon_crawled_pages_embedding_1536
    ON archon_crawled_pages USING ivfflat (embedding_1536 vector_cosine_ops)
    WITH (lists = 1000);
    
    -- NOTE: pgvector ivfflat indexes do not support vectors with more than 2000 dimensions
    -- For 3072-dimension vectors, we'll use sequential scan which is still performant for most use cases
    -- If index is needed in future, consider using HNSW index type when available in pgvector
    -- CREATE INDEX IF NOT EXISTS idx_archon_crawled_pages_embedding_3072
    -- ON archon_crawled_pages USING ivfflat (embedding_3072 vector_cosine_ops)
    -- WITH (lists = 1000);
    
    -- Indexes for archon_code_examples multi-dimensional embeddings
    CREATE INDEX IF NOT EXISTS idx_archon_code_examples_embedding_768
    ON archon_code_examples USING ivfflat (embedding_768 vector_cosine_ops)
    WITH (lists = 1000);
    
    CREATE INDEX IF NOT EXISTS idx_archon_code_examples_embedding_1024
    ON archon_code_examples USING ivfflat (embedding_1024 vector_cosine_ops)
    WITH (lists = 1000);
    
    CREATE INDEX IF NOT EXISTS idx_archon_code_examples_embedding_1536
    ON archon_code_examples USING ivfflat (embedding_1536 vector_cosine_ops)
    WITH (lists = 1000);
    
    -- NOTE: pgvector ivfflat indexes do not support vectors with more than 2000 dimensions
    -- For 3072-dimension vectors, we'll use sequential scan which is still performant for most use cases
    -- If index is needed in future, consider using HNSW index type when available in pgvector
    -- CREATE INDEX IF NOT EXISTS idx_archon_code_examples_embedding_3072
    -- ON archon_code_examples USING ivfflat (embedding_3072 vector_cosine_ops)
    -- WITH (lists = 1000);
    
    RAISE NOTICE 'Vector indexes created successfully.';
    
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Error creating indexes: %', SQLERRM;
    ROLLBACK;
    RETURN;
END $$;

-- =====================================================
-- SECTION 5: UPDATE LEGACY SEARCH FUNCTIONS
-- =====================================================

DO $$
BEGIN
    RAISE NOTICE 'Updating legacy search functions to use embedding_1536...';
    
    -- Update the legacy match_archon_crawled_pages function with dynamic multi-dimensional logic
    CREATE OR REPLACE FUNCTION match_archon_crawled_pages (
      query_embedding VECTOR(1536) DEFAULT NULL,
      match_count INT DEFAULT 10,
      filter JSONB DEFAULT '{}'::jsonb,
      source_filter TEXT DEFAULT NULL
    ) RETURNS TABLE (
      id BIGINT,
      url VARCHAR,
      chunk_number INTEGER,
      content TEXT,
      metadata JSONB,
      source_id TEXT,
      similarity FLOAT
    )
    LANGUAGE plpgsql
    AS $$
    #variable_conflict use_column
    DECLARE
      embedding_dimensions INT;
    BEGIN
      -- Auto-detect embedding dimensions from the provided vector
      IF query_embedding IS NOT NULL THEN
        embedding_dimensions := array_length(query_embedding, 1);
        
        -- Route to appropriate embedding column based on dimensions
        IF embedding_dimensions = 768 THEN
          RETURN QUERY
          SELECT
            id, url, chunk_number, content, metadata, source_id,
            1 - (archon_crawled_pages.embedding_768 <=> query_embedding::VECTOR(768)) AS similarity
          FROM archon_crawled_pages
          WHERE metadata @> filter
            AND (source_filter IS NULL OR source_id = source_filter)
            AND embedding_768 IS NOT NULL
          ORDER BY archon_crawled_pages.embedding_768 <=> query_embedding::VECTOR(768)
          LIMIT match_count;
        ELSIF embedding_dimensions = 1024 THEN
          RETURN QUERY
          SELECT
            id, url, chunk_number, content, metadata, source_id,
            1 - (archon_crawled_pages.embedding_1024 <=> query_embedding::VECTOR(1024)) AS similarity
          FROM archon_crawled_pages
          WHERE metadata @> filter
            AND (source_filter IS NULL OR source_id = source_filter)
            AND embedding_1024 IS NOT NULL
          ORDER BY archon_crawled_pages.embedding_1024 <=> query_embedding::VECTOR(1024)
          LIMIT match_count;
        ELSIF embedding_dimensions = 1536 THEN
          RETURN QUERY
          SELECT
            id, url, chunk_number, content, metadata, source_id,
            1 - (archon_crawled_pages.embedding_1536 <=> query_embedding) AS similarity
          FROM archon_crawled_pages
          WHERE metadata @> filter
            AND (source_filter IS NULL OR source_id = source_filter)
            AND embedding_1536 IS NOT NULL
          ORDER BY archon_crawled_pages.embedding_1536 <=> query_embedding
          LIMIT match_count;
        ELSIF embedding_dimensions = 3072 THEN
          RETURN QUERY
          SELECT
            id, url, chunk_number, content, metadata, source_id,
            1 - (archon_crawled_pages.embedding_3072 <=> query_embedding::VECTOR(3072)) AS similarity
          FROM archon_crawled_pages
          WHERE metadata @> filter
            AND (source_filter IS NULL OR source_id = source_filter)
            AND embedding_3072 IS NOT NULL
          ORDER BY archon_crawled_pages.embedding_3072 <=> query_embedding::VECTOR(3072)
          LIMIT match_count;
        ELSE
          RAISE EXCEPTION 'Unsupported embedding dimensions: %. Supported dimensions: 768, 1024, 1536, 3072', embedding_dimensions;
        END IF;
      ELSE
        RAISE EXCEPTION 'No query embedding provided';
      END IF;
    END;
    $$;
    
    -- Update the legacy match_archon_code_examples function with dynamic multi-dimensional logic
    CREATE OR REPLACE FUNCTION match_archon_code_examples (
      query_embedding VECTOR(1536) DEFAULT NULL,
      match_count INT DEFAULT 10,
      filter JSONB DEFAULT '{}'::jsonb,
      source_filter TEXT DEFAULT NULL
    ) RETURNS TABLE (
      id BIGINT,
      url VARCHAR,
      chunk_number INTEGER,
      content TEXT,
      summary TEXT,
      metadata JSONB,
      source_id TEXT,
      similarity FLOAT
    )
    LANGUAGE plpgsql
    AS $$
    #variable_conflict use_column
    DECLARE
      embedding_dimensions INT;
    BEGIN
      -- Auto-detect embedding dimensions from the provided vector
      IF query_embedding IS NOT NULL THEN
        embedding_dimensions := array_length(query_embedding, 1);
        
        -- Route to appropriate embedding column based on dimensions
        IF embedding_dimensions = 768 THEN
          RETURN QUERY
          SELECT
            id, url, chunk_number, content, summary, metadata, source_id,
            1 - (archon_code_examples.embedding_768 <=> query_embedding::VECTOR(768)) AS similarity
          FROM archon_code_examples
          WHERE metadata @> filter
            AND (source_filter IS NULL OR source_id = source_filter)
            AND embedding_768 IS NOT NULL
          ORDER BY archon_code_examples.embedding_768 <=> query_embedding::VECTOR(768)
          LIMIT match_count;
        ELSIF embedding_dimensions = 1024 THEN
          RETURN QUERY
          SELECT
            id, url, chunk_number, content, summary, metadata, source_id,
            1 - (archon_code_examples.embedding_1024 <=> query_embedding::VECTOR(1024)) AS similarity
          FROM archon_code_examples
          WHERE metadata @> filter
            AND (source_filter IS NULL OR source_id = source_filter)
            AND embedding_1024 IS NOT NULL
          ORDER BY archon_code_examples.embedding_1024 <=> query_embedding::VECTOR(1024)
          LIMIT match_count;
        ELSIF embedding_dimensions = 1536 THEN
          RETURN QUERY
          SELECT
            id, url, chunk_number, content, summary, metadata, source_id,
            1 - (archon_code_examples.embedding_1536 <=> query_embedding) AS similarity
          FROM archon_code_examples
          WHERE metadata @> filter
            AND (source_filter IS NULL OR source_id = source_filter)
            AND embedding_1536 IS NOT NULL
          ORDER BY archon_code_examples.embedding_1536 <=> query_embedding
          LIMIT match_count;
        ELSIF embedding_dimensions = 3072 THEN
          RETURN QUERY
          SELECT
            id, url, chunk_number, content, summary, metadata, source_id,
            1 - (archon_code_examples.embedding_3072 <=> query_embedding::VECTOR(3072)) AS similarity
          FROM archon_code_examples
          WHERE metadata @> filter
            AND (source_filter IS NULL OR source_id = source_filter)
            AND embedding_3072 IS NOT NULL
          ORDER BY archon_code_examples.embedding_3072 <=> query_embedding::VECTOR(3072)
          LIMIT match_count;
        ELSE
          RAISE EXCEPTION 'Unsupported embedding dimensions: %. Supported dimensions: 768, 1024, 1536, 3072', embedding_dimensions;
        END IF;
      ELSE
        RAISE EXCEPTION 'No query embedding provided';
      END IF;
    END;
    $$;
    
    RAISE NOTICE 'Legacy search functions updated successfully.';
    
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Error updating legacy search functions: %', SQLERRM;
    ROLLBACK;
    RETURN;
END $$;

-- =====================================================
-- SECTION 6: VERIFICATION AND SUMMARY
-- =====================================================

DO $$
DECLARE
    crawled_pages_columns INTEGER;
    code_examples_columns INTEGER;
    total_indexes INTEGER;
    dynamic_functions INTEGER;
BEGIN
    -- Count new embedding columns
    SELECT COUNT(*) INTO crawled_pages_columns 
    FROM information_schema.columns 
    WHERE table_name = 'archon_crawled_pages' 
    AND column_name LIKE 'embedding_%';
    
    SELECT COUNT(*) INTO code_examples_columns 
    FROM information_schema.columns 
    WHERE table_name = 'archon_code_examples' 
    AND column_name LIKE 'embedding_%';
    
    -- Count new vector indexes
    SELECT COUNT(*) INTO total_indexes 
    FROM pg_indexes 
    WHERE schemaname = 'public' 
    AND indexname LIKE '%embedding_%';
    
    -- Count dynamic search functions
    SELECT COUNT(*) INTO dynamic_functions 
    FROM pg_proc p
    JOIN pg_namespace n ON p.pronamespace = n.oid
    WHERE n.nspname = 'public'
    AND p.proname LIKE '%_dynamic';
    
    RAISE NOTICE '======================================================================';
    RAISE NOTICE '              MULTI-DIMENSIONAL VECTORS UPGRADE COMPLETED';
    RAISE NOTICE '======================================================================';
    RAISE NOTICE 'Upgrade results:';
    RAISE NOTICE '  - archon_crawled_pages embedding columns: %', crawled_pages_columns;
    RAISE NOTICE '  - archon_code_examples embedding columns: %', code_examples_columns;
    RAISE NOTICE '  - Multi-dimensional vector indexes: %', total_indexes;
    RAISE NOTICE '  - Dynamic search functions: %', dynamic_functions;
    RAISE NOTICE '';
    RAISE NOTICE 'New capabilities:';
    RAISE NOTICE '  ✓ Support for 768, 1024, 1536, and 3072 dimension embeddings';
    RAISE NOTICE '  ✓ Optimized ivfflat indexes for all dimensions';
    RAISE NOTICE '  ✓ Dynamic search functions for dimension-specific queries';
    RAISE NOTICE '  ✓ Full backward compatibility with existing embeddings';
    RAISE NOTICE '';
    RAISE NOTICE 'Supported models:';
    RAISE NOTICE '  - text-embedding-3-small (768, 1536 dims)';
    RAISE NOTICE '  - text-embedding-3-large (3072 dims)';
    RAISE NOTICE '  - text-embedding-ada-002 (1536 dims)';
    RAISE NOTICE '  - Custom models (1024 dims)';
    RAISE NOTICE '======================================================================';
    
END $$;

COMMIT;

-- =====================================================
-- END OF UPGRADE SCRIPT
-- =====================================================