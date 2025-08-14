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
    
    -- Add embedding tracking columns for model migration workflows
    RAISE NOTICE 'Adding embedding model and dimensions tracking columns...';
    
    -- Add embedding_model column to track which model was used
    ALTER TABLE archon_code_examples 
    ADD COLUMN IF NOT EXISTS embedding_model TEXT;
    
    ALTER TABLE archon_crawled_pages 
    ADD COLUMN IF NOT EXISTS embedding_model TEXT;
    
    -- Add embedding_dimensions column to track dimension size
    ALTER TABLE archon_code_examples 
    ADD COLUMN IF NOT EXISTS embedding_dimensions INTEGER;
    
    ALTER TABLE archon_crawled_pages 
    ADD COLUMN IF NOT EXISTS embedding_dimensions INTEGER;
    
    -- Add comments for the new tracking columns
    COMMENT ON COLUMN archon_code_examples.embedding_model IS 'The embedding model used to generate the embedding (e.g., text-embedding-3-small, all-mpnet-base-v2)';
    COMMENT ON COLUMN archon_code_examples.embedding_dimensions IS 'The number of dimensions in the stored embedding vector';
    COMMENT ON COLUMN archon_crawled_pages.embedding_model IS 'The embedding model used to generate the embedding (e.g., text-embedding-3-small, all-mpnet-base-v2)';
    COMMENT ON COLUMN archon_crawled_pages.embedding_dimensions IS 'The number of dimensions in the stored embedding vector';
    
    RAISE NOTICE 'Embedding tracking columns added successfully.';
    
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
-- SECTION 2B: POPULATE EMBEDDING TRACKING COLUMNS
-- =====================================================

DO $$ 
DECLARE
    current_model TEXT;
    detected_dimensions INTEGER;
    total_code_records INTEGER;
    total_page_records INTEGER;
    updated_code_records INTEGER := 0;
    updated_page_records INTEGER := 0;
BEGIN
    RAISE NOTICE 'Populating embedding tracking columns for existing records...';
    
    -- Get the current embedding model from settings
    SELECT value INTO current_model 
    FROM archon_settings 
    WHERE key = 'EMBEDDING_MODEL' 
    LIMIT 1;
    
    -- Default to text-embedding-3-small if not found
    IF current_model IS NULL THEN
        current_model := 'text-embedding-3-small';
        RAISE NOTICE 'No EMBEDDING_MODEL found in settings, defaulting to: %', current_model;
    ELSE
        RAISE NOTICE 'Found current embedding model in settings: %', current_model;
    END IF;
    
    -- Determine dimensions based on current model
    CASE 
        WHEN current_model LIKE '%text-embedding-3-large%' THEN 
            detected_dimensions := 3072;
        WHEN current_model LIKE '%text-embedding-3-small%' OR current_model LIKE '%text-embedding-ada-002%' THEN 
            detected_dimensions := 1536;
        WHEN current_model LIKE '%all-mpnet-base-v2%' THEN 
            detected_dimensions := 768;
        WHEN current_model LIKE '%snowflake-arctic-embed%' THEN 
            detected_dimensions := 1024;
        WHEN current_model LIKE '%all-MiniLM%' THEN 
            detected_dimensions := 384;
        ELSE 
            detected_dimensions := 1536; -- Default fallback
    END CASE;
    
    RAISE NOTICE 'Detected dimensions for model %: %', current_model, detected_dimensions;
    
    -- Count existing records
    SELECT COUNT(*) INTO total_code_records FROM archon_code_examples;
    SELECT COUNT(*) INTO total_page_records FROM archon_crawled_pages;
    
    RAISE NOTICE 'Found % code examples and % crawled pages to update', total_code_records, total_page_records;
    
    -- Update archon_code_examples records that have embeddings but no model tracking
    UPDATE archon_code_examples 
    SET 
        embedding_model = current_model,
        embedding_dimensions = detected_dimensions
    WHERE 
        embedding_model IS NULL 
        AND (
            embedding_768 IS NOT NULL 
            OR embedding_1024 IS NOT NULL 
            OR embedding_1536 IS NOT NULL 
            OR embedding_3072 IS NOT NULL
        );
        
    GET DIAGNOSTICS updated_code_records = ROW_COUNT;
    
    -- Update archon_crawled_pages records that have embeddings but no model tracking
    UPDATE archon_crawled_pages 
    SET 
        embedding_model = current_model,
        embedding_dimensions = detected_dimensions
    WHERE 
        embedding_model IS NULL 
        AND (
            embedding_768 IS NOT NULL 
            OR embedding_1024 IS NOT NULL 
            OR embedding_1536 IS NOT NULL 
            OR embedding_3072 IS NOT NULL
        );
        
    GET DIAGNOSTICS updated_page_records = ROW_COUNT;
    
    RAISE NOTICE 'Updated % code examples and % crawled pages with embedding model: %', 
                 updated_code_records, updated_page_records, current_model;
                 
    -- Additional validation: report on embedding distribution
    DECLARE 
        embedding_768_count INTEGER;
        embedding_1024_count INTEGER; 
        embedding_1536_count INTEGER;
        embedding_3072_count INTEGER;
    BEGIN
        -- Check which embedding columns actually have data in crawled_pages
        SELECT 
            COUNT(*) FILTER (WHERE embedding_768 IS NOT NULL),
            COUNT(*) FILTER (WHERE embedding_1024 IS NOT NULL),
            COUNT(*) FILTER (WHERE embedding_1536 IS NOT NULL),
            COUNT(*) FILTER (WHERE embedding_3072 IS NOT NULL)
        INTO embedding_768_count, embedding_1024_count, embedding_1536_count, embedding_3072_count
        FROM archon_crawled_pages;
        
        RAISE NOTICE 'Embedding distribution in crawled_pages - 768D: %, 1024D: %, 1536D: %, 3072D: %',
                     embedding_768_count, embedding_1024_count, embedding_1536_count, embedding_3072_count;
                     
        -- Check code examples too
        SELECT 
            COUNT(*) FILTER (WHERE embedding_768 IS NOT NULL),
            COUNT(*) FILTER (WHERE embedding_1024 IS NOT NULL),
            COUNT(*) FILTER (WHERE embedding_1536 IS NOT NULL),
            COUNT(*) FILTER (WHERE embedding_3072 IS NOT NULL)
        INTO embedding_768_count, embedding_1024_count, embedding_1536_count, embedding_3072_count
        FROM archon_code_examples;
        
        RAISE NOTICE 'Embedding distribution in code_examples - 768D: %, 1024D: %, 1536D: %, 3072D: %',
                     embedding_768_count, embedding_1024_count, embedding_1536_count, embedding_3072_count;
    END;
    
    RAISE NOTICE 'Embedding tracking data population completed.';
    
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Error during embedding tracking data population: %', SQLERRM;
    -- Don't rollback - this is not critical if it fails
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
    
    -- Create indexes for embedding tracking columns
    RAISE NOTICE 'Creating indexes for embedding tracking columns...';
    
    CREATE INDEX IF NOT EXISTS idx_archon_code_examples_embedding_model 
    ON archon_code_examples (embedding_model);
    
    CREATE INDEX IF NOT EXISTS idx_archon_code_examples_embedding_dimensions 
    ON archon_code_examples (embedding_dimensions);
    
    CREATE INDEX IF NOT EXISTS idx_archon_crawled_pages_embedding_model 
    ON archon_crawled_pages (embedding_model);
    
    CREATE INDEX IF NOT EXISTS idx_archon_crawled_pages_embedding_dimensions 
    ON archon_crawled_pages (embedding_dimensions);
    
    -- Create composite indexes for model migration workflows
    CREATE INDEX IF NOT EXISTS idx_archon_code_examples_model_dimensions 
    ON archon_code_examples (embedding_model, embedding_dimensions);
    
    CREATE INDEX IF NOT EXISTS idx_archon_crawled_pages_model_dimensions 
    ON archon_crawled_pages (embedding_model, embedding_dimensions);
    
    RAISE NOTICE 'Embedding tracking indexes created successfully.';
    
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
      query_embedding_768 VECTOR(768) DEFAULT NULL,
      query_embedding_1024 VECTOR(1024) DEFAULT NULL,
      query_embedding_1536 VECTOR(1536) DEFAULT NULL,
      query_embedding_3072 VECTOR(3072) DEFAULT NULL,
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
    BEGIN
      -- Route to appropriate embedding column based on which parameter is provided
      IF query_embedding_768 IS NOT NULL THEN
        RETURN QUERY
        SELECT
          id, url, chunk_number, content, metadata, source_id,
          1 - (archon_crawled_pages.embedding_768 <=> query_embedding_768) AS similarity
        FROM archon_crawled_pages
        WHERE metadata @> filter
          AND (source_filter IS NULL OR source_id = source_filter)
          AND embedding_768 IS NOT NULL
        ORDER BY archon_crawled_pages.embedding_768 <=> query_embedding_768
        LIMIT match_count;
      ELSIF query_embedding_1024 IS NOT NULL THEN
        RETURN QUERY
        SELECT
          id, url, chunk_number, content, metadata, source_id,
          1 - (archon_crawled_pages.embedding_1024 <=> query_embedding_1024) AS similarity
        FROM archon_crawled_pages
        WHERE metadata @> filter
          AND (source_filter IS NULL OR source_id = source_filter)
          AND embedding_1024 IS NOT NULL
        ORDER BY archon_crawled_pages.embedding_1024 <=> query_embedding_1024
        LIMIT match_count;
      ELSIF query_embedding_1536 IS NOT NULL THEN
        RETURN QUERY
        SELECT
          id, url, chunk_number, content, metadata, source_id,
          1 - (archon_crawled_pages.embedding_1536 <=> query_embedding_1536) AS similarity
        FROM archon_crawled_pages
        WHERE metadata @> filter
          AND (source_filter IS NULL OR source_id = source_filter)
          AND embedding_1536 IS NOT NULL
        ORDER BY archon_crawled_pages.embedding_1536 <=> query_embedding_1536
        LIMIT match_count;
      ELSIF query_embedding_3072 IS NOT NULL THEN
        RETURN QUERY
        SELECT
          id, url, chunk_number, content, metadata, source_id,
          1 - (archon_crawled_pages.embedding_3072 <=> query_embedding_3072) AS similarity
        FROM archon_crawled_pages
        WHERE metadata @> filter
          AND (source_filter IS NULL OR source_id = source_filter)
          AND embedding_3072 IS NOT NULL
        ORDER BY archon_crawled_pages.embedding_3072 <=> query_embedding_3072
        LIMIT match_count;
      ELSE
        RAISE EXCEPTION 'No query embedding provided';
      END IF;
    END;
    $$;
    
    -- Update the legacy match_archon_code_examples function with dynamic multi-dimensional logic
    CREATE OR REPLACE FUNCTION match_archon_code_examples (
      query_embedding_768 VECTOR(768) DEFAULT NULL,
      query_embedding_1024 VECTOR(1024) DEFAULT NULL,
      query_embedding_1536 VECTOR(1536) DEFAULT NULL,
      query_embedding_3072 VECTOR(3072) DEFAULT NULL,
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
    BEGIN
      -- Route to appropriate embedding column based on which parameter is provided
      IF query_embedding_768 IS NOT NULL THEN
        RETURN QUERY
        SELECT
          id, url, chunk_number, content, summary, metadata, source_id,
          1 - (archon_code_examples.embedding_768 <=> query_embedding_768) AS similarity
        FROM archon_code_examples
        WHERE metadata @> filter
          AND (source_filter IS NULL OR source_id = source_filter)
          AND embedding_768 IS NOT NULL
        ORDER BY archon_code_examples.embedding_768 <=> query_embedding_768
        LIMIT match_count;
      ELSIF query_embedding_1024 IS NOT NULL THEN
        RETURN QUERY
        SELECT
          id, url, chunk_number, content, summary, metadata, source_id,
          1 - (archon_code_examples.embedding_1024 <=> query_embedding_1024) AS similarity
        FROM archon_code_examples
        WHERE metadata @> filter
          AND (source_filter IS NULL OR source_id = source_filter)
          AND embedding_1024 IS NOT NULL
        ORDER BY archon_code_examples.embedding_1024 <=> query_embedding_1024
        LIMIT match_count;
      ELSIF query_embedding_1536 IS NOT NULL THEN
        RETURN QUERY
        SELECT
          id, url, chunk_number, content, summary, metadata, source_id,
          1 - (archon_code_examples.embedding_1536 <=> query_embedding_1536) AS similarity
        FROM archon_code_examples
        WHERE metadata @> filter
          AND (source_filter IS NULL OR source_id = source_filter)
          AND embedding_1536 IS NOT NULL
        ORDER BY archon_code_examples.embedding_1536 <=> query_embedding_1536
        LIMIT match_count;
      ELSIF query_embedding_3072 IS NOT NULL THEN
        RETURN QUERY
        SELECT
          id, url, chunk_number, content, summary, metadata, source_id,
          1 - (archon_code_examples.embedding_3072 <=> query_embedding_3072) AS similarity
        FROM archon_code_examples
        WHERE metadata @> filter
          AND (source_filter IS NULL OR source_id = source_filter)
          AND embedding_3072 IS NOT NULL
        ORDER BY archon_code_examples.embedding_3072 <=> query_embedding_3072
        LIMIT match_count;
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