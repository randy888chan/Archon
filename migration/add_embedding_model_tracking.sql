-- =====================================================
-- Add Embedding Model and Dimension Tracking
-- =====================================================
-- This migration adds fields to track which embedding model
-- and dimension was used for each vector in the knowledge base
-- =====================================================

-- Add tracking columns to archon_crawled_pages table
ALTER TABLE archon_crawled_pages 
ADD COLUMN IF NOT EXISTS embedding_model TEXT,
ADD COLUMN IF NOT EXISTS embedding_dimensions INTEGER;

-- Add tracking columns to archon_code_examples table  
ALTER TABLE archon_code_examples
ADD COLUMN IF NOT EXISTS embedding_model TEXT,
ADD COLUMN IF NOT EXISTS embedding_dimensions INTEGER;

-- Create indexes for efficient querying by model and dimensions
CREATE INDEX IF NOT EXISTS idx_archon_crawled_pages_embedding_model 
ON archon_crawled_pages(embedding_model);

CREATE INDEX IF NOT EXISTS idx_archon_crawled_pages_embedding_dimensions
ON archon_crawled_pages(embedding_dimensions);

CREATE INDEX IF NOT EXISTS idx_archon_code_examples_embedding_model
ON archon_code_examples(embedding_model);

CREATE INDEX IF NOT EXISTS idx_archon_code_examples_embedding_dimensions
ON archon_code_examples(embedding_dimensions);

-- Add comments to document the new columns
COMMENT ON COLUMN archon_crawled_pages.embedding_model IS 'Name of the embedding model used to generate the vector (e.g., text-embedding-3-small, nomic-embed-text)';
COMMENT ON COLUMN archon_crawled_pages.embedding_dimensions IS 'Number of dimensions in the embedding vector (768, 1024, 1536, 3072)';
COMMENT ON COLUMN archon_code_examples.embedding_model IS 'Name of the embedding model used to generate the vector (e.g., text-embedding-3-small, nomic-embed-text)';
COMMENT ON COLUMN archon_code_examples.embedding_dimensions IS 'Number of dimensions in the embedding vector (768, 1024, 1536, 3072)';

-- Update existing records to track unknown model/dimensions
-- This is safe as we're setting NULL values to a default that indicates "unknown"
UPDATE archon_crawled_pages 
SET 
    embedding_model = 'text-embedding-3-small',
    embedding_dimensions = 1536
WHERE embedding_model IS NULL 
  AND (embedding_1536 IS NOT NULL OR embedding_768 IS NOT NULL OR embedding_1024 IS NOT NULL OR embedding_3072 IS NOT NULL);

UPDATE archon_code_examples
SET 
    embedding_model = 'text-embedding-3-small', 
    embedding_dimensions = 1536
WHERE embedding_model IS NULL
  AND (embedding_1536 IS NOT NULL OR embedding_768 IS NOT NULL OR embedding_1024 IS NOT NULL OR embedding_3072 IS NOT NULL);

-- =====================================================
-- Migration Complete
-- =====================================================
-- The archon_crawled_pages and archon_code_examples tables now track:
-- - embedding_model: Which model generated the embedding
-- - embedding_dimensions: Vector dimensions used
--
-- This enables:
-- 1. Model usage analytics and reporting
-- 2. Safe model migrations with dimension tracking
-- 3. Mixed-model support in the same database
-- 4. Debugging embedding-related issues
-- =====================================================