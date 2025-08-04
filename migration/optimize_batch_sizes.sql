-- =====================================================
-- Optimize Batch Sizes for Better Performance
-- =====================================================
-- This migration increases batch sizes for improved 
-- throughput in document storage and embedding operations
-- =====================================================

-- Increase document storage batch size from 50 to 100
UPDATE settings 
SET value = '100', description = 'Number of document chunks to process per batch (50-200) - increased for better performance'
WHERE key = 'DOCUMENT_STORAGE_BATCH_SIZE';

-- Increase embedding batch size from 100 to 200 
UPDATE settings 
SET value = '200', description = 'Number of embeddings to create per API call (100-500) - increased for better throughput'
WHERE key = 'EMBEDDING_BATCH_SIZE';

-- Increase delete batch size from 50 to 100
UPDATE settings 
SET value = '100', description = 'Number of URLs to delete in one database operation (50-200) - increased for better performance'
WHERE key = 'DELETE_BATCH_SIZE';

-- Add contextual embedding batch size setting if it doesn't exist
INSERT INTO settings (key, value, is_encrypted, category, description) VALUES
('CONTEXTUAL_EMBEDDING_BATCH_SIZE', '50', false, 'rag_strategy', 'Number of chunks to process in contextual embedding batch API calls (20-100)')
ON CONFLICT (key) DO UPDATE SET
value = '50',
description = 'Number of chunks to process in contextual embedding batch API calls (20-100)';

-- Increase code extraction batch size from 20 to 40
UPDATE settings 
SET value = '40', description = 'Number of code blocks to extract per batch (20-100) - increased for better performance'
WHERE key = 'CODE_EXTRACTION_BATCH_SIZE';