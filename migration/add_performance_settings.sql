-- =====================================================
-- Add Performance Settings to RAG Configuration
-- =====================================================
-- This migration adds configurable performance settings
-- for crawling and document storage operations
-- =====================================================

-- Crawling Performance Settings
INSERT INTO settings (key, value, is_encrypted, category, description) VALUES
('CRAWL_BATCH_SIZE', '50', false, 'rag_strategy', 'Number of URLs to crawl in parallel per batch (10-100)'),
('CRAWL_MAX_CONCURRENT', '10', false, 'rag_strategy', 'Maximum concurrent browser sessions for crawling (1-20)'),
('CRAWL_WAIT_STRATEGY', 'domcontentloaded', false, 'rag_strategy', 'When to consider page loaded: domcontentloaded, networkidle, or load'),
('CRAWL_PAGE_TIMEOUT', '30000', false, 'rag_strategy', 'Maximum time to wait for page load in milliseconds'),
('CRAWL_DELAY_BEFORE_HTML', '0.5', false, 'rag_strategy', 'Time to wait for JavaScript rendering in seconds (0.1-5.0)')
ON CONFLICT (key) DO NOTHING;

-- Document Storage Performance Settings
INSERT INTO settings (key, value, is_encrypted, category, description) VALUES
('DOCUMENT_STORAGE_BATCH_SIZE', '50', false, 'rag_strategy', 'Number of document chunks to process per batch (10-100)'),
('EMBEDDING_BATCH_SIZE', '100', false, 'rag_strategy', 'Number of embeddings to create per API call (20-200)'),
('DELETE_BATCH_SIZE', '50', false, 'rag_strategy', 'Number of URLs to delete in one database operation (10-100)'),
('ENABLE_PARALLEL_BATCHES', 'true', false, 'rag_strategy', 'Enable parallel processing of document batches')
ON CONFLICT (key) DO NOTHING;

-- Advanced Performance Settings
INSERT INTO settings (key, value, is_encrypted, category, description) VALUES
('MEMORY_THRESHOLD_PERCENT', '80', false, 'rag_strategy', 'Memory usage threshold for crawler dispatcher (50-90)'),
('DISPATCHER_CHECK_INTERVAL', '0.5', false, 'rag_strategy', 'How often to check memory usage in seconds (0.1-2.0)'),
('CODE_EXTRACTION_BATCH_SIZE', '20', false, 'rag_strategy', 'Number of code blocks to extract per batch (5-50)'),
('CODE_SUMMARY_MAX_WORKERS', '3', false, 'rag_strategy', 'Maximum parallel workers for code summarization (1-10)')
ON CONFLICT (key) DO NOTHING;

-- Update descriptions for existing CONTEXTUAL_EMBEDDINGS_MAX_WORKERS to be consistent
UPDATE settings 
SET description = 'Maximum parallel workers for contextual embedding generation (1-10)'
WHERE key = 'CONTEXTUAL_EMBEDDINGS_MAX_WORKERS';