-- Hierarchical RAG Database Tables for Supabase
-- Compatible with existing site_pages table

-- Enable the pgvector extension if not already enabled
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- 1. Create the main document nodes table
CREATE TABLE hierarchical_nodes (
    id BIGSERIAL PRIMARY KEY,
    document_id VARCHAR NOT NULL,      -- Unique identifier for the source document
    node_type VARCHAR NOT NULL,         -- 'document', 'header', 'content', etc.
    title VARCHAR,                      -- Node title (for headers/documents)
    content TEXT,                       -- Actual content text
    level INTEGER,                      -- Header level (1, 2, 3, etc.)
    parent_id BIGINT REFERENCES hierarchical_nodes(id),  -- Parent node relationship
    path TEXT NOT NULL,                 -- Full hierarchical path (e.g. "Pydantic > API > Fields")
    section_type VARCHAR,               -- e.g. "API documentation", "Concepts", etc.
    content_type VARCHAR,               -- e.g. "link_list", "descriptive_text", "code_example"
    document_position FLOAT,            -- Normalized position in document (0-1)
    embedding VECTOR(1536),             -- Same dimension as your current embeddings
    created_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc'::text, now()) NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc'::text, now()) NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::JSONB  -- Additional metadata
);

-- 2. Create cross-references table for related sections
CREATE TABLE hierarchical_references (
    id BIGSERIAL PRIMARY KEY,
    source_node_id BIGINT REFERENCES hierarchical_nodes(id) ON DELETE CASCADE,
    target_node_id BIGINT REFERENCES hierarchical_nodes(id) ON DELETE CASCADE,
    reference_type VARCHAR NOT NULL,    -- e.g. "related", "similar", "see_also"
    strength FLOAT,                     -- Relevance score (0-1)
    created_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc'::text, now()) NOT NULL
);

-- 3. Create indexes for efficient queries
-- Main vector similarity search index
CREATE INDEX idx_hierarchical_nodes_embedding ON hierarchical_nodes 
USING IVFFLAT (embedding vector_cosine_ops);

-- Index for fast path lookups
CREATE INDEX idx_hierarchical_nodes_path ON hierarchical_nodes 
USING GIST (path gist_trgm_ops);

-- Index for parent-child relationships
CREATE INDEX idx_hierarchical_nodes_parent ON hierarchical_nodes(parent_id);

-- Index for finding all nodes in a document
CREATE INDEX idx_hierarchical_nodes_document ON hierarchical_nodes(document_id);

-- Index for metadata filtering
CREATE INDEX idx_hierarchical_nodes_metadata ON hierarchical_nodes 
USING GIN (metadata);

-- Indexes for reference table
CREATE INDEX idx_hierarchical_references_source ON hierarchical_references(source_node_id);
CREATE INDEX idx_hierarchical_references_target ON hierarchical_references(target_node_id);
CREATE INDEX idx_hierarchical_references_type ON hierarchical_references(reference_type);

-- 4. Create vector search function for hierarchical nodes
CREATE OR REPLACE FUNCTION match_hierarchical_nodes (
    query_embedding VECTOR(1536),
    match_count INT DEFAULT 10,
    filter JSONB DEFAULT '{}'::JSONB,
    section_filter VARCHAR DEFAULT NULL,
    level_filter INT DEFAULT NULL,
    content_type_filter VARCHAR DEFAULT NULL
)
RETURNS TABLE (
    id BIGINT,
    document_id VARCHAR,
    node_type VARCHAR,
    title VARCHAR,
    content TEXT,
    level INTEGER,
    parent_id BIGINT,
    path TEXT,
    section_type VARCHAR,
    content_type VARCHAR,
    metadata JSONB,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
#variable_conflict use_column
BEGIN
    RETURN QUERY
    SELECT
        id, document_id, node_type, title, content, level, parent_id, 
        path, section_type, content_type, metadata,
        1 - (hierarchical_nodes.embedding <=> query_embedding) AS similarity
    FROM hierarchical_nodes
    WHERE 
        metadata @> filter
        AND (section_filter IS NULL OR section_type = section_filter)
        AND (level_filter IS NULL OR level = level_filter)
        AND (content_type_filter IS NULL OR content_type = content_type_filter)
    ORDER BY hierarchical_nodes.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- 5. Create context expansion function
CREATE OR REPLACE FUNCTION get_node_with_context(node_id BIGINT, context_depth INT DEFAULT 3)
RETURNS TABLE (
    id BIGINT,
    document_id VARCHAR,
    node_type VARCHAR,
    title VARCHAR,
    content TEXT,
    level INTEGER,
    path TEXT,
    section_type VARCHAR,
    content_type VARCHAR,
    context_type VARCHAR,  -- 'self', 'parent', 'child', 'reference'
    context_level INTEGER  -- 0 = self, 1 = parent, 2 = grandparent, etc.
)
LANGUAGE plpgsql
AS $$
BEGIN
    -- Get the target node and its ancestors
    RETURN QUERY
    WITH RECURSIVE node_hierarchy AS (
        -- Base case: start with the specified node
        SELECT 
            n.id, n.document_id, n.node_type, n.title, n.content, n.level, 
            n.path, n.section_type, n.content_type, n.parent_id,
            'self'::VARCHAR AS context_type,
            0 AS context_level
        FROM hierarchical_nodes n
        WHERE n.id = node_id
        
        UNION ALL
        
        -- Recursive case: add each parent
        SELECT 
            p.id, p.document_id, p.node_type, p.title, p.content, p.level, 
            p.path, p.section_type, p.content_type, p.parent_id,
            'parent'::VARCHAR AS context_type,
            nh.context_level + 1
        FROM hierarchical_nodes p
        JOIN node_hierarchy nh ON p.id = nh.parent_id
        WHERE nh.context_level < context_depth
    )
    SELECT 
        id, document_id, node_type, title, content, level, path, 
        section_type, content_type, context_type, context_level
    FROM node_hierarchy
    ORDER BY context_level;
    
    -- Get immediate children of the node
    RETURN QUERY
    SELECT 
        c.id, c.document_id, c.node_type, c.title, c.content, c.level, 
        c.path, c.section_type, c.content_type,
        'child'::VARCHAR AS context_type,
        1 AS context_level
    FROM hierarchical_nodes c
    WHERE c.parent_id = node_id;
    
    -- Get related nodes via cross-references
    RETURN QUERY
    SELECT 
        n.id, n.document_id, n.node_type, n.title, n.content, n.level, 
        n.path, n.section_type, n.content_type,
        'reference'::VARCHAR AS context_type,
        1 AS context_level
    FROM hierarchical_nodes n
    JOIN hierarchical_references r ON n.id = r.target_node_id
    WHERE r.source_node_id = node_id;
END;
$$;

-- 6. Create helper function to find nodes by path
CREATE OR REPLACE FUNCTION find_nodes_by_path(
    path_pattern TEXT,
    max_results INT DEFAULT 20
)
RETURNS TABLE (
    id BIGINT,
    document_id VARCHAR,
    node_type VARCHAR,
    title VARCHAR,
    path TEXT,
    level INTEGER,
    section_type VARCHAR
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        n.id, n.document_id, n.node_type, n.title, n.path, n.level, n.section_type
    FROM hierarchical_nodes n
    WHERE n.path ILIKE '%' || path_pattern || '%'
    ORDER BY n.path
    LIMIT max_results;
END;
$$;

-- 7. Add trigger for updated_at timestamp
CREATE OR REPLACE FUNCTION update_timestamp()
RETURNS TRIGGER AS $$
BEGIN
   NEW.updated_at = NOW();
   RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_hierarchical_nodes_timestamp
BEFORE UPDATE ON hierarchical_nodes
FOR EACH ROW
EXECUTE FUNCTION update_timestamp();

-- 8. Add function to return full subtree
CREATE OR REPLACE FUNCTION get_full_subtree(root_node_id BIGINT)
RETURNS TABLE (
    id BIGINT,
    document_id VARCHAR,
    node_type VARCHAR,
    title VARCHAR,
    content TEXT,
    level INTEGER,
    parent_id BIGINT,
    path TEXT,
    depth INTEGER  -- How deep in the tree
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    WITH RECURSIVE subtree AS (
        -- Base case: the root node
        SELECT 
            n.id, n.document_id, n.node_type, n.title, n.content, n.level, 
            n.parent_id, n.path, 
            0 AS depth
        FROM hierarchical_nodes n
        WHERE n.id = root_node_id
        
        UNION ALL
        
        -- Recursive case: all children
        SELECT 
            c.id, c.document_id, c.node_type, c.title, c.content, c.level, 
            c.parent_id, c.path,
            s.depth + 1
        FROM hierarchical_nodes c
        JOIN subtree s ON c.parent_id = s.id
    )
    SELECT * FROM subtree
    ORDER BY path, depth;
END;
$$;

-- 9. Enable RLS (Row Level Security)
ALTER TABLE hierarchical_nodes ENABLE ROW LEVEL SECURITY;
ALTER TABLE hierarchical_references ENABLE ROW LEVEL SECURITY;

-- 10. Create policies that allow anyone to read
CREATE POLICY "Allow public read access" ON hierarchical_nodes 
    FOR SELECT TO public USING (true);
    
CREATE POLICY "Allow public read access" ON hierarchical_references 
    FOR SELECT TO public USING (true);

-- 11. Create policies for authenticated users to insert/update/delete (for your admin interface)
CREATE POLICY "Allow authenticated insert" ON hierarchical_nodes 
    FOR INSERT TO authenticated WITH CHECK (true);
    
CREATE POLICY "Allow authenticated update" ON hierarchical_nodes 
    FOR UPDATE TO authenticated USING (true);
    
CREATE POLICY "Allow authenticated delete" ON hierarchical_nodes 
    FOR DELETE TO authenticated USING (true);

CREATE POLICY "Allow authenticated insert" ON hierarchical_references 
    FOR INSERT TO authenticated WITH CHECK (true);
    
CREATE POLICY "Allow authenticated update" ON hierarchical_references 
    FOR UPDATE TO authenticated USING (true);
    
CREATE POLICY "Allow authenticated delete" ON hierarchical_references 
    FOR DELETE TO authenticated USING (true);