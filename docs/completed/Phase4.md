# Phase 4 Implementation Plan: Supabase Vector Database for Hierarchical RAG

## Overview

This implementation plan details how to implement Phase 4 (Vector Database Implementation) of our Hierarchical RAG system using Supabase with pgvector and OpenAI embeddings. Building on the document processing, hierarchical chunking, and metadata enrichment completed in Phases 1-3, this phase will focus on efficiently storing and retrieving hierarchical chunks while preserving their contextual relationships.

## 1. Project Structure

```
archon/llms-txt/
├── __init__.py
├── markdown_processor.py  # From Phase 1
├── hierarchical_chunker.py  # From Phase 2
├── metadata_enricher.py  # From Phase 3
├── vector_db/
│   ├── __init__.py
│   ├── supabase_manager.py  # Supabase connection and operations
│   ├── embedding_manager.py  # OpenAI embedding generation
│   ├── index_manager.py  # Database operations optimization (Placeholder)
│   └── query_manager.py  # Complex query construction
├── utils/
│   ├── __init__.py
│   └── env_loader.py  # Load configuration from env_vars.json
└── run_processing.py  # Main execution script (Updated)
```

## 2. Environment Configuration

### 2.1 `utils/env_loader.py`

```python
import json
import os
from typing import Dict, Any

class EnvironmentLoader:
    """Load environment variables from env_vars.json"""
    
    def __init__(self, env_file_path: str = "env_vars.json"):
        """Initialize with path to env_vars.json"""
        self.env_file_path = env_file_path
        self.config = self._load_env()
    
    def _load_env(self) -> Dict[str, Any]:
        """Load environment variables from JSON file"""
        if not os.path.exists(self.env_file_path):
            raise FileNotFoundError(f"Environment file not found: {self.env_file_path}")
        
        with open(self.env_file_path, "r") as f:
            env_data = json.load(f)
        
        # Get current profile
        current_profile = env_data.get("current_profile", "default")
        profile_config = env_data.get("profiles", {}).get(current_profile, {})
        
        if not profile_config:
            raise ValueError(f"Profile '{current_profile}' not found in environment file")
        
        return profile_config
    
    def get_supabase_config(self) -> Dict[str, str]:
        """Get Supabase connection configuration"""
        return {
            "url": self.config.get("SUPABASE_URL"),
            "key": self.config.get("SUPABASE_SERVICE_KEY")
        }
    
    def get_openai_config(self) -> Dict[str, str]:
        """Get OpenAI API configuration"""
        return {
            "api_key": self.config.get("LLM_API_KEY"),
            "base_url": self.config.get("BASE_URL"),
            "embedding_model": self.config.get("EMBEDDING_MODEL")
        }
```

## 3. Supabase Database Connection

### 3.1 `vector_db/supabase_manager.py`

```python
from typing import Dict, List, Optional, Any, Tuple
import json
from supabase import create_client, Client
from ..utils.env_loader import EnvironmentLoader

class SupabaseManager:
    """Manage Supabase database connection and operations"""
    
    def __init__(self, env_loader: Optional[EnvironmentLoader] = None):
        """Initialize Supabase connection"""
        self.env_loader = env_loader or EnvironmentLoader()
        self.supabase_config = self.env_loader.get_supabase_config()
        
        # Create Supabase client
        self.client = create_client(
            self.supabase_config["url"],
            self.supabase_config["key"]
        )
        
        # Initialize tables
        self._check_tables()
    
    def _check_tables(self) -> None:
        """Check if required tables exist"""
        # This is just a check - tables should be created using the SQL init script
        try:
            # Test query
            self.client.table("hierarchical_nodes").select("id").limit(1).execute()
            print("Successfully connected to hierarchical_nodes table")
        except Exception as e:
            print(f"Error connecting to hierarchical_nodes table: {e}")
            print("Make sure to run the initialization SQL script first")
    
    def insert_node(self, node: Dict[str, Any]) -> int:
        """Insert a node into the hierarchical_nodes table
        
        Args:
            node: Dictionary with node data matching hierarchical_nodes schema
            
        Returns:
            The inserted node ID
        """
        # Ensure content types are correct
        if "embedding" in node and isinstance(node["embedding"], list):
            # Convert list to vector format
            # Embedding should already be the correct dimensionality (1536 for text-embedding-3-small)
            pass  # Supabase handles the conversion from list to pgvector
        
        # Handle the metadata field properly
        if "metadata" not in node:
            node["metadata"] = {}
        
        response = self.client.table("hierarchical_nodes").insert(node).execute()
        
        if response.data:
            return response.data[0]["id"]
        else:
            raise Exception(f"Failed to insert node: {response.error}")
    
    def insert_reference(self, reference: Dict[str, Any]) -> int:
        """Insert a cross-reference into the hierarchical_references table
        
        Args:
            reference: Dictionary with reference data matching hierarchical_references schema
            
        Returns:
            The inserted reference ID
        """
        response = self.client.table("hierarchical_references").insert(reference).execute()
        
        if response.data:
            return response.data[0]["id"]
        else:
            raise Exception(f"Failed to insert reference: {response.error}")
    
    def vector_search(
        self, 
        embedding: List[float], 
        match_count: int = 10,
        metadata_filter: Dict[str, Any] = None,
        section_filter: str = None,
        level_filter: int = None,
        content_type_filter: str = None
    ) -> List[Dict[str, Any]]:
        """Perform vector similarity search using the match_hierarchical_nodes function
        
        Args:
            embedding: Query embedding vector (from OpenAI)
            match_count: Maximum number of results to return
            metadata_filter: JSON filter for metadata field
            section_filter: Filter by section_type
            level_filter: Filter by header level
            content_type_filter: Filter by content_type
            
        Returns:
            List of matching nodes with similarity scores
        """
        # Build the parameters
        params = {
            "query_embedding": embedding,
            "match_count": match_count
        }
        
        if metadata_filter:
            params["filter"] = json.dumps(metadata_filter)
        
        if section_filter:
            params["section_filter"] = section_filter
        
        if level_filter is not None:
            params["level_filter"] = level_filter
        
        if content_type_filter:
            params["content_type_filter"] = content_type_filter
        
        # Call the RPC function
        response = self.client.rpc(
            "match_hierarchical_nodes", 
            params
        ).execute()
        
        if response.data:
            return response.data
        else:
            # Return empty list on no results
            if response.error:
                print(f"Search error: {response.error}")
            return []
    
    def get_node_with_context(self, node_id: int, context_depth: int = 3) -> List[Dict[str, Any]]:
        """Get a node with its context (parents, children, references)
        
        Args:
            node_id: ID of the node to get context for
            context_depth: How many levels of parent context to include
            
        Returns:
            List of related nodes with context information
        """
        response = self.client.rpc(
            "get_node_with_context",
            {
                "node_id": node_id,
                "context_depth": context_depth
            }
        ).execute()
        
        if response.data:
            return response.data
        else:
            # Return empty list on no results
            if response.error:
                print(f"Context retrieval error: {response.error}")
            return []
    
    def find_nodes_by_path(self, path_pattern: str, max_results: int = 20) -> List[Dict[str, Any]]:
        """Find nodes by path pattern
        
        Args:
            path_pattern: Text pattern to search in paths
            max_results: Maximum number of results to return
            
        Returns:
            List of matching nodes
        """
        response = self.client.rpc(
            "find_nodes_by_path",
            {
                "path_pattern": path_pattern,
                "max_results": max_results
            }
        ).execute()
        
        if response.data:
            return response.data
        else:
            # Return empty list on no results
            if response.error:
                print(f"Path search error: {response.error}")
            return []
    
    def get_full_subtree(self, root_node_id: int) -> List[Dict[str, Any]]:
        """Get the full subtree starting from a root node
        
        Args:
            root_node_id: ID of the root node
            
        Returns:
            List of all nodes in the subtree
        """
        response = self.client.rpc(
            "get_full_subtree",
            {
                "root_node_id": root_node_id
            }
        ).execute()
        
        if response.data:
            return response.data
        else:
            # Return empty list on no results
            if response.error:
                print(f"Subtree retrieval error: {response.error}")
            return []
```

## 4. OpenAI Embedding Generation

### 4.1 `vector_db/embedding_manager.py`

```python
from typing import Dict, List, Any
import openai
from openai import OpenAI
from ..utils.env_loader import EnvironmentLoader

class OpenAIEmbeddingGenerator:
    """Generate embeddings using OpenAI's API"""
    
    def __init__(self, env_loader: EnvironmentLoader = None):
        """Initialize the OpenAI API client"""
        self.env_loader = env_loader or EnvironmentLoader()
        self.openai_config = self.env_loader.get_openai_config()
        
        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=self.openai_config["api_key"],
            base_url=self.openai_config["base_url"]
        )
        
        self.embedding_model = self.openai_config["embedding_model"]
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding for a single text
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            List of floats representing the embedding vector
        """
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text,
            encoding_format="float"
        )
        
        # Return the embedding
        return response.data[0].embedding
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            List of embedding vectors
        """
        # Check for empty list
        if not texts:
            return []
        
        # OpenAI API supports batching
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=texts,
            encoding_format="float"
        )
        
        # Extract and return embeddings in the same order
        embeddings = [item.embedding for item in response.data]
        return embeddings
    
    def generate_node_embeddings(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate embeddings for hierarchical nodes
        
        Args:
            nodes: List of node dictionaries with 'content' field
            
        Returns:
            The same list of nodes with 'embedding' field added
        """
        # Extract content from each node
        texts = []
        title_texts = []
        
        for node in nodes:
            # Main content embedding
            content = node.get("content", "")
            if node.get("title"):
                # Include title in content for better context
                content_with_title = f"{node['title']}\n\n{content}"
                texts.append(content_with_title)
            else:
                texts.append(content)
            
            # Title/path embedding for specialized search
            if node.get("path"):
                title_texts.append(node["path"])
            elif node.get("title"):
                title_texts.append(node["title"])
            else:
                title_texts.append("")
        
        # Generate embeddings
        content_embeddings = self.generate_embeddings(texts)
        title_embeddings = self.generate_embeddings(title_texts)
        
        # Add embeddings to nodes
        for i, node in enumerate(nodes):
            # Store main embedding (used for vector search)
            node["embedding"] = content_embeddings[i]
            
            # Store title embedding in metadata for possible future use
            if "metadata" not in node:
                node["metadata"] = {}
            
            # We don't store title_embedding directly in the database
            # but keep it in memory for specialized retrieval if needed
            node["metadata"]["has_title_embedding"] = True
        
        return nodes
```

## 5. Query Management

### 5.1 `vector_db/query_manager.py`

```python
from typing import Dict, List, Any, Optional, Tuple
from .supabase_manager import SupabaseManager
from .embedding_manager import OpenAIEmbeddingGenerator

class HierarchicalQueryManager:
    """Manager for complex hierarchical queries"""
    
    def __init__(
        self, 
        supabase_manager: Optional[SupabaseManager] = None, 
        embedding_generator: Optional[OpenAIEmbeddingGenerator] = None
    ):
        """Initialize with database and embedding managers"""
        self.db = supabase_manager or SupabaseManager()
        self.embedder = embedding_generator or OpenAIEmbeddingGenerator()
    
    def search(
        self,
        query: str,
        match_count: int = 10,
        metadata_filter: Dict[str, Any] = None,
        section_filter: str = None,
        level_filter: int = None,
        content_type_filter: str = None
    ) -> List[Dict[str, Any]]:
        """Search for nodes by semantic similarity to query
        
        Args:
            query: Natural language query
            match_count: Maximum number of results to return
            metadata_filter: Filter by metadata fields
            section_filter: Filter by section_type
            level_filter: Filter by header level
            content_type_filter: Filter by content_type
            
        Returns:
            List of matching nodes with similarity scores
        """
        # Generate embedding for query
        query_embedding = self.embedder.generate_embedding(query)
        
        # Perform vector search
        results = self.db.vector_search(
            embedding=query_embedding,
            match_count=match_count,
            metadata_filter=metadata_filter,
            section_filter=section_filter,
            level_filter=level_filter,
            content_type_filter=content_type_filter
        )
        
        return results
    
    def hierarchical_search(
        self,
        query: str,
        match_count: int = 10,
        context_depth: int = 3,
        include_children: bool = True,
        include_references: bool = True
    ) -> List[Dict[str, Any]]:
        """Perform a search and enrich results with hierarchical context
        
        Args:
            query: Natural language query
            match_count: Maximum number of results to return
            context_depth: How many levels of parent context to include
            include_children: Whether to include child nodes
            include_references: Whether to include cross-references
            
        Returns:
            List of node clusters with hierarchical context
        """
        # First, perform the base search
        base_results = self.search(query, match_count=match_count)
        
        # No results? Return empty list
        if not base_results:
            return []
        
        # For each result, get its hierarchical context
        enriched_results = []
        
        for result in base_results:
            node_id = result["id"]
            
            # Get context for this node
            context_nodes = self.db.get_node_with_context(
                node_id=node_id,
                context_depth=context_depth
            )
            
            # Group the nodes by their context type for easier processing
            context_by_type = {
                "self": [],
                "parent": [],
                "child": [],
                "reference": []
            }
            
            for node in context_nodes:
                context_type = node.get("context_type", "self")
                context_by_type[context_type].append(node)
            
            # Build the enriched result
            enriched_result = {
                "main_node": result,
                "parents": sorted(context_by_type["parent"], key=lambda x: x.get("context_level", 0)),
                "similarity": result.get("similarity", 0)
            }
            
            if include_children:
                enriched_result["children"] = context_by_type["child"]
            
            if include_references:
                enriched_result["references"] = context_by_type["reference"]
            
            enriched_results.append(enriched_result)
        
        return enriched_results
    
    def path_based_search(
        self,
        path_query: str,
        semantic_query: Optional[str] = None,
        max_results: int = 20
    ) -> List[Dict[str, Any]]:
        """Search for nodes by path and optionally by semantic similarity
        
        Args:
            path_query: Text pattern to search in paths
            semantic_query: Optional semantic query to refine results
            max_results: Maximum number of results to return
            
        Returns:
            List of matching nodes
        """
        # First, search by path
        path_results = self.db.find_nodes_by_path(
            path_pattern=path_query,
            max_results=max_results * 2  # Get more results for filtering
        )
        
        # If no semantic query, return the path results directly
        if not semantic_query or not path_results:
            return path_results[:max_results]
        
        # If we have a semantic query, re-rank the path results
        query_embedding = self.embedder.generate_embedding(semantic_query)
        
        # Extract node IDs for filtering
        node_ids = [node["id"] for node in path_results]
        
        # Build a metadata filter to only search these nodes
        metadata_filter = {"id": {"in": node_ids}}
        
        # Perform semantic search filtered to these nodes
        semantic_results = self.db.vector_search(
            embedding=query_embedding,
            match_count=max_results,
            metadata_filter=metadata_filter
        )
        
        return semantic_results
```

## 6. Complete Data Pipeline Integration (`run_processing.py`)

The `run_processing.py` script orchestrates the entire pipeline:

1.  **Initialization:** Initializes all components (`MarkdownProcessor`, `HierarchicalChunker`, `MetadataEnricher`, `SupabaseManager`, `OpenAIEmbeddingGenerator`).
2.  **File Reading & Parsing (Phase 1):** Reads the input file and uses `MarkdownProcessor` to parse and build the document tree.
3.  **Chunking (Phase 2):** Uses `HierarchicalChunker` to create chunks from the document tree.
4.  **Metadata Enrichment (Phase 3):** Uses `MetadataEnricher` to add metadata to the chunks.
5.  **Node Preparation & Reference Pre-processing:**
    *   Converts enriched chunks into the format required for the `hierarchical_nodes` table.
    *   Builds an in-memory map (`path_to_original_id_map`) of exact node paths to their original chunk IDs.
    *   Iterates through chunks, attempting to resolve `related_sections` using the exact path map.
    *   Stores successfully resolved exact references (`exact_resolved_references`).
    *   Collects paths that did not have an exact match into `paths_needing_fuzzy_lookup`.
6.  **Batch Fuzzy Lookups:**
    *   Iterates through `paths_needing_fuzzy_lookup`.
    *   Calls `db.find_nodes_by_path` *once* for each unique path pattern.
    *   **Filters** the results to keep only nodes matching the current `effective_document_id`.
    *   Stores the filtered results in `fuzzy_path_to_nodes_map`.
7.  **Embedding Generation:** Uses `OpenAIEmbeddingGenerator` to generate embeddings for the prepared nodes.
8.  **Node Insertion:**
    *   Clears existing nodes for the `effective_document_id` using `db.delete_nodes_by_document_id`.
    *   Inserts the nodes with embeddings into the `hierarchical_nodes` table using `db.insert_node`.
    *   Builds a map (`original_id_to_db_id_map`) from original chunk IDs to the newly assigned database IDs.
9.  **Relationship Creation (Optimized):**
    *   **Parent Links:** Iterates through the inserted nodes (`original_id_to_db_id_map`), finds the original parent ID from `original_id_to_chunk_map`, maps it to the DB parent ID, and calls `db.update_node_parent`.
    *   **Cross-References (Combined Pass):**
        *   Initializes a set `inserted_reference_pairs` to prevent duplicates.
        *   Processes `exact_resolved_references`: Maps original IDs to DB IDs and calls `db.insert_reference` for valid, non-duplicate pairs.
        *   Processes Fuzzy Matches: Iterates through chunks again. For paths requiring fuzzy lookup, retrieves the pre-filtered, pre-fetched target nodes from `fuzzy_path_to_nodes_map`. Calls `db.insert_reference` for valid, non-duplicate pairs.
10. **Command-Line Interface:** Uses `argparse` to handle command-line arguments for file input, document ID, and optional test queries.
11. **Test Query Execution:** If a query is provided via CLI, uses `HierarchicalQueryManager` to perform a search and display results with context.

```python
# Example Snippet (Conceptual - Refer to run_processing.py for full implementation)

# ... (Initialization, Phases 1-3) ...

# --- Phase 4: Prepare Nodes & Pre-process References ---
# ... (Build path_to_original_id_map) ...
# ... (Resolve exact_resolved_references and paths_needing_fuzzy_lookup) ...

# --- Phase 4: Batch Fuzzy Lookups ---
fuzzy_path_to_nodes_map = {}
for path_pattern in paths_needing_fuzzy_lookup:
    target_nodes_raw = db.find_nodes_by_path(f"%{path_pattern}%", max_results=10)
    filtered_nodes = [n for n in target_nodes_raw if n.get("document_id") == effective_document_id]
    if filtered_nodes:
        fuzzy_path_to_nodes_map[path_pattern] = filtered_nodes

# --- Phase 4: Generate Embeddings ---
# ... (Generate embeddings) ...

# --- Phase 4: Insert Nodes into Database ---
# ... (Clear existing nodes) ...
# ... (Insert nodes and build original_id_to_db_id_map) ...

# --- Phase 4: Create Relationships (Optimized) ---
inserted_reference_pairs = set()

# ... (Set Parent Links using db.update_node_parent) ...

# Process Exact Matches
for source_orig_id, target_orig_id in exact_resolved_references:
    source_db_id = original_id_to_db_id_map.get(source_orig_id)
    target_db_id = original_id_to_db_id_map.get(target_orig_id)
    if source_db_id and target_db_id and source_db_id != target_db_id:
        ref_pair = (source_db_id, target_db_id)
        if ref_pair not in inserted_reference_pairs:
            # ... (Create reference_data) ...
            db.insert_reference(reference_data)
            inserted_reference_pairs.add(ref_pair)

# Process Fuzzy Matches
for original_id, chunk_data in original_id_to_chunk_map.items():
    source_db_id = original_id_to_db_id_map.get(original_id)
    # ... (Get related_sections, convert path_item to path_key_str) ...
    if path_key_str in fuzzy_path_to_nodes_map:
        target_nodes = fuzzy_path_to_nodes_map[path_key_str] # Already filtered
        for target_node in target_nodes:
            target_db_id = target_node.get("id")
            if target_db_id and target_db_id in original_id_to_db_id_map.values() and target_db_id != source_db_id:
                ref_pair = (source_db_id, target_db_id)
                if ref_pair not in inserted_reference_pairs:
                    # ... (Create reference_data) ...
                    db.insert_reference(reference_data)
                    inserted_reference_pairs.add(ref_pair)

# ... (main function with argparse) ...
```


## 7. Index Optimization Strategies

### 7.1 Index Management Considerations

The SQL schema already includes optimized indexes for pgvector, but here are some additional optimization strategies to implement:

1. **Batch Processing**: Insert documents in batches to improve performance
2. **Caching**: Implement caching for frequent queries
3. **Connection Pooling**: Configure connection pooling for Supabase
4. **Vacuum and Analyze**: Periodically run maintenance operations

### 7.2 Implementation Notes for Specific Document Types

For Markdown documentation (like the Pydantic docs):

1. **Header Weighting**: Give more weight to header matches than content matches
2. **Code Examples**: Store code examples with special handling in metadata
3. **Link Collections**: Preserve link relationships in the metadata JSON field
4. **Section References**: Create explicit cross-references for related sections

## 8. Testing Approach

1. **Unit Tests**:
   - Test each component individually (parsing, chunking, embedding)
   - Verify database operations work correctly

2. **Integration Tests**:
   - Process sample documentation end-to-end
   - Verify hierarchical relationships are preserved

3. **Query Performance Testing**:
   - Time various query types (direct path, semantic search, hybrid)
   - Optimize based on results

4. **A/B Testing vs. Traditional RAG**:
   - Compare retrieval quality against flat chunking approach
   - Measure relevance, context quality, and retrieval accuracy

## 9. Implementation Steps

1. Set up Supabase tables using the provided SQL script
2. Implement the environment configuration loader
3. Implement the Supabase manager for database operations
4. Implement the OpenAI embedding generator
5. Implement the query manager for advanced searches
6. Integrate all phases into `run_processing.py`
7. Implement optimized relationship creation (hybrid approach) in `run_processing.py`
8. Test with sample documentation (e.g., Pydantic docs)
9. Further optimize based on performance results if needed

## 10. Next Steps

1. **Query Optimization**: Fine-tune search parameters and ranking algorithms
2. **UI Integration**: Create a simple web UI for browsing hierarchical documentation
3. **Feedback Loop**: Implement user feedback collection to improve retrieval
4. **Batch Processing**: Add support for processing multiple documents in one run
5. **Incremental Updates**: Implement efficient updating of existing documents