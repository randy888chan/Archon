from typing import Dict, List, Any, Optional, Tuple

# Corrected import paths assuming query_manager is in vector_db directory
from .supabase_manager import SupabaseManager
from .embedding_manager import OpenAIEmbeddingGenerator

# Import EnvironmentLoader to allow creating default managers if not provided
from ..utils.env_loader import EnvironmentLoader


class HierarchicalQueryManager:
    """Manager for complex hierarchical queries, coordinating embedding and DB interaction."""

    def __init__(
        self,
        supabase_manager: Optional[SupabaseManager] = None,
        embedding_generator: Optional[OpenAIEmbeddingGenerator] = None,
        env_loader: Optional[EnvironmentLoader] = None,  # Allow passing env_loader
    ):
        """Initialize with database and embedding managers.

        If managers are not provided, they will be instantiated using a shared
        EnvironmentLoader (either provided or newly created).
        """
        # Use provided env_loader or create a default one
        # Ensure consistent env_vars path usage
        _env_loader = env_loader or EnvironmentLoader(
            env_file_path="../../workbench/env_vars.json"
        )

        # Use provided managers or instantiate them using the env_loader
        self.db = supabase_manager or SupabaseManager(env_loader=_env_loader)
        self.embedder = embedding_generator or OpenAIEmbeddingGenerator(
            env_loader=_env_loader
        )
        print("HierarchicalQueryManager initialized.", flush=True)

    def search(
        self,
        query: str,
        match_count: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
        section_filter: Optional[str] = None,
        level_filter: Optional[int] = None,
        content_type_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search for nodes by semantic similarity to a query string.

        Args:
            query: Natural language query string.
            match_count: Maximum number of results to return.
            metadata_filter: Filter results based on metadata JSON field.
            section_filter: Filter results by section_type.
            level_filter: Filter results by header level.
            content_type_filter: Filter results by content_type.

        Returns:
            List of matching nodes, potentially including similarity scores,
            or an empty list if no matches or an error occurs.
        """
        if not query:
            print(
                "Warning: Search query is empty. Returning empty results.", flush=True
            )
            return []

        print(f"Performing semantic search for query: '{query[:50]}...'", flush=True)
        try:
            # 1. Generate embedding for the query
            query_embedding = self.embedder.generate_embedding(query)

            # 2. Perform vector search using SupabaseManager
            results = self.db.vector_search(
                embedding=query_embedding,
                match_count=match_count,
                metadata_filter=metadata_filter,
                section_filter=section_filter,
                level_filter=level_filter,
                content_type_filter=content_type_filter,
            )
            print(
                f"Semantic search found {len(results)} potential matches.", flush=True
            )
            return results

        except Exception as e:
            print(
                f"Error during semantic search for query '{query[:50]}...': {e}",
                flush=True,
            )
            return []  # Return empty list on error

    def hierarchical_search(
        self,
        query: str,
        match_count: int = 10,
        context_depth: int = 3,
        include_children: bool = True,  # Keep flags from spec, even if RPC handles logic
        include_references: bool = True,  # Keep flags from spec
    ) -> List[Dict[str, Any]]:
        """Perform semantic search and enrich results with hierarchical context.

        Args:
            query: Natural language query string.
            match_count: Maximum number of initial semantic matches.
            context_depth: How many levels of parent context to retrieve via RPC.
            include_children: (Currently handled by RPC) Flag indicating desire for children.
            include_references: (Currently handled by RPC) Flag indicating desire for references.

        Returns:
            List of enriched results. Each item contains the main matching node
            and its retrieved context (parents, children, references as returned by RPC).
            Returns empty list on error or no matches.
        """
        if not query:
            print(
                "Warning: Hierarchical search query is empty. Returning empty results.",
                flush=True,
            )
            return []

        print(
            f"Performing hierarchical search for query: '{query[:50]}...'", flush=True
        )
        try:
            # 1. Perform the base semantic search
            base_results = self.search(query, match_count=match_count)

            if not base_results:
                print(
                    "Hierarchical search: Base semantic search returned no results.",
                    flush=True,
                )
                return []

            print(
                f"Hierarchical search: Found {len(base_results)} base results. Fetching context...",
                flush=True,
            )
            enriched_results = []
            processed_node_ids = (
                set()
            )  # Avoid processing context for the same node multiple times if returned by base search

            # 2. For each unique base result, get its hierarchical context using the RPC
            for result in base_results:
                node_id = result.get("id")
                if not node_id or node_id in processed_node_ids:
                    continue  # Skip if no ID or already processed

                processed_node_ids.add(node_id)
                print(f"Fetching context for node ID: {node_id}", flush=True)

                # Use the SupabaseManager's get_node_with_context RPC call
                context_nodes = self.db.get_node_with_context(
                    node_id=node_id, context_depth=context_depth
                )

                # The RPC `get_node_with_context` should ideally return nodes tagged
                # with their relationship (self, parent, child, reference).
                # We structure the output based on these tags.
                context_by_type = {
                    "self": [],
                    "parent": [],
                    "child": [],
                    "reference": [],
                }
                for node in context_nodes:
                    # Ensure context_type exists and is valid, default to 'self' otherwise
                    ctx_type = node.get("context_type")
                    if ctx_type not in context_by_type:
                        # print(f"Warning: Node {node.get('id')} has unexpected context_type '{ctx_type}'. Treating as 'self'.")
                        ctx_type = "self"  # Default for safety
                    context_by_type[ctx_type].append(node)

                # Find the original matching node within the context results ('self')
                # It might have more fields than the base search result
                main_node_full = next(
                    (n for n in context_by_type["self"] if n.get("id") == node_id),
                    result,
                )

                # Build the enriched result structure
                enriched_node = {
                    "main_node": main_node_full,  # Use the potentially more detailed node from context
                    "similarity": result.get(
                        "similarity"
                    ),  # Keep original similarity score
                    "parents": sorted(
                        context_by_type["parent"],
                        key=lambda x: x.get("context_level", 0),
                    ),  # Sort parents by level
                    # Children and References are included based on RPC results, flags are illustrative
                    "children": context_by_type["child"] if include_children else [],
                    "references": (
                        context_by_type["reference"] if include_references else []
                    ),
                }
                enriched_results.append(enriched_node)

            print(
                f"Hierarchical search completed. Returning {len(enriched_results)} enriched results.",
                flush=True,
            )
            return enriched_results

        except Exception as e:
            print(
                f"Error during hierarchical search for query '{query[:50]}...': {e}",
                flush=True,
            )
            return []  # Return empty list on error

    def path_based_search(
        self,
        path_query: str,
        semantic_query: Optional[str] = None,
        max_results: int = 20,
    ) -> List[Dict[str, Any]]:
        """Search for nodes primarily by path pattern, optionally refined by semantics.

        Args:
            path_query: Text pattern to search within node paths (e.g., '%Section Title%').
                        Uses SQL LIKE syntax (e.g., % for wildcard).
            semantic_query: Optional natural language query to re-rank path results.
            max_results: Maximum number of final results to return.

        Returns:
            List of matching nodes, ordered by path match or semantic similarity if provided.
            Returns empty list on error or no matches.
        """
        if not path_query:
            print(
                "Warning: Path-based search query is empty. Returning empty results.",
                flush=True,
            )
            return []

        print(f"Performing path-based search for pattern: '{path_query}'", flush=True)
        try:
            # 1. Initial search by path pattern using SupabaseManager RPC
            # Fetch slightly more results initially if re-ranking will occur
            initial_fetch_count = max_results * 2 if semantic_query else max_results
            path_results = self.db.find_nodes_by_path(
                path_pattern=path_query, max_results=initial_fetch_count
            )

            if not path_results:
                print(
                    "Path-based search: No nodes found matching the path pattern.",
                    flush=True,
                )
                return []

            # 2. If no semantic query, return path results directly (limited)
            if not semantic_query:
                print(
                    f"Path-based search: Found {len(path_results)} nodes by path. Returning top {max_results}.",
                    flush=True,
                )
                return path_results[:max_results]

            # 3. If semantic query provided, re-rank path results
            print(
                f"Path-based search: Found {len(path_results)} nodes by path. Re-ranking with semantic query: '{semantic_query[:50]}...'",
                flush=True,
            )
            query_embedding = self.embedder.generate_embedding(semantic_query)

            # Extract node IDs from path results for filtering
            node_ids = [node["id"] for node in path_results if "id" in node]
            if not node_ids:
                print(
                    "Path-based search: Nodes found by path have no IDs for re-ranking.",
                    flush=True,
                )
                return []  # Cannot re-rank without IDs

            # Perform semantic search filtered by the IDs found via path search
            # Use a metadata filter targeting the 'id' field
            # Note: Supabase RPC `match_hierarchical_nodes` needs to support filtering by a list of IDs.
            # Assuming the 'filter' parameter in vector_search can handle something like:
            # filter={"id": {"in": node_ids}} # This syntax might need adjustment based on actual RPC implementation
            # For now, we construct the filter dict as expected by SupabaseManager.vector_search
            id_filter = {
                "id": node_ids
            }  # Assuming direct list works or RPC handles 'id = ANY(ids_array)'

            semantic_results = self.db.vector_search(
                embedding=query_embedding,
                match_count=max_results,  # Limit final results
                metadata_filter=id_filter,  # Apply the ID filter
                # Other filters (section, level, etc.) could also be applied here if needed
            )

            print(
                f"Path-based search: Re-ranked results yield {len(semantic_results)} nodes.",
                flush=True,
            )
            return semantic_results

        except Exception as e:
            print(
                f"Error during path-based search for pattern '{path_query}': {e}",
                flush=True,
            )
            return []  # Return empty list on error
