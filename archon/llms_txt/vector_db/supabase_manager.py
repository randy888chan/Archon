from typing import Dict, List, Optional, Any, Tuple
import json
import os # Added for potential future use, though not strictly needed by spec
from supabase import create_client, Client
# Corrected import path assuming vector_db is sibling to utils
from ..utils.env_loader import EnvironmentLoader

class SupabaseManager:
    """Manage Supabase database connection and operations"""

    def __init__(self, env_loader: Optional[EnvironmentLoader] = None):
        """Initialize Supabase connection"""
        # Allow passing an existing env_loader or create a new one
        # Pass the expected path relative to the project root
        # Assuming archon/llms-txt/ is the root for this module's perspective
        # The env_loader itself handles finding the file
        self.env_loader = env_loader or EnvironmentLoader(env_file_path="../../workbench/env_vars.json") # Adjusted path
        self.supabase_config = self.env_loader.get_supabase_config()

        if not self.supabase_config.get("url") or not self.supabase_config.get("key"):
             raise ValueError("Supabase URL or Key missing in configuration.")

        # Create Supabase client
        try:
            self.client: Client = create_client(
                self.supabase_config["url"],
                self.supabase_config["key"]
            )
            print("Supabase client initialized successfully.")
        except Exception as e:
            print(f"Error initializing Supabase client: {e}")
            raise # Re-raise the exception after printing

        # Initialize tables - Perform a basic check
        # self._check_tables() # Commenting out initial check as it might fail if tables aren't ready

    def _check_tables(self) -> None:
        """Check if required tables exist by attempting a simple query."""
        # This is just a check - tables should be created using the SQL init script
        required_tables = ["hierarchical_nodes", "hierarchical_references"]
        print(f"Checking existence of tables: {required_tables}")
        all_exist = True
        for table_name in required_tables:
            try:
                # Test query
                self.client.table(table_name).select("id", count='exact').limit(1).execute()
                print(f"Successfully connected to '{table_name}' table.")
            except Exception as e:
                print(f"Error connecting to '{table_name}' table: {e}")
                print(f"Ensure '{table_name}' table exists and the SQL initialization script has been run.")
                all_exist = False
        if not all_exist:
             print("Warning: Not all required Supabase tables could be verified.")
        # We don't raise an error here, allowing the application to potentially proceed
        # if only some tables are needed initially, but log a clear warning.

    def insert_node(self, node: Dict[str, Any]) -> int:
        """Insert a node into the hierarchical_nodes table

        Args:
            node: Dictionary with node data matching hierarchical_nodes schema

        Returns:
            The inserted node ID
        """
        # Ensure content types are correct
        if "embedding" in node and isinstance(node["embedding"], list):
            # Supabase python client handles list-to-vector conversion automatically
            pass

        # Handle the metadata field properly - ensure it's a dict
        if "metadata" not in node or not isinstance(node["metadata"], dict):
            node["metadata"] = {} # Default to empty dict if missing or wrong type

        # Remove fields not expected by the table schema before insertion
        # Example: remove 'original_id' if it was only for mapping
        node_to_insert = node.copy()
        if "metadata" in node_to_insert and "original_id" in node_to_insert["metadata"]:
             # Keep original_id in metadata JSONB, don't remove
             pass
        # Remove other potential temporary fields if necessary
        # node_to_insert.pop("some_temp_field", None)

        try:
            response = self.client.table("hierarchical_nodes").insert(node_to_insert).execute()

            if response.data:
                return response.data[0]["id"]
            else:
                # Log more detailed error if available
                error_message = f"Failed to insert node. Response error: {response.error}"
                print(error_message)
                raise Exception(error_message)
        except Exception as e:
            print(f"Exception during node insertion: {e}")
            # Include node details (excluding embedding) for debugging
            node_details = {k: v for k, v in node_to_insert.items() if k != 'embedding'}
            print(f"Node data (excluding embedding): {json.dumps(node_details, indent=2)}")
            raise # Re-raise the exception

    def insert_reference(self, reference: Dict[str, Any]) -> int:
        """Insert a cross-reference into the hierarchical_references table

        Args:
            reference: Dictionary with reference data matching hierarchical_references schema

        Returns:
            The inserted reference ID
        """
        try:
            response = self.client.table("hierarchical_references").insert(reference).execute()

            if response.data:
                return response.data[0]["id"]
            else:
                 error_message = f"Failed to insert reference. Response error: {response.error}"
                 print(error_message)
                 raise Exception(error_message)
        except Exception as e:
            print(f"Exception during reference insertion: {e}")
            print(f"Reference data: {json.dumps(reference, indent=2)}")
            raise # Re-raise the exception

    def vector_search(
        self,
        embedding: List[float],
        match_count: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None, # Made Optional
        section_filter: Optional[str] = None, # Made Optional
        level_filter: Optional[int] = None, # Made Optional
        content_type_filter: Optional[str] = None # Made Optional
    ) -> List[Dict[str, Any]]:
        """Perform vector similarity search using the match_hierarchical_nodes function

        Args:
            embedding: Query embedding vector (from OpenAI)
            match_count: Maximum number of results to return
            metadata_filter: JSON filter for metadata field (e.g., {"source": "pydantic"})
            section_filter: Filter by section_type
            level_filter: Filter by header level
            content_type_filter: Filter by content_type

        Returns:
            List of matching nodes with similarity scores
        """
        # Build the parameters for the RPC call
        params = {
            "query_embedding": embedding,
            "match_count": match_count,
            # Default optional params to None if not provided
            "filter": json.dumps(metadata_filter) if metadata_filter else None,
            "section_filter": section_filter,
            "level_filter": level_filter,
            "content_type_filter": content_type_filter
        }

        # Remove None values from params as RPC might expect missing keys vs null values
        params = {k: v for k, v in params.items() if v is not None}

        print(f"DEBUG: Calling match_hierarchical_nodes with params: {json.dumps(params, indent=2, default=lambda x: '<embedding vector>')}") # Log params, hiding embedding for brevity
        try:
            # Call the RPC function
            response = self.client.rpc(
                "match_hierarchical_nodes",
                params
            ).execute()

            if response.data:
                return response.data
            else:
                # Return empty list on no results or error
                if response.error:
                    print(f"Vector search RPC error: {response.error}")
                return []
        except Exception as e:
            print(f"Exception during vector search RPC call: {e}")
            return [] # Return empty list on exception

    def get_node_with_context(self, node_id: int, context_depth: int = 3) -> List[Dict[str, Any]]:
        """Get a node with its context (parents, children, references) via RPC

        Args:
            node_id: ID of the node to get context for
            context_depth: How many levels of parent context to include

        Returns:
            List of related nodes with context information (or empty list on error/no data)
        """
        try:
            response = self.client.rpc(
                "get_node_with_context",
                {
                    "p_node_id": node_id, # Match param name in SQL function
                    "p_context_depth": context_depth # Match param name in SQL function
                }
            ).execute()

            if response.data:
                return response.data
            else:
                if response.error:
                    print(f"Context retrieval RPC error for node {node_id}: {response.error}")
                return []
        except Exception as e:
            print(f"Exception during context retrieval RPC call for node {node_id}: {e}")
            return []

    def find_nodes_by_path(self, path_pattern: str, max_results: int = 20) -> List[Dict[str, Any]]:
        """Find nodes by path pattern using RPC

        Args:
            path_pattern: Text pattern to search in paths (e.g., '%Introduction%')
            max_results: Maximum number of results to return

        Returns:
            List of matching nodes (or empty list on error/no data)
        """
        try:
            response = self.client.rpc(
                "find_nodes_by_path",
                {
                    "path_pattern": path_pattern, # Corrected param name based on error hint
                    "max_results": max_results  # Corrected param name based on error hint
                }
            ).execute()

            # Check for data first. If data exists, return it.
            if response.data:
                return response.data
            # If no data, check if there was an explicit error attribute (might not exist on success)
            # Supabase client >= 2.0 uses model_dump() for error details if available
            elif hasattr(response, 'model_dump') and response.model_dump().get('error'):
                 error_details = response.model_dump().get('error')
                 print(f"Path search RPC error for pattern '{path_pattern}': {error_details}")
                 return []
            # If no data and no explicit error, assume success with empty results
            else:
                 # print(f"Path search RPC for pattern '{path_pattern}' succeeded but returned no results.") # Optional: Log success/no data
                 return []
        except Exception as e:
            print(f"Exception during path search RPC call for pattern '{path_pattern}': {e}")
            return []

    def get_full_subtree(self, root_node_id: int) -> List[Dict[str, Any]]:
        """Get the full subtree starting from a root node using RPC

        Args:
            root_node_id: ID of the root node

        Returns:
            List of all nodes in the subtree (or empty list on error/no data)
        """
        try:
            response = self.client.rpc(
                "get_full_subtree",
                {
                    "p_root_node_id": root_node_id # Match param name in SQL function
                }
            ).execute()

            if response.data:
                return response.data
            else:
                if response.error:
                    print(f"Subtree retrieval RPC error for root node {root_node_id}: {response.error}")
                return []
        except Exception as e:
            print(f"Exception during subtree retrieval RPC call for root node {root_node_id}: {e}")
            return []

    def update_node_parent(self, node_id: int, parent_id: Optional[int]) -> None:
         """Update the parent_id of a specific node."""
         try:
             response = self.client.table("hierarchical_nodes").update({
                 "parent_id": parent_id
             }).eq("id", node_id).execute()

             if response.error:
                 print(f"Error updating parent for node {node_id}: {response.error}")
             # else:
             #     print(f"Successfully updated parent for node {node_id} to {parent_id}")

         except Exception as e:
             print(f"Exception updating parent for node {node_id}: {e}")

    def delete_nodes_by_document_id(self, document_id: str) -> int:
        """Deletes all nodes associated with a specific document_id.

        Args:
            document_id: The identifier of the document whose nodes should be deleted.

        Returns:
            The number of nodes deleted.

        Raises:
            Exception: If the delete operation fails.
        """
        if not document_id:
            print("Warning: Attempted to delete nodes with empty document_id. Skipping.")
            return 0

        try:
            # The Supabase client's delete().execute() returns the deleted records
            response = self.client.table("hierarchical_nodes") \
                .delete() \
                .eq("document_id", document_id) \
                .execute()

            # Check for errors in the response
            if hasattr(response, 'error') and response.error:
                error_message = f"Failed to delete nodes for document_id '{document_id}'. Response error: {response.error}"
                print(error_message)
                raise Exception(error_message)

            # If successful, response.data contains the list of deleted records
            deleted_count = len(response.data) if response.data else 0
            return deleted_count

        except Exception as e:
            # Catch potential exceptions during the API call or response processing
            print(f"Exception during node deletion for document_id '{document_id}': {e}")
            raise # Re-raise the exception to signal failure
