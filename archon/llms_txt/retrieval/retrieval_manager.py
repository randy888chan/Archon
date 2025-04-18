# archon/llms_txt/retrieval/retrieval_manager.py

from typing import Any, Dict, List, Optional

# Import the other components (assuming they exist in sibling files)
# These imports assume the files exist and contain the respective classes.
# If the actual structure differs, these might need adjustment later.
try:
    from .query_processor import QueryProcessor
except ImportError:
    # TODO:  if QueryProcessor isn't defined yet or path is wrong
    QueryProcessor = Any
try:
    from .ranking import HierarchicalRanker
except ImportError:
    # TODO:  if HierarchicalRanker isn't defined yet or path is wrong
    HierarchicalRanker = Any
try:
    from .response_builder import ResponseBuilder
except ImportError:
    # TODO:  if ResponseBuilder isn't defined yet or path is wrong
    ResponseBuilder = Any


# TODO:  for a potential SearchClient interface/type
# from .search_client import SearchClient # Uncomment when SearchClient is defined


class RetrievalManager:
    """
    Orchestrates the hierarchical retrieval process, coordinating query processing,
    searching, ranking, and response building.
    """

    def __init__(
        self,
        query_processor: QueryProcessor,
        ranker: HierarchicalRanker,
        response_builder: ResponseBuilder,
        search_client: Any,  # Replace 'Any' with 'SearchClient' type hint later
        # TODO: Add other necessary clients or configurations (e.g., vector DB connection)
    ):
        """
        Initializes the RetrievalManager with its core components.

        Args:
            query_processor: An instance of QueryProcessor.
            ranker: An instance of HierarchicalRanker.
            response_builder: An instance of ResponseBuilder.
            search_client: Client/interface for performing searches (e.g., against a vector DB).
        """
        self.query_processor = query_processor
        self.ranker = ranker
        self.response_builder = response_builder
        self.search_client = search_client
        self.debug_info: Dict[str, Any] = (
            {}
        )  # To store intermediate results for debugging

    def _perform_search(
        self, query_embedding: List[float], match_count: int = 10, **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Performs semantic search using the configured search client.

        Args:
            query_embedding: The embedding vector of the query.
            match_count: The maximum number of results to retrieve.
            **kwargs: Additional parameters (currently unused, for future filters).

        Returns:
            A list of node dictionaries matching the query embedding.
        """
        raw_results: List[Dict[str, Any]] = []
        self.debug_info["raw_search_results"] = raw_results  # Initialize in debug info

        if not self.search_client:
            print(
                "Error: Search client is not configured.", flush=True
            )  # Replace with logging
            return raw_results

        # Assuming search_client has a method like 'match_nodes' that calls
        # the 'match_hierarchical_nodes' SQL function.
        # Adjust the method name and parameters if the actual client differs.
        try:
            # TODO: Confirm the actual method name and signature on search_client
            #       It might be rpc('match_hierarchical_nodes', {...}) or a wrapper.
            #       Using 'vector_search' which calls the 'match_hierarchical_nodes' RPC.
            print(
                f"Performing semantic search with match_count={match_count}", flush=True
            )  # TODO:  log
            raw_results = self.search_client.vector_search(
                embedding=query_embedding,  # Corrected parameter name
                match_count=match_count,
                # TODO: Add filter parameters from kwargs if/when implemented
            )
            # Ensure results are a list, even if the client returns None or something else unexpectedly
            if not isinstance(raw_results, list):
                print(
                    f"Warning: Search client did not return a list. Received: {type(raw_results)}. Returning empty list.",
                    flush=True,
                )
                raw_results = []

            print(f"Found {len(raw_results)} raw results.", flush=True)  # TODO:  log
            self.debug_info["raw_search_results"] = raw_results  # Store actual results
        except AttributeError:
            # This specific error might be less likely now, but keep for safety
            print(
                f"Error: Search client object {self.search_client} does not have the method 'vector_search'. Check client implementation and initialization.",
                flush=True,
            )
            raw_results = []
            self.debug_info["raw_search_results"] = raw_results
        except Exception as e:
            # TODO: Implement more specific error handling and logging
            print(f"Error during search: {e}", flush=True)
            # Return empty list on error
            raw_results = []
            self.debug_info["raw_search_results"] = (
                raw_results  # Ensure it's empty on error
            )

        return raw_results

    def retrieve(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Processes a user query, retrieves relevant information, ranks it,
        and builds a final response.

        Args:
            query: The user's natural language query.
            **kwargs: Optional dictionary of additional parameters (e.g., filters, match_count).

        Returns:
            A dictionary representing the final structured response, or an error dictionary.
        """
        self.debug_info = {}  # Reset debug info for new query
        search_results: List[Dict[str, Any]] = []
        ranked_results: List[Dict[str, Any]] = []
        final_response: Dict[str, Any] = {}

        try:
            # 1. Process Query
            # Use process_query as requested, assuming it takes only the query string
            processed_query = self.query_processor.process_query(query)
            self.debug_info["processed_query"] = processed_query

            # Basic check for processing output validity
            if not isinstance(processed_query, dict):
                print(
                    f"Error: QueryProcessor did not return a dictionary for query: {query}",
                    flush=True,
                )  # Replace with logging
                return {
                    "error": "Query processing failed",
                    "details": "Invalid output format from QueryProcessor",
                }

            # 2. Perform Search based on query type
            is_path_query = processed_query.get("is_path_query", False)
            self.debug_info["is_path_query"] = is_path_query

            if is_path_query:
                self.debug_info["search_type"] = "path"
                path_pattern = processed_query.get("original_query", "").strip()
                max_results = kwargs.get(
                    "match_count", 10
                )  # Use match_count or default to 10
                self.debug_info["path_pattern"] = path_pattern
                self.debug_info["match_count"] = max_results

                if not path_pattern:
                    print(
                        "Warning: Path query detected but original query (path) is empty.",
                        flush=True,
                    )
                    search_results = []
                elif not self.search_client:
                    print(
                        "Error: Search client is not configured for path search.",
                        flush=True,
                    )
                    search_results = []
                else:
                    try:
                        print(
                            f"Performing path search for pattern: '{path_pattern}' with max_results={max_results}",
                            flush=True,
                        )
                        # Assume search_client has find_nodes_by_path method
                        path_results = self.search_client.find_nodes_by_path(
                            path_pattern=path_pattern, max_results=max_results
                        )
                        # Ensure results are a list
                        if not isinstance(path_results, list):
                            print(
                                f"Warning: find_nodes_by_path did not return a list. Received: {type(path_results)}. Setting results to empty list.",
                                flush=True,
                            )
                            path_results = []

                        # Add default similarity score
                        search_results = []
                        for node in path_results:
                            if isinstance(node, dict):
                                node["similarity"] = 1.0  # Add default score
                                search_results.append(node)
                            else:
                                print(
                                    f"Warning: Skipping non-dictionary item in path results: {node}",
                                    flush=True,
                                )

                        print(
                            f"Found {len(search_results)} results via path search.",
                            flush=True,
                        )

                    except AttributeError:
                        print(
                            f"Error: Search client object {self.search_client} does not have the method 'find_nodes_by_path'. Check client implementation.",
                            flush=True,
                        )
                        search_results = []
                    except Exception as e:
                        print(
                            f"Error during path search for '{path_pattern}': {e}",
                            flush=True,
                        )
                        search_results = []

                # Ensure raw_search_results is also updated in debug_info for consistency
                self.debug_info["raw_search_results"] = (
                    search_results  # Store potentially modified results (with score)
                )

            else:
                # Semantic Search
                self.debug_info["search_type"] = "semantic"
                query_embedding = processed_query.get("embedding")

                # Check if embedding exists for semantic search
                if not query_embedding or not isinstance(query_embedding, list):
                    print(
                        f"Error: No valid embedding found in processed query for semantic search: {query}",
                        flush=True,
                    )
                    return {"error": "Missing or invalid embedding for semantic search"}

                # Get match_count from kwargs or use default
                match_count = kwargs.get(
                    "match_count", 10
                )  # Default to 10 as requested
                self.debug_info["match_count"] = match_count

                # Call internal search method
                search_results = self._perform_search(
                    query_embedding=query_embedding,
                    match_count=match_count,
                    # Pass other relevant kwargs if _perform_search supports them
                )
                # self.debug_info['raw_search_results'] is set within _perform_search

            # 3. Rank Results
            # Use rerank_results as requested
            # Assuming ranker handles empty search_results gracefully
            ranked_results = self.ranker.rerank_results(search_results)
            self.debug_info["ranked_results"] = ranked_results

            # 4. Build Response
            # Use build_response as requested
            # Assuming builder handles empty ranked_results gracefully
            final_response = self.response_builder.build_response(ranked_results)
            self.debug_info["final_response"] = final_response  # Store for debugging

        except Exception as e:
            # Catch-all for unexpected errors during orchestration
            print(
                f"Unexpected error during retrieval for query '{query}': {e}",
                flush=True,
            )  # Replace with proper logging
            self.debug_info["error"] = str(e)
            # Return a generic error response
            return {
                "error": "An unexpected error occurred during retrieval.",
                "details": str(e),
            }

        # 5. Return Final Response
        return final_response

    def get_debug_info(self) -> Dict[str, Any]:
        """Returns the debug information collected during the last retrieval."""
        return self.debug_info
