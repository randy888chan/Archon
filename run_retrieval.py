import argparse
import json
import sys
from pathlib import Path

# Add project root to path to allow importing archon modules
# Assumes the script is run from the project root directory
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    # Handle cases where the script might be run from a different CWD
    # This ensures 'archon' can be found
    sys.path.insert(0, str(project_root))

# Import necessary components
try:
    from archon.llms_txt.retrieval.query_processor import QueryProcessor
    from archon.llms_txt.retrieval.ranking import HierarchicalRanker
    from archon.llms_txt.retrieval.response_builder import ResponseBuilder
    from archon.llms_txt.retrieval.retrieval_manager import RetrievalManager
    from archon.llms_txt.vector_db.supabase_manager import SupabaseManager
    # EnvironmentLoader is used implicitly by SupabaseManager and EmbeddingManager (in QueryProcessor)
    # No need to explicitly import if components handle their own loading
    # from archon.llms_txt.utils.env_loader import EnvironmentLoader
except ImportError as e:
    print(f"Error importing required Archon components: {e}")
    print(f"Attempted sys.path: {sys.path}")
    print("Please ensure the Archon package structure is correct relative to the project root and all dependencies are installed.")
    sys.exit(1)

def run_retrieval_query(query: str, match_count: int = 10):
    """
    Initializes retrieval components and runs a query.

    Args:
        query (str): The user query.
        match_count (int): Number of results to retrieve initially.
    """
    print(f"--- Running Retrieval Query ---")
    print(f"Query: '{query}'")
    print(f"Match Count: {match_count}")

    try:
        # Instantiate retrieval components
        print("Initializing retrieval components...")
        # Components should handle their own environment loading via EnvironmentLoader

        # QueryProcessor initializes its own embedder
        query_processor = QueryProcessor()

        # Ranker - basic pass-through for now
        ranker = HierarchicalRanker()

        # ResponseBuilder
        response_builder = ResponseBuilder()

        # SupabaseManager acts as the search client
        db_client = SupabaseManager()

        # RetrievalManager orchestrates the process
        retrieval_manager = RetrievalManager(
            search_client=db_client,
            query_processor=query_processor,
            ranker=ranker,
            response_builder=response_builder
        )
        print("Retrieval components initialized.")

        # Execute the retrieval process
        print("Executing retrieval...")
        # Pass match_count as a keyword argument to retrieve method
        retrieval_results = retrieval_manager.retrieve(query, match_count=match_count)

        # Print the results
        print("\n--- Retrieval Results ---")
        if retrieval_results:
            # Pretty print the JSON output
            print(json.dumps(retrieval_results, indent=2, ensure_ascii=False))
        else:
            # Check debug info for potential errors if results are empty/None
            debug_info = retrieval_manager.get_debug_info()
            if debug_info.get("error"):
                 print(f"(Retrieval failed: {debug_info.get('error')})")
            else:
                 print("(No results returned)")

        # Optionally print debug info for detailed analysis
        # print("\n--- Debug Info ---")
        # print(json.dumps(retrieval_manager.get_debug_info(), indent=2, ensure_ascii=False))

    except Exception as e:
        print(f"\nAn unexpected error occurred during the retrieval process: {e}")
        # Uncomment the following lines for full traceback during development
        # import traceback
        # traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(
        description="Run a standalone retrieval query against the hierarchical vector database."
    )
    parser.add_argument(
        "--query", "-q",
        required=True,
        help="The query string to search for."
    )
    parser.add_argument(
        "--match-count", "-k", type=int, default=10,
        help="Number of initial results to retrieve (default: 10)."
    )
    # TODO: Add arguments for filters (metadata, section, level, content_type) later

    args = parser.parse_args()

    run_retrieval_query(args.query, args.match_count)

if __name__ == "__main__":
    main()