# run_retrieval.py
import argparse
from archon.llms_txt.retrieval.query_processor import QueryProcessor
from archon.llms_txt.retrieval.ranking import HierarchicalRanker
from archon.llms_txt.retrieval.response_builder import ResponseBuilder
from archon.llms_txt.retrieval.retrieval_manager import RetrievalManager

def main():
    parser = argparse.ArgumentParser(description="Run the hierarchical RAG retrieval process.")
    parser.add_argument("query", type=str, help="The query to process.")
    parser.add_argument("--section-filter", type=str, help="Filter results by section (e.g., 'Introduction').")
    parser.add_argument("--content-type-filter", type=str, help="Filter results by content type (e.g., 'API Reference').")
    # Add other potential filter arguments here as needed

    args = parser.parse_args()

    # Instantiate retrieval components
    query_processor = QueryProcessor()
    ranker = HierarchicalRanker()
    response_builder = ResponseBuilder()

    # TODO: Instantiate actual search client/vector DB interface based on configuration
    search_client = None # Placeholder

    retrieval_manager = RetrievalManager(
        query_processor=query_processor,
        ranker=ranker,
        response_builder=response_builder,
        search_client=search_client
    )

    # Perform retrieval
    # TODO: Pass filter arguments (args.section_filter, args.content_type_filter, etc.)
    #       to the retrieve method once implemented in RetrievalManager.
    result = retrieval_manager.retrieve(query=args.query)

    # Print the result
    print("Retrieval Result:")
    print(result) # Adjust printing format as needed based on the actual return type

if __name__ == "__main__":
    main()