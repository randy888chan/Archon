import re
from archon.llms_txt.vector_db.embedding_manager import EmbeddingManager
from typing import Any, Dict, List, Optional, Tuple

class QueryProcessor:
    """
    Processes user queries to extract relevant information, detect query types,
    and prepare them for retrieval.
    """

    def __init__(self):
        """
        Initializes the QueryProcessor and the embedding generator.
        """
        try:
            self.embedder = EmbeddingManager()
            print("QueryProcessor: EmbeddingManager initialized.")
        except Exception as e:
            print(f"Error initializing EmbeddingManager in QueryProcessor: {e}")
            self.embedder = None # Handle initialization failure

    def extract_contextual_info(self, query: str) -> Dict[str, Any]:
        """
        Extracts contextual information like requested sections or preferences.
        (Basic implementation using pattern matching for now).

        Args:
            query (str): The user query.

        Returns:
            Dict[str, Any]: A dictionary containing extracted context.
        """
        context = {}
        # TODO: Implement more sophisticated context extraction (e.g., NLP techniques)
        # Example: Look for keywords indicating specific sections
        if "introduction" in query.lower():
            context["requested_section"] = "introduction"
        elif "conclusion" in query.lower():
            context["requested_section"] = "conclusion"
        # Add more patterns as needed
        return context

    def detect_path_query(self, query: str) -> bool:
        """
        Detects if the query is likely a path-based query (e.g., using '>' or '/').

        Args:
            query (str): The user query.

        Returns:
            bool: True if a path-like structure is detected, False otherwise.
        """
        # Simple check for common path separators
        if re.search(r'[>/]', query):
            return True
        return False

    def generate_embeddings(self, query: str) -> List[float]:
        """
        Generates embeddings for the given query using the initialized embedder.

        Args:
            query (str): The user query.

        Returns:
            List[float]: The generated embedding vector.

        Raises:
            ValueError: If the embedder was not initialized successfully.
            Exception: If embedding generation fails.
        """
        if self.embedder is None:
            raise ValueError("Embedder not initialized. Cannot generate embeddings.")

        try:
            embedding = self.embedder.generate_embedding(query)
            return embedding
        except Exception as e:
            print(f"Error generating embedding for query '{query}': {e}")
            # Depending on desired behavior, could return empty list or re-raise
            raise Exception(f"Embedding generation failed: {e}") from e

    def create_hybrid_queries(self, query: str) -> List[str]:
        """
        Creates variations of the query for hybrid search. (Placeholder)

        Args:
            query (str): The original user query.

        Returns:
            List[str]: A list of query variations (including the original).
        """
        # TODO: Implement logic for generating hybrid query variations
        # (e.g., keyword extraction, synonym expansion)
        print(f"INFO: Hybrid query creation for query: '{query}' (Not implemented)")
        return [query] # Placeholder, returns original query

    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Orchestrates the query processing steps: path detection and embedding generation.

        Args:
            query (str): The raw user query.

        Returns:
            Dict[str, Any]: A dictionary containing the processed query information,
                            including original query, path detection result, and embedding.
                            Returns None for embedding if generation fails.
        """
        is_path = self.detect_path_query(query)
        embedding_vector = None # Default to None
        try:
            embedding_vector = self.generate_embeddings(query)
        except ValueError as ve: # Handle embedder initialization error
             print(f"Processing query failed: {ve}")
             # Decide how to handle this - maybe return partial data or raise
        except Exception as e: # Handle embedding generation error
            print(f"Error generating embeddings during query processing: {e}")
            # Embedding vector remains None

        return {
            "original_query": query,
            "embedding": embedding_vector,
            "is_path_query": is_path,
            # Add other processed info later (e.g., extracted context)
        }

# Example Usage (Optional - for testing during development)
if __name__ == '__main__':
    processor = QueryProcessor()
    test_query_1 = "Tell me about the introduction section."
    test_query_2 = "Details on installation > configuration"
    test_query_3 = "How does the system handle errors?"

    processed_1 = processor.process_query(test_query_1)
    print("Processed Query 1:", processed_1)

    processed_2 = processor.process_query(test_query_2)
    print("\nProcessed Query 2:", processed_2)

    processed_3 = processor.process_query(test_query_3)
    print("\nProcessed Query 3:", processed_3)