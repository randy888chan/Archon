# Phase 5 Implementation Plan: Step-by-Step Instructions

## Overview
This implementation plan outlines the steps to build the retrieval strategy for the hierarchical RAG system, focusing on three key components:
1. Context-aware querying
2. Hierarchical results ranking
3. Response generation with preserved context

## Project Setup Steps

1. **Create Directory Structure**
   - Create the `retrieval` directory inside the `archon/llms_txt/` folder
   - Add an empty `__init__.py` file to make it a proper package
   - Create placeholder files for the four main components

2. **Implement Query Processor (`query_processor.py`)**
   - Create the `QueryProcessor` class that:
     - Extracts contextual information from queries (sections, content preferences)
     - Detects path-based queries (using ">" or "/" as separators)
     - Generates query embeddings
     - Creates hybrid query variations for improved retrieval

3. **Implement Hierarchical Ranker (`ranking.py`)**
   - Create the `HierarchicalRanker` class that:
     - Reranks search results based on multiple factors
     - Implements scoring for: semantic similarity, hierarchical relevance, content type match, reference relationships, and path matching
     - Includes utilities for path comparison with fuzzy matching

4. **Implement Response Builder (`response_builder.py`)**
   - Create the `ResponseBuilder` class that:
     - Builds structured responses from search results
     - Preserves hierarchical context in response blocks
     - Includes parent context where relevant
     - Extracts source citations
     - Identifies related sections as suggestions
     - Formats responses as structured markdown

5. **Implement Retrieval Manager (`retrieval_manager.py`)**
   - Create the `RetrievalManager` class that:
     - Orchestrates the entire retrieval process
     - Handles path-based and semantic searches
     - Integrates query processing, search, ranking, and response generation
     - Provides debugging information for analysis

6. **Create Command-Line Interface (`run_retrieval.py`)**
   - Build a CLI script that:
     - Takes a query and optional parameters
     - Invokes the retrieval manager
     - Formats and displays results
     - Provides options for section filtering, content type filtering, etc.

## Testing and Evaluation Steps

7. **Implement Evaluation Framework (`utils/evaluation.py`)**
   - Create methods to:
     - Generate test queries from various categories
     - Compare retrieval results against reference answers
     - Calculate precision, recall, and relevance metrics
     - Measure the impact of hierarchical context on results quality

8. **Create Test Cases**
   - Develop test queries in these categories:
     - Direct section references (e.g., "Tell me about Fields in API documentation")
     - Conceptual queries (e.g., "How does validation work in Pydantic?")
     - Cross-section queries (e.g., "Difference between Fields in API and Concepts")
     - Path-based queries (e.g., "Pydantic > API Documentation > Fields")

9. **Run A/B Tests**
   - Compare results from the hierarchical RAG system against:
     - Traditional flat chunking approach
     - Non-hierarchical vector search
     - Measure improvements in relevance, context quality, and accuracy

## Integration Steps

10. **Update the Main Pipeline**
    - Modify `run_processing.py` to include:
      - Optional test query execution using the new retrieval system
      - Integration with previous phases (1-4)

11. **Create Example Usage Script**
    - Build a simple script demonstrating how to:
      - Process a markdown document through all phases
      - Generate embeddings and store in the vector database
      - Execute queries against the processed document
      - Display formatted responses with preserved context

## Next Steps and Improvements

12. **Plan for Future Enhancements**
    - Prepare for:
      - Multi-document support with cross-document references
      - Feedback loop integration for improved retrieval
      - Hybrid retrievers combining dense and sparse retrievals
      - UI integration for interactive document exploration

By following these steps, you'll implement a complete hierarchical retrieval strategy that preserves document structure, provides contextually relevant responses, and maintains clear citations to source sections.