from typing import Dict, Any, List, Optional
from openai import AsyncOpenAI
from supabase import Client
import sys
import os
import json  # Added for potential JSON parsing if needed

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.utils import get_env_var

# Imports for hierarchical retrieval
try:
    from archon.llms_txt.retrieval.query_processor import QueryProcessor

    # from archon.llms_txt.retrieval.ranking import HierarchicalRanker # Not needed for direct search
    # from archon.llms_txt.retrieval.response_builder import ResponseBuilder # Not needed for direct search
    # from archon.llms_txt.retrieval.retrieval_manager import RetrievalManager # Not needed for direct search
    from archon.llms_txt.vector_db.supabase_manager import SupabaseManager

    HIERARCHICAL_IMPORTS_AVAILABLE = True
except ImportError as e:
    print(
        f"Warning: Failed to import hierarchical retrieval components: {e}. Hierarchical retrieval disabled."
    )
    HIERARCHICAL_IMPORTS_AVAILABLE = False

    # Define dummy classes if imports fail to prevent NameErrors later
    class QueryProcessor:
        pass

    class SupabaseManager:
        pass


embedding_model = get_env_var("EMBEDDING_MODEL") or "text-embedding-3-small"


async def get_embedding(text: str, embedding_client: Any) -> List[float]:
    """Get embedding vector from OpenAI or OpenAIEmbeddingGenerator."""
    try:
        # Check if embedding_client is an AsyncOpenAI client or OpenAIEmbeddingGenerator
        if hasattr(embedding_client, "embeddings"):
            # It's an AsyncOpenAI client
            response = await embedding_client.embeddings.create(
                model=embedding_model, input=text
            )
            return response.data[0].embedding
        elif hasattr(embedding_client, "generate_embedding"):
            # It's an OpenAIEmbeddingGenerator
            return embedding_client.generate_embedding(text)
        else:
            raise ValueError(
                f"Unsupported embedding client type: {type(embedding_client)}"
            )
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error


async def retrieve_relevant_documentation_tool(
    supabase: Client, embedding_client: Any, user_query: str
) -> str:
    """
    Retrieve relevant documentation chunks based on the query.
    Conditionally uses either standard RAG from 'site_pages' or hierarchical RAG
    from 'hierarchical_nodes' based on the DOCS_RETRIEVAL_TABLE environment variable.
    """
    docs_retrieval_table = get_env_var("DOCS_RETRIEVAL_TABLE") or "site_pages"
    match_count = 4  # Keep consistent with original logic

    try:
        if docs_retrieval_table == "hierarchical_nodes":
            if not HIERARCHICAL_IMPORTS_AVAILABLE:
                return "Error: Hierarchical retrieval selected, but required components could not be imported."

            print(f"--- Performing Hierarchical Retrieval for: '{user_query}' ---")
            try:
                # 1. Initialize Hierarchical Components
                # SupabaseManager handles its own env loading and client creation
                db_manager = SupabaseManager()
                # QueryProcessor handles its own env loading and embedder creation
                # We need to ensure QueryProcessor uses the *same* embedding client/model
                # Let's reuse the existing get_embedding function for simplicity and consistency
                # query_processor = QueryProcessor(embedding_client=embedding_client) # Pass existing client

                # 2. Process Query to get Embedding
                query_embedding = await get_embedding(user_query, embedding_client)

                if not query_embedding or len(query_embedding) < 10:  # Basic check
                    return (
                        "Error: Failed to generate embedding for hierarchical search."
                    )

                # 3. Perform Search using SupabaseManager
                # We bypass RetrievalManager/Ranker/ResponseBuilder to get full content directly
                search_results = db_manager.vector_search(
                    embedding=query_embedding,
                    match_count=match_count,
                    # TODO: Add filters if needed, e.g., metadata_filter={'source': 'pydantic_ai_docs'} ?
                    # The llms-txt structure might use different metadata keys. Check later if needed.
                )

                if not search_results:
                    return "No relevant hierarchical documentation found."

                # 4. Format Results (similar to site_pages logic)
                formatted_chunks = []
                for node in search_results:
                    # Extract relevant fields (adjust keys based on actual node structure)
                    title = node.get("title", "Untitled")
                    content = node.get("content", "")
                    path = node.get(
                        "path", ""
                    )  # Hierarchical path might be useful context
                    # Ensure similarity is a float before formatting
                    similarity = float(node.get("similarity", 0.0))  # Score

                    chunk_text = f"""
# {title} (Path: {path}, Score: {similarity:.4f})

{content}
"""
                    formatted_chunks.append(chunk_text)

                print(
                    f"--- Hierarchical Retrieval Found {len(formatted_chunks)} Chunks ---"
                )
                return "\n\n---\n\n".join(formatted_chunks)

            except Exception as hier_e:
                print(f"Error during hierarchical documentation retrieval: {hier_e}")
                # Optionally log traceback: import traceback; traceback.print_exc()
                return (
                    f"Error during hierarchical documentation retrieval: {str(hier_e)}"
                )

        else:  # Default to 'site_pages'
            print(
                f"--- Performing Standard Retrieval ('site_pages') for: '{user_query}' ---"
            )
            # Get the embedding for the query
            query_embedding = await get_embedding(user_query, embedding_client)

            # Query Supabase for relevant documents using the existing RPC
            result = supabase.rpc(
                "match_site_pages",
                {
                    "query_embedding": query_embedding,
                    "match_count": match_count,
                    "filter": {"source": "pydantic_ai_docs"},  # Keep original filter
                },
            ).execute()

            if not result.data:
                return "No relevant standard documentation found."

            # Format the results (original logic)
            formatted_chunks = []
            for doc in result.data:
                chunk_text = f"""
# {doc['title']}

{doc['content']}
"""
                formatted_chunks.append(chunk_text)

            print(f"--- Standard Retrieval Found {len(formatted_chunks)} Chunks ---")
            return "\n\n---\n\n".join(formatted_chunks)

    except Exception as e:
        print(f"Error retrieving documentation (outer scope): {e}")
        # Optionally log traceback: import traceback; traceback.print_exc()
        return f"Error retrieving documentation: {str(e)}"


async def list_documentation_pages_tool(supabase: Client) -> List[str]:
    """
    Function to retrieve a list of all available Pydantic AI documentation pages.
    This is called by the list_documentation_pages tool and also externally
    to fetch documentation pages for the reasoner LLM.

    Returns:
        List[str]: List of unique URLs for all documentation pages
    """
    try:
        # Query Supabase for unique URLs where source is pydantic_ai_docs
        result = (
            supabase.from_("site_pages")
            .select("url")
            .eq("metadata->>source", "pydantic_ai_docs")
            .execute()
        )

        if not result.data:
            return []

        # Extract unique URLs
        urls = sorted(set(doc["url"] for doc in result.data))
        return urls

    except Exception as e:
        print(f"Error retrieving documentation pages: {e}")
        return []


async def get_page_content_tool(supabase: Client, url: str) -> str:
    """
    Retrieve the full content of a specific documentation page by combining all its chunks.

    Args:
        ctx: The context including the Supabase client
        url: The URL of the page to retrieve

    Returns:
        str: The complete page content with all chunks combined in order
    """
    try:
        # Query Supabase for all chunks of this URL, ordered by chunk_number
        result = (
            supabase.from_("site_pages")
            .select("title, content, chunk_number")
            .eq("url", url)
            .eq("metadata->>source", "pydantic_ai_docs")
            .order("chunk_number")
            .execute()
        )

        if not result.data:
            return f"No content found for URL: {url}"

        # Format the page with its title and all chunks
        page_title = result.data[0]["title"].split(" - ")[0]  # Get the main title
        formatted_content = [f"# {page_title}\n"]

        # Add each chunk's content
        for chunk in result.data:
            formatted_content.append(chunk["content"])

        # Join everything together but limit the characters in case the page is massive (there are a coule big ones)
        # This will be improved later so if the page is too big RAG will be performed on the page itself
        return "\n\n".join(formatted_content)[:20000]

    except Exception as e:
        print(f"Error retrieving page content: {e}")
        return f"Error retrieving page content: {str(e)}"

def get_file_content_tool(file_path: str) -> str:
    """
    Retrieves the content of a specific file. Use this to get the contents of an example, tool, config for an MCP server

    Args:
        file_path: The path to the file
        
    Returns:
        The raw contents of the file
    """
    try:
        with open(file_path, "r") as file:
            file_contents = file.read()
        return file_contents
    except Exception as e:
        print(f"Error retrieving file contents: {e}")
        return f"Error retrieving file contents: {str(e)}"           
