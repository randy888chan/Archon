from utils.utils import get_env_var, reorder_by_content, write_to_log
from typing import Dict, Any, List, Optional
from openai import AsyncOpenAI
from supabase import Client
from pinecone import Pinecone
import logfire
import sys
import os
from archon.agent_prompts import LIBRARY_PROMPT
from pydantic import BaseModel, Field
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

embedding_model = get_env_var('EMBEDDING_MODEL') or 'text-embedding-3-small'

provider = get_env_var('LLM_PROVIDER') or 'OpenAI'
logger = logfire.configure(token=os.getenv('LOGFIRE_API_KEY'))
if provider == "OpenAI":
    logfire.instrument_openai()
elif provider == "Anthropic":
    logfire.instrument_anthropic()
else:
    logfire.instrument_aiohttp_client()


class DocumentationFocusState(BaseModel):
    """Represents the selected documentation focus based on the agent's requirements."""
    pydantic_ai_docs: bool = Field(
        default=False,
        description="Select if the agent requires robust Python-based agent construction with strong type safety, dependency injection, and structured response validation, especially for creating agentic tools within complex workflows like LangGraph."
    )
    langgraph_docs: bool = Field(
        default=False,
        description="Select if the agent involves complex, stateful, potentially cyclic workflows with multiple actors or requires human-in-the-loop interaction, using the Python implementation."
    )
    langgraphjs_docs: bool = Field(
        default=False,
        description="Select if the agent involves complex, stateful, potentially cyclic workflows with multiple actors, specifically using the LangGraph JavaScript/Next.js implementation."
    )
    langsmith_docs: bool = Field(
        default=False,
        description="Select if the primary focus is on production-level monitoring, debugging, evaluation, and observability of the LLM application."
    )
    langchain_python_docs: bool = Field(
        default=False,
        description="Select if the agent requires core LangChain Python features like standardized LLM interaction, chains, RAG, or basic agent capabilities."
    )
    langchain_js_docs: bool = Field(
        default=False,
        description="Select if the agent requires core LangChain JavaScript/TypeScript features like standardized LLM interaction, chains, RAG, or basic agent capabilities, typically within a Next.js environment."
    )


async def get_embedding(text: str, embedding_client: AsyncOpenAI) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        logger.info("started : get_embedding")
        response = await embedding_client.embeddings.create(
            model=embedding_model,
            input=text
        )
        logger.info("finished : get_embedding")
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error


async def retrieve_relevant_documentation_tool(supabase: Client, embedding_client: AsyncOpenAI, user_query: str, source_filter: Optional[str] = None) -> str:
    """
    Retrieves documentation chunks relevant to the user query, optionally filtered by source.

    Args:
        supabase: The Supabase client instance.
        embedding_client: The OpenAI client instance for embeddings.
        user_query: The query string to search for.
        source_filter: Optional source name to filter results.

    Returns:
        str: Formatted string containing relevant documentation chunks, or an error message.
    """
    try:
        logger.info("started : retrieve_relevant_documentation_tool")
        # Get the embedding for the query
        query_embedding = await get_embedding(user_query, embedding_client)

        # Prepare parameters for the RPC call
        rpc_params = {
            'query_embedding': query_embedding,
            'match_count': 4  # Keep the match count relatively low
        }

        # Add source filter to the RPC parameters if provided
        if source_filter:
            rpc_params['filter'] = {'source': source_filter}

        # Query Supabase for relevant documents
        result = supabase.rpc('match_site_pages', rpc_params).execute()
        logger.info("finished : retrieve_relevant_documentation_tool")
        if not result.data:
            return "No relevant documentation found."

        # Format the results, including the source in the output for clarity
        formatted_chunks = []
        for doc in result.data:
            source = doc.get('metadata', {}).get('source', 'Unknown Source')
            chunk_text = f"""
                        Source: {source}
                        URL: {doc['url']}
                        # {doc['title']}

                        {doc['content']}
                        """
            formatted_chunks.append(chunk_text)

        # Join all chunks with a separator
        formatted_chunks = "\n\n---\n\n".join(formatted_chunks)
        logger.info("finished : retrieve_relevant_documentation_tool")

        return formatted_chunks

    except Exception as e:
        print(f"Error retrieving documentation: {e}")
        return f"Error retrieving documentation: {str(e)}"


async def list_documentation_pages_tool(supabase: Client, source_filter: Optional[str] = None) -> List[str]:
    """
    Function to retrieve a list of all available documentation page URLs, optionally filtered by source.

    Args:
        supabase: The Supabase client instance.
        source_filter: Optional source name (e.g., 'pydantic_ai_docs') to filter URLs.

    Returns:
        List[str]: List of unique URLs for documentation pages matching the filter (or all if no filter).
    """
    try:
        logger.info("started : list_documentation_pages_tool")
        query = supabase.from_('site_pages').select('url')

        # Apply source filter if provided
        if source_filter:
            query = query.eq('metadata->>source', source_filter)

        result = query.execute()
        logger.info("finished : list_documentation_pages_tool: result returned")
        if not result.data:
            logger.warn(
                "finished : list_documentation_pages_tool: no data returned")
            return []

        # Extract unique URLs
        urls = sorted(set(doc['url'] for doc in result.data))
        logger.info(f"finished : list_documentation_pages_tool: {urls}")
        return urls

    except Exception as e:
        print(f"Error retrieving documentation pages: {e}")
        return []


async def get_page_content_tool(supabase: Client, url: str, source_filter: Optional[str] = None) -> str:
    """
    Retrieve the full content of a specific documentation page by combining all its chunks.
    Optionally filters by source, though URL should generally be unique.

    Args:
        supabase: The Supabase client instance.
        url: The URL of the page to retrieve.
        source_filter: Optional source name to ensure the URL belongs to the correct source.

    Returns:
        str: The complete page content with all chunks combined in order, or an error message.
    """
    try:
        logger.info("started : get_page_content_tool")
        # Query Supabase for all chunks of this URL, ordered by chunk_number
        query = supabase.from_('site_pages') \
            .select('title, content, chunk_number, metadata') \
            .eq('url', url)

        # Add source filter if provided (good practice even if URL is unique)
        if source_filter:
            query = query.eq('metadata->>source', source_filter)

        result = query.order('chunk_number').execute()

        if not result.data:
            if source_filter:
                return f"No content found for URL: {url} with source: {source_filter}"
            else:
                return f"No content found for URL: {url}"

        # Format the page with its title and all chunks
        # Attempt to get a clean title and source
        first_chunk = result.data[0]
        raw_title = first_chunk.get('title', 'Untitled Page')
        if ' - ' in raw_title:
            page_title = raw_title.split(' - ', 1)[0]
        else:
            page_title = raw_title
        source = first_chunk.get('metadata', {}).get(
            'source', 'Unknown Source')

        formatted_content = [f"Source: {source}\nURL: {url}\n# {page_title}\n"]

        # Add each chunk's content
        for chunk in result.data:
            formatted_content.append(
                chunk.get('content', ''))  # Use .get for safety

        # Join everything together but limit the characters
        # TODO: Implement RAG on the page itself if content is too large
        full_content = "\n\n".join(formatted_content)
        limit = 20000
        if len(full_content) > limit:
            print(
                f"Warning: Content for {url} truncated to {limit} characters.")
            full_content = full_content[:limit] + "\n\n... [Content Truncated]"
        logger.info("finished : get_page_content_tool")
        logger.notice(f"finished : get_page_content_tool: {full_content}")
        return full_content

    except Exception as e:
        print(f"Error retrieving page content for {url}: {e}")
        return f"Error retrieving page content for {url}: {str(e)}"


def get_file_content_tool(file_path: str) -> str:
    """
    Retrieves the content of a specific file.

    Args:
        file_path: The path to the file relative to the project root.

    Returns:
        The raw contents of the file or an error message.
    """
    try:
        logger.info("started : get_file_content_tool")
        # Ensure the path is relative to the project root (assuming this script is in 'archon')
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        absolute_path = os.path.join(base_dir, file_path)

        if not os.path.exists(absolute_path):
            return f"Error: File not found at {absolute_path}"

        with open(absolute_path, "r", encoding='utf-8') as file:
            file_contents = file.read()
        logger.info("finished : get_file_content_tool")
        logger.notice(f"finished : get_file_content_tool: {file_contents}")
        return file_contents
    except Exception as e:
        print(f"Error retrieving file contents for {file_path}: {e}")
        return f"Error retrieving file contents: {str(e)}"


# --- Helper Functions for Pinecone Retrieval ---

async def find_relevant_libraries(text: str, client: AsyncOpenAI, model_name: str) -> list[str]:
    """Uses LLM to determine relevant documentation libraries based on user input."""
    try:
        logger.info("started : find_relevant_libraries")
        response = await client.chat.completions.create(
            model=model_name,
            response_model=DocumentationFocusState,
            messages=[
                {"role": "system", "content": LIBRARY_PROMPT},
                {"role": "user", "content": text}
            ],
        )
        logger.info(f"Response: {response} and type: {type(response)}")

        focus_state: DocumentationFocusState = response
        state_dict = focus_state.model_dump()
        relevant_libs = [key for key, value in state_dict.items() if value]
        logger.info("finished : find_relevant_libraries")
        logger.info(f"finished : find_relevant_libraries: {relevant_libs}")
        return relevant_libs
    except Exception as e:
        print(f"Error finding relevant libraries: {e}")
        return []


async def enhance_query_for_embedding(query: str, client: AsyncOpenAI, model_name: str) -> str:
    """Enhances the user query for better semantic search results."""
    prompt = f"""
    Analyze the following user query for creating an AI agent.
    Rephrase it to be more specific and technically detailed, suitable for semantic search against technical documentation.
    Focus on extracting key technical terms, libraries, concepts, and the core task.
    Keep it concise.

    Original Query: {query}

    Enhanced Query for Semantic Search:
    """
    try:
        logger.info("started : enhance_query_for_embedding")
        response = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system",
                    "content": "You are an AI expert and software documentation, you're tasked with enhancing \
                        the user's query for better semantic search results pertaining to the documenation about\
                            Langchain, Langgraph and Pydantic AI. The basis of your knowledge is the following \
                                documentation: " + LIBRARY_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        enhanced_query = response.choices[0].message.content.strip()
        logger.info("finished : enhance_query_for_embedding")
        logger.info(
            f"finished : enhance_query_for_embedding: {enhanced_query}")

        return enhanced_query
    except Exception as e:
        print(f"Error enhancing query: {e}")
        return query


async def get_query_embedding(text: str, embedding_client: AsyncOpenAI, model_name: str) -> list[float]:
    """Generates embeddings for the given text using the specified model."""
    try:
        logger.info("started : get_query_embedding")
        response = await embedding_client.embeddings.create(
            input=[text],
            model=model_name
        )
        logger.info("finished : get_query_embedding")
        logger.info(
            f"finished : get_query_embedding: {response.data[0].embedding}")
        enhanced_query_vector = response.data[0].embedding
        return enhanced_query_vector
    except Exception as e:
        print(f"Error generating query embedding: {e}")
        return [0.0]


async def query_pinecone_with_filter(pinecone_client: Pinecone,
                                     index: Pinecone.Index,
                                     enhanced_query: str,
                                     relevant_libraries: list[str],
                                     query_vector: list[float],
                                     top_k: int = 100,
                                     top_n: int = 10) -> Optional[list[dict]]:
    """Queries Pinecone index with vector and metadata filter."""
    if index is None:
        print("Pinecone index not available. Skipping query.")
        return None
    if not query_vector:
        print("No query vector provided. Skipping Pinecone query.")
        return None

    query_filter = {}
    if not relevant_libraries:
        relevant_libraries = ['langchain_python_docs', 'langchain_js_docs',
                              'langgraph_docs', 'langgraphjs_docs', 'langsmith_docs', 'pydantic_ai_docs']

    try:
        query_response = index.query_namespaces(
            vector=query_vector,
            namespaces=relevant_libraries,
            filter=query_filter if query_filter else None,
            metric="cosine",
            top_k=top_k,
            include_values=True,
            include_metadata=True,
            show_progress=True,
        )
        print(
            f"Pinecone query returned {len(query_response.get('matches', []))} matches.")

        original_ordered_list = []
        content_rerank = []
        new_ordered_matches = []

        for scored_vec in query_response.matches:
            original_ordered_list.append(
                {"content": scored_vec.metadata['content'], "url": scored_vec.metadata['url']})
            truncate_content = scored_vec.metadata['content'][:1000]
            content_rerank.append(truncate_content)

        reranked_results = pinecone_client.inference.rerank(
            model="bge-reranker-v2-m3",
            query=enhanced_query,
            documents=content_rerank,
            top_n=top_n,
            return_documents=True
        )
        # Since we had to turncate the content, we need to update  the original_ordered_list based on the reranked results
        for reranked_result in reranked_results.data:
            new_ordered_matches.append(reranked_result.document.text)

        try:
            original_list_reranked = reorder_by_content(
                new_ordered_matches,
                original_ordered_list,
                max_chunk=100,
                similarity_threshold=0.8
            )
            print("Reordered content:", original_list_reranked)
        except (ValueError, RuntimeError) as e:
            print(f"Error during reordering: {e}")
        pinecone_response = original_list_reranked
        return pinecone_response
    except Exception as e:
        print(f"Error querying Pinecone: {e}")
        return None


def extract_urls_from_pinecone_results(pinecone_response: list[dict]) -> list[str]:
    """Extracts URLs from Pinecone query results.

    Args:
        pinecone_response: List of dictionaries containing 'content' and 'url' keys

    Returns:
        list[str]: List of URLs extracted from the response
    """
    urls = []
    if not pinecone_response:
        return urls

    for item in pinecone_response:
        if url := item.get('url'):
            urls.append(url)
        else:
            print(f"Warning: Missing URL in response item")

    print(f"Extracted {len(urls)} URLs from Pinecone response.")
    return urls
