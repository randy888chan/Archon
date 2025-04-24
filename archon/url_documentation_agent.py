"""
Enhanced URL documentation agent with better config handling and logging.
"""

import os
import asyncio
import logging
from typing import List, Dict, Any, Optional
import logfire

from utils.utils import get_env_var, write_to_log
from pydantic_ai.settings import ModelSettings
from archon.refiner_agents.rag_refiner_agent import run_rag_refiner, RagProcessingService
from instructor import AsyncInstructor
from openai import AsyncOpenAI
from pinecone import Pinecone
from archon.agent_tools import enhance_query_for_embedding

# Setup logging
logger = logfire.configure(token=os.getenv('LOGFIRE_API_KEY'))

# Load environment variables
api_key = get_env_var('LLM_API_KEY') or 'no-llm-api-key-provided'
base_url = get_env_var('BASE_URL') or 'https://api.openai.com/v1'

# Initialize clients - moved to function to allow lazy loading and better error handling


def get_rag_clients():
    """Get clients needed for RAG, with proper error handling"""
    try:
        raw_client = AsyncOpenAI(
            api_key=api_key, base_url=base_url, timeout=300.0)
        llm_client = AsyncInstructor.from_openai(raw_client)

        # Initialize Pinecone
        pinecone_api_key = get_env_var(
            'PINECONE_API_KEY') or 'no-pinecone-api-key-provided'
        pinecone_index_name = get_env_var(
            'PINECONE_INDEX_NAME') or 'ai-docs-index'

        pinecone_client = Pinecone(api_key=pinecone_api_key)
        pinecone_index = pinecone_client.Index(pinecone_index_name)

        logger.info(f"Successfully initialized RAG clients")
        return raw_client, llm_client, pinecone_client, pinecone_index
    except Exception as e:
        logger.error(f"Error initializing RAG clients: {e}")
        return None, None, None, None


async def url_documentation_agent(state: dict, config: Optional[Dict[str, Any]] = None) -> dict:
    """
    Process documentation search with explicit config handling

    Args:
        state: The graph state containing query information
        config: Optional configuration dictionary, properly typed for the graph

    Returns:
        dict: Updated state with retrieved URLs
    """
    logger.info("started : url_documentation_agent (Direct Execution Mode)")

    # Log configuration information for debugging
    if config:
        logger.info(f"Received config: {config}")
        thread_id = config.get('thread_id', 'default')
        recursion_limit = config.get('recursion_limit', 300)
        logger.info(
            f"Using thread_id={thread_id}, recursion_limit={recursion_limit}")
    else:
        logger.info("No config provided, using defaults")

    # Get the initial query from state
    initial_query = state.get('latest_user_message', '')
    if not initial_query:
        initial_query = state.get('enhanced_query', 'No query provided')

    # Truncate for logging
    truncated_query = initial_query[:100] + \
        '...' if len(initial_query) > 100 else initial_query
    logger.info(
        f"url_documentation_agent: preparing to execute RAG pipeline for query: '{truncated_query}'")

    # Configure model settings with extended timeout - create proper ModelSettings object
    # Extended timeout to 5 minutes (300s)
    model_settings = ModelSettings(timeout=300.0)
    logger.info(
        f"Created ModelSettings with timeout={model_settings['timeout']}s")

    try:
        # Get RAG clients
        embedding_client, llm_client, pinecone_client, pinecone_index = get_rag_clients()

        if not all([embedding_client, llm_client, pinecone_client, pinecone_index]):
            logger.error(
                "Failed to initialize one or more required clients for RAG pipeline")
            return {"retrieve_urls": ["https://langchain-ai.github.io/langgraph/", "https://ai.pydantic.dev/"]}

        # Enhance the query for embedding
        logger.info("started : enhance_query_for_embedding")
        try:
            enhance_query_for_embedding_result = await enhance_query_for_embedding(
                query=initial_query,
                client=embedding_client,
                model_name="text-embedding-3-small"
            )
            logger.info(
                f"finished : enhance_query_for_embedding: {enhance_query_for_embedding_result[:100]}...")
        except Exception as e:
            logger.error(f"Error enhancing query: {e}")
            enhance_query_for_embedding_result = initial_query

        # Setup RAG service dependencies
        deps = RagProcessingService(
            llm_client=llm_client,
            embedding_client=embedding_client,
            pinecone_client=pinecone_client,
            pinecone_index=pinecone_index
        )

        # Start tracking progress with heartbeat
        heartbeat_count = 0

        async def heartbeat():
            nonlocal heartbeat_count
            while True:
                heartbeat_count += 1
                logger.info(
                    f"RAG pipeline heartbeat #{heartbeat_count} - still processing")
                await asyncio.sleep(10)  # Send heartbeat every 10 seconds

        # Start heartbeat task
        heartbeat_task = asyncio.create_task(heartbeat())

        try:
            # Run the RAG pipeline with a timeout
            filtered_urls = await asyncio.wait_for(
                run_rag_refiner(
                    enhance_query_for_embedding_result, deps, model_settings),
                # 4.5 minutes (allow buffer for the 5 minute timeout)
                timeout=270
            )
        except asyncio.TimeoutError:
            logger.error(f"RAG pipeline timed out after 270 seconds")
            filtered_urls = [
                "https://langchain-ai.github.io/langgraph/",
                "https://ai.pydantic.dev/"
            ]
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}")
            filtered_urls = [
                "https://langchain-ai.github.io/langgraph/",
                "https://ai.pydantic.dev/"
            ]
        finally:
            # Stop the heartbeat task
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass

        logger.info(
            f"finished : url_documentation_agent: returning {len(filtered_urls)} URLs")
        return {"retrieve_urls": filtered_urls}

    except Exception as e:
        logger.error(f"Error in url_documentation_agent: {e}")
        # Return fallback URLs
        return {"retrieve_urls": ["https://langchain-ai.github.io/langgraph/", "https://ai.pydantic.dev/"]}
