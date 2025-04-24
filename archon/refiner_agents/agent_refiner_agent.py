from __future__ import annotations as _annotations
from archon.agent_tools import (
    retrieve_relevant_documentation_tool,
    list_documentation_pages_tool,
    get_page_content_tool,
)
from archon.agent_prompts import agent_refiner_prompt
from utils.utils import get_env_var

from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import httpx
import os
import sys
import json
from typing import List, Optional
from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from openai import AsyncOpenAI
from supabase import Client
from pinecone import Pinecone
# Add the parent directory to sys.path to allow importing from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

load_dotenv()

provider = get_env_var('LLM_PROVIDER') or 'OpenAI'
llm = get_env_var('PRIMARY_MODEL') or 'gpt-4o-mini'
base_url = get_env_var('BASE_URL') or 'https://api.openai.com/v1'
api_key = get_env_var('LLM_API_KEY') or 'no-llm-api-key-provided'
logfire.configure(token=os.getenv('LOGFIRE_API_KEY'))

if provider == "Anthropic":
    model = AnthropicModel(model_name=llm, api_key=api_key)
    logfire.instrument_anthropic()
elif provider == "OpenAI":
    logfire.instrument_openai()
    openai_provider = OpenAIProvider(api_key=api_key, base_url=base_url)
    model = OpenAIModel(model_name=llm,
                        provider=openai_provider)
else:
    openai_provider = OpenAIProvider(api_key=api_key, base_url=base_url)
    model = OpenAIModel(model_name=llm,
                        provider=openai_provider)
    logfire.instrument_aiohttp_client()


embedding_model = get_env_var('EMBEDDING_MODEL') or 'text-embedding-3-small'


@dataclass
class AgentRefinerDeps:
    supabase: Optional[Client]
    llm_client: Optional[AsyncOpenAI]
    pinecone_client: Optional[Pinecone]
    pinecone_index: Optional[Pinecone.Index]
    embedding_client: Optional[AsyncOpenAI]
    text_query: Optional[str]


agent_refiner_agent = Agent(
    model,
    system_prompt=agent_refiner_prompt,
    deps_type=AgentRefinerDeps,
    retries=2
)


@agent_refiner_agent.tool
async def retrieve_relevant_documentation(ctx: RunContext[AgentRefinerDeps], query: str) -> str:
    """
    Retrieve relevant documentation chunks based on the query with RAG.
    Make sure your searches always focus on implementing the agent itself.

    Args:
        ctx: The context including the Supabase client and OpenAI client
        query: Your query to retrieve relevant documentation for implementing agents

    Returns:
        A formatted string containing the top 4 most relevant documentation chunks
    """
    return await retrieve_relevant_documentation_tool(ctx.deps.supabase, ctx.deps.embedding_client, query)


@agent_refiner_agent.tool
async def list_documentation_pages(ctx: RunContext[AgentRefinerDeps]) -> List[str]:
    """
    Retrieve a list of all available Pydantic AI documentation pages.
    This will give you all pages available, but focus on the ones related to configuring agents and their dependencies.

    Returns:
        List[str]: List of unique URLs for all documentation pages
    """
    return await list_documentation_pages_tool(ctx.deps.supabase)


@agent_refiner_agent.tool
async def get_page_content(ctx: RunContext[AgentRefinerDeps], url: str) -> str:
    """
    Retrieve the full content of a specific documentation page by combining all its chunks.
    Only use this tool to get pages related to setting up agents with Pydantic AI.

    Args:
        ctx: The context including the Supabase client
        url: The URL of the page to retrieve

    Returns:
        str: The complete page content with all chunks combined in order
    """
    return await get_page_content_tool(ctx.deps.supabase, url)
