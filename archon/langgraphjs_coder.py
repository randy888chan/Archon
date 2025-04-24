from __future__ import annotations as _annotations
from archon.agent_tools import (
    retrieve_relevant_documentation_tool,
    list_documentation_pages_tool,
    get_page_content_tool
)
# Assuming same prompt for now
from archon.agent_prompts import primary_coder_prompt
from utils.utils import get_env_var

from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import httpx
import os
import sys
import json
from typing import List
from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.messages import ModelMessagesTypeAdapter
from openai import AsyncOpenAI
from supabase import Client
from pydantic_ai.providers.openai import OpenAIProvider
# Add the parent directory to sys.path to allow importing from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

# --- Configuration ---
PROVIDER = get_env_var('LLM_PROVIDER') or 'OpenAI'
LLM_MODEL_NAME = get_env_var('PRIMARY_MODEL') or 'gpt-4o-mini'
BASE_URL = get_env_var('BASE_URL') or 'https://api.openai.com/v1'
API_KEY = get_env_var('LLM_API_KEY') or 'no-llm-api-key-provided'
SOURCE_NAME = "langgraphjs_docs"  # Specific source for this agent
# ---------------------

provider = get_env_var('LLM_PROVIDER') or 'OpenAI'

logfire.configure(token=os.getenv('LOGFIRE_API_KEY'))

if provider == "Anthropic":
    model = AnthropicModel(model_name=LLM_MODEL_NAME, api_key=API_KEY)
    logfire.instrument_anthropic()
elif provider == "OpenAI":
    logfire.instrument_openai()
    openai_provider = OpenAIProvider(api_key=API_KEY, base_url=BASE_URL)
    model = OpenAIModel(model_name=LLM_MODEL_NAME,
                        provider=openai_provider)
else:
    openai_provider = OpenAIProvider(api_key=API_KEY, base_url=BASE_URL)
    model = OpenAIModel(model_name=LLM_MODEL_NAME,
                        provider=openai_provider)
    logfire.instrument_aiohttp_client()


@dataclass
class LangGraphJsCoderDeps:
    supabase: Client
    embedding_client: AsyncOpenAI
    reasoner_output: str
    advisor_output: str


# Rename agent variable
langgraphjs_coder = Agent(
    model,
    system_prompt=primary_coder_prompt,
    deps_type=LangGraphJsCoderDeps,  # Use specific Deps class
    retries=2
)

# Update decorator and context type hint


@langgraphjs_coder.system_prompt
# Use specific Deps class
def add_reasoner_output(ctx: RunContext[LangGraphJsCoderDeps]) -> str:
    # Update prompt to mention LangGraph context if needed, keeping generic for now
    return f"""

    Additional thoughts/instructions from the reasoner LLM.
    This scope includes documentation pages specific to {SOURCE_NAME} for you to search:
    {ctx.deps.reasoner_output}

    Recommended starting point from the advisor agent:
    {ctx.deps.advisor_output}
    """

# Update decorator and context type hint


@langgraphjs_coder.tool
# Use specific Deps class
async def retrieve_relevant_documentation(ctx: RunContext[LangGraphJsCoderDeps], user_query: str) -> str:
    """
    Retrieve relevant LangGraph JS documentation chunks based on the query with RAG.

    Args:
        ctx: The context including the Supabase client and OpenAI client
        user_query: The user's question or query

    Returns:
        A formatted string containing the top 4 most relevant LangGraph JS documentation chunks
    """
    # Pass the source filter
    return await retrieve_relevant_documentation_tool(ctx.deps.supabase, ctx.deps.embedding_client, user_query, source_filter=SOURCE_NAME)

# Update decorator and context type hint


@langgraphjs_coder.tool
# Use specific Deps class
async def list_documentation_pages(ctx: RunContext[LangGraphJsCoderDeps]) -> List[str]:
    """
    Retrieve a list of all available LangGraph JS documentation pages ({SOURCE_NAME}).

    Returns:
        List[str]: List of unique URLs for LangGraph JS documentation pages
    """
    # Pass the source filter
    return await list_documentation_pages_tool(ctx.deps.supabase, source_filter=SOURCE_NAME)

# Update decorator and context type hint


@langgraphjs_coder.tool
# Use specific Deps class
async def get_page_content(ctx: RunContext[LangGraphJsCoderDeps], url: str) -> str:
    """
    Retrieve the full content of a specific LangGraph JS documentation page ({SOURCE_NAME}).

    Args:
        ctx: The context including the Supabase client
        url: The URL of the page to retrieve

    Returns:
        str: The complete page content with all chunks combined in order
    """
    # Pass the source filter
    return await get_page_content_tool(ctx.deps.supabase, url, source_filter=SOURCE_NAME)

# Override the new_messages_json method to return a string


def get_messages_json(result):
    messages = result.new_messages()
    return json.dumps(ModelMessagesTypeAdapter.dump_python(messages))


langgraphjs_coder.new_messages_json = get_messages_json
