from __future__ import annotations as _annotations
from archon.agent_tools import (
    retrieve_relevant_documentation_tool,
    list_documentation_pages_tool,
    get_page_content_tool,
    get_file_content_tool,
    find_relevant_libraries
)
from archon.agent_prompts import tools_refiner_prompt
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
from pydantic_ai.providers.openai import OpenAIProvider
from openai import AsyncOpenAI
from supabase import Client

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


@dataclass
class ToolsRefinerDeps:
    supabase: Client
    embedding_client: AsyncOpenAI
    file_list: List[str]
    selected_coder: str = "pydantic_ai_coder"  # Default value


tools_refiner_agent = Agent(
    model,
    system_prompt=tools_refiner_prompt,
    deps_type=ToolsRefinerDeps,
    retries=2
)


@tools_refiner_agent.system_prompt
def add_file_list(ctx: RunContext[str]) -> str:
    joined_files = "\n".join(ctx.deps.file_list)
    return f"""
    
    Here is the list of all the files that you can pull the contents of with the
    'get_file_content' tool if the example/tool/MCP server is relevant to the
    agent the user is trying to build:

    {joined_files}
    """


@tools_refiner_agent.tool
async def retrieve_relevant_documentation(ctx: RunContext[ToolsRefinerDeps], query: str) -> str:
    """
    Retrieve relevant documentation chunks based on the query with RAG.
    Make sure your searches always focus on implementing tools.

    Args:
        ctx: The context including the Supabase client and OpenAI client
        query: Your query to retrieve relevant documentation for implementing tools

    Returns:
        A formatted string containing the top 4 most relevant documentation chunks
    """
    # Map coder to source
    source_map = {
        "pydantic_ai_coder": "pydantic_ai_docs",
        "langchain_python_coder": "langchain_python_docs",
        "langchain_js_coder": "langchain_js_docs",
        "langgraph_coder": "langgraph_docs",
        "langgraphjs_coder": "langgraphjs_docs",
        "langsmith_coder": "langsmith_docs"
    }
    source = source_map.get(ctx.deps.selected_coder, "pydantic_ai_docs")
    return await retrieve_relevant_documentation_tool(ctx.deps.supabase, ctx.deps.embedding_client, query, source_filter=source)


@tools_refiner_agent.tool
async def list_documentation_pages(ctx: RunContext[ToolsRefinerDeps]) -> List[str]:
    """
    Retrieve a list of all available documentation pages for the selected coder.
    This will give you all pages available, but focus on the ones related to tools.

    Returns:
        List[str]: List of unique URLs for all documentation pages
    """
    source_map = {
        "pydantic_ai_coder": "pydantic_ai_docs",
        "langchain_python_coder": "langchain_python_docs",
        "langchain_js_coder": "langchain_js_docs",
        "langgraph_coder": "langgraph_docs",
        "langgraphjs_coder": "langgraphjs_docs",
        "langsmith_coder": "langsmith_docs"
    }
    source = source_map.get(ctx.deps.selected_coder, "pydantic_ai_docs")
    return await list_documentation_pages_tool(ctx.deps.supabase, source_filter=source)


@tools_refiner_agent.tool
async def get_page_content(ctx: RunContext[ToolsRefinerDeps], url: str) -> str:
    """
    Retrieve the full content of a specific documentation page by combining all its chunks.
    Only use this tool to get pages related to using tools with the selected coder.

    Args:
        ctx: The context including the Supabase client
        url: The URL of the page to retrieve

    Returns:
        str: The complete page content with all chunks combined in order
    """
    source_map = {
        "pydantic_ai_coder": "pydantic_ai_docs",
        "langchain_python_coder": "langchain_python_docs",
        "langchain_js_coder": "langchain_js_docs",
        "langgraph_coder": "langgraph_docs",
        "langgraphjs_coder": "langgraphjs_docs",
        "langsmith_coder": "langsmith_docs"
    }
    source = source_map.get(ctx.deps.selected_coder, "pydantic_ai_docs")
    return await get_page_content_tool(ctx.deps.supabase, url, source_filter=source)


@tools_refiner_agent.tool_plain
def get_file_content(file_path: str) -> str:
    """
    Retrieves the content of a specific file. Use this to get the contents of an example, tool, config for an MCP server

    Args:
        file_path: The path to the file

    Returns:
        The raw contents of the file
    """
    return get_file_content_tool(file_path)
