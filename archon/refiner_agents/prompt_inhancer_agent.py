from __future__ import annotations as _annotations
from archon.agent_tools import (
    find_relevant_libraries,
    enhance_query_for_embedding,
    get_query_embedding,
    query_pinecone_with_filter,
    extract_urls_from_pinecone_results
)
from archon.agent_prompts import agent_prompt_optimizer_prompt
from utils.utils import get_env_var

from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import os
import sys
from typing import List, Optional
from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIModel
from instructor import AsyncInstructor
from dotenv import load_dotenv
from pydantic_ai.providers.openai import OpenAIProvider
load_dotenv()
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
class AgentPromptOptimizerDeps:
    model_name: Optional[str]
    llm_client: Optional[AsyncInstructor]
    text_query: Optional[str]


agent_prompt_optimizer = Agent(
    model,
    system_prompt=agent_prompt_optimizer_prompt,
    deps_type=AgentPromptOptimizerDeps,
    retries=2
)


@agent_prompt_optimizer.tool
async def enhance_query_for_embedding_tool(ctx: RunContext[AgentPromptOptimizerDeps], query: str) -> str:
    """
    Enhances the user query for better semantic search results.
    """
    return await enhance_query_for_embedding(query=query, client=ctx.deps.llm_client, model_name=ctx.deps.model_name)
