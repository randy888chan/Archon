from __future__ import annotations as _annotations
from archon.agent_prompts import prompt_refiner_prompt
from utils.utils import get_env_var

import logfire
import os
import sys
from pydantic_ai import Agent
from dotenv import load_dotenv
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIModel
from supabase import Client
from pydantic_ai.providers.openai import OpenAIProvider
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

provider = get_env_var('LLM_PROVIDER') or 'OpenAI'
logfire.configure(token=os.getenv('LOGFIRE_API_KEY'))
if provider == "OpenAI":
    logfire.instrument_openai()
elif provider == "Anthropic":
    logfire.instrument_anthropic()
else:
    logfire.instrument_aiohttp_client()

prompt_refiner_agent = Agent(
    model,
    system_prompt=prompt_refiner_prompt
)
