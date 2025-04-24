from __future__ import annotations as _annotations
import asyncio
import instructor
import openai
from archon.agent_tools import (
    find_relevant_libraries,
    enhance_query_for_embedding,
    get_query_embedding,
    query_pinecone_with_filter,
    extract_urls_from_pinecone_results
)
from utils.utils import get_env_var

from dataclasses import dataclass, field
from dotenv import load_dotenv
import logfire
import os
import sys
from typing import List, Optional
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
)
from pydantic_graph.nodes import End
from openai import AsyncOpenAI
from supabase import Client
from pinecone import Pinecone
from instructor import AsyncInstructor
from pydantic_ai.providers.openai import OpenAIProvider
from httpx import AsyncClient
from pydantic_ai.settings import ModelSettings

# Add the parent directory to sys.path to allow importing from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

load_dotenv()

provider = get_env_var('LLM_PROVIDER') or 'OpenAI'
llm_model_name = get_env_var('PRIMARY_MODEL') or 'gpt-4o-mini'
base_url = get_env_var('BASE_URL') or 'https://api.openai.com/v1'
api_key = get_env_var('LLM_API_KEY') or 'no-llm-api-key-provided'
os.environ['OPENAI_API_KEY'] = api_key


embedding_model_name = get_env_var(
    'EMBEDDING_MODEL') or 'text-embedding-3-small'

provider = get_env_var('LLM_PROVIDER') or 'OpenAI'
logger = logfire.configure(token=os.getenv('LOGFIRE_API_KEY'))
custom_http_client = AsyncClient(timeout=3000)

model_settings = ModelSettings(
    http_client=custom_http_client,
    timeout=3000,
    temperature=0.0
)

if provider == "Anthropic":
    agent_llm = AnthropicModel(model_name=llm_model_name, api_key=api_key)
    logfire.instrument_anthropic()
elif provider == "OpenAI":
    logfire.instrument_openai()
    openai_provider = OpenAIProvider(
        api_key=api_key, base_url=base_url, http_client=custom_http_client)
    agent_llm = OpenAIModel(model_name=llm_model_name,
                            provider=openai_provider)
else:
    openai_provider = OpenAIProvider(
        api_key=api_key, base_url=base_url, http_client=custom_http_client)
    agent_llm = OpenAIModel(model_name=llm_model_name,
                            provider=openai_provider)
    logfire.instrument_aiohttp_client()


@dataclass
class RagProcessingService:
    llm_client: AsyncInstructor
    embedding_client: AsyncOpenAI
    pinecone_client: Pinecone
    pinecone_index: Pinecone.Index
    enhanced_query: Optional[str] = None
    llm_model_name: str = field(default=llm_model_name)
    embedding_model_name: str = field(default=embedding_model_name)

    async def enhance_query(self, initial_query: str) -> str:
        print(
            f"\n[Service] enhance_query: Received initial_query='{initial_query}...'")
        enhanced = await enhance_query_for_embedding(query=initial_query, client=self.embedding_client, model_name=self.embedding_model_name)
        print(
            f"[Service] enhance_query: Returning enhanced='{enhanced}...'")
        return enhanced

    async def find_libraries(self, enhanced_query: str) -> List[str]:
        print(
            f"\n[Service] find_libraries: Received enhanced_query='{enhanced_query}...'")
        libs = await find_relevant_libraries(enhanced_query, self.llm_client, self.llm_model_name)
        print(f"[Service] find_libraries: Returning libs={libs}")
        return libs

    async def get_embedding(self, enhanced_query: str) -> List[float]:
        print(
            f"\n[Service] get_embedding: Received enhanced_query='{enhanced_query}...'")
        vector = await get_query_embedding(enhanced_query, self.embedding_client, self.embedding_model_name)
        print(
            f"[Service] get_embedding: Returning vector (length {len(vector) if vector else 'None'})")
        return vector

    async def query_pinecone(self, enhanced_query: str, libraries: List[str], vector: List[float], top_k: int = 100, top_n: int = 10) -> list[dict]:
        print(
            f"\n[Service] query_pinecone: Received enhanced_query='{enhanced_query}...', libraries={libraries}")
        results = await query_pinecone_with_filter(
            pinecone_client=self.pinecone_client,
            index=self.pinecone_index,
            enhanced_query=enhanced_query,
            query_vector=vector,
            relevant_libraries=libraries,
            top_k=top_k,
            top_n=top_n
        )
        print(f"[Service] query_pinecone: Returning {len(results)} results")
        return results

    def extract_urls(self, pinecone_results: list[dict]) -> List[dict]:
        print(
            f"\n[Service] extract_urls: Received {len(pinecone_results)} results")
        urls = extract_urls_from_pinecone_results(pinecone_results)
        print(f"[Service] extract_urls: Returning {len(urls)} URLs")
        return urls


agent_rag_refiner = Agent(
    agent_llm,
    deps_type=RagProcessingService,
    result_type=List[str],
    retries=1,
    model_settings=model_settings
)


@agent_rag_refiner.system_prompt
def simple_rag_prompt(ctx: RunContext[RagProcessingService]) -> str:
    return ("You are an assistant that helps find documentation. "
            "Use the `run_rag_pipeline` tool to process the user's query and find relevant documentation URLs.")


@agent_rag_refiner.tool
async def run_rag_pipeline(ctx: RunContext[RagProcessingService], initial_query: str) -> List[dict]:
    """
    Processes the user's query through a 5-step RAG pipeline to find relevant documentation URLs.

    Args:
        initial_query: The original query from the user.

    Returns:
        List[dict]: A list of dictionaries, each containing a 'url' and 'summary'.
    """
    print(f"\n--- Orchestrator Tool: run_rag_pipeline --- ")
    print(f"Received initial_query: '{initial_query}...'")
    service = ctx.deps

    try:
        print("\nOrchestrator: Executing Step 1: Enhance Query")
        enhanced_query = await service.enhance_query(initial_query)
        if not enhanced_query:
            print("Orchestrator: Failed to enhance query. Aborting.")
            return []

        print("\nOrchestrator: Executing Step 2: Find Libraries")
        libraries = await service.find_libraries(enhanced_query)
        print(f"Orchestrator: Found libraries: {libraries}")

        print("\nOrchestrator: Executing Step 3: Get Embedding")
        query_vector = await service.get_embedding(enhanced_query)
        if not query_vector:
            print("Orchestrator: Failed to get query embedding. Aborting.")
            return []

        print("\nOrchestrator: Executing Step 4: Query Pinecone")
        pinecone_results = await service.query_pinecone(enhanced_query, libraries, query_vector)
        if pinecone_results is None:
            print("Orchestrator: Pinecone query returned None. Aborting.")
            return []

        print("\nOrchestrator: Executing Step 5: Extract URLs")
        final_urls = service.extract_urls(pinecone_results)

        print(
            f"\n--- Orchestrator Tool: Finished. Returning {len(final_urls)} URLs ---")
        return final_urls

    except Exception as e:
        print(f"--- Orchestrator Tool: Error during execution: {e} ---")
        import traceback
        traceback.print_exc()
        return []


async def run_rag_refiner(initial_query: str, service_deps: RagProcessingService) -> List[str]:
    """
    Runs the RAG Refiner agent, which now uses a single orchestrator tool.

    Args:
        initial_query: The user's original query text.
        service_deps: An instance of RagProcessingService containing necessary clients.

    Returns:
        List[dict]: List of relevant URLs and their summaries.
    """
    print(
        f"\n--- Starting RAG Refiner Agent (Orchestrator Pattern) for initial_query: '{initial_query}...' ---")
    final_result: List[dict] = []

    try:
        async with agent_rag_refiner.iter(initial_query, deps=service_deps) as agent_run:
            async for node in agent_run:
                if Agent.is_user_prompt_node(node):
                    print(
                        f"--- Agent Node: UserPrompt (Initial Query='{node.user_prompt}...') ---")
                elif Agent.is_model_request_node(node):
                    print(
                        f"--- Agent Node: ModelRequest --- (Expecting call to 'run_rag_pipeline')")
                elif Agent.is_call_tools_node(node):
                    print(f"--- Agent Node: CallTools --- ")
                    async with node.stream(agent_run.ctx) as handle_stream:
                        async for event in handle_stream:
                            if isinstance(event, FunctionToolCallEvent):
                                args_repr = repr(event.part.args)
                                if len(args_repr) > 150:
                                    args_repr = args_repr[:150] + "...}"
                                print(
                                    f"  [Agent Event] Tool Call: {event.part.tool_name} Args: {args_repr}")
                            elif isinstance(event, FunctionToolResultEvent):
                                result_content = str(event.result.content)
                                print(
                                    f"  [Agent Event] Tool Result (from {event.tool_call_id}): {len(result_content)} chars returned.")
                elif Agent.is_end_node(node):
                    print(f"--- Agent Node: End ---")
                    if agent_run.result:
                        print(f"  Agent run finished. Accessing final result...")
                        if isinstance(agent_run.result.data, list):
                            if all(isinstance(item, list) for item in agent_run.result.data):
                                final_result = agent_run.result.data
                                print(
                                    f"  Successfully obtained final result via agent ({len(final_result)} items).")
                            else:
                                print(
                                    f"  Notice: Final result is list, with strings: {agent_run.result.data}")
                        else:
                            print(
                                f"  Warning: Agent result data is not a list: Type={type(agent_run.result.data)}, Value={agent_run.result.data}")
                    else:
                        print("  Warning: Agent finished but run.result is None.")
                    break
                else:
                    print(
                        f"--- Agent Node: Encountered node type: {type(node).__name__} ---")

    except Exception as e:
        print(f"--- Error during agent execution: {e} ---")
        import traceback
        traceback.print_exc()
        final_result = []

    if not final_result and agent_run and agent_run.result and isinstance(agent_run.result.data, list):
        print("Attempting to retrieve result directly from agent_run.result.data again...")
        final_result = agent_run.result.data

    print(
        f"--- RAG Refiner Agent finished. Returning {len(final_result)} results. ---")
    return final_result


# api_key = get_env_var('LLM_API_KEY') or 'no-llm-api-key-provided'
# os.environ['OPENAI_API_KEY'] = api_key
# raw_client = openai.AsyncOpenAI(api_key=api_key)
# llm_client: AsyncInstructor = instructor.from_openai(client=raw_client)

# pinecone_client = Pinecone(
#     api_key="pcsk_5F7ZmQ_JYcxmbioMJxD27bYF4iuhVL6QjR5P1e7XxdYy7hXMWwiGFuzJuBGLLHrSKz9Qbf")
# pinecone_index = pinecone_client.Index("ai-docs-index")

# # This is the main function that will be used to run the agent from LangGraph but this is for testing purposes to run the agent directly in isolation


# async def url_documentation_agent(query):
#     print("started : url_documentation_agent")
#     enhance_query_for_embedding_result = await enhance_query_for_embedding(query, raw_client, "gpt-4o-mini")

#     deps = RagProcessingService(
#         llm_client=llm_client,
#         embedding_client=raw_client,
#         pinecone_client=pinecone_client,
#         pinecone_index=pinecone_index
#     )
#     # Run the agent using run_rag_refiner to ensure proper tool execution order
#     print("started : url_documentation_agent: running agent")
#     # result = await agent_rag_refiner.run(query, deps=deps)
#     filtered_urls = await run_rag_refiner(enhance_query_for_embedding_result, deps)
#     print("finished : url_documentation_agent: running agent")
#     return {"retrieve_urls": filtered_urls}

# if __name__ == "__main__":
#     initial_query = input("Enter a query: ")
#     # Run the async function using asyncio
#     result = asyncio.run(url_documentation_agent(initial_query))
#     print(result)
