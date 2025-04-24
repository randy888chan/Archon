from utils.utils import get_env_var
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter
)
from utils.utils import get_env_var, get_clients, write_to_log
from archon.agent_tools import find_relevant_libraries
from archon.refiner_agents.agent_refiner_agent import agent_refiner_agent, AgentRefinerDeps
from archon.refiner_agents.tools_refiner_agent import tools_refiner_agent, ToolsRefinerDeps
from archon.refiner_agents.rag_refiner_agent import run_rag_refiner, RagProcessingService
from archon.refiner_agents.prompt_refiner_agent import prompt_refiner_agent
from archon.refiner_agents.prompt_inhancer_agent import agent_prompt_optimizer, AgentPromptOptimizerDeps
from archon.advisor_agent import advisor_agent, AdvisorDeps
from archon.pydantic_ai_coder import pydantic_ai_coder, PydanticAIDeps
from archon.langgraph_coder import langgraph_coder, LangGraphCoderDeps
from archon.langgraphjs_coder import langgraphjs_coder, LangGraphJsCoderDeps
from archon.langsmith_coder import langsmith_coder, LangSmithCoderDeps
from archon.langchain_python_coder import langchain_python_coder, LangChainPyCoderDeps
from archon.langchain_js_coder import langchain_js_coder, LangChainJsCoderDeps
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIModel
from instructor.client import AsyncInstructor
from pydantic_ai import Agent, RunContext
from archon.agent_prompts import LIBRARY_PROMPT
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, List, Any, Optional
from langgraph.config import get_stream_writer
from langgraph.types import interrupt
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator
from openai import AsyncOpenAI
from langsmith import traceable
import openai
import instructor
from langsmith.wrappers import wrap_openai
import logfire
import os
import sys
from pinecone import Pinecone
from pydantic_ai.providers.openai import OpenAIProvider
from anthropic import AsyncAnthropic
import asyncio

logger = logfire.configure(token=os.getenv('LOGFIRE_API_KEY'))

# Import the message classes from Pydantic AI

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

# Configure logfire to suppress warns (optional)
provider = get_env_var('LLM_PROVIDER') or 'OpenAI'

if provider == "OpenAI":
    logfire.instrument_openai()
elif provider == "Anthropic":
    logfire.instrument_anthropic()
else:
    logfire.instrument_aiohttp_client()

pinecone_api_key = get_env_var(
    'PINECONE_API_KEY') or 'no-pinecone-api-key-provided'
pinecone_index_name = get_env_var(
    'PINECONE_INDEX_NAME') or 'no-pinecone-index-name-provided'
# Check if index name is provided before initializing
if pinecone_index_name == 'no-pinecone-index-name-provided':
    logger.warn(
        "Warning: PINECONE_INDEX_NAME not set. Pinecone integration will be skipped.")
    pinecone_index = "default-index"
else:
    try:
        pinecone_client = Pinecone(api_key=pinecone_api_key)
        # Maybe add environment? pinecone_client = Pinecone(api_key=pinecone_api_key, environment=pinecone_region) # Check Pinecone client v3+ syntax
        pinecone_index = pinecone_client.Index(
            pinecone_index_name,
            pool_threads=50,
            connection_pool_maxsize=50,
        )
        logger.info(
            f"Successfully connected to Pinecone index: {pinecone_index_name}")
        # Optional: Check index readiness/stats here if needed
        stats = pinecone_index.describe_index_stats()
        logger.debug(f"Index stats: {str(stats)}")
    except Exception as e:
        logger.error(
            f"Error connecting to Pinecone index '{pinecone_index_name}': {e}")
        pinecone_index = None

provider = get_env_var('LLM_PROVIDER') or 'OpenAI'
base_url = get_env_var('BASE_URL') or 'https://api.openai.com/v1'
api_key = get_env_var('LLM_API_KEY') or 'no-llm-api-key-provided'
embedding_model_name = get_env_var(
    'EMBEDDING_MODEL') or 'text-embedding-3-small'

is_anthropic = provider == "Anthropic"
is_openai = provider == "OpenAI"

reasoner_llm_model_name = get_env_var('REASONER_MODEL') or 'o3-mini'

if provider == "Anthropic":
    reasoner_llm_model = AnthropicModel(
        model_name=reasoner_llm_model_name, api_key=api_key)
    logfire.instrument_anthropic()
elif provider == "OpenAI":
    logfire.instrument_openai()
    openai_provider = OpenAIProvider(api_key=api_key, base_url=base_url, )
    reasoner_llm_model = OpenAIModel(model_name=reasoner_llm_model_name,
                                     provider=openai_provider)
else:
    openai_provider = OpenAIProvider(api_key=api_key, base_url=base_url)
    reasoner_llm_model = OpenAIModel(model_name=reasoner_llm_model_name,
                                     provider=openai_provider)
    logfire.instrument_aiohttp_client()

reasoner = Agent(
    reasoner_llm_model,
    system_prompt='You are an expert at coding AI agents with Pydantic AI and defining the scope for doing so.',
)


primary_llm_model_name = get_env_var('PRIMARY_MODEL') or 'gpt-4o-mini'

if provider == "Anthropic":
    primary_llm_model = AnthropicModel(
        model_name=primary_llm_model_name, api_key=api_key)
    logfire.instrument_anthropic()
elif provider == "OpenAI":
    logfire.instrument_openai()
    openai_provider = OpenAIProvider(api_key=api_key, base_url=base_url)
    primary_llm_model = OpenAIModel(model_name=primary_llm_model_name,
                                    provider=openai_provider)
else:
    openai_provider = OpenAIProvider(api_key=api_key, base_url=base_url)
    primary_llm_model = OpenAIModel(model_name=primary_llm_model_name,
                                    provider=openai_provider)
    logfire.instrument_aiohttp_client()


if provider == "Ollama":
    if api_key == "NOT_REQUIRED":
        api_key = None
    raw_client = openai.AsyncOpenAI(
        api_key=api_key, base_url=base_url, timeout=3000)
    wrapped_client = wrap_openai(raw_client)
    llm_client: AsyncInstructor = instructor.from_openai(client=wrapped_client)
else:
    raw_client = openai.AsyncOpenAI(
        api_key=api_key, base_url=base_url, timeout=3000)
    wrapped_client = wrap_openai(raw_client)
    llm_client: AsyncInstructor = instructor.from_openai(client=wrapped_client)

router_agent = Agent(
    primary_llm_model,
    system_prompt='Your job is to route the user message either to the end of the conversation or to continue coding the AI agent.',
)


# Initialize clients
embedding_client, supabase = get_clients()

# Define state schema


class AgentState(TypedDict):
    latest_user_message: str
    # Stores Pydantic AI message history
    messages: Annotated[List[bytes], lambda x, y: x + y]

    scope: str
    advisor_output: str
    file_list: List[str]  # List of available example files

    # Intermediate steps for scope definition
    enhanced_query: Optional[str] = None
    retrieve_urls: Optional[list[str]] = None

    # Selected coder route
    # Stores the name of the coder node chosen by the router
    selected_coder: Optional[dict]

    # Refinement outputs
    refined_prompt: str
    refined_tools: str
    refined_agent: str


class CoderRouterDeps(BaseModel):
    pydantic_ai_coder_deps: bool = Field(
        default=False, description="Whether to include Pydantic AI coder dependencies")
    langgraph_coder_deps: bool = Field(
        default=False, description="Whether to include LangGraph coder dependencies")
    langgraphjs_coder_deps: bool = Field(
        default=False, description="Whether to include LangGraph NextJS coder dependencies")
    langsmith_coder_deps: bool = Field(
        default=False, description="Whether to include LangSmith coder dependencies")
    langchain_python_coder_deps: bool = Field(
        default=False, description="Whether to include LangChain Python coder dependencies")
    langchain_js_coder_deps: bool = Field(
        default=False, description="Whether to include LangChain NextJS coder dependencies")


@traceable
async def coder_router(prompt: str):
    response = await llm_client.chat.completions.create(
        model=primary_llm_model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that determines which documentation source is most relevant to the user's request or the current scope. Return a True value for any of the following options provided."},
            {"role": "user", "content": prompt}
        ],
        response_model=CoderRouterDeps,
        temperature=0.0,
        max_retries=5,
    )
    logger.info(f"coder_router-response: {response}")
    result = response.model_dump()

    # Get all keys that have True values
    selected_coders = {key.replace('_deps', ''): value
                       for key, value in result.items()
                       if value}

    # Define valid coders
    valid_coders = [
        "pydantic_ai_coder", "langgraph_coder", "langgraphjs_coder",
        "langsmith_coder", "langchain_python_coder", "langchain_js_coder"
    ]

    # Validate selected coders
    validated_coders = {coder: True for coder in selected_coders.keys()
                        if coder in valid_coders}

    if not validated_coders:
        logger.notice(
            f"Warning: No valid coders found in response. Defaulting to langgraph_coder.")
        validated_coders = {"langgraph_coder": True}

    logger.info(f"Validated coders: {validated_coders}")
    return {"selected_coder": validated_coders}


end_conversation_agent = Agent(
    primary_llm_model,
    system_prompt='Your job is to end a conversation for creating an AI agent by giving instructions for how to execute the agent and they saying a nice goodbye to the user.',
)


@traceable
async def define_scope_with_reasoner(state: AgentState):
    # First, get the documentation pages so the reasoner can decide which ones are necessary
    # documentation_pages = await list_documentation_pages_tool(supabase)
    # documentation_pages_str = "\n".join(documentation_pages)
    logger.info("started : define_scope_with_reasoner")
    documentation_urls = state.get('retrieve_urls')
    # Then, use the reasoner to define the scope
    prompt = f"""
    User AI Agent Request: {state['latest_user_message']}

    Create detailed scope document for the AI agent including:
    - Architecture diagram
    - Core components
    - External dependencies
    - Testing strategy

    Also based on these documentation pages available:

    {documentation_urls}

    Include a list of documentation pages that are relevant to creating this agent for the user in the scope document.
    """

    result = await reasoner.run(prompt)
    scope = result.data
    logger.info("finished : define_scope_with_reasoner")
    logger.notice(f"scope: {scope}")

    # Get the directory one level up from the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    scope_path = os.path.join(parent_dir, "workbench", "scope.md")
    os.makedirs(os.path.join(parent_dir, "workbench"), exist_ok=True)

    with open(scope_path, "w", encoding="utf-8") as f:
        f.write(scope)

    return {"scope": scope}

# Advisor agent - create a starting point based on examples and prebuilt tools/MCP servers


@traceable
async def advisor_with_examples(state: AgentState):
    # Get the directory one level up from the current file (archon_graph.py)
    logger.info("started : advisor_with_examples")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)

    # The agent-resources folder is adjacent to the parent folder of archon_graph.py
    agent_resources_dir = os.path.join(parent_dir, "agent-resources")

    # Get a list of all files in the agent-resources directory and its subdirectories
    file_list = []

    for root, dirs, files in os.walk(agent_resources_dir):
        for file in files:
            # Get the full path to the file
            file_path = os.path.join(root, file)
            # Use the full path instead of relative path
            file_list.append(file_path)

    # Then, prompt the advisor with the list of files it can use for examples and tools
    deps = AdvisorDeps(file_list=file_list)
    result = await advisor_agent.run(state['latest_user_message'], deps=deps)
    advisor_output = result.data
    logger.info("finished : advisor_with_examples")
    logger.notice(f"advisor_output: {advisor_output}")

    return {"file_list": file_list, "advisor_output": advisor_output}


# Coding Node with Feedback Handling

@traceable
async def route_to_coder(state: AgentState) -> dict:
    """Determines which coder agent to use based on the scope."""
    logger.info("started : route_to_coder")
    prompt = f"""
    Analyze the following scope document and user request to determine the primary technology focus.

    User Request:
    {state['enhanced_query']}

    Scope Document:
    {state['scope']}

    Which coder agent should handle this? Respond ONLY with the agent name:
    - "pydantic_ai_coder" (Focus on Pydantic AI features, general agent structure)
    - "langgraph_coder" (Focus on LangGraph Python implementation)
    - "langgraphjs_coder" (Focus on LangGraph NextJS implementation)
    - "langsmith_coder" (Focus on LangSmith integration for observability and logging)
    - "langchain_python_coder" (Focus on core LangChain Python components, LCEL)
    - "langchain_js_coder" (Focus on core LangChain NextJS components)

    If the scope mentions multiple, choose the one that seems most central or default to "langgraph_coder".
    """
    result = await coder_router(prompt=prompt)
    selected_coders = result['selected_coder']
    logger.info("finished : route_to_coder")
    logger.notice(f"selected_coders: {selected_coders}")

    # Validate the result
    valid_coders = [
        "pydantic_ai_coder", "langgraph_coder", "langgraphjs_coder",
        "langsmith_coder", "langchain_python_coder", "langchain_js_coder"
    ]

    # If no valid coders or empty dict, default to langgraph_coder
    if not selected_coders:
        logger.notice(
            "Warning: No coders selected. Defaulting to langgraph_coder.")
        selected_coders = {"langgraph_coder": True}

    # Log the selected coders
    logger.info(f"selected_coders: {selected_coders}")
    write_to_log(f"Routing to coders: {list(selected_coders.keys())}")

    return {"selected_coder": selected_coders}


@traceable
async def coder_agent(state: AgentState, writer):
    """
    Dynamically selects, instantiates, and runs specialized coder agents in parallel
    based on the 'selected_coder' dictionary in the state.
    """
    logger.info("started : coder_agent")
    selected_coders = state.get('selected_coder', {})
    if not selected_coders:
        logger.error(
            "State is missing the required 'selected_coder' dictionary.")
        raise ValueError(
            "State must include a 'selected_coder' dictionary to determine agent types.")

    logger.info(f"Attempting to use coders: {list(selected_coders.keys())}")

    # Mapping from coder key to the agent constructor function and its dependencies class
    coder_type_mapping = {
        "pydantic_ai_coder": {"function": pydantic_ai_coder, "deps": PydanticAIDeps},
        "langgraph_coder": {"function": langgraph_coder, "deps": LangGraphCoderDeps},
        "langgraphjs_coder": {"function": langgraphjs_coder, "deps": LangGraphJsCoderDeps},
        "langsmith_coder": {"function": langsmith_coder, "deps": LangSmithCoderDeps},
        "langchain_python_coder": {"function": langchain_python_coder, "deps": LangChainPyCoderDeps},
        "langchain_js_coder": {"function": langchain_js_coder, "deps": LangChainJsCoderDeps}
    }

    # --- Prepare Message History ---
    message_history: list[ModelMessage] = []
    raw_messages = state.get('messages', [])
    if not isinstance(raw_messages, list):
        logger.warn(
            f"State 'messages' is not a list, received {type(raw_messages)}. Skipping history.")
        raw_messages = []

    for i, message_row in enumerate(raw_messages):
        if isinstance(message_row, str):
            try:
                parsed_messages = ModelMessagesTypeAdapter.validate_json(
                    message_row)
                message_history.extend(parsed_messages)
            except Exception as e:
                logger.error(
                    f"Failed to parse message row {i} from JSON string: {message_row}. Error: {e}")
        elif isinstance(message_row, dict) and "role" in message_row and "content" in message_row:
            try:
                message_history.append(ModelMessage(**message_row))
            except Exception as e:
                logger.error(
                    f"Failed to validate message row {i} (dict): {message_row}. Error: {e}")
        elif isinstance(message_row, ModelMessage):
            message_history.append(message_row)
        else:
            logger.warn(
                f"Unexpected message format in state['messages'] at index {i}: {type(message_row)}. Skipping.")

    # --- Determine Agent Prompt ---
    prompt = state.get('latest_user_message',
                       "Generate the agent code based on the conversation history.")
    if state.get('refined_prompt') or state.get('refined_tools') or state.get('refined_agent'):
        prompt = f"""
        Refine the agent based on:
        Prompt Refinement: {state.get('refined_prompt', '[No prompt refinement]')}
        Tools Refinement: {state.get('refined_tools', '[No tools refinement]')}
        Agent Refinement: {state.get('refined_agent', '[No agent refinement]')}
        """

    async def stream_with_prefix(chunk: str, coder_name: str):
        """Stream output with coder name prefix"""
        if chunk.strip():
            writer(f"[{coder_name}] {chunk}\n")

    async def run_single_agent(coder_name: str) -> tuple[str, Optional[str]]:
        """Run a single coder agent and return its messages JSON string."""
        try:
            coder_config = coder_type_mapping.get(coder_name)
            if not coder_config:
                logger.error(f"Unknown coder type specified: {coder_name}")
                return coder_name, None

            agent_instance = coder_config["function"]
            deps_class = coder_config["deps"]

            deps = deps_class(
                supabase=supabase,
                embedding_client=embedding_client,
                reasoner_output=state.get('scope', {}),
                advisor_output=state.get('advisor_output', {})
            )

            result = None
            try:
                if is_openai:
                    async with agent_instance.run_stream(prompt, deps=deps, message_history=message_history) as stream_result:
                        collected_chunks = []
                        async for chunk in stream_result.stream_text(delta=True):
                            if chunk:
                                collected_chunks.append(chunk)
                                await stream_with_prefix(chunk, coder_name)
                        result = stream_result
                        logger.info(
                            f"Completed streaming for {coder_name}. Full response: {''.join(collected_chunks)[:200]}...")
                else:
                    result = await agent_instance.run(prompt, deps=deps, message_history=message_history)
                    if hasattr(result, 'data'):
                        await stream_with_prefix(result.data, coder_name)
                        logger.info(
                            f"Non-streaming response for {coder_name}: {str(result.data)[:200]}...")

                if result:
                    logger.info(
                        f"Result object for {coder_name}: {str(result)[:200]}...")
                    logger.info(
                        f"Result attributes for {coder_name}: {dir(result)}")

                    if hasattr(result, 'new_messages_json'):
                        try:
                            messages_json = result.new_messages_json()
                            logger.info(
                                f"Raw messages JSON for {coder_name}: {str(messages_json)[:200]}...")

                            if isinstance(messages_json, bytes):
                                try:
                                    messages_json = messages_json.decode(
                                        'utf-8')
                                    logger.info(
                                        f"Decoded bytes to string for {coder_name}")
                                except UnicodeDecodeError as e:
                                    logger.error(
                                        f"Failed to decode bytes for {coder_name}: {e}")
                                    return coder_name, None

                            if isinstance(messages_json, str):
                                if messages_json.strip().startswith('['):
                                    logger.info(
                                        f"Valid JSON array found for {coder_name}")
                                    return coder_name, messages_json
                            else:
                                logger.error(
                                    f"Unexpected type for messages_json after processing: {type(messages_json)}")
                        except Exception as e:
                            logger.exception(
                                f"Error processing new_messages_json for {coder_name}: {e}")
                    else:
                        logger.error(
                            f"Result for {coder_name} missing new_messages_json method. Available methods: {dir(result)}")
                else:
                    logger.error(f"No result object returned for {coder_name}")

                logger.error(
                    f"Agent '{coder_name}' failed to produce valid messages")
                return coder_name, None

            except Exception as e:
                logger.exception(
                    f"Error during agent execution for {coder_name}: {e}")
                return coder_name, None

        except Exception as e:
            logger.exception(
                f"Error in run_single_agent for {coder_name}: {e}")
            return coder_name, None

    # Run all selected coders in parallel
    tasks = [run_single_agent(coder_name)
             for coder_name in selected_coders.keys()]
    results = await asyncio.gather(*tasks)

    # Process results
    messages_to_add = []
    failed_coders = []
    successful_coders = []

    for coder_name, messages in results:
        if messages:
            messages_to_add.append(messages)
            successful_coders.append(coder_name)
            logger.info(f"Coder '{coder_name}' successfully produced results")
        else:
            failed_coders.append(coder_name)
            logger.error(f"Coder '{coder_name}' failed to produce results")

    # Log final status
    if successful_coders:
        logger.info(
            f"Successfully completed coders: {', '.join(successful_coders)}")
    if failed_coders:
        logger.warning(f"Failed coders: {', '.join(failed_coders)}")

    if not messages_to_add:
        error_msg = f"All coder agents failed to produce results. Failed coders: {', '.join(failed_coders)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    update_dict = {
        "messages": messages_to_add,
        "refined_prompt": "",
        "refined_tools": "",
        "refined_agent": ""
    }

    logger.info(
        f"Coder agent node finished. Returning state updates including {len(messages_to_add)} new message list(s).")
    write_to_log(f"updated state: {update_dict}")
    return update_dict

# Interrupt the graph to get the user's next message


@traceable
def get_next_user_message(state: AgentState):
    value = interrupt({})

    # Set the user's latest message for the LLM to continue the conversation
    return {
        "latest_user_message": value
    }

# Determine if the user is finished creating their AI agent or not


@traceable
async def route_user_message(state: AgentState):
    logger.info("started : route_user_message")
    prompt = f"""
    The user has sent a message: 
    
    {state['latest_user_message']}

    If the user wants to end the conversation, respond with just the text "finish_conversation".
    If the user wants to continue coding the AI agent and gave feedback, respond with just the text "route_to_coder".
    If the user asks specifically to "refine" the agent, respond with just the text "refine".
    """

    result = await router_agent.run(prompt)

    if result.data == "finish_conversation":
        write_to_log(f"route_user_message: {result.data}")
        logger.info("finished : route_user_message")
        return "finish_conversation"
    if result.data == "refine":
        refine_prompt = ["refine_prompt", "refine_tools", "refine_agent"]
        write_to_log(f"route_user_message: {refine_prompt}")
        logger.info("finished : route_user_message")
        return refine_prompt
    write_to_log("route_user_message: route_to_coder")
    return "route_to_coder"

# Refines the prompt for the AI agent


@traceable
async def enhance_initial_prompt(state: AgentState):
    logger.info("started : enhance_initial_prompt")
    _raw_client = wrap_openai(AsyncOpenAI(api_key=api_key))
    deps = AgentPromptOptimizerDeps(
        model_name=primary_llm_model_name,
        llm_client=_raw_client,
        text_query=state['latest_user_message']
    )

    message_history: list[ModelMessage] = []
    for message_row in state['messages']:
        message_history.extend(
            ModelMessagesTypeAdapter.validate_json(message_row))

    prompt = "Based on the current conversation, refine the prompt for the agent."

    # Run the agent to refine the prompt for the agent being created
    result = await agent_prompt_optimizer.run(prompt, deps=deps, message_history=message_history)
    write_to_log(f"enhanced_query: {result.data}")
    logger.info("finished : enhance_initial_prompt")
    return {"enhanced_query": result.data}


@traceable
async def refine_prompt(state: AgentState):
    # Get the message history into the format for Pydantic AI
    logger.info("started : refine_prompt")
    message_history: list[ModelMessage] = []
    for message_row in state['messages']:
        message_history.extend(
            ModelMessagesTypeAdapter.validate_json(message_row))

    prompt = "Based on the current conversation, refine the prompt for the agent."

    # Run the agent to refine the prompt for the agent being created
    result = await prompt_refiner_agent.run(prompt, message_history=message_history)
    write_to_log(f"refined_prompt: {result.data}")
    logger.info("finished : refine_prompt")
    return {"refined_prompt": result.data}

# Retrieves the URLs from the Pinecone index


@traceable
async def url_documentation_agent(state: AgentState):
    """
    Retrieves documentation URLs from Pinecone index based on the enhanced query.
    Includes fallback mechanisms and detailed logging.
    (Timeout logic temporarily removed for debugging)
    """
    import asyncio
    import time

    # Start timer for performance tracking
    start_time = time.time()
    logger.info(f"STARTED: url_documentation_agent at {start_time}")
    write_to_log("STARTED: url_documentation_agent")

    # Initialize clients
    _raw_client = wrap_openai(AsyncOpenAI(api_key=api_key))
    _llm_client: AsyncInstructor = instructor.from_openai(client=_raw_client)

    # Try to initialize Pinecone with better error handling
    try:
        _pinecone_client = Pinecone(api_key=get_env_var('PINECONE_API_KEY'))
        _pinecone_index_name = get_env_var('PINECONE_INDEX_NAME')

        if not _pinecone_index_name or _pinecone_index_name == 'no-pinecone-index-name-provided':
            logger.warn(
                "PINECONE_INDEX_NAME not configured properly. Using fallback URLs.")
            # Fallback URLs for when Pinecone is not available
            fallback_urls = [
                "https://langchain-ai.github.io/langgraph/concepts/",
                "https://langchain-ai.github.io/langgraph/how-tos/human-in-the-loop/"
            ]
            elapsed = time.time() - start_time
            logger.info(
                f"COMPLETED: url_documentation_agent with fallback URLs in {elapsed:.2f}s")
            write_to_log(
                f"COMPLETED: url_documentation_agent with fallback in {elapsed:.2f}s")
            return {"retrieve_urls": fallback_urls}

        # Initialize Pinecone index
        _pinecone_index = _pinecone_client.Index(_pinecone_index_name)

        # Log success and index stats for diagnostics
        stats = _pinecone_index.describe_index_stats()
        logger.info(
            f"Successfully connected to Pinecone index '{_pinecone_index_name}'. Stats: {stats}")
    except Exception as e:
        logger.error(
            f"ERROR initializing Pinecone in url_documentation_agent: {e}")
        # Return fallback URLs on Pinecone error
        fallback_urls = [
            "https://langchain-ai.github.io/langgraph/concepts/",
            "https://langchain-ai.github.io/langgraph/how-tos/human-in-the-loop/"
        ]
        elapsed = time.time() - start_time
        logger.info(
            f"COMPLETED: url_documentation_agent with fallback URLs in {elapsed:.2f}s")
        write_to_log(
            f"COMPLETED: url_documentation_agent with fallback in {elapsed:.2f}s")
        return {"retrieve_urls": fallback_urls}

    # Initialize RAG service
    deps = RagProcessingService(
        llm_client=_llm_client,
        embedding_client=_raw_client,
        pinecone_client=_pinecone_client,
        pinecone_index=_pinecone_index
    )

    # Get query from state
    initial_query = state.get('enhanced_query')
    if not initial_query:
        logger.error(
            "'enhanced_query' not found in state for url_documentation_agent.")
        fallback_urls = ["https://langchain-ai.github.io/langgraph/concepts/"]
        elapsed = time.time() - start_time
        logger.info(
            f"COMPLETED: url_documentation_agent with fallback URLs in {elapsed:.2f}s")
        return {"retrieve_urls": fallback_urls}

    # Log progress
    logger.info(f"Processing RAG query: '{initial_query[:100]}...'")
    write_to_log(f"Processing RAG query: '{initial_query[:50]}...'")

    # --- Execute RAG pipeline directly (Timeout removed) ---
    try:
        logger.info("Executing run_rag_refiner directly without timeout...")
        results_with_summaries = await run_rag_refiner(initial_query, deps)

        # Log success and data
        elapsed = time.time() - start_time
        result_count = len(
            results_with_summaries) if results_with_summaries else 0
        logger.info(
            f"COMPLETED: url_documentation_agent returned {result_count} results in {elapsed:.2f}s")
        write_to_log(
            f"COMPLETED: url_documentation_agent with {result_count} results in {elapsed:.2f}s")

        if not results_with_summaries or len(results_with_summaries) == 0:
            logger.warn("RAG returned empty results, using fallback URLs")
            fallback_urls = [
                "https://langchain-ai.github.io/langgraph/concepts/",
                "https://langchain-ai.github.io/langgraph/how-tos/human-in-the-loop/"
            ]
            return {"retrieve_urls": fallback_urls}

        return {"retrieve_urls": results_with_summaries}

    except Exception as e:
        # Handle any other exceptions during the RAG process
        elapsed = time.time() - start_time
        # Use logger.exception to include traceback
        logger.exception(
            f"Error during run_rag_refiner execution after {elapsed:.2f}s: {e}")
        write_to_log(f"Error in RAG operation: {str(e)}")

        # Return fallback URLs on error
        fallback_urls = [
            "https://langchain-ai.github.io/langgraph/concepts/",
            "https://langchain-ai.github.io/langgraph/how-tos/human-in-the-loop/"
        ]
        return {"retrieve_urls": fallback_urls}


# Refines the tools for the AI agent

@traceable
async def refine_tools(state: AgentState):
    logger.info("started : refine_tools")

    # First determine which coder to use based on the conversation
    relevant_libs = await find_relevant_libraries(
        state['latest_user_message'],
        embedding_client,
        llm_client
    )

    # Map library names to coder names
    lib_to_coder = {
        'pydantic_ai_docs': 'pydantic_ai_coder',
        'langchain_python_docs': 'langchain_python_coder',
        'langchain_js_docs': 'langchain_js_coder',
        'langgraph_docs': 'langgraph_coder',
        'langgraphjs_docs': 'langgraphjs_coder',
        'langsmith_docs': 'langsmith_coder'
    }

    # Select the first relevant library's coder, default to pydantic_ai_coder if none found
    selected_coder = lib_to_coder.get(
        relevant_libs[0] if relevant_libs else 'pydantic_ai_docs')

    # Prepare dependencies
    deps = ToolsRefinerDeps(
        supabase=supabase,
        embedding_client=embedding_client,
        file_list=state['file_list'],
        selected_coder=selected_coder
    )

    # Get the message history into the format for Pydantic AI
    message_history: list[ModelMessage] = []
    for message_row in state['messages']:
        message_history.extend(
            ModelMessagesTypeAdapter.validate_json(message_row))

    prompt = "Based on the current conversation, refine the tools for the agent: pydantic_ai_coder, langchain_python_coder, langchain_js_coder, langgraph_coder, langgraphjs_coder, and langsmith_coder."

    # Run the agent to refine the tools for the agent being created
    result = await tools_refiner_agent.run(prompt, deps=deps, message_history=message_history)
    write_to_log(f"refined_tools: {result.data}")
    logger.info("finished : refine_tools")
    return {"refined_tools": result.data, "selected_coder": selected_coder}

# Refines the defintion for the AI agent


@traceable
async def refine_agent(state: AgentState):
    logger.info("started : refine_agent")
    # Prepare dependencies
    deps = AgentRefinerDeps(
        supabase=supabase,
        embedding_client=embedding_client,
        pinecone_client=pinecone_client,
        pinecone_index=pinecone_index,
        text_query=state['enhanced_query']
    )

    # Get the message history into the format for Pydantic AI
    message_history: list[ModelMessage] = []
    for message_row in state['messages']:
        message_history.extend(
            ModelMessagesTypeAdapter.validate_json(message_row))

    prompt = "Based on the current conversation, refine the agent definition."

    # Run the agent to refine the definition for the agent being created
    result = await agent_refiner_agent.run(prompt, deps=deps, message_history=message_history)
    write_to_log(f"refined_agent: {result.data}")
    logger.info("finished : refine_agent")
    return {"refined_agent": result.data}

# End of conversation agent to give instructions for executing the agent


@traceable
async def finish_conversation(state: AgentState, writer):
    logger.info("started : finish_conversation")
    # Get the message history into the format for Pydantic AI
    message_history: list[ModelMessage] = []
    for message_row in state['messages']:
        message_history.extend(
            ModelMessagesTypeAdapter.validate_json(message_row))

    # Run the agent in a stream
    if not is_openai:
        writer = get_stream_writer()
        result = await end_conversation_agent.run(state['latest_user_message'], message_history=message_history)
        writer(result)
    else:
        async with end_conversation_agent.run_stream(
            state['latest_user_message'],
            message_history=message_history
        ) as result:
            # Stream partial text as it arrives
            async for chunk in result.stream_text(delta=True):
                writer(chunk)
    write_to_log(f"messages: {result}")
    logger.info("finished : finish_conversation")
    return {"messages": [result.new_messages_json()]}

# Build workflow
builder = StateGraph(AgentState)

# Add nodes

builder.add_node("enhance_initial_prompt", enhance_initial_prompt)
builder.add_node("url_documentation_agent", url_documentation_agent)
builder.add_node("define_scope_with_reasoner", define_scope_with_reasoner)
builder.add_node("advisor_with_examples", advisor_with_examples)
builder.add_node("route_to_coder", route_to_coder)
builder.add_node("coder_agent", coder_agent)
builder.add_node("get_next_user_message", get_next_user_message)
builder.add_node("refine_prompt", refine_prompt)
builder.add_node("refine_tools", refine_tools)
builder.add_node("refine_agent", refine_agent)
builder.add_node("finish_conversation", finish_conversation)

# Set edges
# Ensure sequential execution where dependencies exist

# step 1: Enhance prompt first
builder.add_edge(START, "enhance_initial_prompt")

# step 2: Get documentation based on enhanced prompt
builder.add_edge("enhance_initial_prompt", "url_documentation_agent")

# step 3: Define scope using documentation URLs
builder.add_edge("url_documentation_agent", "define_scope_with_reasoner")

# step 4: Run advisor after scope is defined
builder.add_edge("define_scope_with_reasoner", "advisor_with_examples")

# step 5: Route to coder only after advisor and scope are ready
builder.add_edge("advisor_with_examples", "route_to_coder")

# step 6: Coder agent execution
builder.add_edge("route_to_coder", "coder_agent")

# step 7: Handle user feedback loop
builder.add_edge("coder_agent", "get_next_user_message")

# step 8: Conditional routing based on user message
builder.add_conditional_edges(
    "get_next_user_message",
    route_user_message,
    {
        "route_to_coder": "route_to_coder",  # If user gives feedback
        "refine_prompt": "refine_prompt",  # If user asks to refine prompt
        "refine_tools": "refine_tools",     # If user asks to refine tools
        "refine_agent": "refine_agent",     # If user asks to refine agent
        "finish_conversation": "finish_conversation"  # If user wants to end
    }
)

# step 9: Refinement loops back to router
builder.add_edge("refine_prompt", "route_to_coder")
builder.add_edge("refine_tools", "route_to_coder")
builder.add_edge("refine_agent", "route_to_coder")

# step 10: Finish conversation
builder.add_edge("finish_conversation", END)

# Configure persistence
memory = MemorySaver()
# Compile without checkpointer parameter - checkpointer should only be passed during runtime and removed for debugging in langgraph studio
agentic_flow = builder.compile(
    checkpointer=memory,
    debug=True)
