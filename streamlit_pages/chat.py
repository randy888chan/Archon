from langgraph.types import Command
import streamlit as st
import asyncio
import uuid
import sys
import os
import time
import logging
import traceback
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [CHAT] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("archon.chat")

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from archon.archon_graph import agentic_flow
from utils.utils import write_to_log

# Helper function to log and write to the log file
def log_event(message, level="INFO"):
    """Log a message both to the console and the log file"""
    log_method = getattr(logger, level.lower())
    log_method(message)
    write_to_log(f"[CHAT] {message}")

@st.cache_resource
def get_thread_id():
    thread_id = str(uuid.uuid4())
    log_event(f"Created new thread ID: {thread_id}")
    return thread_id

thread_id = get_thread_id()

async def run_agent_with_streaming(user_input: str):
    """
    Run the agent with streaming text for the user_input prompt,
    while maintaining the entire conversation in `st.session_state.messages`.
    """
    log_event(f"Starting streaming for user input (length: {len(user_input)})")
    
    start_time = time.time()
    chunk_count = 0
    total_tokens = 0
    
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    # First message from user
    if len(st.session_state.messages) == 1:
        log_event("First message in conversation, starting new thread")
        try:
            # Set a timeout for the streaming to prevent hanging
            log_event("Setting up timeout for streaming (60 seconds)")
            async with asyncio.timeout(60):  # 60 second timeout
                log_event("Starting astream for first message")
                last_chunk_time = time.time()
                
                async for msg in agentic_flow.astream(
                        {"latest_user_message": user_input}, config, stream_mode="custom"
                    ):
                        # Log chunk timing and size
                        now = time.time()
                        chunk_delay = now - last_chunk_time
                        last_chunk_time = now
                        chunk_count += 1
                        total_tokens += len(msg) if msg else 0
                        
                        if chunk_count % 10 == 0 or chunk_delay > 1.0:
                            log_event(f"Received chunk #{chunk_count}, size: {len(msg)}, delay: {chunk_delay:.2f}s")
                        
                        # Add timeout check between chunks
                        yield msg
                
                elapsed = time.time() - start_time
                log_event(f"Streaming completed - {chunk_count} chunks, {total_tokens} chars in {elapsed:.2f}s")
                
        except asyncio.TimeoutError:
            error_msg = "\n\n[Timeout occurred: The agent took too long to respond. Please try again.]"
            log_event(f"TIMEOUT ERROR after {time.time() - start_time:.2f}s of streaming", level="ERROR")
            yield error_msg
        except Exception as e:
            error_msg = f"\n\n[An error occurred: {str(e)}]"
            log_event(f"STREAMING ERROR: {str(e)}\n{traceback.format_exc()}", level="ERROR")
            yield error_msg
    # Continue the conversation
    else:
        log_event(f"Continuing conversation with {len(st.session_state.messages)} existing messages")
        try:
            # Set a timeout for the streaming to prevent hanging
            log_event("Setting up timeout for streaming (60 seconds)")
            async with asyncio.timeout(60):  # 60 second timeout
                log_event("Starting astream for continuation")
                last_chunk_time = time.time()
                
                async for msg in agentic_flow.astream(
                    Command(resume=user_input), config, stream_mode="custom"
                ):
                    # Log chunk timing and size
                    now = time.time()
                    chunk_delay = now - last_chunk_time
                    last_chunk_time = now
                    chunk_count += 1
                    total_tokens += len(msg) if msg else 0
                    
                    if chunk_count % 10 == 0 or chunk_delay > 1.0:
                        log_event(f"Received chunk #{chunk_count}, size: {len(msg)}, delay: {chunk_delay:.2f}s")
                    
                    # Add timeout check between chunks
                    yield msg
                
                elapsed = time.time() - start_time
                log_event(f"Streaming completed - {chunk_count} chunks, {total_tokens} chars in {elapsed:.2f}s")
                
        except asyncio.TimeoutError:
            error_msg = "\n\n[Timeout occurred: The agent took too long to respond. Please try again.]"
            log_event(f"TIMEOUT ERROR after {time.time() - start_time:.2f}s of streaming", level="ERROR")
            yield error_msg
        except Exception as e:
            error_msg = f"\n\n[An error occurred: {str(e)}]"
            log_event(f"STREAMING ERROR: {str(e)}\n{traceback.format_exc()}", level="ERROR")
            yield error_msg

async def chat_tab():
    """Display the chat interface for talking to Archon"""
    log_event("Initializing chat tab")
    
    st.write("Describe to me an AI agent you want to build and I'll code it for you with Pydantic AI.")
    st.write("Example: Build me an AI agent that can search the web with the Brave API.")

    # Initialize chat history in session state if not present
    if "messages" not in st.session_state:
        log_event("Initializing new chat history")
        st.session_state.messages = []
    
    # Add a clear conversation button
    if st.button("Clear Conversation"):
        log_event("User clicked Clear Conversation button")
        st.session_state.messages = []
        st.rerun()

    # Display chat messages from history on app rerun
    message_count = len(st.session_state.messages)
    log_event(f"Displaying existing chat history ({message_count} messages)")
    for idx, message in enumerate(st.session_state.messages):
        message_type = message["type"]
        if message_type in ["human", "ai", "system"]:
            with st.chat_message(message_type):
                st.markdown(message["content"])    

    # Chat input for the user
    user_input = st.chat_input("What do you want to build today?")

    if user_input:
        log_event(f"Received new user input: '{user_input[:50]}{'...' if len(user_input) > 50 else ''}'")
        
        # We append a new request to the conversation explicitly
        st.session_state.messages.append({"type": "human", "content": user_input})
        log_event(f"Added user message to history (new length: {len(st.session_state.messages)})")
        
        # Display user prompt in the UI
        with st.chat_message("user"):
            st.markdown(user_input)

        # Display assistant response in chat message container
        response_content = ""
        chat_start_time = time.time()
        
        with st.chat_message("assistant"):
            log_event("Starting assistant response")
            message_placeholder = st.empty()  # Placeholder for updating the message
            
            # Add a spinner while loading
            with st.spinner("Archon is thinking..."):
                # Add a heartbeat to keep the session alive
                last_update_time = time.time()
                heartbeat_count = 0
                
                # Run the async generator to fetch responses
                async for chunk in run_agent_with_streaming(user_input):
                    current_time = time.time()
                    if not chunk:
                        log_event("Received empty chunk, skipping", level="WARNING")
                        continue
                        
                    response_content += chunk
                    chunk_len = len(chunk)
                    log_event(f"Added chunk of length {chunk_len} to response")
                    
                    # Update the placeholder with the current response content
                    message_placeholder.markdown(response_content)
                    
                    # If more than 10 seconds have passed since the last update,
                    # append a heartbeat message to keep the session alive
                    if current_time - last_update_time > 10:
                        heartbeat_count += 1
                        last_update_time = current_time
                        log_event(f"Sending heartbeat #{heartbeat_count} at {current_time - chat_start_time:.2f}s")
                        # This is just to keep the session active
                        st.session_state.last_heartbeat = current_time
        
        chat_elapsed = time.time() - chat_start_time
        log_event(f"Chat response completed in {chat_elapsed:.2f}s")
        log_event(f"Final response length: {len(response_content)} chars")
        
        st.session_state.messages.append({"type": "ai", "content": response_content})
        log_event(f"Added AI response to history (new length: {len(st.session_state.messages)})")

# Import nest_asyncio at the top level to make it available when needed
try:
    import nest_asyncio
    log_event("Successfully imported nest_asyncio")
except ImportError:
    # If nest_asyncio is not installed, the wrapper will handle the error
    log_event("Failed to import nest_asyncio", level="WARNING")
    pass

def chat_tab_wrapper():
    """
    Non-async wrapper for chat_tab that can be imported by streamlit_ui.py
    
    This function properly handles the nested event loop issue by using nest_asyncio
    and implementing proper error handling to prevent the app from crashing.
    """
    log_event("Entering chat_tab_wrapper")
    
    try:
        # Try to import and apply nest_asyncio
        import nest_asyncio
        log_event("Applying nest_asyncio to handle nested event loops")
        nest_asyncio.apply()
    except ImportError:
        # If nest_asyncio is not installed, log a warning
        log_event("WARNING: nest_asyncio is not installed. Chat functionality may be limited", level="WARNING")
        st.warning("nest_asyncio is not installed. Chat functionality may be limited.")
    except RuntimeError as e:
        # Handle potential runtime errors
        error_message = f"Runtime error when setting up async environment: {str(e)}"
        log_event(error_message, level="ERROR")
        st.error(error_message)
        return
    
    try:
        # Get the current event loop
        log_event("Getting current event loop")
        loop = asyncio.get_event_loop()
        
        # Set a timeout for the entire chat_tab function
        # This prevents the entire app from hanging if something goes wrong
        log_event("Setting up wrapper function with timeout")
        
        async def chat_tab_with_timeout():
            try:
                # 120 second timeout for the entire chat_tab function
                log_event("Starting chat_tab with 120 second timeout")
                async with asyncio.timeout(120):
                    await chat_tab()
                log_event("chat_tab completed successfully")
            except asyncio.TimeoutError:
                error_message = "The conversation timed out. Please try again or refresh the page."
                log_event(error_message, level="ERROR")
                st.error(error_message)
            except Exception as e:
                error_message = f"An error occurred: {str(e)}"
                log_event(f"ERROR in chat_tab: {error_message}\n{traceback.format_exc()}", level="ERROR")
                st.error(error_message)
        
        # Run the chat_tab function with timeout
        log_event("Running chat_tab_with_timeout in event loop")
        start_time = time.time()
        loop.run_until_complete(chat_tab_with_timeout())
        elapsed = time.time() - start_time
        log_event(f"chat_tab_wrapper completed in {elapsed:.2f}s")
        
    except RuntimeError as e:
        # Handle event loop errors
        if "This event loop is already running" in str(e):
            error_message = "Cannot run the chat interface in this context. Please try refreshing the page."
            log_event(error_message, level="ERROR")
            st.error(error_message)
        else:
            error_message = f"Error running chat interface: {str(e)}"
            log_event(f"ERROR: {error_message}\n{traceback.format_exc()}", level="ERROR")
            st.error(error_message)
    except Exception as e:
        # Handle any other exceptions
        error_message = f"Unexpected error: {str(e)}"
        log_event(f"CRITICAL ERROR: {error_message}\n{traceback.format_exc()}", level="ERROR")
        st.error(error_message)

# Make sure both functions are exported correctly
__all__ = ["chat_tab", "chat_tab_wrapper"]