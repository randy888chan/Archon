from langgraph.types import Command
import streamlit as st
import asyncio
import uuid
import sys
import os
import time

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from archon.archon_graph import agentic_flow

@st.cache_resource
def get_thread_id():
    return str(uuid.uuid4())

thread_id = get_thread_id()

async def run_agent_with_streaming(user_input: str):
    """
    Run the agent with streaming text for the user_input prompt,
    while maintaining the entire conversation in `st.session_state.messages`.
    """
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    # First message from user
    if len(st.session_state.messages) == 1:
        try:
            # Set a timeout for the streaming to prevent hanging
            async with asyncio.timeout(60):  # 60 second timeout
                async for msg in agentic_flow.astream(
                        {"latest_user_message": user_input}, config, stream_mode="custom"
                    ):
                        # Add timeout check between chunks
                        yield msg
        except asyncio.TimeoutError:
            yield "\n\n[Timeout occurred: The agent took too long to respond. Please try again.]"
        except Exception as e:
            yield f"\n\n[An error occurred: {str(e)}]"
    # Continue the conversation
    else:
        try:
            # Set a timeout for the streaming to prevent hanging
            async with asyncio.timeout(60):  # 60 second timeout
                async for msg in agentic_flow.astream(
                    Command(resume=user_input), config, stream_mode="custom"
                ):
                    # Add timeout check between chunks
                    yield msg
        except asyncio.TimeoutError:
            yield "\n\n[Timeout occurred: The agent took too long to respond. Please try again.]"
        except Exception as e:
            yield f"\n\n[An error occurred: {str(e)}]"

async def chat_tab():
    """Display the chat interface for talking to Archon"""
    st.write("Describe to me an AI agent you want to build and I'll code it for you with Pydantic AI.")
    st.write("Example: Build me an AI agent that can search the web with the Brave API.")

    # Initialize chat history in session state if not present
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Add a clear conversation button
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        message_type = message["type"]
        if message_type in ["human", "ai", "system"]:
            with st.chat_message(message_type):
                st.markdown(message["content"])    

    # Chat input for the user
    user_input = st.chat_input("What do you want to build today?")

    if user_input:
        # We append a new request to the conversation explicitly
        st.session_state.messages.append({"type": "human", "content": user_input})
        
        # Display user prompt in the UI
        with st.chat_message("user"):
            st.markdown(user_input)

        # Display assistant response in chat message container
        response_content = ""
        with st.chat_message("assistant"):
            message_placeholder = st.empty()  # Placeholder for updating the message
            
            # Add a spinner while loading
            with st.spinner("Archon is thinking..."):
                # Add a heartbeat to keep the session alive
                last_update_time = time.time()
                
                # Run the async generator to fetch responses
                async for chunk in run_agent_with_streaming(user_input):
                    current_time = time.time()
                    response_content += chunk
                    
                    # Update the placeholder with the current response content
                    message_placeholder.markdown(response_content)
                    
                    # If more than 10 seconds have passed since the last update,
                    # append a heartbeat message to keep the session alive
                    if current_time - last_update_time > 10:
                        last_update_time = current_time
                        # This is just to keep the session active
                        st.session_state.last_heartbeat = current_time
        
        st.session_state.messages.append({"type": "ai", "content": response_content})

# Import nest_asyncio at the top level to make it available when needed
try:
    import nest_asyncio
except ImportError:
    # If nest_asyncio is not installed, the wrapper will handle the error
    pass

def chat_tab_wrapper():
    """
    Non-async wrapper for chat_tab that can be imported by streamlit_ui.py
    
    This function properly handles the nested event loop issue by using nest_asyncio
    and implementing proper error handling to prevent the app from crashing.
    """
    try:
        # Try to import and apply nest_asyncio
        import nest_asyncio
        nest_asyncio.apply()
    except ImportError:
        # If nest_asyncio is not installed, log a warning
        st.warning("nest_asyncio is not installed. Chat functionality may be limited.")
    except RuntimeError as e:
        # Handle potential runtime errors
        st.error(f"Runtime error when setting up async environment: {str(e)}")
        return
    
    try:
        # Get the current event loop
        loop = asyncio.get_event_loop()
        
        # Set a timeout for the entire chat_tab function
        # This prevents the entire app from hanging if something goes wrong
        async def chat_tab_with_timeout():
            try:
                # 120 second timeout for the entire chat_tab function
                async with asyncio.timeout(120):
                    await chat_tab()
            except asyncio.TimeoutError:
                st.error("The conversation timed out. Please try again or refresh the page.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        
        # Run the chat_tab function with timeout
        loop.run_until_complete(chat_tab_with_timeout())
    except RuntimeError as e:
        # Handle event loop errors
        if "This event loop is already running" in str(e):
            st.error("Cannot run the chat interface in this context. Please try refreshing the page.")
        else:
            st.error(f"Error running chat interface: {str(e)}")
    except Exception as e:
        # Handle any other exceptions
        st.error(f"Unexpected error: {str(e)}")

# Make sure both functions are exported correctly
__all__ = ["chat_tab", "chat_tab_wrapper"]