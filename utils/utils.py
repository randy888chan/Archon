from supabase import Client, create_client
from openai import AsyncOpenAI
from dotenv import load_dotenv
from datetime import datetime
from functools import wraps
from typing import Optional
import streamlit as st
import webbrowser
import importlib
import inspect
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file
load_dotenv()

def write_to_log(message: str):
    """Write a message to the logs.txt file in the workbench directory.
    
    Args:
        message: The message to log
    """
    # Get the directory one level up from the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    workbench_dir = os.path.join(parent_dir, "workbench")
    log_path = os.path.join(workbench_dir, "logs.txt")
    os.makedirs(workbench_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}\n"

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(log_entry)

def get_env_var(var_name: str) -> Optional[str]:
    """Get an environment variable from the saved JSON file or from environment variables.
    
    Args:
        var_name: The name of the environment variable to retrieve
        
    Returns:
        The value of the environment variable or None if not found
    """
    # Path to the JSON file storing environment variables
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    env_file_path = os.path.join(current_dir, "env_vars.json")
    
    # First try to get from JSON file
    if os.path.exists(env_file_path):
        try:
            with open(env_file_path, "r") as f:
                env_vars = json.load(f)
                if var_name in env_vars and env_vars[var_name]:
                    return env_vars[var_name]
        except (json.JSONDecodeError, IOError) as e:
            write_to_log(f"Error reading env_vars.json: {str(e)}")
    
    # If not found in JSON, try to get from environment variables
    return os.environ.get(var_name)

def save_env_var(var_name: str, value: str) -> bool:
    """Save an environment variable to the JSON file.
    
    Args:
        var_name: The name of the environment variable
        value: The value to save
        
    Returns:
        True if successful, False otherwise
    """
    # Path to the JSON file storing environment variables
    current_dir = os.path.dirname(os.path.abspath(__file__))
    env_file_path = os.path.join(current_dir, "env_vars.json")
    
    # Load existing env vars or create empty dict
    env_vars = {}
    if os.path.exists(env_file_path):
        try:
            with open(env_file_path, "r") as f:
                env_vars = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            write_to_log(f"Error reading env_vars.json: {str(e)}")
            # Continue with empty dict if file is corrupted
    
    # Update the variable
    env_vars[var_name] = value
    
    # Save back to file
    try:
        with open(env_file_path, "w") as f:
            json.dump(env_vars, f, indent=2)
        return True
    except IOError as e:
        write_to_log(f"Error writing to env_vars.json: {str(e)}")
        return False

def get_clients():
    openai_client = None
    base_url = get_env_var('BASE_URL') or 'https://api.openai.com/v1'
    api_key = get_env_var('LLM_API_KEY') or 'no-llm-api-key-provided'
    is_ollama = any(kw in base_url.lower() for kw in ["localhost", "host.docker.internal", "ollama"])

    if is_ollama:
        openai_client = AsyncOpenAI(base_url=base_url,api_key=api_key)
    elif get_env_var("OPENAI_API_KEY"):
        openai_client = AsyncOpenAI(api_key=get_env_var("OPENAI_API_KEY"))
    else:
        openai_client = None

    supabase = None
    supabase_url = get_env_var("SUPABASE_URL")
    supabase_key = get_env_var("SUPABASE_SERVICE_KEY")
    if supabase_url and get_env_var("SUPABASE_SERVICE_KEY"):
        try:
            supabase: Client = Client(supabase_url, supabase_key)
        except Exception as e:
            print(f"Failed to initialize Supabase: {e}")
            write_to_log(f"Failed to initialize Supabase: {e}")

    return openai_client, supabase      

def log_node_execution(func):
    """Decorator to log the start and end of graph node execution.
    
    Args:
        func: The async function to wrap
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        func_name = func.__name__
        write_to_log(f"Starting node: {func_name}")
        try:
            result = await func(*args, **kwargs)
            write_to_log(f"Completed node: {func_name}")
            return result
        except Exception as e:
            write_to_log(f"Error in node {func_name}: {str(e)}")
            raise
    return wrapper

# Helper function to create a button that opens a tab in a new window
def create_new_tab_button(label, tab_name, key=None, use_container_width=False):
    """Create a button that opens a specified tab in a new browser window"""
    # Create a unique key if none provided
    if key is None:
        key = f"new_tab_{tab_name.lower().replace(' ', '_')}"
    
    # Get the base URL
    base_url = st.query_params.get("base_url", "")
    if not base_url:
        # If base_url is not in query params, use the default localhost URL
        base_url = "http://localhost:8501"
    
    # Create the URL for the new tab
    new_tab_url = f"{base_url}/?tab={tab_name}"
    
    # Create a button that will open the URL in a new tab when clicked
    if st.button(label, key=key, use_container_width=use_container_width):
        webbrowser.open_new_tab(new_tab_url)

# Function to reload the archon_graph module
def reload_archon_graph():
    """Reload the archon_graph module to apply new environment variables"""
    try:
        # First reload pydantic_ai_coder
        import archon.pydantic_ai_coder
        importlib.reload(archon.pydantic_ai_coder)
        
        # Then reload archon_graph which imports pydantic_ai_coder
        import archon.archon_graph
        importlib.reload(archon.archon_graph)
        
        st.success("Successfully reloaded Archon modules with new environment variables!")
        return True
    except Exception as e:
        st.error(f"Error reloading Archon modules: {str(e)}")
        return False        
