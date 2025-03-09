import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import get_env_var, save_env_var, reload_archon_graph

def environment_tab():
    """Display the environment variables configuration interface"""
    st.header("Environment Variables")
    st.write("- Configure your environment variables for Archon. These settings will be saved and used for future sessions.")
    st.write("- NOTE: Press 'enter' to save after inputting a variable, otherwise click the 'save' button at the bottom.")
    st.write("- HELP: Hover over the '?' icon on the right for each environment variable for help/examples.")
    st.warning("⚠️ If your agent service for MCP is already running, you'll need to restart it after changing environment variables.")

    # Define environment variables and their descriptions from .env.example
    env_vars = {
        "BASE_URL": {
            "description": "Base URL for the OpenAI instance (default is https://api.openai.com/v1)",
            "help": "OpenAI: https://api.openai.com/v1\n\n\n\nAnthropic: https://api.anthropic.com/v1\n\nOllama (example): http://localhost:11434/v1\n\nOpenRouter: https://openrouter.ai/api/v1",
            "sensitive": False
        },
        "LLM_API_KEY": {
            "description": "API key for your LLM provider",
            "help": "For OpenAI: https://help.openai.com/en/articles/4936850-where-do-i-find-my-openai-api-key\n\nFor Anthropic: https://console.anthropic.com/account/keys\n\nFor OpenRouter: https://openrouter.ai/keys\n\nFor Ollama, no need to set this unless you specifically configured an API key",
            "sensitive": True
        },
        "OPENAI_API_KEY": {
            "description": "Your OpenAI API key",
            "help": "Get your Open AI API Key by following these instructions -\n\nhttps://help.openai.com/en/articles/4936850-where-do-i-find-my-openai-api-key\n\nEven if using OpenRouter, you still need to set this for the embedding model.\n\nNo need to set this if using Ollama.",
            "sensitive": True
        },
        "SUPABASE_URL": {
            "description": "URL for your Supabase project",
            "help": "Get your SUPABASE_URL from the API section of your Supabase project settings -\nhttps://supabase.com/dashboard/project/<your project ID>/settings/api",
            "sensitive": False
        },
        "SUPABASE_SERVICE_KEY": {
            "description": "Service key for your Supabase project",
            "help": "Get your SUPABASE_SERVICE_KEY from the API section of your Supabase project settings -\nhttps://supabase.com/dashboard/project/<your project ID>/settings/api\nOn this page it is called the service_role secret.",
            "sensitive": True
        },
        "REASONER_MODEL": {
            "description": "The LLM you want to use for the reasoner",
            "help": "Example: o3-mini\n\nExample: deepseek-r1:7b-8k",
            "sensitive": False
        },
        "PRIMARY_MODEL": {
            "description": "The LLM you want to use for the primary agent/coder",
            "help": "Example: gpt-4o-mini\n\nExample: qwen2.5:14b-instruct-8k",
            "sensitive": False
        },
        "EMBEDDING_MODEL": {
            "description": "Embedding model you want to use",
            "help": "Example for Ollama: nomic-embed-text\n\nExample for OpenAI: text-embedding-3-small",
            "sensitive": False
        }
    }
    
    # Create a form for the environment variables
    with st.form("env_vars_form"):
        updated_values = {}
        
        # Display input fields for each environment variable
        for var_name, var_info in env_vars.items():
            current_value = get_env_var(var_name) or ""
            
            # Display the variable description
            st.subheader(var_name)
            st.write(var_info["description"])
            
            # Display input field (password field for sensitive data)
            if var_info["sensitive"]:
                # If there's already a value, show asterisks in the placeholder
                placeholder = "Set but hidden" if current_value else ""
                new_value = st.text_input(
                    f"Enter {var_name}:", 
                    type="password",
                    help=var_info["help"],
                    key=f"input_{var_name}",
                    placeholder=placeholder
                )
                # Only update if user entered something (to avoid overwriting with empty string)
                if new_value:
                    updated_values[var_name] = new_value
            else:
                new_value = st.text_input(
                    f"Enter {var_name}:", 
                    value=current_value,
                    help=var_info["help"],
                    key=f"input_{var_name}"
                )
                # Always update non-sensitive values (can be empty)
                updated_values[var_name] = new_value
            
            # Add a separator between variables
            st.markdown("---")
        
        # Submit button
        submitted = st.form_submit_button("Save Environment Variables")
        
        if submitted:
            # Save all updated values
            success = True
            for var_name, value in updated_values.items():
                if value:  # Only save non-empty values
                    if not save_env_var(var_name, value):
                        success = False
                        st.error(f"Failed to save {var_name}.")
            
            if success:
                st.success("Environment variables saved successfully!")
                reload_archon_graph()