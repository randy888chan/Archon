import json
import os
from typing import Dict, Any

class EnvironmentLoader:
    """Load environment variables from env_vars.json"""

    def __init__(self, env_file_path: str = "env_vars.json"):
        """Initialize with path to env_vars.json"""
        self.env_file_path = env_file_path
        self.config = self._load_env()

    def _load_env(self) -> Dict[str, Any]:
        """Load environment variables from JSON file"""
        # Construct the absolute path relative to this file's directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        absolute_env_file_path = os.path.join(base_dir, self.env_file_path)

        if not os.path.exists(absolute_env_file_path):
            # Try looking relative to the project root (assuming utils is one level down)
            project_root_path = os.path.join(base_dir, '..', self.env_file_path)
            if os.path.exists(project_root_path):
                 absolute_env_file_path = project_root_path
            else:
                 # As a last resort, check the path mentioned by the user if different
                 user_mentioned_path = "workbench/env_vars.json" # Hardcoding for now, ideally pass this in
                 if self.env_file_path != user_mentioned_path and os.path.exists(user_mentioned_path):
                     absolute_env_file_path = user_mentioned_path
                 else:
                     # If none of the paths exist, raise the error with the original path tried
                     original_path_to_report = os.path.join(base_dir, self.env_file_path)
                     raise FileNotFoundError(f"Environment file not found at expected paths: {original_path_to_report} or {project_root_path} or {user_mentioned_path}")

        with open(absolute_env_file_path, "r") as f:
            env_data = json.load(f)

        # Get current profile
        current_profile = env_data.get("current_profile", "default")
        profile_config = env_data.get("profiles", {}).get(current_profile, {})

        if not profile_config:
            raise ValueError(f"Profile '{current_profile}' not found in environment file: {absolute_env_file_path}")

        # Load environment variables specified in the profile into os.environ
        # This makes them accessible via os.getenv() elsewhere if needed,
        # but the primary access method should be via get_supabase_config etc.
        for key, value in profile_config.items():
             if value is not None: # Ensure value is not None before setting
                 os.environ[key] = str(value) # Env vars must be strings

        return profile_config

    def get_supabase_config(self) -> Dict[str, str]:
        """Get Supabase connection configuration"""
        url = self.config.get("SUPABASE_URL") or os.getenv("SUPABASE_URL")
        key = self.config.get("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_SERVICE_KEY")
        if not url or not key:
             raise ValueError("Supabase URL or Key not found in environment config")
        return {"url": url, "key": key}

    def get_openai_config(self) -> Dict[str, str]:
        """Get OpenAI API configuration"""
        api_key = self.config.get("LLM_API_KEY") or os.getenv("LLM_API_KEY")
        base_url = self.config.get("BASE_URL") or os.getenv("BASE_URL") # Optional
        embedding_model = self.config.get("EMBEDDING_MODEL") or os.getenv("EMBEDDING_MODEL")

        if not api_key:
             raise ValueError("OpenAI API Key (LLM_API_KEY) not found in environment config")
        if not embedding_model:
             raise ValueError("OpenAI Embedding Model (EMBEDDING_MODEL) not found in environment config")

        config = {"api_key": api_key, "embedding_model": embedding_model}
        if base_url:
             config["base_url"] = base_url
        return config