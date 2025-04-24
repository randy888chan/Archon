import os
from supabase import create_client, Client
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_supabase_client() -> Client | None:
    """Initializes and returns the Supabase client."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        print("Error: Supabase URL or Key not found in environment variables.")
        return None
    try:
        return create_client(url, key)
    except Exception as e:
        print(f"Error initializing Supabase client: {e}")
        return None

def get_openai_client():
    """Initializes and returns the OpenAI client."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OpenAI API Key not found in environment variables.")
        return None
    try:
        client = openai.OpenAI(api_key=api_key)
        # Perform a simple test call (optional, but good for validation)
        client.models.list()
        return client
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        return None

# Example usage (optional, for testing)
if __name__ == "__main__":
    supabase = get_supabase_client()
    openai_client = get_openai_client()
    print(f"Supabase client: {'Initialized' if supabase else 'Failed'}")
    print(f"OpenAI client: {'Initialized' if openai_client else 'Failed'}") 