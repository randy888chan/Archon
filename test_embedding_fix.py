import asyncio
import sys
from pathlib import Path

# Add project root to path to allow importing archon modules
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from archon.llms_txt.vector_db.embedding_manager import OpenAIEmbeddingGenerator
from archon.llms_txt.vector_db.supabase_manager import SupabaseManager
from archon.agent_tools import retrieve_relevant_documentation_tool


async def test_embedding_fix():
    print("Initializing components...")

    # Initialize SupabaseManager and OpenAIEmbeddingGenerator
    supabase = SupabaseManager()
    embedding_manager = OpenAIEmbeddingGenerator()

    print("Components initialized successfully.")

    # Test the retrieve_relevant_documentation_tool with OpenAIEmbeddingGenerator
    query = "How do I use embeddings?"
    print(f"Testing retrieval with query: '{query}'")

    result = await retrieve_relevant_documentation_tool(
        supabase, embedding_manager, query
    )

    print("\nRetrieval result:")
    print(result[:500] + "..." if len(result) > 500 else result)
    print("\nTest completed successfully!")


if __name__ == "__main__":
    asyncio.run(test_embedding_fix())
