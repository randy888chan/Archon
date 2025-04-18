import asyncio
import sys
from pathlib import Path

# Add project root to path to allow importing archon modules
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from archon.llms_txt.vector_db.embedding_manager import OpenAIEmbeddingGenerator
import tiktoken


def create_large_text(token_count=10000):
    """Create a text string that exceeds the 8,192 token limit."""
    # Use a repeating pattern to create a large text
    base_text = "This is a test sentence to create a large chunk of text that exceeds the token limit. "
    # Calculate how many repetitions we need to reach the desired token count
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens_per_repetition = len(encoding.encode(base_text))
    repetitions_needed = token_count // tokens_per_repetition + 1

    large_text = base_text * repetitions_needed

    # Verify the token count
    actual_tokens = len(encoding.encode(large_text))
    print(f"Created text with {actual_tokens} tokens (target: {token_count})")

    return large_text


async def test_large_embedding():
    print("Testing embedding generation for large text chunks...")

    # Initialize OpenAIEmbeddingGenerator
    embedding_manager = OpenAIEmbeddingGenerator()

    # Create a text that exceeds the token limit
    large_text = create_large_text(10000)  # 10,000 tokens, well above the 8,192 limit

    print(f"Text length: {len(large_text)} characters")
    token_count = embedding_manager._count_tokens(large_text)
    print(f"Token count: {token_count}")

    print("Generating embedding for large text...")
    try:
        # This should now work with our fix
        embedding = embedding_manager.generate_embedding(large_text)
        print(f"Successfully generated embedding with {len(embedding)} dimensions")
        print(
            "Test passed: Large text was properly truncated and embedding was generated"
        )
    except Exception as e:
        print(f"Test failed: Error generating embedding: {e}")
        raise

    # Test batch embedding with a large text
    print("\nTesting batch embedding with large text...")
    try:
        # Create a batch with one normal text and one large text
        normal_text = "This is a normal-sized text."
        batch = [normal_text, large_text]

        embeddings = embedding_manager.generate_embeddings(batch)
        print(f"Successfully generated {len(embeddings)} embeddings in batch")
        print(f"First embedding dimensions: {len(embeddings[0])}")
        print(f"Second embedding dimensions: {len(embeddings[1])}")
        print("Test passed: Batch with large text was properly processed")
    except Exception as e:
        print(f"Test failed: Error generating batch embeddings: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(test_large_embedding())
