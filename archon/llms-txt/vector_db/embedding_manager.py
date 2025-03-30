from typing import Dict, List, Any, Optional # Added Optional
import os # Added os
# Ensure openai library is installed: pip install openai
try:
    from openai import OpenAI, APIError # Added APIError for specific exception handling
except ImportError:
    raise ImportError("OpenAI library not found. Please install it using: pip install openai")

# Corrected import path assuming vector_db is sibling to utils
from ..utils.env_loader import EnvironmentLoader

class OpenAIEmbeddingGenerator:
    """Generate embeddings using OpenAI's API"""

    def __init__(self, env_loader: Optional[EnvironmentLoader] = None):
        """Initialize the OpenAI API client"""
        # Allow passing an existing env_loader or create a new one
        self.env_loader = env_loader or EnvironmentLoader(env_file_path="../../workbench/env_vars.json") # Adjusted path
        self.openai_config = self.env_loader.get_openai_config()

        # Validate required config
        if not self.openai_config.get("api_key"):
            raise ValueError("OpenAI API Key ('LLM_API_KEY') is missing in configuration.")
        if not self.openai_config.get("embedding_model"):
             raise ValueError("OpenAI Embedding Model ('EMBEDDING_MODEL') is missing in configuration.")

        # Initialize OpenAI client
        try:
            self.client = OpenAI(
                api_key=self.openai_config["api_key"],
                # Pass base_url only if it exists in the config
                base_url=self.openai_config.get("base_url")
            )
            print("OpenAI client initialized successfully.")
        except Exception as e:
            print(f"Error initializing OpenAI client: {e}")
            raise

        self.embedding_model = self.openai_config["embedding_model"]
        print(f"Using OpenAI embedding model: {self.embedding_model}")

    def generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding for a single text string.

        Args:
            text: Text to generate embedding for.

        Returns:
            List of floats representing the embedding vector.

        Raises:
            APIError: If the OpenAI API call fails.
            Exception: For other unexpected errors.
        """
        if not text:
             print("Warning: Attempting to generate embedding for empty text. Returning zero vector.")
             # Determine embedding dimension based on model (common case: 1536 for text-embedding-3-small)
             # This is a simplification; ideally, fetch dimension from model info if possible.
             # For now, assume 1536 if model name suggests it.
             dim = 1536 if "1536" in self.embedding_model or "small" in self.embedding_model or "large" in self.embedding_model else 768 # Default fallback
             return [0.0] * dim

        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text,
                encoding_format="float" # Explicitly request float format
            )

            # Check if response data is valid and contains embeddings
            if response.data and len(response.data) > 0 and response.data[0].embedding:
                 return response.data[0].embedding
            else:
                 raise ValueError("Invalid response received from OpenAI API: No embedding data found.")

        except APIError as e:
            print(f"OpenAI API error generating single embedding: {e}")
            raise # Re-raise the specific API error
        except Exception as e:
            print(f"Unexpected error generating single embedding: {e}")
            raise # Re-raise other exceptions

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of text strings (handles batching).

        Args:
            texts: List of texts to generate embeddings for.

        Returns:
            List of embedding vectors (list of lists of floats). Returns empty list if input is empty.

        Raises:
            APIError: If the OpenAI API call fails.
            Exception: For other unexpected errors.
        """
        # Handle empty input list
        if not texts:
            return []

        # Replace any empty strings with a placeholder or handle them
        # Option 1: Replace with a space (minimal impact)
        processed_texts = [text if text else " " for text in texts]
        # Option 2: Filter out empty strings (might cause index mismatch if not handled carefully later)
        # non_empty_texts = [text for text in texts if text]
        # if not non_empty_texts: return [] # All were empty

        try:
            # OpenAI API handles batching internally
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=processed_texts, # Use processed texts
                encoding_format="float"
            )

            # Check response structure and extract embeddings
            if response.data and len(response.data) == len(processed_texts):
                 embeddings = [item.embedding for item in response.data]
                 # Ensure all items actually have an embedding
                 if not all(embeddings):
                     raise ValueError("Invalid response received from OpenAI API: Missing embedding data in batch.")
                 return embeddings
            else:
                 raise ValueError(f"Invalid response received from OpenAI API: Mismatch in batch size or missing data. Expected {len(processed_texts)}, got {len(response.data) if response.data else 0}.")

        except APIError as e:
            print(f"OpenAI API error generating batch embeddings: {e}")
            raise # Re-raise the specific API error
        except Exception as e:
            print(f"Unexpected error generating batch embeddings: {e}")
            raise # Re-raise other exceptions

    def generate_node_embeddings(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate embeddings for hierarchical nodes and add them to the node dictionaries.

        Args:
            nodes: List of node dictionaries, each expected to have 'content' and optionally 'title'/'path'.

        Returns:
            The same list of node dictionaries with an 'embedding' field added/updated.
            Nodes where embedding generation fails might not have the 'embedding' key.
        """
        if not nodes:
            return []

        texts_to_embed = []
        original_indices = [] # Keep track of which node corresponds to which text

        for i, node in enumerate(nodes):
            # Construct the text for content embedding
            content = node.get("content", "")
            title = node.get("title")
            # Prepend title for better context if available
            content_with_title = f"{title}\n\n{content}" if title else content

            # Add non-empty text to the list for batching
            if content_with_title.strip(): # Check if it's not just whitespace
                 texts_to_embed.append(content_with_title)
                 original_indices.append(i)
            else:
                 print(f"Warning: Node {node.get('metadata', {}).get('original_id', i)} has empty content/title for embedding.")


        # Generate embeddings in batch if there are texts to process
        embeddings = []
        if texts_to_embed:
             try:
                 embeddings = self.generate_embeddings(texts_to_embed)
             except Exception as e:
                 print(f"Error generating batch embeddings for nodes: {e}. Proceeding without embeddings for affected nodes.")
                 # In case of error, embeddings list will be empty or incomplete

        # Add embeddings back to the corresponding nodes
        embedding_map = dict(zip(original_indices, embeddings))

        for i, node in enumerate(nodes):
            if i in embedding_map:
                 node["embedding"] = embedding_map[i]
                 # Ensure metadata exists
                 if "metadata" not in node:
                     node["metadata"] = {}
                 # Indicate that an embedding was successfully generated and added
                 node["metadata"]["embedding_generated"] = True
            else:
                 # Ensure metadata exists even if embedding failed
                 if "metadata" not in node:
                     node["metadata"] = {}
                 node["metadata"]["embedding_generated"] = False
                 # Optionally add a zero vector or leave 'embedding' key absent
                 # node["embedding"] = [0.0] * 1536 # Example: Add zero vector

            # We don't generate/store separate title embeddings as per the spec's note
            # node["metadata"]["has_title_embedding"] = False # Explicitly state not used

        return nodes