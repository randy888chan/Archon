from typing import Dict, List, Any, Optional  # Added Optional
import os  # Added os

# Ensure openai library is installed: pip install openai
try:
    from openai import (
        OpenAI,
        APIError,
    )  # Added APIError for specific exception handling
except ImportError:
    raise ImportError(
        "OpenAI library not found. Please install it using: pip install openai"
    )

# Import tiktoken for token counting
try:
    import tiktoken
except ImportError:
    raise ImportError(
        "Tiktoken library not found. Please install it using: pip install tiktoken"
    )

# Corrected import path assuming vector_db is sibling to utils
from ..utils.env_loader import EnvironmentLoader


class OpenAIEmbeddingGenerator:
    """Generate embeddings using OpenAI's API"""

    def __init__(self, env_loader: Optional[EnvironmentLoader] = None):
        """Initialize the OpenAI API client"""
        # Allow passing an existing env_loader or create a new one
        self.env_loader = env_loader or EnvironmentLoader(
            env_file_path="../../workbench/env_vars.json"
        )  # Adjusted path
        self.openai_config = self.env_loader.get_openai_config()

        # Validate required config
        if not self.openai_config.get("api_key"):
            raise ValueError(
                "OpenAI API Key ('LLM_API_KEY') is missing in configuration."
            )
        if not self.openai_config.get("embedding_model"):
            raise ValueError(
                "OpenAI Embedding Model ('EMBEDDING_MODEL') is missing in configuration."
            )

        # Initialize OpenAI client
        try:
            self.client = OpenAI(
                api_key=self.openai_config["api_key"],
                # Pass base_url only if it exists in the config
                base_url=self.openai_config.get("base_url"),
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
            print(
                "Warning: Attempting to generate embedding for empty text. Returning zero vector."
            )
            # Determine embedding dimension based on model (common case: 1536 for text-embedding-3-small)
            # This is a simplification; ideally, fetch dimension from model info if possible.
            # For now, assume 1536 if model name suggests it.
            dim = (
                1536
                if "1536" in self.embedding_model
                or "small" in self.embedding_model
                or "large" in self.embedding_model
                else 768
            )  # Default fallback
            return [0.0] * dim

        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text,
                encoding_format="float",  # Explicitly request float format
            )

            # Check if response data is valid and contains embeddings
            if response.data and len(response.data) > 0 and response.data[0].embedding:
                return response.data[0].embedding
            else:
                raise ValueError(
                    "Invalid response received from OpenAI API: No embedding data found."
                )

        except APIError as e:
            print(f"OpenAI API error generating single embedding: {e}")
            raise  # Re-raise the specific API error
        except Exception as e:
            print(f"Unexpected error generating single embedding: {e}")
            raise  # Re-raise other exceptions

    def _count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string using the appropriate encoding for the embedding model.

        Args:
            text: The text to count tokens for

        Returns:
            The number of tokens in the text
        """
        # Use cl100k_base encoding for text-embedding-3 models
        encoding_name = "cl100k_base"

        try:
            encoding = tiktoken.get_encoding(encoding_name)
            token_count = len(encoding.encode(text))
            return token_count
        except Exception as e:
            print(f"Error counting tokens: {e}. Using fallback estimation.")
            # Fallback: rough estimation (4 chars per token)
            return len(text) // 4

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
        if not texts:
            return []

        # Replace any empty strings with a single space
        processed_texts = [text if text else " " for text in texts]

        all_embeddings = []
        max_tokens_per_batch = 8192  # Maximum token limit for text-embedding-3-small

        # Create batches based on token count
        current_batch = []
        current_batch_tokens = 0

        for text in processed_texts:
            text_tokens = self._count_tokens(text)

            # If a single text exceeds the token limit, we need to truncate it
            # This is a simplistic approach - in production, consider more sophisticated truncation
            if text_tokens > max_tokens_per_batch:
                print(
                    f"Warning: Text with {text_tokens} tokens exceeds the maximum batch token limit ({max_tokens_per_batch}). It will be truncated."
                )
                # For simplicity, we'll just process it as a single item batch and let OpenAI truncate it
                # A better approach would be to implement proper truncation here

                if current_batch:
                    # Process the current batch first
                    batch_embeddings = self._process_embedding_batch(current_batch)
                    all_embeddings.extend(batch_embeddings)
                    current_batch = []
                    current_batch_tokens = 0

                # Process the oversized text as a single item batch
                batch_embeddings = self._process_embedding_batch([text])
                all_embeddings.extend(batch_embeddings)
                continue

            # If adding this text would exceed the token limit, process the current batch first
            if (
                current_batch_tokens + text_tokens > max_tokens_per_batch
                and current_batch
            ):
                batch_embeddings = self._process_embedding_batch(current_batch)
                all_embeddings.extend(batch_embeddings)
                current_batch = []
                current_batch_tokens = 0

            # Add the text to the current batch
            current_batch.append(text)
            current_batch_tokens += text_tokens

        # Process any remaining texts in the last batch
        if current_batch:
            batch_embeddings = self._process_embedding_batch(current_batch)
            all_embeddings.extend(batch_embeddings)

        # Final check: Ensure the number of embeddings matches the number of processed texts
        if len(all_embeddings) != len(processed_texts):
            print(
                f"Warning: Final embedding count ({len(all_embeddings)}) does not match input text count ({len(processed_texts)}). Some batches may have failed."
            )

        return all_embeddings

    def _process_embedding_batch(self, batch: List[str]) -> List[List[float]]:
        """Process a single batch of texts for embedding generation.

        Args:
            batch: List of texts to generate embeddings for

        Returns:
            List of embedding vectors for the batch

        Raises:
            APIError: If the OpenAI API call fails
            Exception: For other unexpected errors
        """
        batch_index_info = f"(Batch of {len(batch)} texts)"  # For logging

        try:
            print(
                f"Generating embeddings for batch {batch_index_info}..."
            )  # Added print statement
            response = self.client.embeddings.create(
                model=self.embedding_model, input=batch, encoding_format="float"
            )

            if response.data and len(response.data) == len(batch):
                batch_embeddings = [item.embedding for item in response.data]
                if not all(batch_embeddings):
                    # Log which batch failed if possible
                    print(
                        f"Warning: Missing embedding data in response for batch {batch_index_info}."
                    )
                    # Handle missing embeddings - Option: fill with zero vectors of correct dimension
                    # For now, we'll raise an error as before, but the error is now batch-specific.
                    raise ValueError(
                        f"Invalid response received from OpenAI API: Missing embedding data in batch {batch_index_info}."
                    )
                print(
                    f"Successfully processed batch {batch_index_info}."
                )  # Added success print
                return batch_embeddings
            else:
                raise ValueError(
                    f"Invalid response received from OpenAI API for batch {batch_index_info}: Mismatch in batch size or missing data. Expected {len(batch)}, got {len(response.data) if response.data else 0}."
                )

        except APIError as e:
            print(
                f"OpenAI API error generating embeddings for batch {batch_index_info}: {e}"
            )
            # Option 1: Re-raise immediately, stopping the process
            raise
            # Option 2: Log error and continue, potentially skipping this batch (results in incomplete data)
            # print(f"Skipping batch {batch_index_info} due to API error.")
            # continue # This would require careful handling of indices later
        except Exception as e:
            print(
                f"Unexpected error generating embeddings for batch {batch_index_info}: {e}"
            )
            raise  # Re-raise other exceptions

    def generate_node_embeddings(
        self, nodes: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
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
        original_indices = []  # Keep track of which node corresponds to which text

        for i, node in enumerate(nodes):
            # Construct the text for content embedding
            content = node.get("content", "")
            title = node.get("title")
            # Prepend title for better context if available
            content_with_title = f"{title}\n\n{content}" if title else content

            # Add non-empty text to the list for batching
            if content_with_title.strip():  # Check if it's not just whitespace
                texts_to_embed.append(content_with_title)
                original_indices.append(i)
            else:
                print(
                    f"Warning: Node {node.get('metadata', {}).get('original_id', i)} has empty content/title for embedding."
                )

        # Generate embeddings in batch if there are texts to process
        embeddings = []
        if texts_to_embed:
            try:
                embeddings = self.generate_embeddings(texts_to_embed)
            except Exception as e:
                print(
                    f"Error generating batch embeddings for nodes: {e}. Proceeding without embeddings for affected nodes."
                )
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
