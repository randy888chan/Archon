from typing import Dict, List, Any, Optional # Added Optional
import os # Added os
# Ensure necessary libraries are installed: pip install openai langchain-community
try:
    from openai import OpenAI, APIError
except ImportError:
    # Allow running without OpenAI if Ollama is the provider
    OpenAI = None
    APIError = None # Define APIError as None if openai isn't installed
    print("Warning: OpenAI library not found. OpenAI provider will not be available.")

try:
    from langchain_ollama.embeddings import OllamaEmbeddings
except ImportError:
    OllamaEmbeddings = None
    print("Warning: langchain-community library not found. Ollama provider will not be available.")

# Corrected import path assuming vector_db is sibling to utils
from ..utils.env_loader import EnvironmentLoader

class EmbeddingManager:
    """Manages embedding generation using configured provider (OpenAI or Ollama)."""

    def __init__(self, env_loader: Optional[EnvironmentLoader] = None):
        """Initialize the embedding client based on configuration."""
        self.env_loader = env_loader or EnvironmentLoader(env_file_path="../../workbench/env_vars.json") # Adjusted path
        self.config = self.env_loader.config # Access the config directly

        self.provider = self.config.get("EMBEDDING_PROVIDER", "OpenAI").lower() # Default to OpenAI
        self.client = None
        self.embedding_model = None
        # Read dimension from config, default to None if not found
        self.embedding_dim = self.config.get("EMBEDDING_DIMENSION")
        if self.embedding_dim:
            try:
                self.embedding_dim = int(self.embedding_dim)
            except ValueError:
                print(f"Warning: Invalid EMBEDDING_DIMENSION value '{self.embedding_dim}'. Must be an integer. Ignoring.")
                self.embedding_dim = None
        # else: # No need for else if only print was removed


        if self.provider == "ollama":
            if OllamaEmbeddings is None:
                raise ImportError("Ollama provider selected, but langchain-community library is not installed.")

            ollama_base_url = self.config.get("EMBEDDING_BASE_URL")
            self.embedding_model = self.config.get("EMBEDDING_MODEL")

            if not self.embedding_model:
                raise ValueError("Ollama Embedding Model ('EMBEDDING_MODEL') is missing in configuration.")
            if not ollama_base_url:
                 print("Warning: Ollama Base URL ('EMBEDDING_BASE_URL') not found, using default.")
                 # OllamaEmbeddings might have a default, or handle None

            try:
                self.client = OllamaEmbeddings(
                    base_url=ollama_base_url,
                    model=self.embedding_model
                )
                # Note: OllamaEmbeddings doesn't expose dimension easily, may need to infer or hardcode common values
                # Only infer dimension if not provided in config
                if self.embedding_dim is None:
                    try:
                        dummy_embedding = self.client.embed_query("test")
                        self.embedding_dim = len(dummy_embedding)
                    except Exception as e:
                        print(f"Warning: Could not determine Ollama embedding dimension via test query: {e}. Using fallback dimension 768.")
                        self.embedding_dim = 768 # Fallback if inference fails and not configured
            except Exception as e:
                print(f"Error initializing Ollama client: {e}")
                raise

        elif self.provider == "openai":
            if OpenAI is None:
                raise ImportError("OpenAI provider selected, but openai library is not installed.")

            openai_api_key = self.config.get("EMBEDDING_API_KEY") or self.config.get("LLM_API_KEY") # Try embedding-specific key first
            openai_base_url = self.config.get("EMBEDDING_BASE_URL") or self.config.get("BASE_URL") # Try embedding-specific URL first
            self.embedding_model = self.config.get("EMBEDDING_MODEL")

            if not openai_api_key:
                raise ValueError("OpenAI API Key ('LLM_API_KEY') is missing in configuration.")
            if not self.embedding_model:
                raise ValueError("OpenAI Embedding Model ('EMBEDDING_MODEL') is missing in configuration.")

            try:
                self.client = OpenAI(
                    api_key=openai_api_key,
                    base_url=openai_base_url # Pass base_url only if it exists
                )
                # Only infer dimension if not provided in config
                if self.embedding_dim is None:
                    if "ada-002" in self.embedding_model or "3-small" in self.embedding_model or "3-large" in self.embedding_model:
                         self.embedding_dim = 1536
                    elif "ada-001" in self.embedding_model: # Older model? Check specific name
                         self.embedding_dim = 1024
                    else:
                         # Default fallback if model name doesn't match common patterns
                         self.embedding_dim = 1536 # Defaulting to common OpenAI dimension
                         print(f"Warning: Could not infer dimension for model '{self.embedding_model}'. Using fallback: {self.embedding_dim}")
            except Exception as e:
                print(f"Error initializing OpenAI client: {e}")
                raise
        else:
            raise ValueError(f"Unsupported embedding provider: {self.provider}. Choose 'OpenAI' or 'Ollama'.")

        if not self.client:
             raise RuntimeError("Embedding client failed to initialize.")

    def generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding for a single text string using the configured provider.

        Args:
            text: Text to generate embedding for.

        Returns:
            List of floats representing the embedding vector.

        Raises:
            ValueError: If the input text is invalid.
            Exception: If the API call to the provider fails or other errors occur.
        """
        if not text:
             print("Warning: Attempting to generate embedding for empty text. Returning zero vector.")
             # Use the stored dimension, ensure it's set
             if self.embedding_dim is None:
                 # This should ideally not happen if initialization worked
                 print("Error: Embedding dimension not determined. Using fallback 768.")
                 return [0.0] * 768
             return [0.0] * self.embedding_dim

        try:
            if self.provider == "openai":
                if APIError is None: # Check if OpenAI client could be initialized
                     raise RuntimeError("OpenAI client not available.")
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=text,
                    encoding_format="float"
                )
                if response.data and len(response.data) > 0 and response.data[0].embedding:
                    return response.data[0].embedding
                else:
                    raise ValueError("Invalid response received from OpenAI API: No embedding data found.")
            elif self.provider == "ollama":
                embedding = self.client.embed_query(text)
                if embedding and isinstance(embedding, list):
                     return embedding
                else:
                     raise ValueError("Invalid response received from Ollama API: No embedding data found.")
            else:
                # This case should ideally not be reached due to __init__ validation
                raise RuntimeError(f"Invalid provider '{self.provider}' encountered during embedding generation.")

        except APIError as e: # Specific to OpenAI
            print(f"OpenAI API error generating single embedding: {e}")
            raise # Re-raise the specific API error
        except Exception as e:
            # Catch general exceptions which might come from Ollama or other issues
            print(f"Error generating single embedding with {self.provider.upper()} provider: {e}")
            raise # Re-raise other exceptions

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of text strings using the configured provider.

        Args:
            texts: List of texts to generate embeddings for.

        Returns:
            List of embedding vectors (list of lists of floats). Returns empty list if input is empty.

        Raises:
            ValueError: If the input texts are invalid or API response is malformed.
            Exception: If the API call to the provider fails or other errors occur.
        """
        if not texts:
            return []

        # Replace any empty strings with a single space
        processed_texts = [text if text else " " for text in texts]

        all_embeddings = []

        if self.provider == "openai":
            if APIError is None: raise RuntimeError("OpenAI client not available.")
            batch_size = 2000  # OpenAI batch size limit
            for i in range(0, len(processed_texts), batch_size):
                batch = processed_texts[i:i + batch_size]
                batch_index_info = f"(Indices {i} to {i + len(batch) - 1})"
                try:
                    response = self.client.embeddings.create(
                        model=self.embedding_model,
                        input=batch,
                        encoding_format="float"
                    )
                    if response.data and len(response.data) == len(batch):
                        batch_embeddings = [item.embedding for item in response.data]
                        if not all(e is not None for e in batch_embeddings): # Check for None embeddings
                            print(f"Warning: Missing embedding data in OpenAI response for batch {batch_index_info}.")
                            raise ValueError(f"Invalid response from OpenAI API: Missing embedding data in batch {batch_index_info}.")
                        all_embeddings.extend(batch_embeddings)
                    else:
                        raise ValueError(f"Invalid response from OpenAI API for batch {batch_index_info}: Mismatch in batch size or missing data. Expected {len(batch)}, got {len(response.data) if response.data else 0}.")
                except APIError as e:
                    print(f"OpenAI API error generating embeddings for batch {batch_index_info}: {e}")
                    raise
                except Exception as e:
                    print(f"Unexpected error generating OpenAI embeddings for batch {batch_index_info}: {e}")
                    raise

        elif self.provider == "ollama":
            # Ollama's embed_documents handles batching internally (usually)
            try:
                all_embeddings = self.client.embed_documents(processed_texts)
                if len(all_embeddings) != len(processed_texts):
                     raise ValueError(f"Ollama API response length mismatch. Expected {len(processed_texts)}, got {len(all_embeddings)}.")
                if not all(e is not None for e in all_embeddings): # Check for None embeddings
                     raise ValueError("Ollama API returned None for one or more embeddings.")
            except Exception as e:
                # More detailed error logging
                print(f"Error generating Ollama embeddings: Type={type(e).__name__}, Message={e}")
                # Optionally print traceback if needed:
                # import traceback
                # traceback.print_exc()
                raise
        else:
             raise RuntimeError(f"Invalid provider '{self.provider}' encountered during batch embedding generation.")

        # Final check: Ensure the number of embeddings matches the number of processed texts
        if len(all_embeddings) != len(processed_texts):
             print(f"Warning: Final embedding count ({len(all_embeddings)}) does not match input text count ({len(processed_texts)}). Some batches may have failed.")
             # Depending on error handling strategy above, this might indicate partial failure.

        return all_embeddings

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
                 print(f"Error generating batch embeddings for nodes using {self.provider.upper()}: {e}. Proceeding without embeddings for affected nodes.")
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