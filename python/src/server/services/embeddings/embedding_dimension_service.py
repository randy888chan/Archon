"""
Embedding Dimension Service

Handles dynamic detection of embedding model dimensions and manages
schema migrations when embedding models change.
"""

import asyncio
from typing import Dict, Tuple, Optional, List
from contextlib import asynccontextmanager

from ...config.logfire_config import get_logger
from ..llm_provider_service import get_llm_client, get_embedding_model
from ..credential_service import credential_service

logger = get_logger(__name__)

# Recommended embedding models with their dimensions and characteristics
RECOMMENDED_MODELS = {
    "openai": {
        "text-embedding-3-small": {
            "dimensions": 1536,
            "description": "OpenAI's general-purpose model - high quality, balanced performance",
            "use_case": "General purpose, good for most applications",
            "provider": "openai"
        },
        "text-embedding-3-large": {
            "dimensions": 3072,
            "description": "OpenAI's largest model - highest quality but slower",
            "use_case": "When maximum quality is needed",
            "provider": "openai"
        }
    },
    "ollama": {
        "nomic-embed-text": {
            "dimensions": 768,
            "description": "Fast, lightweight model good for quick processing",
            "use_case": "Fast processing, resource-constrained environments",
            "provider": "ollama"
        },
        "mxbai-embed-large": {
            "dimensions": 1536,
            "description": "High-quality model compatible with OpenAI dimensions",
            "use_case": "Drop-in replacement for OpenAI with local processing",
            "provider": "ollama"
        },
        "snowflake-arctic-embed2": {
            "dimensions": 1024,
            "description": "Balanced performance model with good quality/speed ratio",
            "use_case": "Balanced performance and resource usage",
            "provider": "ollama"
        },
        "bge-m3": {
            "dimensions": 1024,
            "description": "Multilingual model supporting many languages",
            "use_case": "Multilingual content and international applications",
            "provider": "ollama"
        }
    },
    "google": {
        "text-embedding-004": {
            "dimensions": 768,
            "description": "Google's embedding model with good performance",
            "use_case": "Google ecosystem integration",
            "provider": "google"
        }
    }
}


class EmbeddingDimensionService:
    """Service for managing embedding dimensions and model transitions"""
    
    def __init__(self):
        self._dimension_cache: Dict[str, int] = {}
    
    async def detect_model_dimensions(self, model_name: str, provider: str = None) -> int:
        """
        Detect the dimensions of an embedding model by creating a test embedding.
        
        Args:
            model_name: Name of the embedding model
            provider: Provider override
            
        Returns:
            Number of dimensions the model produces
        """
        cache_key = f"{provider or 'default'}:{model_name}"
        
        # Check cache first
        if cache_key in self._dimension_cache:
            logger.info(f"Using cached dimensions for {cache_key}: {self._dimension_cache[cache_key]}")
            return self._dimension_cache[cache_key]
        
        # Check if we have a known dimension for this model
        known_dimension = self._get_known_model_dimension(model_name, provider)
        if known_dimension:
            logger.info(f"Using known dimensions for {model_name}: {known_dimension}")
            self._dimension_cache[cache_key] = known_dimension
            return known_dimension
        
        # Try to detect dimensions by creating a test embedding
        try:
            logger.info(f"Detecting dimensions for {model_name} with provider {provider}")
            
            async with get_llm_client(provider=provider, use_embedding_provider=True) as client:
                response = await client.embeddings.create(
                    input="test",
                    model=model_name
                )
                dimensions = len(response.data[0].embedding)
                logger.info(f"Detected {dimensions} dimensions for {model_name}")
                
                # Cache the result
                self._dimension_cache[cache_key] = dimensions
                return dimensions
                
        except Exception as e:
            logger.error(f"Failed to detect dimensions for {model_name}: {e}")
            # Return a sensible default
            return 1536
    
    def _get_known_model_dimension(self, model_name: str, provider: str = None) -> Optional[int]:
        """Get known dimensions for a model from our RECOMMENDED_MODELS."""
        # First try to find in specific provider
        if provider and provider in RECOMMENDED_MODELS:
            model_info = RECOMMENDED_MODELS[provider].get(model_name)
            if model_info:
                return model_info["dimensions"]
        
        # Search across all providers
        for provider_models in RECOMMENDED_MODELS.values():
            model_info = provider_models.get(model_name)
            if model_info:
                return model_info["dimensions"]
        
        return None
    
    def get_recommended_models(self) -> List[Dict]:
        """Get all recommended models as a flat list."""
        models = []
        for provider, provider_models in RECOMMENDED_MODELS.items():
            for model_name, model_info in provider_models.items():
                models.append({
                    "model_name": model_name,
                    "provider": provider,
                    "dimensions": model_info["dimensions"],
                    "description": model_info["description"],
                    "use_case": model_info["use_case"]
                })
        return models
    
    def get_provider_models(self, provider: str) -> List[Dict]:
        """Get recommended models for a specific provider."""
        if provider not in RECOMMENDED_MODELS:
            return []
        
        models = []
        for model_name, model_info in RECOMMENDED_MODELS[provider].items():
            models.append({
                "model_name": model_name,
                "provider": provider,
                "dimensions": model_info["dimensions"],
                "description": model_info["description"],
                "use_case": model_info["use_case"]
            })
        return models


# Global instance
embedding_dimension_service = EmbeddingDimensionService()