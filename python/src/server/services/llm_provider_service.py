"""
LLM Provider Service

Provides a unified interface for creating OpenAI-compatible clients for different LLM providers.
Supports OpenAI, Ollama, and Google Gemini.
"""

import time
from contextlib import asynccontextmanager
from typing import Any

import openai

from ..config.logfire_config import get_logger
from .credential_service import credential_service

logger = get_logger(__name__)

# Settings cache with TTL
_settings_cache: dict[str, tuple[Any, float]] = {}
_CACHE_TTL_SECONDS = 300  # 5 minutes


async def get_best_ollama_instance(model_name: str = None) -> dict | None:
    """
    Select the best Ollama instance for load balancing.
    
    Args:
        model_name: Optional model name for model-specific routing
        
    Returns:
        Best instance dict or None if no healthy instances available
    """
    try:
        # Get healthy instances from database
        healthy_instances = await credential_service.get_healthy_ollama_instances()
        
        if not healthy_instances:
            logger.warning("No healthy Ollama instances available, falling back to primary")
            primary_instance = await credential_service.get_primary_ollama_instance()
            return primary_instance
        
        if len(healthy_instances) == 1:
            # Only one healthy instance, use it
            return healthy_instances[0]
        
        # Load balancing logic: weighted random selection
        import random
        
        total_weight = 0
        weighted_instances = []
        
        for instance in healthy_instances:
            # Base weight from configuration
            base_weight = instance.get("loadBalancingWeight", 100)
            
            # Health factor (prefer faster instances)
            response_time = instance.get("responseTimeMs", 1000)
            health_factor = 1.0 if response_time < 1000 else 0.5
            
            # Model availability factor (prefer instances with the required model)
            model_factor = 1.0
            if model_name:
                available_models = instance.get("modelsAvailable", 0)
                # Simple heuristic: prefer instances with more models
                model_factor = 1.2 if available_models > 5 else 1.0
            
            # Calculate final weight
            final_weight = base_weight * health_factor * model_factor
            total_weight += final_weight
            weighted_instances.append((instance, final_weight))
        
        # Weighted random selection
        if total_weight > 0:
            random_value = random.uniform(0, total_weight)
            cumulative_weight = 0
            
            for instance, weight in weighted_instances:
                cumulative_weight += weight
                if random_value <= cumulative_weight:
                    logger.debug(f"Selected Ollama instance {instance['id']} ({instance['name']}) with weight {weight:.2f}")
                    return instance
        
        # Fallback: return first healthy instance
        logger.debug("Weighted selection failed, using first healthy instance")
        return healthy_instances[0]
        
    except Exception as e:
        logger.error(f"Error selecting best Ollama instance: {e}")
        # Fallback to primary instance
        try:
            primary_instance = await credential_service.get_primary_ollama_instance()
            return primary_instance
        except Exception as fallback_error:
            logger.error(f"Error getting primary Ollama instance: {fallback_error}")
            return None


def _get_cached_settings(key: str) -> Any | None:
    """Get cached settings if not expired."""
    if key in _settings_cache:
        value, timestamp = _settings_cache[key]
        if time.time() - timestamp < _CACHE_TTL_SECONDS:
            return value
        else:
            # Expired, remove from cache
            del _settings_cache[key]
    return None


def _set_cached_settings(key: str, value: Any) -> None:
    """Cache settings with current timestamp."""
    _settings_cache[key] = (value, time.time())


@asynccontextmanager
async def get_llm_client(provider: str | None = None, use_embedding_provider: bool = False):
    """
    Create an async OpenAI-compatible client based on the configured provider.

    This context manager handles client creation for different LLM providers
    that support the OpenAI API format.

    Args:
        provider: Override provider selection
        use_embedding_provider: Use the embedding-specific provider if different

    Yields:
        openai.AsyncOpenAI: An OpenAI-compatible client configured for the selected provider
    """
    client = None

    try:
        # Get provider configuration from database settings
        if provider:
            # Explicit provider requested - get minimal config
            provider_name = provider
            api_key = await credential_service._get_provider_api_key(provider)

            # Check cache for rag_settings
            cache_key = "rag_strategy_settings"
            rag_settings = _get_cached_settings(cache_key)
            if rag_settings is None:
                rag_settings = await credential_service.get_credentials_by_category("rag_strategy")
                _set_cached_settings(cache_key, rag_settings)
                logger.debug("Fetched and cached rag_strategy settings")
            else:
                logger.debug("Using cached rag_strategy settings")

            base_url = credential_service._get_provider_base_url(provider, rag_settings)
        else:
            # Get configured provider from database
            service_type = "embedding" if use_embedding_provider else "llm"

            # Check cache for provider config
            cache_key = f"provider_config_{service_type}"
            provider_config = _get_cached_settings(cache_key)
            if provider_config is None:
                provider_config = await credential_service.get_active_provider(service_type)
                _set_cached_settings(cache_key, provider_config)
                logger.debug(f"Fetched and cached {service_type} provider config")
            else:
                logger.debug(f"Using cached {service_type} provider config")

            provider_name = provider_config["provider"]
            api_key = provider_config["api_key"]
            base_url = provider_config["base_url"]

        logger.info(f"Creating LLM client for provider: {provider_name}")

        if provider_name == "openai":
            if not api_key:
                raise ValueError("OpenAI API key not found")

            client = openai.AsyncOpenAI(api_key=api_key)
            logger.info("OpenAI client created successfully")

        elif provider_name == "ollama":
            # Use load balancing to select the best Ollama instance
            best_instance = await get_best_ollama_instance()
            
            if not best_instance:
                raise ValueError("No Ollama instances available")
            
            # Get the base URL from the selected instance
            instance_base_url = best_instance.get("baseUrl", "http://localhost:11434")
            
            # Ensure base_url has /v1 suffix for OpenAI client compatibility
            if not instance_base_url.endswith("/v1"):
                clean_base_url = f"{instance_base_url}/v1"
            else:
                clean_base_url = instance_base_url
            
            # Ollama requires an API key in the client but doesn't actually use it
            # Add timeout and disable retries for fast failure detection
            client = openai.AsyncOpenAI(
                api_key="ollama",  # Required but unused by Ollama
                base_url=clean_base_url,
                timeout=60.0,  # 1-minute timeout to fail fast on overload
                max_retries=0  # Disable internal retries - handle retries at batch level
            )
            logger.info(f"Ollama client created with load-balanced instance: {best_instance.get('name')} ({clean_base_url})")

        elif provider_name == "google":
            if not api_key:
                raise ValueError("Google API key not found")

            client = openai.AsyncOpenAI(
                api_key=api_key,
                base_url=base_url or "https://generativelanguage.googleapis.com/v1beta/openai/",
            )
            logger.info("Google Gemini client created successfully")

        else:
            raise ValueError(f"Unsupported LLM provider: {provider_name}")

        yield client

    except Exception as e:
        logger.error(
            f"Error creating LLM client for provider {provider_name if 'provider_name' in locals() else 'unknown'}: {e}"
        )
        raise
    finally:
        # Cleanup if needed
        pass


async def get_embedding_model(provider: str | None = None) -> str:
    """
    Get the configured embedding model based on the provider.

    Args:
        provider: Override provider selection

    Returns:
        str: The embedding model to use
    """
    try:
        # Get provider configuration
        if provider:
            # Explicit provider requested
            provider_name = provider
            # Get custom model from settings if any
            cache_key = "rag_strategy_settings"
            rag_settings = _get_cached_settings(cache_key)
            if rag_settings is None:
                rag_settings = await credential_service.get_credentials_by_category("rag_strategy")
                _set_cached_settings(cache_key, rag_settings)
            custom_model = rag_settings.get("EMBEDDING_MODEL", "")
        else:
            # Get configured provider from database
            cache_key = "provider_config_embedding"
            provider_config = _get_cached_settings(cache_key)
            if provider_config is None:
                provider_config = await credential_service.get_active_provider("embedding")
                _set_cached_settings(cache_key, provider_config)
            provider_name = provider_config["provider"]
            custom_model = provider_config["embedding_model"]

        # Use custom model if specified
        if custom_model:
            return custom_model

        # Return provider-specific defaults
        if provider_name == "openai":
            return "text-embedding-3-small"
        elif provider_name == "ollama":
            # Ollama default embedding model
            return "nomic-embed-text"
        elif provider_name == "google":
            # Google's embedding model
            return "text-embedding-004"
        else:
            # Fallback to OpenAI's model
            return "text-embedding-3-small"

    except Exception as e:
        logger.error(f"Error getting embedding model: {e}")
        # Fallback to OpenAI default
        return "text-embedding-3-small"
