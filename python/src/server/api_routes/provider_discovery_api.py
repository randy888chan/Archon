"""
Provider Discovery API endpoints for multi-provider settings UI.

Handles:
- Provider health checks and connectivity status
- Model discovery and specifications 
- Provider configuration validation
- Real-time status monitoring
"""

import asyncio
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import aiohttp
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from ..config.logfire_config import get_logger
from ..services.provider_discovery_service import provider_discovery_service, ModelSpec, ProviderStatus
from ..services.credential_service import credential_service
from .socketio_broadcasts import emit_provider_status_update

logger = get_logger(__name__)

router = APIRouter(prefix="/api/providers", tags=["providers"])

class ProviderHealthResponse(BaseModel):
    """Response for provider health check."""
    provider: str
    is_available: bool
    response_time_ms: Optional[float] = None
    error_message: Optional[str] = None
    models_available: int = 0
    base_url: Optional[str] = None
    last_checked: Optional[datetime] = None

class ModelSpecResponse(BaseModel):
    """Response for model specifications."""
    name: str
    provider: str
    context_window: int
    supports_tools: bool = False
    supports_vision: bool = False
    supports_embeddings: bool = False
    embedding_dimensions: Optional[int] = None
    pricing_input: Optional[float] = None
    pricing_output: Optional[float] = None
    description: str = ""
    aliases: List[str] = []

class ProviderConfigRequest(BaseModel):
    """Request for provider configuration validation."""
    provider: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    additional_config: Dict[str, Any] = {}

class ProviderConfigResponse(BaseModel):
    """Response for provider configuration validation."""
    is_valid: bool
    error_message: Optional[str] = None
    available_models: List[ModelSpecResponse] = []
    health_status: ProviderHealthResponse

class AllProvidersStatusResponse(BaseModel):
    """Response for all providers status."""
    providers: Dict[str, ProviderHealthResponse]
    timestamp: datetime

class OllamaHostsRequest(BaseModel):
    """Request for Ollama model discovery with multiple hosts."""
    hosts: List[str]
    timeout_seconds: Optional[int] = 10

class OllamaModelResponse(BaseModel):
    """Response for Ollama model with host information."""
    name: str
    host: str
    context_window: int
    supports_tools: bool = False
    supports_vision: bool = False
    supports_embeddings: bool = False
    embedding_dimensions: Optional[int] = None
    description: str = ""
    aliases: List[str] = []
    model_type: str  # "chat" or "embedding"
    size_gb: Optional[float] = None
    family: Optional[str] = None

class OllamaModelsDiscoveryResponse(BaseModel):
    """Response for Ollama models discovery."""
    chat_models: List[OllamaModelResponse] = []
    embedding_models: List[OllamaModelResponse] = []
    host_status: Dict[str, Dict[str, Any]] = {}
    total_models: int = 0
    discovery_errors: List[str] = []

# Core endpoints for provider discovery and health checking

@router.get("/status")
async def get_all_providers_status() -> AllProvidersStatusResponse:
    """Get health status for all configured providers."""
    try:
        logger.info("Getting status for all providers")
        
        # Get current provider configurations
        rag_settings = await credential_service.get_credentials_by_category("rag_strategy")
        
        providers_status = {}
        
        # Check OpenAI
        openai_key = await credential_service.get_credential("OPENAI_API_KEY")
        if openai_key:
            status = await provider_discovery_service.check_provider_health(
                "openai", {"api_key": openai_key}
            )
            providers_status["openai"] = _convert_provider_status(status)
        else:
            providers_status["openai"] = ProviderHealthResponse(
                provider="openai",
                is_available=False,
                error_message="API key not configured"
            )
        
        # Check Google
        google_key = await credential_service.get_credential("GOOGLE_API_KEY")
        if google_key:
            status = await provider_discovery_service.check_provider_health(
                "google", {"api_key": google_key}
            )
            providers_status["google"] = _convert_provider_status(status)
        else:
            providers_status["google"] = ProviderHealthResponse(
                provider="google",
                is_available=False,
                error_message="API key not configured"
            )
            
        # Check Ollama
        ollama_url = rag_settings.get("LLM_BASE_URL", "http://localhost:11434")
        status = await provider_discovery_service.check_provider_health(
            "ollama", {"base_url": ollama_url}
        )
        providers_status["ollama"] = _convert_provider_status(status)
        
        # Check Anthropic
        anthropic_key = await credential_service.get_credential("ANTHROPIC_API_KEY")
        if anthropic_key:
            status = await provider_discovery_service.check_provider_health(
                "anthropic", {"api_key": anthropic_key}
            )
            providers_status["anthropic"] = _convert_provider_status(status)
        else:
            providers_status["anthropic"] = ProviderHealthResponse(
                provider="anthropic",
                is_available=False,
                error_message="API key not configured"
            )
        
        logger.info(f"Retrieved status for {len(providers_status)} providers")
        
        return AllProvidersStatusResponse(
            providers=providers_status,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error getting all providers status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{provider}/status")
async def get_provider_status(provider: str, background_tasks: BackgroundTasks) -> ProviderHealthResponse:
    """Get health status for a specific provider."""
    try:
        logger.info(f"Getting status for provider: {provider}")
        
        config = await _get_provider_config(provider)
        status = await provider_discovery_service.check_provider_health(provider, config)
        
        response = _convert_provider_status(status)
        
        # Emit real-time update via Socket.IO
        background_tasks.add_task(
            emit_provider_status_update,
            provider,
            response.dict()
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting provider status for {provider}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{provider}/models")
async def get_provider_models(provider: str) -> List[ModelSpecResponse]:
    """Get available models for a specific provider."""
    try:
        logger.info(f"Getting models for provider: {provider}")
        
        config = await _get_provider_config(provider)
        models = []
        
        if provider == "openai":
            api_key = config.get("api_key")
            if api_key:
                models = await provider_discovery_service.discover_openai_models(api_key)
                
        elif provider == "google":
            api_key = config.get("api_key")
            if api_key:
                models = await provider_discovery_service.discover_google_models(api_key)
                
        elif provider == "ollama":
            base_urls = [config.get("base_url", "http://localhost:11434")]
            models = await provider_discovery_service.discover_ollama_models(base_urls)
            
        elif provider == "anthropic":
            api_key = config.get("api_key")
            if api_key:
                models = await provider_discovery_service.discover_anthropic_models(api_key)
                
        else:
            raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")
        
        response = [_convert_model_spec(model) for model in models]
        logger.info(f"Retrieved {len(response)} models for provider: {provider}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting models for provider {provider}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/{model_id}/specs")
async def get_model_specs(model_id: str, provider: Optional[str] = None) -> ModelSpecResponse:
    """Get detailed specifications for a specific model."""
    try:
        logger.info(f"Getting specs for model: {model_id}, provider: {provider}")
        
        # If provider is specified, search only that provider
        if provider:
            providers_to_search = [provider]
        else:
            # Search all providers to find the model
            providers_to_search = ["openai", "google", "ollama", "anthropic"]
        
        for prov in providers_to_search:
            try:
                models = await get_provider_models(prov)
                for model in models:
                    if model.name == model_id or model_id in model.aliases:
                        return model
            except Exception as e:
                logger.warning(f"Error searching provider {prov} for model {model_id}: {e}")
                continue
        
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model specs for {model_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/validate")
async def validate_provider_config(request: ProviderConfigRequest) -> ProviderConfigResponse:
    """Validate a provider configuration."""
    try:
        logger.info(f"Validating configuration for provider: {request.provider}")
        
        config = {
            "api_key": request.api_key,
            "base_url": request.base_url,
            **request.additional_config
        }
        
        # Check health status
        health_status = await provider_discovery_service.check_provider_health(
            request.provider, config
        )
        
        # Get available models if provider is healthy
        available_models = []
        if health_status.is_available:
            try:
                models = []
                if request.provider == "openai" and request.api_key:
                    models = await provider_discovery_service.discover_openai_models(request.api_key)
                elif request.provider == "google" and request.api_key:
                    models = await provider_discovery_service.discover_google_models(request.api_key)
                elif request.provider == "ollama":
                    base_url = request.base_url or "http://localhost:11434"
                    models = await provider_discovery_service.discover_ollama_models([base_url])
                elif request.provider == "anthropic" and request.api_key:
                    models = await provider_discovery_service.discover_anthropic_models(request.api_key)
                
                available_models = [_convert_model_spec(model) for model in models]
                
            except Exception as e:
                logger.warning(f"Error discovering models during validation: {e}")
        
        return ProviderConfigResponse(
            is_valid=health_status.is_available,
            error_message=health_status.error_message,
            available_models=available_models,
            health_status=_convert_provider_status(health_status)
        )
        
    except Exception as e:
        logger.error(f"Error validating provider config: {e}")
        return ProviderConfigResponse(
            is_valid=False,
            error_message=str(e),
            available_models=[],
            health_status=ProviderHealthResponse(
                provider=request.provider,
                is_available=False,
                error_message=str(e)
            )
        )

@router.get("/models")
async def get_all_available_models() -> Dict[str, List[ModelSpecResponse]]:
    """Get all available models from all configured providers."""
    try:
        logger.info("Getting all available models from all providers")
        
        all_models = await provider_discovery_service.get_all_available_models()
        
        # Convert to response format
        response = {}
        for provider, models in all_models.items():
            response[provider] = [_convert_model_spec(model) for model in models]
        
        total_models = sum(len(models) for models in response.values())
        logger.info(f"Retrieved {total_models} models from {len(response)} providers")
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting all available models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ollama/models")
async def discover_ollama_models_endpoint(request: OllamaHostsRequest) -> OllamaModelsDiscoveryResponse:
    """
    Discover available models from all configured Ollama hosts.
    
    This endpoint queries the /api/tags endpoint for each enabled host to get available models,
    categorizes them into chat models and embedding models based on model names/capabilities,
    and returns structured data with model details including source host, model size, and capabilities.
    
    Connection failures are handled gracefully and return partial results.
    """
    try:
        logger.info(f"Discovering models from {len(request.hosts)} Ollama hosts")
        
        chat_models = []
        embedding_models = []
        host_status = {}
        discovery_errors = []
        
        # Process each host with timeout handling
        async def process_host(host_url: str) -> None:
            try:
                # Clean up URL - remove /v1 suffix if present for raw Ollama API
                parsed = urlparse(host_url)
                if parsed.path.endswith('/v1'):
                    api_url = host_url.replace('/v1', '')
                else:
                    api_url = host_url
                
                # Ensure proper format
                if not api_url.startswith(('http://', 'https://')):
                    api_url = f"http://{api_url}"
                
                session = await provider_discovery_service._get_session()
                
                # Test connectivity and get models
                start_time = time.time()
                timeout = aiohttp.ClientTimeout(total=request.timeout_seconds)
                
                async with session.get(f"{api_url}/api/tags", timeout=timeout) as response:
                    response_time = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        data = await response.json()
                        models_data = data.get("models", [])
                        
                        host_status[host_url] = {
                            "status": "online",
                            "response_time_ms": response_time,
                            "models_count": len(models_data),
                            "api_url": api_url
                        }
                        
                        for model_info in models_data:
                            model_name = model_info.get("name", "")
                            base_name = model_name.split(':')[0]  # Remove tag
                            
                            # Determine model family and capabilities
                            family = _get_model_family(base_name)
                            supports_tools = _supports_tools(base_name)
                            supports_vision = _supports_vision(base_name)
                            supports_embeddings = _supports_embeddings(base_name)
                            
                            # Estimate context window based on model family
                            context_window = _get_context_window(base_name)
                            
                            # Set embedding dimensions for known embedding models
                            embedding_dims = _get_embedding_dimensions(base_name)
                            
                            # Estimate model size (this could be enhanced with actual /api/show calls)
                            size_gb = _estimate_model_size(model_info)
                            
                            # Create model response
                            ollama_model = OllamaModelResponse(
                                name=model_name,
                                host=host_url,
                                context_window=context_window,
                                supports_tools=supports_tools,
                                supports_vision=supports_vision,
                                supports_embeddings=supports_embeddings,
                                embedding_dimensions=embedding_dims,
                                description=f"{family} model on {host_url}",
                                aliases=[base_name] if ':' in model_name else [],
                                model_type="embedding" if supports_embeddings else "chat",
                                size_gb=size_gb,
                                family=family
                            )
                            
                            # Categorize models
                            if supports_embeddings:
                                embedding_models.append(ollama_model)
                            else:
                                chat_models.append(ollama_model)
                    else:
                        error_msg = f"Host {host_url} returned HTTP {response.status}"
                        host_status[host_url] = {
                            "status": "error",
                            "response_time_ms": response_time,
                            "error": error_msg,
                            "api_url": api_url
                        }
                        discovery_errors.append(error_msg)
                        
            except asyncio.TimeoutError:
                error_msg = f"Timeout connecting to {host_url} after {request.timeout_seconds}s"
                host_status[host_url] = {
                    "status": "timeout",
                    "error": error_msg,
                    "api_url": api_url if 'api_url' in locals() else host_url
                }
                discovery_errors.append(error_msg)
                logger.warning(error_msg)
                
            except Exception as e:
                error_msg = f"Error connecting to {host_url}: {str(e)}"
                host_status[host_url] = {
                    "status": "error",
                    "error": error_msg,
                    "api_url": api_url if 'api_url' in locals() else host_url
                }
                discovery_errors.append(error_msg)
                logger.error(error_msg)
        
        # Process all hosts concurrently with individual error handling
        await asyncio.gather(*[process_host(host) for host in request.hosts], return_exceptions=True)
        
        total_models = len(chat_models) + len(embedding_models)
        
        logger.info(f"Discovery complete: {len(chat_models)} chat models, {len(embedding_models)} embedding models from {len(request.hosts)} hosts")
        
        return OllamaModelsDiscoveryResponse(
            chat_models=chat_models,
            embedding_models=embedding_models,
            host_status=host_status,
            total_models=total_models,
            discovery_errors=discovery_errors
        )
        
    except Exception as e:
        logger.error(f"Error in Ollama models discovery: {e}")
        raise HTTPException(status_code=500, detail=f"Discovery failed: {str(e)}")

# Helper functions

async def _get_provider_config(provider: str) -> Dict[str, Any]:
    """Get configuration for a specific provider."""
    config = {}
    
    if provider == "openai":
        api_key = await credential_service.get_credential("OPENAI_API_KEY")
        config = {"api_key": api_key}
        
    elif provider == "google":
        api_key = await credential_service.get_credential("GOOGLE_API_KEY")
        config = {"api_key": api_key}
        
    elif provider == "ollama":
        rag_settings = await credential_service.get_credentials_by_category("rag_strategy")
        base_url = rag_settings.get("LLM_BASE_URL", "http://localhost:11434")
        config = {"base_url": base_url}
        
    elif provider == "anthropic":
        api_key = await credential_service.get_credential("ANTHROPIC_API_KEY")
        config = {"api_key": api_key}
        
    return config

def _convert_provider_status(status: ProviderStatus) -> ProviderHealthResponse:
    """Convert internal ProviderStatus to API response."""
    return ProviderHealthResponse(
        provider=status.provider,
        is_available=status.is_available,
        response_time_ms=status.response_time_ms,
        error_message=status.error_message,
        models_available=status.models_available,
        base_url=status.base_url,
        last_checked=datetime.fromtimestamp(status.last_checked) if status.last_checked else None
    )

def _convert_model_spec(spec: ModelSpec) -> ModelSpecResponse:
    """Convert internal ModelSpec to API response."""
    return ModelSpecResponse(
        name=spec.name,
        provider=spec.provider,
        context_window=spec.context_window,
        supports_tools=spec.supports_tools,
        supports_vision=spec.supports_vision,
        supports_embeddings=spec.supports_embeddings,
        embedding_dimensions=spec.embedding_dimensions,
        pricing_input=spec.pricing_input,
        pricing_output=spec.pricing_output,
        description=spec.description,
        aliases=spec.aliases
    )

# Helper functions for Ollama model categorization and capability detection

def _get_model_family(model_name: str) -> str:
    """Determine model family from name."""
    name_lower = model_name.lower()
    
    if "llama" in name_lower:
        if "code" in name_lower:
            return "CodeLlama"
        elif "llama3" in name_lower or "llama-3" in name_lower:
            return "Llama 3"
        else:
            return "Llama"
    elif "mistral" in name_lower:
        return "Mistral"
    elif "qwen" in name_lower:
        return "Qwen"
    elif "gemma" in name_lower:
        return "Gemma"
    elif "phi" in name_lower:
        return "Phi"
    elif "nomic" in name_lower:
        return "Nomic Embed"
    elif "mxbai" in name_lower:
        return "MxBai Embed"
    elif "embed" in name_lower:
        return "Embedding Model"
    else:
        return "Unknown"

def _supports_tools(model_name: str) -> bool:
    """Check if model supports function calling/tools."""
    name_lower = model_name.lower()
    tool_patterns = ["llama3", "qwen", "mistral", "gemma", "phi3"]
    return any(pattern in name_lower for pattern in tool_patterns)

def _supports_vision(model_name: str) -> bool:
    """Check if model supports vision/image processing."""
    name_lower = model_name.lower()
    vision_patterns = ["vision", "llava", "moondream", "bakllava"]
    return any(pattern in name_lower for pattern in vision_patterns)

def _supports_embeddings(model_name: str) -> bool:
    """Check if model is an embedding model."""
    name_lower = model_name.lower()
    embedding_patterns = ["embed", "embedding", "nomic-embed", "mxbai-embed", "bge-", "e5-"]
    return any(pattern in name_lower for pattern in embedding_patterns)

def _get_context_window(model_name: str) -> int:
    """Estimate context window based on model family."""
    name_lower = model_name.lower()
    
    if "llama3" in name_lower or "llama-3" in name_lower:
        return 8192
    elif "qwen" in name_lower:
        if "72b" in name_lower or "110b" in name_lower:
            return 32768
        else:
            return 8192
    elif "mistral" in name_lower:
        if "large" in name_lower:
            return 32768
        else:
            return 32768
    elif "gemma" in name_lower:
        return 8192
    elif "phi" in name_lower:
        return 4096
    elif "embed" in name_lower:
        return 512  # Embedding models typically have smaller context
    else:
        return 4096  # Default fallback

def _get_embedding_dimensions(model_name: str) -> Optional[int]:
    """Get embedding dimensions for known embedding models."""
    name_lower = model_name.lower()
    
    if "nomic-embed" in name_lower:
        return 768
    elif "mxbai-embed" in name_lower:
        if "large" in name_lower:
            return 1024
        else:
            return 384
    elif "bge-small" in name_lower:
        return 384
    elif "bge-base" in name_lower:
        return 768
    elif "bge-large" in name_lower:
        return 1024
    elif "e5-small" in name_lower:
        return 384
    elif "e5-base" in name_lower:
        return 768
    elif "e5-large" in name_lower:
        return 1024
    elif "embed" in name_lower:
        return 768  # Default for generic embedding models
    else:
        return None

def _estimate_model_size(model_info: Dict[str, Any]) -> Optional[float]:
    """Estimate model size in GB from model info."""
    # This is a rough estimation - for accurate sizes, would need to call /api/show
    # Size information might be in the model info if available
    size = model_info.get("size")
    if size:
        # Convert bytes to GB if size is provided
        if isinstance(size, (int, float)):
            return round(size / (1024**3), 1)
    
    # Fallback estimation based on model name patterns
    name = model_info.get("name", "").lower()
    
    if "7b" in name:
        return 4.1
    elif "13b" in name:
        return 7.3
    elif "30b" in name or "33b" in name:
        return 19.0
    elif "65b" in name or "70b" in name:
        return 39.0
    elif "180b" in name:
        return 101.0
    elif "embed" in name:
        return 0.5  # Embedding models are typically smaller
    else:
        return None