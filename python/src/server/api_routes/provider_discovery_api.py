"""
Provider Discovery API endpoints for multi-provider settings UI.

Handles:
- Provider health checks and connectivity status
- Model discovery and specifications 
- Provider configuration validation
- Real-time status monitoring
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

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