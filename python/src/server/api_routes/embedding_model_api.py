"""
Embedding Model API endpoints for managing embedding models and transitions.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
from pydantic import BaseModel

from ..config.logfire_config import get_logger
from ..services.embeddings.embedding_dimension_service import embedding_dimension_service, RECOMMENDED_MODELS
from ..services.credential_service import credential_service

logger = get_logger(__name__)

router = APIRouter(prefix="/api/embedding-models", tags=["embedding-models"])


class ModelValidationRequest(BaseModel):
    provider: str
    model_name: str


class ModelValidationResponse(BaseModel):
    is_valid: bool
    is_change: bool
    dimensions_change: bool
    requires_migration: bool
    data_loss_warning: bool
    current: Dict[str, Any]
    new: Dict[str, Any]
    error: str = None


class CurrentModelInfo(BaseModel):
    provider: str
    model_name: str
    dimensions: int
    schema_dimensions: Dict[str, int]
    embedding_counts: Dict[str, int]
    total_embeddings: int
    schema_needs_migration: bool


@router.get("/recommendations")
async def get_model_recommendations() -> List[Dict[str, Any]]:
    """Get all recommended embedding models."""
    try:
        return embedding_dimension_service.get_recommended_models()
    except Exception as e:
        logger.error(f"Failed to get model recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/current")
async def get_current_model_info() -> CurrentModelInfo:
    """Get information about the current embedding model."""
    try:
        # Get current provider configuration
        provider_config = await credential_service.get_active_provider("embedding")
        provider = provider_config.get("provider", "openai")
        
        # Get current embedding model
        from ..services.llm_provider_service import get_embedding_model
        current_model = await get_embedding_model()
        
        # Detect current dimensions
        dimensions = await embedding_dimension_service.detect_model_dimensions(current_model, provider)
        
        # For now, we'll return basic info - in a full implementation you'd query the database
        # to get actual schema dimensions and embedding counts
        return CurrentModelInfo(
            provider=provider,
            model_name=current_model,
            dimensions=dimensions,
            schema_dimensions={"documents": dimensions},  # Placeholder
            embedding_counts={"documents": 0},  # Placeholder - would query actual count
            total_embeddings=0,  # Placeholder - would query actual count
            schema_needs_migration=False  # Placeholder - would check if schema matches
        )
    except Exception as e:
        logger.error(f"Failed to get current model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate")
async def validate_model_change(request: ModelValidationRequest) -> ModelValidationResponse:
    """Validate a potential embedding model change."""
    try:
        # Get current model info
        current_info = await get_current_model_info()
        
        # Get new model dimensions
        new_dimensions = await embedding_dimension_service.detect_model_dimensions(
            request.model_name, request.provider
        )
        
        # Check if this is actually a change
        is_change = (
            current_info.provider != request.provider or 
            current_info.model_name != request.model_name
        )
        
        # Check if dimensions change
        dimensions_change = current_info.dimensions != new_dimensions
        
        # Determine if migration is required (dimensions change and embeddings exist)
        requires_migration = dimensions_change and current_info.total_embeddings > 0
        
        # Data loss warning for dimension changes with existing data
        data_loss_warning = requires_migration
        
        return ModelValidationResponse(
            is_valid=True,
            is_change=is_change,
            dimensions_change=dimensions_change,
            requires_migration=requires_migration,
            data_loss_warning=data_loss_warning,
            current={
                "provider": current_info.provider,
                "model": current_info.model_name,
                "dimensions": current_info.dimensions
            },
            new={
                "provider": request.provider,
                "model": request.model_name,
                "dimensions": new_dimensions
            }
        )
    except Exception as e:
        logger.error(f"Failed to validate model change: {e}")
        return ModelValidationResponse(
            is_valid=False,
            is_change=False,
            dimensions_change=False,
            requires_migration=False,
            data_loss_warning=False,
            current={},
            new={},
            error=str(e)
        )


@router.post("/change")
async def change_embedding_model(request: ModelValidationRequest) -> Dict[str, Any]:
    """Change the embedding model (this would trigger data migration in full implementation)."""
    try:
        # In a full implementation, this would:
        # 1. Validate the change
        # 2. Backup existing embeddings if needed
        # 3. Update the model configuration
        # 4. Trigger re-embedding of existing content if dimensions changed
        # 5. Update database schema if needed
        
        # For now, we'll just update the configuration
        await credential_service.set_credential(
            "EMBEDDING_MODEL", 
            request.model_name,
            category="rag_strategy",
            description=f"Embedding model for {request.provider}"
        )
        
        await credential_service.set_credential(
            "LLM_PROVIDER",
            request.provider, 
            category="rag_strategy",
            description="LLM provider for embeddings"
        )
        
        logger.info(f"Changed embedding model to {request.model_name} with provider {request.provider}")
        
        return {
            "success": True,
            "message": f"Successfully changed embedding model to {request.model_name}",
            "provider": request.provider,
            "model": request.model_name
        }
        
    except Exception as e:
        logger.error(f"Failed to change embedding model: {e}")
        raise HTTPException(status_code=500, detail=str(e))