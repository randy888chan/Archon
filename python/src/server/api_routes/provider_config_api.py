"""
Provider Configuration API endpoints for multi-provider settings management.

Extends existing settings capabilities with multi-provider configuration support:
- Manages multiple Ollama instances with load balancing
- Handles provider-specific settings and credentials 
- Supports configuration validation and migration
- Provides compatibility checking and recommendations
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from ..config.logfire_config import get_logger
from ..services.credential_service import credential_service
from ..services.provider_discovery_service import provider_discovery_service
from .socketio_broadcasts import emit_provider_status_update

logger = get_logger(__name__)

router = APIRouter(prefix="/api/provider-config", tags=["provider-config"])

class OllamaInstanceConfig(BaseModel):
    """Configuration for a single Ollama instance."""
    id: str
    name: str
    base_url: str
    is_primary: bool = False
    is_enabled: bool = True
    load_balancing_weight: int = 1
    health_check_enabled: bool = True

class MultiProviderConfig(BaseModel):
    """Configuration for multiple providers and instances."""
    llm_provider: str  # Primary LLM provider
    embedding_provider: str  # Primary embedding provider
    openai_config: Dict[str, Any] = {}
    google_config: Dict[str, Any] = {}
    anthropic_config: Dict[str, Any] = {}
    ollama_instances: List[OllamaInstanceConfig] = []
    provider_preferences: Dict[str, Any] = {}

class ConfigValidationRequest(BaseModel):
    """Request for configuration validation."""
    config: MultiProviderConfig

class ConfigValidationResponse(BaseModel):
    """Response for configuration validation."""
    is_valid: bool
    errors: List[str] = []
    warnings: List[str] = []
    recommendations: List[str] = []
    compatibility_issues: List[str] = []

class ConfigMigrationStatus(BaseModel):
    """Status of configuration migration."""
    needs_migration: bool
    migration_type: str  # "provider_change", "ollama_distributed", "credential_update"
    current_config: Dict[str, Any]
    target_config: Dict[str, Any]
    migration_steps: List[str] = []
    data_loss_risk: bool = False

# Multi-provider configuration endpoints

@router.get("/current")
async def get_current_provider_config() -> MultiProviderConfig:
    """Get current multi-provider configuration."""
    try:
        logger.info("Getting current multi-provider configuration")
        
        # Get current provider settings
        rag_settings = await credential_service.get_credentials_by_category("rag_strategy")
        
        # Get provider configurations
        llm_provider = rag_settings.get("LLM_PROVIDER", "openai")
        embedding_provider = rag_settings.get("LLM_PROVIDER", "openai")  # For now, use same provider
        
        # Get Ollama instances - for now, create from single URL
        ollama_base_url = rag_settings.get("LLM_BASE_URL", "http://localhost:11434")
        ollama_instances = [
            OllamaInstanceConfig(
                id="default",
                name="Default Ollama",
                base_url=ollama_base_url,
                is_primary=True,
                is_enabled=True,
                load_balancing_weight=1
            )
        ]
        
        config = MultiProviderConfig(
            llm_provider=llm_provider,
            embedding_provider=embedding_provider,
            ollama_instances=ollama_instances,
            provider_preferences=rag_settings
        )
        
        logger.info(f"Retrieved multi-provider config with {len(ollama_instances)} Ollama instances")
        return config
        
    except Exception as e:
        logger.error(f"Error getting current provider config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/update")
async def update_provider_config(
    request: MultiProviderConfig, 
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """Update multi-provider configuration."""
    try:
        logger.info(f"Updating provider config for LLM: {request.llm_provider}, Embedding: {request.embedding_provider}")
        
        # Update primary provider settings
        await credential_service.set_credential(
            "LLM_PROVIDER",
            request.llm_provider,
            category="rag_strategy",
            description="Primary LLM provider"
        )
        
        # Update Ollama instances configuration
        if request.ollama_instances:
            primary_instance = next(
                (inst for inst in request.ollama_instances if inst.is_primary), 
                request.ollama_instances[0]
            )
            
            await credential_service.set_credential(
                "LLM_BASE_URL",
                primary_instance.base_url,
                category="rag_strategy",
                description="Primary Ollama base URL"
            )
            
            # Store multi-instance config for future use
            # For now, we'll store this as a JSON string in credentials
            import json
            ollama_config = [inst.dict() for inst in request.ollama_instances]
            await credential_service.set_credential(
                "OLLAMA_INSTANCES",
                json.dumps(ollama_config),
                category="rag_strategy",
                description="Multi-instance Ollama configuration"
            )
        
        # Update provider preferences
        for key, value in request.provider_preferences.items():
            if key not in ["LLM_PROVIDER", "LLM_BASE_URL"]:  # Already handled above
                await credential_service.set_credential(
                    key,
                    value,
                    category="rag_strategy",
                    description=f"Provider preference: {key}"
                )
        
        # Emit real-time update
        background_tasks.add_task(
            emit_provider_status_update,
            "config_updated",
            {
                "llm_provider": request.llm_provider,
                "embedding_provider": request.embedding_provider,
                "ollama_instances_count": len(request.ollama_instances)
            }
        )
        
        logger.info("Multi-provider configuration updated successfully")
        
        return {
            "success": True,
            "message": "Provider configuration updated successfully",
            "llm_provider": request.llm_provider,
            "embedding_provider": request.embedding_provider,
            "ollama_instances_count": len(request.ollama_instances)
        }
        
    except Exception as e:
        logger.error(f"Error updating provider config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/validate")
async def validate_provider_config(request: ConfigValidationRequest) -> ConfigValidationResponse:
    """Validate a multi-provider configuration."""
    try:
        logger.info("Validating multi-provider configuration")
        
        errors = []
        warnings = []
        recommendations = []
        compatibility_issues = []
        
        config = request.config
        
        # Validate primary providers
        if config.llm_provider not in ["openai", "google", "ollama", "anthropic"]:
            errors.append(f"Unknown LLM provider: {config.llm_provider}")
            
        if config.embedding_provider not in ["openai", "google", "ollama"]:
            errors.append(f"Unknown embedding provider: {config.embedding_provider}")
        
        # Validate Ollama instances
        if config.llm_provider == "ollama" or config.embedding_provider == "ollama":
            if not config.ollama_instances:
                errors.append("Ollama provider selected but no instances configured")
            else:
                primary_count = sum(1 for inst in config.ollama_instances if inst.is_primary)
                if primary_count == 0:
                    warnings.append("No primary Ollama instance set - will use first instance")
                elif primary_count > 1:
                    errors.append("Multiple primary Ollama instances configured")
                
                # Check for duplicate URLs
                urls = [inst.base_url for inst in config.ollama_instances]
                if len(urls) != len(set(urls)):
                    errors.append("Duplicate Ollama instance URLs detected")
        
        # Check provider credentials
        if config.llm_provider == "openai":
            openai_key = await credential_service.get_credential("OPENAI_API_KEY")
            if not openai_key:
                errors.append("OpenAI provider selected but API key not configured")
                
        if config.llm_provider == "google":
            google_key = await credential_service.get_credential("GOOGLE_API_KEY")
            if not google_key:
                errors.append("Google provider selected but API key not configured")
                
        if config.llm_provider == "anthropic":
            anthropic_key = await credential_service.get_credential("ANTHROPIC_API_KEY")
            if not anthropic_key:
                errors.append("Anthropic provider selected but API key not configured")
        
        # Generate recommendations
        if len(config.ollama_instances) > 1:
            recommendations.append("Consider enabling load balancing for multiple Ollama instances")
            
        if config.llm_provider != config.embedding_provider:
            recommendations.append("Using different providers for LLM and embeddings may impact performance")
        
        # Check compatibility
        if config.llm_provider == "ollama" and config.embedding_provider == "openai":
            compatibility_issues.append("Mixed Ollama/OpenAI setup may require careful model selection")
        
        is_valid = len(errors) == 0
        
        return ConfigValidationResponse(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            recommendations=recommendations,
            compatibility_issues=compatibility_issues
        )
        
    except Exception as e:
        logger.error(f"Error validating provider config: {e}")
        return ConfigValidationResponse(
            is_valid=False,
            errors=[str(e)]
        )

@router.get("/migration-status")
async def get_migration_status() -> ConfigMigrationStatus:
    """Check if configuration migration is needed."""
    try:
        logger.info("Checking configuration migration status")
        
        # Get current configuration
        current_config = await get_current_provider_config()
        
        # Check if we have the new multi-instance format
        ollama_instances_raw = await credential_service.get_credential("OLLAMA_INSTANCES")
        
        needs_migration = False
        migration_type = "none"
        migration_steps = []
        
        if not ollama_instances_raw and current_config.ollama_instances:
            # Need to migrate from single instance to multi-instance format
            needs_migration = True
            migration_type = "ollama_distributed"
            migration_steps = [
                "Convert single Ollama URL to multi-instance configuration",
                "Set up load balancing configuration",
                "Preserve existing model selections"
            ]
        
        return ConfigMigrationStatus(
            needs_migration=needs_migration,
            migration_type=migration_type,
            current_config=current_config.dict(),
            target_config=current_config.dict(),  # For now, same as current
            migration_steps=migration_steps,
            data_loss_risk=False
        )
        
    except Exception as e:
        logger.error(f"Error checking migration status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/migrate")
async def migrate_configuration(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Perform configuration migration if needed."""
    try:
        logger.info("Starting configuration migration")
        
        migration_status = await get_migration_status()
        
        if not migration_status.needs_migration:
            return {
                "success": True,
                "message": "No migration needed",
                "migration_type": "none"
            }
        
        if migration_status.migration_type == "ollama_distributed":
            # Migrate to multi-instance Ollama configuration
            current_config = await get_current_provider_config()
            
            # The migration is essentially saving the current single-instance 
            # configuration in the new multi-instance format
            await update_provider_config(current_config, background_tasks)
            
            logger.info("Configuration migration completed successfully")
            
            return {
                "success": True,
                "message": "Configuration migrated to multi-instance format",
                "migration_type": migration_status.migration_type
            }
        
        return {
            "success": False,
            "message": f"Unknown migration type: {migration_status.migration_type}"
        }
        
    except Exception as e:
        logger.error(f"Error during configuration migration: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Ollama-specific endpoints for distributed processing

@router.post("/ollama/add-instance")
async def add_ollama_instance(
    instance: OllamaInstanceConfig,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """Add a new Ollama instance to the configuration."""
    try:
        logger.info(f"Adding Ollama instance: {instance.name} at {instance.base_url}")
        
        # Get current configuration
        current_config = await get_current_provider_config()
        
        # Check if instance already exists
        existing_ids = [inst.id for inst in current_config.ollama_instances]
        if instance.id in existing_ids:
            raise HTTPException(status_code=400, detail=f"Instance with ID {instance.id} already exists")
        
        existing_urls = [inst.base_url for inst in current_config.ollama_instances]
        if instance.base_url in existing_urls:
            raise HTTPException(status_code=400, detail=f"Instance with URL {instance.base_url} already exists")
        
        # Add the new instance
        current_config.ollama_instances.append(instance)
        
        # Update configuration
        await update_provider_config(current_config, background_tasks)
        
        # Test connectivity to new instance
        health_status = await provider_discovery_service.check_provider_health(
            "ollama", {"base_url": instance.base_url}
        )
        
        return {
            "success": True,
            "message": f"Ollama instance {instance.name} added successfully",
            "instance_id": instance.id,
            "health_status": health_status.is_available
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding Ollama instance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/ollama/remove-instance/{instance_id}")
async def remove_ollama_instance(
    instance_id: str,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """Remove an Ollama instance from the configuration."""
    try:
        logger.info(f"Removing Ollama instance: {instance_id}")
        
        # Get current configuration
        current_config = await get_current_provider_config()
        
        # Find and remove the instance
        instance_to_remove = None
        for i, inst in enumerate(current_config.ollama_instances):
            if inst.id == instance_id:
                instance_to_remove = current_config.ollama_instances.pop(i)
                break
        
        if not instance_to_remove:
            raise HTTPException(status_code=404, detail=f"Instance with ID {instance_id} not found")
        
        # Ensure at least one instance remains if Ollama is the active provider
        if (current_config.llm_provider == "ollama" or current_config.embedding_provider == "ollama"):
            if not current_config.ollama_instances:
                raise HTTPException(
                    status_code=400, 
                    detail="Cannot remove the last Ollama instance while Ollama provider is active"
                )
        
        # If removing primary instance, promote another
        if instance_to_remove.is_primary and current_config.ollama_instances:
            current_config.ollama_instances[0].is_primary = True
        
        # Update configuration
        await update_provider_config(current_config, background_tasks)
        
        return {
            "success": True,
            "message": f"Ollama instance {instance_to_remove.name} removed successfully",
            "removed_instance_id": instance_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing Ollama instance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ollama/load-balancing-status")
async def get_ollama_load_balancing_status() -> Dict[str, Any]:
    """Get current load balancing status for Ollama instances."""
    try:
        logger.info("Getting Ollama load balancing status")
        
        current_config = await get_current_provider_config()
        ollama_instances = current_config.ollama_instances
        
        if not ollama_instances:
            return {
                "enabled": False,
                "instances": [],
                "total_weight": 0
            }
        
        # Check health of all instances
        instance_status = []
        for instance in ollama_instances:
            health = await provider_discovery_service.check_provider_health(
                "ollama", {"base_url": instance.base_url}
            )
            
            instance_status.append({
                "id": instance.id,
                "name": instance.name,
                "base_url": instance.base_url,
                "is_enabled": instance.is_enabled,
                "is_primary": instance.is_primary,
                "weight": instance.load_balancing_weight,
                "is_healthy": health.is_available,
                "response_time_ms": health.response_time_ms,
                "models_available": health.models_available
            })
        
        total_weight = sum(inst.load_balancing_weight for inst in ollama_instances if inst.is_enabled)
        enabled_count = sum(1 for inst in ollama_instances if inst.is_enabled)
        
        return {
            "enabled": len(ollama_instances) > 1,
            "instances": instance_status,
            "total_weight": total_weight,
            "enabled_instances": enabled_count,
            "load_balancing_active": enabled_count > 1
        }
        
    except Exception as e:
        logger.error(f"Error getting load balancing status: {e}")
        raise HTTPException(status_code=500, detail=str(e))