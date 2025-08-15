"""
Provider Validation Service

Provides comprehensive validation for provider configurations, model compatibility,
and settings migration. Supports validation for multi-provider setups and distributed
processing configurations.
"""

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..config.logfire_config import get_logger
from .credential_service import credential_service
from .provider_discovery_service import provider_discovery_service, ModelSpec

logger = get_logger(__name__)

@dataclass
class ValidationResult:
    """Result of a validation check."""
    is_valid: bool
    severity: str  # "error", "warning", "info"
    message: str
    details: Optional[str] = None
    suggested_action: Optional[str] = None

@dataclass
class CompatibilityCheck:
    """Compatibility check result."""
    models_compatible: bool
    embedding_dimensions_match: bool
    tool_support_available: bool
    performance_impact: str  # "none", "minor", "moderate", "major"
    compatibility_score: float  # 0.0 to 1.0
    recommendations: List[str]

class ProviderValidationService:
    """Service for validating provider configurations and model compatibility."""

    def __init__(self):
        pass

    async def validate_provider_credentials(self, provider: str) -> List[ValidationResult]:
        """Validate credentials for a specific provider."""
        results = []
        
        try:
            if provider == "openai":
                api_key = await credential_service.get_credential("OPENAI_API_KEY")
                if not api_key:
                    results.append(ValidationResult(
                        is_valid=False,
                        severity="error",
                        message="OpenAI API key not configured",
                        suggested_action="Add your OpenAI API key in the credentials section"
                    ))
                else:
                    # Test the API key
                    try:
                        models = await provider_discovery_service.discover_openai_models(api_key)
                        if models:
                            results.append(ValidationResult(
                                is_valid=True,
                                severity="info",
                                message=f"OpenAI API key valid - {len(models)} models available"
                            ))
                        else:
                            results.append(ValidationResult(
                                is_valid=False,
                                severity="warning",
                                message="OpenAI API key configured but no models accessible"
                            ))
                    except Exception as e:
                        results.append(ValidationResult(
                            is_valid=False,
                            severity="error",
                            message="OpenAI API key validation failed",
                            details=str(e),
                            suggested_action="Check your API key and internet connection"
                        ))
                        
            elif provider == "google":
                api_key = await credential_service.get_credential("GOOGLE_API_KEY")
                if not api_key:
                    results.append(ValidationResult(
                        is_valid=False,
                        severity="error",
                        message="Google API key not configured",
                        suggested_action="Add your Google API key in the credentials section"
                    ))
                else:
                    try:
                        models = await provider_discovery_service.discover_google_models(api_key)
                        if models:
                            results.append(ValidationResult(
                                is_valid=True,
                                severity="info",
                                message=f"Google API key valid - {len(models)} models available"
                            ))
                    except Exception as e:
                        results.append(ValidationResult(
                            is_valid=False,
                            severity="error",
                            message="Google API key validation failed",
                            details=str(e)
                        ))
                        
            elif provider == "anthropic":
                api_key = await credential_service.get_credential("ANTHROPIC_API_KEY")
                if not api_key:
                    results.append(ValidationResult(
                        is_valid=False,
                        severity="error",
                        message="Anthropic API key not configured",
                        suggested_action="Add your Anthropic API key in the credentials section"
                    ))
                else:
                    results.append(ValidationResult(
                        is_valid=True,
                        severity="info",
                        message="Anthropic API key configured"
                    ))
                    
            elif provider == "ollama":
                rag_settings = await credential_service.get_credentials_by_category("rag_strategy")
                base_url = rag_settings.get("LLM_BASE_URL", "http://localhost:11434")
                
                health_status = await provider_discovery_service.check_provider_health(
                    "ollama", {"base_url": base_url}
                )
                
                if health_status.is_available:
                    results.append(ValidationResult(
                        is_valid=True,
                        severity="info",
                        message=f"Ollama instance healthy - {health_status.models_available} models available",
                        details=f"Response time: {health_status.response_time_ms:.0f}ms"
                    ))
                else:
                    results.append(ValidationResult(
                        is_valid=False,
                        severity="error",
                        message="Ollama instance not accessible",
                        details=health_status.error_message,
                        suggested_action="Check if Ollama is running and accessible"
                    ))
                    
        except Exception as e:
            results.append(ValidationResult(
                is_valid=False,
                severity="error",
                message=f"Error validating {provider} credentials",
                details=str(e)
            ))
            
        return results

    async def validate_model_compatibility(
        self, 
        llm_model: str, 
        embedding_model: str,
        llm_provider: str,
        embedding_provider: str
    ) -> CompatibilityCheck:
        """Check compatibility between selected LLM and embedding models."""
        try:
            # Get model specifications
            llm_spec = await self._get_model_spec(llm_model, llm_provider)
            embedding_spec = await self._get_model_spec(embedding_model, embedding_provider)
            
            recommendations = []
            performance_impact = "none"
            compatibility_score = 1.0
            
            # Check if models are available
            models_compatible = llm_spec is not None and embedding_spec is not None
            if not models_compatible:
                recommendations.append("One or more selected models are not available")
                compatibility_score -= 0.5
                
            # Check embedding dimensions consistency
            embedding_dimensions_match = True
            if embedding_spec and embedding_spec.embedding_dimensions:
                # This is a simplified check - in reality, you'd check against database schema
                if embedding_spec.embedding_dimensions not in [768, 1024, 1536, 3072]:
                    embedding_dimensions_match = False
                    recommendations.append("Unusual embedding dimensions may require schema migration")
                    performance_impact = "moderate"
                    compatibility_score -= 0.2
                    
            # Check tool support
            tool_support_available = llm_spec.supports_tools if llm_spec else False
            if not tool_support_available:
                recommendations.append("Selected LLM model does not support tool calling")
                compatibility_score -= 0.1
                
            # Provider mixing impact
            if llm_provider != embedding_provider:
                recommendations.append("Mixed providers may increase latency")
                if performance_impact == "none":
                    performance_impact = "minor"
                compatibility_score -= 0.1
                
            # Ollama-specific checks
            if llm_provider == "ollama" or embedding_provider == "ollama":
                recommendations.append("Ensure Ollama models are pulled locally for best performance")
                
            # Add positive recommendations
            if compatibility_score > 0.8:
                recommendations.insert(0, "Good model compatibility - recommended configuration")
            elif compatibility_score > 0.6:
                recommendations.insert(0, "Acceptable model compatibility with minor considerations")
            else:
                recommendations.insert(0, "Model compatibility issues detected - review recommendations")
                
            return CompatibilityCheck(
                models_compatible=models_compatible,
                embedding_dimensions_match=embedding_dimensions_match,
                tool_support_available=tool_support_available,
                performance_impact=performance_impact,
                compatibility_score=compatibility_score,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error checking model compatibility: {e}")
            return CompatibilityCheck(
                models_compatible=False,
                embedding_dimensions_match=False,
                tool_support_available=False,
                performance_impact="major",
                compatibility_score=0.0,
                recommendations=[f"Error checking compatibility: {str(e)}"]
            )

    async def validate_distributed_ollama_config(
        self, 
        ollama_instances: List[Dict[str, Any]]
    ) -> List[ValidationResult]:
        """Validate distributed Ollama configuration."""
        results = []
        
        try:
            if not ollama_instances:
                results.append(ValidationResult(
                    is_valid=False,
                    severity="error",
                    message="No Ollama instances configured",
                    suggested_action="Add at least one Ollama instance"
                ))
                return results
                
            # Check for primary instance
            primary_count = sum(1 for inst in ollama_instances if inst.get("is_primary", False))
            if primary_count == 0:
                results.append(ValidationResult(
                    is_valid=False,
                    severity="warning",
                    message="No primary Ollama instance designated",
                    suggested_action="Set one instance as primary for failover"
                ))
            elif primary_count > 1:
                results.append(ValidationResult(
                    is_valid=False,
                    severity="error",
                    message="Multiple primary Ollama instances configured",
                    suggested_action="Only one instance should be marked as primary"
                ))
                
            # Check for duplicate URLs
            urls = [inst.get("base_url") for inst in ollama_instances]
            if len(urls) != len(set(urls)):
                results.append(ValidationResult(
                    is_valid=False,
                    severity="error",
                    message="Duplicate Ollama instance URLs detected",
                    suggested_action="Each instance must have a unique URL"
                ))
                
            # Check connectivity to all instances
            healthy_instances = 0
            for i, instance in enumerate(ollama_instances):
                try:
                    health = await provider_discovery_service.check_provider_health(
                        "ollama", {"base_url": instance.get("base_url")}
                    )
                    
                    if health.is_available:
                        healthy_instances += 1
                        results.append(ValidationResult(
                            is_valid=True,
                            severity="info",
                            message=f"Instance '{instance.get('name', f'Instance {i+1}')}' is healthy"
                        ))
                    else:
                        results.append(ValidationResult(
                            is_valid=False,
                            severity="warning",
                            message=f"Instance '{instance.get('name', f'Instance {i+1}')}' is not accessible",
                            details=health.error_message,
                            suggested_action="Check if instance is running and network is accessible"
                        ))
                except Exception as e:
                    results.append(ValidationResult(
                        is_valid=False,
                        severity="error",
                        message=f"Failed to check instance '{instance.get('name', f'Instance {i+1}')}'",
                        details=str(e)
                    ))
                    
            # Overall health check
            if healthy_instances == 0:
                results.append(ValidationResult(
                    is_valid=False,
                    severity="error",
                    message="No Ollama instances are accessible",
                    suggested_action="Ensure at least one Ollama instance is running"
                ))
            elif healthy_instances < len(ollama_instances):
                results.append(ValidationResult(
                    is_valid=True,
                    severity="warning",
                    message=f"Only {healthy_instances}/{len(ollama_instances)} instances are healthy",
                    suggested_action="Check connectivity to unavailable instances"
                ))
            else:
                results.append(ValidationResult(
                    is_valid=True,
                    severity="info",
                    message=f"All {healthy_instances} Ollama instances are healthy"
                ))
                
        except Exception as e:
            results.append(ValidationResult(
                is_valid=False,
                severity="error",
                message="Error validating distributed Ollama configuration",
                details=str(e)
            ))
            
        return results

    async def get_configuration_recommendations(
        self, 
        current_config: Dict[str, Any]
    ) -> List[ValidationResult]:
        """Get configuration recommendations based on current setup."""
        recommendations = []
        
        try:
            llm_provider = current_config.get("llm_provider", "openai")
            embedding_provider = current_config.get("embedding_provider", "openai")
            
            # Provider-specific recommendations
            if llm_provider == "ollama":
                recommendations.append(ValidationResult(
                    is_valid=True,
                    severity="info",
                    message="Consider using Ollama for embeddings too for better integration",
                    suggested_action="Set embedding provider to Ollama if you have embedding models installed"
                ))
                
            if llm_provider == "openai" and embedding_provider == "openai":
                recommendations.append(ValidationResult(
                    is_valid=True,
                    severity="info",
                    message="OpenAI provides excellent model compatibility and performance",
                    details="Consider using text-embedding-3-small for cost efficiency or text-embedding-3-large for best quality"
                ))
                
            # Mixed provider recommendations
            if llm_provider != embedding_provider:
                recommendations.append(ValidationResult(
                    is_valid=True,
                    severity="info",
                    message="Mixed providers detected",
                    details="This setup may work well but consider network latency impacts",
                    suggested_action="Monitor performance and consider using same provider for both if issues arise"
                ))
                
            # Security recommendations
            recommendations.append(ValidationResult(
                is_valid=True,
                severity="info",
                message="Security recommendation",
                details="API keys are encrypted and stored securely",
                suggested_action="Regularly rotate API keys and monitor usage patterns"
            ))
            
        except Exception as e:
            recommendations.append(ValidationResult(
                is_valid=False,
                severity="error",
                message="Error generating configuration recommendations",
                details=str(e)
            ))
            
        return recommendations

    async def _get_model_spec(self, model_name: str, provider: str) -> Optional[ModelSpec]:
        """Get model specification for validation."""
        try:
            if provider == "openai":
                api_key = await credential_service.get_credential("OPENAI_API_KEY")
                if api_key:
                    models = await provider_discovery_service.discover_openai_models(api_key)
                    return next((m for m in models if m.name == model_name), None)
                    
            elif provider == "google":
                api_key = await credential_service.get_credential("GOOGLE_API_KEY")
                if api_key:
                    models = await provider_discovery_service.discover_google_models(api_key)
                    return next((m for m in models if m.name == model_name), None)
                    
            elif provider == "ollama":
                rag_settings = await credential_service.get_credentials_by_category("rag_strategy")
                base_url = rag_settings.get("LLM_BASE_URL", "http://localhost:11434")
                models = await provider_discovery_service.discover_ollama_models([base_url])
                return next((m for m in models if m.name == model_name), None)
                
            elif provider == "anthropic":
                api_key = await credential_service.get_credential("ANTHROPIC_API_KEY")
                if api_key:
                    models = await provider_discovery_service.discover_anthropic_models(api_key)
                    return next((m for m in models if m.name == model_name), None)
                    
        except Exception as e:
            logger.error(f"Error getting model spec for {model_name} from {provider}: {e}")
            
        return None

# Global instance
provider_validation_service = ProviderValidationService()