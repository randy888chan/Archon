"""
Configuration models for providers.

Defines configuration structures for different providers.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum


class ProviderType(Enum):
    """Supported provider types."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    GROQ = "groq"
    TOGETHER = "together"
    OLLAMA = "ollama"
    MISTRAL = "mistral"
    COHERE = "cohere"


@dataclass
class ProviderConfig:
    """
    Base configuration for all providers.
    
    TODO: Add validation methods
    """
    provider_type: ProviderType
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    organization_id: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3
    custom_headers: Dict[str, str] = field(default_factory=dict)
    
    def validate(self) -> bool:
        """
        Validate configuration.
        
        TODO: Implement validation logic
        
        Returns:
            True if configuration is valid
        """
        # Basic validation
        if self.provider_type in [ProviderType.OPENAI, ProviderType.ANTHROPIC, ProviderType.GOOGLE]:
            return bool(self.api_key)
        return True


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    model_name: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls."""
        config = {
            "model": self.model_name,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }
        
        if self.max_tokens:
            config["max_tokens"] = self.max_tokens
        
        if self.stop_sequences:
            config["stop"] = self.stop_sequences
        
        return config


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    requests_per_minute: int = 60
    tokens_per_minute: int = 90000
    requests_per_day: Optional[int] = None
    concurrent_requests: int = 10
    
    def get_delay_between_requests(self) -> float:
        """Calculate minimum delay between requests."""
        return 60.0 / self.requests_per_minute


@dataclass
class ProviderModels:
    """
    Model listings for a provider.
    
    TODO: Load from API or configuration
    """
    chat_models: List[str] = field(default_factory=list)
    embedding_models: List[str] = field(default_factory=list)
    vision_models: List[str] = field(default_factory=list)
    
    @classmethod
    def for_openai(cls) -> "ProviderModels":
        """Get OpenAI models."""
        return cls(
            chat_models=[
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4-turbo",
                "gpt-4",
                "gpt-3.5-turbo",
            ],
            embedding_models=[
                "text-embedding-3-small",
                "text-embedding-3-large",
                "text-embedding-ada-002",
            ],
            vision_models=[
                "gpt-4o",
                "gpt-4-turbo",
            ]
        )
    
    @classmethod
    def for_anthropic(cls) -> "ProviderModels":
        """Get Anthropic models."""
        return cls(
            chat_models=[
                "claude-3-5-sonnet-latest",
                "claude-3-5-haiku-latest",
                "claude-3-opus-latest",
                "claude-3-sonnet-20240229",
                "claude-3-haiku-20240307",
            ],
            embedding_models=[],  # Anthropic doesn't provide embeddings
            vision_models=[
                "claude-3-5-sonnet-latest",
                "claude-3-opus-latest",
            ]
        )
    
    @classmethod
    def for_google(cls) -> "ProviderModels":
        """Get Google models."""
        return cls(
            chat_models=[
                "gemini-2.0-flash-exp",
                "gemini-1.5-pro",
                "gemini-1.5-flash",
                "gemini-1.5-flash-8b",
            ],
            embedding_models=[
                "text-embedding-004",
                "text-multilingual-embedding-002",
            ],
            vision_models=[
                "gemini-1.5-pro",
                "gemini-1.5-flash",
            ]
        )
    
    @classmethod
    def for_groq(cls) -> "ProviderModels":
        """Get Groq models."""
        return cls(
            chat_models=[
                "llama-3.3-70b-versatile",
                "llama-3.1-70b-versatile",
                "llama-3.1-8b-instant",
                "mixtral-8x7b-32768",
                "gemma2-9b-it",
            ],
            embedding_models=[],  # Groq doesn't provide embeddings
            vision_models=[]
        )
    
    @classmethod
    def for_together(cls) -> "ProviderModels":
        """Get Together AI models."""
        return cls(
            chat_models=[
                "meta-llama/Llama-3.3-70B-Instruct-Turbo",
                "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
                "Qwen/QwQ-32B-Preview",
                "mistralai/Mixtral-8x7B-Instruct-v0.1",
                "google/gemma-2-27b-it",
            ],
            embedding_models=[
                "togethercomputer/m2-bert-80M-8k-retrieval",
                "BAAI/bge-large-en-v1.5",
            ],
            vision_models=[
                "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
            ]
        )


@dataclass
class ProviderEndpoints:
    """
    API endpoints for a provider.
    
    TODO: Add more endpoint types
    """
    chat_completion: str = "/chat/completions"
    embedding: str = "/embeddings"
    models_list: str = "/models"
    
    @classmethod
    def for_provider(cls, provider_type: ProviderType) -> "ProviderEndpoints":
        """Get endpoints for a specific provider."""
        # Most providers follow OpenAI's API structure
        return cls()