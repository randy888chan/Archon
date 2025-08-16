"""
Credential management service for Archon backend

Handles loading, storing, and accessing credentials with encryption for sensitive values.
Credentials include API keys, service credentials, and application configuration.
"""

import base64
import os
import re
import time
from dataclasses import dataclass

# Removed direct logging import - using unified config
from typing import Any

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from supabase import Client, create_client

from ..config.logfire_config import get_logger

logger = get_logger(__name__)


@dataclass
class CredentialItem:
    """Represents a credential/setting item."""

    key: str
    value: str | None = None
    encrypted_value: str | None = None
    is_encrypted: bool = False
    category: str | None = None
    description: str | None = None


class CredentialService:
    """Service for managing application credentials and configuration."""

    def __init__(self):
        self._supabase: Client | None = None
        self._cache: dict[str, Any] = {}
        self._cache_initialized = False
        self._rag_settings_cache: dict[str, Any] | None = None
        self._rag_cache_timestamp: float | None = None
        self._rag_cache_ttl = 300  # 5 minutes TTL for RAG settings cache

    def _get_supabase_client(self) -> Client:
        """
        Get or create a properly configured Supabase client using environment variables.
        Uses the standard Supabase client initialization.
        """
        if self._supabase is None:
            url = os.getenv("SUPABASE_URL")
            key = os.getenv("SUPABASE_SERVICE_KEY")

            if not url or not key:
                raise ValueError(
                    "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in environment variables"
                )

            try:
                # Initialize with standard Supabase client - no need for custom headers
                self._supabase = create_client(url, key)

                # Extract project ID from URL for logging purposes only
                match = re.match(r"https://([^.]+)\.supabase\.co", url)
                if match:
                    project_id = match.group(1)
                    logger.info(f"Supabase client initialized for project: {project_id}")
                else:
                    logger.info("Supabase client initialized successfully")

            except Exception as e:
                logger.error(f"Error initializing Supabase client: {e}")
                raise

        return self._supabase

    def _get_encryption_key(self) -> bytes:
        """Generate encryption key from environment variables."""
        # Use Supabase service key as the basis for encryption key
        service_key = os.getenv("SUPABASE_SERVICE_KEY", "default-key-for-development")

        # Generate a proper encryption key using PBKDF2
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b"static_salt_for_credentials",  # In production, consider using a configurable salt
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(service_key.encode()))
        return key

    def _encrypt_value(self, value: str) -> str:
        """Encrypt a sensitive value using Fernet encryption."""
        if not value:
            return ""

        try:
            fernet = Fernet(self._get_encryption_key())
            encrypted_bytes = fernet.encrypt(value.encode("utf-8"))
            return base64.urlsafe_b64encode(encrypted_bytes).decode("utf-8")
        except Exception as e:
            logger.error(f"Error encrypting value: {e}")
            raise

    def _decrypt_value(self, encrypted_value: str) -> str:
        """Decrypt a sensitive value using Fernet encryption."""
        if not encrypted_value:
            return ""

        try:
            fernet = Fernet(self._get_encryption_key())
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_value.encode("utf-8"))
            decrypted_bytes = fernet.decrypt(encrypted_bytes)
            return decrypted_bytes.decode("utf-8")
        except Exception as e:
            logger.error(f"Error decrypting value: {e}")
            raise

    async def load_all_credentials(self) -> dict[str, Any]:
        """Load all credentials from database and cache them."""
        try:
            supabase = self._get_supabase_client()

            # Fetch all credentials
            result = supabase.table("archon_settings").select("*").execute()

            credentials = {}
            for item in result.data:
                key = item["key"]
                if item["is_encrypted"] and item["encrypted_value"]:
                    # For encrypted values, we store the encrypted version
                    # Decryption happens when the value is actually needed
                    credentials[key] = {
                        "encrypted_value": item["encrypted_value"],
                        "is_encrypted": True,
                        "category": item["category"],
                        "description": item["description"],
                    }
                else:
                    # Plain text values
                    credentials[key] = item["value"]

            self._cache = credentials
            self._cache_initialized = True
            logger.info(f"Loaded {len(credentials)} credentials from database")

            return credentials

        except Exception as e:
            logger.error(f"Error loading credentials: {e}")
            raise

    async def get_credential(self, key: str, default: Any = None, decrypt: bool = True) -> Any:
        """Get a credential value by key."""
        if not self._cache_initialized:
            await self.load_all_credentials()

        value = self._cache.get(key, default)

        # If it's an encrypted value and we want to decrypt it
        if isinstance(value, dict) and value.get("is_encrypted") and decrypt:
            encrypted_value = value.get("encrypted_value")
            if encrypted_value:
                try:
                    return self._decrypt_value(encrypted_value)
                except Exception as e:
                    logger.error(f"Failed to decrypt credential {key}: {e}")
                    return default

        return value

    async def get_encrypted_credential_raw(self, key: str) -> str | None:
        """Get the raw encrypted value for a credential (without decryption)."""
        if not self._cache_initialized:
            await self.load_all_credentials()

        value = self._cache.get(key)
        if isinstance(value, dict) and value.get("is_encrypted"):
            return value.get("encrypted_value")

        return None

    async def set_credential(
        self,
        key: str,
        value: str,
        is_encrypted: bool = False,
        category: str = None,
        description: str = None,
    ) -> bool:
        """Set a credential value."""
        try:
            supabase = self._get_supabase_client()

            if is_encrypted:
                encrypted_value = self._encrypt_value(value)
                data = {
                    "key": key,
                    "encrypted_value": encrypted_value,
                    "value": None,
                    "is_encrypted": True,
                    "category": category,
                    "description": description,
                }
                # Update cache with encrypted info
                self._cache[key] = {
                    "encrypted_value": encrypted_value,
                    "is_encrypted": True,
                    "category": category,
                    "description": description,
                }
            else:
                data = {
                    "key": key,
                    "value": value,
                    "encrypted_value": None,
                    "is_encrypted": False,
                    "category": category,
                    "description": description,
                }
                # Update cache with plain value
                self._cache[key] = value

            # Upsert to database with proper conflict handling
            result = (
                supabase.table("archon_settings")
                .upsert(
                    data,
                    on_conflict="key",  # Specify the unique column for conflict resolution
                )
                .execute()
            )

            # Invalidate RAG settings cache if this is a rag_strategy setting
            if category == "rag_strategy":
                self._rag_settings_cache = None
                self._rag_cache_timestamp = None
                logger.debug(f"Invalidated RAG settings cache due to update of {key}")

            logger.info(
                f"Successfully {'encrypted and ' if is_encrypted else ''}stored credential: {key}"
            )
            return True

        except Exception as e:
            logger.error(f"Error setting credential {key}: {e}")
            return False

    async def delete_credential(self, key: str) -> bool:
        """Delete a credential."""
        try:
            supabase = self._get_supabase_client()

            result = supabase.table("archon_settings").delete().eq("key", key).execute()

            # Remove from cache
            if key in self._cache:
                del self._cache[key]

            # Invalidate RAG settings cache if this was a rag_strategy setting
            # We check the cache to see if the deleted key was in rag_strategy category
            if self._rag_settings_cache is not None and key in self._rag_settings_cache:
                self._rag_settings_cache = None
                self._rag_cache_timestamp = None
                logger.debug(f"Invalidated RAG settings cache due to deletion of {key}")

            logger.info(f"Successfully deleted credential: {key}")
            return True

        except Exception as e:
            logger.error(f"Error deleting credential {key}: {e}")
            return False

    async def get_credentials_by_category(self, category: str) -> dict[str, Any]:
        """Get all credentials for a specific category."""
        if not self._cache_initialized:
            await self.load_all_credentials()

        # Special caching for rag_strategy category to reduce database calls
        if category == "rag_strategy":
            current_time = time.time()

            # Check if we have valid cached data
            if (
                self._rag_settings_cache is not None
                and self._rag_cache_timestamp is not None
                and current_time - self._rag_cache_timestamp < self._rag_cache_ttl
            ):
                logger.debug("Using cached RAG settings")
                return self._rag_settings_cache

        try:
            supabase = self._get_supabase_client()
            result = (
                supabase.table("archon_settings").select("*").eq("category", category).execute()
            )

            credentials = {}
            for item in result.data:
                key = item["key"]
                if item["is_encrypted"]:
                    credentials[key] = {
                        "encrypted_value": item["encrypted_value"],
                        "is_encrypted": True,
                        "description": item["description"],
                    }
                else:
                    credentials[key] = item["value"]

            # Cache rag_strategy results
            if category == "rag_strategy":
                self._rag_settings_cache = credentials
                self._rag_cache_timestamp = time.time()
                logger.debug(f"Cached RAG settings with {len(credentials)} items")

            return credentials

        except Exception as e:
            logger.error(f"Error getting credentials for category {category}: {e}")
            return {}

    async def list_all_credentials(self) -> list[CredentialItem]:
        """Get all credentials as a list of CredentialItem objects (for Settings UI)."""
        try:
            supabase = self._get_supabase_client()
            result = supabase.table("archon_settings").select("*").execute()

            credentials = []
            for item in result.data:
                # For encrypted values, decrypt them for UI display
                if item["is_encrypted"] and item["encrypted_value"]:
                    try:
                        decrypted_value = self._decrypt_value(item["encrypted_value"])
                        cred = CredentialItem(
                            key=item["key"],
                            value=decrypted_value,
                            encrypted_value=None,  # Don't expose encrypted value
                            is_encrypted=item["is_encrypted"],
                            category=item["category"],
                            description=item["description"],
                        )
                    except Exception as e:
                        logger.error(f"Failed to decrypt credential {item['key']}: {e}")
                        # If decryption fails, show placeholder
                        cred = CredentialItem(
                            key=item["key"],
                            value="[DECRYPTION ERROR]",
                            encrypted_value=None,
                            is_encrypted=item["is_encrypted"],
                            category=item["category"],
                            description=item["description"],
                        )
                else:
                    # Plain text values
                    cred = CredentialItem(
                        key=item["key"],
                        value=item["value"],
                        encrypted_value=None,
                        is_encrypted=item["is_encrypted"],
                        category=item["category"],
                        description=item["description"],
                    )
                credentials.append(cred)

            return credentials

        except Exception as e:
            logger.error(f"Error listing credentials: {e}")
            return []

    def get_config_as_env_dict(self) -> dict[str, str]:
        """
        Get configuration as environment variable style dict.
        Note: This returns plain text values only, encrypted values need special handling.
        """
        if not self._cache_initialized:
            # Synchronous fallback - load from cache if available
            logger.warning("Credentials not loaded, returning empty config")
            return {}

        env_dict = {}
        for key, value in self._cache.items():
            if isinstance(value, dict) and value.get("is_encrypted"):
                # Skip encrypted values in env dict - they need to be handled separately
                continue
            else:
                env_dict[key] = str(value) if value is not None else ""

        return env_dict

    # Provider Management Methods
    async def get_active_provider(self, service_type: str = "llm") -> dict[str, Any]:
        """
        Get the currently active provider configuration.

        Args:
            service_type: Either 'llm' or 'embedding'

        Returns:
            Dict with provider, api_key, base_url, and models
        """
        try:
            # Get RAG strategy settings (where UI saves provider selection)
            rag_settings = await self.get_credentials_by_category("rag_strategy")

            # Get the selected provider
            provider = rag_settings.get("LLM_PROVIDER", "openai")

            # Get API key for this provider
            api_key = await self._get_provider_api_key(provider)

            # Get base URL if needed
            base_url = self._get_provider_base_url(provider, rag_settings)

            # Get models
            chat_model = rag_settings.get("MODEL_CHOICE", "")
            embedding_model = rag_settings.get("EMBEDDING_MODEL", "")

            return {
                "provider": provider,
                "api_key": api_key,
                "base_url": base_url,
                "chat_model": chat_model,
                "embedding_model": embedding_model,
            }

        except Exception as e:
            logger.error(f"Error getting active provider for {service_type}: {e}")
            # Fallback to environment variable
            provider = os.getenv("LLM_PROVIDER", "openai")
            return {
                "provider": provider,
                "api_key": os.getenv("OPENAI_API_KEY"),
                "base_url": None,
                "chat_model": "",
                "embedding_model": "",
            }

    async def _get_provider_api_key(self, provider: str) -> str | None:
        """Get API key for a specific provider."""
        key_mapping = {
            "openai": "OPENAI_API_KEY",
            "google": "GOOGLE_API_KEY",
            "ollama": None,  # No API key needed
        }

        key_name = key_mapping.get(provider)
        if key_name:
            return await self.get_credential(key_name)
        return "ollama" if provider == "ollama" else None

    def _get_provider_base_url(self, provider: str, rag_settings: dict) -> str | None:
        """Get base URL for provider."""
        if provider == "ollama":
            return rag_settings.get("LLM_BASE_URL", "http://localhost:11434/v1")
        elif provider == "google":
            return "https://generativelanguage.googleapis.com/v1beta/openai/"
        return None  # Use default for OpenAI

    async def set_active_provider(self, provider: str, service_type: str = "llm") -> bool:
        """Set the active provider for a service type."""
        try:
            # For now, we'll update the RAG strategy settings
            return await self.set_credential(
                "llm_provider",
                provider,
                category="rag_strategy",
                description=f"Active {service_type} provider",
            )
        except Exception as e:
            logger.error(f"Error setting active provider {provider} for {service_type}: {e}")
            return False

    # Ollama Instance Management Methods
    async def get_ollama_instances(self) -> list[dict]:
        """
        Get all Ollama instances from database.
        
        Returns:
            List of OllamaInstance dictionaries with structure:
            {
                "id": str,
                "name": str,
                "baseUrl": str,
                "isEnabled": bool,
                "isPrimary": bool,
                "loadBalancingWeight": int,
                "isHealthy": bool (optional),
                "responseTimeMs": int (optional),
                "modelsAvailable": int (optional),
                "lastHealthCheck": str (optional)
            }
        """
        try:
            import json
            
            # Get OLLAMA_INSTANCES from rag_strategy category
            instances_raw = await self.get_credential("OLLAMA_INSTANCES", default=None)
            
            if instances_raw:
                # Parse JSON string to list of instances
                if isinstance(instances_raw, str):
                    instances = json.loads(instances_raw)
                else:
                    instances = instances_raw
                    
                # Validate structure
                if not isinstance(instances, list):
                    logger.warning("OLLAMA_INSTANCES is not a list, creating default instance")
                    return await self._create_default_ollama_instances()
                    
                # Validate each instance has required fields
                validated_instances = []
                for instance in instances:
                    if not isinstance(instance, dict):
                        continue
                        
                    # Ensure required fields exist
                    required_fields = ["id", "name", "baseUrl", "isEnabled", "isPrimary", "loadBalancingWeight"]
                    if all(field in instance for field in required_fields):
                        validated_instances.append(instance)
                    else:
                        logger.warning(f"Ollama instance missing required fields: {instance}")
                
                if validated_instances:
                    logger.debug(f"Retrieved {len(validated_instances)} Ollama instances from database")
                    return validated_instances
                else:
                    logger.info("No valid Ollama instances found, creating default")
                    return await self._create_default_ollama_instances()
            else:
                # No instances stored yet, create default based on LLM_BASE_URL
                logger.info("No OLLAMA_INSTANCES found, creating default from LLM_BASE_URL")
                return await self._create_default_ollama_instances()
                
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing OLLAMA_INSTANCES JSON: {e}")
            return await self._create_default_ollama_instances()
        except Exception as e:
            logger.error(f"Error getting Ollama instances: {e}")
            return await self._create_default_ollama_instances()

    async def _create_default_ollama_instances(self) -> list[dict]:
        """Create default Ollama instance based on existing LLM_BASE_URL."""
        import uuid
        
        # Get existing LLM_BASE_URL or use default
        base_url = await self.get_credential("LLM_BASE_URL", default="http://localhost:11434")
        
        # Clean up base URL (remove /v1 suffix if present)
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
            
        default_instance = {
            "id": str(uuid.uuid4()),
            "name": "Primary Ollama Instance",
            "baseUrl": base_url,
            "isEnabled": True,
            "isPrimary": True,
            "loadBalancingWeight": 100
        }
        
        # Save default instance to database
        await self.set_ollama_instances([default_instance])
        
        logger.info(f"Created default Ollama instance: {base_url}")
        return [default_instance]

    async def set_ollama_instances(self, instances: list[dict]) -> bool:
        """
        Store Ollama instances to database.
        
        Args:
            instances: List of OllamaInstance dictionaries
            
        Returns:
            bool: Success status
        """
        try:
            import json
            
            # Validate instances structure
            if not isinstance(instances, list):
                raise ValueError("Instances must be a list")
                
            # Validate each instance
            validated_instances = []
            primary_count = 0
            
            for instance in instances:
                if not isinstance(instance, dict):
                    raise ValueError(f"Instance must be a dict: {instance}")
                    
                # Check required fields
                required_fields = ["id", "name", "baseUrl", "isEnabled", "isPrimary", "loadBalancingWeight"]
                missing_fields = [field for field in required_fields if field not in instance]
                if missing_fields:
                    raise ValueError(f"Instance missing required fields {missing_fields}: {instance}")
                
                # Validate field types
                if not isinstance(instance["id"], str) or not instance["id"]:
                    raise ValueError(f"Instance id must be non-empty string: {instance['id']}")
                    
                if not isinstance(instance["name"], str) or not instance["name"]:
                    raise ValueError(f"Instance name must be non-empty string: {instance['name']}")
                    
                if not isinstance(instance["baseUrl"], str) or not instance["baseUrl"]:
                    raise ValueError(f"Instance baseUrl must be non-empty string: {instance['baseUrl']}")
                    
                if not isinstance(instance["isEnabled"], bool):
                    raise ValueError(f"Instance isEnabled must be boolean: {instance['isEnabled']}")
                    
                if not isinstance(instance["isPrimary"], bool):
                    raise ValueError(f"Instance isPrimary must be boolean: {instance['isPrimary']}")
                    
                if not isinstance(instance["loadBalancingWeight"], int) or instance["loadBalancingWeight"] < 1 or instance["loadBalancingWeight"] > 100:
                    raise ValueError(f"Instance loadBalancingWeight must be int 1-100: {instance['loadBalancingWeight']}")
                
                # Count primary instances
                if instance["isPrimary"]:
                    primary_count += 1
                    
                validated_instances.append(instance)
            
            # Ensure exactly one primary instance
            if primary_count == 0 and validated_instances:
                # Make first instance primary if none specified
                validated_instances[0]["isPrimary"] = True
                primary_count = 1
                logger.info("No primary instance specified, made first instance primary")
            elif primary_count > 1:
                # Make only the first primary instance primary
                primary_found = False
                for instance in validated_instances:
                    if instance["isPrimary"] and not primary_found:
                        primary_found = True
                    elif instance["isPrimary"]:
                        instance["isPrimary"] = False
                logger.warning(f"Multiple primary instances found, kept only the first")
            
            # Serialize to JSON
            instances_json = json.dumps(validated_instances, separators=(',', ':'))
            
            # Store in database
            success = await self.set_credential(
                "OLLAMA_INSTANCES",
                instances_json,
                is_encrypted=False,
                category="rag_strategy",
                description="Ollama instances configuration for load balancing"
            )
            
            if success:
                # Update LLM_BASE_URL to primary instance for backward compatibility
                primary_instance = next((inst for inst in validated_instances if inst["isPrimary"]), None)
                if primary_instance:
                    primary_url = primary_instance["baseUrl"]
                    if not primary_url.endswith("/v1"):
                        primary_url += "/v1"
                    await self.set_credential(
                        "LLM_BASE_URL",
                        primary_url,
                        is_encrypted=False,
                        category="rag_strategy",
                        description="Primary Ollama base URL (auto-updated from instances)"
                    )
                
                logger.info(f"Successfully stored {len(validated_instances)} Ollama instances")
                return True
            else:
                logger.error("Failed to store Ollama instances to database")
                return False
                
        except Exception as e:
            logger.error(f"Error setting Ollama instances: {e}")
            return False

    async def add_ollama_instance(self, instance: dict) -> bool:
        """
        Add a new Ollama instance.
        
        Args:
            instance: OllamaInstance dictionary
            
        Returns:
            bool: Success status
        """
        try:
            # Get existing instances
            existing_instances = await self.get_ollama_instances()
            
            # Check for duplicate ID or URL
            instance_id = instance.get("id")
            instance_url = instance.get("baseUrl")
            
            for existing in existing_instances:
                if existing.get("id") == instance_id:
                    raise ValueError(f"Instance with ID {instance_id} already exists")
                if existing.get("baseUrl") == instance_url:
                    raise ValueError(f"Instance with URL {instance_url} already exists")
            
            # If this is marked as primary, unmark existing primary
            if instance.get("isPrimary", False):
                for existing in existing_instances:
                    existing["isPrimary"] = False
            
            # Add new instance
            existing_instances.append(instance)
            
            # Save updated list
            return await self.set_ollama_instances(existing_instances)
            
        except Exception as e:
            logger.error(f"Error adding Ollama instance: {e}")
            return False

    async def remove_ollama_instance(self, instance_id: str) -> bool:
        """
        Remove an Ollama instance by ID.
        
        Args:
            instance_id: ID of instance to remove
            
        Returns:
            bool: Success status
        """
        try:
            # Get existing instances
            existing_instances = await self.get_ollama_instances()
            
            # Find instance to remove
            instance_to_remove = None
            remaining_instances = []
            
            for instance in existing_instances:
                if instance.get("id") == instance_id:
                    instance_to_remove = instance
                else:
                    remaining_instances.append(instance)
            
            if not instance_to_remove:
                raise ValueError(f"Instance with ID {instance_id} not found")
            
            # Don't allow removing the last instance
            if len(remaining_instances) == 0:
                raise ValueError("Cannot remove the last Ollama instance")
            
            # If removing primary instance, make first remaining instance primary
            if instance_to_remove.get("isPrimary", False) and remaining_instances:
                remaining_instances[0]["isPrimary"] = True
                logger.info(f"Made instance {remaining_instances[0]['id']} primary after removing primary instance")
            
            # Save updated list
            success = await self.set_ollama_instances(remaining_instances)
            
            if success:
                logger.info(f"Successfully removed Ollama instance: {instance_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error removing Ollama instance {instance_id}: {e}")
            return False

    async def update_ollama_instance(self, instance_id: str, updates: dict) -> bool:
        """
        Update an Ollama instance with new data.
        
        Args:
            instance_id: ID of instance to update
            updates: Dictionary of fields to update
            
        Returns:
            bool: Success status
        """
        try:
            # Get existing instances
            existing_instances = await self.get_ollama_instances()
            
            # Find instance to update
            instance_found = False
            
            for i, instance in enumerate(existing_instances):
                if instance.get("id") == instance_id:
                    # Apply updates
                    for key, value in updates.items():
                        if key in ["id", "name", "baseUrl", "isEnabled", "isPrimary", "loadBalancingWeight", 
                                 "isHealthy", "responseTimeMs", "modelsAvailable", "lastHealthCheck"]:
                            instance[key] = value
                    
                    # If this instance is being set as primary, unmark others
                    if updates.get("isPrimary", False):
                        for j, other_instance in enumerate(existing_instances):
                            if j != i:
                                other_instance["isPrimary"] = False
                    
                    instance_found = True
                    break
            
            if not instance_found:
                raise ValueError(f"Instance with ID {instance_id} not found")
            
            # Save updated list
            success = await self.set_ollama_instances(existing_instances)
            
            if success:
                logger.debug(f"Successfully updated Ollama instance {instance_id} with {updates}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error updating Ollama instance {instance_id}: {e}")
            return False

    async def get_primary_ollama_instance(self) -> dict | None:
        """
        Get the primary Ollama instance.
        
        Returns:
            Primary instance dict or None if not found
        """
        try:
            instances = await self.get_ollama_instances()
            
            for instance in instances:
                if instance.get("isPrimary", False) and instance.get("isEnabled", True):
                    return instance
            
            # Fallback to first enabled instance
            for instance in instances:
                if instance.get("isEnabled", True):
                    return instance
            
            # Fallback to first instance regardless of enabled status
            if instances:
                return instances[0]
                
            return None
            
        except Exception as e:
            logger.error(f"Error getting primary Ollama instance: {e}")
            return None

    async def get_healthy_ollama_instances(self) -> list[dict]:
        """
        Get all healthy and enabled Ollama instances for load balancing.
        
        Returns:
            List of healthy instance dicts
        """
        try:
            instances = await self.get_ollama_instances()
            
            healthy_instances = []
            for instance in instances:
                if (instance.get("isEnabled", True) and 
                    instance.get("isHealthy", True)):  # Default to healthy if not specified
                    healthy_instances.append(instance)
            
            # If no healthy instances, return enabled instances
            if not healthy_instances:
                enabled_instances = [inst for inst in instances if inst.get("isEnabled", True)]
                if enabled_instances:
                    logger.warning("No healthy Ollama instances found, returning enabled instances")
                    return enabled_instances
            
            # If no enabled instances, return primary instance as fallback
            if not healthy_instances:
                primary = await self.get_primary_ollama_instance()
                if primary:
                    logger.warning("No enabled Ollama instances found, returning primary as fallback")
                    return [primary]
            
            return healthy_instances
            
        except Exception as e:
            logger.error(f"Error getting healthy Ollama instances: {e}")
            return []


# Global instance
credential_service = CredentialService()


async def get_credential(key: str, default: Any = None) -> Any:
    """Convenience function to get a credential."""
    return await credential_service.get_credential(key, default)


async def set_credential(
    key: str, value: str, is_encrypted: bool = False, category: str = None, description: str = None
) -> bool:
    """Convenience function to set a credential."""
    return await credential_service.set_credential(key, value, is_encrypted, category, description)


async def initialize_credentials() -> None:
    """Initialize the credential service by loading all credentials and setting environment variables."""
    await credential_service.load_all_credentials()

    # Only set infrastructure/startup credentials as environment variables
    # RAG settings will be looked up on-demand from the credential service
    infrastructure_credentials = [
        "OPENAI_API_KEY",  # Required for API client initialization
        "HOST",  # Server binding configuration
        "PORT",  # Server binding configuration
        "MCP_TRANSPORT",  # Server transport mode
        "LOGFIRE_ENABLED",  # Logging infrastructure setup
        "PROJECTS_ENABLED",  # Feature flag for module loading
    ]

    # LLM provider credentials (for sync client support)
    provider_credentials = [
        "GOOGLE_API_KEY",  # Google Gemini API key
        "LLM_PROVIDER",  # Selected provider
        "LLM_BASE_URL",  # Ollama base URL
        "EMBEDDING_MODEL",  # Custom embedding model
        "MODEL_CHOICE",  # Chat model for sync contexts
    ]

    # RAG settings that should NOT be set as env vars (will be looked up on demand):
    # - USE_CONTEXTUAL_EMBEDDINGS
    # - CONTEXTUAL_EMBEDDINGS_MAX_WORKERS
    # - USE_HYBRID_SEARCH
    # - USE_AGENTIC_RAG
    # - USE_RERANKING

    # Code extraction settings (loaded on demand, not set as env vars):
    # - MIN_CODE_BLOCK_LENGTH
    # - MAX_CODE_BLOCK_LENGTH
    # - ENABLE_COMPLETE_BLOCK_DETECTION
    # - ENABLE_LANGUAGE_SPECIFIC_PATTERNS
    # - ENABLE_PROSE_FILTERING
    # - MAX_PROSE_RATIO
    # - MIN_CODE_INDICATORS
    # - ENABLE_DIAGRAM_FILTERING
    # - ENABLE_CONTEXTUAL_LENGTH
    # - CODE_EXTRACTION_MAX_WORKERS
    # - CONTEXT_WINDOW_SIZE
    # - ENABLE_CODE_SUMMARIES

    # Set infrastructure credentials
    for key in infrastructure_credentials:
        try:
            value = await credential_service.get_credential(key, decrypt=True)
            if value:
                os.environ[key] = str(value)
                logger.info(f"Set environment variable: {key}")
        except Exception as e:
            logger.warning(f"Failed to set environment variable {key}: {e}")

    # Set provider credentials with proper environment variable names
    for key in provider_credentials:
        try:
            value = await credential_service.get_credential(key, decrypt=True)
            if value:
                # Map credential keys to environment variable names
                env_key = key.upper()  # Convert to uppercase for env vars
                os.environ[env_key] = str(value)
                logger.info(f"Set environment variable: {env_key}")
        except Exception:
            # This is expected for optional credentials
            logger.debug(f"Optional credential not set: {key}")

    logger.info("✅ Credentials loaded and environment variables set")
