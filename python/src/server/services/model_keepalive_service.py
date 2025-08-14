"""
Model Keep-Alive Service

Manages model heartbeats during crawl operations to prevent model shutdowns.
Provides registration/deregistration for active crawls and monitors model health.
"""

import asyncio
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Set, Callable
from enum import Enum

from ..config.logfire_config import get_logger, safe_logfire_info, safe_logfire_error
from .llm_provider_service import get_llm_client, get_embedding_model
from .embeddings.embedding_service import create_embedding

logger = get_logger(__name__)


class ModelType(Enum):
    """Model types for keep-alive monitoring."""
    CHAT = "chat"
    EMBEDDING = "embedding"


@dataclass
class CrawlRegistration:
    """Information about a registered crawl operation."""
    crawl_id: str
    required_models: Set[ModelType]
    registered_at: datetime
    last_heartbeat: Optional[datetime] = None
    heartbeat_count: int = 0
    
    def __post_init__(self):
        if self.registered_at is None:
            self.registered_at = datetime.now(timezone.utc)


@dataclass
class ModelStatus:
    """Status information for a model."""
    model_type: ModelType
    provider: str
    model_name: str
    is_available: bool = True
    last_check: Optional[datetime] = None
    failure_count: int = 0
    last_error: Optional[str] = None
    
    def __post_init__(self):
        if self.last_check is None:
            self.last_check = datetime.now(timezone.utc)


@dataclass
class KeepAliveStats:
    """Keep-alive system statistics."""
    active_crawls: int = 0
    total_heartbeats_sent: int = 0
    total_failures: int = 0
    uptime_seconds: float = 0
    models_monitored: Dict[str, ModelStatus] = field(default_factory=dict)


class ModelKeepAliveManager:
    """
    Manages model keep-alive operations during crawl processes.
    
    This service ensures that chat and embedding models remain active during
    long-running crawl operations by sending periodic heartbeat requests.
    Includes graceful error handling and automatic recovery mechanisms.
    """
    
    def __init__(self, heartbeat_interval: int = 60, max_retries: int = 3, backoff_multiplier: float = 2.0):
        """
        Initialize the keep-alive manager.
        
        Args:
            heartbeat_interval: Seconds between heartbeat checks (default: 60)
            max_retries: Maximum number of retries for failed heartbeats (default: 3)
            backoff_multiplier: Exponential backoff multiplier for retries (default: 2.0)
        """
        self.heartbeat_interval = heartbeat_interval
        self.max_retries = max_retries
        self.backoff_multiplier = backoff_multiplier
        self.active_crawls: Dict[str, CrawlRegistration] = {}
        self.model_statuses: Dict[str, ModelStatus] = {}
        self.is_running = False
        self.heartbeat_task: Optional[asyncio.Task] = None
        self.start_time = datetime.now(timezone.utc)
        self.stats = KeepAliveStats()
        
        # Error handling and recovery
        self.max_consecutive_failures = 5  # Stop monitoring after 5 consecutive failures
        self.consecutive_failures = 0
        self.last_successful_heartbeat = datetime.now(timezone.utc)
        self.failure_recovery_delay = 30  # Wait 30 seconds before recovery attempt
        
        # Thread lock for thread-safe operations
        self._lock = threading.RLock()
        
        # Callbacks for status updates
        self._status_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []
        
        logger.info(f"ModelKeepAliveManager initialized with {heartbeat_interval}s heartbeat interval, "
                   f"max_retries={max_retries}, backoff_multiplier={backoff_multiplier}")
    
    def add_status_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Add a callback to receive status updates."""
        self._status_callbacks.append(callback)
    
    def remove_status_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Remove a status callback."""
        if callback in self._status_callbacks:
            self._status_callbacks.remove(callback)
    
    async def _notify_status_update(self, event_type: str, data: Dict[str, Any]):
        """Notify all status callbacks of an update."""
        for callback in self._status_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event_type, data)
                else:
                    callback(event_type, data)
            except Exception as e:
                logger.error(f"Error in status callback: {e}")
        
        # Also emit Socket.IO events for real-time updates
        await self._emit_socketio_update(event_type, data)
    
    async def _emit_socketio_update(self, event_type: str, data: Dict[str, Any]):
        """Emit Socket.IO events for model status updates."""
        try:
            # Import socket.IO handlers to avoid circular dependencies
            from ..socketio_app import get_socketio_instance
            
            sio = get_socketio_instance()
            if sio is None:
                logger.debug("Socket.IO instance not available for keep-alive updates")
                return
            
            # Emit different events based on the type
            if event_type == "heartbeat_completed":
                await sio.emit("model_availability_status", {
                    "timestamp": data.get("timestamp"),
                    "successful_models": data.get("successful_models", []),
                    "failed_models": data.get("failed_models", []),
                    "active_crawls": data.get("active_crawls", 0)
                }, room="model_health")
                
            elif event_type == "heartbeat_error":
                await sio.emit("keepalive_failure", {
                    "timestamp": data.get("timestamp"),
                    "error": data.get("error"),
                    "total_failures": data.get("total_failures", 0)
                }, room="model_health")
                
            elif event_type in ["crawl_registered", "crawl_deregistered"]:
                await sio.emit("crawl_model_status", {
                    "event": event_type,
                    "crawl_id": data.get("crawl_id"),
                    "timestamp": data.get("timestamp"),
                    "active_crawls": len(self.active_crawls)
                }, room="model_health")
            
            elif event_type in ["monitoring_started", "monitoring_stopped"]:
                await sio.emit("model_health_dashboard", {
                    "event": event_type,
                    "timestamp": data.get("timestamp"),
                    "is_running": self.is_running,
                    "uptime_seconds": data.get("uptime_seconds"),
                    "total_heartbeats": data.get("total_heartbeats", self.stats.total_heartbeats_sent)
                }, room="model_health")
            
        except Exception as e:
            # Don't let Socket.IO errors crash the keep-alive system
            logger.debug(f"Socket.IO emission failed for keep-alive event {event_type}: {e}")
    
    def register_crawl(self, crawl_id: str, required_models: Optional[List[str]] = None) -> bool:
        """
        Register a crawl operation for keep-alive monitoring.
        
        Args:
            crawl_id: Unique identifier for the crawl operation
            required_models: List of model types needed ("chat", "embedding"). 
                           Defaults to both if None.
        
        Returns:
            bool: True if registration successful, False otherwise
        """
        try:
            with self._lock:
                if crawl_id in self.active_crawls:
                    logger.warning(f"Crawl {crawl_id} already registered for keep-alive")
                    return False
                
                # Default to monitoring both models if not specified
                if required_models is None:
                    required_models = ["chat", "embedding"]
                
                # Convert string model types to enum
                model_set = set()
                for model_str in required_models:
                    try:
                        model_set.add(ModelType(model_str.lower()))
                    except ValueError:
                        logger.warning(f"Unknown model type: {model_str}, skipping")
                
                if not model_set:
                    logger.error(f"No valid model types specified for crawl {crawl_id}")
                    return False
                
                registration = CrawlRegistration(
                    crawl_id=crawl_id,
                    required_models=model_set,
                    registered_at=datetime.now(timezone.utc)
                )
                
                self.active_crawls[crawl_id] = registration
                self.stats.active_crawls = len(self.active_crawls)
                
                logger.info(f"Registered crawl {crawl_id} for keep-alive monitoring "
                          f"with models: {[m.value for m in model_set]}")
                
                # Start heartbeat monitoring if not already running
                if not self.is_running:
                    asyncio.create_task(self.start_monitoring())
                
                # Notify status update
                asyncio.create_task(self._notify_status_update("crawl_registered", {
                    "crawl_id": crawl_id,
                    "required_models": [m.value for m in model_set],
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }))
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to register crawl {crawl_id}: {e}")
            return False
    
    def deregister_crawl(self, crawl_id: str) -> bool:
        """
        Deregister a crawl operation from keep-alive monitoring.
        
        Args:
            crawl_id: Unique identifier for the crawl operation
        
        Returns:
            bool: True if deregistration successful, False otherwise
        """
        try:
            with self._lock:
                if crawl_id not in self.active_crawls:
                    logger.warning(f"Crawl {crawl_id} not registered for keep-alive")
                    return False
                
                registration = self.active_crawls.pop(crawl_id)
                self.stats.active_crawls = len(self.active_crawls)
                
                logger.info(f"Deregistered crawl {crawl_id} from keep-alive monitoring "
                          f"(was active for {datetime.now(timezone.utc) - registration.registered_at})")
                
                # Stop monitoring if no active crawls
                if not self.active_crawls and self.is_running:
                    asyncio.create_task(self.stop_monitoring())
                
                # Notify status update
                asyncio.create_task(self._notify_status_update("crawl_deregistered", {
                    "crawl_id": crawl_id,
                    "duration_seconds": (datetime.now(timezone.utc) - registration.registered_at).total_seconds(),
                    "heartbeat_count": registration.heartbeat_count,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }))
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to deregister crawl {crawl_id}: {e}")
            return False
    
    async def start_monitoring(self) -> bool:
        """
        Start the heartbeat monitoring process.
        
        Returns:
            bool: True if monitoring started successfully
        """
        try:
            if self.is_running:
                logger.warning("Keep-alive monitoring already running")
                return True
            
            self.is_running = True
            self.start_time = datetime.now(timezone.utc)
            
            # Start the heartbeat task
            self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            
            logger.info("Started model keep-alive monitoring")
            safe_logfire_info("Model keep-alive monitoring started")
            
            # Notify status update
            await self._notify_status_update("monitoring_started", {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "heartbeat_interval": self.heartbeat_interval
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start keep-alive monitoring: {e}")
            self.is_running = False
            return False
    
    async def stop_monitoring(self) -> bool:
        """
        Stop the heartbeat monitoring process.
        
        Returns:
            bool: True if monitoring stopped successfully
        """
        try:
            if not self.is_running:
                logger.warning("Keep-alive monitoring not running")
                return True
            
            self.is_running = False
            
            # Cancel the heartbeat task
            if self.heartbeat_task and not self.heartbeat_task.done():
                self.heartbeat_task.cancel()
                try:
                    await asyncio.wait_for(self.heartbeat_task, timeout=2.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
            
            uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()
            self.stats.uptime_seconds = uptime
            
            logger.info(f"Stopped model keep-alive monitoring (uptime: {uptime:.1f}s)")
            safe_logfire_info(f"Model keep-alive monitoring stopped after {uptime:.1f}s")
            
            # Notify status update
            await self._notify_status_update("monitoring_stopped", {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "uptime_seconds": uptime,
                "total_heartbeats": self.stats.total_heartbeats_sent
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop keep-alive monitoring: {e}")
            return False
    
    async def _heartbeat_loop(self):
        """Main heartbeat monitoring loop."""
        logger.info("Starting heartbeat monitoring loop")
        
        while self.is_running:
            try:
                # Check if we have any active crawls
                with self._lock:
                    if not self.active_crawls:
                        logger.debug("No active crawls, skipping heartbeat")
                        await asyncio.sleep(self.heartbeat_interval)
                        continue
                    
                    # Get all required model types from active crawls
                    required_models = set()
                    crawl_list = list(self.active_crawls.keys())
                    for registration in self.active_crawls.values():
                        required_models.update(registration.required_models)
                
                logger.debug(f"Sending heartbeats for {len(crawl_list)} active crawls, "
                           f"models: {[m.value for m in required_models]}")
                
                # Send heartbeats to required models with retry logic
                heartbeat_results = []
                any_successful = False
                
                for model_type in required_models:
                    result = await self._send_heartbeat_with_retry(model_type)
                    heartbeat_results.append((model_type, result))
                    if result:
                        any_successful = True
                
                # Update consecutive failure tracking
                current_time = datetime.now(timezone.utc)
                if any_successful:
                    self.consecutive_failures = 0
                    self.last_successful_heartbeat = current_time
                else:
                    self.consecutive_failures += 1
                
                # Check for critical failure threshold
                if self.consecutive_failures >= self.max_consecutive_failures:
                    logger.error(f"Critical failure: {self.consecutive_failures} consecutive heartbeat failures. "
                               f"Stopping keep-alive monitoring for safety.")
                    
                    await self._notify_status_update("critical_failure", {
                        "timestamp": current_time.isoformat(),
                        "consecutive_failures": self.consecutive_failures,
                        "last_successful": self.last_successful_heartbeat.isoformat(),
                        "action": "monitoring_stopped"
                    })
                    
                    # Stop monitoring to prevent further issues
                    await self.stop_monitoring()
                    break
                
                # Update crawl registration heartbeat times
                with self._lock:
                    for registration in self.active_crawls.values():
                        registration.last_heartbeat = current_time
                        registration.heartbeat_count += 1
                
                # Log heartbeat summary
                successful_models = [mt.value for mt, success in heartbeat_results if success]
                failed_models = [mt.value for mt, success in heartbeat_results if not success]
                
                if successful_models:
                    logger.debug(f"Heartbeat successful for models: {successful_models}")
                if failed_models:
                    logger.warning(f"Heartbeat failed for models: {failed_models} "
                                 f"(consecutive failures: {self.consecutive_failures})")
                
                self.stats.total_heartbeats_sent += len(heartbeat_results)
                
                # Notify status update with aggregated data
                await self._notify_status_update("heartbeat_completed", {
                    "timestamp": current_time.isoformat(),
                    "active_crawls": len(crawl_list),
                    "successful_models": successful_models,
                    "failed_models": failed_models,
                    "consecutive_failures": self.consecutive_failures,
                    "total_heartbeats": self.stats.total_heartbeats_sent
                })
                
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                safe_logfire_error(f"Keep-alive heartbeat loop error: {e}")
                self.stats.total_failures += 1
                
                # Notify error
                await self._notify_status_update("heartbeat_error", {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "error": str(e),
                    "total_failures": self.stats.total_failures
                })
            
            # Wait for next heartbeat interval
            await asyncio.sleep(self.heartbeat_interval)
        
        logger.info("Heartbeat monitoring loop stopped")
    
    async def _send_heartbeat_with_retry(self, model_type: ModelType) -> bool:
        """
        Send a heartbeat with retry logic and exponential backoff.
        
        Args:
            model_type: Type of model to send heartbeat to
        
        Returns:
            bool: True if heartbeat successful after retries, False otherwise
        """
        for attempt in range(self.max_retries + 1):  # +1 for initial attempt
            try:
                if attempt > 0:
                    # Exponential backoff delay
                    delay = min(self.backoff_multiplier ** (attempt - 1), 30)  # Cap at 30 seconds
                    logger.debug(f"Retrying {model_type.value} heartbeat (attempt {attempt + 1}/{self.max_retries + 1}) "
                               f"after {delay:.1f}s delay")
                    await asyncio.sleep(delay)
                
                success = await self._send_heartbeat(model_type)
                
                if success:
                    if attempt > 0:
                        logger.info(f"{model_type.value} heartbeat succeeded on retry attempt {attempt + 1}")
                    return True
                else:
                    if attempt < self.max_retries:
                        logger.debug(f"{model_type.value} heartbeat failed on attempt {attempt + 1}, will retry")
                    else:
                        logger.warning(f"{model_type.value} heartbeat failed after {self.max_retries + 1} attempts")
                
            except Exception as e:
                logger.error(f"Error during {model_type.value} heartbeat attempt {attempt + 1}: {e}")
                if attempt >= self.max_retries:
                    # Update model status with final error
                    model_key = f"{model_type.value}"
                    if model_key in self.model_statuses:
                        self.model_statuses[model_key].last_error = f"Failed after {self.max_retries + 1} attempts: {str(e)}"
                        self.model_statuses[model_key].failure_count += 1
        
        return False
    
    async def _send_heartbeat(self, model_type: ModelType) -> bool:
        """
        Send a heartbeat to a specific model type.
        
        Args:
            model_type: Type of model to send heartbeat to
        
        Returns:
            bool: True if heartbeat successful, False otherwise
        """
        model_key = f"{model_type.value}"
        
        try:
            start_time = time.time()
            
            if model_type == ModelType.CHAT:
                success = await self._chat_model_heartbeat()
            elif model_type == ModelType.EMBEDDING:
                success = await self._embedding_model_heartbeat()
            else:
                logger.error(f"Unknown model type for heartbeat: {model_type}")
                return False
            
            elapsed_time = time.time() - start_time
            
            # Update model status
            if model_key not in self.model_statuses:
                self.model_statuses[model_key] = ModelStatus(
                    model_type=model_type,
                    provider="unknown",
                    model_name="unknown"
                )
            
            status = self.model_statuses[model_key]
            status.last_check = datetime.now(timezone.utc)
            status.is_available = success
            
            if success:
                status.failure_count = 0
                status.last_error = None
                logger.debug(f"Heartbeat successful for {model_type.value} model ({elapsed_time:.2f}s)")
            else:
                status.failure_count += 1
                logger.warning(f"Heartbeat failed for {model_type.value} model (failures: {status.failure_count})")
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending heartbeat to {model_type.value} model: {e}")
            
            # Update failure status
            if model_key in self.model_statuses:
                status = self.model_statuses[model_key]
                status.failure_count += 1
                status.last_error = str(e)
                status.is_available = False
                status.last_check = datetime.now(timezone.utc)
            
            return False
    
    async def _chat_model_heartbeat(self) -> bool:
        """Send heartbeat to chat model with provider-aware handling."""
        try:
            from .credential_service import credential_service
            
            # Get provider configuration
            provider_config = await credential_service.get_active_provider("llm")
            provider_name = provider_config.get("provider", "openai")
            
            async with get_llm_client() as client:
                # Use provider-appropriate model name
                model_name = "gpt-3.5-turbo"  # Default
                if provider_name == "ollama":
                    # For Ollama, get the configured chat model from provider config
                    chat_model = provider_config.get("chat_model", "")
                    if chat_model:
                        model_name = chat_model
                    else:
                        # Skip keep-alive if no specific model is configured
                        logger.info("No Ollama chat model configured, skipping keep-alive")
                        return
                elif provider_name == "google":
                    model_name = "gemini-1.5-flash"
                
                # Send a minimal request to keep the model alive
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=1,
                    temperature=0
                )
                
                # Update model status info
                model_key = "chat"
                if model_key not in self.model_statuses:
                    self.model_statuses[model_key] = ModelStatus(
                        model_type=ModelType.CHAT,
                        provider=provider_name,
                        model_name=model_name
                    )
                else:
                    self.model_statuses[model_key].provider = provider_name
                    self.model_statuses[model_key].model_name = model_name
                
                # Check if we got a valid response
                if response and response.choices and len(response.choices) > 0:
                    return True
                return False
                
        except Exception as e:
            logger.debug(f"Chat model heartbeat failed: {e}")
            # Update error info
            model_key = "chat"
            if model_key in self.model_statuses:
                self.model_statuses[model_key].last_error = str(e)
            return False
    
    async def _embedding_model_heartbeat(self) -> bool:
        """Send heartbeat to embedding model with provider-aware handling."""
        try:
            from .credential_service import credential_service
            
            # Get provider configuration
            provider_config = await credential_service.get_active_provider("embedding")
            provider_name = provider_config.get("provider", "openai")
            
            # Get the actual embedding model name
            embedding_model = await get_embedding_model()
            
            # Create a minimal embedding to keep the model alive
            embedding = await create_embedding("ping")
            
            # Update model status info
            model_key = "embedding"
            if model_key not in self.model_statuses:
                self.model_statuses[model_key] = ModelStatus(
                    model_type=ModelType.EMBEDDING,
                    provider=provider_name,
                    model_name=embedding_model
                )
            else:
                self.model_statuses[model_key].provider = provider_name
                self.model_statuses[model_key].model_name = embedding_model
            
            # Check if we got a valid embedding
            if embedding and len(embedding) > 0:
                # Additional validation: ensure it's not all zeros
                if any(x != 0 for x in embedding):
                    return True
            return False
            
        except Exception as e:
            logger.debug(f"Embedding model heartbeat failed: {e}")
            # Update error info
            model_key = "embedding"
            if model_key in self.model_statuses:
                self.model_statuses[model_key].last_error = str(e)
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of the keep-alive system.
        
        Returns:
            dict: Current status information
        """
        with self._lock:
            uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()
            
            return {
                "is_running": self.is_running,
                "uptime_seconds": uptime,
                "heartbeat_interval": self.heartbeat_interval,
                "active_crawls": len(self.active_crawls),
                "crawl_details": [
                    {
                        "crawl_id": reg.crawl_id,
                        "required_models": [m.value for m in reg.required_models],
                        "registered_at": reg.registered_at.isoformat(),
                        "last_heartbeat": reg.last_heartbeat.isoformat() if reg.last_heartbeat else None,
                        "heartbeat_count": reg.heartbeat_count
                    }
                    for reg in self.active_crawls.values()
                ],
                "model_statuses": {
                    key: {
                        "model_type": status.model_type.value,
                        "provider": status.provider,
                        "model_name": status.model_name,
                        "is_available": status.is_available,
                        "last_check": status.last_check.isoformat() if status.last_check else None,
                        "failure_count": status.failure_count,
                        "last_error": status.last_error
                    }
                    for key, status in self.model_statuses.items()
                },
                "error_handling": {
                    "consecutive_failures": self.consecutive_failures,
                    "max_consecutive_failures": self.max_consecutive_failures,
                    "last_successful_heartbeat": self.last_successful_heartbeat.isoformat(),
                    "failure_recovery_delay": self.failure_recovery_delay,
                    "max_retries": self.max_retries,
                    "backoff_multiplier": self.backoff_multiplier
                },
                "stats": {
                    "total_heartbeats_sent": self.stats.total_heartbeats_sent,
                    "total_failures": self.stats.total_failures
                }
            }
    
    def get_crawl_registration(self, crawl_id: str) -> Optional[Dict[str, Any]]:
        """
        Get registration information for a specific crawl.
        
        Args:
            crawl_id: Crawl operation identifier
        
        Returns:
            dict: Registration information or None if not found
        """
        with self._lock:
            if crawl_id not in self.active_crawls:
                return None
            
            reg = self.active_crawls[crawl_id]
            return {
                "crawl_id": reg.crawl_id,
                "required_models": [m.value for m in reg.required_models],
                "registered_at": reg.registered_at.isoformat(),
                "last_heartbeat": reg.last_heartbeat.isoformat() if reg.last_heartbeat else None,
                "heartbeat_count": reg.heartbeat_count
            }


# Global instance
_keep_alive_manager: Optional[ModelKeepAliveManager] = None


def get_keep_alive_manager() -> ModelKeepAliveManager:
    """Get the global keep-alive manager instance."""
    global _keep_alive_manager
    if _keep_alive_manager is None:
        _keep_alive_manager = ModelKeepAliveManager()
    return _keep_alive_manager


# Convenience functions for integration
async def register_crawl_for_keepalive(crawl_id: str, required_models: Optional[List[str]] = None) -> bool:
    """Register a crawl for keep-alive monitoring."""
    manager = get_keep_alive_manager()
    return manager.register_crawl(crawl_id, required_models)


async def deregister_crawl_from_keepalive(crawl_id: str) -> bool:
    """Deregister a crawl from keep-alive monitoring."""
    manager = get_keep_alive_manager()
    return manager.deregister_crawl(crawl_id)


async def get_keepalive_status() -> Dict[str, Any]:
    """Get current keep-alive system status."""
    manager = get_keep_alive_manager()
    return manager.get_status()