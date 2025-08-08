"""
Blocking Helpers

Legacy helper functions that raise errors when called.
Use async/await patterns directly instead.
"""
import asyncio
from typing import Any, Coroutine, List, Dict, Optional
import threading

from ..config.logfire_config import get_logger

logger = get_logger(__name__)

# Legacy variables
_main_event_loop: Optional[asyncio.AbstractEventLoop] = None
_main_loop_thread_id: Optional[int] = None


def set_main_event_loop(loop: asyncio.AbstractEventLoop):
    """Legacy function that logs a warning when called."""
    global _main_event_loop, _main_loop_thread_id
    _main_event_loop = loop
    _main_loop_thread_id = threading.current_thread().ident
    logger.warning("set_main_event_loop called but should use async patterns instead")


def run_async_in_thread(coro: Coroutine, timeout: float = 30.0) -> Any:
    """Legacy function that raises an error when called."""
    logger.error("run_async_in_thread called - use async tasks directly instead")
    raise RuntimeError(
        "run_async_in_thread should not be used. "
        "Use asyncio.create_task() or await the coroutine directly."
    )


# Legacy wrapper classes

class BlockingEmbeddingWrapper:
    """Legacy wrapper that raises errors when methods are called."""
    
    @staticmethod
    def generate_embeddings_batch(texts: List[str], progress_queue=None) -> List[List[float]]:
        """Legacy method that raises an error when called"""
        logger.error("BlockingEmbeddingWrapper.generate_embeddings_batch called - use async services instead")
        raise RuntimeError(
            "BlockingEmbeddingWrapper should not be used. "
            "Use async embedding services directly: await create_embeddings_batch(texts)"
        )


class BlockingStorageWrapper:
    """Legacy wrapper that raises errors when methods are called."""
    
    def __init__(self, supabase_client):
        """Legacy initialization that logs a warning"""
        self.client = supabase_client
        logger.warning("BlockingStorageWrapper initialized - use async storage services instead")
    
    def store_documents_batch(
        self, 
        documents: List[Dict[str, Any]], 
        progress_queue=None,
        batch_size: int = 10
    ) -> int:
        """Legacy method that raises an error when called"""
        logger.error("BlockingStorageWrapper.store_documents_batch called - use async services instead")
        raise RuntimeError(
            "BlockingStorageWrapper should not be used. "
            "Use async storage services directly: await add_documents_to_supabase(...)"
        )