"""
Progress Tracker Utility

Consolidates all Socket.IO progress tracking operations for cleaner service code.
"""

import asyncio
from datetime import datetime
from typing import Any

from ...config.logfire_config import safe_logfire_error, safe_logfire_info


class ProgressTracker:
    """
    Utility class for tracking and broadcasting progress updates via Socket.IO.
    Consolidates all progress-related Socket.IO operations.
    """

    def __init__(self, sio, progress_id: str, operation_type: str = "crawl"):
        """
        Initialize the progress tracker.

        Args:
            sio: Socket.IO instance
            progress_id: Unique progress identifier
            operation_type: Type of operation (crawl, upload, etc.)
        """
        self.sio = sio
        self.progress_id = progress_id
        self.operation_type = operation_type
        self.state = {
            "progressId": progress_id,
            "startTime": datetime.now().isoformat(),
            "status": "initializing",
            "percentage": 0,
            "logs": [],
            # Enhanced monitoring fields
            "heartbeat": {
                "lastUpdate": datetime.now().isoformat(),
                "isAlive": True,
                "intervalMs": 30000  # 30 seconds
            },
            "performance": {
                "itemsPerMinute": 0,
                "estimatedTimeRemaining": None,
                "processingRate": {
                    "current": 0,
                    "average": 0,
                    "unit": "items/min"
                }
            },
            "detailedProgress": {
                "currentOperation": "initializing",
                "subOperations": [],
                "stageName": "setup",
                "stageProgress": 0
            }
        }
        
        # Performance tracking
        self._start_time = datetime.now()
        self._items_processed = 0
        self._last_rate_calculation = self._start_time
        self._rate_samples = []
        self._heartbeat_task = None

    async def start(self, initial_data: dict[str, Any] | None = None):
        """
        Start progress tracking with initial data.

        Args:
            initial_data: Optional initial data to include
        """
        self.state["status"] = "starting"
        self.state["startTime"] = datetime.now().isoformat()
        self.state["heartbeat"]["lastUpdate"] = self.state["startTime"]
        self.state["detailedProgress"]["currentOperation"] = "starting"
        self.state["detailedProgress"]["stageName"] = "initialization"

        if initial_data:
            self.state.update(initial_data)

        await self._emit_progress()
        safe_logfire_info(
            f"Progress tracking started | progress_id={self.progress_id} | type={self.operation_type}"
        )
        
        # Start heartbeat task
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def update(self, status: str, percentage: int, log: str, **kwargs):
        """
        Update progress with status, percentage, and log message.

        Args:
            status: Current status (analyzing, crawling, processing, etc.)
            percentage: Progress percentage (0-100)
            log: Log message describing current operation
            **kwargs: Additional data to include in update
        """
        current_time = datetime.now()
        
        # Update core state
        self.state.update({
            "status": status,
            "percentage": min(100, max(0, percentage)),
            "log": log,
            "timestamp": current_time.isoformat(),
        })
        
        # Update heartbeat
        self.state["heartbeat"]["lastUpdate"] = current_time.isoformat()
        self.state["heartbeat"]["isAlive"] = True
        
        # Update detailed progress if provided
        if "currentOperation" in kwargs:
            self.state["detailedProgress"]["currentOperation"] = kwargs["currentOperation"]
        if "stageName" in kwargs:
            self.state["detailedProgress"]["stageName"] = kwargs["stageName"]
        if "stageProgress" in kwargs:
            self.state["detailedProgress"]["stageProgress"] = kwargs["stageProgress"]

        # Add log entry with enhanced metadata
        if "logs" not in self.state:
            self.state["logs"] = []
        self.state["logs"].append({
            "timestamp": current_time.isoformat(),
            "message": log,
            "status": status,
            "percentage": percentage,
            "operation": kwargs.get("currentOperation", status),
            "stage": kwargs.get("stageName", "processing")
        })

        # Update performance metrics
        if "itemsProcessed" in kwargs:
            self._update_performance_metrics(kwargs["itemsProcessed"])

        # Add any additional data
        for key, value in kwargs.items():
            if key not in ["currentOperation", "stageName", "stageProgress", "itemsProcessed"]:
                self.state[key] = value

        await self._emit_progress()

    async def complete(self, completion_data: dict[str, Any] | None = None):
        """
        Mark progress as completed with optional completion data.

        Args:
            completion_data: Optional data about the completed operation
        """
        self.state["status"] = "completed"
        self.state["percentage"] = 100
        self.state["endTime"] = datetime.now().isoformat()
        self.state["heartbeat"]["isAlive"] = False

        if completion_data:
            self.state.update(completion_data)

        # Calculate duration and final performance metrics
        if "startTime" in self.state:
            start = datetime.fromisoformat(self.state["startTime"])
            end = datetime.fromisoformat(self.state["endTime"])
            duration = (end - start).total_seconds()
            self.state["duration"] = duration
            self.state["durationFormatted"] = self._format_duration(duration)
            
            # Final performance summary
            if self._items_processed > 0:
                final_rate = (self._items_processed / duration) * 60
                self.state["performance"]["finalRate"] = round(final_rate, 2)

        # Cancel heartbeat task
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()

        await self._emit_progress()
        safe_logfire_info(
            f"Progress completed | progress_id={self.progress_id} | type={self.operation_type} | duration={self.state.get('durationFormatted', 'unknown')}"
        )

    async def error(self, error_message: str, error_details: dict[str, Any] | None = None):
        """
        Mark progress as failed with error information.

        Args:
            error_message: Error message
            error_details: Optional additional error details
        """
        self.state.update({
            "status": "error",
            "error": error_message,
            "errorTime": datetime.now().isoformat(),
        })
        
        self.state["heartbeat"]["isAlive"] = False

        if error_details:
            self.state["errorDetails"] = error_details

        # Cancel heartbeat task
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()

        await self._emit_progress()
        safe_logfire_error(
            f"Progress error | progress_id={self.progress_id} | type={self.operation_type} | error={error_message}"
        )

    async def update_batch_progress(
        self, current_batch: int, total_batches: int, batch_size: int, message: str
    ):
        """
        Update progress for batch operations.

        Args:
            current_batch: Current batch number (1-based)
            total_batches: Total number of batches
            batch_size: Size of each batch
            message: Progress message
        """
        percentage = int((current_batch / total_batches) * 100)
        await self.update(
            status="processing_batch",
            percentage=percentage,
            log=message,
            currentBatch=current_batch,
            totalBatches=total_batches,
            batchSize=batch_size,
        )

    async def update_crawl_stats(
        self, processed_pages: int, total_pages: int, current_url: str | None = None
    ):
        """
        Update crawling statistics.

        Args:
            processed_pages: Number of pages processed
            total_pages: Total pages to process
            current_url: Currently processing URL
        """
        percentage = int((processed_pages / max(total_pages, 1)) * 100)
        log = f"Processing page {processed_pages}/{total_pages}"
        if current_url:
            log += f": {current_url}"

        await self.update(
            status="crawling",
            percentage=percentage,
            log=log,
            processedPages=processed_pages,
            totalPages=total_pages,
            currentUrl=current_url,
        )

    async def update_storage_progress(
        self, chunks_stored: int, total_chunks: int, operation: str = "storing"
    ):
        """
        Update document storage progress.

        Args:
            chunks_stored: Number of chunks stored
            total_chunks: Total chunks to store
            operation: Storage operation description
        """
        percentage = int((chunks_stored / max(total_chunks, 1)) * 100)
        await self.update(
            status="document_storage",
            percentage=percentage,
            log=f"{operation}: {chunks_stored}/{total_chunks} chunks",
            chunksStored=chunks_stored,
            totalChunks=total_chunks,
        )

    async def _emit_progress(self):
        """Emit progress update via Socket.IO."""
        event_name = f"{self.operation_type}_progress"

        # Log detailed progress info for debugging
        safe_logfire_info(f"📢 [SOCKETIO] Broadcasting {event_name} to room: {self.progress_id}")
        safe_logfire_info(
            f"📢 [SOCKETIO] Status: {self.state.get('status')} | Percentage: {self.state.get('percentage')}% | Operation: {self.state['detailedProgress']['currentOperation']}"
        )

        # Emit to the progress room
        await self.sio.emit(event_name, self.state, room=self.progress_id)

    def _format_duration(self, seconds: float) -> str:
        """Format duration in seconds to human-readable string."""
        if seconds < 60:
            return f"{seconds:.1f} seconds"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f} minutes"
        else:
            hours = seconds / 3600
            return f"{hours:.1f} hours"

    def get_state(self) -> dict[str, Any]:
        """Get current progress state."""
        return self.state.copy()

    async def join_room(self, sid: str):
        """Add a socket ID to the progress room."""
        await self.sio.enter_room(sid, self.progress_id)
        safe_logfire_info(f"Socket {sid} joined progress room {self.progress_id}")

    async def leave_room(self, sid: str):
        """Remove a socket ID from the progress room."""
        await self.sio.leave_room(sid, self.progress_id)
        safe_logfire_info(f"Socket {sid} left progress room {self.progress_id}")

    async def update_detailed_batch_progress(
        self, batch_num: int, total_batches: int, operation: str, 
        sub_progress: int = 0, details: dict = None
    ):
        """
        Update progress with detailed batch operation information.
        
        Args:
            batch_num: Current batch number
            total_batches: Total number of batches
            operation: Current operation within batch
            sub_progress: Progress within current operation (0-100)
            details: Additional operation details
        """
        batch_percentage = int(((batch_num - 1) / total_batches) * 100)
        overall_percentage = batch_percentage + int((sub_progress / 100) * (100 / total_batches))
        
        operation_details = details or {}
        
        message = f"Batch {batch_num}/{total_batches}: {operation}"
        if sub_progress > 0:
            message += f" ({sub_progress}%)"
            
        await self.update(
            status="detailed_batch_processing",
            percentage=min(100, overall_percentage),
            log=message,
            currentOperation=operation,
            stageName="batch_processing",
            stageProgress=sub_progress,
            batchDetails={
                "currentBatch": batch_num,
                "totalBatches": total_batches,
                "operation": operation,
                "subProgress": sub_progress,
                **operation_details
            }
        )

    async def update_embedding_progress(
        self, embeddings_created: int, total_embeddings: int, 
        operation_type: str = "creating", provider: str = None
    ):
        """
        Update progress for embedding generation operations.
        
        Args:
            embeddings_created: Number of embeddings processed
            total_embeddings: Total embeddings to process
            operation_type: Type of embedding operation
            provider: Embedding provider being used
        """
        percentage = int((embeddings_created / max(total_embeddings, 1)) * 100)
        
        message = f"{operation_type.title()} embeddings: {embeddings_created}/{total_embeddings}"
        if provider:
            message += f" (via {provider})"
            
        await self.update(
            status="embedding_generation",
            percentage=percentage,
            log=message,
            currentOperation=f"{operation_type}_embeddings",
            stageName="embedding_processing",
            embeddingDetails={
                "created": embeddings_created,
                "total": total_embeddings,
                "operation": operation_type,
                "provider": provider,
                "rate": self._calculate_embedding_rate(embeddings_created)
            },
            itemsProcessed=embeddings_created
        )

    def _update_performance_metrics(self, items_processed: int):
        """Update performance tracking metrics."""
        self._items_processed = items_processed
        current_time = datetime.now()
        
        # Calculate time-based metrics
        elapsed = (current_time - self._start_time).total_seconds()
        if elapsed > 0:
            items_per_second = items_processed / elapsed
            items_per_minute = items_per_second * 60
            
            # Update rate samples for average calculation
            self._rate_samples.append(items_per_minute)
            if len(self._rate_samples) > 10:  # Keep last 10 samples
                self._rate_samples.pop(0)
            
            # Calculate average rate
            avg_rate = sum(self._rate_samples) / len(self._rate_samples)
            
            # Update performance state
            self.state["performance"]["itemsPerMinute"] = round(items_per_minute, 2)
            self.state["performance"]["processingRate"]["current"] = round(items_per_minute, 2)
            self.state["performance"]["processingRate"]["average"] = round(avg_rate, 2)
            
            # Estimate time remaining if we have total items
            if "totalItems" in self.state and avg_rate > 0:
                remaining_items = self.state["totalItems"] - items_processed
                minutes_remaining = remaining_items / avg_rate
                self.state["performance"]["estimatedTimeRemaining"] = self._format_duration(minutes_remaining * 60)

    def _calculate_embedding_rate(self, embeddings_created: int) -> float:
        """Calculate embedding creation rate."""
        elapsed = (datetime.now() - self._start_time).total_seconds()
        return (embeddings_created / elapsed * 60) if elapsed > 0 else 0

    def _calculate_storage_rate(self, chunks_stored: int) -> float:
        """Calculate storage rate.""" 
        elapsed = (datetime.now() - self._start_time).total_seconds()
        return (chunks_stored / elapsed * 60) if elapsed > 0 else 0

    async def _heartbeat_loop(self):
        """Background task to send heartbeat updates during long operations."""
        while self.state.get("status") not in ["completed", "error"]:
            try:
                await asyncio.sleep(30)  # Send heartbeat every 30 seconds
                
                # Update heartbeat timestamp
                self.state["heartbeat"]["lastUpdate"] = datetime.now().isoformat()
                
                # Emit heartbeat if no progress update in last 30 seconds
                last_update = datetime.fromisoformat(self.state.get("timestamp", self.state["startTime"]))
                if (datetime.now() - last_update).total_seconds() > 25:
                    await self._emit_heartbeat()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                safe_logfire_error(f"Heartbeat error: {e}")

    async def _emit_heartbeat(self):
        """Emit a heartbeat event to show system is alive."""
        heartbeat_data = {
            "progressId": self.progress_id,
            "type": "heartbeat",
            "timestamp": datetime.now().isoformat(),
            "currentOperation": self.state["detailedProgress"]["currentOperation"],
            "status": self.state["status"],
            "isAlive": True
        }
        
        event_name = f"{self.operation_type}_heartbeat"
        await self.sio.emit(event_name, heartbeat_data, room=self.progress_id)
        
        safe_logfire_info(f"📢 [HEARTBEAT] {event_name} sent for {self.progress_id}")

    async def detect_stall(self, stall_threshold_seconds: int = 120) -> bool:
        """
        Detect if progress has stalled.
        
        Args:
            stall_threshold_seconds: Seconds without progress to consider a stall
            
        Returns:
            bool: True if stalled, False otherwise
        """
        if "timestamp" not in self.state:
            return False
            
        last_update = datetime.fromisoformat(self.state["timestamp"])
        elapsed = (datetime.now() - last_update).total_seconds()
        
        is_stalled = elapsed > stall_threshold_seconds and self.state["status"] not in ["completed", "error"]
        
        if is_stalled:
            await self._emit_stall_alert(elapsed)
            
        return is_stalled

    async def _emit_stall_alert(self, stall_duration: float):
        """Emit a stall detection alert."""
        alert_data = {
            "progressId": self.progress_id,
            "type": "stall_detected",
            "timestamp": datetime.now().isoformat(),
            "stallDuration": stall_duration,
            "stallDurationFormatted": self._format_duration(stall_duration),
            "currentOperation": self.state["detailedProgress"]["currentOperation"],
            "status": self.state["status"],
            "lastProgress": self.state.get("percentage", 0)
        }
        
        event_name = f"{self.operation_type}_stall_alert"
        await self.sio.emit(event_name, alert_data, room=self.progress_id)
        
        safe_logfire_info(f"🚨 [STALL ALERT] {event_name} sent for {self.progress_id} | Duration: {self._format_duration(stall_duration)}")
