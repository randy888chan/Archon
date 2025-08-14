"""
Enhanced Progress Mapper for Background Tasks

Maps sub-task progress (0-100%) to overall task progress ranges with enhanced granularity.
Provides detailed stage tracking, heartbeat events, and stall detection.
This ensures smooth progress reporting without jumping backwards and eliminates user confusion.
"""

import time


class ProgressMapper:
    """Maps sub-task progress to overall progress ranges with enhanced granularity"""

    # Enhanced progress ranges for more granular tracking
    STAGE_RANGES = {
        "starting": (0, 2),
        "analyzing": (2, 8),
        "crawling": (8, 25),
        "processing": (25, 30),
        # Enhanced document storage with substages
        "document_storage": (30, 70),
        "document_storage.parsing": (30, 35),
        "document_storage.chunking": (35, 45),
        "document_storage.embedding": (45, 65),
        "document_storage.storing": (65, 70),
        # Enhanced code extraction with substages  
        "code_extraction": (70, 85),
        "extracting": (70, 85),  # Alias for code_extraction
        "code_extraction.scanning": (70, 75),
        "code_extraction.processing": (75, 80),
        "code_extraction.indexing": (80, 85),
        "finalization": (85, 98),
        "finalization.cleanup": (85, 90),
        "finalization.validation": (90, 95),
        "finalization.completion": (95, 98),
        "completed": (98, 100),
        "complete": (98, 100),  # Alias
        "error": (-1, -1),  # Special case for errors
    }

    # Heartbeat configuration
    HEARTBEAT_INTERVAL = 5.0  # Send heartbeat every 5 seconds during processing
    STALL_DETECTION_TIMEOUT = 30.0  # Detect stall if no progress for 30 seconds

    def __init__(self):
        """Initialize the progress mapper with enhanced tracking"""
        self.last_overall_progress = 0
        self.current_stage = "starting"
        self.current_substage = None
        self.last_update_time = time.time()
        self.heartbeat_counter = 0
        self.processing_rate = 0.0
        self.items_processed = 0
        self.total_items = 0

    def map_progress(self, stage: str, stage_progress: float, **kwargs) -> dict:
        """
        Map stage-specific progress to overall progress with enhanced metadata.

        Args:
            stage: The current stage name (can include substage with dot notation)
            stage_progress: Progress within the stage (0-100)
            **kwargs: Additional metadata (items_processed, total_items, etc.)

        Returns:
            Dict with overall progress and enhanced metadata
        """
        current_time = time.time()
        
        # Handle error state
        if stage == "error":
            return {
                "percentage": -1,
                "stage": stage,
                "substage": None,
                "status": "error",
                "processing_rate": 0.0,
                "last_update": current_time
            }

        # Parse stage and substage
        if "." in stage:
            main_stage, substage = stage.split(".", 1)
            self.current_substage = substage
        else:
            main_stage = stage
            self.current_substage = None

        # Get stage range
        range_key = stage if stage in self.STAGE_RANGES else main_stage
        if range_key not in self.STAGE_RANGES:
            # Unknown stage - use current progress
            overall_progress = self.last_overall_progress
        else:
            start, end = self.STAGE_RANGES[range_key]
            
            # Handle completion
            if main_stage in ["completed", "complete"]:
                overall_progress = 100
            else:
                # Calculate mapped progress
                stage_progress = max(0, min(100, stage_progress))  # Clamp to 0-100
                stage_range = end - start
                mapped_progress = start + (stage_progress / 100.0) * stage_range
                
                # Ensure progress never goes backwards
                overall_progress = max(self.last_overall_progress, mapped_progress)

        # Update processing rate calculation
        if "items_processed" in kwargs and "total_items" in kwargs:
            self.items_processed = kwargs["items_processed"]
            self.total_items = kwargs["total_items"]
            
            # Calculate processing rate (items per second)
            time_delta = current_time - self.last_update_time
            if time_delta > 0 and self.last_overall_progress > 0:
                progress_delta = overall_progress - self.last_overall_progress
                if progress_delta > 0:
                    # Estimate processing rate based on progress
                    self.processing_rate = progress_delta / time_delta

        # Update state
        self.last_overall_progress = int(round(overall_progress))
        self.current_stage = main_stage
        self.last_update_time = current_time
        self.heartbeat_counter += 1

        # Detect stalls
        stall_detected = (current_time - self.last_update_time) > self.STALL_DETECTION_TIMEOUT

        return {
            "percentage": self.last_overall_progress,
            "stage": main_stage,
            "substage": self.current_substage,
            "status": "stalled" if stall_detected else "processing",
            "processing_rate": round(self.processing_rate, 2),
            "items_processed": self.items_processed,
            "total_items": self.total_items,
            "heartbeat_id": self.heartbeat_counter,
            "last_update": current_time,
            "time_since_update": current_time - self.last_update_time,
            **kwargs
        }

    def get_stage_range(self, stage: str) -> tuple:
        """Get the progress range for a stage"""
        return self.STAGE_RANGES.get(stage, (0, 100))

    def calculate_stage_progress(self, current_value: int, max_value: int) -> float:
        """
        Calculate percentage progress within a stage.

        Args:
            current_value: Current progress value (e.g., processed items)
            max_value: Maximum value (e.g., total items)

        Returns:
            Progress percentage within stage (0-100)
        """
        if max_value <= 0:
            return 0.0

        return (current_value / max_value) * 100.0

    def map_batch_progress(self, stage: str, current_batch: int, total_batches: int, **kwargs) -> dict:
        """
        Convenience method for mapping batch processing progress.

        Args:
            stage: The current stage name
            current_batch: Current batch number (1-based)
            total_batches: Total number of batches
            **kwargs: Additional metadata

        Returns:
            Progress dict with batch information
        """
        if total_batches <= 0:
            return self.map_progress(stage, 0, **kwargs)

        # Calculate stage progress (0-based for calculation)
        stage_progress = ((current_batch - 1) / total_batches) * 100.0
        
        return self.map_progress(stage, stage_progress, 
                               current_batch=current_batch,
                               total_batches=total_batches,
                               **kwargs)

    def map_with_substage(self, stage: str, substage: str, stage_progress: float, **kwargs) -> dict:
        """
        Map progress with substage information for finer control.

        Args:
            stage: Main stage (e.g., 'document_storage')
            substage: Substage (e.g., 'embeddings', 'chunking')
            stage_progress: Progress within the substage
            **kwargs: Additional metadata

        Returns:
            Progress dict with substage information
        """
        full_stage = f"{stage}.{substage}"
        return self.map_progress(full_stage, stage_progress, **kwargs)

    def should_send_heartbeat(self) -> bool:
        """Check if a heartbeat should be sent based on time elapsed."""
        current_time = time.time()
        return (current_time - self.last_update_time) >= self.HEARTBEAT_INTERVAL

    def generate_heartbeat(self) -> dict:
        """Generate a heartbeat event to show system is alive."""
        current_time = time.time()
        return {
            "type": "heartbeat",
            "percentage": self.last_overall_progress,
            "stage": self.current_stage,
            "substage": self.current_substage,
            "status": "processing",
            "heartbeat_id": self.heartbeat_counter,
            "timestamp": current_time,
            "processing_rate": self.processing_rate,
            "uptime": current_time - (self.last_update_time - self.heartbeat_counter * self.HEARTBEAT_INTERVAL)
        }

    def detect_stall(self) -> bool:
        """Detect if processing appears to have stalled."""
        current_time = time.time()
        return (current_time - self.last_update_time) > self.STALL_DETECTION_TIMEOUT

    def reset(self):
        """Reset the mapper to initial state"""
        self.last_overall_progress = 0
        self.current_stage = "starting"
        self.current_substage = None
        self.last_update_time = time.time()
        self.heartbeat_counter = 0
        self.processing_rate = 0.0
        self.items_processed = 0
        self.total_items = 0

    def get_current_stage(self) -> str:
        """Get the current stage name"""
        return self.current_stage

    def get_current_substage(self) -> str:
        """Get the current substage name"""
        return self.current_substage

    def get_current_progress(self) -> int:
        """Get the current overall progress percentage"""
        return self.last_overall_progress

    def get_performance_metrics(self) -> dict:
        """Get current performance metrics."""
        return {
            "processing_rate": self.processing_rate,
            "items_processed": self.items_processed,
            "total_items": self.total_items,
            "completion_percentage": self.last_overall_progress,
            "stage": self.current_stage,
            "substage": self.current_substage,
            "heartbeat_count": self.heartbeat_counter
        }
