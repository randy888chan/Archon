from backend.dependencies import get_supabase_client  # For clear function
from backend.models import CrawlStatusResponse
from crawl_ai_docs import (
    CrawlProgressTracker,
    run_crawl_task,
    clear_existing_records as clear_records_sync,
    get_langchain_python_docs_urls  # Example source for clearing
)
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import logging
import sys
import os
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
import asyncio
from pydantic import BaseModel, Field
from enum import Enum
import redis.asyncio as redis
from datetime import datetime, timezone
from pinecone_migration import (
    run_migration_to_pinecone,
    REDIS_MIGRATION_KEY_STATUS,
    REDIS_MIGRATION_KEY_LOGS,
    REDIS_MIGRATION_KEY_ERRORS,
    REDIS_MIGRATION_KEY_RUNNING_FLAG
)
import sqlite3

import logfire
import httpx
from fastapi import FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize clients, setup resources if needed
    logger.info("Backend starting up...")
    # Ensure Supabase client is available for clear endpoint
    supabase_client = get_supabase_client()
    if not supabase_client:
        logger.warning("Supabase client could not be initialized on startup.")
    app.state.supabase_client = supabase_client  # Store if needed

    # Initialize Redis connection pool
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", 6379))
    try:
        pool = redis.ConnectionPool.from_url(
            f"redis://{redis_host}:{redis_port}", decode_responses=True)  # Decode responses globally
        redis_client = redis.Redis.from_pool(pool)
        await redis_client.ping()  # Check connection
        app.state.redis = redis_client
        logger.info(f"Connected to Redis at {redis_host}:{redis_port}")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}", exc_info=True)
        app.state.redis = None  # Set to None if connection fails

    yield
    # Shutdown: Cleanup resources
    logger.info("Backend shutting down...")
    thread_pool.shutdown(wait=True)
    logger.info("Thread pool shut down.")
    if hasattr(app.state, 'redis') and app.state.redis:
        await app.state.redis.close()
        logger.info("Redis connection closed.")


app = FastAPI(title="Archon Crawler Backend", lifespan=lifespan)

# Basic CORS setup (adjust origins as needed for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Placeholder for shared state (single tracker for simplicity initially)
# In a multi-worker setup, this needs a more robust shared memory solution
# or external store like Redis.


logfire.configure(token=os.getenv('LOGFIRE_API_KEY'))
logfire.instrument_fastapi(app, capture_headers=True)
logfire.instrument_sqlite3()

# sqlite database URL
db_url = 'https://files.pydantic.run/pydantic_pypi.db'

# download the data and create the database
with logfire.span('preparing database'):
    with logfire.span('downloading data'):
        r = httpx.get(db_url)
        r.raise_for_status()

    with logfire.span('create database'):
        with open('pydantic_pypi.db', 'wb') as f:
            f.write(r.content)
        connection = sqlite3.connect('pydantic_pypi.db')


# Add project root to sys.path to allow importing 'archon' module
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now import from archon module

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thread pool for running synchronous tasks
# This might be overly simple; consider libraries like `starlette.concurrency.run_in_threadpool`
# or managing the pool lifecycle more carefully.
thread_pool = ThreadPoolExecutor(max_workers=4)

# Pydantic model for the request body of /crawl/start


class CrawlStartRequest(BaseModel):
    process_only_new: bool = False


# Define allowed sources using Enum (matching frontend KNOWN_SOURCES keys)
class AllowedSources(str, Enum):
    pydantic_ai = "pydantic_ai_docs"
    langgraph = "langgraph_docs"
    langgraphjs = "langgraphjs_docs"
    langsmith = "langsmith_docs"
    langchain_python = "langchain_python_docs"
    langchain_js = "langchain_js_docs"


# Define request body model for clearing documents
class ClearRequest(BaseModel):
    source: AllowedSources = Field(...,
                                   description="The specific documentation source to clear")


# Define a model for Pinecone migration status
class MigrationStatusResponse(BaseModel):
    """Status response for the Pinecone migration process."""
    is_running: bool = False
    # "initializing", "running", "completed", "completed_with_errors", "failed"
    status: str = "unknown"
    message: str = "No migration status available."
    progress: int = 0  # 0-100 percent
    start_time: str = ""
    end_time: str = ""
    duration_seconds: float = 0.0
    rows_processed: int = 0
    vectors_upserted: int = 0
    rows_skipped_embedding: int = 0
    rows_skipped_other: int = 0
    errors_encountered: int = 0
    logs: list[str] = []
    errors: list[str] = []


@app.get("/")
async def read_root():
    return {"message": "Archon Crawler Backend"}

# Constants for Redis keys
REDIS_KEY_STATUS = "crawl:status"
REDIS_KEY_LOGS = "crawl:logs"
REDIS_KEY_ERRORS = "crawl:errors"
REDIS_KEY_RUNNING_FLAG = "crawl:running"
# Check status hash if flag older than 1 hour
STALE_FLAG_TIMEOUT_SECONDS = 3600
# Auto-expire flag after 2 hours as safety net
RUNNING_FLAG_TTL_SECONDS = 7200


@app.post("/crawl/start", status_code=202)
async def start_crawl(background_tasks: BackgroundTasks, request: Request, payload: CrawlStartRequest):
    """Starts the documentation crawling process in the background using Redis for state."""
    redis_client: redis.Redis = request.app.state.redis
    logger.info(f"Redis client: {redis_client}")

    logger.info("Entering /crawl/start endpoint.")

    # Check for existing running flag
    flag_exists = await redis_client.get(REDIS_KEY_RUNNING_FLAG)
    logger.info(f"Flag exists: {flag_exists}")

    if flag_exists:
        logger.warning(
            "Running flag '%s' exists. Checking status for staleness...", REDIS_KEY_RUNNING_FLAG)
        try:
            status_data = await redis_client.hgetall(REDIS_KEY_STATUS)
            is_running_in_status = status_data.get("is_running") == "1"
            start_time_str = status_data.get("start_time")
            is_stale = False

            if not is_running_in_status:
                logger.warning(
                    "Flag '%s' exists, but status hash indicates not running. Clearing stale flag.", REDIS_KEY_RUNNING_FLAG)
                is_stale = True
            elif start_time_str:
                try:
                    start_time = datetime.fromisoformat(start_time_str)
                    if (datetime.now(timezone.utc) - start_time).total_seconds() > STALE_FLAG_TIMEOUT_SECONDS:
                        logger.warning("Flag '%s' exists, status is running, but start time > %d seconds ago (status check). Clearing stale flag.",
                                       REDIS_KEY_RUNNING_FLAG, STALE_FLAG_TIMEOUT_SECONDS)
                        is_stale = True
                except ValueError:
                    logger.warning(
                        "Could not parse start_time '%s' from status while checking flag staleness. Assuming stale.", start_time_str)
                    is_stale = True  # Treat unparseable time as potentially stale

            if is_stale:
                # Clear the stale flag
                await redis_client.delete(REDIS_KEY_RUNNING_FLAG)
                logger.info(
                    "Proceeding to start new crawl after clearing stale flag based on status check.")
            else:
                logger.warning(
                    "Attempted to start crawl while another is running (Redis flag and status confirmed).")
                raise HTTPException(
                    status_code=409, detail="A crawl process is already running.")

        except redis.RedisError as e:
            logger.error(
                f"Redis error while checking stale flag: {e}. Preventing start.", exc_info=True)
            raise HTTPException(
                status_code=503, detail=f"Could not verify running status due to Redis error: {e}")
        except HTTPException:
            # Pass through HTTP exceptions (like 409 Conflict) directly without wrapping
            raise
        except Exception as e_check:
            logger.error(
                f"Unexpected error while checking stale flag: {e_check}. Preventing start.", exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"Could not verify running status due to unexpected error: {e_check}")
    else:
        logger.info(
            "No existing running flag found. Proceeding to start new crawl.")
        # Continue with normal execution - no need to raise an exception here

    # --- Proceed with starting the crawl ---
    logger.info("Proceeding to initialize crawl state in Redis.")

    # Initialize status in Redis
    start_time_iso = datetime.now(timezone.utc).isoformat()
    initial_status = {
        "is_running": "1",
        "processed_count": "0",
        "total_urls": "0",
        "urls_succeeded": "0",
        "urls_failed": "0",
        "urls_skipped": "0",
        "chunks_stored": "0",
        "start_time": start_time_iso,
        "end_time": "",
        "current_url": "",
        "duration_seconds": "0.0",
        "message": "Crawl initiated..."
    }

    try:
        async with redis_client.pipeline(transaction=True) as pipe:
            await pipe.delete(REDIS_KEY_STATUS, REDIS_KEY_LOGS, REDIS_KEY_ERRORS)
            await pipe.hset(REDIS_KEY_STATUS, mapping=initial_status)
            await pipe.set(REDIS_KEY_RUNNING_FLAG, "1")
            await pipe.expire(REDIS_KEY_RUNNING_FLAG, RUNNING_FLAG_TTL_SECONDS)
            await pipe.execute()
        logger.info(
            f"Redis initialization complete. Set flag '{REDIS_KEY_RUNNING_FLAG}' with TTL {RUNNING_FLAG_TTL_SECONDS}s.")
    except redis.RedisError as e_init:
        logger.error(
            f"Redis error during crawl start initialization: {e_init}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to initialize crawl state in Redis: {e_init}")
    except Exception as e_init_other:
        logger.error(
            f"Unexpected error during crawl start initialization: {e_init_other}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Unexpected error initializing crawl state: {e_init_other}")

    # --- Create Tracker ---
    logger.info("Creating tracker instance.")
    try:
        tracker = CrawlProgressTracker()
        logger.info("Tracker created successfully.")
    except Exception as e_tracker:
        logger.error(
            f"Failed to create CrawlProgressTracker: {e_tracker}", exc_info=True)
        # No need to roll back Redis here as the state reflects "initiated" but task hasn't started
        raise HTTPException(
            status_code=500, detail=f"Internal error: Failed to initialize tracker state: {e_tracker}")

    # --- Add Background Task ---
    logger.info(
        f"Adding run_crawl_task to background tasks (process_only_new={payload.process_only_new}).")
    try:
        # Ensure run_crawl_task is correctly imported at the top
        from crawl_ai_docs import run_crawl_task

        background_tasks.add_task(
            run_crawl_task, tracker, payload.process_only_new, redis_client)
        logger.info("Successfully added task to background.")
    except ImportError as e_import:
        logger.error(
            f"Failed to import run_crawl_task: {e_import}", exc_info=True)
        # Attempt rollback
        await _rollback_redis_state(redis_client, "Import error prevented task queuing")
        raise HTTPException(
            status_code=500, detail=f"Internal error: Failed to import crawl task: {e_import}")
    except Exception as e_add_task:
        logger.error(
            f"Failed to add task to background: {e_add_task}", exc_info=True)
        # Attempt rollback
        await _rollback_redis_state(redis_client, f"Error adding task: {e_add_task}")
        raise HTTPException(
            status_code=500, detail=f"Internal error: Failed to queue crawl task: {e_add_task}")

    logger.info("Returning 202 Accepted response from /crawl/start.")
    return {"message": "Crawl process initiated."}


async def _rollback_redis_state(redis_client: redis.Redis, reason: str):
    """Helper to attempt rolling back Redis state on task queuing failure."""
    if not redis_client:
        return
    try:
        logger.warning(f"Rolling back Redis state because: {reason}")
        # Delete the flag first
        await redis_client.delete(REDIS_KEY_RUNNING_FLAG)
        # Update status to reflect failure
        await redis_client.hset(REDIS_KEY_STATUS, mapping={
            "is_running": "0",
            "message": f"Crawl aborted: {reason}",
            "end_time": datetime.now(timezone.utc).isoformat(),
            "urls_failed": "1"  # Indicate failure generically
        })
        logger.warning("Redis state rolled back.")
    except Exception as e_rollback:
        logger.error(
            f"Failed to roll back Redis state: {e_rollback}", exc_info=True)
        # Log error but don't mask the original exception that led here


@app.get("/crawl/status", response_model=CrawlStatusResponse)
async def get_crawl_status(request: Request) -> CrawlStatusResponse:
    """Gets the status of the current or last crawl process from Redis."""
    redis_client: redis.Redis = request.app.state.redis

    try:
        # Fetch status hash, logs, and errors in parallel
        async with redis_client.pipeline(transaction=False) as pipe:
            pipe.hgetall(REDIS_KEY_STATUS)
            pipe.lrange(REDIS_KEY_LOGS, -50, -1)  # Get last 50 logs
            pipe.lrange(REDIS_KEY_ERRORS, -50, -1)  # Get last 50 errors
            results = await pipe.execute()

        status_dict_raw, logs_raw, errors_raw = results

        if not status_dict_raw:
            logger.info(
                "Status requested, but no crawl status found in Redis (key '%s'). Returning default.", REDIS_KEY_STATUS)
            # Return a default status indicating no crawl has run yet
            return CrawlStatusResponse(message="No crawl status found in Redis.", is_running=False)

        # Process the raw data (decode_responses=True handles decoding)
        status_dict = status_dict_raw
        logs = logs_raw
        errors = errors_raw

        # Helper to safely convert/get values
        def safe_get(data, key, type_func=str, default=None):
            val = data.get(key)
            if val is None or val == "":
                return default
            try:
                # Special handling for bool stored as "1" or "0"
                if type_func == bool:
                    return val == "1"
                return type_func(val)
            except (ValueError, TypeError):
                logger.warning(
                    f"Could not convert Redis value '{val}' for key '{key}' to type {type_func.__name__}. Using default: {default}", exc_info=True)
                return default

        # Determine if running based *only* on the 'is_running' field in the status hash
        is_running = safe_get(status_dict, "is_running", bool, False)

        start_time = safe_get(status_dict, "start_time", str, "")
        # Ensure default empty string
        end_time = safe_get(status_dict, "end_time", str, "")

        # Calculate duration if running or finished
        duration = None
        if start_time:
            try:
                # Ensure timezone info is handled correctly (datetime.fromisoformat handles TZ)
                start_dt = datetime.fromisoformat(start_time)
                if end_time:
                    end_dt = datetime.fromisoformat(end_time)
                    duration = (end_dt - start_dt).total_seconds()
                elif is_running:  # Calculate ongoing duration only if actively running
                    duration = (datetime.now(timezone.utc) -
                                start_dt).total_seconds()
            except ValueError:
                logger.warning(
                    f"Could not parse start_time ('{start_time}') or end_time ('{end_time}') as ISO format datetime.", exc_info=True)
                duration = None  # Set duration to None if time parsing fails

        # Adapt the dictionary to the response model fields
        response = CrawlStatusResponse(
            message=status_dict.get("message", "Crawl status retrieved."),
            is_running=is_running,
            processed_count=safe_get(status_dict, "processed_count", int, 0),
            total_urls=safe_get(status_dict, "total_urls", int, 0),
            current_url=safe_get(status_dict, "current_url"),
            errors=errors,
            logs=logs,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            urls_succeeded=safe_get(status_dict, "urls_succeeded", int, 0),
            urls_failed=safe_get(status_dict, "urls_failed", int, 0),
            urls_skipped=safe_get(status_dict, "urls_skipped", int, 0),
        )

        return response

    except redis.RedisError as e:  # Catch specific Redis errors
        logger.error(f"Redis error fetching crawl status: {e}", exc_info=True)
        # Return a default "not running" status on Redis communication error
        return CrawlStatusResponse(message=f"Redis error retrieving status: {e}", is_running=False)
    # Catch other unexpected errors (like AttributeError if redis_client is None)
    except Exception as e:
        logger.error(
            f"Unexpected error fetching crawl status: {e}", exc_info=True)
        # Return a default "not running" status on unexpected errors
        # This now also handles the case where redis_client was None
        return CrawlStatusResponse(message=f"Unexpected error retrieving status: {e}", is_running=False)


@app.post("/crawl/clear")
async def clear_documents(
    request_data: ClearRequest,  # Use the Pydantic model to validate input
    request: Request  # Keep access to request if needed for state etc.
):
    """Clears existing documents for a specific source from the vector store."""
    logger.info(
        f"Received request to clear documents for source: {request_data.source.value}")

    # Get source from validated request data
    source_to_clear = request_data.source.value

    try:
        # Assuming clear_records_sync can run synchronously and handles its own DB connection/client
        # It's imported from crawl_ai_docs which likely uses the Supabase client implicitly or explicitly

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            thread_pool,
            clear_records_sync,  # Pass the synchronous function from crawl_ai_docs
            source_to_clear  # Pass the specific source from the request
        )
        logger.info(
            f"Successfully cleared documents for source: {source_to_clear}")
        return {"message": f"Documents cleared successfully for source: {source_to_clear}"}

    except Exception as e:
        logger.error(
            f"Error clearing documents for {source_to_clear}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to clear documents for {source_to_clear}: {str(e)}")


@app.post("/migrate-to-pinecone", status_code=202)
async def start_pinecone_migration(background_tasks: BackgroundTasks, request: Request):
    """
    Starts the Supabase to Pinecone data migration process in the background.
    Uses Redis for status tracking.
    """
    logger.info("Received request to start Supabase to Pinecone migration.")
    redis_client: redis.Redis = request.app.state.redis

    # Check if migration is already running
    flag_exists = await redis_client.get(REDIS_MIGRATION_KEY_RUNNING_FLAG)

    if flag_exists:
        logger.warning("Migration is already running (flag exists).")
        # Check the status to see if it's stale
        try:
            status_data = await redis_client.hgetall(REDIS_MIGRATION_KEY_STATUS)
            is_running_in_status = status_data.get("is_running") == "1"
            start_time_str = status_data.get("start_time")
            is_stale = False

            if not is_running_in_status:
                logger.warning(
                    "Migration flag exists but status says not running. Flag may be stale.")
                is_stale = True
            elif start_time_str:
                try:
                    start_time = datetime.fromisoformat(start_time_str)
                    if (datetime.now(timezone.utc) - start_time).total_seconds() > STALE_FLAG_TIMEOUT_SECONDS:
                        logger.warning(
                            f"Migration flag exists but start time > {STALE_FLAG_TIMEOUT_SECONDS} seconds ago. Flag may be stale.")
                        is_stale = True
                except ValueError:
                    logger.warning(
                        f"Could not parse migration start_time '{start_time_str}'. Assuming stale.")
                    is_stale = True

            if is_stale:
                # Clear the stale flag
                await redis_client.delete(REDIS_MIGRATION_KEY_RUNNING_FLAG)
                logger.info(
                    "Stale migration flag cleared. Will proceed with new migration.")
            else:
                # Migration is genuinely running
                logger.warning(
                    "Migration is already running. Cannot start another.")
                # Raise the 409 exception, which should now propagate correctly
                raise HTTPException(
                    status_code=409,
                    detail="A migration process is already running."
                )
        except Exception as e:
            # Check if the exception is the specific 409 we raised intentionally
            if isinstance(e, HTTPException) and e.status_code == 409:
                raise e  # Re-raise the 409 exception
            else:
                # Handle other unexpected errors during the status check
                # Log full traceback for unexpected errors
                logger.error(
                    f"Error checking migration status: {e}", exc_info=True)
                raise HTTPException(
                    status_code=500,
                    detail=f"Could not verify migration status: {str(e)}"
                )

    # Start the migration task with redis_client for status tracking
    try:
        # Add the background task that now accepts redis_client
        background_tasks.add_task(run_migration_to_pinecone, redis_client)
        logger.info(
            "Supabase to Pinecone migration task added to background queue.")
        return {"status": "success", "message": "Supabase to Pinecone migration started in the background."}
    except Exception as e:
        logger.error(f"Error starting migration task: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start migration: {str(e)}"
        )


@app.get("/migrate/status", response_model=MigrationStatusResponse)
async def get_migration_status(request: Request) -> MigrationStatusResponse:
    """Gets the status of the current or last Pinecone migration process from Redis."""
    redis_client: redis.Redis = request.app.state.redis

    try:
        # Fetch status hash, logs, and errors in parallel
        async with redis_client.pipeline(transaction=False) as pipe:
            pipe.hgetall(REDIS_MIGRATION_KEY_STATUS)
            pipe.lrange(REDIS_MIGRATION_KEY_LOGS, -50, -1)  # Get last 50 logs
            pipe.lrange(REDIS_MIGRATION_KEY_ERRORS, -
                        50, -1)  # Get last 50 errors
            results = await pipe.execute()

        status_dict, logs, errors = results

        if not status_dict:
            logger.info("No migration status found in Redis.")
            return MigrationStatusResponse(
                message="No migration has been initiated yet.",
                is_running=False
            )

        # Helper function to safely convert Redis string values to Python types
        def safe_get(data, key, type_func=str, default=None):
            val = data.get(key)
            if val is None or val == "":
                return default
            try:
                # Special handling for bool stored as "1" or "0"
                if type_func == bool:
                    return val == "1"
        # Ensure default empty string
            except (ValueError, TypeError):
                logger.warning(
                    f"Could not convert migration Redis value '{val}' for key '{key}' to {type_func.__name__}")
                return default

        # Build the response
        is_running = safe_get(status_dict, "is_running", bool, False)

        # Calculate duration
        start_time = safe_get(status_dict, "start_time", str, "")
        # Ensure default empty string
        end_time = safe_get(status_dict, "end_time", str, "")
        duration = None

        if start_time:
            try:
                start_dt = datetime.fromisoformat(start_time)
                if end_time:
                    end_dt = datetime.fromisoformat(end_time)
                    duration = (end_dt - start_dt).total_seconds()
                elif is_running:
                    duration = (datetime.now(timezone.utc) -
                                start_dt).total_seconds()
            except ValueError:
                logger.warning(
                    f"Could not parse migration time values: start='{start_time}', end='{end_time}'")

        # Check if we need to manually verify running state via flag
        if is_running:
            # Double-check with the flag to avoid stale running state
            flag_exists = await redis_client.exists(REDIS_MIGRATION_KEY_RUNNING_FLAG)
            if not flag_exists:
                logger.warning(
                    "Migration status says running but flag doesn't exist. Status may be stale.")
                is_running = False
                # Update the status record if we can
                try:
                    await redis_client.hset(REDIS_MIGRATION_KEY_STATUS, "is_running", "0")
                except Exception as e:
                    logger.error(
                        f"Failed to update stale migration running state: {e}")

        return MigrationStatusResponse(
            is_running=is_running,
            status=safe_get(status_dict, "status", str, "unknown"),
            message=safe_get(status_dict, "message", str,
                             "Migration status retrieved."),
            progress=safe_get(status_dict, "progress", int, 0),
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration if duration is not None else safe_get(
                status_dict, "duration_seconds", float, 0.0),
            rows_processed=safe_get(status_dict, "rows_processed", int, 0),
            vectors_upserted=safe_get(status_dict, "vectors_upserted", int, 0),
            rows_skipped_embedding=safe_get(
                status_dict, "rows_skipped_embedding", int, 0),
            rows_skipped_other=safe_get(
                status_dict, "rows_skipped_other", int, 0),
            errors_encountered=safe_get(
                status_dict, "errors_encountered", int, 0),
            logs=logs,
            errors=errors
        )

    except redis.RedisError as e:
        logger.error(
            f"Redis error fetching migration status: {e}", exc_info=True)
        return MigrationStatusResponse(
            message=f"Redis error retrieving migration status: {str(e)}",
            is_running=False
        )
    except Exception as e:
        logger.error(
            f"Unexpected error fetching migration status: {e}", exc_info=True)
        return MigrationStatusResponse(
            message=f"Unexpected error retrieving migration status: {str(e)}",
            is_running=False
        )
