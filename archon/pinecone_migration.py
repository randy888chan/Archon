import os
import sys
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from supabase import create_client, Client
from typing import List, Dict, Any, Optional
import time
import asyncio
import redis.asyncio as redis  # Import async Redis
from datetime import datetime, timezone  # Added for timestamps
import json

# Ensure the utils directory is in the Python path
# Adjust the path depending on where rag.py is relative to the utils directory
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Now try importing from utils
try:
    from utils.utils import get_env_var, get_clients
except ImportError as e:
    print(f"Error importing from utils: {e}")
    # Define dummy functions or raise error if utils are critical

    def get_env_var(var_name, default=None):
        return os.environ.get(var_name, default)

    def get_clients():
        # Dummy implementation - replace with actual logic if needed without utils
        print("Warning: Using dummy get_clients(). Supabase/Embedding client might not be initialized.")
        # Attempt to initialize Supabase directly if possible
        supabase_url = get_env_var("SUPABASE_URL")
        supabase_key = get_env_var("SUPABASE_KEY")
        supabase_client = None
        if supabase_url and supabase_key:
            try:
                supabase_client = create_client(supabase_url, supabase_key)
            except Exception as supabase_e:
                print(
                    f"Failed to initialize dummy Supabase client: {supabase_e}")
        # Dummy embedding client - Pinecone migration doesn't use it directly
        return None, supabase_client


load_dotenv()

# --- Configuration ---
PINECONE_API_KEY = get_env_var("PINECONE_API_KEY")
PINECONE_INDEX_NAME = get_env_var("PINECONE_INDEX_NAME", "ai-docs-index")
SUPABASE_TABLE_NAME = "site_pages"
FETCH_CHUNK_SIZE = 1000  # Number of rows to fetch from Supabase at a time
PINECONE_UPSERT_BATCH_SIZE = 100  # Pinecone recommends batches of 100 for upserts
PINECONE_METRIC = "cosine"
VECTOR_DIMENSION = 1536  # OpenAI embedding dimension
PINECONE_CLOUD = get_env_var("PINECONE_CLOUD", "aws")
PINECONE_REGION = get_env_var("PINECONE_REGION", "us-east-1")

# --- Redis Keys for Migration Status ---
# Using 'migration:' prefix to distinguish from 'crawl:'
REDIS_MIGRATION_KEY_STATUS = "migration:status"
REDIS_MIGRATION_KEY_LOGS = "migration:logs"
REDIS_MIGRATION_KEY_ERRORS = "migration:errors"
REDIS_MIGRATION_KEY_RUNNING_FLAG = "migration:running"
# Safety TTL for the running flag
MIGRATION_RUNNING_FLAG_TTL_SECONDS = 7200  # 2 hours


async def _log_migration_message(redis_client: redis.Redis, message: str, is_error: bool = False):
    """Helper to log messages and errors to Redis."""
    if not redis_client:
        # Fallback to print
        print(f"Log ({'Error' if is_error else 'Info'}): {message}")
        return
    try:
        key = REDIS_MIGRATION_KEY_ERRORS if is_error else REDIS_MIGRATION_KEY_LOGS
        timestamp = datetime.now(timezone.utc).strftime(
            '%Y-%m-%d %H:%M:%S UTC')
        log_entry = f"[{timestamp}] {message}"
        async with redis_client.pipeline(transaction=False) as pipe:
            await pipe.lpush(key, log_entry)
            await pipe.ltrim(key, 0, 99)  # Keep last 100 entries
            await pipe.execute()
    except Exception as e:
        print(f"Redis logging failed: {e}")
        print(f"Original message: {message}")


async def _update_migration_status(redis_client: redis.Redis, updates: Dict[str, Any]):
    """Helper to update the migration status hash in Redis."""
    try:
        # Ensure all values are strings for Redis hash
        str_updates = {k: str(v) for k, v in updates.items()}
        await redis_client.hset(REDIS_MIGRATION_KEY_STATUS, mapping=str_updates)
    except Exception as e:
        print(f"Redis status update failed: {e}")
        print(f"Update data: {updates}")


def initialize_pinecone(api_key: str, index_name: str) -> Optional[Pinecone.Index]:
    """
    Initializes the Pinecone client and ensures the target index exists.
    (Synchronous function - should be run in executor)
    """
    if not api_key:
        print("Error: PINECONE_API_KEY environment variable not set.")
        api_key = PINECONE_API_KEY

    try:
        pc = Pinecone(api_key=api_key)

        # Get list of existing indexes
        existing_indexes = pc.list_indexes()
        index_exists = False

        # Check if index exists by comparing names
        if hasattr(existing_indexes, 'names') and isinstance(existing_indexes.names, list):
            # New Pinecone API style
            index_exists = index_name in existing_indexes.names
        else:
            # Fallback method - try to check using different API styles
            try:
                # Check if index exists by iterating through indexes
                index_exists = any(idx.name == index_name for idx in existing_indexes) if hasattr(
                    existing_indexes, '__iter__') else False
            except (AttributeError, TypeError):
                # If that doesn't work, try to access as dictionary or list
                try:
                    if isinstance(existing_indexes, dict) and 'indexes' in existing_indexes:
                        index_exists = any(
                            idx.get('name') == index_name for idx in existing_indexes['indexes'])
                    elif isinstance(existing_indexes, list):
                        index_exists = any(
                            idx.get('name') == index_name for idx in existing_indexes)
                except (AttributeError, TypeError):
                    print(
                        f"Warning: Could not determine if index '{index_name}' exists. Will attempt to create it.")
                    index_exists = False

        if not index_exists:
            print(f"Index '{index_name}' not found. Creating it...")
            # Log info to console, Redis logging happens in async wrapper
            pc.create_index(
                name=index_name,
                dimension=VECTOR_DIMENSION,
                metric=PINECONE_METRIC,
                spec=ServerlessSpec(
                    cloud=PINECONE_CLOUD,
                    region=PINECONE_REGION
                )
            )
            # Wait for index to be ready
            wait_start = time.time()
            while True:
                try:
                    index_status = pc.describe_index(index_name)
                    if hasattr(index_status, 'status') and index_status.status.get('ready', False):
                        break
                    print("Waiting for index to be ready...")
                    time.sleep(5)
                    if time.time() - wait_start > 300:  # 5 minute timeout
                        print(
                            "Error: Timeout waiting for Pinecone index to become ready.")
                        return None  # Indicate failure
                except Exception as e:
                    print(f"Error checking index status: {e}. Retrying...")
                    time.sleep(5)
                    if time.time() - wait_start > 300:  # 5 minute timeout
                        print(
                            "Error: Timeout waiting for Pinecone index to become ready.")
                        return None  # Indicate failure
            print(f"Index '{index_name}' created successfully.")
        else:
            print(f"Using existing index: '{index_name}'")

        return pc.Index(index_name)
    except Exception as e:
        print(f"Error initializing Pinecone or creating index: {e}")
        return None


def fetch_data_from_supabase(supabase: Client, table_name: str, chunk_size: int, last_id: int) -> List[Dict[str, Any]]:
    """
    Fetches data from the Supabase table in chunks.
    (Synchronous function - should be run in executor)
    """
    try:
        response = supabase.table(table_name)\
            .select("id, url, chunk_number, title, summary, content, metadata, embedding")\
            .gt("id", last_id)\
            .order("id", desc=False)\
            .limit(chunk_size)\
            .execute()

        # Changed from response.data to handle potential API variations or errors
        if hasattr(response, 'data') and response.data:
            return response.data
        elif hasattr(response, 'error') and response.error:
            print(f"Supabase fetch error: {response.error}")
            return []
        else:
            # No data and no explicit error
            return []
    except Exception as e:
        print(f"Error fetching data from Supabase: {e}")
        return []


async def migrate_to_pinecone(supabase: Client, pinecone_index: Pinecone.Index, redis_client: redis.Redis) -> Dict[str, Any]:
    """
    Fetches data from Supabase and upserts it into Pinecone by namespace. Reports status to Redis.

    Args:
        supabase: Initialized Supabase client.
        pinecone_index: Initialized Pinecone Index object.
        redis_client: Initialized async Redis client for status updates.

    Returns:
        A dictionary summarizing the migration results.
    """
    namespace_stats = {}
    last_id = 0
    total_rows_processed = 0
    total_vectors_upserted = 0
    total_rows_skipped_embedding = 0
    total_rows_skipped_other = 0
    errors_encountered = 0

    start_time = time.time()
    await _log_migration_message(redis_client, "Starting migration from Supabase to Pinecone...")
    await _update_migration_status(redis_client, {"status": "running", "progress": 0, "message": "Fetching initial data..."})

    while True:
        fetch_start_time = time.time()
        log_msg = f"Fetching next {FETCH_CHUNK_SIZE} rows from Supabase starting after ID {last_id}..."
        await _log_migration_message(redis_client, log_msg)
        print(log_msg)  # Also print to console

        # Run synchronous fetch in an executor to avoid blocking the async loop
        try:
            rows = await asyncio.to_thread(fetch_data_from_supabase, supabase, SUPABASE_TABLE_NAME, FETCH_CHUNK_SIZE, last_id)
        except Exception as fetch_e:
            err_msg = f"CRITICAL: Failed to fetch data from Supabase: {fetch_e}"
            await _log_migration_message(redis_client, err_msg, is_error=True)
            await _update_migration_status(redis_client, {"status": "failed", "message": err_msg})
            errors_encountered += 1
            break  # Stop migration on fetch failure

        fetch_duration = time.time() - fetch_start_time
        await _log_migration_message(redis_client, f"Fetched {len(rows)} rows in {fetch_duration:.2f} seconds.")

        if not rows:
            await _log_migration_message(redis_client, "No more data found in Supabase.")
            break

        print(f"Fetched {len(rows)} rows.")
        total_rows_processed += len(rows)
        await _update_migration_status(redis_client, {"rows_processed": total_rows_processed})

        vectors_by_namespace: Dict[str, List[tuple]] = {}
        rows_processed_in_batch = 0

        for row in rows:
            rows_processed_in_batch += 1
            try:
                namespace = row.get('metadata', {}).get('source', 'unknown')
                if not namespace:
                    namespace = 'unknown'

                # Generate a deterministic ID based on URL and chunk number
                url = row.get('url', '')
                chunk_num = row.get('chunk_number')
                if not url or chunk_num is None:
                    warn_msg = f"Warning: Skipping row id {row.get('id', 'N/A')} due to missing URL or chunk number."
                    # Log as error for visibility
                    await _log_migration_message(redis_client, warn_msg, is_error=True)
                    total_rows_skipped_other += 1
                    errors_encountered += 1
                    continue
                pinecone_id = f"{url}-{chunk_num}"

                pinecone_metadata = {
                    "url": url,
                    "chunk_number": chunk_num,
                    "title": row.get('title'),
                    "summary": row.get('summary'),
                    # Include content if needed, be mindful of size limits
                    "content": row.get('content'),
                    **(row.get('metadata') or {})  # Add original metadata
                }
                # Ensure metadata compatibility with Pinecone (basic types + lists of strings)
                pinecone_metadata_cleaned = {}
                for k, v in pinecone_metadata.items():
                    if isinstance(v, (str, bool, int, float)):
                        pinecone_metadata_cleaned[k] = v
                    elif isinstance(v, list) and all(isinstance(item, str) for item in v):
                        # Allow list of strings
                        pinecone_metadata_cleaned[k] = v
                    # else: skip other types

                embedding = row.get('embedding')
                # Handle embeddings that come as strings instead of lists
                if isinstance(embedding, str):
                    try:
                        # Add debug info for string format
                        print(
                            f"String embedding format sample for ID {row.get('id')}: {embedding[:50]}...")

                        # Try to load as JSON first in case it's a proper JSON string
                        try:
                            json_parsed = json.loads(embedding)
                            if isinstance(json_parsed, list):
                                print(
                                    f"Successfully parsed embedding as JSON array with {len(json_parsed)} elements")
                                embedding = json_parsed
                        except json.JSONDecodeError:
                            # Not JSON, continue with other parsing methods
                            pass

                        if isinstance(embedding, str):  # Still a string after JSON attempt
                            # Handle JSON-like string
                            if (embedding.startswith('[') and embedding.endswith(']')) or (embedding.startswith('{') and embedding.endswith('}')):
                                # Try to handle JSON-like strings by removing quotes and brackets
                                embedding = embedding.strip('[]{}').replace(
                                    ' ', '').replace('"', '').replace("'", '').split(',')
                                embedding = [float(x) for x in embedding if x]
                            # Handle Postgres vector format - possible alternative format from Supabase
                            elif embedding.startswith('(') and embedding.endswith(')'):
                                embedding = embedding.strip(
                                    '()').replace(' ', '').split(',')
                                embedding = [float(x) for x in embedding if x]
                            else:
                                # Handle plain string of numbers
                                embedding = embedding.strip().replace(' ', '').split(',')
                                embedding = [float(x) for x in embedding if x]

                        print(
                            f"Successfully parsed embedding into array with {len(embedding)} elements")
                    except Exception as e:
                        warn_msg = f"Warning: Failed to parse string embedding for row id {row['id']} (Pinecone ID {pinecone_id}): {e}"
                        print(
                            f"DEBUG - Raw embedding causing error: {embedding[:100]}...")
                        print(
                            f"DEBUG - Exception type: {type(e).__name__}, details: {str(e)}")
                        await _log_migration_message(redis_client, warn_msg, is_error=True)
                        total_rows_skipped_embedding += 1
                        errors_encountered += 1
                        continue
                if not isinstance(embedding, list) or not all(isinstance(x, (float, int)) for x in embedding):
                    warn_msg = f"Warning: Skipping row id {row['id']} (Pinecone ID {pinecone_id}) due to invalid embedding format. Type: {type(embedding)}"
                    await _log_migration_message(redis_client, warn_msg, is_error=True)
                    total_rows_skipped_embedding += 1
                    errors_encountered += 1
                    continue

                # Ensure all embedding values are floats
                embedding = [float(x) for x in embedding]

                if len(embedding) != VECTOR_DIMENSION:
                    warn_msg = f"Warning: Skipping row id {row['id']} (Pinecone ID {pinecone_id}) due to incorrect embedding dimension. Expected {VECTOR_DIMENSION}, Got {len(embedding)}"
                    await _log_migration_message(redis_client, warn_msg, is_error=True)
                    total_rows_skipped_embedding += 1
                    errors_encountered += 1
                    continue

                # Use cleaned metadata
                vector_tuple = (pinecone_id, embedding,
                                pinecone_metadata_cleaned)

                if namespace not in vectors_by_namespace:
                    vectors_by_namespace[namespace] = []
                vectors_by_namespace[namespace].append(vector_tuple)

            except Exception as e:
                err_msg = f"Error processing row id {row.get('id', 'N/A')}: {e}. Skipping row."
                await _log_migration_message(redis_client, err_msg, is_error=True)
                total_rows_skipped_other += 1
                errors_encountered += 1
                continue

            # Update progress roughly based on row processing within the batch
            progress_percent = min(99, int((total_rows_processed - len(rows) + rows_processed_in_batch) / (
                total_rows_processed + 1) * 100))  # Avoid division by zero, rough estimate
            if rows_processed_in_batch % 50 == 0:  # Update status periodically within batch
                await _update_migration_status(redis_client, {"progress": progress_percent, "message": f"Processing batch, row {rows_processed_in_batch}/{len(rows)}..."})

        batch_upsert_start_time = time.time()
        batch_vectors_upserted = 0
        for namespace, vectors in vectors_by_namespace.items():
            log_msg = f"Upserting {len(vectors)} vectors to namespace '{namespace}'..."
            await _log_migration_message(redis_client, log_msg)
            print(log_msg)
            upserted_count_ns = 0

            for i in range(0, len(vectors), PINECONE_UPSERT_BATCH_SIZE):
                batch = vectors[i:i + PINECONE_UPSERT_BATCH_SIZE]
                try:
                    # Run synchronous upsert in an executor
                    upsert_response = await asyncio.to_thread(pinecone_index.upsert, vectors=batch, namespace=namespace)

                    batch_upserted = upsert_response.upserted_count
                    if batch_upserted is None:
                        batch_upserted = 0  # Handle case where count might be None
                    log_msg_detail = f"  Batch {i // PINECONE_UPSERT_BATCH_SIZE + 1}: Upserted {batch_upserted} vectors to '{namespace}'."
                    await _log_migration_message(redis_client, log_msg_detail)
                    # print(log_msg_detail) # Optionally print successful batch details
                    upserted_count_ns += batch_upserted
                    batch_vectors_upserted += batch_upserted
                    total_vectors_upserted += batch_upserted

                    # Update status after each successful sub-batch upsert
                    await _update_migration_status(redis_client, {
                        "vectors_upserted": total_vectors_upserted,
                        "message": f"Upserting to {namespace}, batch {i // PINECONE_UPSERT_BATCH_SIZE + 1}..."
                    })

                except Exception as e:
                    err_msg = f"  Error upserting batch to namespace '{namespace}': {e}"
                    await _log_migration_message(redis_client, err_msg, is_error=True)
                    errors_encountered += 1  # Count upsert errors

            if namespace not in namespace_stats:
                namespace_stats[namespace] = 0
            namespace_stats[namespace] += upserted_count_ns
            log_msg = f"Finished upserting for namespace '{namespace}'. Total for namespace: {namespace_stats[namespace]}"
            await _log_migration_message(redis_client, log_msg)
            print(log_msg)

        batch_upsert_duration = time.time() - batch_upsert_start_time
        await _log_migration_message(redis_client, f"Upserted {batch_vectors_upserted} vectors across namespaces in {batch_upsert_duration:.2f} seconds for this batch.")

        if rows:
            # Important: Ensure last_id is updated correctly even if some rows were skipped
            last_id = rows[-1]['id']
        else:
            # This case should be caught by the 'if not rows:' check earlier
            break

        log_msg = f"Processed {total_rows_processed} rows so far. Total vectors upserted: {total_vectors_upserted}. Skipped (embed): {total_rows_skipped_embedding}. Skipped (other): {total_rows_skipped_other}."
        await _log_migration_message(redis_client, log_msg)
        print(log_msg)
        # Optional delay
        await asyncio.sleep(0.1)

    # --- Migration Finished ---
    end_time = time.time()
    duration = end_time - start_time
    final_status = "completed" if errors_encountered == 0 else "completed_with_errors"
    final_message = f"Migration {final_status}. Processed: {total_rows_processed}, Upserted: {total_vectors_upserted}, Skipped (Embed): {total_rows_skipped_embedding}, Skipped (Other): {total_rows_skipped_other}, Errors: {errors_encountered}."

    await _log_migration_message(redis_client, "--- Migration Summary ---")
    await _log_migration_message(redis_client, f"Status: {final_status}")
    await _log_migration_message(redis_client, f"Duration: {duration:.2f} seconds")
    await _log_migration_message(redis_client, f"Total rows processed from Supabase: {total_rows_processed}")
    await _log_migration_message(redis_client, f"Total vectors upserted to Pinecone: {total_vectors_upserted}")
    await _log_migration_message(redis_client, f"Total rows skipped (bad embedding): {total_rows_skipped_embedding}")
    await _log_migration_message(redis_client, f"Total rows skipped (other issues): {total_rows_skipped_other}")
    await _log_migration_message(redis_client, f"Total errors logged: {errors_encountered}")
    await _log_migration_message(redis_client, "Vectors upserted per namespace:")
    if namespace_stats:
        for ns, count in namespace_stats.items():
            await _log_migration_message(redis_client, f"- {ns}: {count}")
    else:
        await _log_migration_message(redis_client, "No vectors were upserted.")
    await _log_migration_message(redis_client, "-------------------------\n")

    await _update_migration_status(redis_client, {
        "status": final_status,
        "message": final_message,
        "end_time": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(duration, 2),
        "progress": 100,
        "rows_processed": total_rows_processed,
        "vectors_upserted": total_vectors_upserted,
        "rows_skipped_embedding": total_rows_skipped_embedding,
        "rows_skipped_other": total_rows_skipped_other,
        "errors_encountered": errors_encountered,
        "is_running": "0"  # Explicitly mark as not running
    })

    print(final_message)

    # Return summary dictionary
    return {
        "status": final_status,
        "duration_seconds": duration,
        "rows_processed": total_rows_processed,
        "vectors_upserted": total_vectors_upserted,
        "rows_skipped_embedding": total_rows_skipped_embedding,
        "rows_skipped_other": total_rows_skipped_other,
        "errors_encountered": errors_encountered,
        "namespace_stats": namespace_stats
    }


async def run_migration_to_pinecone(redis_client: redis.Redis):
    """
    Main async function to run the migration process.
    Initializes clients and calls the migration function, handling Redis status.
    """
    # Add debug logging at the very start
    print("Starting run_migration_to_pinecone function")

    start_time_iso = datetime.now(timezone.utc).isoformat()
    initial_status = {
        "is_running": "1",
        "status": "initializing",
        "message": "Initializing migration clients...",
        "start_time": start_time_iso,
        "end_time": "",
        "duration_seconds": "0.0",
        "progress": "0",
        "rows_processed": "0",
        "vectors_upserted": "0",
        "rows_skipped_embedding": "0",
        "rows_skipped_other": "0",
        "errors_encountered": "0"
    }

    # --- Initialize Redis State ---
    if redis_client:
        try:
            async with redis_client.pipeline(transaction=True) as pipe:
                # Clear previous logs/errors, set initial status, set running flag with TTL
                await pipe.delete(REDIS_MIGRATION_KEY_STATUS, REDIS_MIGRATION_KEY_LOGS, REDIS_MIGRATION_KEY_ERRORS)
                await pipe.hset(REDIS_MIGRATION_KEY_STATUS, mapping=initial_status)
                await pipe.set(REDIS_MIGRATION_KEY_RUNNING_FLAG, "1", ex=MIGRATION_RUNNING_FLAG_TTL_SECONDS)
                await pipe.execute()
            await _log_migration_message(redis_client, f"Migration status initialized in Redis. Running flag set with TTL {MIGRATION_RUNNING_FLAG_TTL_SECONDS}s.")
        except Exception as e_init:
            # Log to console if Redis fails during init
            print(
                f"CRITICAL: Failed to initialize migration state in Redis: {e_init}")
            # Cannot update Redis status, proceed with migration but logging/status might be impaired.
            # If Redis is essential, could raise here to prevent task start. For now, we proceed.
            pass  # Allow migration to attempt running even if initial Redis write fails
    else:
        print("Warning: No Redis client provided. Migration status will not be tracked.")

    # --- Actual Migration Logic ---
    migration_stats = {"status": "failed",
                       "message": "Initialization error"}  # Default result
    try:
        await _log_migration_message(redis_client, "Initializing Supabase and Pinecone clients...")
        # Use the imported get_clients (assuming it works and initializes supabase_client)
        # We don't use embedding_client here
        # Run sync get_clients in executor
        try:
            print("Before calling get_clients()")
            embedding_client, supabase_client = await asyncio.to_thread(get_clients)
            print(
                f"After get_clients(), supabase_client: {supabase_client is not None}")

            if not supabase_client:
                print("Supabase client initialization failed")
                raise ValueError(
                    "Failed to initialize Supabase client via get_clients.")
            if not PINECONE_API_KEY:
                print("PINECONE_API_KEY not set")
                raise ValueError(
                    "PINECONE_API_KEY environment variable not set.")
        except Exception as client_e:
            print(f"Exception during client initialization: {client_e}")
            raise  # Re-raise to be caught by the outer try/except

        # Initialize Pinecone (sync function, run in executor)
        await _update_migration_status(redis_client, {"message": "Initializing Pinecone index..."})
        try:
            print(
                f"Before initializing Pinecone with API key length: {len(PINECONE_API_KEY) if PINECONE_API_KEY else 0}")
            pinecone_index = await asyncio.to_thread(initialize_pinecone, PINECONE_API_KEY, PINECONE_INDEX_NAME)
            print(
                f"After initialize_pinecone(), pinecone_index: {pinecone_index is not None}")

            if not pinecone_index:
                print(
                    f"Failed to initialize Pinecone index '{PINECONE_INDEX_NAME}'")
                raise ValueError(
                    f"Failed to initialize Pinecone index '{PINECONE_INDEX_NAME}'. Check API key and permissions.")
        except Exception as pinecone_e:
            print(f"Exception during Pinecone initialization: {pinecone_e}")
            raise  # Re-raise to be caught by the outer try/except

        if supabase_client and pinecone_index:
            await _log_migration_message(redis_client, "Clients initialized successfully. Starting core migration...")
            await _update_migration_status(redis_client, {"status": "running", "message": "Starting data transfer..."})
            # Call the core migration logic function
            migration_stats = await migrate_to_pinecone(supabase_client, pinecone_index, redis_client)
            # Status updated within migrate_to_pinecone
            await _log_migration_message(redis_client, "Migration function completed.")
        else:
            # This case should be caught by the checks above, but as a fallback:
            raise ValueError(
                "Migration prerequisites failed (Supabase client or Pinecone index initialization).")

    except Exception as e:
        err_msg = f"An error occurred during the migration script execution: {e}"
        print(f"CRITICAL ERROR: {err_msg}")
        await _log_migration_message(redis_client, err_msg, is_error=True)
        # Update status to failed in Redis
        await _update_migration_status(redis_client, {
            "status": "failed",
            "message": err_msg,
            "is_running": "0",
            "end_time": datetime.now(timezone.utc).isoformat(),
            "progress": 100  # Indicate it stopped
        })
        migration_stats = {"status": "failed", "error": str(e)}

    finally:
        # --- Cleanup ---
        # Ensure the running flag is cleared, regardless of success or failure
        if redis_client:
            try:
                await redis_client.delete(REDIS_MIGRATION_KEY_RUNNING_FLAG)
                await _log_migration_message(redis_client, "Migration running flag cleared.")
            except Exception as e_clean:
                print(f"Error clearing migration running flag: {e_clean}")
                await _log_migration_message(redis_client, f"Error clearing migration running flag: {e_clean}", is_error=True)

        print("run_migration_to_pinecone finished.")
        # The final status/result is already captured in migration_stats or updated in Redis
        return migration_stats


# Example of how to run it directly (for testing rag.py)
# if __name__ == "__main__":
#     # Requires a running Redis server accessible
#     async def main():
#          redis_host = os.getenv("REDIS_HOST", "localhost")
#          redis_port = int(os.getenv("REDIS_PORT", 6379))
#          try:
#              pool = redis.ConnectionPool.from_url(f"redis://{redis_host}:{redis_port}", decode_responses=True)
#              redis_client_conn = redis.Redis.from_pool(pool)
#              await redis_client_conn.ping()
#              print(f"Connected to Redis at {redis_host}:{redis_port} for direct testing.")
#              await run_migration_to_pinecone(redis_client_conn)
#              await redis_client_conn.close()
#          except Exception as e:
#              print(f"Failed to run migration test: {e}")
#     asyncio.run(main())
