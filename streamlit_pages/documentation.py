from utils.utils import get_env_var, create_new_tab_button
import streamlit as st
import time
import requests  # Added
import json  # Added for potential JSON error handling
import sys
import os
from datetime import datetime  # Added for parsing time strings

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Define backend URL (consider making this configurable)
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8888")

# Define known documentation sources (match backend Enum keys/values)
# Use descriptive names for labels and the actual keys for the API calls
KNOWN_SOURCES = {
    "pydantic_ai_docs": "Pydantic AI",
    "langgraph_docs": "LangGraph Python",
    "langgraphjs_docs": "LangGraph JS",
    "langsmith_docs": "LangSmith",
    "langchain_python_docs": "LangChain Python",
    "langchain_js_docs": "LangChain JS"
}


def get_backend_status():
    """Fetches crawl status from the backend API."""
    try:
        status_url = f"{BACKEND_URL}/crawl/status"
        response = requests.get(status_url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to backend: {e}", icon="🚨")
        return None
    except json.JSONDecodeError:
        st.error("Error decoding backend response.", icon="🚨")
        return None


def get_migration_status():
    """Fetches migration status from the backend API."""
    try:
        status_url = f"{BACKEND_URL}/migrate/status"
        response = requests.get(status_url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(
            f"Error connecting to backend migration status endpoint: {e}", icon="🚨")
        return None
    except json.JSONDecodeError:
        st.error("Error decoding migration status response.", icon="🚨")
        return None


# Removed supabase_client parameter


def documentation_tab():
    """Displays documentation interface, uses backend API."""
    st.header("Documentation")

    # Initialize session state flags if they don't exist
    if "api_crawl_status" not in st.session_state:
        # Fetch initial status once on load to display last known state
        st.session_state.api_crawl_status = get_backend_status()
    if "crawl_active_polling" not in st.session_state:
        st.session_state.crawl_active_polling = False  # Default to not polling

    # Initialize migration status session state
    if "api_migration_status" not in st.session_state:
        st.session_state.api_migration_status = get_migration_status()
    if "migration_active_polling" not in st.session_state:
        st.session_state.migration_active_polling = False  # Default to not polling

    # --- Fetch Status if Polling ---
    # Only fetch status again if polling is active
    if st.session_state.crawl_active_polling:
        new_status = get_backend_status()
        # If status fetch failed, keep the last known status but stop polling
        if new_status is None:
            st.session_state.crawl_active_polling = False
            st.warning("Failed to fetch status from backend. Stopping polling.")
        else:
            st.session_state.api_crawl_status = new_status
            # Check the fetched status to decide if polling should STOP
            # Check the 'is_running' field specifically
            if not st.session_state.api_crawl_status.get("is_running", False):
                # Stop polling if backend says it's not running
                st.session_state.crawl_active_polling = False

    # --- Fetch Migration Status if Polling ---
    if st.session_state.migration_active_polling:
        new_migration_status = get_migration_status()
        if new_migration_status is None:
            st.session_state.migration_active_polling = False
            st.warning(
                "Failed to fetch migration status from backend. Stopping polling.")
        else:
            st.session_state.api_migration_status = new_migration_status
            # Stop polling if migration is no longer running
            if not st.session_state.api_migration_status.get("is_running", False):
                st.session_state.migration_active_polling = False

    status = st.session_state.api_crawl_status
    migration_status = st.session_state.api_migration_status

    # Determine if the button should be disabled based *only* on active polling state
    is_crawl_button_disabled = st.session_state.crawl_active_polling
    is_migration_button_disabled = st.session_state.migration_active_polling or st.session_state.crawl_active_polling

    # Create tabs for different documentation sources
    doc_tabs = st.tabs(
        ["Pydantic AI Docs", "Data Migration", "Future Sources"])

    with doc_tabs[0]:
        st.subheader("Pydantic AI Documentation")
        st.markdown("""
        This section allows you to crawl and index the Pydantic AI documentation.
        The crawler will:

        1. Fetch URLs from the Pydantic AI sitemap
        2. Crawl each page and extract content
        3. Split content into chunks
        4. Generate embeddings for each chunk
        5. Store the chunks in the Supabase database

        This process may take several minutes depending on the number of pages.
        """)

        # Check environment variables (only needed for displaying warning, not for functionality)
        supabase_url = get_env_var("SUPABASE_URL")
        supabase_key = get_env_var("SUPABASE_SERVICE_KEY")
        if not supabase_url or not supabase_key:
            st.warning(
                "⚠️ Supabase environment variables might not be fully configured, but crawling relies on the backend.")

        # Add checkbox for resuming/processing only new URLs
        process_only_new = st.checkbox("Process only new URLs (skip existing in DB)", value=False, key="process_only_new",
                                       help="If checked, the crawler will fetch URLs already in the database and skip processing them. Useful for resuming an interrupted crawl.")

        col1, col2 = st.columns(2)

        with col1:
            # Button to start crawling - disable if polling is active
            if st.button("Start Crawl (via Backend)", key="crawl_docs_api", disabled=is_crawl_button_disabled):
                try:
                    # Include the checkbox value in the request payload
                    payload = {"process_only_new": process_only_new}
                    response = requests.post(
                        f"{BACKEND_URL}/crawl/start", json=payload)
                    response.raise_for_status()
                    st.success("🚀 Crawl initiated via backend!")
                    # Start polling immediately after successful initiation
                    st.session_state.crawl_active_polling = True
                    # Fetch status right away to update UI
                    st.session_state.api_crawl_status = get_backend_status()
                    st.rerun()  # Rerun to update UI and start polling loop
                except requests.exceptions.RequestException as e:
                    error_detail = e.response.json().get('detail', str(e)) if e.response else str(e)
                    st.error(f"❌ Error starting crawl: {error_detail}")
                    st.session_state.crawl_active_polling = False  # Ensure polling is off on error
                except Exception as e:
                    st.error(f"❌ An unexpected error occurred: {str(e)}")
                    st.session_state.crawl_active_polling = False  # Ensure polling is off on error

        with col2:
            # Expander for Clearing Specific Docs - disable buttons if polling is active
            with st.expander("Clear Specific Documentation", expanded=False):
                st.warning(
                    "⚠️ This action permanently deletes data for the selected source.")
                cols = st.columns(3)  # Arrange buttons in columns
                col_idx = 0
                for source_key, source_label in KNOWN_SOURCES.items():
                    button_key = f"clear_{source_key}_api"
                    # Disable clear buttons also when polling is active
                    if cols[col_idx % 3].button(f"Clear {source_label} Docs", key=button_key, disabled=is_crawl_button_disabled):
                        with st.spinner(f"Sending request to clear {source_label} docs..."):
                            try:
                                # Use the correct source key
                                payload = {"source": source_key}
                                response = requests.post(
                                    f"{BACKEND_URL}/crawl/clear", json=payload)
                                response.raise_for_status()
                                st.success(
                                    f"✅ {response.json().get('message', f'Clear request for {source_label} sent.')}")
                                # Rerun might be needed if this affects stats display
                                st.rerun()
                            except requests.exceptions.RequestException as e:
                                st.error(
                                    f"❌ Error clearing {source_label} docs: {e}")
                            except Exception as e:
                                st.error(
                                    f"❌ An unexpected error occurred: {str(e)}")
                    col_idx += 1

        # Display crawling progress based on API status (always read from session state)
        if status:
            # Note: is_running for display purposes comes from the fetched 'status'
            is_crawl_display_running = status.get("is_running", False)

            progress_container = st.container()
            with progress_container:
                if is_crawl_display_running:
                    st.info(
                        "Crawl in progress... Status updates automatically.", icon="⏳")
                elif status.get("end_time"):
                    # Display completion message based on success/failure counts
                    failed_count = status.get("urls_failed", 0)
                    succeeded_count = status.get("urls_succeeded", 0)
                    if failed_count == 0 and succeeded_count > 0:
                        st.success("✅ Crawl completed successfully!")
                    elif failed_count > 0:
                        st.warning(
                            f"⚠️ Crawl completed with {failed_count} failed URLs.")
                    else:
                        # Or other appropriate message
                        st.info(
                            "Crawl finished. No documents were successfully processed.")

                # Display a progress bar if running or completed
                total_urls = status.get("total_urls", 0)
                processed_count = status.get("processed_count", 0)
                if total_urls > 0:
                    progress = processed_count / total_urls
                    st.progress(
                        progress, text=f"Processing URL {processed_count}/{total_urls}")
                elif is_crawl_display_running:  # Use the display running flag here
                    st.progress(0, text="Waiting for URLs...")

                # Display status metrics
                col1, col2, col3, col4, col5 = st.columns(
                    5)  # Added column for skipped
                col1.metric("URLs Found", total_urls)
                col2.metric("URLs Processed", processed_count)
                col3.metric("Successful", status.get("urls_succeeded", 0))
                col4.metric("Failed", status.get("urls_failed", 0))
                # Display skipped count
                col5.metric("Skipped", status.get("urls_skipped", 0))

                # Display current URL and duration if available
                current_url = status.get("current_url")
                if current_url:
                    st.caption(f"Current URL: {current_url}")
                duration = status.get("duration_seconds")
                if duration is not None:
                    st.caption(
                        f"Elapsed Time: {time.strftime('%H:%M:%S', time.gmtime(duration))}")

                # Display logs and errors in expanders
                with st.expander("Crawling Logs", expanded=False):
                    logs = status.get("logs", [])
                    if logs:
                        st.code("\n".join(logs))
                    else:
                        st.code("No logs available yet...")

                with st.expander("Errors", expanded=bool(status.get("errors"))):
                    errors = status.get("errors", [])
                    if errors:
                        st.code("\n".join(errors))
                    else:
                        st.code("No errors reported.")

            # Auto-refresh logic using st.rerun
            # Only rerun if the polling flag is True
            if st.session_state.crawl_active_polling:
                # Schedule the next rerun
                time.sleep(2)  # Keep the 2-second delay between checks
                st.rerun()

    with doc_tabs[1]:
        st.subheader("Supabase to Pinecone Migration")
        st.markdown("""
        This section allows you to migrate your document embeddings from Supabase to Pinecone vector database.
        The migration process:

        1. Connects to your Supabase database to fetch stored embeddings
        2. Creates a Pinecone index if needed
        3. Transfers embeddings by namespace (source)
        4. Maintains original metadata and IDs

        This migration is necessary to use Pinecone for faster and more scalable vector search capabilities.
        """)

        # Check environment variables for Pinecone configuration
        pinecone_api_key = get_env_var("PINECONE_API_KEY")
        pinecone_index_name = get_env_var(
            "PINECONE_INDEX_NAME")
        if not pinecone_api_key:
            st.warning(
                "⚠️ Pinecone API key is not configured. The migration will fail without it.")

        st.info(f"Target Pinecone index: `{pinecone_index_name}`")

        # Migrate button
        if st.button("Start Supabase to Pinecone Migration", key="start_migration", disabled=is_migration_button_disabled):
            try:
                response = requests.post(f"{BACKEND_URL}/migrate-to-pinecone")
                response.raise_for_status()
                st.success("🚀 Migration initiated!")
                # Start polling for migration status
                st.session_state.migration_active_polling = True
                # Get initial status
                st.session_state.api_migration_status = get_migration_status()
                st.rerun()  # Rerun to update UI and start polling
            except requests.exceptions.RequestException as e:
                error_detail = e.response.json().get('detail', str(e)) if hasattr(
                    e, 'response') and e.response else str(e)
                st.error(f"❌ Error starting migration: {error_detail}")
                st.session_state.migration_active_polling = False
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")
                st.session_state.migration_active_polling = False

        # Display migration status and progress
        if migration_status:
            is_migration_running = migration_status.get("is_running", False)
            migration_progress_container = st.container()

            with migration_progress_container:
                if is_migration_running:
                    st.info(
                        "Migration in progress... Status updates automatically.", icon="⏳")
                elif migration_status.get("end_time"):
                    status_value = migration_status.get("status", "")
                    if status_value == "completed":
                        st.success("✅ Migration completed successfully!")
                    elif status_value == "completed_with_errors":
                        st.warning(
                            f"⚠️ Migration completed with some errors. Check the logs for details.")
                    elif status_value == "failed":
                        st.error(
                            "❌ Migration failed. Check the errors for details.")
                    else:
                        st.info(f"Migration status: {status_value}")

                # Display progress bar
                progress = migration_status.get("progress", 0)
                if progress > 0 or is_migration_running:
                    message = migration_status.get("message", "Processing...")
                    st.progress(progress / 100.0,
                                text=f"{progress}% - {message}")

                # Display metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Rows Processed",
                            migration_status.get("rows_processed", 0))
                col2.metric("Vectors Upserted",
                            migration_status.get("vectors_upserted", 0))

                skipped_total = migration_status.get(
                    "rows_skipped_embedding", 0) + migration_status.get("rows_skipped_other", 0)
                col3.metric("Rows Skipped", skipped_total)

                # Second row of metrics if there are errors or skipped rows
                if skipped_total > 0 or migration_status.get("errors_encountered", 0) > 0:
                    col4, col5, col6 = st.columns(3)
                    col4.metric("Skipped (Bad Embedding)", migration_status.get(
                        "rows_skipped_embedding", 0))
                    col5.metric("Skipped (Other Issues)",
                                migration_status.get("rows_skipped_other", 0))
                    col6.metric("Errors Encountered",
                                migration_status.get("errors_encountered", 0))

                # Display duration if available
                duration = migration_status.get("duration_seconds")
                if duration is not None:
                    st.caption(
                        f"Elapsed Time: {time.strftime('%H:%M:%S', time.gmtime(duration))}")

                # Display logs and errors in expanders
                with st.expander("Migration Logs", expanded=False):
                    logs = migration_status.get("logs", [])
                    if logs:
                        st.code("\n".join(logs))
                    else:
                        st.code("No logs available yet...")

                with st.expander("Migration Errors", expanded=bool(migration_status.get("errors"))):
                    errors = migration_status.get("errors", [])
                    if errors:
                        st.code("\n".join(errors))
                    else:
                        st.code("No errors reported.")

            # Auto-refresh while migration is active
            if st.session_state.migration_active_polling:
                time.sleep(2)  # 2-second polling delay
                st.rerun()

    with doc_tabs[2]:
        st.info("Additional documentation sources will be available in future updates.")

        # --- Keeping original stats logic, assuming supabase_client is available somehow --- #
        st.subheader("Database Statistics")
        try:
            # Attempt to get client (adjust based on actual setup in streamlit_ui.py)
            from utils.utils import get_clients  # Assuming get_clients is the way
            _, supabase_client_for_stats = get_clients()  # Get client

            if not supabase_client_for_stats:
                st.warning("Supabase client not available for statistics.")
                # Changed return to continue to allow the 'Future Sources' tab to render
                pass  # Continue to allow rest of UI to render
            else:
                # Query metadata->>source for all entries
                result = supabase_client_for_stats.table("site_pages").select(
                    "metadata", count="exact").execute()

                total_count = result.count if hasattr(result, "count") else 0
                st.metric("Total Indexed Chunks", total_count)

                if total_count > 0:
                    source_counts = {}
                    for item in result.data:
                        source = item.get("metadata", {}).get(
                            "source", "Unknown")
                        source_counts[source] = source_counts.get(
                            source, 0) + 1

                    num_sources = len(source_counts)
                    if num_sources > 0:
                        st.write("Chunks per Source:")
                        cols = st.columns(min(num_sources, 4))
                        col_idx = 0
                        for source, count in sorted(source_counts.items()):
                            label = f"{source.replace('_', ' ').title()} Chunks"
                            cols[col_idx % len(cols)].metric(label, count)
                            col_idx += 1

                if total_count > 0 and st.button("View Sample Indexed Data", key="view_all_data"):
                    sample_data = supabase_client_for_stats.table("site_pages") \
                        .select("url, title, summary, chunk_number, metadata") \
                        .limit(10) \
                        .execute()

                    if hasattr(sample_data, 'data') and sample_data.data:
                        formatted_data = []
                        for record in sample_data.data:
                            source = record.get(
                                "metadata", {}).get("source", "N/A")
                            formatted_data.append({
                                "Source": source,
                                "URL": record.get("url", "N/A"),
                                "Title": record.get("title", "N/A"),
                                "Summary": record.get("summary", "N/A"),
                                "Chunk": record.get("chunk_number", "N/A")
                            })
                        st.dataframe(formatted_data, use_container_width=True)
                    else:
                        st.info("No sample data found.")
                    st.info("Showing up to 10 sample records from all sources.")

        except Exception as e:
            st.error(f"Error querying database statistics: {str(e)}")
