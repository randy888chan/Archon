import streamlit as st
import time
import sys
import os
import subprocess
from pathlib import Path
import requests
import tempfile  # Keep for now, might be used elsewhere, but remove specific usage below
import re  # For sanitization

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from archon.crawl_pydantic_ai_docs import (
    start_crawl_with_requests,
    clear_existing_records,
)
from utils.utils import get_env_var, create_new_tab_button, write_to_log


def documentation_tab(supabase_client):
    """Display the documentation interface"""
    st.header("Documentation")

    # Create tabs for different documentation sources
    doc_tabs = st.tabs(
        ["Pydantic AI Docs", "Framework Docs (llms.txt)", "Future Sources"]
    )

    with doc_tabs[0]:
        st.subheader("Pydantic AI Documentation")
        st.markdown(
            """
        This section allows you to crawl and index the Pydantic AI documentation.
        The crawler will:
        
        1. Fetch URLs from the Pydantic AI sitemap
        2. Crawl each page and extract content
        3. Split content into chunks
        4. Generate embeddings for each chunk
        5. Store the chunks in the Supabase database
        
        This process may take several minutes depending on the number of pages.
        """
        )

        # Check if the database is configured
        supabase_url = get_env_var("SUPABASE_URL")
        supabase_key = get_env_var("SUPABASE_SERVICE_KEY")

        if not supabase_url or not supabase_key:
            st.warning(
                "⚠️ Supabase is not configured. Please set up your environment variables first."
            )
            create_new_tab_button(
                "Go to Environment Section", "Environment", key="goto_env_from_docs"
            )
        else:
            # Initialize session state for tracking crawl progress
            if "crawl_tracker" not in st.session_state:
                st.session_state.crawl_tracker = None

            if "crawl_status" not in st.session_state:
                st.session_state.crawl_status = None

            if "last_update_time" not in st.session_state:
                st.session_state.last_update_time = time.time()

            # Create columns for the buttons
            col1, col2 = st.columns(2)

            with col1:
                # Button to start crawling
                if st.button("Crawl Pydantic AI Docs", key="crawl_pydantic") and not (
                    st.session_state.crawl_tracker
                    and st.session_state.crawl_tracker.is_running
                ):
                    try:
                        # Clear the Output log at the start of processing a new file
                        # This ensures users can see progress from a clean state
                        if "crawl_status" in st.session_state:
                            st.session_state.crawl_status = None

                        # Define a callback function to update the session state
                        def update_progress(status):
                            st.session_state.crawl_status = status

                        # Start the crawling process in a separate thread
                        st.session_state.crawl_tracker = start_crawl_with_requests(
                            update_progress
                        )
                        st.session_state.crawl_status = (
                            st.session_state.crawl_tracker.get_status()
                        )

                        # Force a rerun to start showing progress
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error starting crawl: {str(e)}")

            with col2:
                # Button to clear existing Pydantic AI docs
                if st.button("Clear Pydantic AI Docs", key="clear_pydantic"):
                    with st.spinner("Clearing existing Pydantic AI docs..."):
                        try:
                            # Clear the Output log at the start of processing
                            # This ensures users can see progress from a clean state
                            if "crawl_status" in st.session_state:
                                st.session_state.crawl_status = None

                            # Run the function to clear records
                            clear_existing_records()
                            st.success(
                                "✅ Successfully cleared existing Pydantic AI docs from the database."
                            )

                            # Force a rerun to update the UI
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error clearing Pydantic AI docs: {str(e)}")

            # Display crawling progress if a crawl is in progress or has completed
            if st.session_state.crawl_tracker:
                # Create a container for the progress information
                progress_container = st.container()

                with progress_container:
                    # Get the latest status
                    current_time = time.time()
                    # Update status every second
                    if current_time - st.session_state.last_update_time >= 1:
                        st.session_state.crawl_status = (
                            st.session_state.crawl_tracker.get_status()
                        )
                        st.session_state.last_update_time = current_time

                    status = st.session_state.crawl_status

                    # Display a progress bar
                    if status and status["urls_found"] > 0:
                        progress = status["urls_processed"] / status["urls_found"]
                        st.progress(progress)

                    # Display status metrics
                    col1, col2, col3, col4 = st.columns(4)
                    if status:
                        col1.metric("URLs Found", status["urls_found"])
                        col2.metric("URLs Processed", status["urls_processed"])
                        col3.metric("Successful", status["urls_succeeded"])
                        col4.metric("Failed", status["urls_failed"])
                    else:
                        col1.metric("URLs Found", 0)
                        col2.metric("URLs Processed", 0)
                        col3.metric("Successful", 0)
                        col4.metric("Failed", 0)

                    # Display logs in an expander
                    with st.expander("Crawling Logs", expanded=True):
                        if status and "logs" in status:
                            logs_text = "\n".join(
                                status["logs"][-20:]
                            )  # Show last 20 logs
                            st.code(logs_text)
                        else:
                            st.code("No logs available yet...")

                    # Show completion message
                    if status and not status["is_running"] and status["end_time"]:
                        if status["urls_failed"] == 0:
                            st.success("✅ Crawling process completed successfully!")
                        else:
                            st.warning(
                                f"⚠️ Crawling process completed with {status['urls_failed']} failed URLs."
                            )

                # Auto-refresh while crawling is in progress
                if not status or status["is_running"]:
                    st.rerun()

        # Display database statistics
        st.subheader("Database Statistics")
        try:
            # Query the count of Pydantic AI docs
            result = (
                supabase_client.table("site_pages")
                .select("count", count="exact")
                .eq("metadata->>source", "pydantic_ai_docs")
                .execute()
            )
            count = result.count if hasattr(result, "count") else 0

            # Display the count
            st.metric("Pydantic AI Docs Chunks", count)

            # Add a button to view the data
            if count > 0 and st.button("View Indexed Data", key="view_pydantic_data"):
                # Query a sample of the data
                sample_data = (
                    supabase_client.table("site_pages")
                    .select("url,title,summary,chunk_number")
                    .eq("metadata->>source", "pydantic_ai_docs")
                    .limit(10)
                    .execute()
                )

                # Display the sample data
                st.dataframe(sample_data.data)
                st.info(
                    "Showing up to 10 sample records. The database contains more records."
                )
        except Exception as e:
            st.error(f"Error querying database: {str(e)}")

    with doc_tabs[1]:  # Framework Docs (llms.txt) Tab
        st.subheader("Framework Documentation (llms.txt)")
        st.markdown(
            """
        Process pre-formatted `llms.txt` files containing documentation for various frameworks.
        This process parses the file, chunks the content hierarchically, generates embeddings,
        and stores the data in the `hierarchical_nodes` table in the database.

        **Note:** Ensure the 'Hierarchical Nodes' table is selected in the Database configuration tab before processing.
        """
        )

        # Check if the database is configured
        supabase_url = get_env_var("SUPABASE_URL")
        supabase_key = get_env_var("SUPABASE_SERVICE_KEY")

        if not supabase_url or not supabase_key:
            st.warning(
                "⚠️ Supabase is not configured. Please set up your environment variables first."
            )
            create_new_tab_button(
                "Go to Environment Section",
                "Environment",
                key="goto_env_from_llms_docs",
            )
        else:
            # Framework Selection
            framework_options = {
                "--- Select a framework ---": None,
                "Pydantic AI (ai.pydantic.dev)": "https://ai.pydantic.dev/llms.txt",
                "LangGraph (langchain-ai.github.io)": "https://langchain-ai.github.io/langgraph/llms-full.txt",
                "CrewAI (docs.crewai.com)": "https://docs.crewai.com/llms-full.txt",
            }
            selected_framework_display = st.selectbox(
                "Select Framework Documentation (llms.txt):",
                options=list(framework_options.keys()),
                key="llms_framework_select",
            )
            selected_framework_file = framework_options[selected_framework_display]

            # Text input for custom URL
            custom_url = st.text_input(
                "Or Enter Custom URL to llms.txt file:",
                key="llms_custom_url_input",
                placeholder="https://example.com/path/to/your_llms.txt",
            )

            # Session state for uploaded file
            if "uploaded_file_path" not in st.session_state:
                st.session_state.uploaded_file_path = None
            if "uploaded_file_original_name" not in st.session_state:
                st.session_state.uploaded_file_original_name = None

            st.subheader("Upload Custom llms.txt File")
            uploaded_file = st.file_uploader(
                "Choose a .txt file (e.g., llms.txt, llms-full.txt)",
                type="txt",
                key="llms_file_uploader",
            )

            if uploaded_file is not None:
                # Process uploaded file immediately upon upload (save it)
                original_filename = uploaded_file.name
                # Use the existing sanitize_filename function defined below/above
                sanitized_filename = sanitize_filename(original_filename)
                docs_dir = "docs"
                saved_file_path = os.path.join(docs_dir, sanitized_filename)
                save_status = st.empty()  # Placeholder for status messages

                try:
                    save_status.info(
                        f"Saving '{original_filename}' as '{sanitized_filename}'..."
                    )
                    # Ensure docs directory exists
                    os.makedirs(docs_dir, exist_ok=True)

                    # Read content and save
                    # Use getvalue() which returns bytes, then decode
                    file_content = uploaded_file.getvalue().decode("utf-8")
                    with open(saved_file_path, "w", encoding="utf-8") as f:
                        f.write(file_content)

                    save_status.success(
                        f"✅ File '{original_filename}' saved as '{saved_file_path}'. Ready for processing."
                    )
                    st.session_state.uploaded_file_path = (
                        saved_file_path  # Store path for the button
                    )
                    st.session_state.uploaded_file_original_name = original_filename
                    # uploaded_file = None # Resetting local var might not be enough

                except Exception as e:
                    save_status.error(f"❌ Error saving uploaded file: {e}")
                    st.session_state.uploaded_file_path = None  # Reset path on error
                    st.session_state.uploaded_file_original_name = None
                    write_to_log(f"Error saving uploaded file {original_filename}: {e}")

            # --- Button to process the *saved* uploaded file ---
            if st.session_state.uploaded_file_path:
                # st.markdown("---") # Separator - removed, button serves as visual break
                if st.button(
                    f"Process Uploaded File: '{st.session_state.uploaded_file_original_name}'",
                    key="process_uploaded_llms_file",
                ):
                    # Clear the Output log at the start of processing a new file
                    # This ensures users can see progress from a clean state
                    st.session_state.llms_processing_output = None
                    st.session_state.llms_processing_error = None
                    st.session_state.llms_processing_complete = False

                    # Check Supabase config and table selection again (important!)
                    supabase_url = get_env_var("SUPABASE_URL")
                    supabase_key = get_env_var("SUPABASE_SERVICE_KEY")
                    docs_retrieval_table = get_env_var("DOCS_RETRIEVAL_TABLE")

                    if not supabase_url or not supabase_key:
                        st.warning(
                            "⚠️ Supabase is not configured. Please set up your environment variables first."
                        )
                        # create_new_tab_button("Go to Environment Section", "Environment", key="goto_env_from_uploaded_llms")
                    elif docs_retrieval_table != "hierarchical_nodes":
                        st.warning(
                            f"⚠️ Incorrect table selected for documentation processing. Expected 'hierarchical_nodes' but found '{docs_retrieval_table}'. Please select 'Hierarchical Nodes' in the Database configuration tab."
                        )
                    else:
                        # Proceed with processing the saved file
                        # Note: Output log already cleared at the beginning of this button click handler

                        script_path = "run_processing.py"
                        file_arg = (
                            st.session_state.uploaded_file_path
                        )  # Use the saved path
                        command = [sys.executable, script_path, "--file", file_arg]
                        # Optional: Add document ID based on sanitized filename stem
                        # doc_id = Path(file_arg).stem
                        # command.extend(["--id", doc_id])

                        process_placeholder = st.empty()
                        # Use a unique key for the status context if needed, or rely on rerun clearing it
                        with process_placeholder.status(
                            f"Processing uploaded file: {st.session_state.uploaded_file_original_name}...",
                            expanded=True,
                        ) as status:
                            try:
                                status.write(f"Running command: `{' '.join(command)}`")

                                # Initialize output and error storage
                                st.session_state.llms_processing_output = ""
                                st.session_state.llms_processing_error = ""

                                # Create placeholders for real-time logs
                                log_output_placeholder = st.empty()
                                log_error_placeholder = st.empty()

                                # Create a text area for stdout
                                log_output = log_output_placeholder.text_area(
                                    "Output Log (Real-time):",
                                    value="",
                                    height=200,
                                    key="uploaded_output_log",
                                )

                                # Create a placeholder for stderr (will only show if errors occur)
                                log_error = None

                                # Start the process with Popen
                                process = subprocess.Popen(
                                    command,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True,
                                    encoding="utf-8",
                                    errors="replace",  # Handle potential encoding issues in output
                                )

                                # Store output/error in session state for display later
                                st.session_state.llms_processing_output = result.stdout
                                st.session_state.llms_processing_error = result.stderr

                                if result.returncode == 0:
                                    status.update(
                                        label="Processing complete!",
                                        state="complete",
                                        expanded=False,
                                    )
                                    st.session_state.llms_processing_complete = True
                                    write_to_log(
                                        f"Successfully processed uploaded file: {file_arg} (Original: {st.session_state.uploaded_file_original_name})."
                                    )
                                    # Clear the saved path state after successful processing to hide the button
                                    st.session_state.uploaded_file_path = None
                                    st.session_state.uploaded_file_original_name = None
                                else:
                                    status.update(
                                        label="Processing failed!",
                                        state="error",
                                        expanded=True,
                                    )
                                    st.session_state.llms_processing_complete = True
                                    write_to_log(
                                        f"Error processing uploaded file: {file_arg}. Return code: {result.returncode}\\nStderr: {result.stderr}"
                                    )

                            except FileNotFoundError:
                                status.update(
                                    label="Error: Script not found!",
                                    state="error",
                                    expanded=True,
                                )
                                st.session_state.llms_processing_error = f"Error: The script '{script_path}' was not found. Ensure it exists and Python can execute it."
                                st.session_state.llms_processing_complete = True
                                write_to_log(
                                    f"Error running processing script for uploaded file: FileNotFoundError - {script_path}"
                                )
                            except Exception as e:
                                status.update(
                                    label="An unexpected error occurred during processing!",
                                    state="error",
                                    expanded=True,
                                )
                                st.session_state.llms_processing_error = (
                                    f"An unexpected error occurred: {str(e)}"
                                )
                                st.session_state.llms_processing_complete = True
                                write_to_log(
                                    f"Unexpected error processing uploaded file {file_arg}: {e}"
                                )

                        # Rerun to display results outside the status context and update UI state (e.g., hide button)
                        st.rerun()

            st.markdown("---")  # Separator before URL processing section
            st.subheader("Process Framework Docs from URL")  # Add subheader for clarity

            # Processing Button
            # Helper function to sanitize URL/filename (moved inside for scope if preferred, or keep global)
            def sanitize_filename(source_identifier: str) -> str:
                """Sanitizes a URL or filename to be used as a valid filename."""
                # Remove protocol
                sanitized = re.sub(r"^https?://", "", source_identifier)
                # Replace invalid filename characters with underscores
                sanitized = re.sub(r'[\\/*?:"<>|]', "_", sanitized)
                # Replace other common separators or problematic chars
                sanitized = re.sub(r"[/.&=%]", "_", sanitized)
                # Limit length to avoid issues (e.g., 100 chars)
                sanitized = sanitized[:100]
                # Ensure it doesn't end with an underscore or period
                sanitized = sanitized.strip("_.")
                # Add .txt extension if not present (handle potential double extensions carefully)
                if not sanitized.lower().endswith(".txt"):
                    sanitized += ".txt"
                return sanitized

            # Determine which URL to process (custom input takes precedence)
            url_to_process = custom_url if custom_url else selected_framework_file
            button_label = (
                f"Process Custom URL"
                if custom_url
                else f"Process Selected Framework: {selected_framework_display}"
            )
            button_disabled = (
                not url_to_process
            )  # Disable if neither custom nor selected URL is available

            if st.button(
                button_label, key="process_llms_docs", disabled=button_disabled
            ):
                # Clear the Output log at the start of processing a new file
                # This ensures users can see progress from a clean state
                st.session_state.llms_processing_output = None
                st.session_state.llms_processing_error = None
                st.session_state.llms_processing_complete = False

                # The original check for selected_framework_file is now covered by url_to_process
                # Check if the correct table is selected in config
                docs_retrieval_table = get_env_var("DOCS_RETRIEVAL_TABLE")
                if docs_retrieval_table != "hierarchical_nodes":
                    st.warning(
                        f"⚠️ Incorrect table selected for documentation processing. Expected 'hierarchical_nodes' but found '{docs_retrieval_table}'. Please select 'Hierarchical Nodes' in the Database configuration tab to process llms.txt files."
                    )
                else:
                    # Note: Output log already cleared at the beginning of this button click handler

                    script_path = "run_processing.py"
                    persistent_file_path = None  # Path for the uniquely named file
                    command = None  # Initialize command
                    docs_dir = "docs"  # Define target directory

                    process_placeholder = st.empty()
                    with process_placeholder.status(
                        "Preparing documentation...", expanded=True
                    ) as status:
                        try:
                            # Ensure 'docs' directory exists
                            os.makedirs(docs_dir, exist_ok=True)
                            status.write(f"Ensured '{docs_dir}' directory exists.")

                            # Download the file from URL (using url_to_process)
                            status.write(
                                f"Downloading documentation from {url_to_process}..."
                            )
                            response = requests.get(
                                url_to_process, timeout=60
                            )  # Increased timeout
                            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                            content = response.text  # Use text for utf-8 handling

                            # Generate unique filename and save persistently (using url_to_process)
                            unique_filename = sanitize_filename(url_to_process)
                            persistent_file_path = os.path.join(
                                docs_dir, unique_filename
                            )
                            status.write(
                                f"Saving content to persistent file: {persistent_file_path}"
                            )
                            with open(persistent_file_path, "w", encoding="utf-8") as f:
                                f.write(content)
                            status.write(
                                f"Content saved successfully to {persistent_file_path}"
                            )

                            # Prepare the command using the persistent file path
                            file_arg = persistent_file_path
                            command = [sys.executable, script_path, "--file", file_arg]
                            # Optionally pass the original URL as document ID
                            # command.extend(["--id", url_to_process]) # Or use sanitized filename stem

                        except requests.exceptions.RequestException as e:
                            status.update(
                                label="Download Failed!", state="error", expanded=True
                            )
                            st.session_state.llms_processing_error = (
                                f"Error downloading file from {url_to_process}: {e}"
                            )
                            st.session_state.llms_processing_complete = True
                            write_to_log(f"Error downloading {url_to_process}: {e}")
                        except OSError as e:
                            status.update(
                                label="File Saving Failed!",
                                state="error",
                                expanded=True,
                            )
                            st.session_state.llms_processing_error = (
                                f"Error saving file to {persistent_file_path}: {e}"
                            )
                            st.session_state.llms_processing_complete = True
                            write_to_log(
                                f"Error saving file {persistent_file_path}: {e}"
                            )
                        except Exception as e:
                            status.update(
                                label="Error preparing file!",
                                state="error",
                                expanded=True,
                            )
                            st.session_state.llms_processing_error = f"An unexpected error occurred while preparing the file: {e}"
                            st.session_state.llms_processing_complete = True
                            write_to_log(
                                f"Unexpected error preparing file from {url_to_process}: {e}"
                            )

                    # Proceed only if download and file saving were successful
                    if command and persistent_file_path:
                        with process_placeholder.status(
                            "Processing documentation...", expanded=True
                        ) as status:
                            try:
                                st.write(f"Running command: `{' '.join(command)}`")

                                # Initialize output and error storage
                                st.session_state.llms_processing_output = ""
                                st.session_state.llms_processing_error = ""

                                # Create placeholders for real-time logs
                                log_output_placeholder = st.empty()
                                log_error_placeholder = st.empty()

                                # Create a text area for stdout
                                log_output = log_output_placeholder.text_area(
                                    "Output Log (Real-time):",
                                    value="",
                                    height=200,
                                    key="url_output_log",
                                )

                                # Create a placeholder for stderr (will only show if errors occur)
                                log_error = None

                                # Start the process with Popen
                                process = subprocess.Popen(
                                    command,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True,
                                    encoding="utf-8",
                                    errors="replace",  # Handle potential encoding errors
                                )

                                st.session_state.llms_processing_output = result.stdout
                                st.session_state.llms_processing_error = result.stderr

                                if result.returncode == 0:
                                    status.update(
                                        label="Processing complete!",
                                        state="complete",
                                        expanded=False,
                                    )
                                    st.session_state.llms_processing_complete = True
                                    write_to_log(
                                        f"Successfully processed {persistent_file_path} (from URL: {url_to_process})."
                                    )
                                else:
                                    status.update(
                                        label="Processing failed!",
                                        state="error",
                                        expanded=True,
                                    )
                                    st.session_state.llms_processing_complete = True
                                    write_to_log(
                                        f"Error processing {persistent_file_path} (from URL: {url_to_process}). Return code: {result.returncode}"
                                    )

                            except FileNotFoundError:
                                status.update(
                                    label="Error: Script not found!",
                                    state="error",
                                    expanded=True,
                                )
                                st.session_state.llms_processing_error = f"Error: The script '{script_path}' was not found. Make sure it's in the correct location."
                                st.session_state.llms_processing_complete = True
                                write_to_log(
                                    f"Error running processing script: FileNotFoundError - {script_path}"
                                )
                            except Exception as e:
                                status.update(
                                    label="An unexpected error occurred during processing!",
                                    state="error",
                                    expanded=True,
                                )
                                st.session_state.llms_processing_error = f"An unexpected error occurred during processing: {str(e)}"
                                st.session_state.llms_processing_complete = True
                                write_to_log(
                                    f"Unexpected error processing {persistent_file_path} (from URL: {url_to_process}): {e}"
                                )

                        # Rerun to display results outside the status context
                        st.rerun()
                    elif not st.session_state.get(
                        "llms_processing_complete"
                    ):  # If command wasn't set due to download error, but not marked complete yet
                        st.error(
                            "Processing could not start due to an error during file preparation."
                        )
                        st.session_state.llms_processing_complete = (
                            True  # Ensure it's marked complete
                        )
                        st.rerun()  # Rerun to show the error message clearly

                    # Rerun to display results outside the status context
                    st.rerun()

            # Display processing results
            if st.session_state.get("llms_processing_complete"):
                st.subheader("Processing Results")
                if st.session_state.get("llms_processing_output"):
                    st.text_area(
                        "Output Log:",
                        st.session_state.llms_processing_output,
                        height=300,
                        key="llms_output_area",
                    )
                if st.session_state.get("llms_processing_error"):
                    st.error("Errors occurred during processing:")
                    st.code(st.session_state.llms_processing_error, language="bash")
                elif st.session_state.get(
                    "llms_processing_output"
                ):  # Check if output exists even if no error
                    st.success("✅ Processing finished.")  # Simple success if no stderr

            # Database Statistics for hierarchical_nodes
            st.subheader("Database Statistics (`hierarchical_nodes`)")
            try:
                # Query total count
                total_result = (
                    supabase_client.table("hierarchical_nodes")
                    .select("count", count="exact")
                    .execute()
                )
                total_count = (
                    total_result.count if hasattr(total_result, "count") else 0
                )
                st.metric("Total Hierarchical Nodes", total_count)

                # Query counts per document_id
                if total_count > 0:
                    # Fetch distinct document_ids first
                    doc_ids_result = (
                        supabase_client.table("hierarchical_nodes")
                        .select("document_id", count="exact")
                        .execute()
                    )

                    if doc_ids_result.data:
                        st.write("Node Counts per Framework:")
                        counts_data = {}
                        # Query count for each document_id (might be slow for many docs, consider alternative if needed)
                        # Supabase python client doesn't directly support GROUP BY with count yet in a simple way.
                        # Fetching all document_ids and counting might be inefficient for very large datasets.
                        # A more performant approach might involve a custom RPC function in Supabase.
                        # For now, fetching distinct IDs and then counting seems feasible for a few frameworks.

                        # Let's fetch a sample of document_ids and their counts for display
                        # This is still not ideal, a GROUP BY would be better.
                        # Fetching all data just to count is bad. Let's just show total for now.
                        # TODO: Implement a more efficient way to get counts per document_id (e.g., RPC function)
                        st.info(
                            "Displaying counts per framework requires a database function (RPC) for efficiency. Showing total count for now."
                        )

                        # Example of how it *could* look if counts were available:
                        # counts_per_doc = {'doc1': 150, 'doc2': 300} # TODO:
                        # st.dataframe(counts_per_doc)

                        # Add a button to view sample data
                        if st.button(
                            "View Sample Hierarchical Data",
                            key="view_hierarchical_data",
                        ):
                            sample_data = (
                                supabase_client.table("hierarchical_nodes")
                                .select(
                                    "id, document_id, node_type, title, path, level"
                                )
                                .limit(10)
                                .execute()
                            )
                            st.dataframe(sample_data.data)
                            st.info("Showing up to 10 sample records.")

            except Exception as e:
                st.error(f"Error querying hierarchical_nodes table: {str(e)}")
                write_to_log(f"Error querying hierarchical_nodes table: {str(e)}")

            # llms-text.ai Search Integration
            st.markdown("---")  # Separator before search section
            st.subheader("Search llms.txt Files via llms-text.ai")

            # Add explanatory text
            st.markdown(
                """
            Search for llms.txt and llms-full.txt files across the web using the llms-text.ai API.
            This search allows you to find websites that have implemented the LLMs protocol.
            """
            )

            # Initialize session state for search results and pagination
            if "llms_search_results" not in st.session_state:
                st.session_state.llms_search_results = None
            if "llms_search_page" not in st.session_state:
                st.session_state.llms_search_page = 1
            if "llms_search_limit" not in st.session_state:
                st.session_state.llms_search_limit = 10
            if "llms_search_total" not in st.session_state:
                st.session_state.llms_search_total = 0
            # Initialize session state for processing selected URL
            if "selected_llms_url" not in st.session_state:
                st.session_state.selected_llms_url = None
            if "process_selected_url" not in st.session_state:
                st.session_state.process_selected_url = False

            # Create a visually distinct search section with a border
            search_container = st.container()
            with search_container:
                st.markdown(
                    """
                <style>
                .search-container {
                    border: 1px solid #ddd;
                    border-radius: 10px;
                    padding: 20px;
                    margin-bottom: 20px;
                    background-color: #f8f9fa;
                }
                </style>
                <div class="search-container">
                """,
                    unsafe_allow_html=True,
                )

                # Search form
                with st.form(key="llms_search_form"):
                    # Search query input (required)
                    search_query = st.text_input(
                        "Search Query (required):",
                        key="llms_search_query",
                        placeholder="Enter search terms (e.g., 'AI', 'documentation', 'cloudflare.com')",
                    )

                    # Create two columns for file type and pagination controls
                    col1, col2 = st.columns(2)

                    with col1:
                        # File type selector
                        file_type_options = {
                            "both": "Both (llms.txt & llms-full.txt)",
                            "llms.txt": "Basic (llms.txt only)",
                            "llms-full.txt": "Comprehensive (llms-full.txt only)",
                        }
                        file_type = st.selectbox(
                            "File Type:",
                            options=list(file_type_options.keys()),
                            format_func=lambda x: file_type_options[x],
                            key="llms_file_type",
                        )

                    with col2:
                        # Pagination controls
                        col2a, col2b = st.columns(2)
                        with col2a:
                            page = st.number_input(
                                "Page:",
                                min_value=1,
                                value=st.session_state.llms_search_page,
                                step=1,
                                key="llms_page_input",
                            )
                        with col2b:
                            limit = st.number_input(
                                "Results per page:",
                                min_value=1,
                                max_value=50,
                                value=st.session_state.llms_search_limit,
                                step=5,
                                key="llms_limit_input",
                            )

                    # Search button
                    search_submitted = st.form_submit_button("Search llms-text.ai")

                st.markdown("</div>", unsafe_allow_html=True)

                # Process search when form is submitted
                if search_submitted:
                    if not search_query:
                        st.error(
                            "Search query is required. Please enter a search term."
                        )
                    else:
                        # Update session state with form values
                        st.session_state.llms_search_page = page
                        st.session_state.llms_search_limit = limit

                        # Show loading spinner during API call
                        with st.spinner("Searching llms-text.ai..."):
                            try:
                                # Construct API URL with parameters
                                api_url = "https://llms-text.ai/api/search-llms"
                                params = {
                                    "q": search_query,
                                    "fileType": file_type,
                                    "page": page,
                                    "limit": limit,
                                }

                                # Make API request
                                response = requests.get(
                                    api_url, params=params, timeout=10
                                )

                                # Check if request was successful
                                if response.status_code == 200:
                                    # Parse JSON response
                                    search_results = response.json()
                                    st.session_state.llms_search_results = (
                                        search_results
                                    )
                                    st.session_state.llms_search_total = (
                                        search_results.get("totalResults", 0)
                                    )

                                    # Display success message
                                    if search_results.get("totalResults", 0) > 0:
                                        st.success(
                                            f"Found {search_results.get('totalResults', 0)} results for '{search_query}'"
                                        )
                                    else:
                                        st.info(
                                            f"No results found for '{search_query}'"
                                        )
                                else:
                                    # Handle API error
                                    error_data = (
                                        response.json()
                                        if response.headers.get("content-type")
                                        == "application/json"
                                        else {"error": "Unknown error"}
                                    )
                                    error_message = error_data.get(
                                        "error",
                                        f"API returned status code {response.status_code}",
                                    )
                                    st.error(
                                        f"Error searching llms-text.ai: {error_message}"
                                    )
                                    st.session_state.llms_search_results = None
                            except requests.exceptions.RequestException as e:
                                # Handle network or timeout errors
                                st.error(
                                    f"Error connecting to llms-text.ai API: {str(e)}"
                                )
                                st.session_state.llms_search_results = None
                            except ValueError as e:
                                # Handle JSON parsing errors
                                st.error(f"Error parsing API response: {str(e)}")
                                st.session_state.llms_search_results = None
                            except Exception as e:
                                # Handle any other unexpected errors
                                st.error(f"Unexpected error during search: {str(e)}")
                                st.session_state.llms_search_results = None

                # Display search results if available
                if st.session_state.llms_search_results:
                    results = st.session_state.llms_search_results

                    # Display pagination information
                    total_results = results.get("totalResults", 0)
                    current_page = results.get("page", 1)
                    results_per_page = results.get("limit", 10)
                    total_pages = (
                        total_results + results_per_page - 1
                    ) // results_per_page  # Ceiling division

                    st.markdown(
                        f"**Showing page {current_page} of {total_pages} ({total_results} total results)**"
                    )

                    # Display results
                    result_items = results.get("results", [])
                    if result_items:
                        for i, item in enumerate(result_items):
                            # Create columns for the result and process button
                            result_col, button_col = st.columns([4, 1])

                            with result_col:
                                # Create a card-like container for each result
                                st.markdown(
                                    f"""
                                <div style="border: 1px solid #ddd; border-radius: 5px; padding: 10px; margin-bottom: 10px;">
                                    <h4 style="margin-top: 0;">{i+1}. {item.get('title', 'Untitled')}</h4>
                                    <p><strong>Domain:</strong> {item.get('domain', 'N/A')}</p>
                                    <p><strong>URL:</strong> <a href="{item.get('url', '#')}" target="_blank">{item.get('url', 'N/A')}</a></p>
                                    <p><strong>Last Updated:</strong> {item.get('last_updated', 'N/A')}</p>
                                    <p><strong>Summary:</strong> {item.get('summary', 'No summary available')}</p>
                                </div>
                                """,
                                    unsafe_allow_html=True,
                                )

                            with button_col:
                                # Add a button to process this file
                                if st.button(
                                    f"Process File", key=f"process_result_{i}"
                                ):
                                    # Set the URL to process
                                    url_to_process = item.get("url", "")
                                    if url_to_process:
                                        # Store the URL in session state and rerun to trigger processing
                                        st.session_state.selected_llms_url = (
                                            url_to_process
                                        )
                                        st.session_state.process_selected_url = True
                                        st.rerun()

                            # Display metadata in an expander
                            metadata = item.get("metadata", {})
                            if metadata:
                                with st.expander(
                                    f"View Metadata for {item.get('domain', 'this result')}"
                                ):
                                    # Display source domain
                                    st.markdown(
                                        f"**Source Domain:** {metadata.get('source_domain', 'N/A')}"
                                    )

                                    # Display URL purpose ranking
                                    url_purpose = metadata.get(
                                        "url_purpose_ranking", []
                                    )
                                    if url_purpose:
                                        st.markdown("**URL Purpose Ranking:**")
                                        for purpose in url_purpose:
                                            st.markdown(f"- {purpose}")

                                    # Display URL topic ranking
                                    url_topics = metadata.get("url_topic_ranking", [])
                                    if url_topics:
                                        st.markdown("**URL Topic Ranking:**")
                                        for topic in url_topics:
                                            if (
                                                isinstance(topic, list)
                                                and len(topic) >= 2
                                            ):
                                                st.markdown(f"- {topic[0]}: {topic[1]}")
                                            else:
                                                st.markdown(f"- {topic}")

                                    # Display domain purpose ranking
                                    domain_purpose = metadata.get(
                                        "domain_purpose_ranking", []
                                    )
                                    if domain_purpose:
                                        st.markdown("**Domain Purpose Ranking:**")
                                        for purpose in domain_purpose:
                                            st.markdown(f"- {purpose}")

                                    # Display domain topic ranking
                                    domain_topics = metadata.get(
                                        "domain_topic_ranking", []
                                    )
                                    if domain_topics:
                                        st.markdown("**Domain Topic Ranking:**")
                                        for topic in domain_topics:
                                            if (
                                                isinstance(topic, list)
                                                and len(topic) >= 2
                                            ):
                                                st.markdown(f"- {topic[0]}: {topic[1]}")
                                            else:
                                                st.markdown(f"- {topic}")

                        # Add pagination controls
                        st.markdown("---")
                        pagination_cols = st.columns([1, 2, 1])

                        with pagination_cols[0]:
                            if current_page > 1:
                                if st.button("← Previous Page"):
                                    st.session_state.llms_search_page = current_page - 1
                                    st.rerun()

                        with pagination_cols[1]:
                            st.markdown(
                                f"<div style='text-align: center;'>Page {current_page} of {total_pages}</div>",
                                unsafe_allow_html=True,
                            )

                        with pagination_cols[2]:
                            if current_page < total_pages:
                                if st.button("Next Page →"):
                                    st.session_state.llms_search_page = current_page + 1
                                    st.rerun()

            # Process selected URL if flag is set
            if st.session_state.get("process_selected_url", False):
                st.session_state.process_selected_url = False  # Reset the flag
                selected_url = st.session_state.get("selected_llms_url")

                if selected_url:
                    # Clear previous processing state
                    st.session_state.llms_processing_output = None
                    st.session_state.llms_processing_error = None
                    st.session_state.llms_processing_complete = False

                    # Check if the correct table is selected in config
                    docs_retrieval_table = get_env_var("DOCS_RETRIEVAL_TABLE")
                    if docs_retrieval_table != "hierarchical_nodes":
                        st.warning(
                            f"⚠️ Incorrect table selected for documentation processing. Expected 'hierarchical_nodes' but found '{docs_retrieval_table}'. Please select 'Hierarchical Nodes' in the Database configuration tab to process llms.txt files."
                        )
                    else:
                        # Set the URL to process and trigger the existing processing logic
                        custom_url = selected_url
                        url_to_process = custom_url

                        # Display a message indicating that processing has started
                        st.success(f"Processing URL: {selected_url}")

                        # Proceed with processing using the existing logic
                        script_path = "run_processing.py"
                        persistent_file_path = None  # Path for the uniquely named file
                        command = None  # Initialize command
                        docs_dir = "docs"  # Define target directory

                        process_placeholder = st.empty()
                        with process_placeholder.status(
                            "Preparing documentation...", expanded=True
                        ) as status:
                            try:
                                # Ensure 'docs' directory exists
                                os.makedirs(docs_dir, exist_ok=True)
                                status.write(f"Ensured '{docs_dir}' directory exists.")

                                # Download the file from URL (using url_to_process)
                                status.write(
                                    f"Downloading documentation from {url_to_process}..."
                                )
                                response = requests.get(
                                    url_to_process, timeout=60
                                )  # Increased timeout
                                response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                                content = response.text  # Use text for utf-8 handling

                                # Generate unique filename and save persistently (using url_to_process)
                                unique_filename = sanitize_filename(url_to_process)
                                persistent_file_path = os.path.join(
                                    docs_dir, unique_filename
                                )
                                status.write(
                                    f"Saving content to persistent file: {persistent_file_path}"
                                )
                                with open(
                                    persistent_file_path, "w", encoding="utf-8"
                                ) as f:
                                    f.write(content)
                                status.write(
                                    f"Content saved successfully to {persistent_file_path}"
                                )

                                # Prepare the command using the persistent file path
                                file_arg = persistent_file_path
                                command = [
                                    sys.executable,
                                    script_path,
                                    "--file",
                                    file_arg,
                                ]

                            except requests.exceptions.RequestException as e:
                                status.update(
                                    label="Download Failed!",
                                    state="error",
                                    expanded=True,
                                )
                                st.session_state.llms_processing_error = (
                                    f"Error downloading file from {url_to_process}: {e}"
                                )
                                st.session_state.llms_processing_complete = True
                                write_to_log(f"Error downloading {url_to_process}: {e}")
                            except OSError as e:
                                status.update(
                                    label="File Saving Failed!",
                                    state="error",
                                    expanded=True,
                                )
                                st.session_state.llms_processing_error = (
                                    f"Error saving file to {persistent_file_path}: {e}"
                                )
                                st.session_state.llms_processing_complete = True
                                write_to_log(
                                    f"Error saving file {persistent_file_path}: {e}"
                                )
                            except Exception as e:
                                status.update(
                                    label="Error preparing file!",
                                    state="error",
                                    expanded=True,
                                )
                                st.session_state.llms_processing_error = f"An unexpected error occurred while preparing the file: {e}"
                                st.session_state.llms_processing_complete = True
                                write_to_log(
                                    f"Unexpected error preparing file from {url_to_process}: {e}"
                                )

                        # Proceed only if download and file saving were successful
                        if command and persistent_file_path:
                            with process_placeholder.status(
                                "Processing documentation...", expanded=True
                            ) as status:
                                try:
                                    st.write(f"Running command: `{' '.join(command)}`")

                                    # Initialize output and error storage
                                    st.session_state.llms_processing_output = ""
                                    st.session_state.llms_processing_error = ""

                                    # Create placeholders for real-time logs
                                    log_output_placeholder = st.empty()
                                    log_error_placeholder = st.empty()

                                    # Create a text area for stdout
                                    log_output = log_output_placeholder.text_area(
                                        "Output Log (Real-time):",
                                        value="",
                                        height=200,
                                        key="search_result_output_log",
                                    )

                                    # Create a placeholder for stderr (will only show if errors occur)
                                    log_error = None

                                    # Start the process with Popen
                                    process = subprocess.Popen(
                                        command,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        text=True,
                                        encoding="utf-8",
                                        errors="replace",  # Handle potential encoding errors
                                    )

                                    # Capture the process output
                                    result = process.communicate()

                                    # Store output/error in session state for display later
                                    st.session_state.llms_processing_output = result[0]
                                    st.session_state.llms_processing_error = result[1]

                                    if process.returncode == 0:
                                        status.update(
                                            label="Processing complete!",
                                            state="complete",
                                            expanded=False,
                                        )
                                        st.session_state.llms_processing_complete = True
                                        write_to_log(
                                            f"Successfully processed {persistent_file_path} (from URL: {url_to_process})."
                                        )
                                    else:
                                        status.update(
                                            label="Processing failed!",
                                            state="error",
                                            expanded=True,
                                        )
                                        st.session_state.llms_processing_complete = True
                                        write_to_log(
                                            f"Error processing {persistent_file_path} (from URL: {url_to_process}). Return code: {process.returncode}"
                                        )

                                except FileNotFoundError:
                                    status.update(
                                        label="Error: Script not found!",
                                        state="error",
                                        expanded=True,
                                    )
                                    st.session_state.llms_processing_error = f"Error: The script '{script_path}' was not found. Make sure it's in the correct location."
                                    st.session_state.llms_processing_complete = True
                                    write_to_log(
                                        f"Error running processing script: FileNotFoundError - {script_path}"
                                    )
                                except Exception as e:
                                    status.update(
                                        label="An unexpected error occurred during processing!",
                                        state="error",
                                        expanded=True,
                                    )
                                    st.session_state.llms_processing_error = f"An unexpected error occurred during processing: {str(e)}"
                                    st.session_state.llms_processing_complete = True
                                    write_to_log(
                                        f"Unexpected error processing {persistent_file_path} (from URL: {url_to_process}): {e}"
                                    )

                            # Rerun to display results outside the status context
                            st.rerun()
                        elif not st.session_state.get(
                            "llms_processing_complete"
                        ):  # If command wasn't set due to download error, but not marked complete yet
                            st.error(
                                "Processing could not start due to an error during file preparation."
                            )
                            st.session_state.llms_processing_complete = (
                                True  # Ensure it's marked complete
                            )
                            st.rerun()  # Rerun to show the error message clearly

    with doc_tabs[2]:  # Future Sources Tab (original content moved here)
        st.info("Additional documentation sources will be available in future updates.")
