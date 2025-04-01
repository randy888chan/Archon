import streamlit as st
import time
import sys
import os
import subprocess
from pathlib import Path
import requests
import tempfile # Keep for now, might be used elsewhere, but remove specific usage below
import re # For sanitization

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from archon.crawl_pydantic_ai_docs import start_crawl_with_requests, clear_existing_records
from utils.utils import get_env_var, create_new_tab_button, write_to_log
def documentation_tab(supabase_client):
    """Display the documentation interface"""
    st.header("Documentation")
    
    # Create tabs for different documentation sources
    doc_tabs = st.tabs(["Pydantic AI Docs", "Framework Docs (llms.txt)", "Future Sources"])
    
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
        
        # Check if the database is configured
        supabase_url = get_env_var("SUPABASE_URL")
        supabase_key = get_env_var("SUPABASE_SERVICE_KEY")
        
        if not supabase_url or not supabase_key:
            st.warning("⚠️ Supabase is not configured. Please set up your environment variables first.")
            create_new_tab_button("Go to Environment Section", "Environment", key="goto_env_from_docs")
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
                if st.button("Crawl Pydantic AI Docs", key="crawl_pydantic") and not (st.session_state.crawl_tracker and st.session_state.crawl_tracker.is_running):
                    try:
                        # Define a callback function to update the session state
                        def update_progress(status):
                            st.session_state.crawl_status = status
                        
                        # Start the crawling process in a separate thread
                        st.session_state.crawl_tracker = start_crawl_with_requests(update_progress)
                        st.session_state.crawl_status = st.session_state.crawl_tracker.get_status()
                        
                        # Force a rerun to start showing progress
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error starting crawl: {str(e)}")
            
            with col2:
                # Button to clear existing Pydantic AI docs
                if st.button("Clear Pydantic AI Docs", key="clear_pydantic"):
                    with st.spinner("Clearing existing Pydantic AI docs..."):
                        try:
                            # Run the function to clear records
                            clear_existing_records()
                            st.success("✅ Successfully cleared existing Pydantic AI docs from the database.")
                            
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
                        st.session_state.crawl_status = st.session_state.crawl_tracker.get_status()
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
                            logs_text = "\n".join(status["logs"][-20:])  # Show last 20 logs
                            st.code(logs_text)
                        else:
                            st.code("No logs available yet...")
                    
                    # Show completion message
                    if status and not status["is_running"] and status["end_time"]:
                        if status["urls_failed"] == 0:
                            st.success("✅ Crawling process completed successfully!")
                        else:
                            st.warning(f"⚠️ Crawling process completed with {status['urls_failed']} failed URLs.")
                
                # Auto-refresh while crawling is in progress
                if not status or status["is_running"]:
                    st.rerun()
        
        # Display database statistics
        st.subheader("Database Statistics")
        try:            
            # Query the count of Pydantic AI docs
            result = supabase_client.table("site_pages").select("count", count="exact").eq("metadata->>source", "pydantic_ai_docs").execute()
            count = result.count if hasattr(result, "count") else 0
            
            # Display the count
            st.metric("Pydantic AI Docs Chunks", count)
            
            # Add a button to view the data
            if count > 0 and st.button("View Indexed Data", key="view_pydantic_data"):
                # Query a sample of the data
                sample_data = supabase_client.table("site_pages").select("url,title,summary,chunk_number").eq("metadata->>source", "pydantic_ai_docs").limit(10).execute()
                
                # Display the sample data
                st.dataframe(sample_data.data)
                st.info("Showing up to 10 sample records. The database contains more records.")
        except Exception as e:
            st.error(f"Error querying database: {str(e)}")
    
    with doc_tabs[1]: # Framework Docs (llms.txt) Tab
        st.subheader("Framework Documentation (llms.txt)")
        st.markdown("""
        Process pre-formatted `llms.txt` files containing documentation for various frameworks.
        This process parses the file, chunks the content hierarchically, generates embeddings,
        and stores the data in the `hierarchical_nodes` table in the database.

        **Note:** Ensure the 'Hierarchical Nodes' table is selected in the Database configuration tab before processing.
        """)

        # Check if the database is configured
        supabase_url = get_env_var("SUPABASE_URL")
        supabase_key = get_env_var("SUPABASE_SERVICE_KEY")

        if not supabase_url or not supabase_key:
            st.warning("⚠️ Supabase is not configured. Please set up your environment variables first.")
            create_new_tab_button("Go to Environment Section", "Environment", key="goto_env_from_llms_docs")
        else:
            # Framework Selection
            framework_options = {
                "--- Select a framework ---": None,
                "Pydantic AI (ai.pydantic.dev)": "https://ai.pydantic.dev/llms.txt",
                "LangGraph (langchain-ai.github.io)": "https://langchain-ai.github.io/langgraph/llms-full.txt",
                "CrewAI (docs.crewai.com)": "https://docs.crewai.com/llms-full.txt"
            }
            selected_framework_display = st.selectbox(
                "Select Framework Documentation (llms.txt):",
                options=list(framework_options.keys()),
                key="llms_framework_select"
            )
            selected_framework_file = framework_options[selected_framework_display]


            # Text input for custom URL
            custom_url = st.text_input(
                "Or Enter Custom URL to llms.txt file:",
                key="llms_custom_url_input",
                placeholder="https://example.com/path/to/your_llms.txt"
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
                key="llms_file_uploader"
            )

            if uploaded_file is not None:
                # Process uploaded file immediately upon upload (save it)
                original_filename = uploaded_file.name
                # Use the existing sanitize_filename function defined below/above
                sanitized_filename = sanitize_filename(original_filename)
                docs_dir = "docs"
                saved_file_path = os.path.join(docs_dir, sanitized_filename)
                save_status = st.empty() # Placeholder for status messages

                try:
                    save_status.info(f"Saving '{original_filename}' as '{sanitized_filename}'...")
                    # Ensure docs directory exists
                    os.makedirs(docs_dir, exist_ok=True)

                    # Read content and save
                    # Use getvalue() which returns bytes, then decode
                    file_content = uploaded_file.getvalue().decode("utf-8")
                    with open(saved_file_path, "w", encoding="utf-8") as f:
                        f.write(file_content)

                    save_status.success(f"✅ File '{original_filename}' saved as '{saved_file_path}'. Ready for processing.")
                    st.session_state.uploaded_file_path = saved_file_path # Store path for the button
                    st.session_state.uploaded_file_original_name = original_filename
                    # uploaded_file = None # Resetting local var might not be enough

                except Exception as e:
                    save_status.error(f"❌ Error saving uploaded file: {e}")
                    st.session_state.uploaded_file_path = None # Reset path on error
                    st.session_state.uploaded_file_original_name = None
                    write_to_log(f"Error saving uploaded file {original_filename}: {e}")

            # --- Button to process the *saved* uploaded file ---
            if st.session_state.uploaded_file_path:
                # st.markdown("---") # Separator - removed, button serves as visual break
                if st.button(f"Process Uploaded File: '{st.session_state.uploaded_file_original_name}'", key="process_uploaded_llms_file"):
                     # Check Supabase config and table selection again (important!)
                    supabase_url = get_env_var("SUPABASE_URL")
                    supabase_key = get_env_var("SUPABASE_SERVICE_KEY")
                    docs_retrieval_table = get_env_var("DOCS_RETRIEVAL_TABLE")

                    if not supabase_url or not supabase_key:
                         st.warning("⚠️ Supabase is not configured. Please set up your environment variables first.")
                         # create_new_tab_button("Go to Environment Section", "Environment", key="goto_env_from_uploaded_llms")
                    elif docs_retrieval_table != "hierarchical_nodes":
                         st.warning(f"⚠️ Incorrect table selected for documentation processing. Expected 'hierarchical_nodes' but found '{docs_retrieval_table}'. Please select 'Hierarchical Nodes' in the Database configuration tab.")
                    else:
                        # Proceed with processing the saved file
                        st.session_state.llms_processing_output = None # Clear previous output
                        st.session_state.llms_processing_error = None
                        st.session_state.llms_processing_complete = False # Reset completion flag for this specific action

                        script_path = "run_processing.py"
                        file_arg = st.session_state.uploaded_file_path # Use the saved path
                        command = [sys.executable, script_path, "--file", file_arg]
                        # Optional: Add document ID based on sanitized filename stem
                        # doc_id = Path(file_arg).stem
                        # command.extend(["--id", doc_id])

                        process_placeholder = st.empty()
                        # Use a unique key for the status context if needed, or rely on rerun clearing it
                        with process_placeholder.status(f"Processing uploaded file: {st.session_state.uploaded_file_original_name}...", expanded=True) as status:
                            try:
                                # --- START MODIFICATION: Pass Environment ---
                                current_env = os.environ.copy()
                                llm_provider = get_env_var("LLM_PROVIDER", "openai")
                                current_env["LLM_PROVIDER"] = llm_provider
                                current_env["EMBEDDING_MODEL"] = get_env_var("EMBEDDING_MODEL", "")
                                current_env["LLM_API_KEY"] = get_env_var("LLM_API_KEY", "")

                                if llm_provider.lower() != "openai":
                                     current_env["LLM_BASE_URL"] = get_env_var("LLM_BASE_URL", "")
                                elif "LLM_BASE_URL" in current_env:
                                     del current_env["LLM_BASE_URL"]

                                current_env["SUPABASE_URL"] = get_env_var("SUPABASE_URL", "")
                                current_env["SUPABASE_SERVICE_KEY"] = get_env_var("SUPABASE_SERVICE_KEY", "")
                                current_env["DOCS_RETRIEVAL_TABLE"] = get_env_var("DOCS_RETRIEVAL_TABLE", "")

                                status.write(f"Using Embedding Model: {current_env.get('EMBEDDING_MODEL', 'N/A')}")
                                status.write(f"Using LLM Provider: {current_env.get('LLM_PROVIDER', 'N/A')}")
                                if "LLM_BASE_URL" in current_env:
                                     status.write(f"Using Base URL: {current_env.get('LLM_BASE_URL')}")
                                # --- END MODIFICATION ---

                                status.write(f"Running command: `{' '.join(command)}`")
                                result = subprocess.run(
                                    command,
                                    capture_output=True,
                                    text=True,
                                    check=False, # Handle non-zero exit codes manually
                                    encoding='utf-8',
                                    errors='replace', # Handle potential encoding issues in output
                                    env=current_env # Pass the modified environment
                                )

                                # Store output/error in session state for display later
                                st.session_state.llms_processing_output = result.stdout
                                st.session_state.llms_processing_error = result.stderr

                                if result.returncode == 0:
                                    status.update(label="Processing complete!", state="complete", expanded=False)
                                    st.session_state.llms_processing_complete = True
                                    write_to_log(f"Successfully processed uploaded file: {file_arg} (Original: {st.session_state.uploaded_file_original_name}).")
                                    # Clear the saved path state after successful processing to hide the button
                                    st.session_state.uploaded_file_path = None
                                    st.session_state.uploaded_file_original_name = None
                                else:
                                    status.update(label="Processing failed!", state="error", expanded=True)
                                    st.session_state.llms_processing_complete = True
                                    write_to_log(f"Error processing uploaded file: {file_arg}. Return code: {result.returncode}\\nStderr: {result.stderr}")

                            except FileNotFoundError:
                                status.update(label="Error: Script not found!", state="error", expanded=True)
                                st.session_state.llms_processing_error = f"Error: The script '{script_path}' was not found. Ensure it exists and Python can execute it."
                                st.session_state.llms_processing_complete = True
                                write_to_log(f"Error running processing script for uploaded file: FileNotFoundError - {script_path}")
                            except Exception as e:
                                status.update(label="An unexpected error occurred during processing!", state="error", expanded=True)
                                st.session_state.llms_processing_error = f"An unexpected error occurred: {str(e)}"
                                st.session_state.llms_processing_complete = True
                                write_to_log(f"Unexpected error processing uploaded file {file_arg}: {e}")

                        # Rerun to display results outside the status context and update UI state (e.g., hide button)
                        st.rerun()

            st.markdown("---") # Separator before URL processing section
            st.subheader("Process Framework Docs from URL") # Add subheader for clarity

            # Processing Button
            # Helper function to sanitize URL/filename (moved inside for scope if preferred, or keep global)
            def sanitize_filename(source_identifier: str) -> str:
                """Sanitizes a URL or filename to be used as a valid filename."""
                # Remove protocol
                sanitized = re.sub(r'^https?://', '', source_identifier)
                # Replace invalid filename characters with underscores
                sanitized = re.sub(r'[\\/*?:"<>|]', '_', sanitized)
                # Replace other common separators or problematic chars
                sanitized = re.sub(r'[/.&=%]', '_', sanitized)
                # Limit length to avoid issues (e.g., 100 chars)
                sanitized = sanitized[:100]
                # Ensure it doesn't end with an underscore or period
                sanitized = sanitized.strip('_.')
                # Add .txt extension if not present (handle potential double extensions carefully)
                if not sanitized.lower().endswith('.txt'):
                     sanitized += ".txt"
                return sanitized

            # Determine which URL to process (custom input takes precedence)
            url_to_process = custom_url if custom_url else selected_framework_file
            button_label = f"Process Custom URL" if custom_url else f"Process Selected Framework: {selected_framework_display}"
            button_disabled = not url_to_process # Disable if neither custom nor selected URL is available

            if st.button(button_label, key="process_llms_docs", disabled=button_disabled):
                # The original check for selected_framework_file is now covered by url_to_process
                # Check if the correct table is selected in config
                docs_retrieval_table = get_env_var("DOCS_RETRIEVAL_TABLE")
                if docs_retrieval_table != "hierarchical_nodes":
                    st.warning(f"⚠️ Incorrect table selected for documentation processing. Expected 'hierarchical_nodes' but found '{docs_retrieval_table}'. Please select 'Hierarchical Nodes' in the Database configuration tab to process llms.txt files.")
                else:
                    st.session_state.llms_processing_output = None # Clear previous output
                    st.session_state.llms_processing_error = None
                    st.session_state.llms_processing_complete = False

                    script_path = "run_processing.py"
                    persistent_file_path = None # Path for the uniquely named file
                    command = None # Initialize command
                    docs_dir = "docs" # Define target directory

                    process_placeholder = st.empty()
                    with process_placeholder.status("Preparing documentation...", expanded=True) as status:
                        try:
                            # Ensure 'docs' directory exists
                            os.makedirs(docs_dir, exist_ok=True)
                            status.write(f"Ensured '{docs_dir}' directory exists.")

                            # Download the file from URL (using url_to_process)
                            status.write(f"Downloading documentation from {url_to_process}...")
                            response = requests.get(url_to_process, timeout=60) # Increased timeout
                            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                            content = response.text # Use text for utf-8 handling

                            # Generate unique filename and save persistently (using url_to_process)
                            unique_filename = sanitize_filename(url_to_process)
                            persistent_file_path = os.path.join(docs_dir, unique_filename)
                            status.write(f"Saving content to persistent file: {persistent_file_path}")
                            with open(persistent_file_path, 'w', encoding='utf-8') as f:
                                f.write(content)
                            status.write(f"Content saved successfully to {persistent_file_path}")

                            # Prepare the command using the persistent file path
                            file_arg = persistent_file_path
                            command = [sys.executable, script_path, "--file", file_arg]
                            # Optionally pass the original URL as document ID
                            # command.extend(["--id", url_to_process]) # Or use sanitized filename stem

                        except requests.exceptions.RequestException as e:
                            status.update(label="Download Failed!", state="error", expanded=True)
                            st.session_state.llms_processing_error = f"Error downloading file from {url_to_process}: {e}"
                            st.session_state.llms_processing_complete = True
                            write_to_log(f"Error downloading {url_to_process}: {e}")
                        except OSError as e:
                            status.update(label="File Saving Failed!", state="error", expanded=True)
                            st.session_state.llms_processing_error = f"Error saving file to {persistent_file_path}: {e}"
                            st.session_state.llms_processing_complete = True
                            write_to_log(f"Error saving file {persistent_file_path}: {e}")
                        except Exception as e:
                            status.update(label="Error preparing file!", state="error", expanded=True)
                            st.session_state.llms_processing_error = f"An unexpected error occurred while preparing the file: {e}"
                            st.session_state.llms_processing_complete = True
                            write_to_log(f"Unexpected error preparing file from {url_to_process}: {e}")

                    # Proceed only if download and file saving were successful
                    if command and persistent_file_path:
                        with process_placeholder.status("Processing documentation...", expanded=True) as status:
                            try:
                                # --- START MODIFICATION: Pass Environment ---
                                current_env = os.environ.copy()
                                llm_provider = get_env_var("LLM_PROVIDER", "openai")
                                current_env["LLM_PROVIDER"] = llm_provider
                                current_env["EMBEDDING_MODEL"] = get_env_var("EMBEDDING_MODEL", "")
                                current_env["LLM_API_KEY"] = get_env_var("LLM_API_KEY", "")

                                if llm_provider.lower() != "openai":
                                     current_env["LLM_BASE_URL"] = get_env_var("LLM_BASE_URL", "")
                                elif "LLM_BASE_URL" in current_env:
                                     del current_env["LLM_BASE_URL"]

                                current_env["SUPABASE_URL"] = get_env_var("SUPABASE_URL", "")
                                current_env["SUPABASE_SERVICE_KEY"] = get_env_var("SUPABASE_SERVICE_KEY", "")
                                current_env["DOCS_RETRIEVAL_TABLE"] = get_env_var("DOCS_RETRIEVAL_TABLE", "")

                                status.write(f"Using Embedding Model: {current_env.get('EMBEDDING_MODEL', 'N/A')}")
                                status.write(f"Using LLM Provider: {current_env.get('LLM_PROVIDER', 'N/A')}")
                                if "LLM_BASE_URL" in current_env:
                                     status.write(f"Using Base URL: {current_env.get('LLM_BASE_URL')}")
                                # --- END MODIFICATION ---

                                status.write(f"Running command: `{' '.join(command)}`")
                                # Use subprocess.run with the modified environment
                                result = subprocess.run(
                                    command,
                                    capture_output=True,
                                    text=True,
                                    check=False, # Don't raise exception on non-zero exit code
                                    encoding='utf-8',
                                    errors='replace', # Handle potential encoding errors
                                    env=current_env # Pass the modified environment
                                )

                                st.session_state.llms_processing_output = result.stdout
                                st.session_state.llms_processing_error = result.stderr

                                if result.returncode == 0:
                                    status.update(label="Processing complete!", state="complete", expanded=False)
                                    st.session_state.llms_processing_complete = True
                                    write_to_log(f"Successfully processed {persistent_file_path} (from URL: {url_to_process}).")
                                else:
                                    status.update(label="Processing failed!", state="error", expanded=True)
                                    st.session_state.llms_processing_complete = True
                                    write_to_log(f"Error processing {persistent_file_path} (from URL: {url_to_process}). Return code: {result.returncode}")

                            except FileNotFoundError:
                                status.update(label="Error: Script not found!", state="error", expanded=True)
                                st.session_state.llms_processing_error = f"Error: The script '{script_path}' was not found. Make sure it's in the correct location."
                                st.session_state.llms_processing_complete = True
                                write_to_log(f"Error running processing script: FileNotFoundError - {script_path}")
                            except Exception as e:
                                status.update(label="An unexpected error occurred during processing!", state="error", expanded=True)
                                st.session_state.llms_processing_error = f"An unexpected error occurred during processing: {str(e)}"
                                st.session_state.llms_processing_complete = True
                                write_to_log(f"Unexpected error processing {persistent_file_path} (from URL: {url_to_process}): {e}")

                        # Rerun to display results outside the status context
                        st.rerun()
                    elif not st.session_state.get("llms_processing_complete"): # If command wasn't set due to download error, but not marked complete yet
                        st.error("Processing could not start due to an error during file preparation.")
                        st.session_state.llms_processing_complete = True # Ensure it's marked complete
                        st.rerun() # Rerun to show the error message clearly

                    # Rerun to display results outside the status context
                    st.rerun()

            # Display processing results
            if st.session_state.get("llms_processing_complete"):
                st.subheader("Processing Results")
                if st.session_state.get("llms_processing_output"):
                    st.text_area("Output Log:", st.session_state.llms_processing_output, height=300, key="llms_output_area")
                if st.session_state.get("llms_processing_error"):
                    st.error("Errors occurred during processing:")
                    st.code(st.session_state.llms_processing_error, language='bash')
                elif st.session_state.get("llms_processing_output"): # Check if output exists even if no error
                     st.success("✅ Processing finished.") # Simple success if no stderr

            # Database Statistics for hierarchical_nodes
            st.subheader("Database Statistics (`hierarchical_nodes`)")
            try:
                # Query total count
                total_result = supabase_client.table("hierarchical_nodes").select("count", count="exact").execute()
                total_count = total_result.count if hasattr(total_result, "count") else 0
                st.metric("Total Hierarchical Nodes", total_count)

                # Query counts per document_id
                if total_count > 0:
                     # Fetch distinct document_ids first
                    doc_ids_result = supabase_client.table("hierarchical_nodes").select("document_id", count="exact").execute()

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
                        st.info("Displaying counts per framework requires a database function (RPC) for efficiency. Showing total count for now.")

                        # Example of how it *could* look if counts were available:
                        # counts_per_doc = {'doc1': 150, 'doc2': 300} # Placeholder
                        # st.dataframe(counts_per_doc)


                        # Add a button to view sample data
                        if st.button("View Sample Hierarchical Data", key="view_hierarchical_data"):
                            sample_data = supabase_client.table("hierarchical_nodes").select("id, document_id, node_type, title, path, level").limit(10).execute()
                            st.dataframe(sample_data.data)
                            st.info("Showing up to 10 sample records.")

            except Exception as e:
                st.error(f"Error querying hierarchical_nodes table: {str(e)}")
                write_to_log(f"Error querying hierarchical_nodes table: {str(e)}")

    with doc_tabs[2]: # Future Sources Tab (original content moved here)
        st.info("Additional documentation sources will be available in future updates.")