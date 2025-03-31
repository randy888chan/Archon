import streamlit as st
import time
import sys
import os
import subprocess
from pathlib import Path

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
                "Pydantic AI (ai.pydantic.dev)": "docs/anthropic-llms.txt",
                "LangGraph (langchain-ai.github.io)": "docs/langgraph-llms-full.txt",
                "CrewAI (docs.crewai.com)": "docs/crewai-llms-full.txt"
            }
            selected_framework_display = st.selectbox(
                "Select Framework Documentation (llms.txt):",
                options=list(framework_options.keys()),
                key="llms_framework_select"
            )
            selected_framework_file = framework_options[selected_framework_display]

            # TODO for file uploader
            # TODO: Add st.file_uploader to allow users to upload their own llms.txt file.
            st.info("Support for uploading custom llms.txt files coming soon!")

            # Processing Button
            if st.button("Process Selected Framework Docs", key="process_llms_docs"):
                if not selected_framework_file:
                    st.warning("Please select a framework documentation file to process.")
                else:
                    # Check if the correct table is selected in config
                    docs_retrieval_table = get_env_var("DOCS_RETRIEVAL_TABLE")
                    if docs_retrieval_table != "hierarchical_nodes":
                        st.warning(f"⚠️ Incorrect table selected for documentation processing. Expected 'hierarchical_nodes' but found '{docs_retrieval_table}'. Please select 'Hierarchical Nodes' in the Database configuration tab to process llms.txt files.")
                    else:
                        st.session_state.llms_processing_output = None # Clear previous output
                        st.session_state.llms_processing_error = None
                        st.session_state.llms_processing_complete = False

                        script_path = "run_processing.py"
                        file_arg = selected_framework_file
                        command = [sys.executable, script_path, "--file", file_arg]

                        process_placeholder = st.empty()
                        with process_placeholder.status("Processing documentation...", expanded=True) as status:
                            try:
                                st.write(f"Running command: `{' '.join(command)}`")
                                # Use subprocess.run for simplicity here, consider Popen for streaming if needed later
                                result = subprocess.run(
                                    command,
                                    capture_output=True,
                                    text=True,
                                    check=False, # Don't raise exception on non-zero exit code
                                    encoding='utf-8',
                                    errors='replace' # Handle potential encoding errors
                                )

                                st.session_state.llms_processing_output = result.stdout
                                st.session_state.llms_processing_error = result.stderr

                                if result.returncode == 0:
                                    status.update(label="Processing complete!", state="complete", expanded=False)
                                    st.session_state.llms_processing_complete = True
                                    write_to_log(f"Successfully processed {file_arg}.")
                                else:
                                    status.update(label="Processing failed!", state="error", expanded=True)
                                    st.session_state.llms_processing_complete = True
                                    write_to_log(f"Error processing {file_arg}. Return code: {result.returncode}")

                            except FileNotFoundError:
                                status.update(label="Error: Script not found!", state="error", expanded=True)
                                st.session_state.llms_processing_error = f"Error: The script '{script_path}' was not found. Make sure it's in the correct location."
                                st.session_state.llms_processing_complete = True
                                write_to_log(f"Error running processing script: FileNotFoundError - {script_path}")
                            except Exception as e:
                                status.update(label="An unexpected error occurred!", state="error", expanded=True)
                                st.session_state.llms_processing_error = f"An unexpected error occurred: {str(e)}"
                                st.session_state.llms_processing_complete = True
                                write_to_log(f"Unexpected error processing {file_arg}: {e}")

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