import streamlit as st
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from archon.crawl_pydantic_ai_docs import start_crawl_with_requests as start_pydantic_crawl, clear_existing_records as clear_pydantic_records
from archon.crawl_craw4ai_docs import start_crawl_with_requests as start_crawl4ai_crawl, clear_existing_records as clear_crawl4ai_records
from archon.crawl_langchain_python_docs import start_crawl_with_requests as start_langchain_crawl, clear_existing_records as clear_langchain_records
from utils.utils import get_env_var, create_new_tab_button

def documentation_tab(supabase_client):
    """Display the documentation interface"""
    st.header("Documentation")
    
    # Create tabs for different documentation sources
    doc_tabs = st.tabs(["Pydantic AI Docs", "Crawl4AI Docs", "Langchain Python Docs", "Future Sources"])
    
    # Check if the database is configured
    supabase_url = get_env_var("SUPABASE_URL")
    supabase_key = get_env_var("SUPABASE_SERVICE_KEY")
    
    if not supabase_url or not supabase_key:
        with doc_tabs[0], doc_tabs[1], doc_tabs[2]:
            st.warning("⚠️ Supabase is not configured. Please set up your environment variables first.")
            create_new_tab_button("Go to Environment Section", "Environment", key="goto_env_from_docs")
    else:
        # Initialize common session state variables
        if "pydantic_crawl_tracker" not in st.session_state:
            st.session_state.pydantic_crawl_tracker = None
        
        if "pydantic_crawl_status" not in st.session_state:
            st.session_state.pydantic_crawl_status = None
            
        if "crawl4ai_crawl_tracker" not in st.session_state:
            st.session_state.crawl4ai_crawl_tracker = None
        
        if "crawl4ai_crawl_status" not in st.session_state:
            st.session_state.crawl4ai_crawl_status = None
                
        if "last_update_time" not in st.session_state:
            st.session_state.last_update_time = time.time()
        
        # === PYDANTIC AI DOCS TAB ===
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
            
            # Create columns for the buttons
            col1, col2 = st.columns(2)
            
            with col1:
                # Button to start crawling
                if st.button("Crawl Pydantic AI Docs", key="crawl_pydantic") and not (st.session_state.pydantic_crawl_tracker and st.session_state.pydantic_crawl_tracker.is_running):
                    try:
                        # Define a callback function to update the session state
                        def update_progress(status):
                            st.session_state.pydantic_crawl_status = status
                        
                        # Start the crawling process in a separate thread
                        st.session_state.pydantic_crawl_tracker = start_pydantic_crawl(update_progress)
                        st.session_state.pydantic_crawl_status = st.session_state.pydantic_crawl_tracker.get_status()
                        
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
                            clear_pydantic_records()
                            st.success("✅ Successfully cleared existing Pydantic AI docs from the database.")
                            
                            # Force a rerun to update the UI
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error clearing Pydantic AI docs: {str(e)}")
            
            # Display crawling progress if a crawl is in progress or has completed
            if st.session_state.pydantic_crawl_tracker:
                # Create a container for the progress information
                progress_container = st.container()
                
                with progress_container:
                    # Get the latest status
                    current_time = time.time()
                    # Update status every second
                    if current_time - st.session_state.last_update_time >= 1:
                        st.session_state.pydantic_crawl_status = st.session_state.pydantic_crawl_tracker.get_status()
                        st.session_state.last_update_time = current_time
                    
                    status = st.session_state.pydantic_crawl_status
                    
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

        # === CRAWL4AI DOCS TAB ===
        with doc_tabs[1]:
            st.subheader("Crawl4AI Documentation")
            st.markdown("""
            This section allows you to crawl and index the Crawl4AI documentation.
            The crawler will:
            
            1. Fetch URLs from the Crawl4AI sitemap
            2. Crawl each page and extract content
            3. Split content into chunks
            4. Generate embeddings for each chunk
            5. Store the chunks in the Supabase database
            
            This process may take several minutes depending on the number of pages.
            """)
            
            # Create columns for the buttons
            col1, col2 = st.columns(2)
            
            with col1:
                # Button to start crawling
                if st.button("Crawl Crawl4AI Docs", key="crawl_crawl4ai") and not (st.session_state.crawl4ai_crawl_tracker and st.session_state.crawl4ai_crawl_tracker.is_running):
                    try:
                        # Define a callback function to update the session state
                        def update_crawl4ai_progress(status):
                            st.session_state.crawl4ai_crawl_status = status
                        
                        # Start the crawling process in a separate thread
                        st.session_state.crawl4ai_crawl_tracker = start_crawl4ai_crawl(update_crawl4ai_progress)
                        st.session_state.crawl4ai_crawl_status = st.session_state.crawl4ai_crawl_tracker.get_status()
                        
                        # Force a rerun to start showing progress
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error starting crawl: {str(e)}")
            
            with col2:
                # Button to clear existing Crawl4AI docs
                if st.button("Clear Crawl4AI Docs", key="clear_crawl4ai"):
                    with st.spinner("Clearing existing Crawl4AI docs..."):
                        try:
                            # Run the function to clear records
                            clear_crawl4ai_records()
                            st.success("✅ Successfully cleared existing Crawl4AI docs from the database.")
                            
                            # Force a rerun to update the UI
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error clearing Crawl4AI docs: {str(e)}")
            
            # Display crawling progress if a crawl is in progress or has completed
            if st.session_state.crawl4ai_crawl_tracker:
                # Create a container for the progress information
                progress_container = st.container()
                
                with progress_container:
                    # Get the latest status
                    current_time = time.time()
                    # Update status every second
                    if current_time - st.session_state.last_update_time >= 1:
                        st.session_state.crawl4ai_crawl_status = st.session_state.crawl4ai_crawl_tracker.get_status()
                        st.session_state.last_update_time = current_time
                    
                    status = st.session_state.crawl4ai_crawl_status
                    
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
                # Query the count of Crawl4AI docs
                result = supabase_client.table("site_pages").select("count", count="exact").eq("metadata->>source", "crawl4ai_docs").execute()
                count = result.count if hasattr(result, "count") else 0
                
                # Display the count
                st.metric("Crawl4AI Docs Chunks", count)
                
                # Add a button to view the data
                if count > 0 and st.button("View Indexed Data", key="view_crawl4ai_data"):
                    # Query a sample of the data
                    sample_data = supabase_client.table("site_pages").select("url,title,summary,chunk_number").eq("metadata->>source", "crawl4ai_docs").limit(10).execute()
                    
                    # Display the sample data
                    st.dataframe(sample_data.data)
                    st.info("Showing up to 10 sample records. The database contains more records.")
            except Exception as e:
                st.error(f"Error querying database: {str(e)}")

        # === LANGCHAIN PYTHON DOCS TAB ===
        with doc_tabs[2]:
            st.subheader("Langchain Python Documentation")
            
            st.write("""
            This will crawl the Langchain Python documentation from the official website.
            The process retrieves the documentation, processes it into chunks, and stores it in your Supabase database.
            """)

            # Create columns for action buttons
            langchain_col1, langchain_col2 = st.columns(2)
            
            with langchain_col1:
                langchain_crawl_button = st.button("Start Langchain Python Docs Crawl", key="start_langchain_crawl", 
                                            use_container_width=True, disabled=not (supabase_url and supabase_key))
            
            with langchain_col2:
                langchain_clear_button = st.button("Clear Existing Langchain Data", key="clear_langchain_data", 
                                          type="secondary", use_container_width=True, 
                                          disabled=not (supabase_url and supabase_key))
            
            langchain_status_container = st.empty()
            langchain_progress_bar = st.empty()
            langchain_log_container = st.empty()

            if langchain_clear_button:
                with st.spinner("Clearing existing Langchain documentation records..."):
                    try:
                        records_cleared = clear_langchain_records(supabase_client)
                        st.success(f"✅ Successfully cleared {records_cleared} Langchain documentation records.")
                    except Exception as e:
                        st.error(f"❌ Error clearing Langchain documentation records: {str(e)}")
            
            if langchain_crawl_button:
                with st.spinner("Starting Langchain Python documentation crawl..."):
                    try:
                        # Display initial progress
                        langchain_progress_bar.progress(0, text="Preparing to crawl Langchain Python documentation...")
                        langchain_log_container.code("Starting crawler...", language="text")
                        
                        # Define the update progress callback
                        def update_langchain_progress(status):
                            """Handle progress updates from the crawl process."""
                            progress_percentage = status.get("progress_percentage", 0)
                            progress_text = f"Progress: {progress_percentage:.1f}%"
                            
                            # Update the progress bar
                            langchain_progress_bar.progress(min(progress_percentage / 100, 1.0), text=progress_text)
                            
                            # Update the status container
                            status_text = ""
                            if status.get("is_running", False):
                                status_text += f"🔄 Crawling in progress...\n"
                            elif status.get("is_completed", False):
                                if status.get("is_successful", False):
                                    status_text += f"✅ Crawl completed successfully!\n"
                                else:
                                    status_text += f"⚠️ Crawl completed with errors.\n"
                            
                            status_text += f"URLs found: {status.get('urls_found', 0)}\n"
                            status_text += f"URLs processed: {status.get('urls_processed', 0)}\n"
                            status_text += f"URLs succeeded: {status.get('urls_succeeded', 0)}\n"
                            status_text += f"URLs failed: {status.get('urls_failed', 0)}\n"
                            status_text += f"Chunks stored: {status.get('chunks_stored', 0)}\n"
                            
                            langchain_status_container.text(status_text)
                            
                            # Update the log container with the latest logs
                            logs = status.get("logs", [])
                            if logs:
                                log_text = "\n".join(logs[-15:])  # Show only the latest 15 logs
                                langchain_log_container.code(log_text, language="text")
                        
                        # Start the crawl with progress updates
                        langchain_crawler_task = start_langchain_crawl(
                            supabase_client,
                            progress_callback=update_langchain_progress
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Error starting Langchain Python documentation crawl: {str(e)}")

        # === FUTURE SOURCES TAB ===
        with doc_tabs[3]:
            st.info("Additional documentation sources will be available in future updates.")