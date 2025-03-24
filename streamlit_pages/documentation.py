import streamlit as st
import uuid
import asyncio
import time
import os
import sys
from typing import Dict, List, Optional, Tuple, Any

# Add the project root to the Python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

# Try to import optional dependencies, but don't error if they're not available
# The documentation_crawler.py module has fallbacks for these
try:
    import html2text
    html2text_available = True
except ImportError:
    html2text_available = False

try:
    from bs4 import BeautifulSoup
    bs4_available = True
except ImportError:
    bs4_available = False

try:
    from lxml import etree
    lxml_available = True
except ImportError:
    lxml_available = False

# Import documentation crawler modules with error handling
try:
    from archon.documentation_crawler import DocumentationSource, DocumentationCrawler, CrawlProgressTracker, get_sitemap_urls
    # Import the registry directly as an already instantiated object
    from archon.documentation_crawler import registry
except ImportError as e:
    st.error(f"Error importing from documentation_crawler: {e}")

from utils.utils import get_clients, get_env_var, create_new_tab_button

def documentation_tab(supabase):
    """
    Documentation tab for crawling and searching documentation.
    """
    if not supabase:
        st.warning("Supabase is not configured. Please add your Supabase credentials in the Environment tab.")
        return

    st.header("Documentation")
    
    # Check if the database is configured
    supabase_url = get_env_var("SUPABASE_URL")
    supabase_key = get_env_var("SUPABASE_SERVICE_KEY")
    
    if not supabase_url or not supabase_key:
        st.warning("⚠️ Supabase is not configured. Please set up your environment variables first.")
        create_new_tab_button("Go to Environment Section", "Environment", key="goto_env_from_docs")
        return
    
    # Create tabs for existing sources and for adding new sources
    existing_tabs, add_source_tab = st.tabs(["Existing Sources", "Add New Source"])
    
    with existing_tabs:
        # Get all registered sources
        all_sources = registry.get_all_sources()
        
        if not all_sources:
            st.warning("No documentation sources registered. Add one in the 'Add New Source' tab.")
        else:
            # Create tabs for each source
            source_tabs = st.tabs([source.name for source in all_sources])
            
            # Initialize session state
            for source in all_sources:
                source_id = source.id
                if f"crawl_progress_{source_id}" not in st.session_state:
                    st.session_state[f"crawl_progress_{source_id}"] = None
                if f"crawl_status_{source_id}" not in st.session_state:
                    st.session_state[f"crawl_status_{source_id}"] = "idle"
                if f"crawl_logs_{source_id}" not in st.session_state:
                    st.session_state[f"crawl_logs_{source_id}"] = []
                if f"clear_status_{source_id}" not in st.session_state:
                    st.session_state[f"clear_status_{source_id}"] = "idle"
            
            # Display content for each source
            for idx, (source, tab) in enumerate(zip(all_sources, source_tabs)):
                source_id = source.id
                with tab:
                    st.subheader(f"{source.name} Documentation")
                    st.write(source.description)
                    
                    # Metrics row with counts
                    try:
                        # Count indexed docs - FIX: Use metadata->>source instead of source column
                        count_result = supabase.table("site_pages").select("*", count="exact").eq("metadata->>source", source_id).execute()
                        indexed_count = count_result.count
                    except Exception as e:
                        st.error(f"Error counting indexed docs: {str(e)}")
                        indexed_count = 0
                    
                    st.write(f"**Indexed Documents**: {indexed_count}")
                    
                    # Buttons for crawling and clearing
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # Start crawl button
                        start_crawl = st.button(f"Start Crawling {source.name}", key=f"start_crawl_{source_id}")
                        if start_crawl:
                            if st.session_state[f"crawl_status_{source_id}"] == "running":
                                st.warning("Crawl already in progress.")
                            else:
                                st.session_state[f"crawl_status_{source_id}"] = "running"
                                st.session_state[f"crawl_logs_{source_id}"] = []
                                
                                # Initialize crawler
                                crawler = DocumentationCrawler(
                                    source=source,
                                )
                                
                                # Start the crawl in a separate thread
                                tracker = crawler.start_crawler()
                                st.session_state[f"crawl_progress_{source_id}"] = tracker
                    
                    with col2:
                        # Clear existing button
                        clear_existing = st.button(f"Clear Existing {source.name}", key=f"clear_{source_id}")
                        if clear_existing:
                            if st.session_state[f"clear_status_{source_id}"] == "running":
                                st.warning("Clear operation already in progress.")
                            else:
                                st.session_state[f"clear_status_{source_id}"] = "running"
                                try:
                                    # Create a crawler with the source and clear records
                                    crawler = DocumentationCrawler(source=source)
                                    crawler.clear_existing_records()
                                    st.session_state[f"clear_status_{source_id}"] = "completed"
                                except Exception as e:
                                    st.error(f"Error clearing records: {str(e)}")
                                    st.session_state[f"clear_status_{source_id}"] = "error"
                    
                    with col3:
                        # View sample button if documents are indexed
                        if indexed_count > 0:
                            view_sample = st.button(f"View Sample {source.name}", key=f"view_{source_id}")
                            if view_sample:
                                try:
                                    # FIX: Use metadata->>source instead of source column
                                    sample_data = supabase.table("site_pages").select("*").eq("metadata->>source", source_id).limit(5).execute()
                                    sample_rows = sample_data.data
                                    st.json(sample_rows)
                                except Exception as e:
                                    st.error(f"Error fetching sample data: {str(e)}")
                    
                    # Display crawl status and progress
                    if st.session_state[f"crawl_status_{source_id}"] == "running":
                        tracker = st.session_state[f"crawl_progress_{source_id}"]
                        if tracker:
                            # Display progress metrics
                            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                            status = tracker.get_status()
                            metrics_col1.metric("URLs Found", status["urls_found"])
                            metrics_col2.metric("Processed", status["urls_processed"])
                            metrics_col3.metric("Succeeded", status["urls_succeeded"])
                            metrics_col4.metric("Failed", status["urls_failed"])
                            
                            # Display progress bar
                            if status["urls_found"] > 0:
                                progress = status["urls_processed"] / status["urls_found"]
                                st.progress(progress)
                            
                            # Display logs
                            if status["logs"]:
                                st.session_state[f"crawl_logs_{source_id}"] = status["logs"][-10:]  # Keep last 10 logs
                                
                            with st.expander("Logs", expanded=True):
                                for log in st.session_state[f"crawl_logs_{source_id}"]:
                                    st.text(log)
                            
                            # Check if crawl is completed
                            if not status["is_running"]:
                                st.success(f"Crawl completed! Processed {status['urls_processed']} URLs with {status['urls_succeeded']} successes and {status['urls_failed']} errors.")
                                st.session_state[f"crawl_status_{source_id}"] = "completed"
                            else:
                                st.rerun()
                    
                    # Display clear status
                    if st.session_state[f"clear_status_{source_id}"] == "completed":
                        st.success(f"Successfully cleared existing {source.name} documentation.")
                        st.session_state[f"clear_status_{source_id}"] = "idle"  # Reset status
                    elif st.session_state[f"clear_status_{source_id}"] == "error":
                        st.error(f"Error clearing {source.name} documentation.")
                        st.session_state[f"clear_status_{source_id}"] = "idle"  # Reset status
    
    # Tab for adding a new source
    with add_source_tab:
        st.subheader("Add a New Documentation Source")
        st.write("Enter the details for a new documentation source to crawl.")
        
        with st.form("add_source_form"):
            # Source ID (auto-generated but shown to user)
            source_id = str(uuid.uuid4())[:8]
            st.text_input("Source ID (auto-generated)", source_id, disabled=True)
            
            # Source name
            source_name = st.text_input("Source Name", placeholder="e.g., FastAPI Docs")
            
            # Source description
            source_description = st.text_area("Source Description", 
                                            placeholder="e.g., FastAPI is a modern, fast web framework for building APIs with Python.")
            
            # Sitemap URL
            sitemap_url = st.text_input("Sitemap URL", placeholder="e.g., https://fastapi.tiangolo.com/sitemap.xml")
            
            # Submit button
            submitted = st.form_submit_button("Add Source")
            
            if submitted:
                if not source_name or not sitemap_url:
                    st.error("Source name and sitemap URL are required!")
                else:
                    # Test the sitemap URL
                    try:
                        # Create a function to get URLs from the sitemap
                        def url_fetcher():
                            return get_sitemap_urls(sitemap_url)
                        
                        # Test fetching some URLs
                        urls = url_fetcher()
                        if not urls:
                            st.error(f"Could not fetch any URLs from the sitemap at {sitemap_url}")
                        else:
                            # Register the new source
                            new_source = DocumentationSource(
                                id=source_id,
                                name=source_name,
                                description=source_description or f"{source_name} documentation",
                                url_fetcher=url_fetcher,
                                sitemap_url=sitemap_url
                            )
                            
                            # Add to registry
                            registry.register(new_source)
                            
                            st.success(f"Successfully added {source_name} as a new documentation source!")
                            st.info("Refresh the page to see the new source in the 'Existing Sources' tab.")
                    except Exception as e:
                        st.error(f"Error testing sitemap URL: {str(e)}")

    # Section on how to add new sources programmatically
    with st.expander("How to add documentation sources programmatically"):
        st.write("""
        You can also add new documentation sources by editing the `documentation_crawler.py` file.
        Here's an example of how to add a new source:
        
        ```python
        from archon.documentation_crawler import DocumentationSource, registry

        def get_my_docs_urls():
            # Function to fetch URLs from your documentation sitemap
            # Example:
            from archon.documentation_crawler import get_sitemap_urls
            return get_sitemap_urls("https://mydocs.example.com/sitemap.xml")
            
        # Create and register the source
        my_docs_source = DocumentationSource(
            id="my-docs",
            name="My Documentation",
            description="Description of my documentation",
            url_fetcher=get_my_docs_urls,
            sitemap_url="https://mydocs.example.com/sitemap.xml"
        )
        
        # Register the source
        registry.register(my_docs_source)
        ```
        """)