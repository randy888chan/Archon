import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import get_env_var, save_env_var

@st.cache_data
def load_sql_template():
    """Load the SQL template file and cache it"""
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "utils", "site_pages.sql"), "r") as f:
        return f.read()


@st.cache_data
def load_llms_txt_sql():
    """Load the llms_txt SQL file and cache it"""
    try:
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "utils", "llms_txt.sql"), "r") as f:
            return f.read()
    except FileNotFoundError:
        return "Error: utils/llms_txt.sql not found."

def get_supabase_sql_editor_url(supabase_url):
    """Get the URL for the Supabase SQL Editor"""
    try:
        # Extract the project reference from the URL
        # Format is typically: https://<project-ref>.supabase.co
        if '//' in supabase_url and 'supabase' in supabase_url:
            parts = supabase_url.split('//')
            if len(parts) > 1:
                domain_parts = parts[1].split('.')
                if len(domain_parts) > 0:
                    project_ref = domain_parts[0]
                    return f"https://supabase.com/dashboard/project/{project_ref}/sql/new"
        
        # Fallback to a generic URL
        return "https://supabase.com/dashboard"
    except Exception:
        return "https://supabase.com/dashboard"

def show_manual_sql_instructions(sql, vector_dim, recreate=False):
    """Show instructions for manually executing SQL in Supabase"""
    st.info("### Manual SQL Execution Instructions")
    
    # Provide a link to the Supabase SQL Editor
    supabase_url = get_env_var("SUPABASE_URL")
    if supabase_url:
        dashboard_url = get_supabase_sql_editor_url(supabase_url)
        st.markdown(f"**Step 1:** [Open Your Supabase SQL Editor with this URL]({dashboard_url})")
    else:
        st.markdown("**Step 1:** Open your Supabase Dashboard and navigate to the SQL Editor")
    
    st.markdown("**Step 2:** Create a new SQL query")
    
    if recreate:
        st.markdown("**Step 3:** Copy and execute the following SQL:")
        drop_sql = f"DROP FUNCTION IF EXISTS match_site_pages(vector({vector_dim}), int, jsonb);\nDROP TABLE IF EXISTS site_pages CASCADE;"
        st.code(drop_sql, language="sql")
        
        st.markdown("**Step 4:** Then copy and execute this SQL:")
        st.code(sql, language="sql")
    else:
        st.markdown("**Step 3:** Copy and execute the following SQL:")
        st.code(sql, language="sql")
    
    st.success("After executing the SQL, return to this page and refresh to see the updated table status.")

def database_tab(supabase):
    """Display the database configuration interface"""
    st.header("Database Configuration")
    st.write("Set up and manage your Supabase database tables for Archon.")
    
    # Check if Supabase is configured
    if not supabase:
        st.error("Supabase is not configured. Please set your Supabase URL and Service Key in the Environment tab.")
        return
    
    # Site Pages Table Setup
    st.subheader("Site Pages Table")
    st.write("This table stores web page content and embeddings for semantic search.")
    
    # Add information about the table
    with st.expander("About the Site Pages Table", expanded=False):
        st.markdown("""
        This table is used to store:
        - Web page content split into chunks
        - Vector embeddings for semantic search
        - Metadata for filtering results
        
        The table includes:
        - URL and chunk number (unique together)
        - Title and summary of the content
        - Full text content
        - Vector embeddings for similarity search
        - Metadata in JSON format
        
        It also creates:
        - A vector similarity search function
        - Appropriate indexes for performance
        - Row-level security policies for Supabase
        """)
    
    # Check if the table already exists
    table_exists = False
    table_has_data = False
    
    try:
        # Try to query the table to see if it exists
        response = supabase.table("site_pages").select("id").limit(1).execute()
        table_exists = True
        
        # Check if the table has data
        count_response = supabase.table("site_pages").select("*", count="exact").execute()
        row_count = count_response.count if hasattr(count_response, 'count') else 0
        table_has_data = row_count > 0
        
        st.success("✅ The site_pages table already exists in your database.")
        if table_has_data:
            st.info(f"The table contains data ({row_count} rows).")
        else:
            st.info("The table exists but contains no data.")
    except Exception as e:
        error_str = str(e)
        if "relation" in error_str and "does not exist" in error_str:
            st.info("The site_pages table does not exist yet. You can create it below.")
        else:
            st.error(f"Error checking table status: {error_str}")
            st.info("Proceeding with the assumption that the table needs to be created.")
        table_exists = False
    
    # Vector dimensions selection
    st.write("### Vector Dimensions")
    st.write("Select the embedding dimensions based on your embedding model:")
    
    vector_dim = st.selectbox(
        "Embedding Dimensions",
        options=[1536, 768, 384, 1024],
        index=0,
        help="Use 1536 for OpenAI embeddings, 768 for nomic-embed-text with Ollama, or select another dimension based on your model."
    )
    
    # Get the SQL with the selected vector dimensions
    sql_template = load_sql_template()
    
    # Replace the vector dimensions in the SQL
    sql = sql_template.replace("vector(1536)", f"vector({vector_dim})")
    
    # Also update the match_site_pages function dimensions
    sql = sql.replace("query_embedding vector(1536)", f"query_embedding vector({vector_dim})")
    
    # Show the SQL
    with st.expander("View SQL", expanded=False):
        st.code(sql, language="sql")
    
    # Create table button
    if not table_exists:
        if st.button("Get Instructions for Creating Site Pages Table"):
            show_manual_sql_instructions(sql, vector_dim)
    else:
        # Option to recreate the table or clear data
        col1, col2 = st.columns(2)
        
        with col1:
            st.warning("⚠️ Recreating will delete all existing data.")
            if st.button("Get Instructions for Recreating Site Pages Table"):
                show_manual_sql_instructions(sql, vector_dim, recreate=True)
        
        with col2:
            if table_has_data:
                st.warning("⚠️ Clear all data but keep structure.")
                if st.button("Clear Table Data"):
                    try:
                        with st.spinner("Clearing table data..."):
                            # Use the Supabase client to delete all rows
                            response = supabase.table("site_pages").delete().neq("id", 0).execute()
                            st.success("✅ Table data cleared successfully!")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error clearing table data: {str(e)}")
                        # Fall back to manual SQL
                        truncate_sql = "TRUNCATE TABLE site_pages;"
                        st.code(truncate_sql, language="sql")
                        st.info("Execute this SQL in your Supabase SQL Editor to clear the table data.")
                        
                        # Provide a link to the Supabase SQL Editor
                        supabase_url = get_env_var("SUPABASE_URL")
                        if supabase_url:
                            dashboard_url = get_supabase_sql_editor_url(supabase_url)
                            st.markdown(f"[Open Your Supabase SQL Editor with this URL]({dashboard_url})")    

    st.divider()

    # Documentation Retrieval Table Selection
    st.subheader("Documentation Retrieval Table")
    st.write("Select which database table structure to use for documentation retrieval.")

    # Define options and the key for storing the preference
    retrieval_options = {
        "Site Pages (Default - Pydantic AI Docs)": "site_pages",
        "Hierarchical Nodes (llms.txt Framework Docs)": "hierarchical_nodes"
    }
    env_var_key = "DOCS_RETRIEVAL_TABLE"

    # Get current preference from env_vars.json, default to 'site_pages'
    current_preference_value = get_env_var(env_var_key) or "site_pages"

    # Find the display label corresponding to the stored value
    current_preference_label = "Site Pages (Default - Pydantic AI Docs)" # Default label
    for label, value in retrieval_options.items():
        if value == current_preference_value:
            current_preference_label = label
            break

    # Get the index of the current preference for the radio button
    options_list = list(retrieval_options.keys())
    try:
        current_index = options_list.index(current_preference_label)
    except ValueError:
        current_index = 0 # Default to first option if stored value is invalid

    # Callback function to save the selection
    def save_retrieval_preference():
        selected_label = st.session_state.docs_retrieval_table_radio
        selected_value = retrieval_options[selected_label]
        if save_env_var(env_var_key, selected_value):
            st.toast(f"Documentation retrieval table set to: {selected_label}", icon="✅")
        else:
            st.toast(f"Error saving preference for {env_var_key}", icon="❌")

    # Display the radio button
    selected_table_label = st.radio(
        "Select Documentation Table:",
        options=options_list,
        index=current_index,
        key="docs_retrieval_table_radio", # Unique key for session state
        on_change=save_retrieval_preference,
        help="Choose 'Site Pages' for standard web scraping results or 'Hierarchical Nodes' for structured llms.txt processing results."
    )

    st.divider()

    # Hierarchical Nodes Schema Display
    st.subheader("Alternative Schema: Hierarchical Nodes (for llms.txt)")
    with st.expander("View SQL and Instructions", expanded=False):
        st.markdown("""
        This alternative schema is designed for storing hierarchically structured documentation, 
        typically generated by processing `llms.txt` files. It allows for more granular retrieval 
        based on document structure (headers, sections, etc.).
        
        **Instructions:**
        1.  Ensure you have the `pgvector` extension enabled in your Supabase project.
        2.  Run the following SQL commands in your Supabase SQL Editor to create the necessary 
            `hierarchical_nodes` table and related functions/indexes.
        3.  **Important:** Adjust the `VECTOR(1536)` dimension in the SQL below if your embedding model uses a different dimension (e.g., 768 for `nomic-embed-text`).
        """)
        
        # Load and display the llms_txt SQL
        llms_txt_sql = load_llms_txt_sql()
        if "Error:" in llms_txt_sql:
            st.error(llms_txt_sql)
        else:
            # Replace vector dimension placeholder if needed (similar to site_pages)
            # Assuming the same vector_dim variable applies, otherwise fetch separately
            llms_txt_sql_adjusted = llms_txt_sql.replace("VECTOR(1536)", f"VECTOR({vector_dim})")
            st.code(llms_txt_sql_adjusted, language="sql")

        # Provide a link to the Supabase SQL Editor
        supabase_url = get_env_var("SUPABASE_URL")
        if supabase_url:
            dashboard_url = get_supabase_sql_editor_url(supabase_url)
            st.markdown(f"[Open Your Supabase SQL Editor]({dashboard_url})")
        else:
            st.warning("Configure Supabase URL in Environment tab to get a direct link to the SQL Editor.")

