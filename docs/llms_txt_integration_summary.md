# LLMS.txt Hierarchical RAG Integration Summary

This document summarizes the changes made to integrate the `llms_txt` hierarchical RAG workflow into the Archon Streamlit application.

## 1. Database Configuration (`streamlit_pages/database.py`)

- **Table Selection:** Added a radio button (`st.radio`) allowing users to select the database table used for documentation retrieval:
  - "Site Pages (Default - Pydantic AI Docs)"
  - "Hierarchical Nodes (llms.txt Framework Docs)"
- **Preference Storage:** The selected table preference is now stored in `workbench/env_vars.json` under the key `DOCS_RETRIEVAL_TABLE` using the `save_env_var` utility.
- **Schema Display:** Added a new expander section ("Alternative Schema: Hierarchical Nodes (for llms.txt)") that displays the SQL commands from `utils/llms_txt.sql` using `st.code`. Includes instructions for executing the SQL in Supabase. The displayed vector dimension is dynamically adjusted based on environment settings.

## 2. Documentation Processing UI (`streamlit_pages/documentation.py`)

- **New Tab:** Added a new tab titled "Framework Docs (llms.txt)" to the documentation section.
- **Framework Selection:** Included a dropdown (`st.selectbox`) for users to select predefined `llms.txt` documentation sources (local paths or URLs):
  - Pydantic AI (`https://ai.pydantic.dev/llms.txt`)
  - LangGraph (`https://langchain-ai.github.io/langgraph/llms-full.txt`)
  - CrewAI (`https://docs.crewai.com/llms-full.txt`)
- **Processing Trigger:** Added a button ("Process Selected Framework Docs") that:
  - Checks if `DOCS_RETRIEVAL_TABLE` is set to `hierarchical_nodes`.
  - Executes the `run_processing.py` script via `subprocess` with the selected file path.
  - Displays the script's output logs and errors.
- **Custom File Upload:** Implemented a file uploader for users to upload their own llms.txt or llms-full.txt files.
  - Files are saved to the `docs/` directory with sanitized filenames
  - Clear status indicators for successful saves and processing
  - Real-time processing log display
- **Database Statistics:** Included a section to display the total count of nodes in the `hierarchical_nodes` table and a button to view sample data.

## 3. llms-text.ai Search Integration

- **Search Interface:** Added a dedicated search section to find llms.txt and llms-full.txt files across the web using the llms-text.ai API.

  - Custom styled container with a form-based interface
  - Search query input with placeholder text
  - File type selector dropdown with options for:
    - Both llms.txt and llms-full.txt formats
    - Basic (llms.txt only)
    - Comprehensive (llms-full.txt only)
  - Pagination controls with configurable results per page

- **Search Results Display:**

  - Card-like containers for each result showing:
    - Title and domain information
    - URL with external link
    - Last updated timestamp
    - Content summary
  - "Process File" button for immediate importing of discovered files
  - Expandable metadata sections showing:
    - Source domain information
    - URL purpose rankings
    - Domain and URL topic rankings
  - Pagination navigation for browsing through result pages

- **One-Click Processing:** Implemented direct processing of search results:

  - Downloads the selected file when "Process File" is clicked
  - Automatically saves to the docs directory with a sanitized filename
  - Executes the processing pipeline on the downloaded file
  - Displays real-time processing logs
  - Updates the hierarchical nodes database with the content

- **Error Handling:** Comprehensive error handling for:
  - API connection issues
  - File download problems
  - Processing failures
  - Proper JSON parsing of API responses

## 4. Chat RAG Integration (`archon/agent_tools.py`)

- **Configuration Reading:** The RAG logic now reads the `DOCS_RETRIEVAL_TABLE` setting from `workbench/env_vars.json` at runtime.
- **Conditional Retrieval:** Implemented conditional logic within the relevant agent tool function(s):
  - If `DOCS_RETRIEVAL_TABLE` is `site_pages` (or default), the existing RAG process querying the `site_pages` table is used.
  - If `DOCS_RETRIEVAL_TABLE` is `hierarchical_nodes`, the code now queries the `hierarchical_nodes` table using `SupabaseManager` and associated components from `archon.llms_txt` to perform a vector search and retrieve relevant node content.
- **Context Formatting:** The retrieved context (from either table) is formatted appropriately before being passed to the language model.

## Recent Merges and Pull Requests

The following merges and PRs have contributed to the `llms_txt` integration:

1. **PR #8: LLMS-TXT-INTEGRATION** (Merged April 1, 2025)

   - Main integration PR that introduced the hierarchical RAG workflow
   - Added core functionality for processing llms.txt format documents
   - Implemented database schema and vector search capabilities
   - Added UI components for document selection and processing

2. **Fix: Embedding Batch Token Limit** (Merged April 18, 2025)

   - Implemented token-aware batching for OpenAI embedding requests
   - Fixed issues with the 8192 token limit in the embedding API
   - Enhanced the embedding_manager.py to handle large documents more efficiently
   - Improved UI by clearing output logs at the start of new file processing

3. **Fix: Retrieval Embedding Attribute Error** (Merged April 18, 2025)

   - Fixed an issue with similarity score formatting in the retrieval process
   - Ensured proper type handling for similarity scores

4. **Feature: Real-time Log Streaming** (Merged April 18, 2025)

   - Implemented real-time log streaming for large llms.txt/llms-full.txt processing
   - Replaced subprocess.run with subprocess.Popen for non-blocking execution
   - Added incremental log updates to the UI for better user experience
   - Allows users to see processing progress in real time

5. **Feature: llms-text.ai API Integration** (Merged April 18, 2025)
   - Added search capabilities for discovering llms.txt and llms-full.txt files on the web
   - Integrated with the llms-text.ai API for comprehensive search functionality
   - Implemented rich result display with metadata visualization
   - Added one-click processing of discovered framework documentation
   - Enhanced UI with styled containers and intuitive search controls
