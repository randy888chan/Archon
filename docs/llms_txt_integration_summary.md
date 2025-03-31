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
- **Framework Selection:** Included a dropdown (`st.selectbox`) for users to select predefined `llms.txt` documentation files:
  - Pydantic AI (`docs/anthropic-llms.txt`)
  - LangGraph (`docs/langgraph-llms-full.txt`)
  - CrewAI (`docs/crewai-llms-full.txt`)
- **Processing Trigger:** Added a button ("Process Selected Framework Docs") that:
  - Checks if `DOCS_RETRIEVAL_TABLE` is set to `hierarchical_nodes`.
  - Executes the `run_processing.py` script via `subprocess` with the selected file path.
  - Displays the script's output logs and errors.
- **Custom Upload Placeholder:** Added a `st.info` message and a `# TODO` comment indicating that custom file uploads will be added later.
- **Database Statistics:** Included a section to display the total count of nodes in the `hierarchical_nodes` table and a button to view sample data.

## 3. Chat RAG Integration (`archon/agent_tools.py`)

- **Configuration Reading:** The RAG logic now reads the `DOCS_RETRIEVAL_TABLE` setting from `workbench/env_vars.json` at runtime.
- **Conditional Retrieval:** Implemented conditional logic within the relevant agent tool function(s):
  - If `DOCS_RETRIEVAL_TABLE` is `site_pages` (or default), the existing RAG process querying the `site_pages` table is used.
  - If `DOCS_RETRIEVAL_TABLE` is `hierarchical_nodes`, the code now queries the `hierarchical_nodes` table using `SupabaseManager` and associated components from `archon.llms_txt` to perform a vector search and retrieve relevant node content.
- **Context Formatting:** The retrieved context (from either table) is formatted appropriately before being passed to the language model.
