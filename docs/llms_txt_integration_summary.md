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
- **Custom Upload Placeholder:** Added a `st.info` message and a `# TODO` comment indicating that custom file uploads will be added later.
- **Database Statistics:** Included a section to display the total count of nodes in the `hierarchical_nodes` table and a button to view sample data.

## 3. Chat RAG Integration (`archon/agent_tools.py`)

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

4. **Feature: Real-time Log Streaming** (April 18, 2025)
   - Implemented real-time log streaming for large llms.txt/llms-full.txt processing
   - Replaced subprocess.run with subprocess.Popen for non-blocking execution
   - Added incremental log updates to the UI for better user experience
   - Allows users to see processing progress in real time

## How to Create a Pull Request to Upstream

This section provides step-by-step instructions for creating a pull request from this fork to the original repository (https://github.com/coleam00/Archon).

### Prerequisites

1. Ensure your changes are committed to a feature branch in your fork
2. Test your changes thoroughly to ensure they work as expected
3. Make sure your code follows the project's coding standards and conventions

### Step 1: Update Your Fork with the Latest Changes from Upstream

```bash
# Ensure you have the upstream repository configured
git remote -v
# If upstream is not listed, add it
git remote add upstream https://github.com/coleam00/Archon.git

# Fetch the latest changes from upstream
git fetch upstream

# Switch to your main branch
git checkout main

# Merge upstream changes into your main branch
git merge upstream/main

# Push the updated main branch to your fork
git push origin main
```

### Step 2: Rebase Your Feature Branch

```bash
# Switch to your feature branch
git checkout your-feature-branch

# Rebase your branch on top of the updated main
git rebase main

# Resolve any conflicts that may arise during the rebase
# After resolving conflicts:
git rebase --continue

# Force push your rebased branch to your fork
git push --force-with-lease origin your-feature-branch
```

### Step 3: Create the Pull Request

1. Go to the original repository: https://github.com/coleam00/Archon
2. Click on the "Pull Requests" tab
3. Click the "New Pull Request" button
4. Click on the link "compare across forks"
5. Set the base repository to `coleam00/Archon` and the base branch to `main`
6. Set the head repository to `HillviewCap/Archon` and the compare branch to your feature branch
7. Click "Create Pull Request"
8. Fill in the PR template with the following information:

### Pull Request Template

```markdown
## Description

[Provide a detailed description of the changes in this PR]

## Related Issues

[Link to any related issues or tickets]

## Type of Change

- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring
- [ ] Other (please describe)

## Testing Performed

[Describe the testing you've done to validate your changes]

## Screenshots (if applicable)

[Add screenshots to help explain your changes]

## Checklist

- [ ] My code follows the project's coding style
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have updated the documentation accordingly
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing tests pass locally with my changes
```

### Step 4: Respond to Review Feedback

1. Be responsive to any feedback or change requests from the maintainers
2. Make requested changes to your branch and push them to your fork
3. The PR will automatically update with your new changes

### Best Practices for Upstream PRs

1. **Keep PRs Focused**: Each PR should address a single concern or feature
2. **Write Clear Commit Messages**: Use descriptive commit messages that explain what and why
3. **Document Your Changes**: Update relevant documentation as part of your PR
4. **Follow Project Conventions**: Adhere to the coding style and patterns used in the project
5. **Be Patient and Respectful**: Maintainers are often busy; be patient and respectful in communications
