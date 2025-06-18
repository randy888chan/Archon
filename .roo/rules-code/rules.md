## Memory Bank Integration

### Initialization

Upon entering Code mode, perform the following steps to initialize or load the Memory Bank:

1.  **Check for Memory Bank Directory:** Use `list_files` to check for the existence of the `memory-bank/` directory in the workspace root.
    <list_files>
    <path>.</path>
    <recursive>false</recursive>
    </list_files>

2.  **Handle Missing Memory Bank:** If the `memory-bank/` directory is NOT found:
    a.  Inform the user: "No Memory Bank was found. I recommend creating one to maintain project context. Would you like to switch to Architect mode to do this?"
    b.  Conditional Actions:
        *   If the user declines:
            - Inform the user that the Memory Bank will not be created.
            - Set the status to `[MEMORY BANK: INACTIVE]`.
            - Proceed with the task using the current context if needed or if no task is provided, use the `ask_followup_question` tool .
        *   If the user agrees:
            Switch to Architect mode to create the Memory Bank.

3.  **Handle Existing Memory Bank:** If the `memory-bank/` directory IS found:
    a.  Read the contents of the mandatory Memory Bank files (`productContext.md`, `activeContext.md`, `systemPatterns.md`, `decisionLog.md`, `progress.md`) sequentially using `read_file`.
    b.  Set the status to `[MEMORY BANK: ACTIVE]` and inform user.
    c.  Proceed with the task using the context loaded from the Memory Bank or if no task is provided, use the `ask_followup_question` tool.

---

## Implements, documents specific sub-tasks, returns results to `Test`.

### Context7
- **Always use the context7 MCP** to reference documentation for libraries like Pydantic AI and Streamlit.
- For the tokens, **start with 5000** but then increase to **20000** if your first search didn't give relevant documentation.
- **Only search three times maximum for any specific piece of documentation**. If you don't get what you need, use the Brave MCP server to perform a wider search.

### Code Structure & Modularity
- **Never create a file longer than 500 lines of code.** If a file approaches this limit, refactor by splitting it into modules or helper files.
- **Organize code into clearly separated modules**, grouped by feature or responsibility.
- **Use clear, consistent imports** (prefer relative imports within packages).

### Style & Conventions
- **Use Python** as the primary language.
- **Follow PEP8**, use type hints, and format with `black`.
- **Use `pydantic` for data validation**.
- Use `FastAPI` for APIs and `SQLAlchemy` or `SQLModel` for ORM if applicable.
- Write **docstrings for every function** using the Google style:
  ```python
  def example():
      """
      Brief summary.

      Args:
          param1 (type): Description.

      Returns:
          type: Description.
      """
  ```