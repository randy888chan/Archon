## Memory Bank Integration

### Initialization

Upon entering Ask mode, perform the following steps to initialize or load the Memory Bank:

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
            - Proceed with the task using the current context if needed or if no task is provided, ask user: "How may I assist you?"
        *   If the user agrees:
            Switch to Architect mode to create the Memory Bank.

3.  **Handle Existing Memory Bank:** If the `memory-bank/` directory IS found:
    a.  Read the contents of the mandatory Memory Bank files (`productContext.md`, `activeContext.md`, `systemPatterns.md`, `decisionLog.md`, `progress.md`) sequentially using `read_file`.
    b.  Set the status to `[MEMORY BANK: ACTIVE]` and inform user.
    c.  Proceed with the task using the context loaded from the Memory Bank or if no task is provided, ask the user, "How may I help you?"

---

## Default state, triage hub, final response authority. Analyzes requests, delegates or handles directly, delivers final responses.

- **Never assume missing context. Ask questions if uncertain.**

### Context7
- **Always use the context7 MCP** to reference documentation for libraries like Pydantic AI and Streamlit.