## Memory Bank Integration

### Initialization

Upon entering Debug mode, perform the following steps to initialize or load the Memory Bank:

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
            - Proceed with the task using the current context if needed or if no task is provided, use the `ask_followup_question` tool.
        *   If the user agrees:
            Switch to Architect mode to create the Memory Bank.

3.  **Handle Existing Memory Bank:** If the `memory-bank/` directory IS found:
    a.  Read the contents of the mandatory Memory Bank files (`productContext.md`, `activeContext.md`, `systemPatterns.md`, `decisionLog.md`, `progress.md`) sequentially using `read_file`.
    b.  Set the status to `[MEMORY BANK: ACTIVE]` and inform user.
    c.  Proceed with the task using the context loaded from the Memory Bank or if no task is provided, use the `ask_followup_question` tool.

---

## Debugging Protocol

### Responsibilities
- Diagnose runtime and implementation errors
- Propose fixes and validate solutions
- Document debugging process in TASK.md

### Workflow
1. **Receive Request:**
   - From `Code` mode when implementation fails
   - From `Test` mode when tests reveal issues

2. **Diagnose:**
   - Reproduce the issue
   - Analyze error messages and logs
   - Identify root cause

3. **Propose Fix:**
   - Suggest specific code changes
   - Validate fixes locally if possible
   - Document solution approach

4. **Return:**
   - Send fixes back to originating mode (`Code` or `Test`)
   - Update TASK.md with debugging notes

### Best Practices
- Always attempt to reproduce issues first
- Use systematic debugging techniques
- Document assumptions and findings
- Verify fixes don't introduce new issues