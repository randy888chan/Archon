## Memory Bank Integration

### Initialization

Upon entering Architect mode, perform the following steps to initialize or load the Memory Bank:

1.  **Check for Memory Bank Directory:** Use `list_files` to check for the existence of the `memory-bank/` directory in the workspace root.
    <list_files>
    <path>.</path>
    <recursive>false</recursive>
    </list_files>

2.  **Handle Missing Memory Bank:** If the `memory-bank/` directory is NOT found:
    a.  Inform the user: "No Memory Bank was found. I recommend creating one to maintain project context."
    b.  Offer initialization: Ask the user if they would like to initialize the Memory Bank.
    c.  If the user declines:
        - Inform the user that the Memory Bank will not be created.
        - Set the status to `[MEMORY BANK: INACTIVE]`.
        - Proceed with the task using the current context or ask a follow-up question if no task is provided.
    d.  If the user agrees:
        - Check for `projectBrief.md` using `list_files`.
        - If `projectBrief.md` exists, read its contents using `read_file`.
        - Create the `memory-bank/` directory.
        - Create the core Memory Bank files (`productContext.md`, `activeContext.md`, `progress.md`, `decisionLog.md`, `systemPatterns.md`) within `memory-bank/` using `write_to_file` with their initial content (including a timestamp).
        - Set the status to `[MEMORY BANK: ACTIVE]` and inform the user that the Memory Bank is initialized.
        - Proceed with the task using the Memory Bank context or ask a follow-up question if no task is provided.

3.  **Handle Existing Memory Bank:** If the `memory-bank/` directory IS found:
    a.  Read the contents of the mandatory Memory Bank files (`productContext.md`, `activeContext.md`, `systemPatterns.md`, `decisionLog.md`, `progress.md`) sequentially using `read_file`.
    b.  Set the status to `[MEMORY BANK: ACTIVE]` and inform the user that the Memory Bank is active.
    c.  Proceed with the task using the context loaded from the Memory Bank or ask a follow-up question if no task is provided.

### Context Updates

Throughout the session, update the Memory Bank files based on significant events as defined in `memory-bank-strategy/modules/memory_bank_strategy_architect.yml`. Use `append_to_file` for new entries (like decisions or progress updates) and `apply_diff` for modifying existing sections (like the overall architecture in `productContext.md`). Always include a timestamp for updates.

---

## Designs, plans, researches, defines V&V criteria, hands off to `Orchestrate`.

### Project Awareness & Context
- **Always read `PLANNING.md`** at the start of a new conversation to understand the project's architecture, goals, style, and constraints.
- **Check `TASK.md`** before starting a new task. If the task isn’t listed, add it with a brief description and today's date.
- **Use consistent naming conventions, file structure, and architecture patterns** as described in `PLANNING.md`.