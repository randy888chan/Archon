## Writes and executes tests. If tests fail, pass back to `code` mode to rectify. If tests fail three times in a row, report back to `Orchestrate` with results.

- **Always create Pytest unit tests for new features** (functions, classes, routes, etc).
- **After updating any logic**, check whether existing unit tests need to be updated. If so, do it.
- **Tests should live in a `/tests` folder** mirroring the main app structure.
  - Include at least:
    - 1 test for expected use
    - 1 edge case
    - 1 failure case