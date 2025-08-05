# Claude Code GitHub Workflows

This directory contains GitHub Actions workflows for Claude Code integration in Archon V2 Alpha.

## Available Commands

### `/claude-review` - Code Review (Read-Only)
- **Purpose**: Performs code review without making changes
- **Trigger**: Comment `/claude-review` on any PR or issue
- **Permissions**: Read-only access to code
- **Use Cases**:
  - Review PR changes for quality and bugs
  - Analyze code for security issues
  - Suggest improvements
  - Check adherence to coding standards

### `/claude-fix` - Fix Issues (Write Access)
- **Purpose**: Implements fixes and creates pull requests
- **Trigger**: Comment `/claude-fix` on any PR or issue
- **Permissions**: Full write access to create branches and PRs
- **Use Cases**:
  - Fix reported bugs
  - Implement requested changes
  - Add missing tests
  - Update documentation

## Authorized Users

Only the following users can trigger Claude:
- @Wirasm
- @coleam00
- @sean-eskerium

Unauthorized users will receive an error message if they try to trigger Claude.

## Usage Examples

### Request a Code Review
```
/claude-review Please review this PR for security issues and performance concerns
```

### Fix an Issue
```
/claude-fix Fix the TypeError in the bug report above and add appropriate error handling
```

### Specific Fix Request
```
/claude-fix Implement the missing validation for user input in the settings page
```

## Security Features

1. **User Authorization**: Only approved maintainers can trigger Claude
2. **Command Separation**: Different commands for read vs write operations
3. **No Auto-Triggers**: Claude never runs automatically on PR creation
4. **Timeout Protection**: Workflows have timeout limits to prevent runaway sessions
5. **Limited Permissions**: Each workflow only gets necessary permissions

## Adding New Authorized Users

To add a new authorized user:
1. Edit both `claude-review.yml` and `claude-fix.yml`
2. Add the GitHub username to the `ALLOWED_USERS` environment variable
3. Commit and push the changes

Example:
```yaml
env:
  ALLOWED_USERS: '["Wirasm", "coleam00", "sean-eskerium", "new-username"]'
```

## Workflow Files

- `claude-review.yml` - Read-only code review workflow
- `claude-fix.yml` - Write-access fix implementation workflow
- `claude.yml.deprecated` - Old workflow (to be removed)

## Migrating from Old Workflow

The old `claude.yml` workflow has been deprecated because it:
- Triggered automatically on PR creation
- Had no user authorization
- Mixed read and write permissions

Please use the new command-based workflows instead.

## Troubleshooting

### Claude doesn't respond to command
- Verify you're an authorized user
- Check the command syntax (must start with `/claude-review` or `/claude-fix`)
- Ensure the workflow files are in the main branch

### Unauthorized error
- Only users in the `ALLOWED_USERS` list can trigger Claude
- Contact a repository maintainer to be added

### Workflow times out
- Complex fixes may take time
- Check the PR/issue for partial progress
- Re-run with more specific instructions if needed