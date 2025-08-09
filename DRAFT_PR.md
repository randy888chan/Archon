# Pull Request

## Summary
Remove all hardcoded ports and require explicit environment variable configuration for better deployment flexibility and to prevent configuration mismatches in alpha testing.

## Changes Made
- Fixed critical port mismatches between services (8080 vs 8181)
- Removed all hardcoded localhost URLs in frontend services
- Replaced all fallback ports with explicit environment variable requirements
- Updated inter-service communication to use consistent port configuration
- Added clear error messages when required port environment variables are missing

## Type of Change
- [x] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [x] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [x] Code refactoring

## Affected Services
- [x] Frontend (React UI)
- [x] Server (FastAPI backend)
- [x] MCP Server (Model Context Protocol)
- [x] Agents (PydanticAI service)
- [ ] Database (migrations/schema)
- [x] Docker/Infrastructure
- [ ] Documentation site

## Implementation Plan

### 🚨 CRITICAL FIXES (Blocking Alpha)

#### 1. Fix Bug Report Service URL (bugReportService.ts:162)
**Current:** `http://localhost:8181/api/bug-report/github`
**Solution:** Use the API configuration from `src/config/api.ts`
**Impact:** Bug reporting will work regardless of port configuration

#### 2. Fix main.py Port Configuration (main.py:267)
**Current:** `port=8080`
**Solution:** Use `int(os.getenv("ARCHON_SERVER_PORT"))` with required check
**Impact:** Server runs on correct configured port

#### 3. Fix Inter-service Communication Ports
**Files:**
- `python/src/agents/server.py:67` - Update from 8080 to use ARCHON_SERVER_PORT
- `python/src/server/fastapi/agent_chat_api.py:164` - Use service discovery
- `python/src/server/fastapi/internal_api.py:85` - Use proper port variables

**Impact:** Services can communicate correctly

### 🟡 NICE TO HAVE FIXES

#### 4. Remove Hardcoded Fallbacks in Frontend
**Files:**
- `archon-ui-main/src/services/mcpClientService.ts:404`
- `archon-ui-main/vite.config.ts:22`
- `archon-ui-main/src/config/api.ts:23`

**Solution:** Require ARCHON_SERVER_PORT and ARCHON_MCP_PORT with clear error messages

#### 5. Update Service Discovery
**File:** `python/src/server/config/service_discovery.py`
**Solution:** Remove DEFAULT_PORTS fallbacks, require env vars with validation

#### 6. Update All Python Service Ports
**Files:**
- `python/src/mcp/mcp_server.py:66`
- `python/src/agents/server.py:291`
- `python/src/server/config/config.py:82`

**Solution:** Remove fallbacks, add startup validation

#### 7. Fix Documentation Examples
**Files:**
- `python/src/mcp/modules/project_module.py:706-707`
- Various docs files

**Solution:** Update to use correct port examples (8181, not 8000/8080)

## Error Message Strategy

When environment variables are missing, provide clear, actionable error messages:

```python
if not os.getenv("ARCHON_SERVER_PORT"):
    raise ValueError(
        "ARCHON_SERVER_PORT environment variable is required. "
        "Please set it in your .env file or environment. "
        "Default: 8181"
    )
```

```typescript
if (!import.meta.env.ARCHON_SERVER_PORT) {
  throw new Error(
    'ARCHON_SERVER_PORT is not configured. ' +
    'Please set it in your environment variables. ' +
    'Default value: 8181'
  );
}
```

## Testing
- [ ] All existing tests pass
- [ ] Added new tests for new functionality
- [ ] Manually tested affected user flows
- [ ] Docker builds succeed for all services

### Test Evidence
```bash
# Test with custom ports
export ARCHON_SERVER_PORT=9191
export ARCHON_MCP_PORT=9051  
export ARCHON_AGENTS_PORT=9052
export ARCHON_UI_PORT=4737

# Run services and verify they use custom ports
docker-compose up --build

# Test inter-service communication
curl http://localhost:9191/health
curl http://localhost:9051/health
curl http://localhost:9052/health

# Test frontend can connect to backend on custom port
# Visit http://localhost:4737 and verify all features work
```

## Checklist
- [x] My code follows the service architecture patterns
- [x] If using an AI coding assistant, I used the CLAUDE.md rules
- [ ] I have added tests that prove my fix/feature works
- [ ] All new and existing tests pass locally
- [ ] My changes generate no new warnings
- [ ] I have updated relevant documentation
- [ ] I have verified no regressions in existing features

## Breaking Changes
This PR introduces breaking changes that require environment variables to be explicitly set:

### Required Environment Variables (no more fallbacks):
- `ARCHON_SERVER_PORT` (previously defaulted to 8181)
- `ARCHON_MCP_PORT` (previously defaulted to 8051)
- `ARCHON_AGENTS_PORT` (previously defaulted to 8052)
- `ARCHON_UI_PORT` (previously defaulted to 3737)

### Migration Steps:
1. Update your `.env` file with all required port variables
2. If using Docker, ensure docker-compose.yml passes these variables
3. For local development, export the variables before running services

## Additional Notes
This change improves deployment flexibility and prevents the "works on my machine" problem by making configuration explicit. It also fixes critical port mismatches that were causing service communication failures.

The approach of requiring explicit configuration over silent fallbacks follows the principle of "fail fast and loud" established in our CLAUDE.md guidelines for alpha development.