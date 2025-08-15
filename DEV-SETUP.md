# Archon V2 Alpha - Development Setup

## Current Development Workflow

### Server Configuration

**Development Server (192.168.5.11)**:
- **Workshop Version**: Running on standard ports (80, 8051, 8052, 8181)
- **Development Code**: Located at `/root/Archon-V2-Alpha`
- **Purpose**: Code development and editing

**Testing Server (192.168.5.12)**:  
- **Database**: Supabase running on port 8000
- **Purpose**: Testing the V2 development code

### Development Process

1. **Develop** on 192.168.5.11 in `/root/Archon-V2-Alpha`
2. **Test** by deploying to 192.168.5.12 server
3. **Workshop version** remains stable on 192.168.5.11

### Environment Configuration

The V2 development version is configured to use:
- **Supabase**: `http://192.168.5.12:8000` (external testing server)
- **Project Name**: `archon-v2-alpha` (prevents conflicts)
- **Ports**: Non-conflicting ports (3737, 8182, 8053, 8054)

### Deployment Commands

**On Development Server (192.168.5.11)**:
```bash
cd /root/Archon-V2-Alpha
# Develop and test code locally
```

**For Testing (deploy to 192.168.5.12)**:
```bash
# Push code to testing server and run containers there
# (Commands to be added based on deployment method)
```

### Access Points

- **Workshop Version**: http://192.168.5.11 (port 80)
- **V2 Testing Version**: http://192.168.5.12:3737 (when deployed)

### Benefits

✅ **Clean Separation**: Workshop stable, development isolated  
✅ **No Port Conflicts**: Each version uses different ports  
✅ **Shared Database**: Both can use same Supabase instance  
✅ **Independent Testing**: V2 testing doesn't affect workshop  