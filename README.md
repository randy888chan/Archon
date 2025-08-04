# Archon - Knowledge Engine MCP Server

<p align="center">
  <em>Build Your AI's Knowledge Base with Web Crawling and Document Management</em>
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-connecting-to-cursor-ide">Cursor Setup</a> •
  <a href="#-documentation">Documentation</a>
</p>

---

## 🎯 What is Archon?

Archon is a **Model Context Protocol (MCP) server** that creates a centralized knowledge base for your AI coding assistants. Connect Cursor, Windsurf, or Claude Desktop to give your AI agents access to:

- **Your documentation** (crawled websites, uploaded PDFs/docs)
- **Smart search capabilities** with advanced RAG strategies  
- **Task management** integrated with your knowledge base
- **Real-time updates** as you add new content

## 🚀 Quick Start

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [Supabase](https://supabase.com/) account (free tier works)
- [OpenAI API key](https://platform.openai.com/api-keys)

### Initial Setup

```bash
# Clone the repository
git clone https://github.com/coleam00/archon.git
cd archon

# Create environment file
cp .env.example .env
```

Edit `.env` and add your Supabase credentials:

```bash
# Required
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your-service-key-here
```

## Initial Setup

1. **Set Up Database**: In your [Supabase project](https://supabase.com/dashboard) SQL Editor, run:
   ```sql
   -- Copy and paste the contents of migration/complete_setup.sql
   ```

2. **Start Archon**:
   ```bash
   docker-compose up --build -d
   ```
   
   This starts the core microservices:
   - **Archon-MCP**: Lightweight MCP server (Default Port: 8051)
   - **Archon-FastAPI**: Web crawling & document processing (Default Port: 8181)
   - **Archon-Agents**: AI/ML operations & reranking (Default Port: 8052)
   - **Archon-UI**: Web interface (Default Port: 3737)

3. **Configure API Key**:
   - Open http://localhost:3737
   - Go to **Settings** → Add your OpenAI API key
   - Test by uploading a document or crawling a website

## 🔄 Database Reset (Start Fresh if Needed)

If you need to completely reset your database and start fresh:

<details>
<summary>⚠️ <strong>Reset Database - This will delete ALL data for Archon!</strong></summary>

1. **Run Reset Script**: In your Supabase SQL Editor, run:
   ```sql
   -- Copy and paste the contents of migration/RESET_DB.sql
   -- ⚠️ WARNING: This will delete all data for ARchon!
   ```

2. **Rebuild Database**: After reset, run `migration/complete_setup.sql` to create all the tables again:

3. **Restart Services**:
   ```bash
   docker-compose restart
   ```

4. **Reconfigure**: 
   - Add your OpenAI API key in Settings
   - Re-upload any documents or re-crawl websites
   - Enable Projects feature if needed

The reset script safely removes all tables, functions, triggers, and policies with proper dependency handling.

</details>

## 📚 Documentation

### Core Services

| Service | Container Name | Default URL | Purpose |
|---------|---------------|-------------|---------|
| **Web Interface** | Archon-UI | http://localhost:3737 | Main dashboard and controls |
| **API Service** | Archon-FastAPI | http://localhost:8181 | Web crawling, document processing |
| **MCP Server** | Archon-MCP | http://localhost:8051 | Model Context Protocol interface |
| **Agents Service** | Archon-Agents | http://localhost:8052 | AI/ML operations, reranking |

### Optional Documentation Service

The documentation service is optional. To run it:

```bash
# Start core services + documentation
docker-compose -f docker-compose.yml -f docker-compose.docs.yml up --build -d
```

Then access documentation at: **http://localhost:3838**

## ⚡ Quick Test

Once everything is running:

1. **Test Document Upload**: Go to http://localhost:3737 → Knowledge Base → Upload a PDF
2. **Test Web Crawling**: Knowledge Base → "Crawl Website" → Enter a docs URL
3. **Test Projects**: Projects → Create a new project and add tasks
4. **Test AI Integration**: MCP Dashboard → Copy connection config for your AI client

## 🛠️ What's Included

### Features
- **Smart Web Crawling**: Automatically detects sitemaps, text files, or webpages
- **Document Processing**: Upload PDFs, Word docs, markdown, and text files
- **AI Integration**: Connect any MCP-compatible client (Cursor, Windsurf, etc.)
- **Task Management**: Organize projects and tasks with AI agent integration
- **Real-time Updates**: WebSocket-based live progress tracking

### Architecture
Archon uses a true microservices architecture with:
- **Lightweight MCP container**: ~150MB using distroless base
- **Service separation**: Each service has only its required dependencies
- **HTTP-based communication**: Services communicate via internal REST APIs
- **Optimized containers**: 50-90% size reduction compared to monolithic approach

## 🔧 Configuring Custom Ports & Hostname

By default, Archon services run on the following ports:
- **Archon-UI**: 3737
- **Archon-Server**: 8181  
- **Archon-MCP**: 8051
- **Archon-Agents**: 8052
- **Archon-Docs**: 3838 (optional)

### Changing Ports

To use custom ports, add these variables to your `.env` file:

```bash
# Service Ports Configuration
ARCHON_UI_PORT=3737
ARCHON_SERVER_PORT=8181
ARCHON_MCP_PORT=8051
ARCHON_AGENTS_PORT=8052
ARCHON_DOCS_PORT=3838
```

Example: Running on different ports:
```bash
ARCHON_SERVER_PORT=8282
ARCHON_MCP_PORT=8151
```

### Configuring Hostname

By default, Archon uses `localhost` as the hostname. You can configure a custom hostname or IP address by setting the `HOST` variable in your `.env` file:

```bash
# Hostname Configuration
HOST=localhost  # Default

# Examples of custom hostnames:
HOST=192.168.1.100     # Use specific IP address
HOST=archon.local      # Use custom domain
HOST=myserver.com      # Use public domain
```

This is useful when:
- Running Archon on a different machine and accessing it remotely
- Using a custom domain name for your installation
- Deploying in a network environment where `localhost` isn't accessible

After changing hostname or ports:
1. Restart Docker containers: `docker-compose down && docker-compose up -d`
2. Access the UI at: `http://${HOST}:${ARCHON_UI_PORT}`
3. Update your AI client configuration with the new hostname and MCP port

## 🔧 Development

For development with hot reload:

```bash
# Backend services (with auto-reload)
docker-compose up Archon-FastAPI Archon-MCP Archon-Agents --build

# Frontend (with hot reload) 
cd archon-ui-main && npm run dev

# Documentation (with hot reload)
cd docs && npm start
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

<p align="center">
  <strong>Build once, query everywhere</strong><br>
  <em>Transform your AI coding experience with Archon</em>
</p>