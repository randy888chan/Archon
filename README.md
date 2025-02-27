# Archon - AI Agent Builder

<img src="public/Archon.png" alt="Archon Logo" />

<div align="center" style="margin-top: 20px;margin-bottom: 30px">

<h3>🚀 **CURRENT VERSION** 🚀</h3>

**[ V3 - MCP Support ]**
*Using LangGraph + Pydantic AI with AI IDE integration*

</div>

Archon is an AI meta-agent designed to autonomously build, refine, and optimize other AI agents. 

It serves both as a practical tool for developers and as an educational framework demonstrating the evolution of agentic systems.
Archon will be developed in iterations, starting with just a simple Pydantic AI agent that can build other Pydantic AI agents,
all the way to a full agentic workflow using LangGraph that can build other AI agents with any framework.
Through its iterative development, Archon showcases the power of planning, feedback loops, and domain-specific knowledge in creating robust AI agents.

## Important Links

- The current version of Archon is V3 as mentioned above - see [V3 Documentation](iterations/v3-mcp-support/README.md) for details.

- I **just** created the [Archon community](https://thinktank.ottomator.ai/c/archon/30) forum over in the oTTomator Think Tank! Please post any questions you have there!

- [GitHub Kanban board](https://github.com/users/coleam00/projects/1) for feature implementation and bug squashing.

## Vision

Archon demonstrates three key principles in modern AI development:

1. **Agentic Reasoning**: Planning, iterative feedback, and self-evaluation overcome the limitations of purely reactive systems
2. **Domain Knowledge Integration**: Seamless embedding of frameworks like Pydantic AI and LangGraph within autonomous workflows
3. **Scalable Architecture**: Modular design supporting maintainability, cost optimization, and ethical AI practices

## Project Evolution

### V1: Single-Agent Foundation
- Basic RAG-powered agent using Pydantic AI
- Supabase vector database for documentation storage
- Simple code generation without validation
- [Learn more about V1](iterations/v1-single-agent/README.md)

### V2: Agentic Workflow (LangGraph)
- Multi-agent system with planning and execution separation
- Reasoning LLM (O3-mini/R1) for architecture planning
- LangGraph for workflow orchestration
- Support for local LLMs via Ollama
- [Learn more about V2](iterations/v2-agentic-workflow/README.md)

### V3: Current - MCP Support
- Integration with AI IDEs like Windsurf and Cursor
- Automated file creation and dependency management
- FastAPI service for agent generation
- Improved project structure and organization
- [Learn more about V3](iterations/v3-mcp-support/README.md)

### Future Iterations
- V4: Self-Feedback Loop - Automated validation and error correction
- V5: Tool Library Integration - Pre-built external tool incorporation
- V6: Multi-Framework Support - Framework-agnostic agent generation
- V7: Autonomous Framework Learning - Self-updating framework adapters

### Future Integrations
- Docker
- LangSmith
- Other frameworks besides Pydantic AI
- Other vector databases besides Supabase

## Getting Started with V3 (current version)

Since V3 is the current version of Archon, all the code for V3 is in both the `archon` and `archon/iterations/v3-mcp-support` directories.

### Prerequisites
- Python 3.11+
- Supabase account and database
- OpenAI/OpenRouter API key or Ollama for local LLMs
- Streamlit (for web interface)
- Windsurf, Cursor, or another MCP-compatible AI IDE (optional)

### Installation

There are two ways to install Archon V3:

#### Option 1: Standard Installation (for Streamlit UI)

1. Clone the repository:
```bash
git clone https://github.com/coleam00/archon.git
cd archon
```

2. Install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### Option 2: MCP Server Setup (for AI IDE integration)

1. Clone the repository as above

2. Run the MCP setup script:
```bash
python setup_mcp.py
```

For running the crawler and graph service later, activate the virtual environment too:

```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

This will:
- Create a virtual environment if it doesn't exist
- Install dependencies from requirements.txt
- Generate an MCP configuration file

3. Configure your AI IDE:
   - **In Windsurf**:
        - Click on the hammer icon above the chat input
        - Click on "Configure"
        - Paste the JSON that `setup_mcp.py` gave you as the MCP config
        - Click "Refresh" next to "Configure"
   - **In Cursor**:
        - Go to Cursor Settings > Features > MCP
        - Click on "+ Add New MCP Server"
        - Name: Archon
        - Type: command (equivalent to stdio)
        - Command: Paste the command that `setup_mcp.py` gave for Cursor

### Environment Setup

1. Configure environment:
   - Rename `.env.example` to `.env`
   - Edit `.env` with your settings:
   ```env
   BASE_URL=https://api.openai.com/v1 for OpenAI, https://api.openrouter.ai/v1 for OpenRouter, or your Ollama URL
   LLM_API_KEY=your_openai_or_openrouter_api_key
   OPENAI_API_KEY=your_openai_api_key  # Required for embeddings
   SUPABASE_URL=your_supabase_url
   SUPABASE_SERVICE_KEY=your_supabase_service_key
   PRIMARY_MODEL=gpt-4o-mini  # Main agent model
   REASONER_MODEL=o3-mini    # Planning model
   ```

### Quick Start

1. Set up the database:
   - Execute `utils/site_pages.sql` in your Supabase SQL Editor
   - This creates tables and enables vector similarity search
   - See the Database Setup section for more details

2. Crawl documentation:
```bash
python archon/crawl_pydantic_ai_docs.py
```

3. Run Archon either as an MCP Server or with Streamlit:

### Using with AI IDEs (MCP Support)

1. After crawling the documentation, start the graph service:

```bash
python graph_service.py
```

Archon runs as a separate API endpoint for MCP instead of directly in the MCP server because that way Archon can be updated separately without having to restart the MCP server, and the communication protocols for MCP seemed to interfere with LLM calls when done directly within the MCP server.

2. Restart the MCP server in your AI IDE
3. You can now ask your AI IDE to create agents with Archon
4. Be sure to specify when you want to use Archon - not necessary but it helps a lot

### Using the Streamlit UI

For an interactive web interface:

```bash
streamlit run streamlit_ui.py
```

The interface will be available at `http://localhost:8501`
## Architecture

### Core Files
- `mcp_server.py`: MCP server script for AI IDE integration
- `graph_service.py`: FastAPI service that handles the agentic workflow
- `setup_mcp.py`: MCP setup script
- `streamlit_ui.py`: Web interface with streaming support
- `requirements.txt`: Project dependencies

### Archon Package
- `archon/`: Core agent and workflow implementation
  - `archon_graph.py`: LangGraph workflow definition and agent coordination
  - `pydantic_ai_coder.py`: Main coding agent with RAG capabilities
  - `crawl_pydantic_ai_docs.py`: Documentation crawler and processor

### Utilities
- `utils/`: Utility functions and database setup
  - `utils.py`: Shared utility functions
  - `site_pages.sql`: Database setup commands
  - `site_pages_ollama.sql`: Database setup commands with vector dimensions updated for nomic-embed-text  

### Database Setup

The Supabase database uses the following schema:

```sql
CREATE TABLE site_pages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    url TEXT,
    chunk_number INTEGER,
    title TEXT,
    summary TEXT,
    content TEXT,
    metadata JSONB,
    embedding VECTOR(1536) -- Adjust dimensions as necessary (i.e. 768 for nomic-embed-text)
);
```

Execute the SQL commands in `utils/site_pages.sql` to:
1. Create the necessary tables
2. Enable vector similarity search
3. Set up Row Level Security policies

In Supabase, do this by going to the "SQL Editor" tab and pasting in the SQL into the editor there. Then click "Run".

If using Ollama with the nomic-embed-text embedding model or another with 786 dimensions, either update site_pages.sql so that the dimensions are 768 instead of 1536 or use `utils/ollama_site_pages.sql`

## Contributing

We welcome contributions! Whether you're fixing bugs, adding features, or improving documentation, please feel free to submit a Pull Request.

## License

[MIT License](LICENSE)

---

For version-specific details:
- [V1 Documentation](iterations/v1-single-agent/README.md)
- [V2 Documentation](iterations/v2-agentic-workflow/README.md)
- [V3 Documentation](iterations/v3-mcp-support/README.md)
