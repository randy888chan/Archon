# Archon V4 - Streamlit UI Overhaul

This is the fourth iteration of the Archon project, building upon V3 by adding a comprehensive Streamlit UI for managing all aspects of Archon. The system retains the core LangGraph workflow and MCP support from V3, but now provides a unified interface for environment configuration, database setup, documentation crawling, agent service management, and MCP integration.

What makes V4 special is its guided setup process that walks users through each step of configuring and running Archon. The Streamlit UI eliminates the need for manual configuration of environment variables, database setup, and service management, making Archon much more accessible to users without extensive technical knowledge.

The core remains an intelligent documentation crawler and RAG (Retrieval-Augmented Generation) system built using Pydantic AI, LangGraph, and Supabase. The system crawls the Pydantic AI documentation, stores content in a vector database, and provides Pydantic AI agent code by retrieving and analyzing relevant documentation chunks.

This version continues to support both local LLMs with Ollama and cloud-based LLMs through OpenAI/OpenRouter.

## Features

- Comprehensive Streamlit UI with multiple tabs for different functions
- Guided setup process with interactive instructions
- Environment variable management through the UI
- Database setup and configuration simplified
- Documentation crawling with progress tracking
- Agent service control and monitoring
- MCP configuration through the UI
- Multi-agent workflow using LangGraph
- Specialized agents for reasoning, routing, and coding
- Pydantic AI documentation crawling and chunking
- Vector database storage with Supabase
- Semantic search using OpenAI embeddings
- RAG-based question answering
- Support for code block preservation
- MCP server support for AI IDE integration

## Prerequisites

- Python 3.11+
- Supabase account (for vector database)
- OpenAI/OpenRouter API key or Ollama for local LLMs

## Installation

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

## Usage

Start the Streamlit UI:

```bash
streamlit run streamlit_ui.py
```

The interface will be available at `http://localhost:8501`

### Streamlit UI Tabs

The Streamlit UI provides the following tabs:

1. **Intro**: Overview and guided setup process
2. **Environment**: Configure API keys and model settings
3. **Database**: Set up your Supabase vector database
4. **Documentation**: Crawl and index the Pydantic AI documentation
5. **Agent Service**: Start and monitor the agent service
6. **Chat**: Interact with Archon to create AI agents
7. **MCP**: Configure integration with AI IDEs

### Environment Configuration

The Environment tab allows you to set and manage all environment variables through the UI:

- Base URL for API endpoints
- API keys for LLM providers
- Supabase connection details
- Model selections for different agent roles
- Embedding model configuration

All settings are saved to an `env_vars.json` file, which is automatically loaded when Archon starts.

### Database Setup

The Database tab simplifies the process of setting up your Supabase database:

- Select embedding dimensions based on your model
- View SQL commands for table creation
- Get instructions for executing SQL in Supabase
- Clear existing data if needed

### Documentation Management

The Documentation tab provides an interface for crawling and managing documentation:

- Start and monitor the crawling process with progress tracking
- View logs of the crawling process
- Clear existing documentation
- View database statistics

### Agent Service Control

The Agent Service tab allows you to manage the agent service:

- Start, restart, and stop the service
- Monitor service output in real-time
- Clear output logs
- Auto-refresh for continuous monitoring

### MCP Configuration

The MCP tab simplifies the process of configuring MCP for AI IDEs:

- Select your IDE (Windsurf, Cursor, or Cline)
- Generate configuration commands or JSON
- Copy configuration to clipboard
- Get step-by-step instructions for your specific IDE

## Project Structure

### Core Files
- `streamlit_ui.py`: Comprehensive web interface for managing all aspects of Archon
- `graph_service.py`: FastAPI service that handles the agentic workflow
- `mcp_server.py`: MCP server script for AI IDE integration
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
  - `env_vars.json`: Environment variables defined in the UI

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
