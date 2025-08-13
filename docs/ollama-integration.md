# Ollama Integration Guide

## Overview

Archon now supports **Ollama** as a local LLM provider, enabling you to run large language models and embedding models locally without external API dependencies. This is perfect for privacy-conscious users, air-gapped environments, or those wanting to reduce costs.

## Features

- ✅ Local embedding model support with 4 pre-configured models
- ✅ Safe model switching with dimension change detection
- ✅ Data migration warnings when changing embedding dimensions
- ✅ Model recommendation system with use case guidance
- ✅ Complete integration with Archon's RAG pipeline
- ✅ OpenAI-compatible API endpoints

## Quick Start

### 1. Install Ollama

**macOS/Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:**
Download from [ollama.com/download](https://ollama.com/download)

### 2. Start Ollama Service

```bash
ollama serve
```

Ollama will start on `http://localhost:11434` by default.

### 3. Pull Embedding Models

Choose one of the recommended embedding models:

```bash
# Fast, lightweight model (768 dimensions)
ollama pull nomic-embed-text

# High-quality, OpenAI-compatible dimensions (1536 dimensions)  
ollama pull mxbai-embed-large

# Balanced performance model (1024 dimensions)
ollama pull snowflake-arctic-embed2

# Multilingual support (1024 dimensions)
ollama pull bge-m3
```

### 4. Configure Archon

1. Open Archon Settings
2. Navigate to **RAG Settings**
3. Set **LLM Provider** to "Ollama"
4. Set **Ollama Base URL** to `http://localhost:11434/v1`
5. Set **Embedding Model** to your chosen model (e.g., `nomic-embed-text`)
6. Click **Save Settings**

## Supported Embedding Models

| Model | Dimensions | Use Case | Command |
|-------|------------|----------|---------|
| `nomic-embed-text` | 768 | Fast processing, resource-constrained environments | `ollama pull nomic-embed-text` |
| `mxbai-embed-large` | 1536 | OpenAI replacement, high accuracy | `ollama pull mxbai-embed-large` |
| `snowflake-arctic-embed2` | 1024 | Balanced performance and efficiency | `ollama pull snowflake-arctic-embed2` |
| `bge-m3` | 1024 | Multilingual content, international applications | `ollama pull bge-m3` |

## Model Selection Guidance

### Choose `nomic-embed-text` if:
- You need fast processing
- Running on resource-constrained hardware
- Working with English content primarily
- Want the smallest memory footprint

### Choose `mxbai-embed-large` if:
- You want drop-in OpenAI compatibility (1536 dimensions)
- Need highest accuracy for complex queries
- Working with technical or specialized content
- Have sufficient compute resources

### Choose `snowflake-arctic-embed2` if:
- You want balanced performance and speed
- Working with mixed content types
- Need good performance without maximum resource usage

### Choose `bge-m3` if:
- You have multilingual content
- Working with international applications
- Need support for non-English languages

## Understanding Embedding Dimensions

**IMPORTANT:** Changing embedding models with different dimensions will require re-embedding all your existing content.

### Dimension Impact

- **768d models**: Faster processing, lower storage requirements, good performance
- **1024d models**: Balanced performance, moderate storage requirements
- **1536d models**: Higher accuracy, compatible with OpenAI models, larger storage requirements

### Data Migration Warnings

When switching between models with different dimensions:

1. ⚠️ **Archon will warn you** about potential data loss
2. 🔄 **All existing embeddings** will need to be regenerated
3. 📊 **Your knowledge base** will need to be re-processed
4. ⏱️ **This process takes time** depending on your content volume

## Advanced Configuration

### Custom Ollama Installation

If running Ollama on a different host or port:

```bash
# Custom host/port
OLLAMA_HOST=0.0.0.0:11435 ollama serve
```

Update Archon settings:
- **Ollama Base URL**: `http://your-host:11435/v1`

### Performance Tuning

Environment variables for Ollama:

```bash
# Increase context length
export OLLAMA_NUM_CTX=4096

# Set GPU memory usage
export OLLAMA_GPU_MEMORY=8GB

# Enable debug logging
export OLLAMA_DEBUG=1
```

### Model Management

```bash
# List installed models
ollama list

# Remove a model
ollama rm model-name

# Update a model
ollama pull model-name
```

## Integration with Chat Models

You can also use Ollama for chat completions:

```bash
# Install a chat model
ollama pull llama2
ollama pull mistral
ollama pull codellama
```

Set **Chat Model** in Archon to: `llama2`, `mistral`, `codellama`, etc.

## Troubleshooting

### Common Issues

**Issue: "Connection refused" error**
```
Solution: Ensure Ollama is running with `ollama serve`
```

**Issue: "Model not found" error**
```
Solution: Pull the model first with `ollama pull model-name`
```

**Issue: "Slow embedding generation"**
```
Solutions:
- Use smaller models (nomic-embed-text)
- Increase GPU memory allocation
- Check CPU/GPU utilization
```

**Issue: "Out of memory" errors**
```
Solutions:
- Use smaller embedding models
- Reduce batch sizes in RAG settings
- Close other applications using GPU
```

### Performance Issues

1. **CPU Usage High**: Switch to GPU-accelerated models if available
2. **Memory Usage High**: Use smaller models or reduce context length
3. **Slow Responses**: Check network connectivity to Ollama service

### Debugging

Enable debug logging:

```bash
export OLLAMA_DEBUG=1
ollama serve
```

Check Ollama logs:
```bash
# macOS/Linux
tail -f ~/.ollama/logs/server.log

# Check service status
curl http://localhost:11434/api/tags
```

## Migration from OpenAI/Google

### Step 1: Backup Current Settings
Take note of your current:
- Chat model name
- Embedding model name  
- Any custom configurations

### Step 2: Choose Compatible Models
- For **1536d OpenAI compatibility**: Use `mxbai-embed-large`
- For **balanced performance**: Use `snowflake-arctic-embed2` 
- For **fastest processing**: Use `nomic-embed-text`

### Step 3: Expect Re-embedding
- **Save your sources list** before switching
- **Budget time** for re-processing your knowledge base
- **Monitor the process** through Archon's UI

## Security Considerations

### Local Network Only
- Ollama runs on localhost by default
- No data leaves your machine
- Complete privacy for sensitive documents

### Firewall Configuration
If running Ollama on a network:
```bash
# Allow specific IP range
sudo ufw allow from 192.168.1.0/24 to any port 11434
```

## Performance Benchmarks

Approximate performance on modern hardware:

| Model | Speed (tokens/sec) | Memory Usage | Quality |
|-------|-------------------|--------------|---------|
| nomic-embed-text | ~2000 | 2GB | Good |
| mxbai-embed-large | ~1200 | 4GB | Excellent |
| snowflake-arctic-embed2 | ~1500 | 3GB | Very Good |
| bge-m3 | ~1400 | 3GB | Very Good |

*Results may vary based on hardware configuration*

## Getting Help

1. **Archon Issues**: Check the Archon GitHub repository
2. **Ollama Issues**: Visit [ollama.com/docs](https://ollama.com/docs)
3. **Model Issues**: Check the model's documentation on Ollama Hub

## What's Next?

After setting up Ollama:

1. ✅ Test embedding generation with sample content
2. ✅ Verify RAG search works with your models
3. ✅ Monitor performance and adjust settings
4. ✅ Consider setting up multiple models for different use cases
5. ✅ Explore Ollama's chat models for complete local AI stack

---

**Need help?** Create an issue on the Archon repository or check the troubleshooting section above.