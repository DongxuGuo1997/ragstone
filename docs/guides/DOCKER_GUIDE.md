# 🐳 Docker Setup Guide
## LangChain RAG Pipeline Containerization

---

## 🚀 **Quick Start**

### **1. Basic Setup**
```bash
# Clone and navigate to your project
cd /path/to/your/langchain-rag-pipeline

# Copy environment template
cp .env.example .env

# Edit .env with your API keys
nano .env

# Build and start the container
docker-compose up -d rag-pipeline
```

### **2. Test the Setup**
```bash
# Check if container is running
docker ps

# Test the MCP server
curl http://localhost:8000/health

# View logs
docker-compose logs -f rag-pipeline
```

---

## 📋 **Available Services**

### **Core Service**
- **`rag-pipeline`** - Main MCP server (Port 8000)

### **Optional Services** (use profiles to enable)
- **`chromadb`** - Vector database (Port 8001)
- **`ollama`** - Local LLM server (Port 11434)  
- **`redis`** - Caching layer (Port 6379)
- **`rag-dev`** - Development environment (Port 8080, 8501)

---

## 🛠️ **Configuration Options**

### **Environment Variables**

Create a `.env` file with your configuration:

```bash
# Required for OpenAI models
OPENAI_API_KEY=your_openai_api_key_here

# Optional for Anthropic models
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Vector store configuration (defaults to faiss)
VECTOR_STORE_TYPE=faiss

# Logging level
MCP_LOG_LEVEL=INFO

# Custom model settings
DEFAULT_OPENAI_MODEL=gpt-3.5-turbo
DEFAULT_OLLAMA_MODEL=llama3
```

### **Volume Mounts**

The container uses these volume mounts:
- `./data` → `/app/data` (read-only) - Your documents
- `./store` → `/app/store` - Pipeline storage
- `./vs_data` → `/app/vs_data` - Vector store data
- `./logs` → `/app/logs` - Application logs
- `./config.json` → `/app/config.json` (read-only) - Configuration

---

## 🗂️ **Vector Store Configuration**

### **Default: FAISS (Recommended for Docker)**

FAISS is now the default vector store for Docker deployments. It's:
- ✅ **Self-contained** - No external services required
- ✅ **Fast** - Excellent performance for similarity search
- ✅ **Persistent** - Data persists between container restarts
- ✅ **Memory efficient** - Lower resource usage than ChromaDB

```bash
# FAISS is the default - no configuration needed
docker-compose up -d rag-pipeline

# Or explicitly set FAISS
echo "VECTOR_STORE_TYPE=faiss" >> .env
docker-compose up -d rag-pipeline
```

### **Alternative: ChromaDB**

Use ChromaDB if you need:
- Advanced querying capabilities
- Web UI for database management
- Multi-client access

```bash
# Set ChromaDB as vector store
echo "VECTOR_STORE_TYPE=chroma" >> .env

# Start ChromaDB service
docker-compose --profile chromadb up -d

# ChromaDB UI: http://localhost:8001
```

### **Switching Vector Stores**

```bash
# Stop current services
docker-compose down

# Change vector store type
sed -i 's/VECTOR_STORE_TYPE=.*/VECTOR_STORE_TYPE=faiss/' .env

# Start with new configuration
docker-compose up -d rag-pipeline

# Note: Existing embeddings will need to be recreated
```

---

## 🎯 **Usage Scenarios**

### **Scenario 1: Basic MCP Server**
```bash
# Start just the RAG pipeline
docker-compose up -d rag-pipeline

# Use with Cursor
# Configure Cursor MCP to point to: http://localhost:8000
```

### **Scenario 2: Using ChromaDB Vector Store**
```bash
# Set ChromaDB as vector store
echo "VECTOR_STORE_TYPE=chroma" >> .env

# Start with ChromaDB service
docker-compose --profile chromadb up -d

# Your pipeline will use ChromaDB for vector storage
# ChromaDB UI available at: http://localhost:8001
```

### **Scenario 3: Local LLMs with Ollama**
```bash
# Start with Ollama for local models
docker-compose --profile ollama up -d

# Pull a model in Ollama
docker exec -it ollama ollama pull llama3

# Create pipeline with local model
# @langchain-rag-pipeline create_ollama_pipeline model="llama3"
```

### **Scenario 4: Development Environment**
```bash
# Start development setup with Streamlit UI
docker-compose --profile dev up -d

# Access Streamlit UI at: http://localhost:8501
# Access MCP server at: http://localhost:8080
```

### **Scenario 5: Full Production Stack**
```bash
# Start everything
docker-compose --profile chromadb --profile ollama --profile cache up -d

# You now have:
# - RAG Pipeline (8000)
# - ChromaDB (8001)  
# - Ollama (11434)
# - Redis Cache (6379)
```

---

## 🔧 **Advanced Configuration**

### **Custom Dockerfile**

For custom builds, modify the `Dockerfile`:

```dockerfile
# Add custom dependencies
RUN pip install your-custom-package

# Add custom configuration
COPY your-config.json /app/
```

### **Multiple Environments**

Create environment-specific compose files:

```bash
# docker-compose.prod.yml
version: '3.8'
services:
  rag-pipeline:
    extends:
      file: docker-compose.yml
      service: rag-pipeline
    environment:
      - MCP_LOG_LEVEL=WARNING
    deploy:
      replicas: 2
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G
```

### **GPU Support for Ollama**

Uncomment GPU configuration in `docker-compose.yml`:

```yaml
ollama:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

---

## 🐛 **Troubleshooting**

### **Container Won't Start**

```bash
# Check logs
docker-compose logs rag-pipeline

# Common issues:
# 1. Missing .env file
# 2. Invalid API keys
# 3. Port conflicts
# 4. Missing dependencies
```

### **MCP Server Not Responding**

```bash
# Check if server is running
docker exec -it langchain-rag-pipeline python -c "from src.pipeline import Pipeline; print('OK')"

# Check port binding
docker port langchain-rag-pipeline

# Test connection
curl http://localhost:8000/health
```

### **Volume Mount Issues**

```bash
# Check volume mounts
docker inspect langchain-rag-pipeline | grep -A 10 "Mounts"

# Fix permissions
sudo chown -R $(id -u):$(id -g) ./data ./store ./vs_data ./logs
```

### **Memory Issues**

```bash
# Check memory usage
docker stats langchain-rag-pipeline

# Increase memory limits in docker-compose.yml
services:
  rag-pipeline:
    deploy:
      resources:
        limits:
          memory: 4G
```

---

## 📊 **Monitoring & Maintenance**

### **Health Checks**

The container includes health checks:

```bash
# Check health status
docker ps --format "table {{.Names}}\t{{.Status}}"

# Manual health check
docker exec langchain-rag-pipeline python -c "from src.pipeline import Pipeline; print('✅ Healthy')"
```

### **Log Management**

```bash
# View real-time logs
docker-compose logs -f rag-pipeline

# Rotate logs
docker-compose exec rag-pipeline logrotate /etc/logrotate.conf

# Export logs
docker-compose logs rag-pipeline > rag-pipeline.log
```

### **Backup & Restore**

```bash
# Backup volumes
docker run --rm -v test-langchain_chroma_data:/data -v $(pwd):/backup alpine tar czf /backup/chroma_backup.tar.gz -C /data .

# Restore volumes  
docker run --rm -v test-langchain_chroma_data:/data -v $(pwd):/backup alpine tar xzf /backup/chroma_backup.tar.gz -C /data
```

---

## 🚀 **Deployment Options**

### **Local Development**
```bash
docker-compose --profile dev up -d
```

### **Production Single Node**
```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### **Docker Swarm**
```bash
docker stack deploy -c docker-compose.yml langchain-rag
```

### **Kubernetes**

Create Kubernetes manifests:

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: langchain-rag-pipeline
spec:
  replicas: 2
  selector:
    matchLabels:
      app: langchain-rag-pipeline
  template:
    metadata:
      labels:
        app: langchain-rag-pipeline
    spec:
      containers:
      - name: rag-pipeline
        image: langchain-rag-pipeline:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: openai
```

---

## 🎛️ **Container Modes**

The container supports multiple modes via the `MODE` environment variable:

### **MCP Server Mode** (default)
```bash
docker run -e MODE=mcp langchain-rag-pipeline
```

### **Streamlit UI Mode**
```bash
docker run -e MODE=streamlit -p 8501:8501 langchain-rag-pipeline
```

### **Chat Interface Mode**
```bash
docker run -e MODE=chat -it langchain-rag-pipeline
```

### **Test Mode**
```bash
docker run -e MODE=test langchain-rag-pipeline
```

### **Shell Mode** (debugging)
```bash
docker run -e MODE=shell -it langchain-rag-pipeline
```

---

## 📚 **Integration Examples**

### **With Cursor IDE**

Configure Cursor MCP settings:

```json
{
  "mcpServers": {
    "langchain-rag-pipeline": {
      "command": "docker",
      "args": ["exec", "-i", "langchain-rag-pipeline", "python", "mcp_rag_server_fastmcp.py"],
      "env": {
        "USER_AGENT": "Cursor-Docker-MCP/1.0"
      }
    }
  }
}
```

### **With CI/CD**

```yaml
# .github/workflows/docker.yml
name: Docker Build and Test
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Build Docker image
      run: docker build -t langchain-rag-pipeline .
    - name: Test container
      run: |
        docker run -d --name test-container langchain-rag-pipeline
        docker exec test-container python mcp_connection_test.py
```

---

## 🔐 **Security Considerations**

### **API Key Management**
- Use `.env` files (never commit to git)
- Consider Docker secrets for production
- Use environment-specific configurations

### **Network Security**
```yaml
# Restrict network access
services:
  rag-pipeline:
    networks:
      - internal
networks:
  internal:
    internal: true
```

### **User Permissions**
- Container runs as non-root user
- Volume mounts with appropriate permissions
- Read-only mounts where possible

---

## 🎯 **Best Practices**

1. **Use multi-stage builds** - Smaller production images
2. **Pin dependency versions** - Reproducible builds
3. **Health checks** - Monitor container health
4. **Resource limits** - Prevent resource exhaustion
5. **Logging** - Centralized log management
6. **Secrets management** - Secure API key handling
7. **Regular updates** - Keep base images updated

---

## 🆘 **Quick Commands Reference**

```bash
# Build and start
docker-compose up -d rag-pipeline

# View logs
docker-compose logs -f rag-pipeline

# Stop services
docker-compose down

# Rebuild after changes
docker-compose build rag-pipeline

# Shell into container
docker exec -it langchain-rag-pipeline /bin/bash

# Test MCP connection
docker exec langchain-rag-pipeline python mcp_connection_test.py

# Start with all services
docker-compose --profile chromadb --profile ollama --profile cache up -d

# Production deployment
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

---

**Last Updated:** December 2024  
**Status:** ✅ Complete Docker setup with multiple deployment options 