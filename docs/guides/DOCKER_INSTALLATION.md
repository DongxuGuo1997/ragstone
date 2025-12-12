# 🐳 Docker Installation Guide for macOS

## 🚀 **Quick Installation Options**

### **Option 1: Docker Desktop (Recommended)**

#### **Download and Install:**
1. **Visit Docker Desktop for Mac**: https://docs.docker.com/desktop/mac/install/
2. **Download** the appropriate version:
   - **Intel Macs**: Docker Desktop for Mac with Intel chip
   - **Apple Silicon (M1/M2/M3)**: Docker Desktop for Mac with Apple chip
3. **Install** by dragging Docker.app to Applications folder
4. **Launch** Docker Desktop from Applications
5. **Complete setup** - Docker will start automatically

#### **Verify Installation:**
```bash
docker --version
docker-compose --version
```

### **Option 2: Homebrew (Command Line)**

```bash
# Install Docker Desktop via Homebrew
brew install --cask docker

# Launch Docker Desktop (required for first-time setup)
open /Applications/Docker.app

# Wait for Docker to start, then verify
docker --version
docker-compose --version
```

### **Option 3: Homebrew with Docker Engine Only**

```bash
# Install Docker engine and compose separately
brew install docker docker-compose

# Note: You'll need Docker Desktop or colima for the Docker daemon
brew install colima

# Start colima (Docker daemon alternative)
colima start

# Verify
docker --version
docker-compose --version
```

---

## ⚡ **Quick Start After Installation**

Once Docker is installed:

```bash
# Navigate to your project
cd /Users/dguo/MyProjects/test-langchain

# Copy environment template
cp .env.example .env

# Edit with your API keys (optional for testing)
nano .env

# Build and start the RAG pipeline
docker-compose up -d rag-pipeline

# Check if it's running
docker ps

# View logs
docker-compose logs -f rag-pipeline
```

---

## 🛠️ **Alternative: Run Without Docker**

If you prefer not to install Docker right now, you can still run the containerized setup using your existing Python environment:

### **Simulate Container Environment:**

```bash
# Set container-like environment variables
export PYTHONPATH=/Users/dguo/MyProjects/test-langchain/src
export USER_AGENT="Local-MCP-Server/1.0"
export MCP_LOG_LEVEL=INFO

# Create directories that would be mounted in container
mkdir -p logs

# Run the MCP server (simulating container)
python mcp_rag_server_fastmcp.py
```

### **Run Streamlit UI (Container Mode):**

```bash
# Run Streamlit like it would run in container
streamlit run run.py --server.address=0.0.0.0 --server.port=8501
```

### **Test Connection (Container Style):**

```bash
# Test the setup
python mcp_connection_test.py
```

---

## 🔧 **Troubleshooting Docker Installation**

### **Common Issues:**

#### **"Docker daemon not running"**
```bash
# Start Docker Desktop manually
open /Applications/Docker.app

# Or if using colima
colima start
```

#### **Permission Issues**
```bash
# Add your user to docker group (if using Docker engine)
sudo dkpg -l docker.io  # Check if installed
sudo usermod -aG docker $USER
# Logout and login again
```

#### **Docker Desktop Won't Start**
1. **Check system requirements** - macOS 10.15 or later
2. **Enable virtualization** in System Preferences → Security
3. **Restart Mac** after installation
4. **Check Activity Monitor** for Docker processes

#### **Port Conflicts**
```bash
# Check what's using port 8000
lsof -i :8000

# Kill processes if needed
sudo kill -9 <PID>
```

---

## 📋 **System Requirements**

### **For Docker Desktop:**
- **macOS 10.15** or later
- **4GB RAM** minimum (8GB recommended)
- **VirtualBox** not running (conflicts with Docker Desktop)

### **For Homebrew Installation:**
- **Homebrew** installed: `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`
- **Xcode Command Line Tools**: `xcode-select --install`

---

## 🎯 **Recommended Installation Steps**

### **Step 1: Install Docker Desktop**
```bash
# Using Homebrew (easiest)
brew install --cask docker

# Launch Docker Desktop
open /Applications/Docker.app
```

### **Step 2: Verify Installation**
```bash
# Wait for Docker to start, then test
docker --version
docker-compose --version
docker run hello-world
```

### **Step 3: Test Your RAG Pipeline**
```bash
# In your project directory
cd /Users/dguo/MyProjects/test-langchain

# Quick test without building
docker run --rm -v $(pwd):/app -w /app python:3.11-slim python -c "print('✅ Docker works!')"

# Build and run your pipeline
docker-compose up -d rag-pipeline
```

---

## 🚀 **Next Steps After Installation**

1. **Follow the main Docker Guide**: `DOCKER_GUIDE.md`
2. **Configure your `.env` file** with API keys
3. **Start with basic setup**: `docker-compose up -d rag-pipeline`
4. **Test MCP integration** with Cursor
5. **Explore advanced features** like ChromaDB and Ollama

---

## 💡 **Pro Tips**

1. **Docker Desktop** is the easiest option for beginners
2. **Allocate sufficient resources** in Docker Desktop preferences
3. **Use Docker Desktop's built-in terminal** for Docker commands
4. **Enable "Use Docker Compose V2"** in Docker Desktop settings
5. **Keep Docker Desktop updated** for best performance

---

## 🆘 **Still Having Issues?**

### **Check Docker Status:**
```bash
docker info
docker system df
docker system prune  # Clean up if needed
```

### **Alternative Approaches:**
1. **Use the existing Python setup** (no Docker needed)
2. **Try Podman** as Docker alternative: `brew install podman`
3. **Use GitHub Codespaces** for cloud-based development
4. **Run on a Linux VM** if macOS issues persist

---

**Installation Help:**
- **Docker Documentation**: https://docs.docker.com/desktop/mac/install/
- **Homebrew**: https://brew.sh/
- **Docker Community**: https://forums.docker.com/

**Next:** Once Docker is installed, return to `DOCKER_GUIDE.md` for usage instructions! 