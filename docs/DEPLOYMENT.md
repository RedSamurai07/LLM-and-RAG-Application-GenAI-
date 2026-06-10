# 🚀 End-to-End GenAI Deployment — Hybrid RAG API on AWS EC2 with Docker & GitHub Actions

This guide details the environment configuration, dependency management, container deployment strategy, and validation steps to host the LLM and RAG Application (GenAI) on a cloud instance using Docker or Hugging Face Spaces.

---

## System Architecture Endpoints

- **Core API Layer (FastAPI):** `http://<EC2_PUBLIC_IP>:5000`
- **API Interactive Playground (Swagger UI):** `http://<EC2_PUBLIC_IP>:5000/docs`
- **AI-Powered RAG Interface (Gradio UI):** `http://<EC2_PUBLIC_IP>:7860`

---

## Step 1: Launch and Configure Your Environment

1. **Provision Compute:** Launch an Amazon EC2 instance using **Ubuntu 22.04 LTS** (a `t2.medium` or `t3.medium` instance is recommended due to embedding model memory requirements).
2. **Configure Firewall / Security Groups:** Expose the minimum necessary ingress ports to authorize external traffic pipelines securely:
   - **SSH (Port 22):** For secure remote shell administration.
   - **FastAPI (Port 5000):** To handle client requests and RAG inference payloads.
   - **Gradio UI (Port 7860):** To serve the interactive AI-powered Q&A web interface.

### Establish Secure SSH Connection

#### On Windows (PowerShell):

```
# Restrict file permissions to the current user (Windows equivalent of chmod 400)
icacls "rag-app-key.pem" /inheritance:r
icacls "rag-app-key.pem" /grant:r "${env:USERNAME}:R"

# Connect to the remote instance
ssh -i "rag-app-key.pem" ubuntu@<EC2_PUBLIC_IP>
```

#### On Linux/Mac:

```
# Set strict read-only permissions for the private key
chmod 400 rag-app-key.pem

# Connect to the remote instance
ssh -i "rag-app-key.pem" ubuntu@<EC2_PUBLIC_IP>
```

## Step 2: Install Container Runtime Environment

Once authenticated within the remote Ubuntu shell, initialize and configure the Docker engine:

```
# Update local package indexes
sudo apt-get update

# Install the standard Docker runtime
sudo apt-get install -y docker.io

# Enable the Docker daemon to automatically initialize on system boot
sudo systemctl start docker
sudo systemctl enable docker

# Add the default ubuntu user to the docker group to execute commands without sudo
sudo usermod -aG docker $USER

# CRITICAL: Terminate session and reconnect via SSH for group updates to take effect
exit
```

## Step 3: Deploy the RAG Application Service

Reconnect to your EC2 instance and run the following deployment script to build the image layer and run the container with persistence guardrails:

```
# 1. Clone the production source code from the repository
git clone https://github.com/RedSamurai07/LLM-and-RAG-Application-GenAI-.git
cd LLM-and-RAG-Application-GenAI-

# 2. Install all required dependencies
pip install -r requirements.txt

# 3. Build the Docker application image layer
docker build -t llm-rag-api .

# 4. Instantiate the production container engine
# Maps runtime ports, ensures data persistence, and establishes crash auto-restart logic
docker run -d \
  -p 5000:5000 \
  -p 7860:7860 \
  -v chroma_data:/app/chroma_db \
  -v model_cache:/app/models \
  --name rag-service \
  --restart unless-stopped \
  llm-rag-api
```

💡 **Production Enhancements Added:**

- `--restart unless-stopped`: Ensures the RAG service automatically reboots if the application crashes or the underlying EC2 server undergoes a hardware reboot.

- `-v chroma_data:/app/chroma_db`: Mounts a persistent named Docker volume so your ChromaDB vector store and indexed document embeddings survive container updates and deletions.

- `-v model_cache:/app/models`: Preserves downloaded Sentence Transformer embedding model weights across container lifecycle events, ensuring inference continuity without re-downloading.

## Step 4: Infrastructure & Service Verification

**1. Health-Check Endpoint API Validation**

Test the baseline responsiveness of the FastAPI engine from your local machine terminal:

```
curl http://<EC2_PUBLIC_IP>:5000/health
```

Expected response:

```
{
  "status": "healthy",
  "embedding_model_loaded": true,
  "vector_store_ready": true
}
```

**2. RAG Query Endpoint Validation**

Submit a test query payload to verify the hybrid search and generation pipeline is operational:

```
curl -X POST http://<EC2_PUBLIC_IP>:5000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is Retrieval-Augmented Generation?",
    "top_k": 5
  }'
```

**3. Interactive Swagger API Playground**

FastAPI natively serves interactive OpenAPI documentation. You can test live RAG query payloads directly through your web browser at:

```
http://<EC2_PUBLIC_IP>:5000/docs
```

**4. AI-Powered Gradio RAG Interface**

To interact with the visual Q&A interface featuring document ingestion, query input, and real-time retrieved context results, navigate to:

```
http://<EC2_PUBLIC_IP>:7860
```

## Alternative: Deploy on Hugging Face Spaces

As an alternative to AWS EC2, the RAG application can be deployed directly on Hugging Face Spaces for zero-infrastructure hosting:

```
# 1. Install the Hugging Face Hub CLI
pip install huggingface_hub

# 2. Login to your Hugging Face account
huggingface-cli login

# 3. Create a new Gradio Space and push the application
huggingface-cli repo create llm-rag-app --type space --space_sdk gradio
git remote add hf https://huggingface.co/spaces/<YOUR_HF_USERNAME>/llm-rag-app
git push hf main
```

## CI/CD Pipeline Status

The operational integrity of the master codebase is continuously protected via automated integration testing gates:

[![CI Pipeline](https://github.com/RedSamurai07/LLM-and-RAG-Application-GenAI-/actions/workflows/main.yml/badge.svg)](https://github.com/RedSamurai07/LLM-and-RAG-Application-GenAI-/actions/workflows/main.yml)