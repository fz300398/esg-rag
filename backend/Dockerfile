# === Base Image ===
FROM python:3.11-slim

# === System Setup ===
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# === Python Dependencies ===
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# === Copy Application Files ===
COPY . .

# === Environment ===
ENV CHROMA_DIR=/app/chroma \
    DATA_DIR=/app/data \
    OLLAMA_BASE_URL=http://ollama:11434 \
    EMBEDDING_MODEL=intfloat/multilingual-e5-base \
    RERANKER_MODEL=BAAI/bge-reranker-v2-m3

# === Expose API Port ===
EXPOSE 8000

# === Volume Mount (persist ChromaDB) ===
VOLUME ["/app/chroma", "/app/data"]

# === Healthcheck ===
HEALTHCHECK CMD curl --fail http://localhost:8000/healthz || exit 1

# === Start the FastAPI Server ===
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]