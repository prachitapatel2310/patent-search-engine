# Dockerfile — Patent Search Engine (Part 2 Proof-of-Concept)
# ─────────────────────────────────────────────────────────────
# Build:  docker build -t patent-search .
# Run:    docker run -p 8000:8000 \
#           -v $(pwd)/patent_index.pkl:/app/patent_index.pkl \
#           -v $(pwd)/patent_data:/app/patent_data \
#           patent-search
#
# The pre-built index (patent_index.pkl) is mounted as a volume so the
# container doesn't need to re-embed on every start.  If no index file
# is found, the container will build one on startup (slower first start).

FROM python:3.11-slim

WORKDIR /app

# System deps for sentence-transformers / tokenizers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps first (cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY search_engine.py reranker.py api_server.py ./

# Data is expected to be mounted at runtime:
#   /app/patent_data/data/patent_data_small/*.json
#   /app/patent_index.pkl  (optional pre-built cache)

EXPOSE 8000

ENV PORT=8000

CMD ["python", "api_server.py"]
