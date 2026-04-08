# SupportOpsEnv — Dockerfile
# Python 3.11 slim base for smaller image size

FROM python:3.11-slim

# ── Metadata ───────────────────────────────────────────────────────────────────
LABEL maintainer="MetaHack Team"
LABEL description="SupportOpsEnv — OpenEnv Customer Support Operations Environment"
LABEL version="1.0.0"

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────────────────────
WORKDIR /app

# ── Copy dependency file first (for Docker layer caching) ─────────────────────
COPY requirements.txt .

# ── Install Python dependencies ────────────────────────────────────────────────
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Copy project files ─────────────────────────────────────────────────────────
COPY . .

# ── Create __init__ files for packages ────────────────────────────────────────
RUN touch tasks/__init__.py graders/__init__.py

# ── Environment variables (can be overridden at runtime) ──────────────────────
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=7860

# API configuration (override via docker run -e or docker-compose)
ENV API_BASE_URL="https://api.openai.com/v1"
ENV MODEL_NAME="gpt-4o-mini"
ENV HF_TOKEN=""

# ── Expose port ────────────────────────────────────────────────────────────────
EXPOSE 7860

# ── Health check ───────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# ── Default command: run the FastAPI server ────────────────────────────────────
CMD ["python", "app.py"]
