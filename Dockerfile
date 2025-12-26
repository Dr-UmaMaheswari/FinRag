FROM python:3.11-slim

WORKDIR /app
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*
# Good defaults for containers
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Copy packaging metadata first to leverage Docker layer caching
COPY pyproject.toml README.md /app/
# Copy source so editable install has actual code present
COPY src /app/src

# If you truly do not use Chroma anymore, remove it from pyproject.toml.
# Install base + Milvus + Hybrid extras (adjust as needed)
RUN pip install -U pip && \
    pip install -e ".[milvus,dev,hybrid]"

# App runtime assets
COPY samples /app/samples

EXPOSE 8000

CMD ["uvicorn", "rag_starterkit.main:app", "--host", "0.0.0.0", "--port", "8000"]
