# syntax=docker/dockerfile:1
# ------------------------------------------------------------------------------
# Demand Consumption Pipeline - Production Runtime Dockerfile
# Python 3.13-slim multi-stage hardened container
# ------------------------------------------------------------------------------

FROM python:3.13-slim AS runtime

# Set environment flags
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIPELINE_ENV=production \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    COORDINATOR_DB_PATH=/app/data/coordinator/coordinator.db \
    FULFILMENT_DB_PATH=/app/data/fulfilment/fulfilment.db \
    DATASET_PATH=/app/data/processed/df_featured.parquet \
    MODELS_DIR=/app/outputs \
    PYTHONPATH=/app

WORKDIR /app

# Install minimal system dependencies (libgomp1 required for LightGBM OpenMP support)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create dedicated non-root application user and group
RUN groupadd -g 10001 appuser && \
    useradd -u 10001 -g appuser -s /bin/bash -m appuser

# Pre-create directory layout with correct permissions
RUN mkdir -p /app/data/coordinator /app/data/fulfilment /app/outputs /app/logs && \
    chown -R appuser:appuser /app

# Copy dependency requirements first to leverage Docker layer caching
COPY requirements.txt /app/

# Install python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application configuration, source, data assets, and models
COPY pyproject.toml README.md /app/
COPY src/ /app/src/
COPY data/processed/ /app/data/processed/
COPY data/raw/ /app/data/raw/
COPY outputs/ /app/outputs/

# Install application package
RUN pip install --no-cache-dir -e .

# Final ownership adjustment
RUN chown -R appuser:appuser /app

# Persistent database storage mount point
VOLUME ["/app/data"]

# Expose HTTP API port
EXPOSE 8000

# Built-in health check using Python standard library (no curl dependency required)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')" || exit 1

# Run as non-root user
USER appuser

# Default entrypoint and command: agy-pipeline serve
ENTRYPOINT ["agy-pipeline"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]

