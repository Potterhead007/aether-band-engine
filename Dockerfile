# AETHER Band Engine
# Production Docker Image
#
# Build full:  docker build -t aether-band-engine .
# Build API:   docker build -t aether-api --target api .
# Run:         docker run -e ANTHROPIC_API_KEY=... aether-band-engine

# =============================================================================
# Stage 1a: Builder (API - lightweight)
# =============================================================================
FROM python:3.11-slim AS builder-api

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies (API only - no ML libs)
COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir --upgrade pip hatchling && \
    pip install --no-cache-dir ".[api]"

# =============================================================================
# Stage 1b: Builder (Full - includes ML)
# =============================================================================
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install package with all extras (non-editable for production)
RUN pip install --no-cache-dir --upgrade pip hatchling && \
    pip install --no-cache-dir ".[full]"

# =============================================================================
# Stage 2: Runtime
# =============================================================================
FROM python:3.11-slim AS runtime

LABEL maintainer="AETHER Team"
LABEL version="0.1.0"
LABEL description="AETHER Band Engine - AI Music Generation"

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    fluidsynth \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN useradd --create-home --shell /bin/bash aether
USER aether
WORKDIR /home/aether

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application
COPY --chown=aether:aether src/ ./src/
COPY --chown=aether:aether data/ ./data/
COPY --chown=aether:aether pyproject.toml ./

# Create directories
RUN mkdir -p /home/aether/.aether/output \
             /home/aether/.aether/projects \
             /home/aether/.aether/cache

# Environment configuration
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV AETHER_PATHS__BASE_DIR=/home/aether/.aether

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from aether.core.health import HealthCheck; import asyncio; h = HealthCheck(); asyncio.run(h.check_all())" || exit 1

# Default command
ENTRYPOINT ["aether"]
CMD ["--help"]

# Expose metrics port (if running API server)
EXPOSE 9090

# =============================================================================
# Stage 3: API Server (production - lightweight)
# =============================================================================
FROM python:3.11-slim AS api

LABEL maintainer="AETHER Team"
LABEL version="0.1.0"
LABEL description="AETHER Band Engine - Production API Server"

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN useradd --create-home --shell /bin/bash aether
USER aether
WORKDIR /home/aether

# Copy virtual environment from API builder (lightweight, no ML libs)
COPY --from=builder-api /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application
COPY --chown=aether:aether src/ ./src/
COPY --chown=aether:aether data/ ./data/
COPY --chown=aether:aether pyproject.toml ./

# Create directories
RUN mkdir -p /home/aether/.aether/output \
             /home/aether/.aether/cache

# Environment configuration
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV AETHER_ENVIRONMENT=production
ENV AETHER_LOG_FORMAT=json
ENV AETHER_API_HOST=0.0.0.0
ENV AETHER_API_PORT=8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose API port
EXPOSE 8000

# Run API server
CMD ["python", "-m", "uvicorn", "aether.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
