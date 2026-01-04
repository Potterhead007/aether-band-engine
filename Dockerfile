# AETHER Band Engine
# Production Docker Image
#
# Build: docker build -t aether-band-engine .
# Run:   docker run -e ANTHROPIC_API_KEY=... aether-band-engine

# =============================================================================
# Stage 1: Builder
# =============================================================================
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY pyproject.toml ./
COPY src/ ./src/

# Install package with all extras
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[full]"

# =============================================================================
# Stage 2: Runtime
# =============================================================================
FROM python:3.11-slim as runtime

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
