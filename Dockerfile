# Multi-stage build for RLaaS platform
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN groupadd -r rlaas && useradd -r -g rlaas rlaas

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY pyproject.toml .
COPY README.md .

# Install the package
RUN pip install -e .

# Change ownership
RUN chown -R rlaas:rlaas /app

# Switch to non-root user
USER rlaas

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["uvicorn", "rlaas.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Development stage
FROM base as development

USER root

# Install development dependencies
RUN pip install -e ".[dev]"

# Install additional development tools
RUN apt-get update && apt-get install -y \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

USER rlaas

CMD ["uvicorn", "rlaas.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Production stage
FROM base as production

# Copy only necessary files
COPY --from=base /app /app

# Set production environment
ENV ENVIRONMENT=production

# Use production WSGI server
CMD ["gunicorn", "rlaas.api.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
