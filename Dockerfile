# TelNet - Seasonal Precipitation Forecasting
# Multi-stage Dockerfile for training and inference

# ========================================
# Stage 1: Base image with dependencies
# ========================================
FROM python:3.11-slim as base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    libgeos-dev \
    libproj-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir pip --upgrade && \
    pip install --no-cache-dir .

# ========================================
# Stage 2: Training image with GPU support
# ========================================
FROM nvidia/cuda:12.1-runtime-ubuntu22.04 as training

WORKDIR /app

# Install Python and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    python3.11-venv \
    git \
    curl \
    libgeos-dev \
    libproj-dev \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Copy application code
COPY . .

# Install Python dependencies with GPU support
RUN pip install --no-cache-dir pip --upgrade && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir .

# Create data and results directories
RUN mkdir -p /data /results

# Set environment variables
ENV TELNET_DATADIR=/data
ENV PYTHONUNBUFFERED=1

# Entry point for training
ENTRYPOINT ["python", "-m", "telnet.cli"]
CMD ["--help"]

# ========================================
# Stage 3: Inference image (CPU only, smaller)
# ========================================
FROM python:3.11-slim as inference

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgeos-dev \
    libproj-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY . .

# Install Python dependencies (CPU only)
RUN pip install --no-cache-dir pip --upgrade && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir .

# Create data and results directories
RUN mkdir -p /data /results

# Set environment variables
ENV TELNET_DATADIR=/data
ENV PYTHONUNBUFFERED=1

# Entry point for inference
ENTRYPOINT ["python", "-m", "telnet.cli"]
CMD ["--help"]

# ========================================
# Stage 4: Development image
# ========================================
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir pytest black ruff

# Copy application code
COPY . .

# Set environment variables
ENV TELNET_DATADIR=/data
ENV PYTHONUNBUFFERED=1

CMD ["/bin/bash"]
