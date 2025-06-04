# =========================
# 🔧 Stage 1: Builder Stage
# =========================
FROM python:3.10-slim AS builder

# Set working directory
WORKDIR /app

# Install dependencies manually here (edit as needed)
COPY src /app/src
COPY setup.sh /app/setup.sh
RUN chmod +x /app/setup.sh && ./setup.sh

# Export a full list of installed packages
RUN pip freeze > /app/requirements.txt

# ===============================
# 🚀 Stage 2: Final Runtime Image
# ===============================
FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python and basic tools
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3-pip \
    git curl wget sudo nano \
    && ln -s /usr/bin/python3.10 /usr/bin/python \
    && apt-get clean \

# Set working directory
WORKDIR /app

# Copy source code
COPY src /app/src

# Copy frozen requirements from builder
COPY --from=builder /app/requirements.txt /app/requirements.txt

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install exactly the frozen dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Expose Jupyter port
EXPOSE 8888

# Run Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]