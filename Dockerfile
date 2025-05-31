# Base image with Python and Jupyter
#FROM python:3.10-slim

# Use CUDA-enabled base image with Python (Ubuntu 22.04 recommended)
FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.10 and other dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3-pip \
    git curl wget sudo nano \
    && ln -s /usr/bin/python3.10 /usr/bin/python \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Install Python packages
RUN pip install --upgrade pip &&  \
    pip install --no-cache-dir -r requirements.txt

# Expose port for Jupyter Notebook
EXPOSE 8888

# Start Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]
